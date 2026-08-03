# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that the installed wheel is usable as a C++ SDK.

The wheel ships a prebuilt runtime library plus a CMake package config, so a
standalone application can find_package(executorch) and link
executorch::runtime without building ExecuTorch from source. These checks run
against the installed wheel only; they never look at the source tree's build
directory.

Two properties are verified:

1. Exactly one shipped library defines the backend registry. Backends register
   into a process-wide table owned by the runtime, so a second definition would
   silently give the process two tables and let a backend register into the one
   nobody reads.
2. A C++ consumer builds and runs against the wheel, and records a dependency
   on the shipped runtime with a relocatable RUNPATH.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Registry entry points. A second definer of any of these means a second
# process-wide registry.
_REGISTRY_SYMBOLS = (
    "executorch::runtime::register_backend",
    "executorch::runtime::get_num_registered_backends",
    "executorch::runtime::get_backend_class",
)

# The thread pool accessor. A second definer means a second pool, which
# oversubscribes the CPU because each pool sizes itself to all cores.
_THREADPOOL_SYMBOLS = ("executorch::extension::threadpool::get_threadpool",)

# A representative operator from the merged CPU kernels. A second definer means
# the operators are registered twice, which aborts at startup.
_KERNEL_SYMBOLS = ("torch::executor::native::abs_out",)

# The registry entry points, kept separate from the kernel implementations above.
# A library that carries its own copy of these has its own registration code, which
# is what this split is meant to prevent: one owner of the operator table. Checking
# only a kernel implementation would miss that entirely.
_KERNEL_REGISTRY_SYMBOLS = (
    "executorch::runtime::register_kernels",
    "executorch::runtime::get_registered_kernels",
)

# A representative symbol from the XNNPACK delegate. A second definer means the
# process carries two copies of the delegate.
_XNNPACK_SYMBOLS = (
    "executorch::backends::xnnpack::XnnpackBackendOptions::workspace_manager",
)

# A representative symbol from the CUDA delegate's shim layer. The delegate's own
# methods are weak symbols, so this checks a strong one instead.
_CUDA_SYMBOLS = ("executorch::backends::cuda::clearCurrentCUDAStream",)

# `nm -DC` prints "<hexaddr> <kind> <name>" for a definition and
# "                 U <name>" for an undefined reference.
_DEFINED = re.compile(r"^[0-9a-fA-F]+\s+(?P<kind>[A-Za-z])\s+(?P<name>.+)$")

# Symbol kinds that mean the object owns the code or storage. Weak and unique
# kinds are included because a definition is still a definition, but every
# symbol probed here is a strong one, which is what makes a second definer a
# real second copy rather than ordinary vague linkage.
_OWNING_KINDS = frozenset("TtBbDdGgSsRrWV")

_CONSUMER_SOURCE = """\
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>
#include <vector>

using namespace executorch::extension;

int main() {
  executorch::runtime::runtime_init();
  // Printed rather than asserted on purpose. This consumer links only the
  // runtime, exactly as the documented two-line example does, and the runtime
  // alone registers no backend. Requiring a nonzero count here would be
  // asserting that the runtime does something it is not supposed to do.
  std::printf(
      "registered backends: %zu\\n",
      (size_t)executorch::runtime::get_num_registered_backends());

  // Use the Module and tensor APIs, which are how an application is expected to
  // load and run a program. Constructing them proves the shipped headers and the
  // shipped library agree, which taking sizeof alone would not: a declaration is
  // enough for that, while these need real definitions at link time.
  module::Module module("nonexistent.pte");
  std::vector<float> data(4, 1.0f);
  auto input = make_tensor_ptr({2, 2}, data.data());
  std::printf("tensor holds %zu values\\n", (size_t)input->numel());

  // A load failure is expected here, since no program is shipped for this check.
  // What matters is that the call links and returns an error rather than failing
  // to resolve a symbol.
  const auto error = module.load();
  std::printf("module load returned 0x%x as expected\\n", (unsigned)error);
  return 0;
}
"""

_CONSUMER_CMAKE = """\
cmake_minimum_required(VERSION 3.28)
project(executorch_wheel_consumer CXX)
find_package(executorch REQUIRED)
add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE executorch::runtime)
"""


# Dependencies that come from outside the wheel. Torch libraries resolve once the
# torch package is imported, libpython comes from the running interpreter, and the
# CUDA and TensorRT runtimes are deliberately not bundled, so they come from the
# environment or the separate nvidia packages. None is reachable from an ldd process,
# and a wheel must not carry an absolute path to a build machine's copy just to
# satisfy a check. Anything the wheel itself ships still has to resolve.
_EXTERNAL_LIBRARY_PREFIXES = (
    "libpython",
    "libtorch",
    "libc10",
    "libcuda",
    "libcurand",
    "libcublas",
    "libnvinfer",
)


def _provided_externally(name: str) -> bool:
    """Whether a shared library is expected to come from outside the wheel."""
    return name.startswith(_EXTERNAL_LIBRARY_PREFIXES)


# The component library each target is expected to expose. Keyed by the library base
# name as shipped, so the test can start from what is in the wheel and require a target
# for it, rather than only inspecting targets that happen to exist.
_COMPONENT_LIBRARIES = {
    "libexecutorch_threadpool": "threadpool",
    "libexecutorch_optimized_native_cpu_ops_lib": "kernels",
    "libexecutorch_xnnpack_backend": "xnnpack_backend",
    "libexecutorch_cuda_backend": "cuda_backend",
}


def _shipped_components(package_dir: Path) -> set:
    """Component names the wheel ships a library for."""
    found = set()
    for library in _shipped_shared_objects(package_dir):
        for base, component in _COMPONENT_LIBRARIES.items():
            if library.name.startswith(base):
                found.add(component)
    return found


def _needs_external_cuda_runtime(package_dir: Path) -> bool:
    """Whether the wheel's libraries depend on a CUDA runtime it does not bundle.

    Used to skip checks that assume every dependency is either shipped or reachable
    without help. The CUDA runtime is deliberately not bundled, so on a machine where
    it comes from the separate nvidia packages those checks would report a fault that
    is by design.

    Decided from the shipped file names rather than by reading each ELF, so the answer
    does not depend on a tool being installed. Getting this wrong in the absent-tool
    direction would treat a CUDA wheel as a CPU one and fail the checks it should skip.
    """
    return any(
        "cuda" in library.name for library in _shipped_shared_objects(package_dir)
    )


def _installed_package_dir() -> Path:
    """The installed executorch package, never the source checkout."""
    import executorch

    return Path(list(executorch.__path__)[0]).resolve()


def _shipped_shared_objects(package_dir: Path):
    return [
        path
        for path in sorted(package_dir.rglob("*.so*"))
        if path.is_file() and not path.is_symlink()
    ]


def _defines_symbol(library: Path, symbol: str) -> bool:
    result = subprocess.run(
        ["nm", "-DC", str(library)], capture_output=True, text=True, check=False
    )
    # A library nm cannot read would otherwise look like one that simply defines
    # nothing, letting the single-definer checks pass without having actually
    # inspected every shipped library.
    assert result.returncode == 0, (
        f"nm could not read {library}, so the symbol checks cannot be trusted: "
        f"{result.stderr.strip()}"
    )
    for line in result.stdout.splitlines():
        if symbol not in line:
            continue
        match = _DEFINED.match(line)
        if not match or match.group("kind") not in _OWNING_KINDS:
            continue
        # Exact, or followed by the argument list that nm -C prints. A plain prefix
        # test would also match a longer name that merely starts the same way.
        name = match.group("name")
        if name == symbol or name.startswith(symbol + "("):
            return True
    return False


def test_shipped_libraries_load() -> None:
    """Every shipped library must depend only on things that exist.

    The symbol checks prove each component is defined exactly once, but a library
    can still be unloadable if it needs something nothing provides, which is a
    packaging bug rather than a duplication bug.

    A dependency the wheel ships elsewhere is fine even when `ldd` cannot resolve
    it: some extensions are loaded after `import torch` has already brought their
    dependencies into the process, so they intentionally carry no path to them.
    Only a name nothing in the wheel provides is a real problem.
    """
    if shutil.which("ldd") is None:
        print("- ldd not available, skipping the load check")
        return
    # Torch has to be installed for this to mean anything: several shipped
    # libraries depend on it and resolve once it is imported. Without it every one
    # of them looks broken, which would report a packaging fault that does not
    # exist.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the load check")
        return

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    shipped = {library.name for library in libraries}

    # For a CUDA wheel the undefined-symbol half of this check cannot be sound. When
    # the CUDA runtime is reachable only through LD_LIBRARY_PATH, which is how the
    # separate nvidia packages provide it, every CUDA entry point is reported
    # unresolved. Matching those by library name is not possible either, because ldd
    # names the library that references a symbol, not the one that would provide it.
    # The missing-library half below still runs.
    skip_undefined = _needs_external_cuda_runtime(package_dir)
    if skip_undefined:
        print(
            "- a CUDA wheel gets its CUDA runtime from the environment, so undefined "
            "symbol reporting is skipped for it"
        )

    # A dependency is only excusable when the wheel ships it AND the loader can
    # actually reach it from the library that needs it. Loaded-later extensions
    # such as the Torch libraries are the real exception: they resolve once the
    # Python package that owns them is imported. Anything the wheel itself ships
    # must resolve here, because a RUNPATH applies to the library carrying it and
    # is not inherited on behalf of a dependency's own dependencies.
    broken = {}
    unreachable = {}
    unresolved = {}
    for library in libraries:
        resolved = subprocess.run(
            # -r resolves data and function symbols too, not just the NEEDED
            # entries. A SHARED link does not error on undefined symbols, so
            # without this an under-linked library passes here and fails at first
            # use instead.
            ["ldd", "-r", str(library)],
            capture_output=True,
            text=True,
            check=False,
            # Any LD_LIBRARY_PATH in the build environment would paper over a
            # RUNPATH the shipped library is actually missing.
            env={
                key: value
                for key, value in os.environ.items()
                if key != "LD_LIBRARY_PATH"
            },
        )
        # ldd reports missing libraries on stdout but undefined symbols on stderr,
        # so both streams matter.
        combined = resolved.stdout + resolved.stderr
        missing = [
            line.split("=>")[0].strip()
            for line in combined.splitlines()
            if "not found" in line
        ]
        # Interpreter symbols are excluded rather than whole files. A library that
        # is loaded by Python, whether a extension module or an ahead-of-time
        # plugin, resolves those only once an interpreter is running, so ldd can
        # never resolve them and their absence says nothing about packaging.
        # Filtering the symbols rather than guessing from the file name keeps the
        # check active for everything else those libraries need.
        undefined = [
            line.strip()
            for line in combined.splitlines()
            if "undefined symbol" in line
            and not re.search(r"undefined symbol:\s+_?Py", line)
            # On a CUDA wheel the CUDA entry points are unresolved because the runtime
            # comes from the environment, so only those are excused. Blanking the whole
            # list instead would hide a genuinely under-linked symbol on the same wheel.
            # The leading-underscore forms matter too: nvcc emits host stubs such as
            # __cudaRegisterFatBinary for every compiled .cu file.
            and not (
                skip_undefined
                and re.search(
                    r"undefined symbol:\s+_*(cu|cuda|curand|cublas|cudnn)", line
                )
            )
        ]
        if undefined:
            unresolved[str(library.relative_to(package_dir))] = undefined[:5]

        absent = [
            name
            for name in missing
            if name not in shipped and not _provided_externally(name)
        ]
        present_but_unreachable = [name for name in missing if name in shipped]
        if absent:
            broken[str(library.relative_to(package_dir))] = absent
        if present_but_unreachable:
            unreachable[str(library.relative_to(package_dir))] = present_but_unreachable

    assert not broken, (
        "shipped libraries need dependencies that nothing provides, so they will "
        f"fail to load: {broken}"
    )
    assert not unreachable, (
        "shipped libraries need dependencies the wheel ships but the loader "
        "cannot reach from them, which usually means a missing RUNPATH entry: "
        f"{unreachable}"
    )
    assert not unresolved, (
        "shipped libraries reference symbols nothing provides, so they will fail "
        f"at first use rather than at load: {unresolved}"
    )
    print("✓ every shipped library resolves in an environment with torch present")


def test_shipped_libraries_resolve_without_build_tree() -> None:
    """A shipped library must resolve using only its relative runtime paths.

    Packaging copies binaries out of the build directory, so they still carry the
    absolute paths they were linked with. On the machine that produced the wheel
    those paths exist, which means a library whose relative path is wrong can still
    resolve and look correct. Anywhere else it would fail.

    Copy each library and its wheel-provided dependencies into a fresh tree that
    mirrors the wheel layout, drop every absolute runtime path, and check what is
    left is enough.
    """
    if shutil.which("ldd") is None or shutil.which("patchelf") is None:
        print("- ldd or patchelf unavailable, skipping the relocated load check")
        return

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    environment = {
        key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
    }

    with tempfile.TemporaryDirectory() as work_dir:
        root = Path(work_dir) / package_dir.name
        # Mirror the layout so a relative path such as $ORIGIN/../../lib still
        # points where it would in a real install.
        for library in libraries:
            target = root / library.relative_to(package_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, target)

        broken = {}
        for library in libraries:
            target = root / library.relative_to(package_dir)
            current = subprocess.run(
                ["patchelf", "--print-rpath", str(target)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            relative = [
                entry for entry in current.split(":") if entry.startswith("$ORIGIN")
            ]
            subprocess.run(
                ["patchelf", "--set-rpath", ":".join(relative), str(target)],
                # A failure here would leave the original absolute build paths in
                # place, and the check below would then pass by resolving through
                # them, which is exactly what this test exists to rule out.
                check=True,
            )
            resolved = subprocess.run(
                ["ldd", str(target)],
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            ).stdout
            shipped = {item.name for item in libraries}
            all_missing = [
                line.split("=>")[0].strip()
                for line in resolved.splitlines()
                if "not found" in line
            ]
            # Only wheel-provided dependencies are asserted on, because an external
            # one is expected to come from the environment. They are still reported,
            # since silently dropping them would hide a library that resolves only
            # through an absolute build path.
            missing = [name for name in all_missing if name in shipped]
            external = [name for name in all_missing if name not in shipped]
            if external:
                print(
                    f"- {library.relative_to(package_dir)} also needs "
                    f"{external} from the environment"
                )
            if missing:
                broken[str(library.relative_to(package_dir))] = missing

        assert not broken, (
            "shipped libraries only resolve their wheel-provided dependencies "
            "through absolute build paths, so they would fail on any other "
            f"machine: {broken}"
        )
    print("✓ every shipped library resolves without the build tree")


def _assert_single_definer(symbols, what: str, optional: bool = False) -> None:
    """Exactly one shipped library may define each of `symbols`.

    `optional` allows a component that is only present in some wheel flavors,
    such as an accelerator delegate, to be absent without failing.
    """
    assert shutil.which("nm") is not None, "nm is required to inspect the wheel"

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    assert libraries, f"no shared libraries found under {package_dir}"

    # Resolve every symbol first, so a component that is only half present is
    # reported rather than being mistaken for one that is absent entirely.
    found = {
        symbol: [lib for lib in libraries if _defines_symbol(lib, symbol)]
        for symbol in symbols
    }
    if optional and not any(found.values()):
        print(f"- no {what} in this wheel, skipping")
        return

    for symbol, definers in found.items():
        pretty = [str(lib.relative_to(package_dir)) for lib in definers]
        assert len(definers) == 1, (
            f"expected exactly one library to define {symbol}, found "
            f"{len(definers)}: {pretty}. More than one definition means the "
            f"process has more than one {what}."
        )
    print(f"✓ single {what} across {len(libraries)} shipped libraries")


def test_single_backend_registry() -> None:
    """Exactly one shipped library may define the backend registry."""
    _assert_single_definer(_REGISTRY_SYMBOLS, "backend registry")


def test_single_threadpool() -> None:
    """Exactly one shipped library may define the thread pool accessor."""
    _assert_single_definer(_THREADPOOL_SYMBOLS, "thread pool")


def test_single_kernel_registration() -> None:
    """Exactly one shipped library may define the merged CPU kernels."""
    _assert_single_definer(_KERNEL_SYMBOLS, "set of CPU kernels")
    # Ownership of the operator table, not just of a kernel implementation. A
    # second copy means a second table, and a static initializer registering into
    # a table nothing else reads shows up as an operator that is missing at run
    # time rather than as a link error.
    _assert_single_definer(_KERNEL_REGISTRY_SYMBOLS, "operator registry")


def test_single_xnnpack_delegate() -> None:
    """Exactly one shipped library may define the XNNPACK delegate."""
    _assert_single_definer(_XNNPACK_SYMBOLS, "XNNPACK delegate")


def test_single_cuda_delegate() -> None:
    """Exactly one shipped library may define the CUDA delegate, if present.

    Presence is decided from the shipped library file, not from whether the symbol
    resolves. Inferring absence from a missing symbol means a rename on a CUDA
    wheel would skip the check instead of failing it.
    """
    package_dir = _installed_package_dir()
    if not list(package_dir.rglob("libexecutorch_cuda_backend.so*")):
        print("- no CUDA delegate in this wheel, skipping")
        return
    _assert_single_definer(_CUDA_SYMBOLS, "CUDA delegate")


def test_cpp_consumer(work_dir: Path) -> None:
    """A standalone C++ app builds and runs against the installed wheel."""
    assert shutil.which("cmake") is not None, "cmake is required to build a consumer"

    package_dir = _installed_package_dir()
    config = package_dir / "share" / "cmake" / "executorch-config.cmake"
    assert config.is_file(), f"wheel is missing its CMake package config: {config}"

    source_dir = work_dir / "consumer"
    build_dir = work_dir / "consumer-build"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "consumer.cpp").write_text(_CONSUMER_SOURCE)
    (source_dir / "CMakeLists.txt").write_text(_CONSUMER_CMAKE)

    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={config.parent}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)

    consumer = build_dir / "consumer"
    # No LD_LIBRARY_PATH: the imported target is responsible for making the
    # shipped runtime findable.
    environment = {
        key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
    }
    subprocess.run([str(consumer)], check=True, env=environment)
    print("✓ C++ consumer builds and runs against the installed wheel")

    _assert_runs_relocated(consumer, package_dir, work_dir, environment)

    assert shutil.which("readelf") is not None, "readelf is required to check the ELF"

    dynamic = subprocess.run(
        ["readelf", "-d", str(consumer)], capture_output=True, text=True, check=True
    ).stdout
    assert "libexecutorch.so" in dynamic, (
        "the consumer does not depend on the shipped runtime; "
        f"dynamic section was:\n{dynamic}"
    )
    assert "$ORIGIN" in dynamic, (
        "the consumer has no $ORIGIN-relative RUNPATH, so it is not "
        f"relocatable; dynamic section was:\n{dynamic}"
    )
    print("✓ consumer depends on the shipped runtime with a relocatable RUNPATH")


def _assert_runs_relocated(consumer, package_dir, work_dir, environment) -> None:
    """The app still runs after being moved away from the wheel.

    Building in place leaves an absolute path to the wheel's lib directory in the
    binary's RUNPATH, which resolves the runtime no matter what `$ORIGIN` says.
    Copying the app next to a copy of the runtime, with that absolute entry
    removed, is what actually proves the package is relocatable.

    The layout mirrors what the package config supports: the app in `bin/` with
    the libraries in a sibling `lib/`, which is what `$ORIGIN/../lib` resolves.
    """
    if shutil.which("patchelf") is None:
        print("- patchelf not available, skipping the relocated run")
        return

    deploy = work_dir / "deployed"
    (deploy / "bin").mkdir(parents=True, exist_ok=True)
    (deploy / "lib").mkdir(parents=True, exist_ok=True)
    moved = deploy / "bin" / consumer.name
    shutil.copy2(consumer, moved)
    for library in (package_dir / "lib").glob("*.so*"):
        shutil.copy2(library, deploy / "lib" / library.name)

    # Keep only the $ORIGIN-relative entries, so nothing absolute can help.
    current = subprocess.run(
        ["patchelf", "--print-rpath", str(moved)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    relative = [entry for entry in current.split(":") if entry.startswith("$ORIGIN")]
    assert relative, (
        "the consumer has no $ORIGIN-relative RUNPATH entry, so it cannot be "
        f"relocated; RUNPATH was: {current}"
    )
    subprocess.run(
        ["patchelf", "--set-rpath", ":".join(relative), str(moved)], check=True
    )

    subprocess.run([str(moved)], check=True, env=environment, cwd=str(deploy))
    print("✓ consumer still runs when deployed beside a copy of the runtime")


def test_python_extensions_import() -> None:
    """Every shipped Python extension must import from a clean environment.

    The symbol and dependency checks work on the files. This covers the other
    half: an extension can be packaged correctly and still fail to load because a
    runtime path does not reach one of its dependencies. Run in a subprocess with
    `LD_LIBRARY_PATH` removed so a value from the build environment cannot supply
    a path the shipped library is missing.

    A CUDA wheel is the documented exception. Its libraries need the CUDA runtime,
    which the wheel deliberately does not bundle, so the environment has to provide
    it and removing the search path would fail for a reason that is by design.
    """
    modules = [
        "executorch.extension.pybindings.portable_lib",
        "executorch.extension.training",
    ]
    # Torch has to be installed, the same as for the dependency check: these
    # extensions link it, so without it they cannot import for a reason that says
    # nothing about packaging.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the extension import check")
        return
    package_dir = _installed_package_dir()
    if _needs_external_cuda_runtime(package_dir):
        print(
            "- a CUDA wheel needs the CUDA runtime from the environment, so the "
            "clean-environment import check does not apply"
        )
        return
    environment = {
        key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
    }
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if result.returncode == 0:
            print(f"✓ {module} imports from a clean environment")
            continue
        # A Python dependency that is simply not installed here, including torch,
        # says nothing about how the wheel was built. Only a failure to load a
        # native library does.
        # A Python package this environment simply does not have says nothing
        # about how the wheel was built. Match only that shape, so a native load
        # failure reported as ModuleNotFoundError is still caught below.
        missing_python_package = re.search(
            r"ModuleNotFoundError: No module named '(?!executorch)", result.stderr
        )
        if missing_python_package:
            print(f"- {module} needs a package this environment lacks, skipping")
            continue
        # Anything else is a real failure to load what the wheel ships: a missing
        # native library, an unresolved symbol, or an ABI mismatch.
        raise AssertionError(
            f"{module} ships in the wheel but does not import: "
            f"{result.stderr.strip()[-500:]}"
        )


_CUSTOM_OP_SOURCE = """\
// A custom operator, built the way an out-of-tree project builds one: against the
// shipped Python extension rather than an ExecuTorch source tree.
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace {

executorch::aten::Tensor& custom_double_out(
    executorch::runtime::KernelRuntimeContext& context,
    const executorch::aten::Tensor& input,
    executorch::aten::Tensor& out) {
  (void)context;
  const float* in = input.const_data_ptr<float>();
  float* dst = out.mutable_data_ptr<float>();
  for (ssize_t i = 0; i < input.numel(); ++i) {
    dst[i] = in[i] * 2.0f;
  }
  return out;
}

} // namespace

// The registration macro is the point of the check: it has to compile and resolve
// against the registry the shipped extension provides.
EXECUTORCH_LIBRARY(wheel_check, "custom_double.out", custom_double_out);
"""


_CUSTOM_OP_CMAKE = """\
cmake_minimum_required(VERSION 3.28)
project(custom_op_check CXX)

find_package(executorch REQUIRED)

add_library(custom_op_check SHARED custom_op.cpp)
# The legacy contract: a custom-op library links the shipped Python extension,
# which owns the operator registry it registers into.
target_link_libraries(custom_op_check PRIVATE _portable_lib)
"""


def test_custom_op_compiles(work_dir: Path) -> None:
    """A custom operator compiles and links against the shipped extension.

    This is how an out-of-tree project adds its own kernels, and it points at the
    Python extension rather than the runtime, so it is not covered by the consumer
    check above.
    """
    assert shutil.which("cmake") is not None, "cmake is required to build a consumer"

    package_dir = _installed_package_dir()
    if not list(package_dir.glob("extension/pybindings/_portable_lib*")):
        print("- the wheel ships no Python extension, skipping the custom op check")
        return

    source_dir = work_dir / "custom-op"
    build_dir = work_dir / "custom-op-build"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "custom_op.cpp").write_text(_CUSTOM_OP_SOURCE)
    (source_dir / "CMakeLists.txt").write_text(_CUSTOM_OP_CMAKE)

    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={package_dir}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert configure.returncode == 0, (
        "a custom operator project cannot configure against the wheel: "
        f"{(configure.stderr or configure.stdout).strip()[-600:]}"
    )

    compiled = subprocess.run(
        ["cmake", "--build", str(build_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, (
        "a custom operator does not compile or link against the shipped extension: "
        f"{(compiled.stderr or compiled.stdout).strip()[-800:]}"
    )
    assert list(build_dir.rglob("libcustom_op_check.so")) or list(
        build_dir.rglob("custom_op_check.dll")
    ), "the custom operator library was not produced"
    print("✓ a custom operator compiles against the shipped Python extension")


def _find_wheel_files() -> list:
    """The built wheel files, searched where a build actually leaves them.

    WHEEL_DIR is honoured when set, but it is not set in the wheel-build job, so the
    usual output directories are searched too. Without this the check has nothing to
    inspect and skips.
    """
    candidates = []
    configured = os.environ.get("WHEEL_DIR")
    if configured:
        candidates.append(Path(configured))
    # The build leaves the wheel in dist/ at the repository root, and this file sits at a
    # fixed depth below that root, so the location follows from __file__ rather than from
    # the current directory. The release job runs the smoke test from the workspace above
    # the repository, where a cwd-relative guess finds nothing.
    #
    # Guarded because a copy of this file can live outside that layout, where indexing
    # past the available parents would raise instead of falling through to the other
    # candidates.
    here = Path(__file__).resolve()
    repository_root = here.parents[3] if len(here.parents) > 3 else here.parent
    candidates += [
        repository_root / "dist",
        Path.cwd() / "dist",
        Path.cwd(),
        repository_root / "wheelhouse",
    ]
    for directory in candidates:
        try:
            found = sorted(directory.glob("executorch-*.whl"))
        except OSError:
            continue
        if found:
            return found
    return []


def test_wheel_platform_tag() -> None:
    """The wheel's declared platform tag must match what its libraries need.

    A library that quietly picks up a newer dependency, or a newer minimum glibc,
    makes the wheel unusable on machines the tag says it supports. auditwheel is
    the tool that decides this, so ask it rather than guessing.

    Only a contradiction between the tag and the contents fails here. Reports about
    instruction set extensions are left to the caller, because a prebuilt tool that
    ships in the wheel can legitimately require a newer baseline than the tag
    implies.
    """
    # Not installed here on purpose. A release check should not need the network or
    # change the environment it is verifying, so auditwheel is provisioned by the wheel
    # build script alongside the other prerequisites.
    if importlib.util.find_spec("auditwheel") is None:
        print("- auditwheel is not installed, skipping the platform tag check")
        return

    wheels = _find_wheel_files()
    if not wheels:
        print("- no wheel file to inspect, skipping the platform tag check")
        return

    result = subprocess.run(
        [sys.executable, "-m", "auditwheel", "show", str(wheels[-1])],
        capture_output=True,
        text=True,
        check=False,
    )
    # auditwheel wraps its verdict across lines, so compare on collapsed
    # whitespace rather than the literal output.
    combined = " ".join((result.stdout + result.stderr).split())
    match = re.search(
        r'consistent with the following platform tag: "([^"]+)"', combined
    )
    assert match, (
        "auditwheel reported no platform tag for the wheel, so its contents could "
        f"not be checked against what it claims: {combined[-400:]}"
    )
    # The tag auditwheel derives from the contents has to be the one the file name
    # claims. A wheel that names a stricter tag than its libraries support installs
    # on machines it cannot actually run on.
    claimed = wheels[-1].name.split("-")[-1].removesuffix(".whl")
    assert match.group(1) in claimed, (
        f"the wheel claims platform tag {claimed} but its contents only support "
        f"{match.group(1)}"
    )
    print(f"✓ the wheel contents match its declared platform tag {match.group(1)}")


def test_no_absolute_runtime_paths() -> None:
    """No shipped library may carry an absolute runtime search path.

    Packaging copies libraries out of the build tree rather than installing them,
    so anything CMake recorded at build time ships as-is. An absolute entry both
    names the build machine and points somewhere that will not exist for a user.

    The check reads the shipped file directly, with nothing stripped, which is what
    a user actually receives.
    """
    if shutil.which("patchelf") is None:
        print("- patchelf unavailable, skipping the runtime path check")
        return

    package_dir = _installed_package_dir()
    offenders = {}
    for library in sorted(package_dir.rglob("*.so*")):
        if not library.is_file() or library.is_symlink():
            continue
        result = subprocess.run(
            ["patchelf", "--print-rpath", str(library)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        absolute = [
            entry for entry in result.stdout.strip().split(":") if entry.startswith("/")
        ]
        if absolute:
            offenders[str(library.relative_to(package_dir))] = absolute

    assert not offenders, (
        "shipped libraries carry absolute runtime search paths, so they are not "
        f"relocatable and name the build machine: {offenders}"
    )
    print("✓ no shipped library carries an absolute runtime search path")


_COMPONENT_CONSUMER_CMAKE = """\
cmake_minimum_required(VERSION 3.28)
project(component_consumer CXX)

find_package(executorch REQUIRED)

add_executable(component_consumer consumer.cpp)
target_link_libraries(component_consumer PRIVATE executorch::runtime)

# Link every component this wheel offers, and report which ones those are so the test
# can check the result. Guarded individually because the set depends on the wheel.
foreach(_component threadpool kernels xnnpack_backend cuda_backend)
  if(TARGET executorch::${_component})
    target_link_libraries(component_consumer PRIVATE executorch::${_component})
    # Report the library file, not just the target name: the two differ, and the test
    # needs the file name to look for in the built binary.
    get_target_property(_location executorch::${_component} IMPORTED_LOCATION)
    get_filename_component(_file "${_location}" NAME)
    message(STATUS "LINKED_COMPONENT=${_component}:${_file}")
  endif()
endforeach()
"""


def test_component_targets_link(work_dir: Path) -> None:
    """Every component target the wheel offers must link and be retained.

    A component library exists to register something, so nothing in the application
    references a symbol from it. That is exactly the case a normal link drops, which is
    why the targets carry retention options. This checks the options do their job
    instead of trusting them.
    """
    assert shutil.which("cmake") is not None, "cmake is required to build a consumer"
    if shutil.which("readelf") is None:
        print("- readelf unavailable, skipping the component link check")
        return

    package_dir = _installed_package_dir()
    source_dir = work_dir / "components"
    build_dir = work_dir / "components-build"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "consumer.cpp").write_text(_CONSUMER_SOURCE)
    (source_dir / "CMakeLists.txt").write_text(_COMPONENT_CONSUMER_CMAKE)

    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={package_dir}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert configure.returncode == 0, (
        "a consumer linking the component targets cannot configure: "
        f"{(configure.stderr or configure.stdout).strip()[-500:]}"
    )
    linked = dict(
        match.split(":", 1)
        for match in re.findall(r"LINKED_COMPONENT=(\S+)", configure.stdout)
    )
    # Compare against the libraries the wheel actually ships. Checking only the targets
    # that exist would pass a component whose target silently failed to be created,
    # which is the failure mode a glob-based definition has.
    expected = _shipped_components(package_dir)
    absent = sorted(expected - set(linked))
    assert not absent, (
        f"the wheel ships libraries for {absent} but the package config defines no "
        "target for them, so a consumer cannot link them"
    )
    if not linked:
        print("- this wheel offers no component targets, skipping the component check")
        return

    built = subprocess.run(
        ["cmake", "--build", str(build_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, (
        "a consumer linking the component targets does not build: "
        f"{(built.stderr or built.stdout).strip()[-700:]}"
    )

    consumer = build_dir / "component_consumer"
    needed = subprocess.run(
        ["readelf", "-d", str(consumer)], capture_output=True, text=True, check=True
    ).stdout
    # Each component has to appear in DT_NEEDED. Absent means the retention options did
    # not hold and whatever the library registers would never happen at runtime.
    dropped = [
        component for component, library in linked.items() if library not in needed
    ]
    assert not dropped, (
        f"components {dropped} were linked but do not appear in the consumer's "
        "DT_NEEDED, so their registration would never run"
    )

    # Run it, so the check covers a registration constructor actually firing rather than
    # only the library being named in DT_NEEDED.
    #
    # On a CUDA wheel the consumer links the CUDA delegate, which needs a CUDA runtime
    # the wheel does not bundle. Stripping the search path would then fail for a reason
    # that is by design, so the environment is left alone there, matching what the other
    # checks do for the same wheel.
    if _needs_external_cuda_runtime(package_dir):
        environment = dict(os.environ)
        print(
            "- a CUDA wheel needs the CUDA runtime from the environment, so the "
            "consumer runs with the search path left in place"
        )
    else:
        environment = {
            key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
        }
    run = subprocess.run(
        [str(consumer)], capture_output=True, text=True, check=False, env=environment
    )
    assert run.returncode == 0, (
        "the consumer links every component but does not run: "
        f"{(run.stderr or run.stdout).strip()[-400:]}"
    )
    print(f"✓ every offered component links, is retained, and runs: {sorted(linked)}")


def run_tests(work_dir: Path) -> None:
    test_shipped_libraries_load()
    test_shipped_libraries_resolve_without_build_tree()
    test_single_backend_registry()
    test_python_extensions_import()
    test_wheel_platform_tag()
    test_custom_op_compiles(work_dir)
    test_no_absolute_runtime_paths()
    test_single_threadpool()
    test_single_kernel_registration()
    test_single_xnnpack_delegate()
    test_single_cuda_delegate()
    test_cpp_consumer(work_dir)
    test_component_targets_link(work_dir)
