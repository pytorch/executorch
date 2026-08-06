# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that the wheel ships its runtime as separate shared libraries.

The Python bindings extension used to contain the runtime, the registries, the
CPU kernels, the XNNPACK delegate and the profiler, all statically linked into
one file. It now links them as shared libraries the wheel ships alongside it.
These checks run against the installed wheel only; they never look at the source
tree's build directory, because a checkout on the module search path makes every
check below pass while inspecting the wrong thing.

The properties verified here are the ones the split exists to create:

1. Each component has exactly one definer, and it is the library that is meant
   to own it. Counting definers alone is not enough: the monolithic layout also
   had exactly one of each, inside the Python extension.
2. The extension contains none of those components and depends on every shipped
   library instead.
3. Every shipped library loads with no absolute runtime search path, including
   after being moved, so the wheel is relocatable rather than only working on
   the machine that built it.
"""

import importlib.util
import json
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

# `nm -DC` prints "<hexaddr> <kind> <name>" for a definition and
# "                 U <name>" for an undefined reference.
_DEFINED = re.compile(r"^[0-9a-fA-F]+\s+(?P<kind>[A-Za-z])\s+(?P<name>.+)$")

# Symbol kinds that mean the object owns the code or storage.
_OWNING_KINDS = frozenset("TtBbDdGgSsRrWV")


def _declared_requirements() -> set:
    """The top-level distributions the installed wheel declares as requirements.

    Used to tell a package this environment simply does not have from one the wheel
    said it needed, because only the second is a packaging defect.
    """
    try:
        from importlib.metadata import requires
    except ImportError:  # pragma: no cover
        return set()
    try:
        declared = requires("executorch") or []
    except Exception:
        return set()
    names = set()
    for requirement in declared:
        name = re.split(r"[\s;\[<>=!~(]", requirement.strip(), maxsplit=1)[0]
        if name:
            # Distribution names use hyphens where the import name uses underscores.
            names.add(name)
            names.add(name.replace("-", "_"))
    return names


def _installed_package_dir() -> Path:
    """The installed executorch package, never the source checkout.

    Enforced rather than assumed. Python puts the working directory on the module
    search path, so running from a checkout resolves `executorch` to the source
    tree, where there are no shipped libraries and every check below passes while
    testing nothing. That is worse than a failure, because it looks like a pass.
    """
    import executorch

    paths = list(executorch.__path__)
    directory = Path(paths[0]).resolve()
    assert "site-packages" in directory.parts or "dist-packages" in directory.parts, (
        f"executorch resolves to {directory}, which is not an installed package. "
        "Run this from a directory that contains no executorch checkout, or the "
        "checks silently inspect the source tree instead of the wheel."
    )
    return directory


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
    if result.returncode != 0:
        # A file that is not an object file at all is not this check's concern: something whose
        # name merely ends in .so must not abort the run. A shipped library the reader cannot
        # parse is different, because a real definition could be hiding inside it, and reporting
        # "defines nothing" would let a duplicate pass. The ELF magic bytes tell them apart
        # without depending on the reader's wording.
        with library.open("rb") as handle:
            is_object_file = handle.read(4) == b"\x7fELF"
        assert not is_object_file, (
            f"nm could not read {library.name}, which is a shipped object file, so the symbol "
            f"checks cannot be trusted: {result.stderr.strip()[:200]}"
        )
        return False
    for line in result.stdout.splitlines():
        if symbol not in line:
            continue
        match = _DEFINED.match(line)
        if (
            match
            and match.group("name").startswith(symbol)
            and match.group("kind") in _OWNING_KINDS
        ):
            return True
    return False


def _assert_single_definer(symbols, what: str, owner: str | None = None) -> None:
    """At most one shipped library may define each of `symbols`.

    The owner is named where one is expected, because counting definers alone does
    not prove the split happened: the monolithic layout has exactly one definer too,
    the Python extension. Requiring the symbol to live in the library that is
    supposed to own it is what distinguishes the two.

    A component the wheel does not ship at all is a valid configuration, not a
    fault. Delegates and kernel sets are build options, so a wheel built without one
    has zero definers and is reported as such. What must never happen is two.
    """
    assert shutil.which("nm") is not None, "nm is required to inspect the wheel"

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    assert libraries, f"no shared libraries found under {package_dir}"

    # Every symbol is resolved before anything is reported, so a component that is only
    # half present is described as such rather than looking like one that is absent.
    found = {
        symbol: [lib for lib in libraries if _defines_symbol(lib, symbol)]
        for symbol in symbols
    }
    # A component is either wholly present or wholly absent. Some symbols defined and
    # others not means a partial build, which is neither of those and is a fault.
    present = {symbol for symbol, definers in found.items() if definers}
    if not present:
        print(f"- this wheel ships no {what}, nothing to check")
        return
    assert len(present) == len(found), (
        f"the wheel defines only part of the {what}: {sorted(present)} are present "
        f"and {sorted(set(found) - present)} are not, so it is neither shipped nor "
        "absent"
    )

    for symbol, definers in found.items():
        pretty = [str(lib.relative_to(package_dir)) for lib in definers]
        assert len(definers) == 1, (
            f"expected at most one library to define {symbol}, found "
            f"{len(definers)}: {pretty}. More than one definition means the "
            f"process has more than one {what}."
        )
        if owner is not None:
            assert definers[0].name.startswith(owner), (
                f"{symbol} is defined by {pretty[0]}, but it belongs in {owner}. One "
                "definer is not enough on its own: the monolithic layout this change "
                "replaces also had exactly one, inside the Python extension."
            )
    where = f" owned by {owner}" if owner else ""
    print(f"✓ single {what}{where} across {len(libraries)} shipped libraries")


def test_single_backend_registry() -> None:
    """Exactly one shipped library may define the backend registry."""
    _assert_single_definer(_REGISTRY_SYMBOLS, "backend registry", "libexecutorch.so")


def test_single_threadpool() -> None:
    """Exactly one shipped library may define the thread pool accessor."""
    _assert_single_definer(
        _THREADPOOL_SYMBOLS, "thread pool", "libexecutorch_threadpool.so"
    )


def test_single_kernel_registration() -> None:
    """Exactly one shipped library may define the merged CPU kernels."""
    _assert_single_definer(
        _KERNEL_SYMBOLS, "set of CPU kernels", "libexecutorch_kernels_optimized.so"
    )
    # Ownership of the operator table, not just of a kernel implementation. A
    # second copy means a second table, and a static initializer registering into
    # a table nothing else reads shows up as an operator that is missing at run
    # time rather than as a link error.
    _assert_single_definer(
        _KERNEL_REGISTRY_SYMBOLS, "operator registry", "libexecutorch.so"
    )


def test_single_xnnpack_delegate() -> None:
    """Exactly one shipped library may define the XNNPACK delegate."""
    _assert_single_definer(
        _XNNPACK_SYMBOLS, "XNNPACK delegate", "libexecutorch_backend_xnnpack.so"
    )


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

    The list is discovered from the installed package rather than written here, so
    an extension added later is covered without anyone remembering to add it. A
    hardcoded list is how the ones this change relinked went untested.
    """
    package_dir = _installed_package_dir()
    modules = []
    for extension in sorted(package_dir.rglob("*.so")):
        # Only Python extensions, which carry the interpreter's suffix. The plain
        # shared libraries under lib/ are checked by the load test instead.
        if ".cpython-" not in extension.name:
            continue
        relative = extension.relative_to(package_dir).parent
        module = extension.name.split(".", 1)[0]
        modules.append(
            ".".join(["executorch", *relative.parts, module]).replace("..", ".")
        )
    assert modules, "the wheel ships no Python extension, which cannot be right"

    # Torch has to be installed, the same as for the dependency check: these
    # extensions link it, so without it they cannot import for a reason that says
    # nothing about packaging.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the extension import check")
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
        # A Python package this environment simply does not have says nothing about
        # how the wheel was built, PROVIDED the wheel does not claim to require it.
        # Match only that shape, so a native load failure reported as
        # ModuleNotFoundError is still caught below.
        missing_python_package = re.search(
            r"ModuleNotFoundError: No module named '(?!executorch)([\w.]+)",
            result.stderr,
        )
        if missing_python_package:
            absent = missing_python_package.group(1).split(".")[0]
            # A package the wheel declares as a requirement should have been
            # installed with it, so its absence is a packaging defect rather than a
            # property of this environment. Skipping there would report a broken
            # dependency list as coverage.
            assert absent not in _declared_requirements(), (
                f"{module} cannot import because {absent} is missing, and the wheel "
                "declares it as a requirement, so installing the wheel should have "
                "provided it"
            )
            print(f"- {module} needs {absent}, which this environment lacks, skipping")
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
# The runtime headers include c10 headers, which belong to torch rather than to
# this wheel, so an out-of-tree operator project supplies them the same way it
# supplies torch itself. The package config does not and should not ship them.
find_package(Torch REQUIRED)

add_library(custom_op_check SHARED custom_op.cpp)
# The legacy contract: a custom-op library links the shipped Python extension,
# which owns the operator registry it registers into.
target_link_libraries(custom_op_check PRIVATE _portable_lib)
target_include_directories(custom_op_check PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_features(custom_op_check PRIVATE cxx_std_20)
"""


# Libraries that belong to torch rather than to this wheel. A library here resolves when the
# Python package that owns it is imported, so it is not something this wheel can or should ship.
_TORCH_LIBRARY_PREFIXES = (
    "libtorch",
    "libc10",
    "libshm",
    "libgomp",
    "libcudnn",
    "libcublas",
)


def _is_torch_library(name: str) -> bool:
    return name.startswith(_TORCH_LIBRARY_PREFIXES)


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
    # Torch has to be installed for this to mean anything: several shipped libraries
    # depend on it and resolve once it is imported. Without it every one of them looks
    # broken, which would report a packaging fault that does not exist.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the load check")
        return

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    shipped = {library.name for library in libraries}

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
        ]
        if undefined:
            unresolved[str(library.relative_to(package_dir))] = undefined[:5]
        # Torch's own libraries are the documented exception. They are not in this wheel, and a
        # library that needs them resolves once the Python package owning them is imported, which
        # is how every accelerator and AOT library in this package is used. Treating them as
        # missing fails a wheel that works, and it fires only where torch installs its libraries
        # somewhere the plain loader search does not reach.
        absent = [
            name
            for name in missing
            if name not in shipped and not _is_torch_library(name)
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

    # Torch's CMake directory as well as the wheel's, because the runtime headers
    # include c10 headers that belong to torch. A real out-of-tree project does the
    # same; the package config has no business shipping another project's headers.
    prefixes = [str(package_dir / "share" / "cmake")]
    if importlib.util.find_spec("torch") is not None:
        import torch

        prefixes.append(str(Path(torch.__path__[0]) / "share" / "cmake"))
    else:
        print("- torch is not installed, skipping the custom op check")
        return

    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_PREFIX_PATH=" + ";".join(prefixes),
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
    produced = list(build_dir.rglob("libcustom_op_check.so")) or list(
        build_dir.rglob("custom_op_check.dll")
    )
    assert produced, "the custom operator library was not produced"

    # Loaded, not just built. A shared library on Linux is allowed to have
    # unresolved symbols, so an under-linked custom operator links successfully and
    # fails only when something dlopens it and its registration initialiser runs.
    # That is exactly the failure this contract exists to prevent.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, so the custom operator is only built")
        return
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch\n"
            "from executorch.extension.pybindings import portable_lib\n"
            f"torch.ops.load_library({str(produced[0])!r})\n"
            "print('loaded')",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            key: value for key, value in os.environ.items() if key != "LD_LIBRARY_PATH"
        },
    )
    assert loaded.returncode == 0, (
        "a custom operator built against the shipped extension cannot be loaded, so "
        "it would fail at first use rather than at link time: "
        f"{(loaded.stderr or loaded.stdout).strip()[-800:]}"
    )
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
    if importlib.util.find_spec("auditwheel") is None:
        print("- auditwheel unavailable, skipping the platform tag check")
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


def test_extension_contains_no_component() -> None:
    """The Python extension must link the components, not contain them.

    This is the property the change exists to create, and no count of definers
    proves it: the monolithic layout has exactly one definer of every symbol too,
    inside the extension. The direct statement is that the extension defines none of
    what the shipped libraries own, and records a dependency on each instead.
    """
    assert shutil.which("nm") is not None, "nm is required to inspect the wheel"
    if shutil.which("readelf") is None:
        print("- readelf unavailable, skipping the extension composition check")
        return

    package_dir = _installed_package_dir()
    extensions = sorted(
        (package_dir / "extension" / "pybindings").glob("_portable_lib.*.so")
    )
    assert len(extensions) == 1, f"expected one _portable_lib, found {extensions}"
    extension = extensions[0]

    lib_dir = package_dir / "lib"
    if not lib_dir.is_dir():
        print("- this wheel ships no lib directory, nothing to check")
        return

    # Every symbol group a shipped library owns. The extension holding any of these
    # means it still carries its own copy of that component.
    owned = (
        *_REGISTRY_SYMBOLS,
        *_THREADPOOL_SYMBOLS,
        *_KERNEL_SYMBOLS,
        *_KERNEL_REGISTRY_SYMBOLS,
        *_XNNPACK_SYMBOLS,
    )
    contained = [symbol for symbol in owned if _defines_symbol(extension, symbol)]
    assert not contained, (
        f"{extension.name} defines {contained}, which the shipped libraries own. The "
        "extension is supposed to link them rather than contain them, so this is the "
        "monolithic layout the split removes."
    )

    # And it has to actually depend on each shipped library. Defining nothing while
    # depending on nothing would be an extension that cannot work at all.
    needed = {
        line.split("[", 1)[1].rstrip("]").strip()
        for line in subprocess.run(
            ["readelf", "-d", str(extension)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if "NEEDED" in line
    }
    shipped = {path.name for path in lib_dir.glob("lib*.so.*") if path.is_file()}
    unused = sorted(shipped - needed)
    assert not unused, (
        f"the wheel ships {unused} but {extension.name} does not depend on them, so "
        "either they are dead weight or a retention option did not hold"
    )
    print(
        f"✓ {extension.name} ({extension.stat().st_size // 1024} KiB) contains no "
        f"component and depends on all {len(shipped)} shipped libraries"
    )


def test_shipped_libraries_are_all_versioned() -> None:
    """Every library in lib/ must be one this build could have produced.

    Packaging copies binaries out of a staging directory rather than running an
    install step, so anything left there from an earlier build ships too. That
    really happened: a wheel picked up three libraries from a different revision
    and still passed every symbol check, because those checks only ask how many
    definers a symbol has, never whether a file belongs in the wheel at all.

    Three properties catch it. A library built by this project carries VERSION and
    SOVERSION, so its file name ends in a major. Its recorded soname matches its
    file name, or a consumer records a dependency the wheel does not contain. And
    its name is one packaging knows how to produce, which is what a leftover from
    an older layout fails.
    """
    package_dir = _installed_package_dir()
    lib_dir = package_dir / "lib"
    if not lib_dir.is_dir():
        print("- this wheel ships no lib directory, nothing to check")
        return

    shipped = sorted(p for p in lib_dir.glob("*.so*") if p.is_file())
    assert shipped, f"the wheel ships a lib directory with no libraries: {lib_dir}"

    # The names packaging can put here. Listed rather than derived because setup.py
    # names each one literally, and a file with any other name did not come from
    # this build. Which of them are present depends on the build options, so
    # absence is fine and an unknown name is not.
    known = {
        "libexecutorch.so",
        "libexecutorch_kernels_optimized.so",
        "libexecutorch_backend_xnnpack.so",
        "libexecutorch_threadpool.so",
        "libexecutorch_etdump.so",
    }
    unknown = sorted(
        p.name for p in shipped if p.name.split(".so")[0] + ".so" not in known
    )
    assert not unknown, (
        f"the wheel ships {unknown} under lib/, which packaging does not produce. "
        "A file packaging did not put there came from a stale staging directory, "
        "and it ships while looking correct to every other check."
    )

    # A bare .so pins no major, so nothing constrains what a consumer that linked
    # it will load later, which is the guarantee a soname exists to provide.
    unversioned = [p.name for p in shipped if not re.search(r"\.so\.\d+$", p.name)]
    assert not unversioned, (
        f"shipped libraries carry no major version: {unversioned}. Either the build "
        "did not set VERSION and SOVERSION, or these are leftovers from an earlier "
        "build that packaging copied out of a stale staging directory."
    )

    if shutil.which("readelf") is None:
        print(f"✓ {len(shipped)} shipped libraries are versioned")
        return

    # The recorded soname has to match the file name, or a consumer records a
    # dependency on a name the wheel does not contain.
    mismatched = {}
    for library in shipped:
        dynamic = subprocess.run(
            ["readelf", "-d", str(library)], capture_output=True, text=True, check=False
        ).stdout
        soname = next(
            (
                line.split("[", 1)[1].rstrip("]").strip()
                for line in dynamic.splitlines()
                if "SONAME" in line
            ),
            None,
        )
        if soname != library.name:
            mismatched[library.name] = soname
    assert not mismatched, (
        "shipped libraries record a soname that is not their file name, so a "
        f"consumer would look for a file the wheel does not ship: {mismatched}"
    )
    print(f"✓ {len(shipped)} shipped libraries are versioned with matching sonames")


_PARITY_MODEL = '''
import json
import sys

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)


class Net(torch.nn.Module):
    """Several operator kinds rather than one, so the run exercises the merged CPU
    kernels rather than a single add."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)

    def forward(self, x, image):
        a = torch.relu(self.linear(x))
        b = self.conv(image).flatten(1)
        return a.sum(dim=1, keepdim=True) + b.mean(dim=1, keepdim=True)


delegate = sys.argv[1] == "delegate"
torch.manual_seed(0)
model = Net().eval()
example = (torch.randn(2, 8), torch.randn(2, 1, 6, 6))
with torch.no_grad():
    expected = model(*example)

partitioners = []
if delegate:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    partitioners = [XnnpackPartitioner()]

program = to_edge_transform_and_lower(
    torch.export.export(model, example), partitioner=partitioners
).to_executorch()
buffer = program.buffer

actual = _load_for_executorch_from_buffer(buffer).forward(list(example))[0]
# Compared rather than merely run. The point of the split is that behaviour does
# not change, and only a numeric comparison shows that; a model that returns
# wrong values without erroring passes everything else.
torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

print(json.dumps({"delegated": delegate, "has_xnnpack": b"XnnpackBackend" in bytes(buffer)}))
'''


def test_model_matches_eager_pytorch(work_dir: Path) -> None:
    """A model exported and run through the bindings must match eager PyTorch.

    Twice: once plain, so the CPU kernels resolve from the shared library, and once
    delegated to XNNPACK, so the delegate does. The delegated program is also
    checked for the delegate's own identity, because a partitioner that claimed
    nothing would silently fall back to the CPU kernels and still match.

    Separate processes, because a fault in one export leaves state that makes the
    next look broken when it is not.
    """
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the eager comparison")
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    script = work_dir / "parity.py"
    script.write_text(_PARITY_MODEL)

    for mode in ("plain", "delegate"):
        result = subprocess.run(
            [sys.executable, str(script), mode],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(work_dir),
        )
        assert result.returncode == 0, (
            f"the {mode} model does not export, run, and match eager PyTorch: "
            f"{(result.stderr or result.stdout).strip()[-1500:]}"
        )
        report = json.loads(result.stdout.strip().splitlines()[-1])
        if mode == "delegate":
            assert report["has_xnnpack"], (
                "the delegated export produced a program with no XNNPACK partition, "
                "so the delegate was never exercised and the comparison only proves "
                "the CPU kernels work"
            )
        print(f"✓ the {mode} model matches eager PyTorch")


def run_tests(work_dir: Path) -> None:
    test_single_backend_registry()
    test_python_extensions_import()
    test_extension_contains_no_component()
    test_shipped_libraries_are_all_versioned()
    test_shipped_libraries_load()
    test_shipped_libraries_resolve_without_build_tree()
    test_wheel_platform_tag()
    test_custom_op_compiles(work_dir)
    test_no_absolute_runtime_paths()
    test_model_matches_eager_pytorch(work_dir)
    test_single_threadpool()
    test_single_kernel_registration()
    test_single_xnnpack_delegate()
