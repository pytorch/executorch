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

import importlib.metadata
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

# The quantized kernels, whose own library the wheel ships when they are built.
# A separate group because they have a separate owner, and because a wheel built
# without them ships neither the library nor these symbols.
# The CUDA delegate and its stream helper, for a wheel built from a CUDA index. The
# stream helper matters most: two copies means two notions of the caller's stream, so
# work queued through one is invisible to the other.
# A strongly defined symbol, chosen by reading the built library rather than guessed.
# The CudaBackend methods are emitted weak, and the definer count only looks at strong
# definitions, so naming one of those would report zero definers for a library that is
# plainly present.
_CUDA_BACKEND_SYMBOLS = ("executorch::backends::cuda::load_library",)
_CUDA_STREAM_SYMBOLS = ("executorch::extension::cuda::getCallerStream",)

_QUANTIZED_KERNEL_SYMBOLS = (
    "torch::executor::native::quantize_per_tensor_out",
    "torch::executor::native::dequantize_per_tensor_out",
)

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

# Third-party code the shipped libraries bundle rather than depend on. These are C
# symbols with default visibility, so a second copy in the same process is not a
# duplicate of ExecuTorch's own code but it is still two thread pools or two
# XNNPACK runtimes, and which one a caller reaches depends on load order.
#
# Checked separately from the wrapper symbols above because the wrappers can each
# have exactly one owner while the bundled code underneath them does not. That is
# the same failure the split exists to prevent, reached by a different route.
_BUNDLED_THREADPOOL_SYMBOLS = ("pthreadpool_create", "cpuinfo_initialize")
_BUNDLED_XNNPACK_SYMBOLS = ("xnn_create_runtime_v4",)

# A representative symbol from the profiler. A second definer means two event
# tracers, so a trace records only part of what ran.
_ETDUMP_SYMBOLS = ("executorch::etdump::ETDumpGen::ETDumpGen",)

# `nm -DC` prints "<hexaddr> <kind> <name>" for a definition and
# "                 U <name>" for an undefined reference.
_DEFINED = re.compile(r"^[0-9a-fA-F]+\s+(?P<kind>[A-Za-z])\s+(?P<name>.+)$")

# Symbol kinds that mean the object owns the code or storage.
_OWNING_KINDS = frozenset("TtBbDdGgSsRrWV")


def _declared_requirements() -> set:
    """Import names the installed wheel declares a requirement for.

    Used to tell a package this environment simply does not have from one the wheel
    said it needed, because only the second is a packaging defect.

    Two sources, because neither alone is sufficient. importlib's reverse map is
    authoritative where a distribution is INSTALLED, which is what makes
    PyYAML -> yaml, ruamel.yaml -> ruamel and hydra-core -> hydra resolve correctly.
    But it enumerates installed distributions only, so a declared dependency that is
    MISSING can never appear in it, which is precisely the case this function exists
    to catch. The transformed distribution name covers that case.
    """
    try:
        from importlib.metadata import packages_distributions, requires
    except ImportError:  # pragma: no cover
        return set()
    try:
        declared = requires("executorch") or []
    except Exception:
        return set()

    wanted, names = set(), set()
    for requirement in declared:
        # Only the distribution name, dropping any version specifier, extra or
        # environment marker.
        name = re.split(r"[\s;\[<>=!~(]", requirement.strip(), maxsplit=1)[0]
        if not name:
            continue
        wanted.add(name.lower().replace("-", "_").replace(".", "_"))
        # The likely import name, so a MISSING declared dependency is still
        # recognised as declared rather than silently skipped.
        names.add(name)
        names.add(name.replace("-", "_"))
        names.add(name.split(".")[0])

    for import_name, distributions in packages_distributions().items():
        for distribution in distributions:
            if distribution.lower().replace("-", "_").replace(".", "_") in wanted:
                names.add(import_name)
    return names


def _installed_package_dir() -> Path:
    """The installed executorch package, never the source checkout.

    Enforced rather than assumed. Python puts the working directory on the module
    search path, so running from a checkout resolves `executorch` to the source
    tree, where there are no shipped libraries and every check below passes while
    testing nothing. That is worse than a failure, because it looks like a pass.
    """
    import executorch

    paths = [Path(entry).resolve() for entry in executorch.__path__]
    # Every entry, not just the first. This is a namespace package, so a checkout on
    # the module search path adds a second entry, and a module can then resolve from
    # the checkout while the first entry still looks like a clean install.
    outside = [
        path
        for path in paths
        if "site-packages" not in path.parts and "dist-packages" not in path.parts
    ]
    assert not outside, (
        f"executorch also resolves through {outside}, which is not an installed "
        "package. Run this from a directory that contains no executorch checkout, "
        "or the checks silently inspect the source tree instead of the wheel."
    )
    assert len(paths) == 1, (
        f"executorch resolves through {len(paths)} paths ({paths}). Even when all of "
        "them are installs, a module could come from either, so which artifact is "
        "under test is ambiguous."
    )
    return paths[0]


def _tool(name: str):
    """Locate a build tool, including one pip installed beside this interpreter.

    `shutil.which` searches PATH only, and a virtual environment's `bin` is on PATH
    only when the environment is activated. These tests are normally run by invoking
    the interpreter directly, so a tool pip installed into that environment is present
    on disk and invisible to a PATH search.
    """
    found = shutil.which(name)
    if found:
        return found
    beside = Path(sys.executable).parent / name
    return str(beside) if beside.is_file() else None


def _shipped_shared_objects(package_dir: Path):
    return [
        path
        for path in sorted(package_dir.rglob("*.so*"))
        if path.is_file() and not path.is_symlink()
    ]


def _shipped_runtime_libraries(package_dir: Path):
    """The libraries the wheel ships under lib/, whatever it names them.

    One place, because three checks previously spelled the pattern themselves as
    `*.so.*` and every one of them silently stopped matching when the build moved to
    unversioned names. The failure was invisible: a check that finds nothing reports
    that the component is absent, which each of those treats as acceptable.

    Matches a versioned name too, so a build that does set SOVERSION is still found.
    """
    lib_dir = package_dir / "lib"
    if not lib_dir.is_dir():
        return []
    return [
        path
        for path in sorted(lib_dir.glob("lib*.so*"))
        if path.is_file() and not path.is_symlink()
    ]


def _defines_symbol(library: Path, symbol: str) -> bool:
    result = subprocess.run(
        [_tool("nm"), "-DC", str(library)], capture_output=True, text=True, check=False
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


def _is_export_only(library: Path) -> bool:
    """Whether a library exists to export a model rather than to run one.

    The ahead-of-time operator libraries register kernels into torch so a model can be
    exported, and they link torch to do it. They deliberately carry their own copy of
    the kernels, because export happens in a Python process that never loads the
    runtime libraries a C++ application links.

    Excluded from the single-owner checks for that reason. Counting them would report a
    duplicate for something that is not one, and the alternative, making them resolve
    the kernels from the shipped library, would mean an export-time library depending on
    a runtime layout it never uses.

    Matched by what the file links rather than by its name, so a library renamed later
    is still recognised.
    """
    if _tool("readelf") is None:
        return library.name.endswith("_aot_lib.so")
    dynamic = subprocess.run(
        [_tool("readelf"), "-d", str(library)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return "libtorch.so" in dynamic and "_aot_lib" in library.name


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
    assert _tool("nm") is not None, "nm is required to inspect the wheel"

    package_dir = _installed_package_dir()
    libraries = [
        library
        for library in _shipped_shared_objects(package_dir)
        if not _is_export_only(library)
    ]
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


# Each component the wheel ships as its own library, the symbols that identify it,
# and the library that must own them. `required` says whether the owner has to be
# present: the optimized kernels are optional, because a wheel built without them
# deliberately links the portable ops into the Python extension instead, which is a
# supported configuration rather than a duplicate.
#
# A table rather than one function per component, because the per-function form let
# one of them drift: it looked up its library with its own glob, which silently
# stopped matching when the libraries were renamed while the others kept working.
_OWNED_COMPONENTS = (
    ("backend registry", _REGISTRY_SYMBOLS, "libexecutorch.so", True),
    ("operator registry", _KERNEL_REGISTRY_SYMBOLS, "libexecutorch.so", True),
    ("thread pool", _THREADPOOL_SYMBOLS, "libexecutorch_threadpool.so", True),
    ("profiler", _ETDUMP_SYMBOLS, "libexecutorch_etdump.so", True),
    (
        "XNNPACK delegate",
        _XNNPACK_SYMBOLS,
        "libexecutorch_backend_xnnpack.so",
        True,
    ),
    (
        "set of CPU kernels",
        _KERNEL_SYMBOLS,
        "libexecutorch_kernels_optimized.so",
        False,
    ),
    (
        "set of quantized kernels",
        _QUANTIZED_KERNEL_SYMBOLS,
        "libexecutorch_kernels_quantized.so",
        False,
    ),
    (
        "CUDA delegate",
        _CUDA_BACKEND_SYMBOLS,
        "libexecutorch_backend_cuda.so",
        False,
    ),
    (
        "CUDA stream helper",
        _CUDA_STREAM_SYMBOLS,
        "libexecutorch_extension_cuda.so",
        False,
    ),
    # The third-party code these libraries bundle, checked separately from the
    # wrappers above. A wrapper can have a single owner while the implementation
    # underneath it is bundled into two of these, which is two real thread pools or
    # two XNNPACK runtimes.
    #
    # One copy among the libraries this wheel ships, which is what this change
    # controls. torch links the same projects and exports the same symbols, so the
    # process still holds two definitions and which one a caller reaches depends on
    # load order. Fixing that needs an explicit export list: hiding them wholesale
    # with --exclude-libs,ALL breaks aarch64, where the optimized kernels resolve
    # cpuinfo_initialize from the thread pool across a library boundary.
    (
        "bundled thread pool implementation",
        _BUNDLED_THREADPOOL_SYMBOLS,
        "libexecutorch_threadpool.so",
        True,
    ),
    (
        "bundled XNNPACK runtime",
        _BUNDLED_XNNPACK_SYMBOLS,
        "libexecutorch_backend_xnnpack.so",
        True,
    ),
)


def test_each_component_has_one_owner() -> None:
    """No component may be defined by more than one library the wheel ships.

    This is the property the split exists to create. Two copies of a component mean
    two registries or two thread pools in one process, and a static initializer that
    registers into a table nothing else reads shows up as an operator missing at run
    time rather than as a link error.
    """
    shipped = {
        path.name for path in _shipped_runtime_libraries(_installed_package_dir())
    }
    for what, symbols, owner, required in _OWNED_COMPONENTS:
        present = any(name.startswith(owner) for name in shipped)
        assert present or not required, (
            f"the wheel ships no {owner}, which owns the {what}. Either packaging "
            "dropped it or the build did not produce it."
        )
        _assert_single_definer(symbols, what, owner if present else None)


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

add_library(custom_op_check SHARED custom_op.cpp)
# The legacy contract: a custom-op library links the shipped Python extension,
# which owns the operator registry it registers into.
target_link_libraries(custom_op_check PRIVATE _portable_lib)
# The runtime headers include c10 headers, which belong to torch rather than to
# this wheel, so an out-of-tree operator project supplies them the same way it
# supplies torch itself. The package config does not and should not ship them.
#
# The include directory is passed in rather than found with find_package(Torch),
# because that enables the CUDA language and fails on a machine with a CUDA
# toolkit it cannot probe, which has nothing to do with compiling an operator.
target_include_directories(custom_op_check PRIVATE ${TORCH_INCLUDE_DIR})
# Deliberately no target_compile_features here. These headers need C++20, and the
# package config is what has to say so. Setting it here would compile the check
# correctly while leaving a real consumer to fail.
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
    if _tool("ldd") is None:
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
            [_tool("ldd"), "-r", str(library)],
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
        # A non-zero exit with none of the expected text means ldd could not inspect
        # the file at all, which a text-only search reads as "nothing wrong". A file
        # under lib/ that is not a loadable object is a packaging defect, so treat it
        # as one rather than passing it.
        if (
            resolved.returncode != 0
            and "not found" not in combined
            and "undefined symbol" not in combined
        ):
            unresolved[str(library.relative_to(package_dir))] = [
                f"ldd could not inspect this file: {combined.strip()[:160]}"
            ]
            continue
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
    if _tool("ldd") is None or _tool("patchelf") is None:
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
                [_tool("patchelf"), "--print-rpath", str(target)],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            relative = [
                entry for entry in current.split(":") if entry.startswith("$ORIGIN")
            ]
            subprocess.run(
                [_tool("patchelf"), "--set-rpath", ":".join(relative), str(target)],
                # A failure here would leave the original absolute build paths in
                # place, and the check below would then pass by resolving through
                # them, which is exactly what this test exists to rule out.
                check=True,
            )
            resolved = subprocess.run(
                [_tool("ldd"), str(target)],
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
    # Skipped rather than asserted, the same as every other tool this suite needs.
    # A missing compiler says nothing about the wheel, and aborting here would take
    # the whole run down with it rather than reporting the one check it prevents.
    if _tool("cmake") is None:
        print("- cmake unavailable, skipping the custom op check")
        return

    package_dir = _installed_package_dir()
    if not list(package_dir.glob("extension/pybindings/_portable_lib*")):
        print("- the wheel ships no Python extension, skipping the custom op check")
        return

    source_dir = work_dir / "custom-op"
    build_dir = work_dir / "custom-op-build"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "custom_op.cpp").write_text(_CUSTOM_OP_SOURCE)
    (source_dir / "CMakeLists.txt").write_text(_CUSTOM_OP_CMAKE)

    # Torch's include directory is handed over directly, because the runtime headers
    # include c10 headers that belong to torch. A real out-of-tree project supplies
    # them the same way; the package config has no business shipping another
    # project's headers.
    if importlib.util.find_spec("torch") is None:
        print("- torch is not installed, skipping the custom op check")
        return
    import torch

    torch_include = Path(torch.__path__[0]) / "include"

    configure = subprocess.run(
        [
            _tool("cmake"),
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={package_dir / 'share' / 'cmake'}",
            f"-DTORCH_INCLUDE_DIR={torch_include}",
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
        [_tool("cmake"), "--build", str(build_dir)],
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


# The architectures a Linux wheel tag can name. Matched by suffix rather than parsed
# positionally, because the version part differs between spellings (linux_x86_64,
# manylinux_2_28_x86_64, manylinux2014_x86_64) enough that a positional pattern picks
# it up as part of the name.
#
# Linux only, which is what this check reads. `arm64` deliberately absent: it is the
# macOS spelling, Linux uses aarch64, and including it made a macosx_11_0_arm64 tag
# look like something this could compare. A tag naming none of these is reported as
# unreadable rather than as a mismatch.
_WHEEL_ARCHITECTURES = ("x86_64", "aarch64", "i686", "ppc64le", "s390x", "armv7l")


def _wheel_architecture(tag: str):
    """The architecture a platform tag names, or None if it names none of ours."""
    for architecture in _WHEEL_ARCHITECTURES:
        if tag.endswith("_" + architecture):
            return architecture
    return None


def _tag_architectures_match(claimed: str, supported: str):
    """Whether two platform tags name the same architecture.

    Returned rather than asserted so the same decision can be unit tested without a
    wheel. The previous arrangement duplicated the comparison in the test, which meant
    the test could pass while the shipped check was wrong, and that is exactly what
    happened: the original defect was in how the caller compared the two tags, and a
    test that re-implemented the comparison could not see it.

    None for either side means the tag names no architecture this project builds for,
    which is a different failure from a mismatch and is reported separately.
    """
    claimed_arch = _wheel_architecture(claimed)
    supported_arch = _wheel_architecture(supported)
    if claimed_arch is None or supported_arch is None:
        return None
    return claimed_arch == supported_arch


def test_wheel_platform_tag() -> None:
    """The wheel's declared platform tag must name the architecture it was built for.

    Only the architecture. auditwheel cannot certify a glibc baseline for this wheel:
    it depends on torch without vendoring torch's libraries, so the contents reference
    libtorch.so from outside any manylinux policy and auditwheel reports a plain
    linux_<arch>. That is the expected answer for a torch-dependent wheel rather than
    a defect, and the manylinux tag on the file comes from the build image. Asserting
    the baseline here failed every correct wheel.

    The architecture is still worth checking, because a wheel labelled with the wrong
    one installs on machines it cannot run on at all, and that is a mistake this can
    actually catch.
    """
    if importlib.util.find_spec("auditwheel") is None:
        # Installed here rather than skipped, because auditwheel is not in any CI
        # image and a skip is indistinguishable from a pass in the summary. This
        # check is the only thing that compares the wheel's declared tag against
        # what its libraries actually need, and this change adds five libraries
        # under that tag.
        print("- auditwheel not present, installing it so this check can run")
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "auditwheel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if installed.returncode != 0 or importlib.util.find_spec("auditwheel") is None:
            print(
                "- auditwheel could not be installed, skipping the platform tag "
                f"check: {installed.stderr.strip()[-200:]}"
            )
            return
        importlib.invalidate_caches()

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
    claimed = wheels[-1].name.split("-")[-1].removesuffix(".whl")
    supported = match.group(1)

    claimed_arch = _wheel_architecture(claimed)
    supported_arch = _wheel_architecture(supported)
    matches = _tag_architectures_match(claimed, supported)
    assert matches is not None, (
        f"could not read an architecture from the declared tag {claimed} or from what "
        f"auditwheel reported, {supported}"
    )
    assert matches, (
        f"the wheel claims architecture {claimed_arch} but its contents are built for "
        f"{supported_arch}, so it would install where it cannot run"
    )
    print(f"✓ the wheel is tagged for the architecture it contains ({claimed_arch})")


def test_no_absolute_runtime_paths() -> None:
    """No shipped library may search a directory a user does not have.

    Packaging copies libraries out of the build tree rather than installing them, so
    every directory the linker recorded ships as-is. Two kinds are rejected on every
    shipped library, not only the lib/ payload:

    - a directory inside the build, which names the machine that produced the wheel
    - an empty entry, which the loader reads as the process working directory

    Torch's own directory is accepted. The extensions link torch and resolve it
    through the directory the linker recorded, so that entry is load-bearing rather
    than leftover. Narrowing this check to lib/ once hid seven extensions carrying
    build-tree paths, so the exclusion is by what an entry POINTS AT, never by which
    file carries it.

    The check reads the shipped file directly, with nothing stripped, which is what
    a user actually receives.
    """
    # Fatal, not a skip. Packaging strips these paths best-effort, because it cannot
    # guarantee patchelf on PATH, so this is the only place the guarantee can be
    # enforced. If both went quiet on the same missing tool, a wheel carrying the
    # build machine's directories would ship looking correct.
    if _tool("patchelf") is None:
        print("- patchelf not present, installing it so this check can run")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "patchelf"],
            capture_output=True,
            text=True,
            check=False,
        )
    patchelf = _tool("patchelf")
    assert patchelf is not None, (
        "patchelf is required to check the shipped runtime paths and could not be "
        "installed. Packaging uses it to strip build-tree directories, and without it "
        "here neither side would notice that they were left in place."
    )

    package_dir = _installed_package_dir()

    # Directories that only exist inside a build of this project.
    build_markers = ("/pip-out/", "/cmake-out", "/build/lib.")

    offenders = {}
    checked = 0
    for library in sorted(package_dir.rglob("*.so*")):
        if not library.is_file() or library.is_symlink():
            continue
        result = subprocess.run(
            [patchelf, "--print-rpath", str(library)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        checked += 1
        # An absent RPATH and one containing a single empty entry both print as an
        # empty string, so treat empty output as "no runtime path" rather than as an
        # empty entry. A library with nothing to search is fine; the defect is
        # searching somewhere unusable.
        raw = result.stdout.strip()
        if not raw:
            continue
        bad = []
        for entry in raw.split(":"):
            if not entry:
                bad.append("<empty>")
            elif entry.startswith("/") and any(m in entry for m in build_markers):
                bad.append(entry)
        if bad:
            offenders[str(library.relative_to(package_dir))] = bad

    assert not offenders, (
        "shipped libraries search directories a user does not have, either inside "
        "the build tree that produced the wheel or, for an empty entry, the process "
        f"working directory: {offenders}"
    )
    print(
        f"✓ none of the {checked} shipped libraries searches a build-tree or empty "
        "runtime path"
    )


def test_extension_contains_no_component() -> None:
    """The Python extension must link the components, not contain them.

    This is the property the change exists to create, and no count of definers
    proves it: the monolithic layout has exactly one definer of every symbol too,
    inside the extension. The direct statement is that the extension defines none of
    what the shipped libraries own, and records a dependency on each instead.
    """
    assert _tool("nm") is not None, "nm is required to inspect the wheel"
    if _tool("readelf") is None:
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
    # Derived from the ownership table rather than restated. A hand-written copy drifts: this once
    # listed five symbol groups while the table covered eleven, so the components added later, including
    # both CUDA ones, were never checked here.
    #
    # The bundled third-party groups are left out on purpose. That code is also linked by torch, and the
    # extension links torch, so seeing those symbols there says nothing about this split.
    owned = tuple(
        symbol
        for _, symbols, _, _ in _OWNED_COMPONENTS
        if symbols not in (_BUNDLED_THREADPOOL_SYMBOLS, _BUNDLED_XNNPACK_SYMBOLS)
        for symbol in symbols
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
            [_tool("readelf"), "-d", str(extension)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if "NEEDED" in line
    }
    # Only the libraries whose code the extension used to contain. Those are the ones
    # this split moved out of it, so the extension must now resolve them from outside or
    # a retention option silently failed.
    #
    # Not every shipped library serves Python. The quantized kernels exist for a C++
    # application, since Python registers those operators through the torch-linked
    # ahead-of-time library at export time, and requiring a dependency would demand the
    # extension link code it has no use for.
    #
    # The CUDA delegate is NOT in that category. The build deliberately links it into the
    # extension with a retention option, so it does carry a dependency, and excluding it
    # switched off the one check that would notice if that retention stopped working. The
    # stream helper stays excluded because the extension reaches it only through the
    # delegate's public link, with no retention of its own to protect.
    expected = {
        name
        for name in (path.name for path in _shipped_runtime_libraries(package_dir))
        if not any(marker in name for marker in ("kernels_quantized", "extension_cuda"))
    }
    unused = sorted(expected - needed)
    assert not unused, (
        f"the wheel ships {unused} but {extension.name} does not depend on them, so "
        "either they are dead weight or a retention option did not hold"
    )

    # Positive proof that the extension resolves these from elsewhere, rather than
    # only the absence of a visible definition. A hidden or local copy would not
    # appear in the dynamic symbol table at all, so "defines nothing" on its own is
    # satisfiable by an extension that still carries its own private runtime. An
    # UNDEFINED reference cannot be faked that way: it says the definition is not
    # here and has to come from a dependency.
    undefined = subprocess.run(
        [_tool("nm"), "-DC", "--undefined-only", str(extension)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    imported = [
        symbol
        for symbol in (*_REGISTRY_SYMBOLS, *_THREADPOOL_SYMBOLS)
        if symbol in undefined
    ]
    assert len(imported) == len(_REGISTRY_SYMBOLS) + len(_THREADPOOL_SYMBOLS), (
        f"{extension.name} does not import every registry and thread pool symbol it "
        "uses, so it may carry a hidden copy that the visible symbol table does not "
        f"show. Imported: {imported}"
    )
    print(
        f"✓ {extension.name} ({extension.stat().st_size // 1024} KiB) contains no "
        f"component, imports the runtime symbols it uses, and depends on all "
        f"{len(expected)} shipped libraries it used to contain"
    )


def test_shipped_library_names_are_expected() -> None:
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

    # Regular files only. A symlink here would be a deliberate alias rather than the
    # stale-artifact case this check is about, and a leftover from an earlier build is
    # a real file, so it is still caught.
    shipped = sorted(
        p for p in lib_dir.glob("*.so*") if p.is_file() and not p.is_symlink()
    )
    assert shipped, f"the wheel ships a lib directory with no libraries: {lib_dir}"

    # The names packaging can put here. Listed rather than derived because setup.py
    # names each one literally, and a file with any other name did not come from
    # this build. Which of them are present depends on the build options, so
    # absence is fine and an unknown name is not.
    #
    # Matched in full rather than by taking the part before the first ".so", because
    # that prefix is satisfied by a name like libexecutorch.so.old.so.1, which is
    # exactly the shape a leftover file takes.
    known = (
        "libexecutorch",
        "libexecutorch_kernels_optimized",
        "libexecutorch_kernels_quantized",
        "libexecutorch_backend_cuda",
        "libexecutorch_extension_cuda",
        "libexecutorch_backend_xnnpack",
        "libexecutorch_threadpool",
        "libexecutorch_etdump",
    )
    # A plain .so, because the wheel build does not version these. A trailing
    # .so.<digits> would also be a name packaging did not produce here.
    permitted = re.compile(rf"(?:{'|'.join(known)})\.so")
    unknown = sorted(p.name for p in shipped if not permitted.fullmatch(p.name))
    assert not unknown, (
        f"the wheel ships {unknown} under lib/, which packaging does not produce. "
        "A file packaging did not put there came from a stale staging directory, "
        "and it ships while looking correct to every other check."
    )

    if _tool("readelf") is None:
        print(f"✓ {len(shipped)} shipped libraries have expected names")
        return

    # The recorded soname has to match the file name, or a consumer records a
    # dependency on a name the wheel does not contain.
    mismatched = {}
    for library in shipped:
        dynamic = subprocess.run(
            [_tool("readelf"), "-d", str(library)],
            capture_output=True,
            text=True,
            check=False,
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
    print(f"✓ {len(shipped)} shipped libraries have expected names and sonames")


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

    # The delegated half only applies to a wheel that ships the delegate. A wheel
    # built without XNNPACK is a supported configuration, and requiring delegated
    # execution there would reject a correct artifact.
    modes = ["plain"]
    if any(
        path.name.startswith("libexecutorch_backend_xnnpack.so")
        for path in _shipped_runtime_libraries(_installed_package_dir())
    ):
        modes.append("delegate")
    else:
        print("- this wheel ships no XNNPACK delegate, checking the plain model only")

    for mode in modes:
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


def test_declared_dependencies_match_the_wheel_tag() -> None:
    """A CPU wheel must not declare the CUDA runtime, and a CUDA wheel must declare it.

    The tag is what a user resolves against, so a mismatch is a promise the wheel cannot keep in
    either direction: a CPU wheel that pulls the CUDA packages costs a user hundreds of megabytes it
    never loads, and a CUDA wheel that declares nothing leaves the runtime unresolvable.

    This is metadata only, so no library check can see it. A CPU wheel that wrongly declared the CUDA
    runtime passed every other check in this file.
    """
    requirements = importlib.metadata.requires("executorch") or []
    cuda = sorted(r.split()[0] for r in requirements if r.lower().startswith("nvidia"))

    # The local version segment of the installed version states what the wheel was built for.
    version = importlib.metadata.version("executorch")
    local = version.partition("+")[2]
    is_cuda_wheel = local.startswith("cu")

    if is_cuda_wheel:
        assert cuda, (
            f"version {version} says this is a CUDA wheel, but it declares no CUDA runtime "
            "packages, so nothing resolves the runtime it links"
        )
        print(f"✓ this CUDA wheel declares the runtime ({len(cuda)} packages)")
    else:
        assert not cuda, (
            f"version {version} is not a CUDA wheel, yet it declares {cuda}. A user installing it "
            "would download the CUDA runtime this wheel never loads."
        )
        print("✓ this non-CUDA wheel declares no CUDA runtime")


def run_tests(work_dir: Path) -> None:
    # Ordered by what a failure tells you, because these run in sequence and the
    # first failure stops the rest. The checks that prove the split behaves
    # correctly come first; packaging metadata comes last.
    #
    # This is not hypothetical ordering advice. The platform tag check used to sit
    # in the middle, and when it failed it took the custom-operator, runtime-path
    # and numeric-parity checks with it in all eleven wheel jobs, so the weakest
    # check in the file hid the three strongest.
    test_each_component_has_one_owner()
    test_python_extensions_import()
    test_declared_dependencies_match_the_wheel_tag()
    test_extension_contains_no_component()
    test_shipped_library_names_are_expected()
    test_shipped_libraries_load()
    test_shipped_libraries_resolve_without_build_tree()
    test_custom_op_compiles(work_dir)
    test_no_absolute_runtime_paths()
    test_model_matches_eager_pytorch(work_dir)
    # Last: a wrong answer here says the wheel is labelled wrong, not that the
    # split is broken.
    test_wheel_platform_tag()
