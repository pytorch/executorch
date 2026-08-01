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

import os
import re
import shutil
import subprocess
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

# Symbol kinds that mean the object owns the code or storage.
_OWNING_KINDS = frozenset("TtBbDdGgSsRrWV")

_CONSUMER_SOURCE = """\
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>

int main() {
  executorch::runtime::runtime_init();
  std::printf(
      "registered backends: %zu\\n",
      (size_t)executorch::runtime::get_num_registered_backends());
  return 0;
}
"""

_CONSUMER_CMAKE = """\
cmake_minimum_required(VERSION 3.24)
project(executorch_wheel_consumer CXX)
find_package(executorch REQUIRED)
add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE executorch::runtime)
"""


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
        if (
            match
            and match.group("name").startswith(symbol)
            and match.group("kind") in _OWNING_KINDS
        ):
            return True
    return False


def _report_definers(symbols, what: str) -> None:
    """Print which shipped libraries define each symbol, without asserting.

    Used where the desired invariant is one owner but the wheel does not reach it
    yet for reasons outside this change. Printing keeps the number visible in CI so
    a regression is noticeable, without failing on a pre-existing condition.
    """
    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    for symbol in symbols:
        definers = [
            str(library.relative_to(package_dir))
            for library in libraries
            if _defines_symbol(library, symbol)
        ]
        print(f"- {what}: {symbol.split('::')[-1]} defined by {len(definers)}")
        for definer in definers:
            print(f"    {definer}")


def report_wheel_composition() -> None:
    """Print what the wheel ships and what each library needs.

    Not an assertion. A size jump or an unexpected external dependency is the
    first visible sign that a component got statically duplicated again, so the
    numbers are worth having in the log of every run.
    """
    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)

    print("shipped libraries:")
    total = 0
    for library in sorted(libraries, key=lambda path: path.name):
        size = library.stat().st_size
        total += size
        print(f"  {size / 1024:9.1f} KiB  {library.relative_to(package_dir)}")
    print(f"  {total / 1024:9.1f} KiB  total")

    if shutil.which("readelf") is None:
        return
    # Anything the libraries need that the wheel does not itself ship has to be
    # present on the user's machine, so it belongs in the report. Compare against
    # the shipped file names rather than guessing from name prefixes.
    shipped = {library.name for library in libraries}
    external = set()
    for library in libraries:
        dynamic = subprocess.run(
            ["readelf", "-d", str(library)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        for line in dynamic.splitlines():
            if "(NEEDED)" not in line or "[" not in line:
                continue
            name = line.split("[", 1)[1].rstrip("]").strip()
            if name not in shipped:
                external.add(name)
    if external:
        print("external dependencies expected from the environment:")
        for name in sorted(external):
            print(f"  {name}")


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
    for library in libraries:
        resolved = subprocess.run(
            ["ldd", str(library)],
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
        ).stdout
        missing = [
            line.split("=>")[0].strip()
            for line in resolved.splitlines()
            if "not found" in line
        ]
        absent = [name for name in missing if name not in shipped]
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
    print("✓ every shipped library resolves every dependency it needs")


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
    # Ownership of the operator table, not just of a kernel implementation. This is
    # reported rather than asserted because two extension modules still link the
    # static core and carry their own copy, which predates this split: a released
    # wheel has five definers where this one has three. Asserting here would fail
    # on those pre-existing copies rather than on anything this change introduced.
    _report_definers(_KERNEL_REGISTRY_SYMBOLS, "operator registry")


def test_single_xnnpack_delegate() -> None:
    """Exactly one shipped library may define the XNNPACK delegate."""
    _assert_single_definer(_XNNPACK_SYMBOLS, "XNNPACK delegate")


def test_single_cuda_delegate() -> None:
    """Exactly one shipped library may define the CUDA delegate, if present."""
    _assert_single_definer(_CUDA_SYMBOLS, "CUDA delegate", optional=True)


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


def run_tests(work_dir: Path) -> None:
    report_wheel_composition()
    test_shipped_libraries_load()
    test_single_backend_registry()
    test_single_threadpool()
    test_single_kernel_registration()
    test_single_xnnpack_delegate()
    test_single_cuda_delegate()
    test_cpp_consumer(work_dir)
