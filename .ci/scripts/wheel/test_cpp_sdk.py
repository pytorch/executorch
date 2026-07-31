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


def _assert_single_definer(symbols, what: str) -> None:
    """Exactly one shipped library may define each of `symbols`."""
    assert shutil.which("nm") is not None, "nm is required to inspect the wheel"

    package_dir = _installed_package_dir()
    libraries = _shipped_shared_objects(package_dir)
    assert libraries, f"no shared libraries found under {package_dir}"

    for symbol in symbols:
        definers = [lib for lib in libraries if _defines_symbol(lib, symbol)]
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


def run_tests(work_dir: Path) -> None:
    test_single_backend_registry()
    test_single_threadpool()
    test_cpp_consumer(work_dir)
