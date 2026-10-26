# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _compile_flags(entry: dict) -> set:
    """The flags of one compile command, without the compiler, source and outputs."""
    flags = set()
    arguments = iter(shlex.split(entry["command"])[1:])
    for argument in arguments:
        if argument in ("-o", "-c", "-MT", "-MF"):
            next(arguments)
        elif argument == "-isystem":
            flags.add(f"-isystem {next(arguments)}")
        elif argument != "-MD":
            flags.add(argument)
    return flags


def _copied_from_torch(flag: str) -> bool:
    """Whether a flag comes from what find_package_torch copies from torch.

    A CUDA build of torch also gives torch the CUDA toolkit include dir and torch_cuda's
    definitions, which torch_cpu_library correctly does not carry.
    """
    return (
        flag == "-DET_TORCH_CXX_FLAGS_PROBE"
        or flag.startswith("-std=")
        or (flag.startswith("-isystem ") and flag.endswith("torch/csrc/api/include"))
    )


@unittest.skipIf(sys.platform == "win32", "Visual Studio writes no compile commands")
class TestTorchCpuLibraryCMake(unittest.TestCase):
    def test_torch_cpu_library_gets_torch_settings(self):
        cmake = shutil.which("cmake")
        if cmake is None or importlib.util.find_spec("torch") is None:
            self.skipTest("CMake and torch are required")

        found_first = (
            "add_torch_to_cmake_prefix_path()\nfind_package(Torch CONFIG REQUIRED)"
        )
        for find_torch in ("", found_first):
            with self.subTest(found_first=bool(find_torch)):
                self._check(cmake, find_torch)

    def _check(self, cmake: str, find_torch: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            build = source / "build"
            (source / "probe.cpp").write_text("int probe() { return 0; }\n")
            (source / "CMakeLists.txt").write_text(
                f"""cmake_minimum_required(VERSION 3.24)
project(TorchCpuLibraryProbe LANGUAGES CXX)
include("${{EXECUTORCH_ROOT}}/tools/cmake/Utils.cmake")
{find_torch}
find_package_torch()
add_library(links_torch OBJECT probe.cpp)
target_link_libraries(links_torch PRIVATE torch)
add_library(links_torch_cpu_library OBJECT probe.cpp)
target_link_libraries(links_torch_cpu_library PRIVATE torch_cpu_library)
add_library(links_it_publicly STATIC probe.cpp)
target_link_libraries(links_it_publicly PUBLIC torch_cpu_library)
add_library(consumer OBJECT probe.cpp)
target_link_libraries(consumer PRIVATE links_it_publicly)
"""
            )
            result = subprocess.run(
                [
                    cmake,
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    f"-DEXECUTORCH_ROOT={ROOT}",
                    f"-DPYTHON_EXECUTABLE={sys.executable}",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                    # Recent torch sets no TORCH_CXX_FLAGS, so set one to check it travels too.
                    "-DTORCH_CXX_FLAGS=-DET_TORCH_CXX_FLAGS_PROBE",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            flags = {}
            for entry in json.loads((build / "compile_commands.json").read_text()):
                target = Path(entry["output"]).parent.name.removesuffix(".dir")
                flags[target] = _compile_flags(entry)
            expected = {
                flag for flag in flags["links_torch"] if _copied_from_torch(flag)
            }
            self.assertIn("-DET_TORCH_CXX_FLAGS_PROBE", expected)
            self.assertTrue(any(flag.startswith("-isystem ") for flag in expected))
            for target in ("links_torch_cpu_library", "links_it_publicly", "consumer"):
                self.assertEqual(expected - flags[target], set(), target)

    def test_find_package_torch_without_torch_cpu_library(self):
        """A parent project or a static libtorch can define torch alone."""
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("CMake is required")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.24)
project(TorchWithoutTorchCpuLibrary LANGUAGES NONE)
include("${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake")
add_library(torch INTERFACE)
find_package_torch()
"""
            )
            result = subprocess.run(
                [
                    cmake,
                    "-S",
                    str(source),
                    "-B",
                    str(source / "build"),
                    f"-DEXECUTORCH_ROOT={ROOT}",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
