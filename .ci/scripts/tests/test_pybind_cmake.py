# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipIf(sys.platform == "win32", "Unix extensions resolve Python at load time")
class TestPybindCMake(unittest.TestCase):
    def test_extension_without_embedding_target_remains_linkable(self):
        cmake = shutil.which("cmake")
        if cmake is None or not (ROOT / "third-party/pybind11/CMakeLists.txt").exists():
            self.skipTest("CMake and the pybind11 submodule are required")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            build = source / "build"
            (source / "CMakeLists.txt").write_text(
                """cmake_minimum_required(VERSION 3.26)
project(PybindRegression LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
add_subdirectory("${EXECUTORCH_ROOT}/third-party/pybind11" pybind11)
include("${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake")
if(TARGET Python::Python)
  message(FATAL_ERROR "This test requires no Python embedding target")
endif()
executorch_add_pybind_module(et_pybind_probe probe.cpp)
get_target_property(extension_type et_pybind_probe TYPE)
if(NOT extension_type STREQUAL "SHARED_LIBRARY")
  message(FATAL_ERROR "The extension must remain linkable as a shared library")
endif()
add_library(consumer SHARED consumer.cpp)
target_link_libraries(consumer PRIVATE et_pybind_probe)
if(APPLE)
  target_link_options(consumer PRIVATE "LINKER:-undefined,dynamic_lookup")
endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/extension-path.txt"
     CONTENT "$<TARGET_FILE:et_pybind_probe>")
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-path.txt"
     CONTENT "$<TARGET_FILE:consumer>")
"""
            )
            (source / "probe.cpp").write_text(
                """#include <pybind11/pybind11.h>
extern "C" PYBIND11_EXPORT int et_probe_add(int a, int b) { return a + b; }
PYBIND11_MODULE(et_pybind_probe, module) { module.def("add", &et_probe_add); }
"""
            )
            (source / "consumer.cpp").write_text(
                """extern "C" int et_probe_add(int, int);
extern "C" int consume_extension() { return et_probe_add(34, 42); }
"""
            )
            for command in (
                [
                    cmake,
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    f"-DEXECUTORCH_ROOT={ROOT}",
                    f"-DPython_EXECUTABLE={sys.executable}",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                [cmake, "--build", str(build), "--parallel", "2"],
            ):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            extension_path = (build / "extension-path.txt").read_text()
            spec = importlib.util.spec_from_file_location(
                "et_pybind_probe", extension_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.add(17, 25), 42)

            consumer = ctypes.CDLL((build / "consumer-path.txt").read_text())
            self.assertEqual(consumer.consume_extension(), 76)


if __name__ == "__main__":
    unittest.main()
