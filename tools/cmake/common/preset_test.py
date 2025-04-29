# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from functools import cache
from typing import Any, Dict, List, Optional

_DEV_DO_NOT_DELETE_FOLDER = True

_TESTABLE_CMAKE_FILES = [
    "preset.cmake",
]


@cache
def _list_cmake_cache(cache_path: str) -> Dict[str, str]:
    result = {}
    with open(cache_path, "r") as cache_file:
        for line in cache_file:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                if ":" in key:
                    key, _ = key.split(":")
                result[key.strip()] = value.strip()
    return result


class TestPresetCmake(unittest.TestCase):
    def tearDown(self) -> None:
        if not _DEV_DO_NOT_DELETE_FOLDER:
            if self.workspace:
                shutil.rmtree(self.workspace)
            self.assertFalse(os.path.exists(self.workspace))

    def createWorkspace(self, tree: Dict[Any, Any], cwd: Optional[str] = None) -> None:
        if cwd is None:
            self.workspace = tempfile.mkdtemp()
            cwd = self.workspace
            if _DEV_DO_NOT_DELETE_FOLDER:
                print(
                    f"{self._testMethodName} workspace:",
                    self.workspace,
                    file=sys.stderr,
                )

            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            for testable_cmake_file in _TESTABLE_CMAKE_FILES:
                source_path = os.path.join(current_file_dir, testable_cmake_file)
                assert os.path.exists(
                    source_path
                ), f"{testable_cmake_file} does not exist in {source_path}"
                destination_path = os.path.join(self.workspace, testable_cmake_file)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy(source_path, destination_path)

        for name, value in tree.items():
            if isinstance(value, str):
                file_path = os.path.join(cwd, name)
                assert not os.path.exists(
                    file_path
                ), f"file already exists: {file_path}"
                os.makedirs(cwd, exist_ok=True)
                with open(file_path, "w") as new_file:
                    new_file.write(value)
            elif isinstance(value, dict):
                new_cwd = os.path.join(cwd, name)
                os.makedirs(new_cwd, exist_ok=True)
                self.createWorkspace(value, cwd=new_cwd)
            else:
                raise AssertionError("invalid tree value", value)

    def assertFileContent(self, relativePath: str, expectedContent: str) -> None:
        path = os.path.join(self.workspace, relativePath)
        self.assertTrue(os.path.exists(path), f"expected path does not exist: {path}")

        with open(path, "r") as path_file:
            self.assertEqual(path_file.read(), expectedContent)

    def test_create_workspace(self):
        self.createWorkspace(
            {
                "CMakeLists.txt": "move fast",
                ".gitignore": ".DS_Store",
                "build": {
                    "CMakeLists.txt": "move faster",
                    "github": {
                        "README.md": "yeet",
                    },
                },
                "README.md": "Meta Platforms",
            }
        )

        self.assertIsNotNone(self.workspace)
        self.assertFileContent("CMakeLists.txt", "move fast")
        self.assertFileContent("README.md", "Meta Platforms")
        self.assertFileContent(".gitignore", ".DS_Store")
        self.assertFileContent("build/CMakeLists.txt", "move faster")
        self.assertFileContent("build/github/README.md", "yeet")

        # Test implicitly copied cmake files
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.assertTrue(len(_TESTABLE_CMAKE_FILES) > 0)
        for testable_cmake_file in _TESTABLE_CMAKE_FILES:
            with open(
                os.path.join(current_file_dir, testable_cmake_file), "r"
            ) as source_file:
                self.assertFileContent(testable_cmake_file, source_file.read())

    def runCmake(
        self, cmake_args: List[str] = [], error_contains: Optional[str] = None
    ):
        cmake_args += [
            "--no-warn-unused-cli"
        ]

        result = subprocess.run(
            ["cmake", *cmake_args, "-S", ".", "-B", "cmake-out"],
            cwd=self.workspace,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if error_contains else None,
            check=False,
        )

        if error_contains is not None:
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(error_contains in result.stderr.decode("utf-8"))
        else:
            self.assertEqual(result.returncode, 0)
            self.assertTrue(os.path.exists(os.path.join(self.workspace, "cmake-out")))

    def assertCmakeCacheValue(self, key: str, expected: str):
        cache = _list_cmake_cache(
            os.path.join(self.workspace, "cmake-out", "CMakeCache.txt")
        )
        self.assertEqual(cache[key], expected, f"invalid value for {key}")

    def test_set_option(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            add_subdirectory(numbers)

            set(SECRET_MESSAGE "move fast" CACHE STRING "")
        """
        _numbers_cmake_lists_txt = """
            set(PI 3.14 CACHE STRING "")
        """

        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "numbers": {
                    "CMakeLists.txt": _numbers_cmake_lists_txt,
                },
            }
        )
        self.runCmake()
        self.assertCmakeCacheValue("SECRET_MESSAGE", "move fast")
        self.assertCmakeCacheValue("PI", "3.14")

    def testdefine_overridable_config_invalid_name(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)

            define_overridable_config(IAM_AN_INVALID_NAME "test example" "move fast")
        """
        self.createWorkspace({"CMakeLists.txt": _cmake_lists_txt})
        self.runCmake(
            error_contains="Config name 'IAM_AN_INVALID_NAME' must start with EXECUTORCH_"
        )

    def testdefine_overridable_config_default(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            add_subdirectory(build)
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake()
        self.assertCmakeCacheValue("EXECUTORCH_TEST_MESSAGE", "move fast")

    def testdefine_overridable_config_cli_override(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            add_subdirectory(build)
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake(cmake_args=["-DEXECUTORCH_TEST_MESSAGE='from the cli'"])
        self.assertCmakeCacheValue("EXECUTORCH_TEST_MESSAGE", "from the cli")

    def testdefine_overridable_config_set_override_before(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)

            set(EXECUTORCH_TEST_MESSAGE "from set")
            add_subdirectory(build)
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake()
        self.assertCmakeCacheValue("EXECUTORCH_TEST_MESSAGE", "from set")

    def testdefine_overridable_config_set_override_after(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)

            add_subdirectory(build)
            set(EXECUTORCH_TEST_MESSAGE "from set")
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake()
        # Setting the value after should not affect the cache.
        self.assertCmakeCacheValue("EXECUTORCH_TEST_MESSAGE", "move fast")

    def testdefine_overridable_config_set_override_after_with_cache(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)

            add_subdirectory(build)
            set(EXECUTORCH_TEST_MESSAGE "from set with cache override" CACHE STRING "")
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake()
        # Setting the value after should not affect the cache.
        self.assertCmakeCacheValue("EXECUTORCH_TEST_MESSAGE", "move fast")

    def testdefine_overridable_config_cli_override_with_set_override(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)

            set(EXECUTORCH_TEST_MESSAGE "from set")
            add_subdirectory(build)
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake(cmake_args=["-DEXECUTORCH_TEST_MESSAGE='from the cli'"])
        # If an option is set through cmake, it should be be overridable from the CLI.
        self.assertCmakeCacheValue("EXECUTORCH_TEST_MESSAGE", "from set")

    def test_set_overridable_config_before(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)

            set_overridable_config(EXECUTORCH_TEST_MESSAGE "from set_overridable_config")
            add_subdirectory(build)
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake()
        self.assertCmakeCacheValue(
            "EXECUTORCH_TEST_MESSAGE", "from set_overridable_config"
        )

    def test_set_overridable_config_after(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)

            add_subdirectory(build)
            set_overridable_config(EXECUTORCH_TEST_MESSAGE "from set_overridable_config")
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake()
        self.assertCmakeCacheValue("EXECUTORCH_TEST_MESSAGE", "move fast")

    def test_set_overridable_config_with_cli_override(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)

            include(${PROJECT_SOURCE_DIR}/preset.cmake)

            set_overridable_config(EXECUTORCH_TEST_MESSAGE "from set_overridable_config")
            add_subdirectory(build)
        """
        _build_cmake_lists_txt = """
            define_overridable_config(EXECUTORCH_TEST_MESSAGE "test message" "move fast")
        """
        self.createWorkspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.runCmake(cmake_args=["-DEXECUTORCH_TEST_MESSAGE='from the cli'"])
        self.assertCmakeCacheValue("EXECUTORCH_TEST_MESSAGE", "from the cli")


if __name__ == "__main__":
    unittest.main()
