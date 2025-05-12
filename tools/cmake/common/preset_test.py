# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from tools.cmake.common import CMakeTestCase, TESTABLE_CMAKE_FILES


class TestPreset(CMakeTestCase):

    def test_create_workspace(self):
        self.create_workspace(
            {
                ".gitignore": ".DS_Store",
                "CMakeLists.txt": "move fast",
                "README.md": "Meta Platforms",
                "example": {
                    "CMakeLists.txt": "move faster",
                    "cmake": {
                        "README.md": "godspeed you!",
                    },
                },
            }
        )

        self.assertIsNotNone(self.workspace)
        self.assert_file_content("CMakeLists.txt", "move fast")
        self.assert_file_content("README.md", "Meta Platforms")
        self.assert_file_content(".gitignore", ".DS_Store")
        self.assert_file_content("example/CMakeLists.txt", "move faster")
        self.assert_file_content("example/cmake/README.md", "godspeed you!")

        # Test implicitly copied cmake files
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.assertTrue(len(TESTABLE_CMAKE_FILES) > 0)
        for testable_cmake_file in TESTABLE_CMAKE_FILES:
            with open(
                os.path.join(this_file_dir, testable_cmake_file), "r"
            ) as source_file:
                self.assert_file_content(testable_cmake_file, source_file.read())

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

        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "numbers": {
                    "CMakeLists.txt": _numbers_cmake_lists_txt,
                },
            }
        )
        self.run_cmake()
        self.assert_cmake_cache("SECRET_MESSAGE", "move fast", "STRING")
        self.assert_cmake_cache("PI", "3.14", "STRING")

    def test_define_overridable_option_invalid_name(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            define_overridable_option(IAM_AN_INVALID_NAME "test example" STRING "default value")
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake(
            error_contains="Option name 'IAM_AN_INVALID_NAME' must start with EXECUTORCH_"
        )

    def test_define_overridable_option_default(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            add_subdirectory(example)
        """
        _example_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "default value")
            define_overridable_option(EXECUTORCH_TEST_OPTION "test option" BOOL ON)
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "example": {
                    "CMakeLists.txt": _example_cmake_lists_txt,
                },
            }
        )
        self.run_cmake()
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "default value", "STRING")
        self.assert_cmake_cache("EXECUTORCH_TEST_OPTION", "ON", "BOOL")

    def test_define_overridable_option_invalid_type(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test example" NUMBER "default value")
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake(
            error_contains="Invalid option (EXECUTORCH_TEST_MESSAGE) value type 'NUMBER'"
        )

    def test_define_overridable_option_cli_override(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            add_subdirectory(example)
        """
        _example_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "default value")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "example": {
                    "CMakeLists.txt": _example_cmake_lists_txt,
                },
            }
        )
        self.run_cmake(cmake_args=["-DEXECUTORCH_TEST_MESSAGE='cli value'"])
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "cli value", "STRING")

    def test_define_overridable_option_set_override_before(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            set(EXECUTORCH_TEST_MESSAGE "set value")
            add_subdirectory(example)
        """
        _example_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "default value")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "example": {
                    "CMakeLists.txt": _example_cmake_lists_txt,
                },
            }
        )
        self.run_cmake()
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "set value", "STRING")

    def test_define_overridable_option_set_override_after(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            add_subdirectory(example)
            set(EXECUTORCH_TEST_MESSAGE "set value")
        """
        _example_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "default value")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "example": {
                    "CMakeLists.txt": _example_cmake_lists_txt,
                },
            }
        )
        self.run_cmake()
        # Setting the value after should not affect the cache.
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "default value", "STRING")

    def test_define_overridable_option_set_override_after_with_cache(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            add_subdirectory(example)
            set(EXECUTORCH_TEST_MESSAGE "set value" CACHE STRING "")
        """
        _example_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "default value")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "example": {
                    "CMakeLists.txt": _example_cmake_lists_txt,
                },
            }
        )
        self.run_cmake()
        # Setting the value after should not affect the cache.
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "default value", "STRING")

    def test_define_overridable_option_cli_override_with_set_override(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            set(EXECUTORCH_TEST_MESSAGE "set value")
            add_subdirectory(example)
        """
        _example_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "default value")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "example": {
                    "CMakeLists.txt": _example_cmake_lists_txt,
                },
            }
        )
        self.run_cmake(cmake_args=["-DEXECUTORCH_TEST_MESSAGE='cli value'"])
        # If an option is set through cmake, it should NOT be overridable from the CLI.
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "set value", "STRING")

    def test_set_overridable_option_before(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            set_overridable_option(EXECUTORCH_TEST_MESSAGE "from set_overridable_option")
            add_subdirectory(build)
        """
        _build_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "move fast")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.run_cmake()
        self.assert_cmake_cache(
            "EXECUTORCH_TEST_MESSAGE", "from set_overridable_option", "STRING"
        )

    def test_set_overridable_option_after(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            add_subdirectory(build)
            set_overridable_option(EXECUTORCH_TEST_MESSAGE "from set_overridable_option")
        """
        _build_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "move fast")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.run_cmake()
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "move fast", "STRING")

    def test_set_overridable_option_with_cli_override(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            add_subdirectory(build)
        """
        _build_cmake_lists_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "move fast")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "CMakeLists.txt": _build_cmake_lists_txt,
                },
            }
        )
        self.run_cmake(cmake_args=["-DEXECUTORCH_TEST_MESSAGE='from the cli'"])
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "from the cli", "STRING")
