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

    def test_define_overridable_option_override_existing_cache_with_cli(self):
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
        self.run_cmake()
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "default value", "STRING")

        self.run_cmake(cmake_args=["-DEXECUTORCH_TEST_MESSAGE='cli value'"])
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "cli value", "STRING")

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

    def test_set_overridable_option_loaded_from_file(self):
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            include(${PROJECT_SOURCE_DIR}/build/my_preset.cmake)
            include(${PROJECT_SOURCE_DIR}/build/default.cmake)
        """
        _my_preset_txt = """
            set_overridable_option(EXECUTORCH_FOO "hello world")
        """
        _default_preset_txt = """
            define_overridable_option(EXECUTORCH_TEST_MESSAGE "test message" STRING "move fast")
            define_overridable_option(EXECUTORCH_FOO "another test message" STRING "break things")
        """
        self.create_workspace(
            {
                "CMakeLists.txt": _cmake_lists_txt,
                "build": {
                    "my_preset.cmake": _my_preset_txt,
                    "default.cmake": _default_preset_txt,
                },
            }
        )
        self.run_cmake(cmake_args=["-DEXECUTORCH_TEST_MESSAGE='from the cli'"])
        self.assert_cmake_cache("EXECUTORCH_TEST_MESSAGE", "from the cli", "STRING")
        self.assert_cmake_cache("EXECUTORCH_FOO", "hello world", "STRING")

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

    def test_check_required_options_on_if_on_off(self):
        """Test that when IF_ON is OFF, no checks are performed."""

        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            
            set(FEATURE_FLAG OFF)
            set(REQUIRED_OPTION1 OFF)
            set(REQUIRED_OPTION2 OFF)
            
            check_required_options_on(
                IF_ON 
                    FEATURE_FLAG
                REQUIRES 
                    REQUIRED_OPTION1 
                    REQUIRED_OPTION2
            )
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake()  # Should succeed

    def test_check_required_options_on_all_required_on(self):
        """Test that when IF_ON is ON and all required options are ON, no error occurs."""

        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            
            set(FEATURE_FLAG ON)
            set(REQUIRED_OPTION1 ON)
            set(REQUIRED_OPTION2 ON)
            
            check_required_options_on(
                IF_ON 
                    FEATURE_FLAG
                REQUIRES 
                    REQUIRED_OPTION1 
                    REQUIRED_OPTION2
            )
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake()

    def test_check_required_options_on_one_required_off(self):
        """Test that when IF_ON is ON but one required option is OFF, a fatal error occurs."""

        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            
            set(FEATURE_FLAG ON)
            set(REQUIRED_OPTION1 ON)
            set(REQUIRED_OPTION2 OFF)
            
            check_required_options_on(
                IF_ON 
                    FEATURE_FLAG
                REQUIRES 
                    REQUIRED_OPTION1 
                    REQUIRED_OPTION2
            )
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake(
            error_contains="Use of 'FEATURE_FLAG' requires 'REQUIRED_OPTION2'"
        )

    def test_check_required_options_on_multiple_required_off(self):
        """Test that when IF_ON is ON but multiple required options are OFF, a fatal error occurs for the first one."""

        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            
            set(FEATURE_FLAG ON)
            set(REQUIRED_OPTION1 OFF)
            set(REQUIRED_OPTION2 OFF)
            
            # This should cause a fatal error
            check_required_options_on(
                IF_ON 
                    FEATURE_FLAG
                REQUIRES 
                    REQUIRED_OPTION1 
                    REQUIRED_OPTION2
            )
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake(
            error_contains="Use of 'FEATURE_FLAG' requires 'REQUIRED_OPTION1'"
        )

    def test_check_conflicting_options_on_if_on_off(self):
        """Test that when IF_ON is OFF, no conflict checks are performed."""

        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            set(FEATURE_FLAG OFF)
            set(CONFLICTING_OPTION1 ON)
            set(CONFLICTING_OPTION2 ON)
            check_conflicting_options_on(
                IF_ON
                    FEATURE_FLAG
                CONFLICTS_WITH
                    CONFLICTING_OPTION1
                    CONFLICTING_OPTION2
            )
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake()

    def test_check_conflicting_options_on_no_conflicts(self):
        """Test that when IF_ON is ON but no conflicting options are ON, no error occurs."""
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            set(FEATURE_FLAG ON)
            set(CONFLICTING_OPTION1 OFF)
            set(CONFLICTING_OPTION2 OFF)
            check_conflicting_options_on(
                IF_ON
                    FEATURE_FLAG
                CONFLICTS_WITH
                    CONFLICTING_OPTION1
                    CONFLICTING_OPTION2
            )
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake()

    def test_check_conflicting_options_on_one_conflict(self):
        """Test that when IF_ON is ON and one conflicting option is also ON, a fatal error occurs."""
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            set(FEATURE_FLAG ON)
            set(CONFLICTING_OPTION1 ON)
            set(CONFLICTING_OPTION2 OFF)
            check_conflicting_options_on(
                IF_ON
                    FEATURE_FLAG
                CONFLICTS_WITH
                    CONFLICTING_OPTION1
                    CONFLICTING_OPTION2
            )
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake(
            error_contains="Both 'FEATURE_FLAG' and 'CONFLICTING_OPTION1' can't be ON"
        )

    def test_check_conflicting_options_on_multiple_conflicts(self):
        """Test that when IF_ON is ON and multiple conflicting options are ON, a fatal error occurs for the first conflict."""
        _cmake_lists_txt = """
            cmake_minimum_required(VERSION 3.24)
            project(test_preset)
            include(${PROJECT_SOURCE_DIR}/preset.cmake)
            set(FEATURE_FLAG ON)
            set(CONFLICTING_OPTION1 ON)
            set(CONFLICTING_OPTION2 ON)
            check_conflicting_options_on(
                IF_ON
                    FEATURE_FLAG
                CONFLICTS_WITH
                    CONFLICTING_OPTION1
                    CONFLICTING_OPTION2
            )
        """
        self.create_workspace({"CMakeLists.txt": _cmake_lists_txt})
        self.run_cmake(
            error_contains="Both 'FEATURE_FLAG' and 'CONFLICTING_OPTION1' can't be ON"
        )
