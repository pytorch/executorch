# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess
import tempfile
import unittest
from functools import cache
from typing import Any, Dict, List, Optional

from tools.cmake.cmake_cache import CMakeCache

# Files to copy from this directory into the temporary workspaces.
TESTABLE_CMAKE_FILES = [
    "preset.cmake",
]


# If KEEP_WORKSPACE is set, then keep the workspace instead of deleting it. Useful
# when debugging tests.
@cache
def _keep_workspace() -> bool:
    keep_workspace_env = os.environ.get("KEEP_WORKSPACE")
    if keep_workspace_env is None:
        return False
    return keep_workspace_env.lower() not in ("false", "0", "no", "n")


# Create a file tree in the current working directory (cwd). The structure of the
# tree maps to the structure of the file tree. The key of the tree is the name
# of the folder or file. If the value is dict, it creates a folder. If the value
# is a string, it creates a file.
#
# Example:
#
#     {
#       "README.md": "this is a read me file",
#       "build": {
#         "cmake": {
#           "utils.cmake": "this is a cmake file",
#         }
#       }
#     }
# Results in:
#
#     ├── README.md
#     └── build
#         └── cmake
#             └── utils.cmake
#
def _create_file_tree(tree: Dict[Any, Any], cwd: str) -> None:
    for name, value in tree.items():
        if isinstance(value, str):
            file_path = os.path.join(cwd, name)
            assert not os.path.exists(file_path), f"file already exists: {file_path}"
            os.makedirs(cwd, exist_ok=True)
            with open(file_path, "w") as new_file:
                new_file.write(value)
        elif isinstance(value, dict):
            new_cwd = os.path.join(cwd, name)
            os.makedirs(new_cwd, exist_ok=True)
            _create_file_tree(tree=value, cwd=new_cwd)
        else:
            raise AssertionError("invalid tree value", value)


class CMakeTestCase(unittest.TestCase):

    def tearDown(self) -> None:
        super().tearDown()

        if self.workspace and not _keep_workspace():
            shutil.rmtree(self.workspace)
            self.assertFalse(os.path.exists(self.workspace))

    def create_workspace(self, tree: Dict[Any, Any]) -> None:
        self.workspace = tempfile.mkdtemp()
        if _keep_workspace():
            print("created workspace", self.workspace)

        # Copy testable tree
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        for testable_cmake_file in TESTABLE_CMAKE_FILES:
            source_path = os.path.join(this_file_dir, testable_cmake_file)
            assert os.path.exists(
                source_path
            ), f"{testable_cmake_file} does not exist in {source_path}"
            destination_path = os.path.join(self.workspace, testable_cmake_file)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(source_path, destination_path)

        _create_file_tree(tree=tree, cwd=self.workspace)

    def assert_file_content(self, relativePath: str, expectedContent: str) -> None:
        path = os.path.join(self.workspace, relativePath)
        self.assertTrue(os.path.exists(path), f"expected path does not exist: {path}")

        with open(path, "r") as path_file:
            self.assertEqual(path_file.read(), expectedContent)

    def run_cmake(
        self,
        cmake_args: Optional[List[str]] = None,
        error_contains: Optional[str] = None,
    ):
        cmake_args = (cmake_args or []) + ["--no-warn-unused-cli"]

        result = subprocess.run(
            ["cmake", *cmake_args, "-S", ".", "-B", "cmake-out"],
            cwd=self.workspace,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if error_contains else None,
            check=False,
        )

        if error_contains is not None:
            self.assertNotEqual(result.returncode, 0)
            actual_error = result.stderr.decode("utf-8")
            self.assertTrue(
                error_contains in actual_error,
                f"\n\nWanted: {error_contains}\n\nActual: {actual_error}",
            )
        else:
            self.assertEqual(result.returncode, 0)
            self.assertTrue(os.path.exists(os.path.join(self.workspace, "cmake-out")))

    def assert_cmake_cache(
        self,
        key: str,
        expected_value: str,
        expected_type: str,
    ):
        cache = CMakeCache(os.path.join(self.workspace, "cmake-out", "CMakeCache.txt"))
        self.assertEqual(
            cache.get(key).value, expected_value, f"unexpected value for {key}"
        )
        self.assertEqual(
            cache.get(key).value_type, expected_type, f"unexpected value type for {key}"
        )
