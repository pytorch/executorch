#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from unittest.mock import NonCallableMock, patch

import executorch.codegen.tools.gen_all_oplist as gen_all_oplist


class TestGenAllOplist(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = tempfile.NamedTemporaryFile(dir=self.temp_dir.name)

    @patch("tools_copy.code_analyzer.gen_oplist_copy_from_core.main")
    def test_model_file_list_path_is_file_pass(
        self, mock_main: NonCallableMock
    ) -> None:
        args = [
            f"--output_dir={self.temp_dir.name}",
            f"--model_file_list_path={self.temp_file.name}",
        ]
        gen_all_oplist.main(args)
        mock_main.assert_called_once_with(args)

    @patch("tools_copy.code_analyzer.gen_oplist_copy_from_core.main")
    def test_model_file_list_path_is_directory_with_file_pass(
        self, mock_main: NonCallableMock
    ) -> None:
        file_ = tempfile.NamedTemporaryFile()
        with open(file_.name, "w") as f:
            f.write(self.temp_file.name)
        args = [
            f"--model_file_list_path=@{file_.name}",
            f"--output_dir={self.temp_dir.name}",
        ]
        gen_all_oplist.main(args)
        mock_main.assert_called_once_with(args)
        file_.close()

    @patch("tools_copy.code_analyzer.gen_oplist_copy_from_core.main")
    def test_model_file_list_path_is_empty_directory_throws(
        self, mock_main: NonCallableMock
    ) -> None:
        file_ = tempfile.NamedTemporaryFile()
        args = [
            f"--model_file_list_path=@{file_.name}",
            f"--output_dir={self.temp_dir.name}",
        ]
        with self.assertRaises(AssertionError):
            gen_all_oplist.main(args)

    def tearDown(self):
        self.temp_dir.cleanup()
