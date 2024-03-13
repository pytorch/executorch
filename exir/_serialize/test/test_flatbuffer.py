#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import shutil
import tempfile
import unittest
from typing import Dict, Optional, Sequence
from unittest.mock import patch

from executorch.exir._serialize import _flatbuffer
from executorch.exir._serialize._flatbuffer import (
    _program_json_to_flatbuffer,
    _ResourceFiles,
    _SchemaInfo,
)


def read_file(dir: str, filename: str) -> bytes:
    """Returns the contents of the given file."""
    with open(os.path.join(dir, filename), "rb") as fp:
        return fp.read()


# Fake resource files to use when testing _ResourceFiles.
FAKE_RESOURCES: Dict[str, bytes] = {
    "resource-1": b"resource-1 data",
    "resource-2": b"resource-2 data",
}


class TestResourceFiles(unittest.TestCase):
    def make_resource_files(self, files: Dict[str, bytes]) -> _ResourceFiles:
        """Returns a _ResourceFiles containing the injected fake files.

        Args:
            files: Mapping of filename to contents.
        """
        with patch.object(
            _flatbuffer.importlib.resources, "read_binary"
        ) as mock_read_binary:
            # Use the fake resource files when looking up resources.
            mock_read_binary.side_effect = lambda _, name: files[name]
            return _ResourceFiles(tuple(files.keys()))

    def test_load_and_write(self) -> None:
        rf: _ResourceFiles = self.make_resource_files(FAKE_RESOURCES)
        with tempfile.TemporaryDirectory() as out_dir:
            # Write the unmodified inputs to the filesystem.
            rf.write_to(out_dir)
            self.assertEqual(read_file(out_dir, "resource-1"), b"resource-1 data")
            self.assertEqual(read_file(out_dir, "resource-2"), b"resource-2 data")

    def test_load_patch_and_write(self) -> None:
        rf: _ResourceFiles = self.make_resource_files(FAKE_RESOURCES)

        # Append something to the end of each file.
        rf.patch_files(lambda data: data + b" PATCHED")

        with tempfile.TemporaryDirectory() as out_dir:
            rf.write_to(out_dir)
            self.assertEqual(
                read_file(out_dir, "resource-1"), b"resource-1 data PATCHED"
            )
            self.assertEqual(
                read_file(out_dir, "resource-2"), b"resource-2 data PATCHED"
            )


# Fake resource files to use when testing alignment-patching.
SCHEMA_FILES: Dict[str, bytes] = {
    "program.fbs": b"\n".join(
        [
            b"table Program {",
            # Space after the colon.
            b"  tensor_data: [ubyte] (force_align: 8); // @executorch-tensor-alignment",
            # No spaces around the colon.
            b"  delegate_data: [ubyte] (force_align:16); // @executorch-delegate-alignment",
            b"  other_data: [ubyte] (force_align: 32);",
            b"}",
        ]
    ),
    "scalar_type.fbs": b"\n".join(
        [
            b"table ScalarType {",
            # Spaces around the colon.
            b"  tensor_data: [ubyte] (force_align : 8); // @executorch-tensor-alignment",
            # Spaces between all tokens.
            b"  delegate_data: [ubyte] ( force_align : 16 ); // @executorch-delegate-alignment",
            b"  other_data: [ubyte] (force_align: 64);",
            b"}",
        ]
    ),
}


# Bad alignment values; not whole powers of 2.
BAD_ALIGNMENTS: Sequence[int] = (-1, 0, 5)


class TestPrepareSchema(unittest.TestCase):
    def call_prepare_schema(
        self,
        schema_files: Dict[str, bytes],
        out_dir: str,
        constant_tensor_alignment: Optional[int] = None,
        delegate_alignment: Optional[int] = None,
    ) -> _SchemaInfo:
        """Calls _prepare_schema(), using `files` to get the original contents
        of the schema files.
        """
        with patch.object(
            _flatbuffer.importlib.resources, "read_binary"
        ) as mock_read_binary:
            # Use the fake resource files when looking up resources.
            mock_read_binary.side_effect = lambda _, name: schema_files[name]
            return _flatbuffer._prepare_schema(
                out_dir=out_dir,
                constant_tensor_alignment=constant_tensor_alignment,
                delegate_alignment=delegate_alignment,
            )

    def test_unmodified(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            info: _SchemaInfo = self.call_prepare_schema(SCHEMA_FILES, out_dir)
            self.assertEqual(info.root_path, os.path.join(out_dir, "program.fbs"))
            # Files should not have been modified.
            for fname in SCHEMA_FILES.keys():
                self.assertEqual(read_file(out_dir, fname), SCHEMA_FILES[fname])
            # Max alignment should be the largest value in the input.
            self.assertEqual(info.max_alignment, 64)

    def test_update_tensor_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            info: _SchemaInfo = self.call_prepare_schema(
                SCHEMA_FILES, out_dir, constant_tensor_alignment=128
            )
            self.assertEqual(info.root_path, os.path.join(out_dir, "program.fbs"))
            # Only the tensor alignment lines should have been modified.
            self.assertEqual(
                read_file(out_dir, "program.fbs"),
                b"\n".join(
                    [
                        b"table Program {",
                        # Now 128:
                        b"  tensor_data: [ubyte] (force_align: 128); // @executorch-tensor-alignment",
                        b"  delegate_data: [ubyte] (force_align:16); // @executorch-delegate-alignment",
                        b"  other_data: [ubyte] (force_align: 32);",
                        b"}",
                    ]
                ),
            )
            self.assertEqual(
                read_file(out_dir, "scalar_type.fbs"),
                b"\n".join(
                    [
                        b"table ScalarType {",
                        # Now 128, and reformatted:
                        b"  tensor_data: [ubyte] (force_align: 128); // @executorch-tensor-alignment",
                        b"  delegate_data: [ubyte] ( force_align : 16 ); // @executorch-delegate-alignment",
                        b"  other_data: [ubyte] (force_align: 64);",
                        b"}",
                    ]
                ),
            )
            # Max alignment should reflect this change.
            self.assertEqual(info.max_alignment, 128)

    def test_update_delegate_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            info: _SchemaInfo = self.call_prepare_schema(
                SCHEMA_FILES, out_dir, delegate_alignment=256
            )
            self.assertEqual(info.root_path, os.path.join(out_dir, "program.fbs"))
            # Only the delegate alignment lines should have been modified.
            self.assertEqual(
                read_file(out_dir, "program.fbs"),
                b"\n".join(
                    [
                        b"table Program {",
                        b"  tensor_data: [ubyte] (force_align: 8); // @executorch-tensor-alignment",
                        # Now 256:
                        b"  delegate_data: [ubyte] (force_align: 256); // @executorch-delegate-alignment",
                        b"  other_data: [ubyte] (force_align: 32);",
                        b"}",
                    ]
                ),
            )
            self.assertEqual(
                read_file(out_dir, "scalar_type.fbs"),
                b"\n".join(
                    [
                        b"table ScalarType {",
                        b"  tensor_data: [ubyte] (force_align : 8); // @executorch-tensor-alignment",
                        # Now 256, and reformatted:
                        b"  delegate_data: [ubyte] (force_align: 256); // @executorch-delegate-alignment",
                        b"  other_data: [ubyte] (force_align: 64);",
                        b"}",
                    ]
                ),
            )
            # Max alignment should reflect this change.
            self.assertEqual(info.max_alignment, 256)

    def test_update_tensor_and_delegate_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            info: _SchemaInfo = self.call_prepare_schema(
                SCHEMA_FILES,
                out_dir,
                constant_tensor_alignment=1,
                delegate_alignment=2,
            )
            self.assertEqual(info.root_path, os.path.join(out_dir, "program.fbs"))
            # Only the delegate alignment lines should have been modified.
            self.assertEqual(
                read_file(out_dir, "program.fbs"),
                b"\n".join(
                    [
                        b"table Program {",
                        # Now 1:
                        b"  tensor_data: [ubyte] (force_align: 1); // @executorch-tensor-alignment",
                        # Now 2:
                        b"  delegate_data: [ubyte] (force_align: 2); // @executorch-delegate-alignment",
                        b"  other_data: [ubyte] (force_align: 32);",
                        b"}",
                    ]
                ),
            )
            self.assertEqual(
                read_file(out_dir, "scalar_type.fbs"),
                b"\n".join(
                    [
                        b"table ScalarType {",
                        # Now 1, and reformatted:
                        b"  tensor_data: [ubyte] (force_align: 1); // @executorch-tensor-alignment",
                        # Now 2, and reformatted:
                        b"  delegate_data: [ubyte] (force_align: 2); // @executorch-delegate-alignment",
                        b"  other_data: [ubyte] (force_align: 64);",
                        b"}",
                    ]
                ),
            )
            self.assertEqual(info.max_alignment, 64)

    def test_bad_tensor_alignment_fails(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            for bad_alignment in BAD_ALIGNMENTS:
                # subTest will create a different top-level test entry for each
                # value, whose full names have a suffix like "(bad_alignment=5)".
                with self.subTest(bad_alignment=bad_alignment):
                    with self.assertRaises(ValueError):
                        self.call_prepare_schema(
                            SCHEMA_FILES,
                            out_dir,
                            constant_tensor_alignment=bad_alignment,
                        )

    def test_bad_delegate_alignment_fails(self) -> None:
        with tempfile.TemporaryDirectory() as out_dir:
            for bad_alignment in BAD_ALIGNMENTS:
                # subTest will create a different top-level test entry for each
                # value, whose full names have a suffix like "(bad_alignment=5)".
                with self.subTest(bad_alignment=bad_alignment):
                    with self.assertRaises(ValueError):
                        self.call_prepare_schema(
                            SCHEMA_FILES,
                            out_dir,
                            delegate_alignment=bad_alignment,
                        )


class TestProgramJsonToFlatbuffer(unittest.TestCase):
    @patch.dict(os.environ, {_flatbuffer._SAVE_FLATC_ENV: "1"})
    def test_save_json_on_failure(self) -> None:
        err_msg: Optional[str] = None
        try:
            _program_json_to_flatbuffer("} some bad json {")
            self.fail("Should have raised an exception")
        except RuntimeError as err:
            err_msg = err.args[0]

        self.assertIsNotNone(err_msg)
        match = re.search(r"Moved input files to '(.*?)'", err_msg)
        self.assertTrue(match, msg=f"Unexpected error message: {err_msg}")
        path = match.group(1)

        files = frozenset(os.listdir(path))
        # Delete the files otherwise they'll accumulate every time the
        # test is run.
        shutil.rmtree(path)
        # Check for a couple of the files that should be there.
        self.assertIn("data.json", files)
        self.assertIn("program.fbs", files)

    @patch.dict(os.environ, {_flatbuffer._SAVE_FLATC_ENV: "1"})
    def test_unable_to_save_json_on_failure(self) -> None:
        err_msg: Optional[str] = None
        try:
            with patch.object(
                _flatbuffer.shutil,
                "move",
                side_effect=Exception("shutil.move mock failure"),
            ):
                _program_json_to_flatbuffer("} some bad json {")
            self.fail("Should have raised an exception")
        except RuntimeError as err:
            err_msg = err.args[0]

        self.assertIsNotNone(err_msg)
        self.assertIn("Failed to save input files", err_msg)

    @patch.dict(os.environ, {_flatbuffer._SAVE_FLATC_ENV: ""})
    def test_no_save_json_on_failure(self) -> None:
        err_msg: Optional[str] = None
        try:
            _program_json_to_flatbuffer("} some bad json {")
            self.fail("Should have raised an exception")
        except RuntimeError as err:
            err_msg = err.args[0]

        self.assertIsNotNone(err_msg)
        self.assertIn(
            f"Set {_flatbuffer._SAVE_FLATC_ENV}=1 to save input files", err_msg
        )
        self.assertNotIn("Moved input files", err_msg)
        self.assertNotIn("Failed to save input files", err_msg)
