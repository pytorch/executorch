# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from executorch.codegen.tools.combine_prim_ops_headers import (
    combine_prim_ops_headers,
    main,
    UNBOUNDED_SELECTION_MACRO,
)
from executorch.codegen.tools.gen_selected_prim_ops import (
    main as generate_header,
    UNBOUNDED_SELECTION_MACRO as GENERATOR_UNBOUNDED_MACRO,
)

ENABLE = "#define EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD"
INCLUDE_ET_VIEW = "#define INCLUDE_EXECUTORCH_PRIM_ET_VIEW_DEFAULT"
INCLUDE_ET_COPY_INDEX = "#define INCLUDE_EXECUTORCH_PRIM_ET_COPY_INDEX_DEFAULT"
INCLUDE_SYM_SIZE = "#define INCLUDE_ATEN_SYM_SIZE_INT"


class CombinePrimOpsHeadersTest(unittest.TestCase):
    """The combined header is the selected primitive-operator set.

    The combiner only runs for a build that asked for a selected set, so the
    union it produces is that set: an empty union means the build selected no
    primitive operator, not that the selection is unknown. A dependency that
    reports nothing at all is an error instead, since that is the case the
    empty union must not be confused with.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _input_dir(self, name: str, contents: str | None = None) -> Path:
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        if contents is not None:
            (directory / "selected_prim_ops.h").write_text(contents)
        return directory

    def _combine(self, sources: list[Path]) -> str:
        output = self.root / f"out_{len(list(self.root.iterdir()))}/selected_prim_ops.h"
        combine_prim_ops_headers([str(s) for s in sources], str(output))
        return output.read_text()

    def test_empty_union_selects_no_prim_ops(self) -> None:
        source = self._input_dir("empty", "#pragma once\n")
        text = self._combine([source])
        self.assertIn(ENABLE, text)
        self.assertNotIn("#define INCLUDE_", text)

    def test_union_of_empty_and_selected_keeps_the_selection(self) -> None:
        empty = self._input_dir("empty", "#pragma once\n")
        single = self._input_dir("single", INCLUDE_ET_VIEW + "\n")
        text = self._combine([empty, single])
        self.assertIn(ENABLE, text)
        self.assertIn(INCLUDE_ET_VIEW, text)
        self.assertEqual(text.count("#define INCLUDE_"), 1)

    def test_known_prim_ops_survive_the_union(self) -> None:
        first = self._input_dir("first", f"{INCLUDE_ET_VIEW}\n{INCLUDE_SYM_SIZE}\n")
        second = self._input_dir("second", INCLUDE_ET_COPY_INDEX + "\n")
        text = self._combine([first, second])
        for macro in (INCLUDE_ET_VIEW, INCLUDE_ET_COPY_INDEX, INCLUDE_SYM_SIZE):
            self.assertIn(macro, text)

    def test_non_prim_macros_are_carried_through(self) -> None:
        source = self._input_dir(
            "mixed", f"{INCLUDE_ET_VIEW}\n#define INCLUDE_ATEN_ADD_OUT\n"
        )
        text = self._combine([source])
        self.assertIn(INCLUDE_ET_VIEW, text)
        self.assertIn("#define INCLUDE_ATEN_ADD_OUT", text)

    def test_union_is_deduplicated_and_sorted(self) -> None:
        first = self._input_dir(
            "first", f"{INCLUDE_ET_VIEW}\n{INCLUDE_ET_COPY_INDEX}\n"
        )
        second = self._input_dir("second", INCLUDE_ET_VIEW + "\n")
        text = self._combine([first, second])
        self.assertEqual(text.count(INCLUDE_ET_VIEW), 1)
        self.assertLess(text.index(INCLUDE_ET_COPY_INDEX), text.index(INCLUDE_ET_VIEW))

    def test_a_missing_header_is_an_error(self) -> None:
        missing = self._input_dir("missing")
        with self.assertRaises(FileNotFoundError):
            self._combine([missing])

    def test_yaml_all_overloads_are_rejected_before_combined_header(self) -> None:
        for name in ("aten::sym_size", "executorch_prim::add"):
            for metadata in ({}, {"include_all_overloads": True}):
                with self.subTest(operator=name, metadata=metadata):
                    source = self._input_dir(
                        name.replace("::", "_") + str(len(metadata))
                    )
                    selection = source / "selected_operators.yaml"
                    selection.write_text(
                        json.dumps(
                            {
                                "include_all_operators": False,
                                "operators": {name: metadata},
                            }
                        )
                    )
                    generate_header(
                        [
                            f"--op-selection-yaml-path={selection}",
                            f"--output-dir={source}",
                        ]
                    )
                    output = source / "combined" / "selected_prim_ops.h"
                    with self.assertRaisesRegex(
                        ValueError, "include_all_prim_ops = True"
                    ):
                        combine_prim_ops_headers([str(source)], str(output))
                    self.assertFalse(output.exists())

    def test_an_unbounded_selection_is_an_error(self) -> None:
        unbounded = self._input_dir(
            "unbounded", f"#define {UNBOUNDED_SELECTION_MACRO}\n"
        )
        with self.assertRaises(ValueError) as caught:
            self._combine([unbounded])
        self.assertIn("include_all_prim_ops = True", str(caught.exception))

    def test_an_unbounded_selection_beside_a_real_one_is_still_an_error(self) -> None:
        selected = self._input_dir("selected", INCLUDE_ET_VIEW + "\n")
        unbounded = self._input_dir(
            "unbounded", f"#define {UNBOUNDED_SELECTION_MACRO}\n"
        )
        with self.assertRaises(ValueError):
            self._combine([selected, unbounded])

    def test_the_unbounded_marker_matches_the_generator(self) -> None:
        self.assertEqual(UNBOUNDED_SELECTION_MACRO, GENERATOR_UNBOUNDED_MACRO)

    def test_an_unreadable_header_is_an_error(self) -> None:
        source = self._input_dir("unreadable", "#pragma once\n")
        with mock.patch("builtins.open", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                combine_prim_ops_headers(
                    [str(source)], str(self.root / "out/selected_prim_ops.h")
                )

    def test_cli_writes_the_selected_set(self) -> None:
        source = self._input_dir("selected", INCLUDE_ET_VIEW + "\n")
        query_output = self.root / "query_output.txt"
        query_output.write_text(str(source) + "\n")
        output_dir = self.root / "cli"
        argv = [
            "combine_prim_ops_headers",
            "--header_files",
            "@" + str(query_output),
            "--output_dir",
            str(output_dir),
        ]
        with mock.patch.object(sys, "argv", argv):
            main()
        text = (output_dir / "selected_prim_ops.h").read_text()
        self.assertIn(ENABLE, text)
        self.assertIn(INCLUDE_ET_VIEW, text)

    def test_cli_writes_a_zero_prim_op_header(self) -> None:
        source = self._input_dir("empty", "#pragma once\n")
        query_output = self.root / "query_output.txt"
        query_output.write_text(str(source) + "\n")
        output_dir = self.root / "cli_empty"
        argv = [
            "combine_prim_ops_headers",
            "--header_files",
            "@" + str(query_output),
            "--output_dir",
            str(output_dir),
        ]
        with mock.patch.object(sys, "argv", argv):
            main()
        text = (output_dir / "selected_prim_ops.h").read_text()
        self.assertIn(ENABLE, text)
        self.assertNotIn("#define INCLUDE_", text)

    def test_cli_rejects_an_empty_header_list(self) -> None:
        query_output = self.root / "query_output.txt"
        query_output.write_text("")
        argv = [
            "combine_prim_ops_headers",
            "--header_files",
            "@" + str(query_output),
            "--output_dir",
            str(self.root / "cli_none"),
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 1)
        self.assertFalse(os.path.exists(self.root / "cli_none"))


if __name__ == "__main__":
    unittest.main()
