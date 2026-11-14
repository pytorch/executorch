# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from executorch.codegen.tools.gen_selected_prim_ops import (
    main,
    normalize_op_name,
    read_prim_op_names_from_yaml,
    selection_is_unbounded,
    UNBOUNDED_SELECTION_MACRO,
    write_selected_prim_ops,
)


class TestGenSelectedPrimOps(unittest.TestCase):
    def test_normalizes_aten_op_with_leading_underscore(self) -> None:
        self.assertEqual(
            normalize_op_name("aten::_local_scalar_dense"),
            "INCLUDE_ATEN_LOCAL_SCALAR_DENSE",
        )

    def test_writes_selected_prim_ops_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "operators": {
                            "executorch_prim::et_view.default": {
                                "include_all_overloads": False
                            },
                            "aten::_local_scalar_dense": {
                                "include_all_overloads": False
                            },
                        },
                        "et_kernel_metadata": {
                            "aten::sym_size.int": ["default"],
                        },
                    }
                )
            )

            main(
                [
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )

            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn("#define INCLUDE_ATEN_LOCAL_SCALAR_DENSE", header)
            self.assertIn("#define INCLUDE_ATEN_SYM_SIZE_INT", header)
            self.assertIn("#define INCLUDE_EXECUTORCH_PRIM_ET_VIEW_DEFAULT", header)

    def test_cli_does_not_mix_selection_snapshots(self) -> None:
        versions = [
            yaml.safe_dump(
                {"operators": {"aten::sym_size.int": {"include_all_overloads": False}}}
            ),
            yaml.safe_dump({"include_all_operators": True, "operators": {}}),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(versions[0])
            snapshots = iter(versions)
            original_open = open

            def changing_open(path, mode="r", *args, **kwargs):
                if path == str(yaml_path) and mode == "r":
                    return io.StringIO(next(snapshots, versions[-1]))
                return original_open(path, mode, *args, **kwargs)

            with mock.patch("builtins.open", side_effect=changing_open):
                main(
                    [
                        f"--op-selection-yaml-path={yaml_path}",
                        f"--output-dir={temp_dir}",
                    ]
                )
            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn(
                (
                    "#define INCLUDE_ATEN_SYM_SIZE_INT" in header,
                    f"#define {UNBOUNDED_SELECTION_MACRO}" in header,
                ),
                ((True, False), (False, True)),
            )

    def test_all_overloads_are_not_an_enumerated_prim_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "include_all_operators": False,
                        "operators": {
                            "aten::sym_size": {
                                "is_root_operator": True,
                                "is_used_for_training": False,
                                "include_all_overloads": True,
                            },
                        },
                    }
                )
            )
            main(
                [
                    "--prim-op-names=executorch_prim::et_view.default",
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )
            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn(f"#define {UNBOUNDED_SELECTION_MACRO}", header)
            self.assertIn("EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD", header)
            self.assertIn("#error", header)
            self.assertIn("#define INCLUDE_EXECUTORCH_PRIM_ET_VIEW_DEFAULT", header)
            self.assertNotIn("#define INCLUDE_ATEN_SYM_SIZE\n", header)

    def test_missing_overload_flag_uses_selector_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(yaml.safe_dump({"operators": {"aten::sym_size": {}}}))
            main(
                [
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )
            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn(f"#define {UNBOUNDED_SELECTION_MACRO}", header)
            self.assertNotIn("#define INCLUDE_ATEN_SYM_SIZE\n", header)

    def test_invalid_overload_flag_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {"operators": {"aten::sym_size": {"include_all_overloads": None}}}
                )
            )
            with self.assertRaises(AssertionError):
                main(
                    [
                        f"--op-selection-yaml-path={yaml_path}",
                        f"--output-dir={temp_dir}",
                    ]
                )
            self.assertFalse((Path(temp_dir) / "selected_prim_ops.h").exists())

    def test_kernel_metadata_does_not_bound_all_overloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "operators": {
                            "aten::sym_size": {"include_all_overloads": True}
                        },
                        "et_kernel_metadata": {"aten::sym_size.int": ["default"]},
                    }
                )
            )
            main(
                [
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )
            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn(f"#define {UNBOUNDED_SELECTION_MACRO}", header)
            self.assertIn("#define INCLUDE_ATEN_SYM_SIZE_INT", header)

    def test_explicit_overloads_remain_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "include_all_operators": False,
                        "operators": {
                            "aten::sym_size.int": {"include_all_overloads": False},
                            "executorch_prim::add.Scalar": {
                                "include_all_overloads": False
                            },
                        },
                    }
                )
            )
            main(
                [
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )
            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn("#define INCLUDE_ATEN_SYM_SIZE_INT", header)
            self.assertIn("#define INCLUDE_EXECUTORCH_PRIM_ADD_SCALAR", header)
            self.assertNotIn(UNBOUNDED_SELECTION_MACRO, header)

    def test_writes_empty_header_for_empty_op_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_selected_prim_ops([], temp_dir)

            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertNotIn("#define INCLUDE_", header)

    def test_empty_selection_yaml_selects_no_prim_ops(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(yaml.safe_dump({"operators": {}}))

            self.assertEqual(read_prim_op_names_from_yaml(str(yaml_path)), [])

    def test_missing_selection_yaml_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                read_prim_op_names_from_yaml(
                    str(Path(temp_dir) / "selected_operators.yaml")
                )

    def test_malformed_selection_yaml_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text("- not\n- a mapping\n")

            with self.assertRaises(ValueError):
                read_prim_op_names_from_yaml(str(yaml_path))

    def test_prim_op_names_and_yaml_are_unioned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {"operators": {"aten::add.out": {"include_all_overloads": False}}}
                )
            )

            main(
                [
                    "--prim-op-names=executorch_prim::et_view.default",
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )

            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn("#define INCLUDE_EXECUTORCH_PRIM_ET_VIEW_DEFAULT", header)
            self.assertIn("#define INCLUDE_ATEN_ADD_OUT", header)

    def test_an_all_operators_selection_is_reported_as_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump({"operators": {}, "include_all_operators": True})
            )

            self.assertTrue(selection_is_unbounded(str(yaml_path)))

            main(
                [
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )

            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertIn(f"#define {UNBOUNDED_SELECTION_MACRO}", header)
            self.assertIn("#error", header)
            self.assertIn("EXECUTORCH_ENABLE_PRIM_OPS_SELECTIVE_BUILD", header)

    def test_a_listed_selection_is_not_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "selected_operators.yaml"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "operators": {
                            "aten::add.out": {"include_all_overloads": False}
                        },
                        "include_all_operators": False,
                    }
                )
            )

            self.assertFalse(selection_is_unbounded(str(yaml_path)))

            main(
                [
                    f"--op-selection-yaml-path={yaml_path}",
                    f"--output-dir={temp_dir}",
                ]
            )

            header = (Path(temp_dir) / "selected_prim_ops.h").read_text()
            self.assertNotIn(UNBOUNDED_SELECTION_MACRO, header)
            self.assertNotIn("#error", header)


if __name__ == "__main__":
    unittest.main()
