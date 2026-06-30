# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import yaml

from executorch.codegen.tools.gen_max_kernel_num import (
    _count_prim_ops,
    _count_yaml_kernels,
    gen_max_kernel_num,
)


PRIM_OPS_STUB = """\
// Helper defined above the array; its Kernel( must not be counted.
void build_helper_kernel() { something::Kernel(unrelated); }

static Kernel prim_ops[] = {
    Kernel("aten::sym_size.int", sym_size_int),
    Kernel(
        "executorch_prim::add.int_int",
        add_int_int),
    Kernel("executorch_prim::mul.int_int", mul_int_int),
};

// Another stray Kernel( below the array; also must not be counted.
auto misleading = Kernel("not_a_prim_op", nullptr);
"""


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload))


class TestGenMaxKernelNum(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.prim_ops_source = self.tmp / "register_prim_ops.cpp"
        self.prim_ops_source.write_text(PRIM_OPS_STUB)
        self.output = self.tmp / "selected_max_kernel_num.h"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_counts_prim_ops_from_source(self) -> None:
        self.assertEqual(_count_prim_ops(self.prim_ops_source), 3)

    def test_counts_prim_ops_errors_when_array_missing(self) -> None:
        empty = self.tmp / "empty.cpp"
        empty.write_text("// no kernels here\n")
        with self.assertRaises(RuntimeError):
            _count_prim_ops(empty)

    def test_counts_prim_ops_errors_when_array_empty(self) -> None:
        empty_array = self.tmp / "empty_array.cpp"
        empty_array.write_text("static Kernel prim_ops[] = {\n};\n")
        with self.assertRaises(RuntimeError):
            _count_prim_ops(empty_array)

    def test_counts_single_variant_per_op(self) -> None:
        yaml_path = self.tmp / "selected_operators.yaml"
        _write_yaml(
            yaml_path,
            {
                "operators": {
                    "aten::add.out": {"is_root_operator": True},
                    "aten::mul.out": {"is_root_operator": True},
                },
                "et_kernel_metadata": {
                    "aten::add.out": ["v1/6;0,1|6;0,1|6;0,1|6;0,1"],
                    "aten::mul.out": ["v1/6;0,1|6;0,1|6;0,1|6;0,1"],
                },
            },
        )
        self.assertEqual(_count_yaml_kernels(yaml_path), 2)

    def test_counts_multiple_variants_per_op(self) -> None:
        yaml_path = self.tmp / "selected_operators.yaml"
        _write_yaml(
            yaml_path,
            {
                "operators": {"aten::add.out": {"is_root_operator": True}},
                "et_kernel_metadata": {
                    "aten::add.out": [
                        "v1/6;0,1|6;0,1|6;0,1|6;0,1",
                        "v1/3;0,1|3;0,1|3;0,1|3;0,1",
                    ],
                },
            },
        )
        self.assertEqual(_count_yaml_kernels(yaml_path), 2)

    def test_counts_ops_without_metadata(self) -> None:
        yaml_path = self.tmp / "selected_operators.yaml"
        _write_yaml(
            yaml_path,
            {
                "operators": {
                    "aten::add.out": {"is_root_operator": True},
                    "aten::mul.out": {"is_root_operator": True},
                },
                "et_kernel_metadata": {},
            },
        )
        self.assertEqual(_count_yaml_kernels(yaml_path), 2)

    def test_include_all_operators_returns_none(self) -> None:
        yaml_path = self.tmp / "selected_operators.yaml"
        _write_yaml(
            yaml_path,
            {
                "include_all_operators": True,
                "operators": {},
                "et_kernel_metadata": {},
            },
        )
        self.assertIsNone(_count_yaml_kernels(yaml_path))

    def test_include_all_overloads_returns_none(self) -> None:
        yaml_path = self.tmp / "selected_operators.yaml"
        _write_yaml(
            yaml_path,
            {
                "operators": {
                    "aten::add": {"include_all_overloads": True},
                },
                "et_kernel_metadata": {},
            },
        )
        self.assertIsNone(_count_yaml_kernels(yaml_path))

    def test_end_to_end_single_yaml(self) -> None:
        yaml_path = self.tmp / "selected_operators.yaml"
        _write_yaml(
            yaml_path,
            {
                "operators": {"aten::add.out": {"is_root_operator": True}},
                "et_kernel_metadata": {
                    "aten::add.out": ["v1/6;0,1|6;0,1|6;0,1|6;0,1"],
                },
            },
        )
        total = gen_max_kernel_num(
            oplist_yamls=[yaml_path],
            prim_ops_source=self.prim_ops_source,
            output_path=self.output,
        )
        self.assertEqual(total, 1 + 3)
        self.assertIn(
            "#define EXECUTORCH_SELECTED_MAX_KERNEL_NUM 4",
            self.output.read_text(),
        )

    def test_end_to_end_multiple_yamls(self) -> None:
        yaml_a = self.tmp / "a.yaml"
        yaml_b = self.tmp / "b.yaml"
        _write_yaml(
            yaml_a,
            {
                "operators": {"aten::add.out": {}},
                "et_kernel_metadata": {"aten::add.out": ["v1/6", "v1/3"]},
            },
        )
        _write_yaml(
            yaml_b,
            {
                "operators": {"aten::mul.out": {}},
                "et_kernel_metadata": {"aten::mul.out": ["v1/6"]},
            },
        )
        total = gen_max_kernel_num(
            oplist_yamls=[yaml_a, yaml_b],
            prim_ops_source=self.prim_ops_source,
            output_path=self.output,
        )
        self.assertEqual(total, 2 + 1 + 3)

    def test_include_all_writes_opt_out_header(self) -> None:
        yaml_path = self.tmp / "selected_operators.yaml"
        _write_yaml(
            yaml_path,
            {"include_all_operators": True, "operators": {}, "et_kernel_metadata": {}},
        )
        total = gen_max_kernel_num(
            oplist_yamls=[yaml_path],
            prim_ops_source=self.prim_ops_source,
            output_path=self.output,
        )
        self.assertIsNone(total)
        self.assertTrue(self.output.exists())
        contents = self.output.read_text()
        self.assertNotIn("#define EXECUTORCH_SELECTED_MAX_KERNEL_NUM", contents)
        self.assertIn("opted into all operators", contents)

    def test_write_if_different_preserves_mtime(self) -> None:
        yaml_path = self.tmp / "selected_operators.yaml"
        _write_yaml(
            yaml_path,
            {
                "operators": {"aten::add.out": {}},
                "et_kernel_metadata": {"aten::add.out": ["v1/6"]},
            },
        )
        gen_max_kernel_num(
            oplist_yamls=[yaml_path],
            prim_ops_source=self.prim_ops_source,
            output_path=self.output,
        )
        first_mtime = self.output.stat().st_mtime_ns
        gen_max_kernel_num(
            oplist_yamls=[yaml_path],
            prim_ops_source=self.prim_ops_source,
            output_path=self.output,
        )
        self.assertEqual(self.output.stat().st_mtime_ns, first_mtime)


if __name__ == "__main__":
    unittest.main()
