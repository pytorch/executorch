# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest
from typing import Dict, List
from unittest.mock import NonCallableMock, patch

import executorch.codegen.tools.gen_oplist as gen_oplist
import yaml


class TestGenOpList(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.ops_schema_yaml = os.path.join(self.temp_dir.name, "test.yaml")
        with open(self.ops_schema_yaml, "w") as f:
            f.write(
                """
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
            """
            )

    @patch("executorch.codegen.tools.gen_oplist._get_operators")
    @patch("executorch.codegen.tools.gen_oplist._dump_yaml")
    def test_gen_op_list_with_wrong_path(
        self,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
    ) -> None:
        args = ["--output_path=wrong_path", "--model_file_path=path2"]
        with self.assertRaises(RuntimeError):
            gen_oplist.main(args)

    @patch("executorch.codegen.tools.gen_oplist._get_kernel_metadata_for_model")
    @patch("executorch.codegen.tools.gen_oplist._get_operators")
    @patch("executorch.codegen.tools.gen_oplist._dump_yaml")
    def test_gen_op_list_with_valid_model_path(
        self,
        mock_get_kernel_metadata_for_model: NonCallableMock,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
    ) -> None:
        temp_file = tempfile.NamedTemporaryFile()
        args = [
            f"--output_path={os.path.join(self.temp_dir.name, 'output.yaml')}",
            f"--model_file_path={temp_file.name}",
        ]
        gen_oplist.main(args)
        mock_get_operators.assert_called_once_with(temp_file.name)
        temp_file.close()

    @patch("executorch.codegen.tools.gen_oplist._dump_yaml")
    def test_gen_op_list_with_valid_root_ops(
        self,
        mock_dump_yaml: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        args = [
            f"--output_path={output_path}",
            "--root_ops=aten::add,aten::mul",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add", "aten::mul"],
            output_path,
            None,
            {"aten::add": ["default"], "aten::mul": ["default"]},
            False,
        )

    @patch("executorch.codegen.tools.gen_oplist._dump_yaml")
    def test_gen_op_list_with_root_ops_and_dtypes(
        self,
        mock_dump_yaml: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        ops_dict = {
            "aten::add": ["v1/3;0,1|3;0,1|3;0,1|3;0,1", "v1/6;0,1|6;0,1|6;0,1|6;0,1"],
            "aten::mul": [],
        }
        args = [
            f"--output_path={output_path}",
            f"--ops_dict={json.dumps(ops_dict)}",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add", "aten::mul"],
            output_path,
            None,
            {
                "aten::add": [
                    "v1/3;0,1|3;0,1|3;0,1|3;0,1",
                    "v1/6;0,1|6;0,1|6;0,1|6;0,1",
                ],
                "aten::mul": ["default"],
            },
            False,
        )

    @patch("executorch.codegen.tools.gen_oplist._get_operators")
    @patch("executorch.codegen.tools.gen_oplist._dump_yaml")
    def test_gen_op_list_with_both_op_list_and_ops_schema_yaml_merges(
        self,
        mock_dump_yaml: NonCallableMock,
        mock_get_operators: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        args = [
            f"--output_path={output_path}",
            "--root_ops=aten::relu.out",
            f"--ops_schema_yaml_path={self.ops_schema_yaml}",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add.out", "aten::mul.out", "aten::relu.out"],
            output_path,
            "test.yaml",
            {
                "aten::relu.out": ["default"],
                "aten::add.out": ["default"],
                "aten::mul.out": ["default"],
            },
            False,
        )

    @patch("executorch.codegen.tools.gen_oplist._dump_yaml")
    def test_gen_op_list_with_include_all_operators(
        self,
        mock_dump_yaml: NonCallableMock,
    ) -> None:
        output_path = os.path.join(self.temp_dir.name, "output.yaml")
        args = [
            f"--output_path={output_path}",
            "--root_ops=aten::add,aten::mul",
            "--include_all_operators",
        ]
        gen_oplist.main(args)
        mock_dump_yaml.assert_called_once_with(
            ["aten::add", "aten::mul"],
            output_path,
            None,
            {"aten::add": ["default"], "aten::mul": ["default"]},
            True,
        )

    def test_get_custom_build_selector_with_both_allowlist_and_yaml(
        self,
    ) -> None:
        op_list = ["aten::add", "aten::mul"]
        filename = os.path.join(self.temp_dir.name, "selected_operators.yaml")
        gen_oplist._dump_yaml(op_list, filename, "model.pte")
        self.assertTrue(os.path.isfile(filename))
        with open(filename) as f:
            es = yaml.safe_load(f)
        ops = es["operators"]
        self.assertEqual(len(ops), 2)
        self.assertSetEqual(set(ops.keys()), set(op_list))

    def test_gen_oplist_generates_from_root_ops(
        self,
    ) -> None:
        filename = os.path.join(self.temp_dir.name, "selected_operators.yaml")
        op_list = ["aten::add.out", "aten::mul.out", "aten::relu.out"]
        comma = ","
        args = [
            f"--output_path={filename}",
            f"--root_ops={comma.join(op_list)}",
        ]
        gen_oplist.main(args)
        self.assertTrue(os.path.isfile(filename))
        with open(filename) as f:
            es = yaml.safe_load(f)
        ops = es["operators"]
        self.assertEqual(len(ops), 3)
        self.assertSetEqual(set(ops.keys()), set(op_list))

    def test_dump_operator_from_ops_schema_yaml(self) -> None:
        ops = gen_oplist._get_et_kernel_metadata_from_ops_yaml(self.ops_schema_yaml)
        self.assertListEqual(sorted(ops.keys()), ["aten::add.out", "aten::mul.out"])

    def test_dump_operator_from_ops_schema_yaml_with_op_syntax(self) -> None:
        ops_yaml = os.path.join(self.temp_dir.name, "ops.yaml")
        with open(ops_yaml, "w") as f:
            f.write(
                """
- op: add.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- op: mul.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
            """
            )
        ops = gen_oplist._get_et_kernel_metadata_from_ops_yaml(ops_yaml)
        self.assertListEqual(sorted(ops.keys()), ["aten::add.out", "aten::mul.out"])

    def test_dump_operator_from_ops_schema_yaml_with_mix_syntax(self) -> None:
        mix_yaml = os.path.join(self.temp_dir.name, "mix.yaml")
        with open(mix_yaml, "w") as f:
            f.write(
                """
- op: add.out
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::add_out_kernel

- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU: torch::executor::mul_out_kernel
            """
            )
        ops = gen_oplist._get_et_kernel_metadata_from_ops_yaml(mix_yaml)
        self.assertListEqual(sorted(ops.keys()), ["aten::add.out", "aten::mul.out"])

    def test_get_kernel_metadata_from_ops_yaml(self) -> None:
        metadata: Dict[str, List[str]] = (
            gen_oplist._get_et_kernel_metadata_from_ops_yaml(self.ops_schema_yaml)
        )

        self.assertEqual(len(metadata), 2)

        self.assertIn("aten::add.out", metadata)
        # We only have one dtype/dim-order combo for add (float/0,1)
        self.assertEqual(len(metadata["aten::add.out"]), 1)
        self.assertEqual(
            metadata["aten::add.out"][0],
            "default",
        )

        self.assertIn("aten::mul.out", metadata)
        self.assertEqual(len(metadata["aten::mul.out"]), 1)
        self.assertEqual(
            metadata["aten::mul.out"][0],
            "default",
        )

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
