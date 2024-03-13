# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from typing import Any, Dict, List, Set

import torch
from executorch.exir.dialects.edge.dtype.supported import regular_tensor_dtypes_to_str

from executorch.exir.dialects.edge.spec.gen import (
    EdgeOpYamlInfo,
    gen_op_yaml,
    get_sample_input,
)


class TestEdgeYaml(unittest.TestCase):
    def assertTypeAliasEqual(
        self, type_alias_1: List[List[str]], type_alias_2: List[List[str]]
    ) -> None:
        """Helper function to assert two type alias equal"""
        self.assertEqual(len(type_alias_1), len(type_alias_2))
        type_alias_set_1: List[Set[str]] = []
        type_alias_set_2: List[Set[str]] = []
        for ta1, ta2 in zip(type_alias_1, type_alias_2):
            self.assertEqual(len(ta1), len(set(ta1)))
            self.assertEqual(len(ta2), len(set(ta2)))
            type_alias_set_1.append(set(ta1))
            type_alias_set_2.append(set(ta2))

        for tas1, tas2 in zip(type_alias_set_1, type_alias_set_2):
            self.assertTrue(tas1 in type_alias_set_2)
            self.assertTrue(tas2 in type_alias_set_1)

    def assertOpYamlEqual(
        self, op_yaml_1: Dict[str, Any], op_yaml_2: Dict[str, Any]
    ) -> None:
        """Helper function to assert two edge operator yaml object equal"""

        for op_yaml_key in op_yaml_1:
            self.assertTrue(op_yaml_key in op_yaml_2)
            if op_yaml_key == "type_alias":
                self.assertEqual(
                    len(op_yaml_1[op_yaml_key]), len(op_yaml_2[op_yaml_key])
                )
                type_alias_list_1: List[List[str]] = []
                type_alias_list_2: List[List[str]] = []
                for type_alias_key in op_yaml_1[op_yaml_key]:
                    self.assertTrue(type_alias_key in op_yaml_2[op_yaml_key])
                    type_alias_list_1.append(op_yaml_1[op_yaml_key][type_alias_key])
                    type_alias_list_2.append(op_yaml_2[op_yaml_key][type_alias_key])

                self.assertTypeAliasEqual(type_alias_list_1, type_alias_list_2)
            else:
                self.assertEqual(op_yaml_1[op_yaml_key], op_yaml_2[op_yaml_key])

        self.assertEqual(op_yaml_1["func"], op_yaml_2["func"])
        self.assertEqual(op_yaml_1["namespace"], op_yaml_2["namespace"])

    def assertEdgeYamlEqual(
        self, edge_yaml_1: List[Dict[str, Any]], edge_yaml_2: List[Dict[str, Any]]
    ) -> None:
        """Helper function to assert two edge dialect yaml object equal"""
        self.assertEqual(len(edge_yaml_1), len(edge_yaml_2))
        dict_edge_yaml_1: Dict[str, Dict[str, Any]] = {
            op["func"]: op for op in edge_yaml_1
        }
        dict_edge_yaml_2: Dict[str, Dict[str, Any]] = {
            op["func"]: op for op in edge_yaml_2
        }

        for op_yaml_key in dict_edge_yaml_1:
            assert op_yaml_key in dict_edge_yaml_2
            op_yaml_1, op_yaml_2 = (
                dict_edge_yaml_1[op_yaml_key],
                dict_edge_yaml_2[op_yaml_key],
            )
            self.assertOpYamlEqual(op_yaml_1, op_yaml_2)

    def test_edge_op_yaml_info_combine_types_with_all_same_types(self) -> None:
        """This test aims to check if EdgeOpYamlInfo can a. generate correct type
        alias and type constraint and b. properly combine the type combinations with
        all same input types (e.g. (FloatTensor, FloatTensor, FloatTensor),
        (DoubleTensor, DoubleTensor, DoubleTensor)).
        """

        example_yaml_info = EdgeOpYamlInfo(
            func_name="add.Tensor",
            tensor_variable_names=["self", "other", "__ret"],
            inherits="aten::add.Tensor",
            allowed_types={
                ("Float", "Float", "Float"),
                ("Double", "Double", "Double"),
                ("Char", "Char", "Int"),
            },
        )

        self.assertEqual(example_yaml_info.func_name, "add.Tensor")
        self.assertEqual(
            example_yaml_info.tensor_variable_names, ["self", "other", "__ret"]
        )
        self.assertEqual(example_yaml_info.inherits, "aten::add.Tensor")
        self.assertEqual(example_yaml_info.custom, "")
        self.assertEqual(
            example_yaml_info.type_alias,
            [("Char",), ("Double", "Float"), ("Int",)],
        )
        self.assertEqual(example_yaml_info.type_constraint, [(0, 0, 2), (1, 1, 1)])

    def test_edge_op_yaml_info_combine_same_format(self) -> None:
        """This test aims to check if EdgeOpYamlInfo can a. generate correct type
        alias and type constraint and b. properly combine the inputs with same format.
        Two inputs having same format here means one and only one of their corresponding
        input tensors is different. e.g. {DoubleTensor, DoubleTensor), FloatTensor}
        shares same format with {DoubleTensor, DoubleTensor, DoubleTensor},
        but not {DoubleTensor, FloatTensor, DoubleTensor}.

        """

        example_yaml_info = EdgeOpYamlInfo(
            func_name="tanh",
            tensor_variable_names=["self", "__ret_0"],
            inherits="aten::tanh",
            allowed_types={
                ("Bool", "Float"),
                ("Byte", "Float"),
                ("Char", "Float"),
                ("Short", "Float"),
                ("Int", "Float"),
                ("Long", "Int"),
                ("Float", "Float"),
                ("Double", "Double"),
            },
        )

        self.assertEqual(example_yaml_info.func_name, "tanh")
        self.assertEqual(example_yaml_info.tensor_variable_names, ["self", "__ret_0"])
        self.assertEqual(example_yaml_info.inherits, "aten::tanh")
        self.assertEqual(example_yaml_info.custom, "")
        self.assertEqual(
            example_yaml_info.type_alias,
            [
                ("Bool", "Byte", "Char", "Float", "Int", "Short"),
                ("Double",),
                ("Float",),
                ("Int",),
                ("Long",),
            ],
        )
        self.assertEqual(example_yaml_info.type_constraint, [(0, 2), (1, 1), (4, 3)])

    def test_optional_tensor_supported(self) -> None:
        # Two of three tensor inputs of native_layer_norm are in optional tensor type.
        ret = gen_op_yaml("native_layer_norm.default")
        self.assertTrue(ret is not None)
        self.assertEqual(ret.func_name, "aten::native_layer_norm")
        self.assertEqual(ret.inherits, "aten::native_layer_norm")
        self.assertEqual(ret.custom, "")
        self.assertEqual(ret.type_alias, [("Double", "Float", "Half")])
        self.assertEqual(ret.type_constraint, [(0, 0, 0, 0, 0, 0)])
        self.assertEqual(
            ret.tensor_variable_names,
            ["input", "weight", "bias", "__ret_0", "__ret_1", "__ret_2"],
        )

    def test_tensor_list_supported(self) -> None:
        # Input of cat is tensor list.
        ret = gen_op_yaml("cat.default")
        self.assertTrue(ret is not None)
        self.assertEqual(ret.func_name, "aten::cat")
        self.assertEqual(ret.inherits, "aten::cat")
        self.assertEqual(ret.custom, "")
        self.assertEqual(
            ret.type_alias,
            [
                (
                    "Bool",
                    "Byte",
                    "Char",
                    "Double",
                    "Float",
                    "Half",
                    "Int",
                    "Long",
                    "Short",
                )
            ],
        )
        self.assertEqual(ret.type_constraint, [(0, 0)])
        self.assertEqual(ret.tensor_variable_names, ["tensors", "__ret_0"])

    # Check if any function updated by comparing the current yaml file with
    # previous one. If anything mismatch, please follow the instructions at the
    # top of //executorch/exir/dialects/edge/edge.yaml.
    # TODO(gasoonjia, T159593834): Should be updated after support other models and infer methods.
    # def test_need_update_edge_yaml(self) -> None:
    #     model = <need OSS model example>
    #     model_edge_dialect_operators: List[str] = get_all_ops(model)
    #     with tempfile.NamedTemporaryFile(mode="w+") as yaml_stream:
    #         _ = gen_edge_yaml(model_edge_dialect_operators, yaml_stream)
    #         yaml_stream.seek(0, 0)
    #         self.assertTrue(
    #             filecmp.cmp(
    #                 yaml_stream.name,
    #                 "executorch/exir/dialects/edge/edge.yaml",
    #             ),
    #             "Please run `//executorch/exir/dialects/edge:yaml_generator -- --regenerate` to regenerate the file.",
    #         )

    def test_to_copy_sample_input_has_enough_coverage(self) -> None:
        """Make sure sample input to _to_copy(Tensor self, *, ScalarType dtype, ...) has enough coverage"""
        sample_input = get_sample_input(
            key="to", overload_name="", edge_type=torch.float32
        )
        dtype_set: Set[torch.dtype] = set()
        for _, kwargs in sample_input:
            self.assertTrue("dtype" in kwargs)
            dtype_set.add(kwargs["dtype"])

        self.assertTrue(dtype_set == regular_tensor_dtypes_to_str.keys())
