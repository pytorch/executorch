#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import List, Optional

import torch

from executorch.exir.dialects._ops import ops
from executorch.exir.dialects.edge._ops import (
    _edge_dialect_info,
    AllowedDtypeSet,
    EdgeOpOverload,
    FunctionDtypeConstraint,
)
from torch._ops import OpOverload
from torch.library import impl, Library

lib = Library("test_op", "DEF")

# Fake a operator for testing.
# This operator takes two tensors as input and returns the first one.
lib.define("foo(Tensor self, Tensor other) -> Tensor")


@impl(lib, "foo", "CPU")
def foo(a, b):
    # do nothing and return a.
    return a


def foo_dtype_constraint():
    # Update the type constraint for function foo.
    _edge_dialect_info["test_op::foo"] = {
        "func": "foo",
        "namespace": "edge",
        "inherits": "test_op::foo",
        "type_alias": {
            "T0": [
                "Float",
                "Double",
            ],
            "T1": [
                "Char",
            ],
            "T2": [
                "Int",
            ],
        },
        "type_constraint": [
            {
                "self": "T0",
                "other": "T0",
                "__ret_0": "T0",
            },
            {
                "self": "T1",
                "other": "T1",
                "__ret_0": "T2",
            },
        ],
    }


# Fake a operator not been included by edge.yaml for testing.
# This operator takes three tensors as input and returns the second one.
lib.define(
    "yaml_unincluded(Tensor self, Tensor[] other_list, Tensor? other_optional) -> Tensor[]"
)


@impl(lib, "yaml_unincluded", "CPU")
def yaml_unincluded(
    a: torch.Tensor, b: List[torch.Tensor], c: Optional[torch.Tensor]
) -> List[torch.Tensor]:
    # do nothing and return b.
    return b


class TestEdgeOps(unittest.TestCase):
    def setUp(self) -> None:
        self.aten_add: OpOverload = torch.ops.aten.add.Tensor
        self.edge_add: EdgeOpOverload = ops.edge.aten.add.Tensor

        foo_dtype_constraint()
        self.edge_foo: EdgeOpOverload = ops.edge.test_op.foo.default

    def test_callable_gives_same_result(self) -> None:
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) * 2
        c = torch.ones(2, 3) * 3
        self.assertTrue(torch.allclose(c, self.edge_add(a, b)))
        self.assertTrue(torch.allclose(self.edge_add(a, b), self.aten_add(a, b)))

    def test_schema_name_same_as_aten_op(self) -> None:
        self.assertEqual(self.aten_add._schema.name, self.edge_add._schema.name)

    def test_edge_argument_dtype_constraints(self) -> None:
        edge_log_softmax: OpOverload = ops.edge.aten._log_softmax.default
        arguments = edge_log_softmax._schema.arguments
        returns = edge_log_softmax._schema.returns
        for arg in arguments:
            if isinstance(arg.type, torch.TensorType):
                self.assertTrue(isinstance(arg.allowed_types, set))
                self.assertEqual(
                    arg.allowed_types, {torch.float16, torch.float32, torch.float64}
                )

        for ret in returns:
            if isinstance(ret.type, torch.TensorType):
                self.assertTrue(isinstance(ret.allowed_types, set))
                self.assertEqual(
                    ret.allowed_types, {torch.float16, torch.float32, torch.float64}
                )

    def test_allowed_dtype_set(self) -> None:
        allowed_dtype_set = AllowedDtypeSet({torch.int8, torch.int32})
        self.assertTrue(torch.int8 in allowed_dtype_set)
        self.assertTrue(torch.int32 in allowed_dtype_set)

        # torch.int16 is not a legal dtype for allowed_dtype_set
        self.assertFalse(allowed_dtype_set.reduce_to(torch.int16))
        self.assertTrue(allowed_dtype_set.reduce_to(torch.int32))

        # now allowed_dtype_set is reduced to torch.int32
        self.assertFalse(torch.int8 in allowed_dtype_set)
        self.assertTrue(torch.int32 in allowed_dtype_set)

        # clear it to make it back
        allowed_dtype_set.clear()
        self.assertTrue(torch.int8 in allowed_dtype_set)
        self.assertTrue(torch.int32 in allowed_dtype_set)

    def test_edge_add_dtype_constraints_content(self) -> None:
        edge_foo_schema = self.edge_foo._schema
        self.assertTrue(
            isinstance(edge_foo_schema.dtype_constraint, FunctionDtypeConstraint)
        )
        self.assertTrue(isinstance(edge_foo_schema.dtype_constraint.type_alias, dict))
        for key, value in edge_foo_schema.dtype_constraint.type_alias.items():
            self.assertTrue(key in ["T0", "T1", "T2"])
            self.assertTrue(isinstance(value, AllowedDtypeSet))
            if key == "T0":
                self.assertEqual(value.types, {torch.float32, torch.float64})
            elif key == "T1":
                self.assertEqual(
                    value.types,
                    {
                        torch.int8,
                    },
                )
            elif key == "T2":
                self.assertEqual(
                    value.types,
                    {
                        torch.int32,
                    },
                )

        self.assertEqual(
            edge_foo_schema.dtype_constraint.type_constraint,
            [
                {
                    "self": "T0",
                    "other": "T0",
                    "__ret_0": "T0",
                },
                {
                    "self": "T1",
                    "other": "T1",
                    "__ret_0": "T2",
                },
            ],
        )

    def test_edge_op_dtype_constraints_validation_function(self) -> None:
        edge_foo_schema = self.edge_foo._schema
        self.assertTrue(
            edge_foo_schema.dtype_constraint.validate(
                {
                    "self": torch.float32,
                    "other": torch.float32,
                    "__ret_0": torch.float32,
                }
            )
        )
        self.assertFalse(
            edge_foo_schema.dtype_constraint.validate(
                {
                    "self": torch.float32,
                    "other": torch.float32,
                    "__ret_0": torch.float64,
                }
            )
        )
        self.assertFalse(
            edge_foo_schema.dtype_constraint.validate(
                {
                    "self": torch.float32,
                    "other": torch.float32,
                }
            )
        )
        self.assertFalse(
            edge_foo_schema.dtype_constraint.validate(
                {
                    "other": torch.float32,
                    "__ret_0": torch.float32,
                }
            )
        )
        self.assertTrue(
            edge_foo_schema.dtype_constraint.validate(
                {"self": torch.int8, "other": torch.int8, "__ret_0": torch.int32}
            )
        )

        self.assertFalse(
            edge_foo_schema.dtype_constraint.validate(
                {"self": torch.int8, "other": torch.int8, "__ret": torch.int32}
            )
        )

        self.assertFalse(
            edge_foo_schema.dtype_constraint.validate(
                {"self": torch.int8, "other": torch.int8, "__ret_0": torch.int8}
            )
        )

    def test_edge_op_dtype_constraints_validation_function_with_optional_tensor_input(
        self,
    ) -> None:
        edge_native_layer_norm = ops.edge.aten.native_layer_norm.default
        edge_native_layer_norm_schema = edge_native_layer_norm._schema
        # In native layer norm, there have six tensor inputs and outputs, but weight
        # and bias are all optional. Therefore, the dtype validator should return True
        # if user does not provide the corresponding argument, or provide optional
        # argument in correct dtype.

        self.assertTrue(
            edge_native_layer_norm_schema.dtype_constraint.validate(
                {
                    "input": torch.float32,
                    "__ret_0": torch.float32,
                    "__ret_1": torch.float32,
                    "__ret_2": torch.float32,
                }
            )
        )

        self.assertTrue(
            edge_native_layer_norm_schema.dtype_constraint.validate(
                {
                    "input": torch.float32,
                    "weight": torch.float32,
                    "__ret_0": torch.float32,
                    "__ret_1": torch.float32,
                    "__ret_2": torch.float32,
                }
            )
        )

        self.assertTrue(
            edge_native_layer_norm_schema.dtype_constraint.validate(
                {
                    "input": torch.float32,
                    "bias": torch.float32,
                    "__ret_0": torch.float32,
                    "__ret_1": torch.float32,
                    "__ret_2": torch.float32,
                }
            )
        )

        self.assertTrue(
            edge_native_layer_norm_schema.dtype_constraint.validate(
                {
                    "input": torch.float32,
                    "weight": torch.float32,
                    "bias": torch.float32,
                    "__ret_0": torch.float32,
                    "__ret_1": torch.float32,
                    "__ret_2": torch.float32,
                }
            )
        )

        self.assertFalse(
            edge_native_layer_norm_schema.dtype_constraint.validate(
                {
                    "input": torch.float32,
                    "weight": torch.float32,
                    "bias": torch.int32,
                    "__ret_0": torch.float32,
                    "__ret_1": torch.float32,
                    "__ret_2": torch.float32,
                }
            )
        )

        # Any other tensor input/output should be essential input/output.
        # The dtype validator should return False if user does not forward all essential inputs.
        self.assertFalse(
            edge_native_layer_norm_schema.dtype_constraint.validate(
                {
                    "weight": torch.float32,
                    "bias": torch.float32,
                    "__ret_0": torch.float32,
                    "__ret_1": torch.float32,
                    "__ret_2": torch.float32,
                }
            )
        )

        self.assertFalse(
            edge_native_layer_norm_schema.dtype_constraint.validate(
                {
                    "weight": torch.float32,
                    "bias": torch.float32,
                    "__ret_0": torch.float32,
                    "__ret_1": torch.float32,
                }
            )
        )

    def test_edge_op_dtype_constraints_validation_function_with_tensor_list_input(
        self,
    ) -> None:
        edge_cat = ops.edge.aten.cat.default
        edge_cat_schema = edge_cat._schema
        # The input of cat, `tensors`, is a tensor list.
        # Test if edge dialect can validate the correctness of tensor list type.

        self.assertTrue(
            edge_cat_schema.dtype_constraint.validate(
                {"tensors": torch.float32, "__ret_0": torch.float32}
            )
        )
        self.assertTrue(
            edge_cat_schema.dtype_constraint.validate(
                {"tensors": torch.half, "__ret_0": torch.half}
            )
        )
        self.assertFalse(
            edge_cat_schema.dtype_constraint.validate(
                {"tensors": torch.half, "__ret_0": torch.float}
            )
        )
        self.assertFalse(
            edge_cat_schema.dtype_constraint.validate(
                {"tensors": torch.half, "non-sense": torch.half}
            )
        )

    def test_op_not_included_by_yaml(self) -> None:
        # We should support operator not listed in edge.yaml
        # For such function, any given dtype combinations will be legal as long as:
        # a. each dtype is supported by executorch
        # b. all essential tensor-like inputs are provided
        # c. provided inputs rather than essential tensor-like inputs are optional tensor-like inputs.
        edge_op_test = ops.edge.test_op.yaml_unincluded.default
        edge_op_test_schema = edge_op_test._schema
        self.assertTrue(
            edge_op_test_schema.dtype_constraint.validate(
                {
                    "self": torch.float32,
                    "other_list": torch.float32,
                    "other_optional": torch.float32,
                    "__ret_0": torch.float32,
                }
            )
        )
        self.assertTrue(
            edge_op_test_schema.dtype_constraint.validate(
                {
                    "self": torch.float32,
                    "other_list": torch.int32,
                    "other_optional": torch.int8,
                    "__ret_0": torch.int32,
                }
            )
        )
        self.assertTrue(
            edge_op_test_schema.dtype_constraint.validate(
                {
                    "self": torch.float32,
                    "other_list": torch.int32,
                    "other_optional": torch.int8,
                    "__ret_0": torch.bool,
                }
            )
        )
        self.assertTrue(
            edge_op_test_schema.dtype_constraint.validate(
                {
                    "self": torch.float32,
                    "other_list": torch.float32,
                    "__ret_0": torch.float32,
                }
            )
        )
        self.assertFalse(
            edge_op_test_schema.dtype_constraint.validate(
                {
                    "self": torch.float32,
                    "other_optional": torch.float32,
                    "__ret_0": torch.float32,
                }
            )
        )

    def test_to_out_variant_returns_correct_op(self) -> None:
        out = self.edge_add.to_out_variant()
        self.assertEqual(out, torch.ops.aten.add.out)

    def test_to_out_variant_raises_exception_when_no_out_variant(self) -> None:
        view_op = ops.edge.aten.view.default
        with self.assertRaisesRegex(
            RuntimeError,
            "SchemaKind.out variant of operator aten::view can't be found.",
        ):
            view_op.to_out_variant()

    def test_get_new_registered_out_var(
        self,
    ) -> None:
        library = Library("TEST_ONLY", "DEF")
        library.define("foo.Tensor(Tensor a, Tensor b) -> Tensor")
        op = ops.edge.TEST_ONLY.foo.Tensor

        self.assertRaises(RuntimeError, op.to_out_variant)
        library.define(
            "foo.Tensor_out(Tensor a, Tensor b, *, Tensor(a!) out) -> Tensor(a!)"
        )
        out = op.to_out_variant()
        self.assertEqual(out, torch.ops.TEST_ONLY.foo.Tensor_out)
