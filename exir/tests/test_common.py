# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
import unittest

import torch
import torch.fx

from executorch.exir.common import extract_out_arguments, get_schema_for_operators
from executorch.exir.print_program import add_cursor_to_graph


class TestExirCommon(unittest.TestCase):
    def test_get_schema_for_operators(self) -> None:
        op_list = [
            "torch.ops._caffe2.RoIAlign.default",
            "torch.ops.aten.add.Tensor",
            "torch.ops.aten.batch_norm.default",
            "torch.ops.aten.cat.default",
            "torch.ops.aten.clamp.default",
        ]

        schemas = get_schema_for_operators(op_list)
        pat = re.compile(r"[^\(]+\([^\)]+\) -> ")
        for _op_name, schema in schemas.items():
            self.assertIsNotNone(re.match(pat, schema))

    def test_get_out_args(self) -> None:
        schema1 = torch._C.parse_schema(
            "aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)"
        )
        schema2 = torch._C.parse_schema(
            "split_copy.Tensor_out(Tensor self, int split_size, int dim=0, *, Tensor(a!)[] out) -> ()"
        )

        out_args_1 = extract_out_arguments(schema1, {"out": torch.ones(5)})
        out_args_2 = extract_out_arguments(
            schema2, {"out": [torch.ones(5), torch.ones(5)]}
        )

        out_arg_name_1, _ = out_args_1
        self.assertEqual(out_arg_name_1, "out")

        out_arg_name_2, _ = out_args_2
        self.assertEqual(out_arg_name_2, "out")

    def test_add_cursor(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        module = MyModule()

        from torch.fx import symbolic_trace

        symbolic_traced = symbolic_trace(module)

        # Graph we are testing:
        # graph():
        #   %x : [#users=1] = placeholder[target=x]
        #   %param : [#users=1] = get_attr[target=param]
        #   %add : [#users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
        # --> %linear : [#users=1] = call_module[target=linear](args = (%add,), kwargs = {})
        #   %clamp : [#users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
        #   return clamp

        actual_str = add_cursor_to_graph(
            symbolic_traced.graph, list(symbolic_traced.graph.nodes)[3]
        )
        self.assertTrue(actual_str.split("\n")[4].startswith("-->"))
