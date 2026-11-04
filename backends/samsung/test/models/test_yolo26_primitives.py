# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig


class Primitive(torch.nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, *inputs):
        return self.operator(*inputs)


class StrideMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("strides", torch.ones(1, 2100))

    def forward(self, x):
        return x * self.strides


class TestYolo26Primitives(unittest.TestCase):
    def test_compile_primitives(self):
        cases = (
            ("attention_softmax", lambda x: torch.softmax(x, -1), ((1, 2, 100, 100),)),
            ("query_key_bmm", torch.bmm, ((2, 100, 32), (2, 32, 100))),
            ("value_attention_bmm", torch.bmm, ((2, 64, 100), (2, 100, 100))),
            ("scores_amax", lambda x: torch.amax(x, dim=-1), ((1, 2100, 80),)),
            (
                "indices_topk",
                lambda x: torch.topk(x, 300, dim=-1).indices,
                ((1, 2100),),
            ),
            (
                "values_topk",
                lambda x: torch.topk(x, 300, dim=-1).values,
                ((1, 2100),),
            ),
            ("both_topk", lambda x: tuple(torch.topk(x, 300)), ((1, 2100),)),
            ("stride_mul", StrideMul(), ((1, 4, 2100),)),
        )
        for name, operator, shapes in cases:
            with self.subTest(operator=name):
                inputs = tuple(torch.randn(shape) for shape in shapes)
                tester = SamsungTester(
                    Primitive(operator),
                    inputs,
                    [gen_samsung_backend_compile_spec(TestConfig.chipset)],
                )
                tester.export().to_edge_transform_and_lower()
                if name.endswith("topk"):
                    tester.check_count(
                        {"executorch_exir_dialects_edge__ops_aten_topk_default": 1}
                    ).check_not(["torch.ops.higher_order.executorch_call_delegate"])
                else:
                    tester.check(["torch.ops.higher_order.executorch_call_delegate"])
                tester.to_executorch()


if __name__ == "__main__":
    unittest.main()
