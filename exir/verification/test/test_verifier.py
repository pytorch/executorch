# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import contextmanager

import torch
from executorch.exir import EdgeCompileConfig, to_edge

from executorch.exir.dialects._ops import ops
from torch._export.verifier import SpecViolationError
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.export import export

from ..verifier import EXIREdgeDialectVerifier


class TestEdgeDialectVerifier(unittest.TestCase):
    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException("{} raised".format(exc_type.__name__))

    def test_edge_verifier_check_valid_op_succeed_given_custom_op(self) -> None:
        edge_op = ops.edge.quantized_decomposed.quantize_per_tensor.default
        verifier = EXIREdgeDialectVerifier(check_edge_ops=True)
        with self.assertNotRaises(SpecViolationError):
            verifier.check_valid_edge_op(edge_op)
            verifier.check_valid_op(edge_op)

    def test_edge_verifier_check_valid_dim_order_graph(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                t1 = x.to(dtype=torch.double, memory_format=torch.channels_last)
                t2 = t1 + t1
                return t1 * t2

        m = Model().eval()

        example_input = (
            torch.rand_like(
                torch.zeros([2, 2, 2, 2]),
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            ),
        )
        verifier = EXIREdgeDialectVerifier(check_edge_ops=True, dim_order=True)

        export_model = export(m, example_input)
        stride_edge_model = to_edge(
            export_model, compile_config=EdgeCompileConfig(_skip_dim_order=True)
        )
        dim_order_edge_model = to_edge(
            export_model, compile_config=EdgeCompileConfig(_skip_dim_order=False)
        )
        verifier(dim_order_edge_model.exported_program())
        with self.assertRaises(SpecViolationError):
            verifier(stride_edge_model.exported_program())
