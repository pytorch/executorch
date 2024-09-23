# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from contextlib import contextmanager
from typing import Any

import torch
from executorch.exir import EdgeCompileConfig, to_edge

from executorch.exir.dialects._ops import ops
from torch._export.verifier import SpecViolationError
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.export import export

from ..verifier import EXIREdgeDialectVerifier


class TestEdgeDialectVerifier(unittest.TestCase):
    @contextmanager
    def assertNotRaises(self, exc_type: Any) -> Any:
        try:
            yield None
        except exc_type:
            raise self.failureException("{} raised".format(exc_type.__name__))

    def test_edge_verifier_check_valid_op_succeed_given_custom_op(self) -> None:
        edge_op = ops.edge.quantized_decomposed.quantize_per_tensor.default
        verifier = EXIREdgeDialectVerifier()
        with self.assertNotRaises(SpecViolationError):
            verifier.check_valid_edge_op(edge_op)
            verifier.check_valid_op(edge_op)

    def test_edge_verifier_enablement(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x, y):
                z = y.item()
                torch._check(z > 0)
                torch._check(z < 4)
                return x[z : z + y.shape[0]]

        ep = torch.export.export(M(), (torch.randn(10), torch.tensor([3])))

        compile_config_with_disable_ir_validity = EdgeCompileConfig(
            _check_ir_validity=False
        )
        edge_manager = to_edge(
            ep, compile_config=compile_config_with_disable_ir_validity
        )

        normal_verifier = EXIREdgeDialectVerifier()
        disable_ir_validity_verifier = EXIREdgeDialectVerifier(
            compile_config_with_disable_ir_validity
        )

        # exported model can not pass normal verifier due to
        # aten.sym_constrain_range.default is illegal to be edge op
        with self.assertRaises(SpecViolationError):
            normal_verifier(edge_manager.exported_program())

        # exported model can pass disable_ir_validity_verifier due to verifier
        # is disabled by compile_config_with_disable_ir_validity
        # (_check_ir_validity=False). Noted that this verifation has been done
        # when calling `to_edge`. Explicitly calling verifier here just for better
        # demonstration and is unnecessary in real world for ir verification.
        disable_ir_validity_verifier(edge_manager.exported_program())

    def test_edge_verifier_check_edge_op(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.transpose(0, 1)

        m = Model().eval()

        example_input = (torch.zeros([2, 2]),)

        export_model = export(m, example_input)

        compile_config_without_edge_op = EdgeCompileConfig(
            _use_edge_ops=False, _skip_dim_order=False
        )

        edge_manager = to_edge(
            export_model, compile_config=compile_config_without_edge_op
        )

        normal_verifier = EXIREdgeDialectVerifier()
        disable_edge_op_check_verifier = EXIREdgeDialectVerifier(
            compile_config_without_edge_op
        )

        # exported model can not pass normal verifier due to
        # incontiguous memory layout tensor is not supported in ET
        with self.assertRaises(SpecViolationError):
            normal_verifier(edge_manager.exported_program())

        # exported model can pass disable_edge_op_check_verifier due to the
        # incontiguous memory layout tensor verification is disabled by
        # compile_config_without_edge_op (_use_edge_ops=False). Noted that this
        # verifation has been done when calling `to_edge`. Explicitly calling
        # verifier here just for better demonstration and is unnecessary
        # in real world for ir verification.
        disable_edge_op_check_verifier(edge_manager.exported_program())

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

        export_model = export(m, example_input)

        compile_config_with_dim_order = EdgeCompileConfig(_skip_dim_order=False)
        compile_config_with_stride = EdgeCompileConfig(_skip_dim_order=True)

        dim_order_edge_model = to_edge(
            export_model, compile_config=compile_config_with_dim_order
        )
        stride_edge_model = to_edge(
            export_model, compile_config=compile_config_with_stride
        )

        dim_order_verifier = EXIREdgeDialectVerifier(
            edge_compile_config=compile_config_with_dim_order
        )
        stride_verifier = EXIREdgeDialectVerifier(
            edge_compile_config=compile_config_with_stride
        )

        dim_order_verifier(dim_order_edge_model.exported_program())
        stride_verifier(stride_edge_model.exported_program())

        with self.assertRaises(SpecViolationError):
            dim_order_verifier(stride_edge_model.exported_program())
        with self.assertRaises(SpecViolationError):
            stride_verifier(dim_order_edge_model.exported_program())
