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
from torch import nn
from torch._export.verifier import SpecViolationError
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.export import export
from torch.export.experimental import _export_forward_backward

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

    def test_edge_verifier_check_edge_op(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.transpose(0, 1)

        m = Model().eval()

        example_input = (torch.zeros([2, 2]),)

        export_model = export(m, example_input, strict=True)

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
                t2 = torch.empty(t1.size(), memory_format=torch.channels_last)
                t2.copy_(t1)
                return t2

        m = Model().eval()

        example_input = (
            torch.rand_like(
                torch.zeros([2, 2, 2, 2]),
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            ),
        )

        export_model = export(m, example_input, strict=True)

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

    def test_none_return_verifier(self) -> None:
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(6, 6, 5)
                self.linear = nn.Linear(6, 2)

            def forward(self, x):
                return self.linear(self.conv1(x).flatten(1))

        class TrainingNet(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
                self.loss = nn.CrossEntropyLoss()

            def forward(self, input, label):
                pred = self.net(input)
                return self.loss(pred, label)

        # conv returns (None, Tensor, Tensor) which is uncommon to see since
        # the schema is (Tensor, Tensor, Tensor). This is to test that
        # the verifier just ignores the None return value (since itll be
        # unused in the runtime).
        net = TrainingNet(Net())
        inputs = (torch.randn(1, 6, 5, 5), torch.ones(1, dtype=torch.int64))

        export_model = export(net, inputs, strict=True)
        export_model = _export_forward_backward(export_model)

        edge = to_edge(export_model)

        edge_verifier = EXIREdgeDialectVerifier()

        edge_verifier(edge.exported_program())

    def test_verifier_preserve_ops_view(self) -> None:
        class TestExpand(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.expand(2, 2, 2, 2)

        model = TestExpand()
        config = EdgeCompileConfig(preserve_ops=[torch.ops.aten.expand.default])
        export_model = export(model, (torch.randn(2, 2, 2, 2),), strict=True)
        with self.assertRaises(RuntimeError):
            to_edge(export_model, compile_config=config)
