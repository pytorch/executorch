# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from executorch.backends.fused_quant.compile import compile_to_fused_quant
from executorch.backends.fused_quant.decompose_fused_quant import DecomposeFusedQuant
from executorch.backends.fused_quant.frontend import trace
from executorch.backends.fused_quant.optimization_passes.to_channels_first import (
    ToChannelsFirst,
)
from executorch.backends.fused_quant.quantizer.defaults import (
    make_fused_quant_quantizer,
)
from executorch.exir import EdgeProgramManager, ExecutorchProgramManager
from executorch.exir.pass_manager import ExportedProgramPassManager
from torch import nn


class ConvReluLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(8 * 4 * 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        return self.linear(x.flatten(1))


def _calibration_inputs() -> list[tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    return [(torch.randn(2, 3, 4, 4),) for _ in range(4)]


def _decompose_for_portable_kernels(
    edge_program_manager: EdgeProgramManager,
) -> EdgeProgramManager:
    """The tail a backend runs after :func:`compile_to_fused_quant`.

    This is the no-backend case written out: nothing is colored, so every
    fused_quant op has to go back to quantize/dequantize plus ATen for the
    portable kernels to run it. A real backend colors first, lowers what it
    claimed, and runs this over the remainder.

    ``ToChannelsFirst`` comes first because no backend is consuming NHWC here, so
    the layout ``ToChannelsLast`` introduced has to be undone before decomposing.
    """
    return edge_program_manager.transform(
        ExportedProgramPassManager([ToChannelsFirst(), DecomposeFusedQuant()])
    )


def _op_targets(program: ExecutorchProgramManager) -> list[str]:
    return [
        str(node.target)
        for node in program.exported_program().graph_module.graph.nodes
        if node.op == "call_function"
    ]


class CompileTest(unittest.TestCase):
    def test_compile_to_fused_quant_produces_fused_quant_ops(self) -> None:
        """The pre-coloring seam hands back a graph in fused_quant form."""
        inputs = _calibration_inputs()

        edge_program_manager = compile_to_fused_quant(ConvReluLinear(), inputs)

        targets = [
            str(node.target)
            for node in edge_program_manager.exported_program().graph_module.graph.nodes
            if node.op == "call_function"
        ]
        self.assertTrue(
            any("fused_quant" in target for target in targets),
            f"expected fused_quant ops, got {targets}",
        )

    def test_passes_can_be_overridden(self) -> None:
        """An explicit pass list replaces the default pipeline."""
        inputs = _calibration_inputs()

        unoptimized = compile_to_fused_quant(ConvReluLinear(), inputs, passes=[])

        targets = [
            str(node.target)
            for node in unoptimized.exported_program().graph_module.graph.nodes
            if node.op == "call_function"
        ]
        # ToConvolution is part of the default pipeline and rewrites conv2d into
        # the generic convolution op, so its absence shows the default did not run.
        self.assertTrue(
            any("fused_quant.conv2d" in target for target in targets),
            f"expected the unoptimized conv2d form, got {targets}",
        )

    def test_decomposing_clears_every_fused_quant_op(self) -> None:
        """With nothing colored, the tail must leave no fused_quant op behind."""
        inputs = _calibration_inputs()

        edge = _decompose_for_portable_kernels(
            compile_to_fused_quant(ConvReluLinear(), inputs)
        )

        leftover = [
            str(node.target)
            for node in edge.exported_program().graph_module.graph.nodes
            if node.op == "call_function" and "fused_quant" in str(node.target)
        ]
        self.assertEqual(leftover, [])

    def test_decomposed_program_has_no_delegates(self) -> None:
        """Nothing is claimed, so nothing should be delegated."""
        inputs = _calibration_inputs()

        edge = _decompose_for_portable_kernels(
            compile_to_fused_quant(ConvReluLinear(), inputs)
        )
        program = edge.to_executorch()

        self.assertNotIn("executorch_call_delegate", _op_targets(program))

    def test_decomposed_program_matches_the_quantized_reference(self) -> None:
        """Compiled output tracks the eager float model within quantization error."""
        model = ConvReluLinear().eval()
        inputs = _calibration_inputs()

        edge = _decompose_for_portable_kernels(compile_to_fused_quant(model, inputs))

        (example,) = inputs[0]
        with torch.no_grad():
            expected = model(example)
            actual = edge.exported_program().module()(example)

        self.assertEqual(actual.shape, expected.shape)
        # int8 per-tensor activations over this range; compare against the
        # magnitude of the reference rather than an absolute epsilon.
        tolerance = 0.1 * expected.abs().max().item()
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)


class SoftmaxAttention(nn.Module):
    """SDPA, which traces to aten._safe_softmax."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class CoreOnlyAttentionTest(unittest.TestCase):
    """A quantizer that does not claim _masked_softmax on an attention model.

    `trace` holds `aten._safe_softmax` back from decomposition unconditionally,
    but the recompose into `_masked_softmax` only runs for a quantizer that
    claims it, so with the core set the op survives tracing. It is not core ATen
    and it is not in the verifier exception list, which would be a problem if it
    reached the verifier -- it does not, because `to_edge` decomposes it. This
    pins that down, since the exception list only covers ops re-emitted *after*
    `to_edge` by `DecomposeFusedQuant`.
    """

    def _inputs(self) -> list[tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)
        shape = (1, 2, 8, 16)
        return [
            (torch.randn(shape), torch.randn(shape), torch.randn(shape))
            for _ in range(2)
        ]

    def test_safe_softmax_survives_tracing_but_not_to_edge(self) -> None:
        quantizer = make_fused_quant_quantizer(core_only=True)
        self.assertNotIn(
            torch.ops.aten._masked_softmax.default, quantizer.preserved_ops()
        )

        program = trace(
            SoftmaxAttention().eval(),
            self._inputs()[0],
            ops_to_keep=quantizer.preserved_ops(),
        )
        traced = {n.target for n in program.graph.nodes if n.op == "call_function"}
        self.assertIn(torch.ops.aten._safe_softmax.default, traced)

        edge = compile_to_fused_quant(
            SoftmaxAttention().eval(), self._inputs(), quantizer
        )

        edge_ops = {
            str(n.target)
            for n in edge.exported_program().graph.nodes
            if n.op == "call_function"
        }
        self.assertFalse(
            any("_safe_softmax" in op for op in edge_ops),
            f"_safe_softmax reached the edge graph: {sorted(edge_ops)}",
        )
