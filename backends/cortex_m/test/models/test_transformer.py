# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.backends.arm._passes import (
    ConstantFoldingPass,
    DecomposeLayerNormPass,
    DecomposeSDPAWithRegularSoftmaxPass,
    FoldAndAnnotateQParamsPass,
)
from executorch.backends.cortex_m.passes.replace_scalar_with_tensor_pass import (
    CortexMReplaceScalarWithTensorArgPass,
)
from executorch.backends.cortex_m.test.tester import CortexMQuantize, CortexMTester
from executorch.backends.test.harness.stages import StageType
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.program._program import _transform
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class AttentionModel(torch.nn.Module):
    """Minimal SDPA model used to isolate attention-scale MUL lowering."""

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class TransformerModel(torch.nn.Module):
    """Small Transformer used to exercise decomposed attention lowering."""

    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Transformer(
            d_model=32,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=False,
        )

    def forward(self, src, tgt):
        return self.transformer(src, tgt)


class TransformerDecompositionQuantize(CortexMQuantize):
    """Apply the decomposition workflow before Cortex-M PT2E quantization."""

    def run(self, artifact, inputs):
        assert inputs is not None

        captured_graph = export(
            artifact,
            inputs,
            strict=True,
        ).module()

        captured_graph = ConstantFoldingPass().call(captured_graph).graph_module
        captured_graph = (
            DecomposeSDPAWithRegularSoftmaxPass().call(captured_graph).graph_module
        )
        captured_graph = DecomposeLayerNormPass().call(captured_graph).graph_module

        prepared = prepare_pt2e(
            captured_graph,
            self.quantizer,
        )

        with torch.no_grad():
            prepared(*inputs)

        self.converted_graph = convert_pt2e(
            prepared,
            fold_quantize=self.fold_quantize,
        )


def test_quantized_bmm_retrace_preserves_output_dtype(cortex_m_target):
    """Retracing folded quantized BMM must preserve its requantized dtype."""

    torch.manual_seed(0)

    model = AttentionModel().eval()
    inputs = (
        torch.randn(1, 2, 4, 16),
        torch.randn(1, 2, 4, 16),
        torch.randn(1, 2, 4, 16),
    )

    tester = CortexMTester(
        model,
        inputs,
        target_config=cortex_m_target,
    )

    tester.quantize(TransformerDecompositionQuantize())
    tester.export()
    tester.to_edge()

    exported_program = tester.get_artifact(StageType.TO_EDGE).exported_program()

    exported_program = _transform(
        exported_program,
        RemoveGetItemPass(),
    )
    exported_program = _transform(
        exported_program,
        FoldAndAnnotateQParamsPass(exported_program),
    )
    exported_program = _transform(
        exported_program,
        CortexMReplaceScalarWithTensorArgPass(),
    )

    bmm_nodes = [
        node
        for node in exported_program.graph_module.graph.nodes
        if node.target == exir_ops.edge.aten.bmm.default
    ]

    assert len(bmm_nodes) == 2

    for node in bmm_nodes:
        output_qparams = node.meta["output_qparams"]
        assert node.meta["val"].dtype == output_qparams[0].dtype


def test_decomposed_attention_scale_mul_lowering(cortex_m_target):
    """The two SDPA scaling MULs should lower to Cortex-M quantized MUL."""

    torch.manual_seed(0)

    model = AttentionModel().eval()
    inputs = (
        torch.randn(1, 2, 4, 16),
        torch.randn(1, 2, 4, 16),
        torch.randn(1, 2, 4, 16),
    )

    tester = CortexMTester(
        model,
        inputs,
        target_config=cortex_m_target,
    )

    tester.quantize(TransformerDecompositionQuantize())
    tester.export()
    tester.to_edge()
    tester.run_passes()

    tester.check_count(
        {
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 2,
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 2,
            "executorch_exir_dialects_edge__ops_cortex_m_softmax_default": 1,
        }
    )

    tester.check_not(
        [
            "executorch_exir_dialects_edge__ops_aten_mul_Tensor",
        ]
    )

    tester.run_method_and_compare_outputs(
        inputs=inputs,
        qtol=2,
    )


def test_transformer_decomposed_attention_lowering(cortex_m_target):
    """Decomposed attention ops should lower to Cortex-M quantized kernels."""

    torch.manual_seed(0)

    model = TransformerModel().eval()
    inputs = (
        torch.randn(8, 1, 32),
        torch.randn(8, 1, 32),
    )

    tester = CortexMTester(
        model,
        inputs,
        target_config=cortex_m_target,
    )

    tester.quantize(TransformerDecompositionQuantize())
    tester.export()
    tester.to_edge()
    tester.run_passes()

    # Decomposition details may change with PyTorch, so require Cortex-M
    # lowering without depending on exact generated-op counts.
    tester.check(
        [
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default",
            "executorch_exir_dialects_edge__ops_cortex_m_softmax_default",
            "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default",
        ]
    )

    tester.check_not(
        [
            "executorch_exir_dialects_edge__ops_aten_bmm_default",
            "executorch_exir_dialects_edge__ops_aten__softmax_default",
            "executorch_exir_dialects_edge__ops_aten__safe_softmax_default",
        ]
    )

    tester.run_method_and_compare_outputs(
        inputs=inputs,
        qtol=2,
    )
