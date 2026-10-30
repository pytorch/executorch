# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from executorch.backends.arm._passes import DecomposeGatedDeltaRulePass
from executorch.backends.arm.constants import DISALLOW_TFA_META_KEY
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

aten_op_fused = "torch.ops.llama.gated_delta_rule.default"
aten_op_bmm = "torch.ops.aten.bmm.default"


class GatedDeltaRule(torch.nn.Module):
    def forward(self, query, key, value, decay, beta, state):
        return torch.ops.llama.gated_delta_rule.default(
            query, key, value, decay, beta, state
        )


def _inputs(seq_len, channelwise=False):
    torch.manual_seed(0)
    decay_shape = (1, 4, seq_len, 4) if channelwise else (1, 4, seq_len)
    return (
        torch.randn(1, 4, seq_len, 4) * 0.2,
        torch.randn(1, 4, seq_len, 4) * 0.2,
        torch.randn(1, 4, seq_len, 4) * 0.2,
        torch.rand(decay_shape),
        torch.rand(1, 4, seq_len),
        torch.randn(1, 4, 4, 4) * 0.2,
    )


@pytest.mark.parametrize("seq_len", [1, 3])
@pytest.mark.parametrize("channelwise", [False, True], ids=["scalar", "channelwise"])
def test_gated_delta_rule_pass_matches_fused_op(seq_len, channelwise):
    inputs = _inputs(seq_len, channelwise)
    graph = torch.export.export(GatedDeltaRule(), inputs).module()
    result = DecomposeGatedDeltaRulePass()(graph)

    assert result.modified
    assert all(
        "gated_delta_rule" not in str(node.target)
        for node in result.graph_module.graph.nodes
    )
    for actual, expected in zip(
        result.graph_module(*inputs), GatedDeltaRule()(*inputs)
    ):
        torch.testing.assert_close(actual, expected)


def test_gated_delta_rule_pass_keeps_grouped_heads_portable():
    inputs = list(_inputs(1))
    inputs[0] = torch.randn(1, 8, 1, 4)
    graph = torch.export.export(GatedDeltaRule(), tuple(inputs)).module()

    assert not DecomposeGatedDeltaRulePass()(graph).modified


@pytest.mark.parametrize(
    "global_enabled, selective_config, disallow_tfa",
    [
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
    ids=["global", "no_quantization", "selective_config", "explicit_disallow"],
)
def test_gated_delta_rule_composable_tfa_decomposition(
    global_enabled, selective_config, disallow_tfa
):
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    config = get_symmetric_quantization_config()
    quantizer.set_global(config if global_enabled else None)
    if selective_config:
        quantizer.set_node_target(torch.ops.llama.gated_delta_rule.default, None)

    graph = torch.export.export(GatedDeltaRule(), _inputs(1)).module()
    if disallow_tfa:
        node = next(
            node
            for node in graph.graph.nodes
            if node.target == torch.ops.llama.gated_delta_rule.default
        )
        node.meta[DISALLOW_TFA_META_KEY] = True
    graph = quantizer.transform_for_annotation(graph)
    fused = [
        node
        for node in graph.graph.nodes
        if node.target == torch.ops.llama.gated_delta_rule.default
    ]
    bmm = [
        node for node in graph.graph.nodes if node.target == torch.ops.aten.bmm.default
    ]
    assert not fused
    assert len(bmm) == 2

    if global_enabled and not selective_config and not disallow_tfa:
        graph = quantizer.annotate(graph)
        assert all(node.meta[Q_ANNOTATION_KEY].output_qspec is not None for node in bmm)


@pytest.mark.parametrize("seq_len", [1, 3])
def test_gated_delta_rule_tosa_FP(seq_len):
    pipeline = TosaPipelineFP(
        GatedDeltaRule(),
        _inputs(seq_len),
        aten_op_fused,
        transform_passes=[DecomposeGatedDeltaRulePass()],
    )
    pipeline.run()


@pytest.mark.parametrize("seq_len", [1, 3])
def test_gated_delta_rule_tosa_INT(seq_len):
    pipeline = TosaPipelineINT(GatedDeltaRule(), _inputs(seq_len), aten_op_bmm)
    pipeline.add_stage_after("export", pipeline.tester.check_not, [aten_op_fused])
    pipeline.run()


@pytest.mark.parametrize("seq_len", [1, 3])
@common.XfailIfNoCorstone320
def test_gated_delta_rule_u85_INT_16a8w(seq_len):
    pipeline = EthosU85PipelineINT(
        GatedDeltaRule(),
        _inputs(seq_len),
        aten_op_bmm,
        run_on_fvp=True,
        a16w8_quantization=True,
    )
    pipeline.add_stage_after(
        "export",
        pipeline.tester.check_not,
        [aten_op_fused],
    )
    pipeline.run()


@pytest.mark.parametrize("seq_len", [1, 3])
@common.SkipIfNoModelConverter
def test_gated_delta_rule_vgf_quant(seq_len):
    pipeline = VgfPipeline(
        GatedDeltaRule(),
        _inputs(seq_len),
        aten_op_bmm,
        quantize=True,
    )
    pipeline.add_stage_after(
        "export",
        pipeline.tester.check_not,
        [aten_op_fused],
    )
    pipeline.run()


@pytest.mark.parametrize("seq_len", [1, 3])
@common.SkipIfNoModelConverter
def test_gated_delta_rule_vgf_no_quant(seq_len):
    pipeline = VgfPipeline(
        GatedDeltaRule(),
        _inputs(seq_len),
        aten_op_fused,
        quantize=False,
        transform_passes=[DecomposeGatedDeltaRulePass()],
    )
    pipeline.add_stage_after(
        "to_edge_transform_and_lower",
        pipeline.tester.check_not,
        ["executorch_exir_dialects_edge__ops_llama_gated_delta_rule_default"],
    )
    pipeline.run()
