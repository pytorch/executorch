# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Tuple

import torch

from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.test import common
from executorch.backends.arm.test.tester.test_pipeline import (
    EthosU55PipelineINT,
    EthosU85PipelineINT,
    TosaPipelineFP,
    TosaPipelineINT,
    VgfPipeline,
)
from torch.export import export


class SDPA(torch.nn.Module):
    def __init__(self, attn_mask=None, is_causal=False):
        super().__init__()
        self.attn_mask = attn_mask
        self.is_causal = is_causal

    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=self.attn_mask, is_causal=self.is_causal
        )


input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
test_case_t = Callable[[], Tuple[SDPA, input_t]]

test_suite = {
    # test_name: generator(model, inputs)
    "randn_no_mask_non_causal": lambda: (
        SDPA(attn_mask=None, is_causal=False),
        tuple(torch.randn(1, 3, 197, 64) for _ in range(3)),
    ),
    "randn_no_mask_causal": lambda: (
        SDPA(attn_mask=None, is_causal=True),
        tuple(torch.randn(1, 3, 197, 64) for _ in range(3)),
    ),
    "randn_with_bool_mask_non_causal": lambda: (
        SDPA(attn_mask=(torch.rand(1, 3, 197, 1) > 0.5), is_causal=False),
        tuple(torch.randn(1, 3, 197, 64) for _ in range(3)),
    ),
    "randn_with_additive_mask_non_causal": lambda: (
        SDPA(
            attn_mask=torch.where(torch.rand(1, 3, 197, 1) > 0.5, 0.0, -float("inf")),
            is_causal=False,
        ),
        tuple(torch.randn(1, 3, 197, 64) for _ in range(3)),
    ),
    # causal with mask is not supported in PyTorch (https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
}


@common.parametrize("test_case", test_suite)
def test_sdpa_tosa_FP(test_case: test_case_t):
    model, test_input = test_case()
    pipeline = TosaPipelineFP[input_t](model, test_input, [], [])
    pipeline.pop_stage("check_count.exir")
    pipeline.run()


@common.parametrize("test_case", test_suite)
def test_sdpa_tosa_INT(test_case: test_case_t):
    model, test_input = test_case()
    pipeline = TosaPipelineINT[input_t](
        model, test_input, [], [], frobenius_threshold=None, cosine_threshold=None
    )
    pipeline.pop_stage("check.quant_nodes")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage(
        "run_method_and_compare_outputs"
    )  # TODO: reference is not quantized
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_case", test_suite)
def test_sdpa_vgf_no_quant(test_case: test_case_t):
    model, test_input = test_case()
    pipeline = VgfPipeline[input_t](
        model,
        test_input,
        [],
        [],
        quantize=False,
    )
    pipeline.run()


@common.SkipIfNoModelConverter
@common.parametrize("test_case", test_suite)
def test_sdpa_vgf_quant(test_case: test_case_t):
    model, test_input = test_case()
    pipeline = VgfPipeline[input_t](
        model, test_input, [], [], quantize=True, run_on_vulkan_runtime=False
    )
    pipeline.run()


@common.parametrize("test_case", test_suite)
def test_sdpa_u55_INT(test_case: test_case_t):
    model, test_input = test_case()
    pipeline = EthosU55PipelineINT[input_t](model, test_input, [], [])
    pipeline.pop_stage("check.quant_nodes")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()


@common.parametrize("test_case", test_suite)
def test_sdpa_u55_INT_annotation_pipeline_decomposes_safe_softmax(
    test_case: test_case_t,
):
    """Verify the U55 annotation pipeline decomposes SDPA _safe_softmax.

    U55 now matches U85 and VGF: the annotation pipeline lowers
    _safe_softmax to the stable softmax primitive sequence instead of leaving
    it in the graph for partitioning.

    """
    model, test_input = test_case()
    exported_program = export(model, test_input)
    graph_module = ArmPassManager(
        EthosUCompileSpec("ethos-u55-128")
    ).transform_for_annotation_pipeline(exported_program.graph_module)

    softmax_targets = {
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function" and "softmax" in str(node.target)
    }

    assert "aten._safe_softmax.default" not in softmax_targets


@common.parametrize("test_case", test_suite)
@common.XfailIfNoCorstone320
def test_sdpa_u85_INT(test_case: test_case_t):
    model, test_input = test_case()
    pipeline = EthosU85PipelineINT[input_t](model, test_input, [], [])
    pipeline.pop_stage("check.quant_nodes")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
