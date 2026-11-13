# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import Counter
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from executorch.backends.arm._passes.arm_pass_manager import ArmPassManager
from executorch.backends.arm._passes.decompose_sdpa_pass import (
    DecomposeScaledDotProductAttentionPass,
)
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


@dataclass(frozen=True)
class SdpaDecodeConfig:
    batch_size: int = 1
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    query_len: int = 1
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class SdpaCase:
    name: str
    module: torch.nn.Module
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    config: SdpaDecodeConfig
    kv_len: int


class DecodeSdpaModule(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True,
        )


def build_decode_case(
    kv_len: int,
    config: SdpaDecodeConfig,
    seed: int = 0,
) -> SdpaCase:
    if config.query_heads % config.kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads for GQA")

    generator = torch.Generator().manual_seed(seed + kv_len)
    shape_q = (
        config.batch_size,
        config.query_heads,
        config.query_len,
        config.head_dim,
    )
    shape_kv = (
        config.batch_size,
        config.kv_heads,
        kv_len,
        config.head_dim,
    )
    q = torch.randn(shape_q, generator=generator, dtype=config.dtype)
    k = torch.randn(shape_kv, generator=generator, dtype=config.dtype)
    v = torch.randn(shape_kv, generator=generator, dtype=config.dtype)
    return SdpaCase(
        name=(
            f"b{config.batch_size}_hq{config.query_heads}_hkv{config.kv_heads}"
            f"_d{config.head_dim}_q{config.query_len}_kv{kv_len}"
        ),
        module=DecodeSdpaModule().eval(),
        inputs=(q, k, v),
        config=config,
        kv_len=kv_len,
    )


def export_and_decompose(
    case: SdpaCase,
) -> tuple[torch.fx.GraphModule, Counter[str]]:
    graph_module = export(case.module, case.inputs, strict=True).module()
    graph_module = DecomposeScaledDotProductAttentionPass()(graph_module).graph_module
    op_counts = summarize_graph_ops(graph_module)
    return graph_module, op_counts


def summarize_graph_ops(graph_module: torch.fx.GraphModule) -> Counter[str]:
    return Counter(
        str(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    )


def check_case_correctness(
    case: SdpaCase,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    eager_output = case.module(*case.inputs)
    decomposed_module, _ = export_and_decompose(case)
    decomposed_output = decomposed_module(*case.inputs)
    torch.testing.assert_close(decomposed_output, eager_output, atol=atol, rtol=rtol)


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


def test_sdpa_decode_case_matches_decomposed_graph() -> None:
    case = build_decode_case(128, SdpaDecodeConfig())
    check_case_correctness(case, atol=1e-5, rtol=1e-5)


def test_sdpa_decomposition_removes_sdpa_op() -> None:
    case = build_decode_case(512, SdpaDecodeConfig())
    graph_module, _ = export_and_decompose(case)
    assert not any(
        node.target == torch.ops.aten.scaled_dot_product_attention.default
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    )


@common.parametrize("test_case", test_suite)
@common.XfailIfNoCorstone320
def test_sdpa_u85_INT(test_case: test_case_t):
    model, test_input = test_case()
    pipeline = EthosU85PipelineINT[input_t](model, test_input, [], [])
    pipeline.pop_stage("check.quant_nodes")
    pipeline.pop_stage("check_count.exir")
    pipeline.pop_stage("run_method_and_compare_outputs")
    pipeline.run()
