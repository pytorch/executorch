# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import Counter

import torch
from executorch.backends.arm.test.common import parametrize, xfail_type
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase


class CortexMSDPA(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 2,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_softmax_default": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 0,
    }

    def __init__(self, attn_mask: torch.Tensor | None = None, **sdpa_kwargs):
        super().__init__()
        self.register_buffer("attn_mask", attn_mask)
        self.sdpa_kwargs = sdpa_kwargs

    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=self.attn_mask, **self.sdpa_kwargs
        )


class CortexMMultiheadAttention(torch.nn.Module):
    ops_before_transforms = {
        "executorch_exir_dialects_edge__ops_aten_linear_default": 2,
        "executorch_exir_dialects_edge__ops_aten_bmm_default": 2,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 1,
        "executorch_exir_dialects_edge__ops_aten__softmax_default": 1,
    }

    ops_after_transforms = {
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_linear_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_quantized_batch_matmul_default": 2,
        "executorch_exir_dialects_edge__ops_cortex_m_softmax_default": 1,
        "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 0,
    }

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

    def forward(self, x):
        return self.attention(x, x, x, need_weights=False)[0]


def _qkv(query_shape, key_shape, value_shape, dtype=torch.float32):
    return (
        torch.randn(query_shape, dtype=dtype),
        torch.randn(key_shape, dtype=dtype),
        torch.randn(value_shape, dtype=dtype),
    )


test_cases = {
    "self_attention": McuTestCase(
        CortexMSDPA(), _qkv((1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16))
    ),
    "cross_attention": McuTestCase(
        CortexMSDPA(), _qkv((1, 2, 4, 32), (1, 2, 12, 32), (1, 2, 12, 8))
    ),
    "rank3": McuTestCase(CortexMSDPA(), _qkv((2, 8, 64), (2, 8, 64), (2, 8, 64))),
    "explicit_scale": McuTestCase(
        CortexMSDPA(scale=0.3), _qkv((1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16))
    ),
    "multihead_attention": McuTestCase(
        CortexMMultiheadAttention(embed_dim=32, num_heads=4),
        (torch.randn(1, 8, 32),),
    ),
}

not_decomposed_test_cases = {
    "attn_mask": McuTestCase(
        CortexMSDPA(attn_mask=torch.randn(8, 8)),
        _qkv((1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16)),
    ),
    "causal": McuTestCase(
        CortexMSDPA(is_causal=True), _qkv((1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16))
    ),
    "dropout": McuTestCase(
        CortexMSDPA(dropout_p=0.5), _qkv((1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16))
    ),
    "gqa": McuTestCase(
        CortexMSDPA(enable_gqa=True), _qkv((1, 4, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16))
    ),
    "float16": McuTestCase(
        CortexMSDPA(),
        _qkv((1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16), dtype=torch.float16),
    ),
    "empty_key": McuTestCase(
        CortexMSDPA(), _qkv((1, 2, 8, 16), (1, 2, 0, 16), (1, 2, 0, 16))
    ),
    "broadcast_batch": McuTestCase(
        CortexMSDPA(), _qkv((2, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16))
    ),
    "rank2": McuTestCase(CortexMSDPA(), _qkv((8, 16), (8, 16), (8, 16))),
}

_SCALE_NOT_FOLDED = (
    "The attention scale mul is left as an fp32 aten.mul between dequantize "
    "and quantize, no pass folds it into the quantize yet"
)
_LINEAR_RETRACED_AS_INT32 = (
    "Retracing a folded quantized linear gives an int32 output, which fails "
    "the dtype check of the bmm consuming it"
)

xfail_cases_dialect: dict[str, xfail_type] = {
    "self_attention": (_SCALE_NOT_FOLDED, RuntimeError),
    "cross_attention": (_SCALE_NOT_FOLDED, RuntimeError),
    "rank3": (_SCALE_NOT_FOLDED, RuntimeError),
    "explicit_scale": (_SCALE_NOT_FOLDED, RuntimeError),
    "multihead_attention": (_LINEAR_RETRACED_AS_INT32, Exception),
}
xfail_cases_impl: dict[str, xfail_type] = {
    "multihead_attention": (_LINEAR_RETRACED_AS_INT32, Exception),
}


def _transform_for_annotation(model, example_inputs, dynamic_shapes=None):
    # Without check_guards=False, whether export adds a _guards_fn call_module,
    # which ExportPass rejects, depends on the path of the calling file.
    graph_module = torch.export.export(
        model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
    ).module(check_guards=False)
    return CortexMQuantizer().transform_for_annotation(graph_module)


def _op_counts(graph_module) -> Counter:
    return Counter(
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    )


@parametrize("test_case", test_cases, xfails=xfail_cases_dialect)
def test_dialect_sdpa(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_dialect(
        test_case.model.ops_before_transforms,
        test_case.model.ops_after_transforms,
        qtol=2,
    )


@parametrize("test_case", test_cases, xfails=xfail_cases_impl)
def test_implementation_sdpa(test_case, cortex_m_target):
    tester = CortexMTester(
        test_case.model, test_case.example_inputs, target_config=cortex_m_target
    )
    tester.test_implementation(qtol=2)


@parametrize("test_case", test_cases)
def test_sdpa_decomposition(test_case):
    inputs = test_case.get_example_inputs()
    graph_module = _transform_for_annotation(test_case.model, inputs)

    op_counts = _op_counts(graph_module)
    assert op_counts[torch.ops.aten.scaled_dot_product_attention.default] == 0
    assert op_counts[torch.ops.aten.matmul.default] == 0
    assert op_counts[torch.ops.aten.bmm.default] == 2
    assert op_counts[torch.ops.aten.mul.Tensor] == 1
    assert op_counts[torch.ops.aten.softmax.int] == 1
    torch.testing.assert_close(graph_module(*inputs), test_case.model(*inputs))


@parametrize("test_case", not_decomposed_test_cases)
def test_sdpa_not_decomposed(test_case):
    graph_module = _transform_for_annotation(
        test_case.model, test_case.get_example_inputs()
    )

    assert (
        _op_counts(graph_module)[torch.ops.aten.scaled_dot_product_attention.default]
        == 1
    )


def test_sdpa_with_dynamic_shape_not_decomposed():
    seq_len = torch.export.Dim("seq_len", min=2, max=64)
    graph_module = _transform_for_annotation(
        CortexMSDPA(),
        _qkv((1, 2, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16)),
        dynamic_shapes=({2: seq_len}, {2: seq_len}, {2: seq_len}),
    )

    assert (
        _op_counts(graph_module)[torch.ops.aten.scaled_dot_product_attention.default]
        == 1
    )


@parametrize("test_case", test_cases)
def test_sdpa_scale_mul_qparams_allow_folding(test_case):
    """The scale mul can only be folded into the quantize after it without
    changing any int8 value if that quantize uses the scale of the dequantize
    before it times the attention scale, and the same zero point."""
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.quantize()
    graph_module = tester.get_artifact()

    muls = [
        node
        for node in graph_module.graph.nodes
        if node.target == torch.ops.aten.mul.Tensor
    ]
    assert len(muls) == 1
    dequantize, scale_constant = muls[0].args
    (quantize,) = muls[0].users
    assert (
        dequantize.target
        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
    )
    assert quantize.target == torch.ops.quantized_decomposed.quantize_per_tensor.default

    scale = getattr(graph_module, scale_constant.target).item()
    in_scale, in_zero_point = dequantize.args[1:3]
    out_scale, out_zero_point = quantize.args[1:3]
    assert out_zero_point == in_zero_point
    assert math.isclose(out_scale, in_scale * scale, rel_tol=1e-6)
