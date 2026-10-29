# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from functools import partial
from unittest.mock import patch

import pytest
import torch
from executorch.backends.arm.arm_vela import VelaCompileResult
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.quantizer.quantization_config import (
    _derive_matmul_accumulator_qparams,
    get_int8_matmul_accumulator_qspec,
)
from executorch.backends.cortex_m.quantizer.node_finders import (
    SDPAFinder,
    SDPAMatmulFinder,
    SDPAScoreFinder,
)
from executorch.examples.models.llama.attention import KVCache, SDPA
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    StaticQuantizedKVCache,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import DerivedQuantizationSpec
from tosa.DType import DType  # type: ignore[import-untyped]
from tosa.Op import Op  # type: ignore[import-untyped]
from tosa.TosaGraph import TosaGraph  # type: ignore[import-untyped]


class AttentionAndMatmul(torch.nn.Module):
    def forward(self, q, k, v, mask, lhs, rhs):
        unrelated = torch.bmm(lhs, rhs)
        first = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask
        )
        second = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask
        )
        return unrelated, first + second


def _inputs():
    generator = torch.Generator().manual_seed(47)
    return (
        torch.randn(1, 2, 1, 8, generator=generator),
        torch.randn(1, 2, 8, 8, generator=generator),
        torch.randn(1, 2, 8, 8, generator=generator),
        torch.arange(8).reshape(1, 1, 1, 8) < 4,
        torch.randn(2, 3, 4, generator=generator),
        torch.randn(2, 4, 5, generator=generator),
    )


def _configure_attention(quantizer, config):
    quantizer.set_node_finder(config, SDPAFinder())
    if config is not None and config.input_activation.dtype == torch.int8:
        quantizer.set_node_finder(
            get_symmetric_a16w8_quantization_config(), SDPAScoreFinder()
        )
        input_qspec = replace(
            config.input_activation, qscheme=torch.per_tensor_symmetric
        )
        quantizer.set_node_finder(
            replace(
                config,
                input_activation=input_qspec,
                output_activation=partial(
                    get_int8_matmul_accumulator_qspec, input_qspec=input_qspec
                ),
            ),
            SDPAMatmulFinder("qk"),
        )
    return quantizer


@pytest.mark.parametrize("policy", ["default", "a8", "a16", "disabled"])
def test_sdpa_policy_is_opt_in_and_preserves_unrelated_matmul(policy):
    inputs = _inputs()
    model = AttentionAndMatmul()
    exported = torch.export.export(model, inputs, strict=True).module()
    quantizer = EthosUQuantizer(EthosUCompileSpec("ethos-u85-256"))
    if policy != "default":
        _configure_attention(
            quantizer,
            (
                get_symmetric_quantization_config()
                if policy == "a8"
                else get_symmetric_a16w8_quantization_config()
            ),
        )
    if policy == "disabled":
        quantizer.set_node_finder(None, SDPAFinder())
    prepared = prepare_pt2e(exported, quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared, fold_quantize=False)
    attention_matmuls = list(SDPAMatmulFinder().find_nodes(converted))
    unrelated_matmuls = [
        node
        for node in converted.graph.nodes
        if node.target == torch.ops.aten.bmm.default and node not in attention_matmuls
    ]
    assert len(unrelated_matmuls) == 1
    assert unrelated_matmuls[0].args[0].op == "placeholder"
    assert len(attention_matmuls) == (4 if policy in ("a8", "a16") else 0)
    if policy in ("a8", "a16"):
        assert len(SDPAFinder._groups(converted)) == 2
        for node in attention_matmuls:
            for operand in node.args[:2]:
                assert (
                    operand.target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                )
                assert operand.args[5] == (
                    torch.int8 if policy == "a8" else torch.int16
                )
        qk_matmuls = set(SDPAMatmulFinder("qk").find_nodes(converted))
        assert len(qk_matmuls) == 2
        assert len(list(SDPAMatmulFinder("av").find_nodes(converted))) == 2
        if policy == "a8":
            masked_scores = [
                node
                for node in SDPAFinder().find_nodes(converted)
                if node.target == torch.ops.aten.add.Tensor
            ]
            assert len(masked_scores) == 2
            for score in masked_scores:
                for operand in score.args[:2]:
                    assert (
                        operand.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                    )
                    assert operand.args[5] == torch.int16
            for node in attention_matmuls:
                if node not in qk_matmuls:
                    continue
                score_quantizers = [
                    user
                    for user in node.users
                    if user.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ]
                assert len(score_quantizers) == 1
                score_quantizer = score_quantizers[0]
                assert score_quantizer.args[5] == torch.int32
                assert score_quantizer.args[2] == 0
                assert score_quantizer.args[1] == pytest.approx(
                    node.args[0].args[1] * node.args[1].args[1]
                )
    expected = model(*inputs)
    actual = converted(*inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[1],
        expected[1],
        rtol=0.03 if policy == "a8" else 0.01,
        atol=0.04 if policy == "a8" else 0.006,
    )


def test_sdpa_policy_requires_composable_quantizer():
    quantizer = EthosUQuantizer(
        EthosUCompileSpec("ethos-u85-256"), use_composable_quantizer=False
    )
    with pytest.raises(NotImplementedError, match="composable"):
        quantizer.set_node_finder(
            get_symmetric_a16w8_quantization_config(), SDPAFinder()
        )


@pytest.mark.parametrize("mask_kind", ["boolean", "additive", "causal"])
def test_native_a8_attention_masks_gqa_and_explicit_scale(mask_kind):
    class Attention(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if mask_kind == "causal" else mask,
                is_causal=mask_kind == "causal",
                scale=0.37,
                enable_gqa=True,
            )

    generator = torch.Generator().manual_seed(29)
    keep = torch.arange(8).reshape(1, 1, 1, 8) < 4
    mask = keep if mask_kind == "boolean" else torch.where(keep, 0.0, float("-inf"))
    inputs = (
        torch.randn(1, 4, 3, 8, generator=generator),
        torch.randn(1, 2, 8, 8, generator=generator),
        torch.randn(1, 2, 8, 8, generator=generator),
        mask,
    )
    model = Attention()
    quantizer = _configure_attention(
        EthosUQuantizer(EthosUCompileSpec("ethos-u85-256")),
        get_symmetric_quantization_config(),
    )
    prepared = prepare_pt2e(
        torch.export.export(model, inputs, strict=True).module(), quantizer
    )
    prepared(*inputs)
    converted = convert_pt2e(prepared, fold_quantize=False)
    actual = converted(*inputs)
    torch.testing.assert_close(actual, model(*inputs), rtol=0.03, atol=0.04)
    changed_v = inputs[2].clone()
    changed_v[:, :, 4:] = 10000
    torch.testing.assert_close(
        converted(inputs[0], inputs[1], changed_v, mask), actual, rtol=0, atol=0
    )


class CachedAttention(torch.nn.Module):
    def __init__(self, per_channel):
        super().__init__()
        self.cache = StaticQuantizedKVCache.from_float(
            KVCache(1, 8, 2, 8, False),
            scale=0.025,
            use_custom_update_cache_op=False,
            use_per_channel=per_channel,
        )
        self.attention = SDPA(32, 8, 2, 8)

    def forward(self, q, k, v, position, mask):
        k, v = self.cache.update(position, k, v)
        return self.attention(position, q, k, v, 1, 1, mask)


def test_attention_policy_does_not_change_mutable_cache_storage_dtype():
    model = CachedAttention(False)
    model.cache = KVCache(1, 8, 2, 8, False)
    inputs = (
        torch.ones(1, 4, 1, 8),
        torch.ones(1, 2, 1, 8),
        torch.ones(1, 2, 1, 8),
        torch.tensor([0]),
        torch.arange(8).reshape(1, 1, 1, 8) < 1,
    )
    exported = torch.export.export(model, inputs, strict=True).module()
    quantizer = _configure_attention(
        EthosUQuantizer(EthosUCompileSpec("ethos-u85-256")),
        get_symmetric_a16w8_quantization_config(),
    )
    prepared = prepare_pt2e(exported, quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared, fold_quantize=False)
    buffers = dict(converted.named_buffers())
    assert buffers["cache.k_cache"].dtype == torch.float32
    assert buffers["cache.v_cache"].dtype == torch.float32


@pytest.mark.parametrize("policy", ["native_a8", "native_a16", "global_a8_per_channel"])
def test_attention_quantization_preserves_delegated_int8_variable_cache(policy):
    generator = torch.Generator().manual_seed(71)
    inputs = (
        torch.randn(1, 4, 1, 8, generator=generator),
        torch.randn(1, 2, 1, 8, generator=generator),
        torch.randn(1, 2, 1, 8, generator=generator),
        torch.tensor([0]),
        torch.arange(8).reshape(1, 1, 1, 8) < 1,
    )
    spec = EthosUCompileSpec("ethos-u85-256").set_delegate_mutable_buffers()
    quantizer = EthosUQuantizer(spec)
    per_channel = policy == "global_a8_per_channel"
    if per_channel:
        quantizer.set_global(get_symmetric_quantization_config())
    else:
        _configure_attention(
            quantizer,
            (
                get_symmetric_quantization_config()
                if policy == "native_a8"
                else get_symmetric_a16w8_quantization_config()
            ),
        )
    model = torch.export.export(
        CachedAttention(per_channel), inputs, strict=True
    ).module()
    prepared = prepare_pt2e(model, quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared, fold_quantize=False)
    for name, buffer in converted.named_buffers():
        if name.endswith(("k_cache", "v_cache")):
            assert buffer.dtype == torch.int8
            buffer.zero_()
    exported = torch.export.export(converted, inputs, strict=True)
    captured = []

    def capture(tosa, *args, **kwargs):
        captured.append(tosa)
        return VelaCompileResult(processed_bytes=b"test", scratch_size=0)

    with patch("executorch.backends.arm.ethosu.backend.vela_compile", capture):
        edge = to_edge_transform_and_lower(
            exported,
            partitioner=[EthosUPartitioner(spec)],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    assert len(captured) == (2 if per_channel else 1)
    variable_types = []
    matmul_types = []
    reads = writes = 0
    for data in captured:
        block = TosaGraph.GetRootAs(data, 0).Regions(0).Blocks(0)
        tensors = {
            block.Tensors(i).Name(): block.Tensors(i)
            for i in range(block.TensorsLength())
        }
        variable_types.extend(
            tensor.Type() for tensor in tensors.values() if tensor.Variable()
        )
        operators = [block.Operators(i) for i in range(block.OperatorsLength())]
        reads += sum(op.Op() == Op.VARIABLE_READ for op in operators)
        writes += sum(op.Op() == Op.VARIABLE_WRITE for op in operators)
        matmul_types.extend(
            (
                tensors[op.Inputs(0)].Type(),
                tensors[op.Inputs(1)].Type(),
                tensors[op.Outputs(0)].Type(),
            )
            for op in operators
            if op.Op() == Op.MATMUL
        )
    assert variable_types == [DType.INT8, DType.INT8]
    assert reads == 2 and writes == 2
    operand_type = DType.INT16 if policy == "native_a16" else DType.INT8
    accumulator_type = DType.INT48 if policy == "native_a16" else DType.INT32
    assert matmul_types == [(operand_type, operand_type, accumulator_type)] * 2
    assert not any(
        "aten.copy" in str(node.target) for node in edge.exported_program().graph.nodes
    )


def test_matmul_accumulator_rejects_other_ops_and_overflow():
    config = get_symmetric_quantization_config()
    input_qspec = replace(config.input_activation, qscheme=torch.per_tensor_symmetric)
    config = replace(
        config,
        input_activation=input_qspec,
        output_activation=partial(
            get_int8_matmul_accumulator_qspec, input_qspec=input_qspec
        ),
    )
    graph = torch.fx.Graph()
    q = graph.placeholder("q")
    k = graph.placeholder("k")
    score = graph.call_function(torch.ops.aten.bmm.default, (q, k))
    other = graph.call_function(torch.ops.aten.add.Tensor, (q, k))
    with pytest.raises(ValueError, match="requires a matmul"):
        config.get_output_act_qspec(other)
    q.meta["val"] = torch.empty(1, 1, 131072, device="meta")
    with pytest.raises(ValueError, match="overflow INT32"):
        config.get_output_act_qspec(score)
    q.meta["val"] = torch.empty(1, 1, 64, device="meta")
    qspec = config.get_output_act_qspec(score)
    assert isinstance(qspec, DerivedQuantizationSpec)
    assert qspec.derived_from == [(q, score), (k, score)]


@pytest.mark.parametrize(
    "invalid", ["zero", "negative", "nan", "infinite", "zero_point"]
)
def test_sdpa_policy_rejects_invalid_derived_score_qparams(invalid):
    class Observer:
        def __init__(self, scale, zero=0):
            self.scale = scale
            self.zero = zero

        def calculate_qparams(self):
            return torch.tensor([self.scale]), torch.tensor([self.zero])

    scales = {
        "zero": 0.0,
        "negative": -0.1,
        "nan": float("nan"),
        "infinite": float("inf"),
        "zero_point": 0.1,
    }
    with pytest.raises(ValueError, match="zero-centered inputs"):
        _derive_matmul_accumulator_qparams(
            [Observer(scales[invalid], int(invalid == "zero_point")), Observer(0.2)]
        )


@pytest.mark.parametrize("invalid", ["dynamic", "affine", "int16", "missing_bounds"])
def test_matmul_accumulator_requires_static_symmetric_int8(invalid):
    config = get_symmetric_quantization_config()
    activation = replace(
        config.input_activation,
        qscheme=(
            torch.per_tensor_affine
            if invalid == "affine"
            else torch.per_tensor_symmetric
        ),
        dtype=torch.int16 if invalid == "int16" else torch.int8,
        is_dynamic=invalid == "dynamic",
        quant_min=(
            None if invalid == "missing_bounds" else config.input_activation.quant_min
        ),
    )
    accumulator = replace(
        config,
        input_activation=activation,
        output_activation=partial(
            get_int8_matmul_accumulator_qspec, input_qspec=activation
        ),
    )
    graph = torch.fx.Graph()
    lhs, rhs = graph.placeholder("lhs"), graph.placeholder("rhs")
    matmul = graph.call_function(torch.ops.aten.bmm.default, (lhs, rhs))
    with pytest.raises(ValueError, match="static symmetric INT8"):
        accumulator.get_output_act_qspec(matmul)
