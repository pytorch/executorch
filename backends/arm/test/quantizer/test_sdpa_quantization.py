# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from dataclasses import replace
from functools import partial
from unittest.mock import patch

import pytest
import torch
from executorch.backends.arm.arm_vela import vela_compile, VelaCompileResult
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer.arm_quantizer import (
    EthosUQuantizer,
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
    TOSAQuantizer,
)
from executorch.backends.arm.quantizer.quantization_config import (
    _derive_matmul_accumulator_qparams,
    get_int8_matmul_accumulator_qspec,
)
from executorch.backends.arm.test.tester.quantize import ArmQuantize
from executorch.backends.arm.test.tester.test_pipeline import TosaPipelineINT
from executorch.backends.cortex_m.quantizer.node_finders import (
    SDPAFinder,
    SDPAMatmulFinder,
    SDPAScoreFinder,
)
from executorch.backends.test.harness.stages import StageType
from executorch.examples.models.llama.attention import KVCache, SDPA
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    StaticQuantizedKVCache,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import LoweredBackendModule
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


@pytest.mark.parametrize("policy", ["default", "a8", "a16", "disabled", "a8_disabled"])
@pytest.mark.parametrize("strict", [False, True])
def test_sdpa_policy_is_opt_in_and_preserves_unrelated_matmul(policy, strict):
    inputs = _inputs()
    model = AttentionAndMatmul()
    exported = torch.export.export(model, inputs, strict=strict).module()
    quantizer = EthosUQuantizer(EthosUCompileSpec("ethos-u85-256"))
    if policy != "default":
        _configure_attention(
            quantizer,
            (
                get_symmetric_quantization_config()
                if policy in ("a8", "a8_disabled")
                else get_symmetric_a16w8_quantization_config()
            ),
        )
    if policy in ("disabled", "a8_disabled"):
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
    if policy in ("default", "disabled", "a8_disabled"):
        assert not any(
            str(node.target).startswith("quantized_decomposed.")
            for node in converted.graph.nodes
        )
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


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "finder", [SDPAMatmulFinder("qk"), SDPAMatmulFinder("av"), SDPAScoreFinder()]
)
def test_sdpa_role_only_policy_does_not_enable_native_decomposition(finder, strict):
    inputs = _inputs()
    model = torch.export.export(AttentionAndMatmul(), inputs, strict=strict).module()
    quantizer = EthosUQuantizer(EthosUCompileSpec("ethos-u85-256"))
    quantizer.set_node_finder(get_symmetric_a16w8_quantization_config(), finder)

    prepared = prepare_pt2e(model, quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared, fold_quantize=False)

    assert (
        sum(
            node.target == torch.ops.aten.scaled_dot_product_attention.default
            for node in converted.graph.nodes
        )
        == 2
    )
    assert not list(finder.find_nodes(converted))
    assert not any(
        str(node.target).startswith("quantized_decomposed.")
        for node in converted.graph.nodes
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
@pytest.mark.parametrize("strict", [False, True])
def test_native_a8_attention_masks_gqa_and_explicit_scale(mask_kind, strict):
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
        torch.export.export(model, inputs, strict=strict).module(), quantizer
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


def test_native_a8_attention_masked_softmax_tosa_INT():
    class Attention(torch.nn.Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=0.37
            )

    q = torch.zeros(1, 1, 1, 8)
    q[..., 0] = 1.0
    k = torch.zeros(1, 1, 8, 8)
    k[..., 0] = torch.tensor([2.0, 1.0, -1.0, -2.0, 4.0, 3.0, -3.0, -4.0])
    # Basis values expose the masked attention probabilities in the output.
    inputs = (
        q,
        k,
        torch.eye(8).reshape(1, 1, 8, 8),
        torch.arange(8).reshape(1, 1, 1, 8) < 4,
    )

    def compare_outputs(reference, actual, qparams):
        torch.testing.assert_close(actual, reference, atol=0.04, rtol=0.03)
        torch.testing.assert_close(
            actual[..., 4:], torch.zeros_like(actual[..., 4:]), atol=0, rtol=0
        )

    pipeline = TosaPipelineINT(
        Attention(),
        inputs,
        [],
        [],
        tosa_extensions=["int16"],
        run_on_tosa_ref_model=True,
    )
    quantizer = _configure_attention(
        TOSAQuantizer(pipeline.tester.compile_spec),
        get_symmetric_quantization_config(),
    )
    pipeline.change_args("quantize", ArmQuantize(quantizer, set_global=False))
    pipeline.change_args(
        "run_method_and_compare_outputs",
        stage=StageType.TO_EXECUTORCH,
        inputs=inputs,
        reference_stage_type=StageType.QUANTIZE,
        compare_callback=compare_outputs,
    )
    pipeline.run()

    edge = pipeline.tester.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
    delegates = [
        module
        for module in edge.exported_program().graph_module.modules()
        if isinstance(module, LoweredBackendModule)
    ]
    assert len(delegates) == 1
    assert delegates[0].backend_id == "TOSABackend"
    block = TosaGraph.GetRootAs(delegates[0].processed_bytes, 0).Regions(0).Blocks(0)
    tensors = {
        block.Tensors(i).Name(): block.Tensors(i) for i in range(block.TensorsLength())
    }
    operators = [block.Operators(i) for i in range(block.OperatorsLength())]
    matmuls = [op for op in operators if op.Op() == Op.MATMUL]
    assert len(matmuls) == 2
    producers = {
        op.Outputs(i): op for op in operators for i in range(op.OutputsLength())
    }
    qk_rescales = []
    for op in operators:
        if (
            op.Op() != Op.RESCALE
            or tensors[op.Inputs(0)].Type() != DType.INT32
            or tensors[op.Outputs(0)].Type() != DType.INT16
        ):
            continue
        producer = producers.get(op.Inputs(0))
        while producer is not None and producer.Op() in (Op.RESHAPE, Op.TRANSPOSE):
            producer = producers.get(producer.Inputs(0))
        if producer in matmuls:
            qk_rescales.append((producer, op))
    assert len(qk_rescales) == 1
    qk, _ = qk_rescales[0]
    assert [tensors[qk.Inputs(i)].Type() for i in range(2)] == [DType.INT8] * 2
    assert tensors[qk.Outputs(0)].Type() == DType.INT32
    pipeline.tester.run_method_and_compare_outputs(
        stage=StageType.TO_EXECUTORCH,
        inputs=inputs,
        reference_stage_type=StageType.INITIAL_MODEL,
        compare_callback=compare_outputs,
    )


class CachedAttention(torch.nn.Module):
    def __init__(self, per_channel, cache_dtype=torch.int8):
        super().__init__()
        self.cache = StaticQuantizedKVCache.from_float(
            KVCache(1, 8, 2, 8, False),
            scale=0.025 * 127 / torch.iinfo(cache_dtype).max,
            use_custom_update_cache_op=False,
            use_per_channel=per_channel,
            quantized_cache_dtype=cache_dtype,
        )
        self.attention = SDPA(32, 8, 2, 8)

    def forward(self, q, k, v, position, mask):
        k, v = self.cache.update(position, k, v)
        return self.attention(position, q, k, v, 1, 1, mask)


@pytest.mark.parametrize("cache_dtype", [torch.int8, torch.int16])
def test_attention_policy_preserves_cache_storage_and_history(cache_dtype):
    model = CachedAttention(False, cache_dtype)
    reference = deepcopy(model)
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
    for name in ("cache.k_cache", "cache.v_cache"):
        assert buffers[name].dtype == cache_dtype
        buffers[name].zero_()
    with torch.no_grad():
        expected = reference(*inputs)
        actual = converted(*inputs)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.006)
        second_inputs = (
            inputs[0],
            inputs[1] * 0.5,
            inputs[2] * 0.5,
            torch.tensor([1]),
            torch.arange(8).reshape(1, 1, 1, 8) < 2,
        )
        expected = reference(*second_inputs)
        actual = converted(*second_inputs)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.006)
    for name, reference_buffer in reference.named_buffers():
        if name.endswith(("k_cache", "v_cache")):
            torch.testing.assert_close(buffers[name], reference_buffer, rtol=0, atol=0)
            assert torch.count_nonzero(buffers[name][:, :2]) > 0
            assert torch.count_nonzero(buffers[name][:, 2:]) == 0


@pytest.mark.parametrize(
    "policy,cache_dtype,compile_with_vela",
    [
        ("native_a8", torch.int8, False),
        ("native_a16", torch.int8, False),
        ("global_a8_per_channel", torch.int8, False),
        ("native_a16", torch.int16, False),
        pytest.param("native_a16", torch.int16, True, id="int16-vela"),
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_attention_quantization_preserves_delegated_variable_cache(
    policy, cache_dtype, compile_with_vela, strict, capsys
):
    if compile_with_vela:
        vela = pytest.importorskip("ethosu.vela.vela")

        with pytest.raises(SystemExit) as help_exit:
            vela.main(["--help"])
        assert help_exit.value.code == 0
        if "--separate-persistent-region" not in capsys.readouterr().out:
            pytest.skip(
                "Vela lacks --separate-persistent-region (repository pin: 5.2.0); "
                "real INT16 compilation requires Vela 5.3.0 or a compatible dev build"
            )
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
        CachedAttention(per_channel, cache_dtype), inputs, strict=strict
    ).module()
    prepared = prepare_pt2e(model, quantizer)
    prepared(*inputs)
    converted = convert_pt2e(prepared, fold_quantize=False)
    for name, buffer in converted.named_buffers():
        if name.endswith(("k_cache", "v_cache")):
            assert buffer.dtype == cache_dtype
            buffer.zero_()
    exported = torch.export.export(converted, inputs, strict=strict)
    captured = []

    def capture(tosa, *args, **kwargs):
        captured.append(tosa)
        if compile_with_vela:
            result = vela_compile(tosa, *args, **kwargs)
            assert b"persistent_size" in result.processed_bytes
            return result
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
    variable_type = DType.INT16 if cache_dtype == torch.int16 else DType.INT8
    assert variable_types == [variable_type, variable_type]
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
    with pytest.raises(ValueError, match=f"requires a matmul.*{other.name}"):
        config.get_output_act_qspec(other)
    q.meta["val"] = torch.empty(1, 1, 131072, device="meta")
    with pytest.raises(ValueError, match=f"overflow INT32.*{score.name}"):
        config.get_output_act_qspec(score)
    q.meta["val"] = torch.empty(1, 1, 64, device="meta")
    qspec = config.get_output_act_qspec(score)
    assert isinstance(qspec, DerivedQuantizationSpec)
    assert qspec.derived_from == [(q, score), (k, score)]


def test_matmul_accumulator_does_not_trust_factory_range_for_overflow():
    config = get_symmetric_quantization_config()
    input_qspec = replace(
        config.input_activation,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-1,
        quant_max=1,
    )
    graph = torch.fx.Graph()
    lhs, rhs = graph.placeholder("lhs"), graph.placeholder("rhs")
    matmul = graph.call_function(torch.ops.aten.bmm.default, (lhs, rhs))
    lhs.meta["val"] = torch.empty(1, 1, 131072, device="meta")
    with pytest.raises(ValueError, match=f"overflow INT32.*{matmul.name}"):
        get_int8_matmul_accumulator_qspec(matmul, input_qspec=input_qspec)


@pytest.mark.parametrize("shape_kind", ["missing", "scalar", "symbolic"])
def test_matmul_accumulator_requires_static_shape_metadata(shape_kind):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    config = get_symmetric_quantization_config()
    input_qspec = replace(config.input_activation, qscheme=torch.per_tensor_symmetric)
    graph = torch.fx.Graph()
    lhs, rhs = graph.placeholder("lhs"), graph.placeholder("rhs")
    matmul = graph.call_function(torch.ops.aten.bmm.default, (lhs, rhs))
    if shape_kind == "scalar":
        lhs.meta["val"] = torch.empty((), device="meta")
    elif shape_kind == "symbolic":
        reduction = ShapeEnv().create_unbacked_symint()
        lhs.meta["val"] = torch.empty(1, 1, reduction, device="meta")
    with pytest.raises(
        ValueError, match=rf"(shape metadata|static reduction).*{matmul.name}"
    ):
        get_int8_matmul_accumulator_qspec(matmul, input_qspec=input_qspec)


@pytest.mark.parametrize("input_count", [0, 1, 2])
def test_matmul_accumulator_requires_tensor_inputs(input_count):
    graph = torch.fx.Graph()
    lhs = graph.placeholder("lhs")
    matmul = graph.call_function(torch.ops.aten.bmm.default, (lhs, 1)[:input_count])
    input_qspec = replace(
        get_symmetric_quantization_config().input_activation,
        qscheme=torch.per_tensor_symmetric,
    )
    with pytest.raises(ValueError, match=f"two tensor input nodes.*{matmul.name}"):
        get_int8_matmul_accumulator_qspec(matmul, input_qspec=input_qspec)


@pytest.mark.parametrize("invalid", ["int16", "affine", "dynamic"])
def test_matmul_accumulator_rejects_mismatched_actual_input_observers(invalid):
    from torchao.quantization.pt2e.observer import MinMaxObserver

    lhs = MinMaxObserver(
        dtype=torch.int16 if invalid == "int16" else torch.int8,
        qscheme=(
            torch.per_tensor_affine
            if invalid == "affine"
            else torch.per_tensor_symmetric
        ),
        is_dynamic=invalid == "dynamic",
    )
    rhs = MinMaxObserver(dtype=torch.int8, qscheme=torch.per_tensor_symmetric)
    lhs(torch.tensor([-1.0, 1.0]))
    rhs(torch.tensor([-2.0, 2.0]))
    graph = torch.fx.Graph()
    q, k = graph.placeholder("q"), graph.placeholder("k")
    matmul = graph.call_function(torch.ops.aten.bmm.default, (q, k))
    q.meta["val"] = torch.empty(1, 1, 64, device="meta")
    input_qspec = replace(
        get_symmetric_quantization_config().input_activation,
        qscheme=torch.per_tensor_symmetric,
    )
    qspec = get_int8_matmul_accumulator_qspec(matmul, input_qspec=input_qspec)
    with pytest.raises(
        ValueError, match=f"static symmetric INT8 observers.*{matmul.name}"
    ):
        qspec.derive_qparams_fn([lhs, rhs])


@pytest.mark.parametrize(
    "invalid", ["zero", "negative", "nan", "infinite", "zero_point"]
)
def test_sdpa_policy_rejects_invalid_derived_score_qparams(invalid):
    class Observer:
        dtype = torch.int8
        qscheme = torch.per_tensor_symmetric
        is_dynamic = False

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


@pytest.mark.parametrize(
    "invalid",
    ["dynamic", "affine", "int16", "missing_bounds", "wide_bounds", "empty_bounds"],
)
def test_matmul_accumulator_requires_static_symmetric_int8(invalid):
    config = get_symmetric_quantization_config()
    quant_min = {
        "missing_bounds": None,
        "wide_bounds": -129,
        "empty_bounds": 127,
    }.get(invalid, config.input_activation.quant_min)
    activation = replace(
        config.input_activation,
        qscheme=(
            torch.per_tensor_affine
            if invalid == "affine"
            else torch.per_tensor_symmetric
        ),
        dtype=torch.int16 if invalid == "int16" else torch.int8,
        is_dynamic=invalid == "dynamic",
        quant_min=quant_min,
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
    with pytest.raises(ValueError, match=f"static symmetric INT8.*{matmul.name}"):
        accumulator.get_output_act_qspec(matmul)
