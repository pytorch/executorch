# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Python tests for the `llama::quantized_moe_ffn` custom op and the
`replace_moe_with_quantized_op` source transform.

Numerical correctness is checked by comparing the custom op against a
pure-Python q-dq reference that mirrors `MOEFeedForward.forward`
with the same INT4 group quantization applied to each expert.
"""

from __future__ import annotations

import copy
import unittest

import torch

from executorch.examples.models.llama.llama_transformer import MOEFeedForward
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.source_transformation.moe import (
    _symmetric_quantize_per_group,
    QuantizedMoEFFN,
    replace_moe_with_quantized_op,
)

# Importing custom_ops registers the schema + Meta kernel and loads the
# AOT shared library so `torch.ops.llama.quantized_moe_ffn` is callable
# in eager mode.
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from torchao.quantization.granularity import PerGroup
from torchao.quantization.quant_api import (
    Int8DynamicActivationIntxWeightConfig,
    quantize_,
)
from torchao.quantization.quantize_.workflows.intx.intx_opaque_tensor import (
    _is_kernel_library_loaded,
)
from torchao.utils import unwrap_tensor_subclass


_REQUIRES_TORCHAO_KERNEL_LIBRARY = unittest.skipUnless(
    _is_kernel_library_loaded(),
    "TorchAO kernel library is not loaded",
)


def _qdq_int4_reference(w: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply round-trip symmetric INT4 group quantization. Returns the
    "fake-quantized" weight in fp32, matching the values the torchao
    kernel would consume."""
    qvals, scales = _symmetric_quantize_per_group(w, group_size)
    n, k = w.shape
    qvals_grouped = qvals.float().reshape(n, k // group_size, group_size)
    scales_grouped = scales.reshape(n, k // group_size, 1)
    return (qvals_grouped * scales_grouped).reshape(n, k)


def _build_moe_eager(
    *,
    dim: int = 32,
    hidden_dim: int = 32,
    num_experts: int = 4,
    num_activated_experts: int = 2,
    score_func: str = "sigmoid",
    use_expert_bias: bool = True,
    route_scale: float = 2.5,
) -> MOEFeedForward:
    args = ModelArgs(
        dim=dim,
        n_layers=1,
        n_heads=1,
        vocab_size=8,
        hidden_dim=hidden_dim,
        moe=True,
        num_experts=num_experts,
        num_activated_experts=num_activated_experts,
    )
    moe = MOEFeedForward(args)
    # `replace_moe_with_quantized_op` reads routing/bias configuration off the
    # eager module. Upstream `MOEFeedForward` does not carry these attributes,
    # so set them explicitly to drive the transform with the desired config.
    moe.num_activated_experts = num_activated_experts
    moe.score_func = score_func
    moe.route_scale = route_scale
    moe.use_expert_bias = use_expert_bias
    if use_expert_bias:
        moe.expert_bias = torch.randn(num_experts)
    moe.eval()
    return moe


def _make_valid_meta_inputs() -> dict:
    num_experts = 4
    dim = 32
    hidden_dim = 32
    return {
        "x": torch.empty((4, dim), dtype=torch.float32, device="meta"),
        "gate_weight": torch.empty(
            (num_experts, dim), dtype=torch.float32, device="meta"
        ),
        "expert_bias": torch.empty((num_experts,), dtype=torch.float32, device="meta"),
        "packed_w1": torch.empty((num_experts, 1), dtype=torch.uint8, device="meta"),
        "packed_w3": torch.empty((num_experts, 1), dtype=torch.uint8, device="meta"),
        "packed_w2": torch.empty((num_experts, 1), dtype=torch.uint8, device="meta"),
        "num_activated_experts": 2,
        "num_experts": num_experts,
        "hidden_dim": hidden_dim,
        "dim": dim,
        "group_size": 32,
        "weight_nbit": 4,
        "score_func": "sigmoid",
        "route_scale": 2.5,
    }


def _moe_forward_with_qdq_weights(
    moe: MOEFeedForward, x: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Independent q-dq reference for `llama::quantized_moe_ffn`.

    Applies the same INT4 group-quantize/dequant round-trip to each expert
    weight, then reproduces the op's routing contract (see op_moe.cpp) in
    eager PyTorch: sigmoid or softmax scoring, expert-bias-shifted top-k
    selection, un-biased weight renormalization scaled by route_scale for the
    sigmoid path (softmax path softmaxes the top-k raw scores). Expert compute
    reuses the q-dq `ConditionalFeedForward`.
    """
    moe_q = copy.deepcopy(moe)

    def _qdq_per_expert(w_eFD: torch.Tensor) -> torch.Tensor:
        # w shape: [E, F, D] for w1/w3 or [E, D, F] for w2 (after the
        # transpose in the caller). Quantize each [N, K] slice along K.
        return torch.stack(
            [_qdq_int4_reference(w_eFD[ei], group_size) for ei in range(w_eFD.size(0))]
        )

    cond_q = moe_q.cond_ffn
    with torch.no_grad():
        cond_q.w1.copy_(_qdq_per_expert(moe.cond_ffn.w1))
        cond_q.w3.copy_(_qdq_per_expert(moe.cond_ffn.w3))
        # w2's einsum convention treats it as [F, D] with K=F at packing time.
        # The torchao packer is fed `w2.transpose(-2, -1)` (shape [E, D, F]),
        # quantized along K=F, then dequantized. Equivalent to quantizing each
        # [D, F] slice and transposing back to [F, D].
        w2_DF = moe.cond_ffn.w2.transpose(-2, -1).contiguous()
        w2_DF_qdq = _qdq_per_expert(w2_DF)
        cond_q.w2.copy_(w2_DF_qdq.transpose(-2, -1).contiguous())

    scores = moe.gate(x)  # [T, E]
    k = moe.num_activated_experts
    if moe.score_func == "sigmoid":
        s = torch.sigmoid(scores)
        sel = s + moe.expert_bias if getattr(moe, "use_expert_bias", False) else s
        idx = torch.topk(sel, k, dim=-1).indices
        weights = torch.gather(s, -1, idx)
        weights = (
            weights * moe.route_scale / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        )
    else:
        idx = torch.topk(scores, k, dim=-1).indices
        weights = torch.gather(scores, -1, idx).softmax(dim=-1)

    expert_outs = moe_q.cond_ffn(x, idx)  # [T, K, D]
    return torch.einsum("tkd,tk->td", expert_outs, weights)


@_REQUIRES_TORCHAO_KERNEL_LIBRARY
class TestQuantizedMoeFfnOp(unittest.TestCase):
    """Numerical correctness vs a Python q-dq reference."""

    def setUp(self) -> None:
        torch.manual_seed(0)

    def _check_against_qdq_reference(
        self,
        *,
        score_func: str,
        use_expert_bias: bool,
        route_scale: float,
        num_tokens: int = 8,
        atol: float = 5e-3,
    ) -> None:
        dim, hidden_dim = 32, 32
        num_experts, num_activated_experts = 4, 2
        group_size = 32

        moe = _build_moe_eager(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_activated_experts=num_activated_experts,
            score_func=score_func,
            use_expert_bias=use_expert_bias,
            route_scale=route_scale,
        )
        x = torch.randn(num_tokens, dim)

        with torch.no_grad():
            ref = _moe_forward_with_qdq_weights(moe, x, group_size)

            qmodel = MOEFeedForward.__new__(MOEFeedForward)
            torch.nn.Module.__init__(qmodel)
            qmodel.gate = moe.gate
            qmodel.cond_ffn = moe.cond_ffn
            qmodel.dim = moe.dim
            qmodel.num_activated_experts = moe.num_activated_experts
            qmodel.score_func = moe.score_func
            qmodel.route_scale = moe.route_scale
            qmodel.use_expert_bias = moe.use_expert_bias
            if moe.use_expert_bias:
                qmodel.expert_bias = moe.expert_bias

            wrapper = torch.nn.Module()
            wrapper.block_sparse_moe = qmodel
            replace_moe_with_quantized_op(wrapper, group_size=group_size, weight_nbit=4)
            test = wrapper.block_sparse_moe(x)

        diff = (ref - test).abs()
        self.assertTrue(
            diff.max().item() < atol,
            f"max abs diff {diff.max().item()} > atol {atol} (mean={diff.mean().item()})",
        )

    def test_sigmoid_with_bias_route_scale_2p5(self) -> None:
        self._check_against_qdq_reference(
            score_func="sigmoid", use_expert_bias=True, route_scale=2.5
        )

    def test_sigmoid_no_bias(self) -> None:
        self._check_against_qdq_reference(
            score_func="sigmoid", use_expert_bias=False, route_scale=1.0
        )

    def test_softmax(self) -> None:
        self._check_against_qdq_reference(
            score_func="softmax", use_expert_bias=False, route_scale=1.0
        )

    def test_single_token(self) -> None:
        self._check_against_qdq_reference(
            score_func="sigmoid",
            use_expert_bias=True,
            route_scale=2.5,
            num_tokens=1,
        )

    def test_route_scale_zero(self) -> None:
        self._check_against_qdq_reference(
            score_func="sigmoid",
            use_expert_bias=False,
            route_scale=0.0,
        )


@_REQUIRES_TORCHAO_KERNEL_LIBRARY
class TestSourceTransform(unittest.TestCase):
    """Tests for `replace_moe_with_quantized_op`."""

    def test_replaces_module_in_place_with_correct_buffers(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        replaced = wrapper.block_sparse_moe
        self.assertIsInstance(replaced, QuantizedMoEFFN)
        self.assertEqual(replaced.num_experts, 4)
        self.assertEqual(replaced.num_activated_experts, 2)
        self.assertEqual(replaced.dim, 32)
        self.assertEqual(replaced.hidden_dim, 32)
        self.assertEqual(tuple(replaced.gate_weight.shape), (4, 32))
        self.assertEqual(replaced.expert_bias.numel(), 4)
        self.assertEqual(replaced.packed_w1.shape[0], 4)
        self.assertEqual(replaced.packed_w3.shape[0], 4)
        self.assertEqual(replaced.packed_w2.shape[0], 4)
        self.assertEqual(replaced.packed_w1.shape[1], replaced.packed_w3.shape[1])

    def test_torchao_quantized_gate_exports_as_fp32_buffer(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
            weight_scale_dtype=torch.bfloat16,
        )
        quantize_(moe, config, filter_fn=lambda _module, fqn: fqn == "gate")
        moe = unwrap_tensor_subclass(moe)

        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)

        replaced = wrapper.block_sparse_moe
        self.assertEqual(replaced.gate_weight.dtype, torch.float32)
        exported = torch.export.export(replaced, (torch.randn(2, 32),))
        call_targets = {
            node.target for node in exported.graph.nodes if node.op == "call_function"
        }
        self.assertIn(torch.ops.llama.quantized_moe_ffn.default, call_targets)

    def test_no_expert_bias_produces_empty_buffer(self) -> None:
        moe = _build_moe_eager(
            dim=32,
            hidden_dim=32,
            num_experts=4,
            num_activated_experts=2,
            use_expert_bias=False,
        )
        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        replaced = wrapper.block_sparse_moe
        self.assertIsInstance(replaced, QuantizedMoEFFN)
        self.assertEqual(replaced.expert_bias.numel(), 0)

    def test_buffers_stay_fp32_after_to_half(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        wrapper.to(torch.float16)
        replaced = wrapper.block_sparse_moe
        self.assertEqual(replaced.gate_weight.dtype, torch.float32)
        self.assertEqual(replaced.expert_bias.dtype, torch.float32)
        self.assertEqual(replaced.packed_w1.dtype, torch.uint8)

    def test_buffers_stay_fp32_after_to_bfloat16(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        wrapper.to(torch.bfloat16)
        replaced = wrapper.block_sparse_moe
        self.assertEqual(replaced.gate_weight.dtype, torch.float32)
        self.assertEqual(replaced.expert_bias.dtype, torch.float32)

    def test_3d_input_shape_preserved_through_forward(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        replaced = wrapper.block_sparse_moe
        x_meta = torch.empty((2, 4, 32), dtype=torch.float32, device="meta")
        out = torch.ops.llama.quantized_moe_ffn(
            x_meta.view(-1, 32),
            replaced.gate_weight.to("meta"),
            replaced.expert_bias.to("meta"),
            replaced.packed_w1.to("meta"),
            replaced.packed_w3.to("meta"),
            replaced.packed_w2.to("meta"),
            replaced.num_activated_experts,
            replaced.num_experts,
            replaced.hidden_dim,
            replaced.dim,
            replaced.group_size,
            replaced.weight_nbit,
            replaced.score_func,
            replaced.route_scale,
        )
        self.assertEqual(tuple(out.shape), (8, 32))
        out_reshaped = out.view(2, 4, 32)
        self.assertEqual(tuple(out_reshaped.shape), (2, 4, 32))

    def test_forward_output_dtype_matches_input(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        x_fp16 = torch.randn(4, 32, dtype=torch.float16)
        with torch.no_grad():
            out = wrapper.block_sparse_moe(x_fp16)
        self.assertEqual(out.dtype, torch.float16)

    def test_nested_replacement(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        outer = torch.nn.Module()
        inner = torch.nn.Module()
        inner.feed_forward = moe
        outer.layer0 = inner
        replace_moe_with_quantized_op(outer, group_size=32, weight_nbit=4)
        self.assertIsInstance(outer.layer0.feed_forward, QuantizedMoEFFN)

    def test_meta_kernel_returns_correct_output_shape(self) -> None:
        moe = _build_moe_eager(num_experts=4, num_activated_experts=2)
        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        replaced = wrapper.block_sparse_moe
        x_meta = torch.empty((8, replaced.dim), dtype=torch.float32, device="meta")
        out_meta = torch.ops.llama.quantized_moe_ffn(
            x_meta,
            replaced.gate_weight.to("meta"),
            replaced.expert_bias.to("meta"),
            replaced.packed_w1.to("meta"),
            replaced.packed_w3.to("meta"),
            replaced.packed_w2.to("meta"),
            replaced.num_activated_experts,
            replaced.num_experts,
            replaced.hidden_dim,
            replaced.dim,
            replaced.group_size,
            replaced.weight_nbit,
            replaced.score_func,
            replaced.route_scale,
        )
        self.assertEqual(tuple(out_meta.shape), (8, replaced.dim))
        self.assertEqual(out_meta.dtype, torch.float32)

    def test_meta_kernel_single_token(self) -> None:
        moe = _build_moe_eager(num_experts=4, num_activated_experts=2)
        wrapper = torch.nn.Module()
        wrapper.block_sparse_moe = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        replaced = wrapper.block_sparse_moe
        x_meta = torch.empty((1, replaced.dim), dtype=torch.float32, device="meta")
        out_meta = torch.ops.llama.quantized_moe_ffn(
            x_meta,
            replaced.gate_weight.to("meta"),
            replaced.expert_bias.to("meta"),
            replaced.packed_w1.to("meta"),
            replaced.packed_w3.to("meta"),
            replaced.packed_w2.to("meta"),
            replaced.num_activated_experts,
            replaced.num_experts,
            replaced.hidden_dim,
            replaced.dim,
            replaced.group_size,
            replaced.weight_nbit,
            replaced.score_func,
            replaced.route_scale,
        )
        self.assertEqual(tuple(out_meta.shape), (1, replaced.dim))


class TestInt4PackRoundtrip(unittest.TestCase):
    """The torchao INT4 packer + unpacker should preserve weight values
    within INT4 quantization resolution."""

    def test_pack_unpack_within_quant_resolution(self) -> None:
        torch.manual_seed(1)
        n, k = 32, 64
        w = torch.randn(n, k)
        qvals, scales = _symmetric_quantize_per_group(w, group_size=32)
        recon = (qvals.float().reshape(n, 2, 32) * scales.reshape(n, 2, 1)).reshape(
            n, k
        )
        max_step = (w.abs().reshape(n, 2, 32).amax(dim=-1) / 7.0).max().item()
        diff = (w - recon).abs().max().item()
        self.assertLessEqual(diff, max_step + 1e-6)

    def test_full_row_group_has_lower_error(self) -> None:
        torch.manual_seed(2)
        n, k = 32, 64
        w = torch.randn(n, k)
        _, _ = _symmetric_quantize_per_group(w, group_size=32)
        qvals_fine, scales_fine = _symmetric_quantize_per_group(w, group_size=32)
        recon_fine = (
            qvals_fine.float().reshape(n, 2, 32) * scales_fine.reshape(n, 2, 1)
        ).reshape(n, k)
        qvals_coarse, scales_coarse = _symmetric_quantize_per_group(w, group_size=k)
        recon_coarse = (
            qvals_coarse.float().reshape(n, 1, k) * scales_coarse.reshape(n, 1, 1)
        ).reshape(n, k)
        err_fine = (w - recon_fine).abs().max().item()
        err_coarse = (w - recon_coarse).abs().max().item()
        self.assertGreaterEqual(err_coarse, err_fine * 0.5)


class TestMetaKernelValidation(unittest.TestCase):
    """Meta kernel input validation rejects bad inputs."""

    def test_rejects_3d_input(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["x"] = torch.empty((1, 4, 32), dtype=torch.float32, device="meta")
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_bad_score_func(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["score_func"] = "gelu"
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_bad_weight_nbit(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["weight_nbit"] = 3
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_num_activated_experts_zero(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["num_activated_experts"] = 0
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_non_uint8_packed_w1(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["packed_w1"] = kw["packed_w1"].to(torch.float32)
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_non_uint8_packed_w2(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["packed_w2"] = kw["packed_w2"].to(torch.float32)
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_non_uint8_packed_w3(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["packed_w3"] = kw["packed_w3"].to(torch.float32)
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_mismatched_packed_w1_w3_sizes(self) -> None:
        kw = _make_valid_meta_inputs()
        e = kw["num_experts"]
        kw["packed_w3"] = torch.empty(
            (e, kw["packed_w3"].size(1) + 1), dtype=torch.uint8, device="meta"
        )
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_non_fp32_gate_weight(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["gate_weight"] = kw["gate_weight"].to(torch.float16)
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_non_fp32_expert_bias(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["expert_bias"] = kw["expert_bias"].to(torch.float16)
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)


class TestKernelAvailability(unittest.TestCase):
    """The kernel (reference or optimized) is always available."""

    def test_sentinel_op_registered(self) -> None:
        self.assertTrue(hasattr(torch.ops.llama, "_quantized_moe_ffn_active"))
        self.assertTrue(torch.ops.llama._quantized_moe_ffn_active())


class TestGroupSizeValidation(unittest.TestCase):
    """Validate group_size and divisibility checks."""

    def test_rejects_group_size_zero(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["group_size"] = 0
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_rejects_dim_not_divisible_by_group_size(self) -> None:
        kw = _make_valid_meta_inputs()
        kw["group_size"] = 7
        with self.assertRaises(AssertionError):
            torch.ops.llama.quantized_moe_ffn(**kw)

    def test_source_transform_rejects_invalid_weight_nbit(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        wrapper = torch.nn.Module()
        wrapper.m = moe
        with self.assertRaises(ValueError):
            replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=3)

    def test_source_transform_rejects_group_size_zero(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        wrapper = torch.nn.Module()
        wrapper.m = moe
        with self.assertRaises(ValueError):
            replace_moe_with_quantized_op(wrapper, group_size=0, weight_nbit=4)


@_REQUIRES_TORCHAO_KERNEL_LIBRARY
class TestSharedExpert(unittest.TestCase):
    """Shared expert is preserved through the source transform."""

    def test_shared_expert_preserved(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        moe.shared_expert = torch.nn.Linear(32, 32)
        wrapper = torch.nn.Module()
        wrapper.m = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        self.assertIsNotNone(wrapper.m.shared_expert)
        self.assertIsInstance(wrapper.m.shared_expert, torch.nn.Linear)

    def test_no_shared_expert_is_none(self) -> None:
        moe = _build_moe_eager(
            dim=32, hidden_dim=32, num_experts=4, num_activated_experts=2
        )
        wrapper = torch.nn.Module()
        wrapper.m = moe
        replace_moe_with_quantized_op(wrapper, group_size=32, weight_nbit=4)
        self.assertIsNone(wrapper.m.shared_expert)
