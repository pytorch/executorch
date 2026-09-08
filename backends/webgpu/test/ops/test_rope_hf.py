# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace rotate-half rotary positional embedding
(`et_vk.apply_rotary_emb_hf`) export + goldens for the WebGPU backend.

Qwen3 (and other HF-derived models) use the rotate-half RoPE convention, which
fuses under VulkanPartitioner into `et_vk.apply_rotary_emb_hf.default` (a full
[max_seq, rotary_dim] freqs table + a start_pos offset, two outputs xq_out,
xk_out as a ValueList). This is the counterpart of test_rope.py for the
interleaved (Llama) convention. Full rotary only (rotary_dim == head_dim), which
is what Qwen3 uses; the runtime rejects partial rotary.

Inputs are deterministic /16 ramps so the native binary reconstructs them
bit-for-bit; the two torch-computed goldens are written for the native binary to
compare (it has no ATen).
"""

import os
import unittest
from collections import namedtuple

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import torch
from executorch.backends.vulkan import VulkanPartitioner
from executorch.examples.models.llama.rope import (
    hf_apply_rotary_emb,
    hf_precompute_freqs_cis,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.utils import get_delegates, get_non_lowered_nodes

# B batch, S tokens, NH query heads, NKV kv heads (NH != NKV so the two outputs
# are distinguishable by numel), HD head dim (even; full rotary, rotary_dim==HD).
Shape = namedtuple("Shape", ["name", "b", "s", "nh", "nkv", "hd"])
SHAPES = [
    Shape("multi", 1, 5, 8, 2, 64),
    # Single-token decode at a Qwen3-0.6B-like head config (GQA 16:8, head_dim
    # 128) so the seq=1 / batch decompositions are covered at decode too.
    Shape("decode", 1, 1, 16, 8, 128),
]

DYNAMIC_BATCH = 1
DYNAMIC_SEQ = 1
DYNAMIC_N_HEADS_Q = 16
DYNAMIC_N_HEADS_K = 8
DYNAMIC_HEAD_DIM = 128
DYNAMIC_MAX_SEQ = 16
DYNAMIC_POSITIONS = (0, 7, 15)
DYNAMIC_SEQUENCE_CASES = (
    (DYNAMIC_MAX_SEQ, 0),
    (5, 7),
    (1, DYNAMIC_MAX_SEQ - 1),
    (DYNAMIC_MAX_SEQ, 0),
)


class HfRope(torch.nn.Module):
    # unsqueeze_dim=1: freqs [S, HD] -> [S, 1, HD] broadcasts over (B, NH) of the
    # [B, S, NH, HD] q/k, matching the WebGPU kernel + HfRotaryEmbeddingPattern.
    def forward(self, xq, xk, freqs_cos, freqs_sin):
        return hf_apply_rotary_emb(xq, xk, freqs_cos, freqs_sin, unsqueeze_dim=1)


class DynamicHfRope(torch.nn.Module):
    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_pos = input_pos[0].item()
        torch._check_is_size(start_pos)
        torch._check(start_pos + xq.shape[1] <= freqs_cos.shape[0])
        return hf_apply_rotary_emb(
            xq,
            xk,
            freqs_cos.narrow(0, start_pos, xq.shape[1]),
            freqs_sin.narrow(0, start_pos, xq.shape[1]),
        )


def _ramp(numel: int, mod: int, off: int) -> torch.Tensor:
    # ((i % mod) - off) / 16: exact in fp32, matches test_webgpu_native.cpp.
    idx = torch.arange(numel, dtype=torch.int64)
    return ((idx % mod) - off).to(torch.float32) / 16.0


def _inputs(
    shape: Shape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xq = _ramp(shape.b * shape.s * shape.nh * shape.hd, 17, 8).reshape(
        shape.b, shape.s, shape.nh, shape.hd
    )
    xk = _ramp(shape.b * shape.s * shape.nkv * shape.hd, 13, 6).reshape(
        shape.b, shape.s, shape.nkv, shape.hd
    )
    # HF freqs are the FULL rotary_dim (== head_dim) table, not head_dim/2.
    freqs_cos = _ramp(shape.s * shape.hd, 11, 5).reshape(shape.s, shape.hd)
    freqs_sin = _ramp(shape.s * shape.hd, 7, 3).reshape(shape.s, shape.hd)
    return xq, xk, freqs_cos, freqs_sin


def _dynamic_inputs(seq: int = DYNAMIC_SEQ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    xq = _ramp(
        DYNAMIC_BATCH * seq * DYNAMIC_N_HEADS_Q * DYNAMIC_HEAD_DIM,
        17,
        8,
    ).reshape(DYNAMIC_BATCH, seq, DYNAMIC_N_HEADS_Q, DYNAMIC_HEAD_DIM)
    xk = _ramp(
        DYNAMIC_BATCH * seq * DYNAMIC_N_HEADS_K * DYNAMIC_HEAD_DIM,
        13,
        6,
    ).reshape(DYNAMIC_BATCH, seq, DYNAMIC_N_HEADS_K, DYNAMIC_HEAD_DIM)
    freqs_cos, freqs_sin = hf_precompute_freqs_cis(
        DYNAMIC_HEAD_DIM,
        DYNAMIC_MAX_SEQ,
        theta=10000.0,
    )
    input_pos = torch.tensor([0], dtype=torch.long)
    return xq, xk, freqs_cos, freqs_sin, input_pos


def _golden(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Reference = the registered et_vk op the kernel implements (start_pos=0).
    return torch.ops.et_vk.apply_rotary_emb_hf.default(xq, xk, freqs_cos, freqs_sin, 0)


def _dynamic_golden(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return hf_apply_rotary_emb(
        xq,
        xk,
        freqs_cos[position : position + xq.shape[1]],
        freqs_sin[position : position + xq.shape[1]],
    )


def _assert_fully_delegated(edge) -> None:
    graph = edge.exported_program().graph_module.graph
    delegates = get_delegates(graph)
    portable = get_non_lowered_nodes(graph)
    if len(delegates) != 1:
        raise AssertionError(f"expected one delegate, got {len(delegates)}")
    if portable:
        raise AssertionError(f"unexpected non-lowered nodes: {portable}")


def _lower(inputs):
    ep = torch.export.export(HfRope().eval(), inputs)
    edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
    _assert_fully_delegated(edge)
    return edge


def _export(inputs):
    return _lower(inputs).to_executorch()


def _lower_dynamic_program():
    inputs = _dynamic_inputs()
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        ep = torch.export.export(DynamicHfRope().eval(), inputs)

    symints = [
        node
        for node in ep.graph_module.graph.nodes
        if isinstance(node.meta.get("val"), torch.SymInt)
    ]
    if not symints:
        raise AssertionError("input_pos did not lower to a SymInt")

    edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
    _assert_fully_delegated(edge)
    return edge


def _lower_dynamic_sequence_program():
    inputs = _dynamic_inputs(DYNAMIC_MAX_SEQ)
    s_dim = torch.export.Dim("rope_hf_s", min=1, max=DYNAMIC_MAX_SEQ)
    dynamic_shapes = ({1: s_dim}, {1: s_dim}, None, None, None)
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        ep = torch.export.export(
            DynamicHfRope().eval(),
            inputs,
            dynamic_shapes=dynamic_shapes,
        )

    scalar_symints = [
        node
        for node in ep.graph_module.graph.nodes
        if isinstance(node.meta.get("val"), torch.SymInt)
    ]
    if not scalar_symints:
        raise AssertionError("input_pos did not lower to a SymInt")
    xq_placeholder = next(
        node
        for node in ep.graph_module.graph.nodes
        if node.op == "placeholder" and node.target == "xq"
    )
    if not isinstance(xq_placeholder.meta["val"].shape[1], torch.SymInt):
        raise AssertionError("query sequence dimension did not remain symbolic")

    edge = to_edge_transform_and_lower(ep, partitioner=[VulkanPartitioner()])
    _assert_fully_delegated(edge)
    return edge


def _export_dynamic_program():
    edge = _lower_dynamic_program()

    et = edge.to_executorch()
    delegate_ids = [
        delegate.id
        for plan in et.executorch_program.execution_plan
        for delegate in plan.delegates
    ]
    if delegate_ids != ["VulkanBackend"]:
        raise AssertionError(f"unexpected delegates: {delegate_ids}")
    return et


def _export_dynamic_sequence_program():
    edge = _lower_dynamic_sequence_program()
    et = edge.to_executorch()
    delegate_ids = [
        delegate.id
        for plan in et.executorch_program.execution_plan
        for delegate in plan.delegates
    ]
    if delegate_ids != ["VulkanBackend"]:
        raise AssertionError(f"unexpected delegates: {delegate_ids}")
    return et


class TestRopeHf(unittest.TestCase):
    def test_export_delegates(self) -> None:
        for shape in SHAPES:
            with self.subTest(shape=shape.name):
                self.assertIsNotNone(_lower(_inputs(shape)))

    def test_golden_matches_eager(self) -> None:
        # The et_vk golden must equal the real HF rotate-half apply_rotary_emb,
        # so a buggy golden can't fake-pass the native kernel. Run at both shapes
        # so the S=1 decode position indexing is covered.
        for shape in SHAPES:
            with self.subTest(shape=shape.name):
                xq, xk, fc, fs = _inputs(shape)
                gq, gk = _golden(xq, xk, fc, fs)
                eq, ek = hf_apply_rotary_emb(xq, xk, fc, fs, unsqueeze_dim=1)
                torch.testing.assert_close(gq, eq, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(gk, ek, atol=1e-5, rtol=1e-5)

    def test_dynamic_export_is_fully_delegated(self) -> None:
        self.assertIsNotNone(_lower_dynamic_program())

    def test_dynamic_position_goldens_match_custom_op(self) -> None:
        xq, xk, freqs_cos, freqs_sin, _ = _dynamic_inputs()
        self.assertNotEqual(xq.shape[2], xk.shape[2])
        position_outputs = []
        for position in DYNAMIC_POSITIONS:
            with self.subTest(position=position):
                expected_q, expected_k = _dynamic_golden(
                    xq, xk, freqs_cos, freqs_sin, position
                )
                position_outputs.append(expected_q)
                actual_q, actual_k = torch.ops.et_vk.apply_rotary_emb_hf.default(
                    xq, xk, freqs_cos, freqs_sin, position
                )
                torch.testing.assert_close(actual_q, expected_q)
                torch.testing.assert_close(actual_k, expected_k)
        self.assertFalse(torch.allclose(position_outputs[0], position_outputs[1]))
        self.assertFalse(torch.allclose(position_outputs[1], position_outputs[2]))

    def test_dynamic_sequence_export_is_fully_delegated(self) -> None:
        self.assertIsNotNone(_lower_dynamic_sequence_program())

    def test_dynamic_sequence_goldens_match_custom_op(self) -> None:
        _, _, freqs_cos, freqs_sin, _ = _dynamic_inputs(DYNAMIC_MAX_SEQ)
        for seq, position in dict.fromkeys(DYNAMIC_SEQUENCE_CASES):
            with self.subTest(seq=seq, position=position):
                xq, xk, _, _, _ = _dynamic_inputs(seq)
                expected_q, expected_k = _dynamic_golden(
                    xq, xk, freqs_cos, freqs_sin, position
                )
                actual_q, actual_k = torch.ops.et_vk.apply_rotary_emb_hf.default(
                    xq, xk, freqs_cos, freqs_sin, position
                )
                torch.testing.assert_close(actual_q, expected_q)
                torch.testing.assert_close(actual_k, expected_k)


def export_rope_hf_model(
    pte_path: str, xq_golden_path: str, xk_golden_path: str, shape_name: str = "multi"
) -> None:
    """Write the apply_rotary_emb_hf .pte + the xq_out and xk_out torch goldens
    (raw LE fp32). Inputs are /16 ramps reconstructed in the native test."""
    shape = next(s for s in SHAPES if s.name == shape_name)
    xq, xk, fc, fs = _inputs(shape)
    gq, gk = _golden(xq, xk, fc, fs)
    et = _export((xq, xk, fc, fs))
    with open(pte_path, "wb") as f:
        f.write(et.buffer)
    gq.detach().numpy().astype("<f4").tofile(xq_golden_path)
    gk.detach().numpy().astype("<f4").tofile(xk_golden_path)
    print(
        f"Exported {pte_path} (shape={shape_name}); xq_out golden {xq_golden_path} "
        f"({gq.numel()} floats); xk_out golden {xk_golden_path} ({gk.numel()} floats)"
    )


def export_rope_hf_dynamic(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    xq, xk, freqs_cos, freqs_sin, _ = _dynamic_inputs()
    et = _export_dynamic_program()
    with open(os.path.join(out_dir, "rope_hf_dynamic.pte"), "wb") as output:
        output.write(et.buffer)

    for name, tensor in (
        ("xq", xq),
        ("xk", xk),
        ("freqs_cos", freqs_cos),
        ("freqs_sin", freqs_sin),
    ):
        tensor.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"rope_hf_dynamic.{name}.bin")
        )

    for position in DYNAMIC_POSITIONS:
        gq, gk = _dynamic_golden(xq, xk, freqs_cos, freqs_sin, position)
        gq.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"rope_hf_dynamic.pos{position}.xq.golden.bin")
        )
        gk.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"rope_hf_dynamic.pos{position}.xk.golden.bin")
        )


def export_rope_hf_dynamic_sequence(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _, _, freqs_cos, freqs_sin, _ = _dynamic_inputs(DYNAMIC_MAX_SEQ)
    et = _export_dynamic_sequence_program()
    prefix = "rope_hf_dynamic_sequence"
    with open(os.path.join(out_dir, f"{prefix}.pte"), "wb") as output:
        output.write(et.buffer)

    for name, tensor in (("freqs_cos", freqs_cos), ("freqs_sin", freqs_sin)):
        tensor.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"{prefix}.{name}.bin")
        )

    for seq, position in dict.fromkeys(DYNAMIC_SEQUENCE_CASES):
        xq, xk, _, _, _ = _dynamic_inputs(seq)
        golden_q, golden_k = _dynamic_golden(xq, xk, freqs_cos, freqs_sin, position)
        case_prefix = os.path.join(out_dir, f"{prefix}.S{seq}.pos{position}")
        for name, tensor in (
            ("xq", xq),
            ("xk", xk),
            ("xq.golden", golden_q),
            ("xk.golden", golden_k),
        ):
            tensor.detach().numpy().astype("<f4").tofile(f"{case_prefix}.{name}.bin")


if __name__ == "__main__":
    unittest.main()
