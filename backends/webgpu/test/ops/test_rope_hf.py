# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic HuggingFace RoPE export and native-test goldens."""

import os
import unittest

import executorch.backends.vulkan.custom_ops_lib  # noqa: F401

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.examples.models.llama.rope import (
    hf_apply_rotary_emb,
    hf_precompute_freqs_cis,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.utils import get_delegates, get_non_lowered_nodes

BATCH = 1
SEQ = 1
N_HEADS_Q = 16
N_HEADS_K = 8
HEAD_DIM = 128
MAX_SEQ = 16
POSITIONS = (0, 7, 15)


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
    idx = torch.arange(numel, dtype=torch.int64)
    return ((idx % mod) - off).to(torch.float32) / 16.0


def _inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    xq = _ramp(BATCH * SEQ * N_HEADS_Q * HEAD_DIM, 17, 8).reshape(
        BATCH, SEQ, N_HEADS_Q, HEAD_DIM
    )
    xk = _ramp(BATCH * SEQ * N_HEADS_K * HEAD_DIM, 13, 6).reshape(
        BATCH, SEQ, N_HEADS_K, HEAD_DIM
    )
    freqs_cos, freqs_sin = hf_precompute_freqs_cis(
        HEAD_DIM,
        MAX_SEQ,
        theta=10000.0,
    )
    input_pos = torch.tensor([0], dtype=torch.long)
    return xq, xk, freqs_cos, freqs_sin, input_pos


def _golden(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return hf_apply_rotary_emb(
        xq,
        xk,
        freqs_cos[position : position + SEQ],
        freqs_sin[position : position + SEQ],
    )


def _export_program():
    inputs = _inputs()
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
    graph = edge.exported_program().graph_module.graph
    delegates = get_delegates(graph)
    portable = get_non_lowered_nodes(graph)
    if len(delegates) != 1:
        raise AssertionError(f"expected one delegate, got {len(delegates)}")
    if portable:
        raise AssertionError(f"unexpected non-lowered nodes: {portable}")

    et = edge.to_executorch()
    delegate_ids = [
        delegate.id
        for plan in et.executorch_program.execution_plan
        for delegate in plan.delegates
    ]
    if delegate_ids != ["VulkanBackend"]:
        raise AssertionError(f"unexpected delegates: {delegate_ids}")
    return et


def export_rope_hf_dynamic(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    xq, xk, freqs_cos, freqs_sin, _ = _inputs()
    et = _export_program()
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

    for position in POSITIONS:
        gq, gk = _golden(xq, xk, freqs_cos, freqs_sin, position)
        gq.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"rope_hf_dynamic.pos{position}.xq.golden.bin")
        )
        gk.detach().numpy().astype("<f4").tofile(
            os.path.join(out_dir, f"rope_hf_dynamic.pos{position}.xk.golden.bin")
        )


class TestRopeHf(unittest.TestCase):
    def test_dynamic_export_is_fully_delegated(self) -> None:
        self.assertIsNotNone(_export_program())

    def test_position_goldens_match_custom_op(self) -> None:
        xq, xk, freqs_cos, freqs_sin, _ = _inputs()
        self.assertNotEqual(xq.shape[2], xk.shape[2])
        position_outputs = []
        for position in POSITIONS:
            with self.subTest(position=position):
                expected_q, expected_k = _golden(xq, xk, freqs_cos, freqs_sin, position)
                position_outputs.append(expected_q)
                actual_q, actual_k = torch.ops.et_vk.apply_rotary_emb_hf.default(
                    xq, xk, freqs_cos, freqs_sin, position
                )
                torch.testing.assert_close(actual_q, expected_q)
                torch.testing.assert_close(actual_k, expected_k)
        self.assertFalse(torch.allclose(position_outputs[0], position_outputs[1]))
        self.assertFalse(torch.allclose(position_outputs[1], position_outputs[2]))


if __name__ == "__main__":
    unittest.main()
