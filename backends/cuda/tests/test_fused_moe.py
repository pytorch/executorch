# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export and validate fused_moe triton kernel on CUDA backend.

Usage:
  python -m pytest backends/cuda/tests/test_fused_moe.py -v

  # Standalone export (produces .pte + .ptd):
  python backends/cuda/tests/test_fused_moe.py --output-dir /tmp/exports
"""

import argparse
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.backends.cuda.triton.kernels.fused_moe import (
    fused_moe as triton_fused_moe,
    fused_moe_batched as triton_fused_moe_batched,
    moe_align_block_size,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from torch.export import export

EXECUTORCH_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
RUNNER_PATH = os.path.join(EXECUTORCH_ROOT, "cmake-out", "executor_runner")

# Test configurations: (seed, M, hidden, intermediate, num_experts, top_k, group_size)
TEST_CONFIGS = [
    (42, 1, 64, 32, 4, 2, 32, "basic_1tok_4exp_top2"),
    (0, 1, 64, 32, 8, 2, 32, "1tok_8exp_top2"),
    (7, 2, 64, 32, 4, 2, 32, "2tok_4exp_top2"),
    (99, 1, 128, 64, 4, 2, 32, "1tok_128d_4exp"),
    (13, 1, 64, 32, 4, 1, 32, "1tok_top1"),
    (55, 4, 64, 32, 8, 4, 32, "4tok_8exp_top4"),
    (77, 1, 128, 64, 8, 2, 64, "1tok_gs64"),
]


def _quantize_weights_int4(weight, group_size):
    """Quantize [E, N, K] bf16 weight to packed INT4 + scales."""
    E, N, K = weight.shape
    w = weight.float()
    w_grouped = w.reshape(E, N, K // group_size, group_size)
    amax = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = amax / 7.0
    int4_val = (w_grouped / scale).round().clamp(-8, 7).to(torch.int8)
    uint4_val = (int4_val + 8).to(torch.int16)
    uint4_flat = uint4_val.reshape(E, N, K)
    low = uint4_flat[:, :, 0::2]
    high = uint4_flat[:, :, 1::2]
    packed = (low | (high << 4)).to(torch.int8)
    scale_squeezed = scale.squeeze(-1).to(torch.bfloat16)
    return packed, scale_squeezed


def _reference_moe(hidden_states, w1_weight, w2_weight, topk_weights, topk_ids, top_k):
    """Reference MoE implementation in eager PyTorch (no quantization)."""
    M, K = hidden_states.shape
    num_pairs = M * top_k
    topk_ids_flat = topk_ids.reshape(-1)
    topk_weights_flat = topk_weights.reshape(-1)

    outputs = []
    for p in range(num_pairs):
        expert_id = topk_ids_flat[p].item()
        token_idx = p // top_k
        x = hidden_states[token_idx]

        # Gate+up
        gate_up = w1_weight[expert_id] @ x
        intermediate = gate_up.shape[0] // 2
        gate = gate_up[:intermediate]
        up = gate_up[intermediate:]
        activated = F.silu(gate) * up

        # Down
        down = w2_weight[expert_id] @ activated
        outputs.append(topk_weights_flat[p] * down)

    result = torch.stack(outputs).view(M, top_k, K).sum(dim=1)
    return result


class FusedMoEModel(nn.Module):
    """Model that calls triton::fused_moe for export testing."""

    def __init__(self, hidden, intermediate, num_experts, top_k, group_size):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.group_size = group_size
        self.router = nn.Linear(hidden, num_experts, bias=False)

        # Generate and quantize expert weights
        torch.manual_seed(0)
        w1_weight = torch.randn(
            num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16
        )
        w2_weight = torch.randn(num_experts, hidden, intermediate, dtype=torch.bfloat16)
        w1, w1_scale = _quantize_weights_int4(w1_weight, group_size)
        w2, w2_scale = _quantize_weights_int4(w2_weight, group_size)

        self.register_buffer("w1", w1)
        self.register_buffer("w1_scale", w1_scale)
        self.register_buffer("w2", w2)
        self.register_buffer("w2_scale", w2_scale)

    def forward(self, x):
        scores = self.router(x)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = weights.softmax(dim=-1).float()
        return triton_fused_moe(
            x,
            self.w1,
            self.w1_scale,
            self.w2,
            self.w2_scale,
            weights,
            indices,
            self.top_k,
            self.num_experts,
            self.group_size,
        )


def _make_inputs(seed, M, hidden, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(seed)
    return (torch.randn(M, hidden, dtype=dtype, device=device),)


def _save_tensor(t, path):
    t_cpu = t.cpu().contiguous()
    with open(path, "wb") as f:
        f.write(bytes(t_cpu.untyped_storage()))


def _load_output(path, shape, dtype):
    data = np.fromfile(path, dtype=np.uint8)
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def export_fused_moe(output_dir):
    """Export a FusedMoEModel to .pte + .ptd."""
    torch.manual_seed(42)
    model = (
        FusedMoEModel(hidden=64, intermediate=32, num_experts=4, top_k=2, group_size=32)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    inputs = _make_inputs(42, 1, 64)

    with torch.no_grad():
        ep = export(model, inputs, strict=True)

    os.makedirs(output_dir, exist_ok=True)

    specs = [CudaBackend.generate_method_name_compile_spec("forward")]
    et_prog = to_edge_transform_and_lower(
        ep,
        partitioner=[CudaPartitioner(specs)],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False, _skip_dim_order=True
        ),
    )
    et_program = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            do_quant_fusion_and_const_prop=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    pte_path = os.path.join(output_dir, "fused_moe.pte")
    with open(pte_path, "wb") as f:
        et_program.write_to_file(f)

    if hasattr(et_program, "_tensor_data") and et_program._tensor_data:
        et_program.write_tensor_data_to_file(output_dir)

    return pte_path, model


def _run_cpp_runner(runner_path, pte_path, ptd_path, input_files, output_base):
    cmd = [
        runner_path,
        f"--model_path={pte_path}",
        f"--data_path={ptd_path}",
        f"--inputs={','.join(input_files)}",
        f"--output_file={output_base}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


class TestFusedMoE(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_eager_shapes(self):
        """Triton fused_moe produces correct output shape and dtype."""
        M, K, intermediate, E, top_k, gs = 1, 64, 32, 4, 2, 32
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w1_weight = torch.randn(
            E, 2 * intermediate, K, dtype=torch.bfloat16, device="cuda"
        )
        w2_weight = torch.randn(E, K, intermediate, dtype=torch.bfloat16, device="cuda")
        w1, w1s = _quantize_weights_int4(w1_weight.cpu(), gs)
        w2, w2s = _quantize_weights_int4(w2_weight.cpu(), gs)
        w1, w1s = w1.cuda(), w1s.cuda()
        w2, w2s = w2.cuda(), w2s.cuda()

        topk_weights = torch.tensor([[0.6, 0.4]], dtype=torch.float32, device="cuda")
        topk_ids = torch.tensor([[0, 2]], dtype=torch.int64, device="cuda")

        out = triton_fused_moe(
            x, w1, w1s, w2, w2s, topk_weights, topk_ids, top_k, E, gs
        )
        self.assertEqual(out.shape, torch.Size([M, K]))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_eager_correctness(self):
        """Triton fused_moe matches reference across configs (with quantization error)."""
        for seed, M, hidden, intermediate, num_experts, top_k, gs, desc in TEST_CONFIGS:
            with self.subTest(desc=desc):
                torch.manual_seed(seed)
                x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
                w1_weight = torch.randn(
                    num_experts,
                    2 * intermediate,
                    hidden,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                w2_weight = torch.randn(
                    num_experts,
                    hidden,
                    intermediate,
                    dtype=torch.bfloat16,
                    device="cuda",
                )

                # Quantize
                w1, w1s = _quantize_weights_int4(w1_weight.cpu(), gs)
                w2, w2s = _quantize_weights_int4(w2_weight.cpu(), gs)
                w1, w1s = w1.cuda(), w1s.cuda()
                w2, w2s = w2.cuda(), w2s.cuda()

                # Router
                scores = torch.randn(M, num_experts, device="cuda")
                topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
                topk_weights = topk_weights.softmax(dim=-1).float()

                # Triton kernel
                out = triton_fused_moe(
                    x,
                    w1,
                    w1s,
                    w2,
                    w2s,
                    topk_weights,
                    topk_ids,
                    top_k,
                    num_experts,
                    gs,
                )

                # Reference (using same quantized weights via dequant)
                # Dequantize for reference
                w1_dq = _dequantize_int4(w1.cpu(), w1s.cpu(), gs).cuda()
                w2_dq = _dequantize_int4(w2.cpu(), w2s.cpu(), gs).cuda()
                ref = _reference_moe(
                    x,
                    w1_dq,
                    w2_dq,
                    topk_weights,
                    topk_ids,
                    top_k,
                )

                diff = (out.float() - ref.float()).abs().max().item()
                rel = diff / (ref.float().abs().max().item() + 1e-10)
                self.assertLess(
                    rel,
                    0.05,
                    f"{desc}: relative diff {rel:.4f} (abs {diff:.6f})",
                )

    def test_single_expert(self):
        """All tokens routed to same expert gives consistent results."""
        M, K, intermediate, E, gs = 2, 64, 32, 4, 32
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w1_weight = torch.randn(E, 2 * intermediate, K, dtype=torch.bfloat16)
        w2_weight = torch.randn(E, K, intermediate, dtype=torch.bfloat16)
        w1, w1s = _quantize_weights_int4(w1_weight, gs)
        w2, w2s = _quantize_weights_int4(w2_weight, gs)
        w1, w1s, w2, w2s = w1.cuda(), w1s.cuda(), w2.cuda(), w2s.cuda()

        # Both tokens to expert 1, top_k=1
        topk_weights = torch.ones(M, 1, dtype=torch.float32, device="cuda")
        topk_ids = torch.ones(M, 1, dtype=torch.int64, device="cuda")

        out = triton_fused_moe(x, w1, w1s, w2, w2s, topk_weights, topk_ids, 1, E, gs)

        # Each token should get same expert applied independently
        w1_dq = _dequantize_int4(w1.cpu(), w1s.cpu(), gs).cuda()
        w2_dq = _dequantize_int4(w2.cpu(), w2s.cpu(), gs).cuda()
        for t in range(M):
            gate_up = w1_dq[1] @ x[t]
            activated = F.silu(gate_up[:intermediate]) * gate_up[intermediate:]
            ref = w2_dq[1] @ activated
            diff = (out[t].float() - ref.float()).abs().max().item()
            rel = diff / (ref.float().abs().max().item() + 1e-10)
            self.assertLess(rel, 0.05, f"token {t}: relative diff {rel:.4f}")

    def test_batched_correctness(self):
        """Batched kernel matches reference across M values."""
        test_cases = [
            (42, 8, 64, 32, 4, 2, 32, "8tok_small"),
            (7, 16, 64, 32, 8, 4, 32, "16tok_8exp_top4"),
            (13, 32, 128, 64, 8, 2, 64, "32tok_gs64"),
            (55, 64, 64, 32, 4, 2, 32, "64tok"),
            (99, 128, 128, 64, 8, 2, 32, "128tok"),
        ]
        for seed, M, hidden, intermediate, num_experts, top_k, gs, desc in test_cases:
            with self.subTest(desc=desc):
                torch.manual_seed(seed)
                x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda")
                w1_weight = torch.randn(
                    num_experts,
                    2 * intermediate,
                    hidden,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                w2_weight = torch.randn(
                    num_experts,
                    hidden,
                    intermediate,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                w1, w1s = _quantize_weights_int4(w1_weight.cpu(), gs)
                w2, w2s = _quantize_weights_int4(w2_weight.cpu(), gs)
                w1, w1s, w2, w2s = w1.cuda(), w1s.cuda(), w2.cuda(), w2s.cuda()

                scores = torch.randn(M, num_experts, device="cuda")
                topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
                topk_weights = topk_weights.softmax(dim=-1).float()

                out = triton_fused_moe_batched(
                    x,
                    w1,
                    w1s,
                    w2,
                    w2s,
                    topk_weights,
                    topk_ids,
                    top_k,
                    num_experts,
                    gs,
                )

                w1_dq = _dequantize_int4(w1.cpu(), w1s.cpu(), gs).cuda()
                w2_dq = _dequantize_int4(w2.cpu(), w2s.cpu(), gs).cuda()
                ref = _reference_moe(x, w1_dq, w2_dq, topk_weights, topk_ids, top_k)

                diff = (out.float() - ref.float()).abs().max().item()
                rel = diff / (ref.float().abs().max().item() + 1e-10)
                self.assertLess(
                    rel,
                    0.05,
                    f"{desc}: relative diff {rel:.4f} (abs {diff:.6f})",
                )

    def test_batched_matches_fused(self):
        """Batched kernel matches the existing fused_moe kernel at Qwen-scale dims."""
        E, top_k, K, inter, gs = 256, 8, 2048, 512, 128
        torch.manual_seed(42)
        vals = torch.randint(0, 16, (E, 2 * inter, K), dtype=torch.uint8, device="cuda")
        w1 = ((vals[:, :, 1::2] << 4) | vals[:, :, 0::2]).to(torch.int8)
        w1s = (
            torch.randn(E, 2 * inter, K // gs, device="cuda", dtype=torch.bfloat16)
            * 0.01
        )
        vals = torch.randint(0, 16, (E, K, inter), dtype=torch.uint8, device="cuda")
        w2 = ((vals[:, :, 1::2] << 4) | vals[:, :, 0::2]).to(torch.int8)
        w2s = torch.randn(E, K, inter // gs, device="cuda", dtype=torch.bfloat16) * 0.01

        for M in [16, 64, 256]:
            with self.subTest(M=M):
                x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                logits = torch.randn(M, E, device="cuda", dtype=torch.float32)
                tw, ti = torch.topk(logits, top_k, dim=-1)
                tw = tw.softmax(dim=-1)
                ti = ti.to(torch.int64)

                out_fused = triton_fused_moe(x, w1, w1s, w2, w2s, tw, ti, top_k, E, gs)
                out_batched = triton_fused_moe_batched(
                    x, w1, w1s, w2, w2s, tw, ti, top_k, E, gs
                )

                err = (out_fused.float() - out_batched.float()).abs().max().item()
                self.assertLess(err, 0.5, f"M={M}: max abs error {err:.4e}")

    def test_export_cuda(self):
        """Export succeeds and produces non-empty .pte."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pte_path, _ = export_fused_moe(tmpdir)
            self.assertTrue(os.path.exists(pte_path))
            self.assertGreater(os.path.getsize(pte_path), 0)

    def test_e2e_cpp_runner(self):
        """Export once, run executor_runner with multiple inputs, compare."""
        self.assertTrue(
            os.path.exists(RUNNER_PATH),
            f"executor_runner not found at {RUNNER_PATH}. "
            "Build with: cmake --build cmake-out --target executor_runner",
        )

        M, hidden = 1, 64
        e2e_seeds = [0, 7, 42, 99, 123]

        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = os.path.join(tmpdir, "export")
            pte_path, model = export_fused_moe(export_dir)
            ptd_path = os.path.join(export_dir, "aoti_cuda_blob.ptd")

            for seed in e2e_seeds:
                with self.subTest(seed=seed):
                    inputs = _make_inputs(seed, M, hidden)

                    with torch.no_grad():
                        ref = model(*inputs)

                    run_dir = os.path.join(tmpdir, f"run_seed{seed}")
                    os.makedirs(run_dir)

                    input_files = []
                    for i, tensor in enumerate(inputs):
                        path = os.path.join(run_dir, f"{i}.bin")
                        _save_tensor(tensor, path)
                        input_files.append(path)

                    output_base = os.path.join(run_dir, "output")
                    result = _run_cpp_runner(
                        RUNNER_PATH, pte_path, ptd_path, input_files, output_base
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        f"seed={seed}: executor_runner failed:\n{result.stderr}",
                    )

                    cpp_out = _load_output(
                        f"{output_base}-0.bin", (M, hidden), torch.bfloat16
                    )

                    diff = (cpp_out.float() - ref.cpu().float()).abs().max().item()
                    ref_scale = ref.cpu().float().abs().max().item() + 1e-10
                    rel_diff = diff / ref_scale
                    self.assertLess(
                        rel_diff,
                        0.02,
                        f"seed={seed}: abs diff {diff}, rel diff {rel_diff:.4f}",
                    )


class TestMoeAlignBlockSize(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_basic_correctness(self):
        M, top_k, num_experts, block_size = 4, 2, 4, 4
        # Token 0 -> experts 0, 1
        # Token 1 -> experts 2, 3
        # Token 2 -> experts 0, 2
        # Token 3 -> experts 1, 3
        topk_ids = torch.tensor(
            [[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.int64, device="cuda"
        )
        num_pairs = M * top_k
        sentinel = num_pairs

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, num_experts
        )

        max_num_tokens_padded = num_pairs + num_experts * block_size
        max_num_expert_blocks = max_num_tokens_padded // block_size
        self.assertEqual(sorted_token_ids.shape[0], max_num_tokens_padded)
        self.assertEqual(expert_ids.shape[0], max_num_expert_blocks)

        # Each expert gets exactly 2 tokens, padded to block_size=4
        # So num_tokens_post_padded should be 4 * 4 = 16
        self.assertEqual(num_tokens_post_padded.item(), 16)

        # Verify tokens are grouped by expert within the active region
        flat_ids = topk_ids.reshape(-1)
        active = sorted_token_ids[: num_tokens_post_padded.item()]
        for block_idx in range(num_tokens_post_padded.item() // block_size):
            expert = expert_ids[block_idx].item()
            block = active[block_idx * block_size : (block_idx + 1) * block_size]
            for pair_id in block.tolist():
                if pair_id == sentinel:
                    continue
                self.assertEqual(
                    flat_ids[pair_id].item(),
                    expert,
                    f"pair {pair_id} expected expert {expert}, got {flat_ids[pair_id].item()}",
                )

    def test_all_tokens_same_expert(self):
        M, top_k, num_experts, block_size = 4, 2, 4, 4
        topk_ids = torch.full((M, top_k), 2, dtype=torch.int64, device="cuda")
        num_pairs = M * top_k
        sentinel = num_pairs

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, num_experts
        )

        # All 8 pairs go to expert 2, padded to block_size=4 -> 8 slots
        self.assertEqual(num_tokens_post_padded.item(), 8)

        active = sorted_token_ids[: num_tokens_post_padded.item()]
        real_ids = active[active != sentinel]
        self.assertEqual(real_ids.shape[0], num_pairs)
        self.assertTrue(
            (sorted(real_ids.tolist()) == list(range(num_pairs))),
            "All pair indices should appear exactly once",
        )

    def test_single_token(self):
        num_experts, block_size = 4, 4
        topk_ids = torch.tensor([[2]], dtype=torch.int64, device="cuda")
        num_pairs = 1
        sentinel = num_pairs

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, num_experts
        )

        # 1 token to expert 2, padded to block_size=4
        self.assertEqual(num_tokens_post_padded.item(), block_size)

        active = sorted_token_ids[: num_tokens_post_padded.item()]
        real_ids = active[active != sentinel].tolist()
        self.assertEqual(real_ids, [0])
        sentinel_count = (active == sentinel).sum().item()
        self.assertEqual(sentinel_count, block_size - 1)

    def test_num_pairs_less_than_block_size(self):
        M, top_k, num_experts, block_size = 1, 2, 4, 16
        topk_ids = torch.tensor([[0, 3]], dtype=torch.int64, device="cuda")
        num_pairs = M * top_k
        sentinel = num_pairs

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, num_experts
        )

        # 1 token per expert -> each padded to block_size=16, total=32
        self.assertEqual(num_tokens_post_padded.item(), 2 * block_size)

        active = sorted_token_ids[: num_tokens_post_padded.item()]
        real_ids = sorted(active[active != sentinel].tolist())
        self.assertEqual(real_ids, [0, 1])

    def test_sentinel_value(self):
        M, top_k, num_experts, block_size = 2, 2, 4, 4
        topk_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device="cuda")
        num_pairs = M * top_k
        sentinel = num_pairs

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, block_size, num_experts
        )

        # Padding positions within the active region use sentinel = num_pairs
        active = sorted_token_ids[: num_tokens_post_padded.item()]
        for val in active.tolist():
            self.assertTrue(
                0 <= val <= sentinel,
                f"Value {val} outside valid range [0, {sentinel}]",
            )

        # Tail beyond active region should also be sentinel
        tail = sorted_token_ids[num_tokens_post_padded.item() :]
        self.assertTrue((tail == sentinel).all())

    def test_determinism(self):
        M, top_k, num_experts, block_size = 8, 4, 8, 4
        torch.manual_seed(42)
        topk_ids = torch.randint(0, num_experts, (M, top_k), device="cuda")

        results = [
            moe_align_block_size(topk_ids, block_size, num_experts) for _ in range(5)
        ]
        for i in range(1, len(results)):
            self.assertTrue(torch.equal(results[0][0], results[i][0]))
            self.assertTrue(torch.equal(results[0][1], results[i][1]))
            self.assertEqual(results[0][2].item(), results[i][2].item())


def _dequantize_int4(packed, scale, group_size):
    """Dequantize packed INT4 [E, N, K//2] back to [E, N, K] float."""
    E, N, K_half = packed.shape
    K = K_half * 2
    packed_uint8 = packed.to(torch.int16) & 0xFF
    low = packed_uint8 & 0xF
    high = (packed_uint8 >> 4) & 0xF
    # Interleave: [E, N, K]
    uint4 = torch.stack([low, high], dim=-1).reshape(E, N, K)
    int4 = uint4.float() - 8.0
    # Apply scales: scale is [E, N, K // group_size]
    scale_expanded = (
        scale.float().unsqueeze(-1).expand(E, N, K // group_size, group_size)
    )
    scale_flat = scale_expanded.reshape(E, N, K)
    return (int4 * scale_flat).to(torch.bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args, remaining = parser.parse_known_args()

    if args.output_dir:
        export_fused_moe(args.output_dir)
    else:
        sys.argv = [sys.argv[0]] + remaining
        unittest.main()
