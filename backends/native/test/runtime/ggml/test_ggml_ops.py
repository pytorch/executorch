# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export single operators and a tiny Llama to .ptn and check the ggml engine
against eager PyTorch. GGML_PTN_DEVICE selects the runner's device (cpu or
gpu)."""

import collections
import contextlib
import copy
import os
import re
import subprocess
import tempfile
import unittest
from importlib.resources import as_file, files
from pathlib import Path

import torch
import torch.nn.functional as F
from executorch.backends.native.custom_ops import gated_delta_rule_op
from executorch.backends.native.examples.hf_llm.export_hf_llm import (
    export_prefill,
    LastTokenLM,
    native_gdn_rules,
)
from executorch.exir.native import to_native
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
from torch.export import Dim
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    MappingType,
    quantize_,
)
from torchao.quantization.granularity import PerGroup
from transformers import (
    AutoModelForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)


# Tests that turn an engine testing option on run only when asked for.
requires_option_tests = unittest.skipUnless(
    os.environ.get("NATIVE_GGML_OPTION_TESTS") == "1",
    "engine testing option; set NATIVE_GGML_OPTION_TESTS=1 to run",
)


class Lambda(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class Linear(torch.nn.Module):
    def __init__(self, bias: bool):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=bias)

    def forward(self, x):
        return self.linear(x)


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 8)

    def forward(self, indices):
        return self.embedding(indices)


class HalfEmbeddingCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 8).half()

    def forward(self, x, indices):
        return torch.cat([x, self.embedding(indices)], dim=-1)


class RMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(16) + 0.5)

    def forward(self, x):
        return F.rms_norm(x, (16,), self.weight, eps=1e-6)


class Rope(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.rotary = LlamaRotaryEmbedding(config)

    def forward(self, x, position_ids):
        cos, sin = self.rotary(x, position_ids)
        q, _ = apply_rotary_pos_emb(x, x, cos, sin)
        return q


class RopeOfPermute(torch.nn.Module):
    """Rotates `x` permuted by `dims` to [B, H, T, D], then transposes the
    result to [B, T, H, D]."""

    def __init__(self, config: LlamaConfig, dims):
        super().__init__()
        self.rope = Rope(config)
        self.dims = dims

    def forward(self, x, position_ids):
        return self.rope(x.permute(*self.dims), position_ids).transpose(1, 2)


def batch_norm(channels: int) -> torch.nn.BatchNorm2d:
    norm = torch.nn.BatchNorm2d(channels)
    norm.running_mean.uniform_(-1.0, 1.0)
    norm.running_var.uniform_(0.5, 2.0)
    norm.weight.data.uniform_(0.5, 1.5)
    norm.bias.data.uniform_(-0.5, 0.5)
    return norm


class RopeQK(torch.nn.Module):
    """Rotates q and k at positions computed in the graph."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.rotary = LlamaRotaryEmbedding(config)

    def forward(self, q, k, start):
        position_ids = (torch.arange(q.shape[2]) + start).unsqueeze(0)
        cos, sin = self.rotary(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q + k


class CastIndexCacheWrite(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 2, 8, 4))

    def forward(self, values, positions):
        self.cache.index_copy_(2, positions.to(torch.int32).long(), values)
        return self.cache * 1.0


class SnapshotThenWrite(torch.nn.Module):
    """Reads the cache, then writes it in place: the read sees the old rows."""

    def __init__(self, clone: bool):
        super().__init__()
        self.clone = clone
        self.register_buffer("cache", torch.zeros(1, 2, 4, 2))

    def forward(self, values, positions):
        old = self.cache.clone() if self.clone else self.cache * 1.0
        self.cache.index_copy_(2, positions, values)
        return old


class ConvBN(torch.nn.Module):
    def __init__(self, groups: int, bias: bool, kernel_size: int = 3):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            8,
            8,
            kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias,
        )
        self.bn = batch_norm(8)
        # Exports as hardtanh_, as in torchvision's MobileNetV2.
        self.relu6 = torch.nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu6(self.bn(self.conv(x)))


class SharedConvBN(torch.nn.Module):
    """Batch norms reading a shared convolution output, a convolution with
    shared weights, and a ReLU."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1)
        self.bn_shared_output = batch_norm(4)
        self.bn_shared_weight = batch_norm(4)
        self.bn_relu = batch_norm(4)

    def forward(self, x):
        y = self.conv(x)
        z = self.conv(torch.relu(x))
        return (
            self.bn_shared_output(y) + y,
            self.bn_shared_weight(z) + self.bn_relu(torch.relu(x)),
        )


class CacheWrite(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 2, 8, 4))

    def forward(self, values, positions):
        self.cache.index_copy_(2, positions, values)
        return self.cache * 1.0


class SharedCacheWrite(torch.nn.Module):
    """A method writing rows of a cache other methods share."""

    def __init__(self, cache, persistent):
        super().__init__()
        self.register_buffer("cache", cache, persistent=persistent)

    def forward(self, values, positions):
        self.cache.index_copy_(2, positions, values)
        return self.cache * 1.0


class SharedCacheRead(torch.nn.Module):
    """A method reading a cache other methods write."""

    def __init__(self, cache, persistent):
        super().__init__()
        self.register_buffer("cache", cache, persistent=persistent)

    def forward(self, x):
        return self.cache + x


class WindowWriteBack(torch.nn.Module):
    """A causal conv's state update: the buffer becomes the last columns of
    itself concatenated with new ones."""

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.randn(1, 6, 4))

    def forward(self, x):
        window = torch.cat([self.state, x], dim=-1)
        self.state.copy_(window[..., -4:])
        return window * 2.0


class TransposedCacheWrite(torch.nn.Module):
    """Writes [B, T, H, D] values into a head-major cache through a transpose."""

    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 2, 8, 4))

    def forward(self, values, positions):
        self.cache.index_copy_(2, positions, values.transpose(1, 2))
        return self.cache * 1.0


class CachedAttention(torch.nn.Module):
    """Writes each call's keys and values into zero-initialized F32 caches and
    attends every written position, as the HF exporter's static-cache
    attention does."""

    def __init__(self):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(1, 2, 16, 64), persistent=False)
        self.register_buffer("v_cache", torch.zeros(1, 2, 16, 64), persistent=False)

    def forward(self, q, k, v, positions):
        self.k_cache.index_copy_(2, positions, k)
        self.v_cache.index_copy_(2, positions, v)
        mask = torch.arange(16).le(positions[:, None])
        return F.scaled_dot_product_attention(
            q, self.k_cache, self.v_cache, attn_mask=mask
        )


class HalfCacheWrite(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 2, 8, 4, dtype=torch.float16))

    def forward(self, values, positions):
        self.cache.index_copy_(2, positions, values.to(torch.float16))
        return self.cache * 1.0


class TinyLlama(torch.nn.Module):
    """A tiny 8da4w Llama behind the HF exporter's static-cache wrapper, which
    returns only the last position's logits."""

    def __init__(self, max_cache_len: int):
        super().__init__()
        config = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=512,
            rope_theta=500000.0,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
            },
            initializer_range=0.1,
            tie_word_embeddings=True,
        )
        model = LlamaForCausalLM(config).eval()
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        _quantize_8da4w(model, group_size=64)
        self.lm = LastTokenLM(model, max_cache_len)

    def forward(self, input_ids, cache_position):
        return self.lm(input_ids, cache_position)


class WrittenCacheMask(torch.nn.Module):
    """Writes keys into a cache at `cache_position`, or at `rows` if given,
    then masks it as transformers 5 does, with query positions arange(T) + the
    first position."""

    def __init__(self, cache_len: int):
        super().__init__()
        self.register_buffer("keys", torch.zeros(1, 2, cache_len, 32), persistent=False)
        self.register_buffer("values", torch.randn(1, 2, cache_len, 32))

    def forward(self, q, k, cache_position, rows=None):
        self.keys.index_copy_(2, cache_position if rows is None else rows, k)
        positions = (torch.arange(q.shape[2]) + cache_position[:1]).view(-1, 1)
        mask = torch.arange(self.keys.shape[2]) + 0 <= positions
        return F.scaled_dot_product_attention(q, self.keys, self.values, attn_mask=mask)


class LongCacheMask(torch.nn.Module):
    """Attention over a static cache with the mask transformers 5 builds,
    `arange(kv_length) + kv_offset <= positions`, optionally with a sliding
    window: over 2^16 keys, more than the host computes exactly."""

    def __init__(self, cache_len: int, window: int | None = None):
        super().__init__()
        self.window = window
        self.register_buffer("keys", torch.randn(1, 2, cache_len, 32))
        self.register_buffer("values", torch.randn(1, 2, cache_len, 32))

    def forward(self, q, cache_position):
        kv_arange = torch.arange(self.keys.shape[2]) + 0
        positions = cache_position.view(-1, 1)
        mask = kv_arange <= positions
        if self.window is not None:
            mask = mask * ((positions - self.window) < kv_arange)
        return F.scaled_dot_product_attention(q, self.keys, self.values, attn_mask=mask)


class SlicedQueryPositions(torch.nn.Module):
    """Attention over a static cache whose query positions are one sliced cache
    position plus an arange, the form transformers 5 builds its causal mask
    from: the first position, or with `last` the last one, whose slice starts
    at a computed SymInt."""

    def __init__(self, cache_len: int, last: bool = False):
        super().__init__()
        self.last = last
        self.register_buffer("keys", torch.randn(1, 2, cache_len, 32))
        self.register_buffer("values", torch.randn(1, 2, cache_len, 32))

    def forward(self, q, cache_position):
        count = q.shape[2]
        offset = cache_position[count - 1 : count] if self.last else cache_position[:1]
        positions = torch.arange(count) + offset
        mask = torch.arange(self.keys.shape[2]) <= positions[:, None]
        return F.scaled_dot_product_attention(q, self.keys, self.values, attn_mask=mask)


class BatchedExperts(torch.nn.Module):
    """A top-k mixture of experts in transformers' batched_mm form: each token
    repeated once per chosen expert, the experts' weights gathered by id, and
    one bmm per projection. `transposed` stores weights as [E, I, O], which
    multiplies the activations by the gathered weights instead."""

    def __init__(
        self, experts=4, k=2, hidden=64, inter=32, transposed=False, shared=False
    ):
        super().__init__()
        self.k, self.transposed, self.shared = k, transposed, shared
        self.router = torch.nn.Linear(hidden, experts, bias=False)
        gate_up, down = (experts, 2 * inter, hidden), (experts, hidden, inter)
        if transposed:
            gate_up, down = (experts, hidden, 2 * inter), (experts, inter, hidden)
        self.gate_up = torch.nn.Parameter(torch.randn(gate_up) / hidden**0.5)
        self.down = torch.nn.Parameter(torch.randn(down) / inter**0.5)

    def _project(self, x, weight):
        if self.transposed:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1)
        return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1)

    def forward(self, x):
        tokens, hidden = x.shape
        probs = torch.softmax(self.router(x), dim=-1)
        weights, ids = torch.topk(probs, self.k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).reshape(-1)
        ids = ids.reshape(-1)
        selected = self.gate_up[ids]
        gate, up = self._project(x.repeat_interleave(self.k, dim=0), selected).chunk(
            2, dim=-1
        )
        out = self._project(F.silu(gate) * up, self.down[ids])
        out = (out * weights.unsqueeze(-1)).view(tokens, self.k, hidden).sum(1)
        # The gathered gate_up weights also read by another op.
        return out + selected.sum() if self.shared else out


class ExpertMatmul(torch.nn.Module):
    """y[s] = experts[ids[s]] @ x[s], with the ids an input."""

    def __init__(self, experts, out, inp, dtype):
        super().__init__()
        self.experts = torch.nn.Parameter(
            (torch.randn(experts, out, inp) / inp**0.5).to(dtype)
        )

    def forward(self, x, ids):
        return torch.bmm(self.experts[ids], x.unsqueeze(-1)).squeeze(-1)


def _quantize_8da4w(
    model: torch.nn.Module,
    group_size: int,
    embedding_mapping: MappingType = MappingType.SYMMETRIC,
) -> None:
    quantize_(
        model,
        IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(group_size),
            mapping_type=embedding_mapping,
        ),
        filter_fn=lambda m, _: isinstance(m, torch.nn.Embedding),
    )
    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4, weight_granularity=PerGroup(group_size)
        ),
        filter_fn=lambda m, _: isinstance(m, torch.nn.Linear),
    )


def _shape_text(inputs) -> str:
    return ",".join("x".join(str(d) for d in t.shape) for t in inputs)


def _run_ggml(  # noqa: C901
    package: Path,
    steps,
    outdir: Path,
    num_outputs: int,
    resize: bool = False,
    max_cached_graphs: int = 1,
    cuda_f32_accumulation_for_testing: bool = False,
    graph_ops: collections.Counter | None = None,
    method: str = "forward",
    allow_activation_quant_for_weight_only: bool = False,
    exact_topk_for_testing: bool = False,
    exact_gelu_for_testing: bool = False,
    f32_gpu_kv_cache_for_testing: bool = False,
    fold_constants: bool = True,
    constant_bytes: list[int] | None = None,
    memory: list[tuple[str, int, int]] | None = None,
):
    """Runs each step's inputs in order on one executable and returns each
    step's outputs as (bytes, shape) pairs. `graph_ops`, if given, collects
    the built graphs' node counts per ggml op; `constant_bytes`, the context's
    constant bytes after each method's compile; `memory`, the context's compute
    buffer bytes and allocations after each compile and at exit."""
    input_steps, output_steps = [], []
    for step, inputs in enumerate(steps):
        input_paths = []
        for i, tensor in enumerate(inputs):
            path = outdir / f"step{step}_input{i}.bin"
            path.write_bytes(tensor.contiguous().flatten().view(torch.uint8).numpy())
            input_paths.append(str(path))
        input_steps.append(",".join(input_paths))
        output_steps.append(
            ",".join(
                str(outdir / f"step{step}_output{i}.bin") for i in range(num_outputs)
            )
        )
    command = [
        f"--device={os.environ.get('GGML_PTN_DEVICE', 'cpu')}",
        f"--package={package}",
        f"--method={method}",
        f"--inputs={';'.join(input_steps)}",
        f"--outputs={';'.join(output_steps)}",
        f"--max_cached_graphs={max_cached_graphs}",
    ]
    if resize:
        command.append(f"--input_sizes={';'.join(map(_shape_text, steps))}")
    if cuda_f32_accumulation_for_testing:
        command.append("--cuda_f32_accumulation_for_testing")
    if graph_ops is not None:
        command.append("--print_graph_ops")
    if allow_activation_quant_for_weight_only:
        command.append("--allow_activation_quant_for_weight_only")
    if exact_topk_for_testing:
        command.append("--exact_topk_for_testing")
    if exact_gelu_for_testing:
        command.append("--exact_gelu_for_testing")
    if f32_gpu_kv_cache_for_testing:
        command.append("--f32_gpu_kv_cache_for_testing")
    if not fold_constants:
        command.append("--nofold_constants")
    if constant_bytes is not None or memory is not None:
        command.append("--print_memory")
    with as_file(files(__package__) / "ggml_ptn_runner") as runner:
        result = subprocess.run(
            [str(runner), *command],
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        raise AssertionError(f"ggml_ptn_runner failed:\n{result.stderr}")
    for line in result.stderr.splitlines() if graph_ops is not None else []:
        if line.startswith("ggml host integer checks per run:"):
            graph_ops["host integer checks"] += int(line.split()[-1])
        elif line.startswith("ggml graph ops:"):
            graph_ops.update(
                {op: int(n) for op, n in (item.split("=") for item in line.split()[3:])}
            )
    for line in result.stderr.splitlines() if memory is not None else []:
        if match := re.match(
            r"ggml memory (.+): constants \d+ bytes, compute (\d+) bytes in (\d+)",
            line,
        ):
            memory.append((match.group(1), int(match.group(2)), int(match.group(3))))
    for line in result.stderr.splitlines() if constant_bytes is not None else []:
        if match := re.match(r"ggml memory after compiling \S+: constants (\d+)", line):
            constant_bytes.append(int(match.group(1)))
    shapes = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    paths = [path for step in output_steps for path in step.split(",")]
    outputs = [(Path(path).read_bytes(), shape) for path, shape in zip(paths, shapes)]
    return [outputs[i : i + num_outputs] for i in range(0, len(outputs), num_outputs)]


def _assert_same_as_unfolded(test, folded, unfolded):
    """Float32 outputs computed with constants folded are bitwise those of an
    unfolded run on the CPU. GPU backends fuse a graph's ops by its shape, and
    a folded constant changes that shape, which can move a result by an ulp."""
    if os.environ.get("GGML_PTN_DEVICE", "cpu") == "cpu":
        test.assertEqual(folded, unfolded)
        return
    for step_folded, step_unfolded in zip(folded, unfolded, strict=True):
        for (raw, shape), (raw_unfolded, shape_unfolded) in zip(
            step_folded, step_unfolded, strict=True
        ):
            test.assertEqual(shape, shape_unfolded)
            torch.testing.assert_close(
                torch.frombuffer(bytearray(raw), dtype=torch.float32),
                torch.frombuffer(bytearray(raw_unfolded), dtype=torch.float32),
                rtol=1e-5,
                atol=1e-6,
            )


def _hip_accumulates_half_gemms_in_f32() -> bool:
    """ggml-hip accumulates F16 GEMMs in F32 on CDNA and RDNA4 GPUs only."""
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except (AssertionError, AttributeError, RuntimeError):
        return False
    return re.match(r"gfx(908|90a|94\d|95\d|12\d\d)", arch) is not None


# ggml-cuda runs F32 cuBLAS matmuls in TF32, with a 10-bit mantissa.
_MATMUL_TOL = (
    {"atol": 5e-3, "rtol": 1e-2}
    if os.environ.get("GGML_PTN_DEVICE", "cpu") == "gpu"
    else {"atol": 1e-4, "rtol": 1e-5}
)

# GGUF quant types the random-weight tests cover, with the fp16 super-block
# scales of their random blocks by byte offset in a block. Each min scale
# centers the weights near zero.
_GGUF_SUPER_SCALES = {
    "q4_k": {0: 2**-8, 2: 2**-5},
    "q4_0": {0: 2**-4},
    "q4_1": {0: 2**-4, 2: -15 * 2**-5},
    "q5_0": {0: 2**-5},
    "q5_1": {0: 2**-5, 2: -15 * 2**-5},
    "q8_0": {0: 2**-8},
    "q2_k": {80: 2**-6, 82: 3 * 2**-7},
    "q3_k": {108: 2**-6},
    "q5_k": {0: 2**-8, 2: 2**-4},
    "q6_k": {208: 2**-14},
}


# The ggml backend under test: cpu, cuda or hip.
_BACKEND = os.environ.get("GGML_PTN_BACKEND", "cpu")

# ggml-cuda is built with -use_fast_math, whose tanhf is a few 1e-6 off.
_GELU_TOL = (
    {"atol": 1e-5, "rtol": 1e-5} if _BACKEND == "cuda" else {"atol": 1e-6, "rtol": 1e-5}
)


def _half_linear(weight: list[list[float]]) -> torch.nn.Linear:
    linear = torch.nn.Linear(len(weight[0]), len(weight), bias=False).half()
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(weight))
    return linear.requires_grad_(False)


class GgmlOpsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    @staticmethod
    def _q6_k_weights(rows):
        # Valid Q6_K blocks: 128 low-bit, 64 high-bit, 16 signed-scale
        # bytes, then an FP16 super-block scale of 1.0 (0x3c00).
        raw = torch.zeros(rows, 210, dtype=torch.uint8)
        raw[:, :128] = torch.arange(128, dtype=torch.uint8)
        raw[:, 128:192] = torch.arange(64, dtype=torch.uint8)
        raw[:, 192:208] = 1
        raw[:, 209] = 0x3C
        return ExportableGGUFTensor.from_raw(raw, "q6_k", torch.float32)

    @staticmethod
    def _q5_k_weights(rows, cols):
        # Random Q5_K blocks under an FP16 super-block scale of 2^-10 and min
        # of 2^-6, so sub-block scales, mins and fifth bits vary by row and
        # block.
        raw = torch.randint(0, 256, (rows, cols // 256, 176), dtype=torch.uint8)
        raw[..., 0:2] = torch.tensor([2**-10], dtype=torch.float16).view(torch.uint8)
        raw[..., 2:4] = torch.tensor([2**-6], dtype=torch.float16).view(torch.uint8)
        return ExportableGGUFTensor.from_raw(
            raw.reshape(rows, -1), "q5_k", torch.float32
        )

    def test_gguf_q5_k_linear(self):
        self._check_gguf_linear(self._q5_k_weights(8, 512))

    def test_gguf_q6_k_linear(self):
        self._check_gguf_linear(self._q6_k_weights(8))

    def _check_gguf_linear(self, weight):
        rows, cols = weight.shape
        model = torch.nn.Linear(cols, rows, bias=False)
        model.weight = torch.nn.Parameter(weight, requires_grad=False)
        representable = torch.full((2, cols), 0.125)
        representable[1, ::2] = -0.125
        # GGML requantizes F32 activations before the dot product: to Q8_K on
        # CPU, which these values survive exactly, and to Q8_1 with an FP16
        # block scale on GPUs. Test the weight layout tightly with exact
        # activations, then separately bound the randomized requantization loss.
        gpu = os.environ.get("GGML_PTN_DEVICE", "cpu") == "gpu"
        self._check(
            model,
            (representable,),
            atol=5e-4,
            rtol=2e-4 if gpu else 1e-5,
            allow_activation_quant_for_weight_only=True,
        )
        self._check(
            model,
            (torch.randn(2, cols) / 8,),
            atol=0.35,
            rtol=0.02,
            allow_activation_quant_for_weight_only=True,
        )

    def test_gguf_q6_k_embedding(self):
        model = torch.nn.Embedding(8, 256)
        model.weight = torch.nn.Parameter(self._q6_k_weights(8), requires_grad=False)
        self._check(model, (torch.tensor([[0, 3, 6], [7, 1, 4]]),))

    @staticmethod
    def _random_gguf_weights(ggml_type, rows, cols):
        # Random blocks, with each fp16 super-block scale set at its byte
        # offset so that the weights are finite and of moderate size.
        qtype = GGMLQuantizationType[ggml_type.upper()]
        block, block_bytes = GGML_QUANT_SIZES[qtype]
        raw = torch.randint(
            0, 256, (rows, cols // block, block_bytes), dtype=torch.uint8
        )
        for offset, scale in _GGUF_SUPER_SCALES[ggml_type].items():
            raw[..., offset : offset + 2] = torch.tensor(
                [scale], dtype=torch.float16
            ).view(torch.uint8)
        return ExportableGGUFTensor.from_raw(
            raw.reshape(rows, -1), ggml_type, torch.float32
        )

    def _check_random_gguf_linear(self, ggml_type):
        weight = self._random_gguf_weights(ggml_type, 8, 512)
        model = torch.nn.Linear(512, 8, bias=False)
        model.weight = torch.nn.Parameter(weight, requires_grad=False)
        # Activations of +-127/1024 survive every int8 requantization ggml
        # does exactly: each block's scale, max|x| / 127 = 2^-10, is exact even
        # in fp16, and so are the block sums some kernels keep. The GPU Q4_1 and
        # Q5_1 kernels multiply those by the weight scales in fp16, which the
        # scales above keep exact too.
        representable = torch.full((2, 512), 127 / 1024)
        representable[1, ::2] *= -1
        self._check(
            model,
            (representable,),
            atol=5e-4,
            rtol=1e-5,
            allow_activation_quant_for_weight_only=True,
        )
        # ggml rounds each block of activations to int8 in steps of at most
        # max|x| / 127. Modeling the rounding errors as independent and
        # uniform, an output's error has a standard deviation of
        # step / sqrt(12) times its weight row's norm; allow five. The model
        # is rough on GPUs, whose kernels for some types take a block's min
        # term from the sum of its unquantized activations.
        x = torch.randn(2, 512) / 8
        rounding = x.abs().max() / 127 / 12**0.5
        row_norm = weight.dequantize(torch.float32).norm(dim=1).max()
        self._check(
            model,
            (x,),
            atol=float(5 * rounding * row_norm),
            rtol=0,
            allow_activation_quant_for_weight_only=True,
        )

    def _check_random_gguf_embedding(self, ggml_type):
        model = torch.nn.Embedding(8, 512)
        model.weight = torch.nn.Parameter(
            self._random_gguf_weights(ggml_type, 8, 512), requires_grad=False
        )
        self._check(model, (torch.tensor([[0, 3, 6], [7, 1, 4]]),))

    def _check_random_gguf_dynamic_tokens(self, ggml_type):
        # A linear and an embedding, each exported once with a dynamic token
        # count and run at 9, 1 and 33 tokens on one executable; GPUs switch
        # matmul kernels above 8 rows. The linear gets the exact activations
        # of _check_random_gguf_linear, alternating by token.
        linear = torch.nn.Linear(512, 8, bias=False)
        linear.weight = torch.nn.Parameter(
            self._random_gguf_weights(ggml_type, 8, 512), requires_grad=False
        )
        embedding = torch.nn.Embedding(8, 512)
        embedding.weight = torch.nn.Parameter(
            self._random_gguf_weights(ggml_type, 8, 512), requires_grad=False
        )
        representable = torch.full((2, 512), 127 / 1024)
        representable[1, ::2] *= -1
        lengths = (9, 1, 33)
        cases = {
            "linear": (linear, [representable[torch.arange(n) % 2] for n in lengths]),
            "embedding": (embedding, [torch.arange(n) % 8 for n in lengths]),
        }
        tokens = Dim("tokens", min=1, max=64)
        for name, (model, inputs) in cases.items():
            steps = [(x,) for x in inputs]
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                ep = torch.export.export(
                    model, steps[0], dynamic_shapes={"input": {0: tokens}}
                )
                outdir = Path(tmp)
                package = outdir / "model.ptn"
                to_native(ep).save(str(package))
                actual = _run_ggml(
                    package,
                    steps,
                    outdir,
                    1,
                    resize=True,
                    allow_activation_quant_for_weight_only=True,
                )
                self.assertEqual(len(actual), len(steps))
                for x, [(raw, shape)] in zip(inputs, actual):
                    with torch.no_grad():
                        expected = model(x)
                    self.assertEqual(shape, tuple(expected.shape))
                    torch.testing.assert_close(
                        torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(
                            shape
                        ),
                        expected,
                        atol=5e-4,
                        rtol=1e-5,
                    )

    def test_gguf_q4_k_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q4_k")

    def test_gguf_q5_k_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q5_k")

    def test_gguf_q6_k_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q6_k")

    def test_gguf_q4_0_linear(self):
        self._check_random_gguf_linear("q4_0")

    def test_gguf_q4_0_embedding(self):
        self._check_random_gguf_embedding("q4_0")

    def test_gguf_q4_0_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q4_0")

    def test_gguf_q4_1_linear(self):
        self._check_random_gguf_linear("q4_1")

    def test_gguf_q4_1_embedding(self):
        self._check_random_gguf_embedding("q4_1")

    def test_gguf_q4_1_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q4_1")

    def test_gguf_q5_0_linear(self):
        self._check_random_gguf_linear("q5_0")

    def test_gguf_q5_0_embedding(self):
        self._check_random_gguf_embedding("q5_0")

    def test_gguf_q5_0_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q5_0")

    def test_gguf_q5_1_linear(self):
        self._check_random_gguf_linear("q5_1")

    def test_gguf_q5_1_embedding(self):
        self._check_random_gguf_embedding("q5_1")

    def test_gguf_q5_1_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q5_1")

    def test_gguf_q8_0_linear(self):
        self._check_random_gguf_linear("q8_0")

    def test_gguf_q8_0_embedding(self):
        self._check_random_gguf_embedding("q8_0")

    def test_gguf_q8_0_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q8_0")

    def test_gguf_q2_k_linear(self):
        self._check_random_gguf_linear("q2_k")

    def test_gguf_q2_k_embedding(self):
        self._check_random_gguf_embedding("q2_k")

    def test_gguf_q2_k_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q2_k")

    def test_gguf_q3_k_linear(self):
        self._check_random_gguf_linear("q3_k")

    def test_gguf_q3_k_embedding(self):
        self._check_random_gguf_embedding("q3_k")

    def test_gguf_q3_k_dynamic_tokens(self):
        self._check_random_gguf_dynamic_tokens("q3_k")

    def test_gguf_q5_k_linear_within_rounding_bound(self):
        self._check_random_gguf_linear("q5_k")

    def test_gguf_q5_k_embedding(self):
        self._check_random_gguf_embedding("q5_k")

    def _check(
        self,
        model,
        inputs,
        atol=1e-5,
        rtol=1e-5,
        graph_ops=None,
        allow_activation_quant_for_weight_only=False,
        exact_topk_for_testing=False,
        exact_gelu_for_testing=False,
    ):
        model = model.eval()
        # Eager writes buffers in place; the package must keep them as constructed.
        with torch.no_grad():
            expected = copy.deepcopy(model)(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(model, inputs)).save(str(package))
            (outputs,) = _run_ggml(
                package,
                [inputs],
                outdir,
                len(expected),
                graph_ops=graph_ops,
                allow_activation_quant_for_weight_only=(
                    allow_activation_quant_for_weight_only
                ),
                exact_topk_for_testing=exact_topk_for_testing,
                exact_gelu_for_testing=exact_gelu_for_testing,
            )
        self.assertEqual(len(outputs), len(expected))
        for (raw, shape), reference in zip(outputs, expected):
            self.assertEqual(shape, tuple(reference.shape))
            actual = torch.frombuffer(bytearray(raw), dtype=reference.dtype)
            torch.testing.assert_close(
                actual.reshape(shape), reference, atol=atol, rtol=rtol, equal_nan=True
            )

    def _run_exported(self, model, example, inputs):
        """Exports `model` at `example`, before any eager call writes its
        buffers, and returns its float outputs on `inputs`."""
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(model, example)).save(str(package))
            (outputs,) = _run_ggml(package, [inputs], outdir, 1)
        return [
            torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
            for raw, shape in outputs
        ]

    def _check_steps(self, model, steps, graph_ops=None):
        """Exports `model` at the first step, runs every step on one
        executable and checks each float output against eager; returns them."""
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(model, steps[0])).save(str(package))
            actual = _run_ggml(package, steps, outdir, 1, graph_ops=graph_ops)
        outputs = [
            torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
            for [(raw, shape)] in actual
        ]
        for step, output in zip(steps, outputs):
            with torch.no_grad():
                expected = model(*step)
            torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
        return outputs

    def _binary(self, fn):
        self._check(Lambda(fn), (torch.randn(2, 3, 4), torch.randn(2, 3, 4)))

    def test_add(self):
        self._binary(torch.add)

    def test_sub(self):
        self._binary(torch.sub)

    def test_mul(self):
        self._binary(torch.mul)

    def test_div(self):
        self._check(
            Lambda(torch.div), (torch.randn(2, 3, 4), torch.rand(2, 3, 4) + 0.5)
        )

    def test_multiple_outputs(self):
        self._binary(lambda x, y: (x + y, x - y))

    def test_add_broadcast(self):
        self._check(Lambda(torch.add), (torch.randn(2, 3, 4), torch.randn(4)))

    def test_sub_broadcast_left(self):
        self._check(Lambda(torch.sub), (torch.randn(3, 1), torch.randn(2, 3, 4)))

    def test_div_broadcast_both(self):
        self._check(Lambda(torch.div), (torch.randn(2, 1, 4), torch.rand(3, 1) + 0.5))

    def test_add_bool_is_logical_or(self):
        self._check(
            Lambda(lambda x, y, z: (x.le(y) + x.le(z)).to(x.dtype)),
            (
                torch.zeros(4),
                torch.tensor([1.0, 1.0, -1.0, -1.0]),
                torch.tensor([1.0, -1.0, 1.0, -1.0]),
            ),
        )

    def test_add_scalar(self):
        self._check(Lambda(lambda x: x + 2.0), (torch.randn(2, 3, 4),))

    def test_int64_arithmetic(self):
        x = torch.arange(6).reshape(2, 3)
        for name, fn in {
            "add scalar": lambda x: x + 1,
            "mul scalar": lambda x: x * 2,
            "sub tensor": lambda x: x - x * x,
            "add then float": lambda x: (x + 1).float(),
        }.items():
            with self.subTest(op=name):
                self._check(Lambda(fn), (x,))

    def test_sub_scalar(self):
        self._check(Lambda(lambda x: x - 2.0), (torch.randn(2, 3, 4),))

    def test_add_scalar_permuted(self):
        self._check(Lambda(lambda x: x.permute(1, 0) + 1.0), (torch.randn(2, 3),))

    def test_rank5_with_a_leading_dim_above_one_is_refused(self):
        with self.assertRaisesRegex(AssertionError, "extra leading dim to be 1"):
            self._check(Lambda(lambda x: x * 2.0), (torch.randn(2, 1, 2, 3, 4),))

    def test_rank5_permute_moving_its_leading_one_inward_is_refused(self):
        with self.assertRaisesRegex(AssertionError, "leading 1s in front"):
            self._check(
                Lambda(lambda x: x.permute(1, 0, 2, 3, 4) * 2.0),
                (torch.randn(1, 2, 3, 4, 5),),
            )

    def test_rank5_views_along_a_folded_leading_one(self):
        cases = {
            "slice": lambda x: x[0:1] * 2.0,
            "select": lambda x: x[0] * 2.0,
            "split": lambda x: torch.split(x, 1, dim=0)[0] * 2.0,
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                self._check(Lambda(fn), (torch.randn(1, 2, 3, 4, 5),))

    def test_rank5_cat_along_a_folded_leading_one_is_refused(self):
        with self.assertRaisesRegex(AssertionError, "folded away above rank 4"):
            self._check(
                Lambda(lambda x, y: torch.cat([x, y], 0)),
                (torch.randn(1, 2, 3, 4, 5), torch.randn(1, 2, 3, 4, 5)),
            )

    def test_mul_scalar(self):
        self._check(Lambda(lambda x: x * 0.5), (torch.randn(2, 3, 4),))

    def test_div_scalar(self):
        self._check(Lambda(lambda x: x / 4.0), (torch.randn(2, 3, 4),))

    def test_squeeze_to_scalar(self):
        self._check(Lambda(lambda x: x.squeeze(0) * 2.0), (torch.randn(1),))

    def test_select_to_scalar(self):
        self._check(Lambda(lambda x: x.select(0, 1) * 2.0), (torch.randn(3),))

    def test_scalar_input(self):
        self._check(Lambda(torch.mul), (torch.randn(2, 3), torch.randn(())))

    def test_resize_with_scalar_last_input(self):
        class Scale(torch.nn.Module):
            def forward(self, x, scale):
                return x * scale

        # A rank-0 shape is an empty --input_sizes entry.
        cases = {
            "1x5,;1x3,": (
                Scale(),
                [(torch.randn(1, n), torch.randn(())) for n in (5, 3)],
                {"x": {1: Dim("n", min=1, max=8)}, "scale": None},
            ),
            ";": (
                Lambda(lambda x: x * 2.0),
                [(torch.randn(()),) for _ in range(2)],
                None,
            ),
        }
        for input_sizes, (model, steps, dynamic_shapes) in cases.items():
            with self.subTest(input_sizes=input_sizes):
                ep = torch.export.export(model, steps[0], dynamic_shapes=dynamic_shapes)
                with tempfile.TemporaryDirectory() as tmp:
                    outdir = Path(tmp)
                    package = outdir / "model.ptn"
                    to_native(ep).save(str(package))
                    actual = _run_ggml(package, steps, outdir, 1, resize=True)
                self.assertEqual(len(actual), len(steps))
                for inputs, [(raw, shape)] in zip(steps, actual):
                    expected = model(*inputs)
                    self.assertEqual(shape, tuple(expected.shape))
                    torch.testing.assert_close(
                        torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(
                            shape
                        ),
                        expected,
                    )

    def test_silu(self):
        self._check(Lambda(F.silu), (torch.randn(2, 3, 4),))

    def test_softplus(self):
        # Above 88, exp overflows in the branch softplus does not take.
        x = torch.tensor([[-100.0, -20.0, -1.0, 0.0, 1e-3, 1.0, 19.0, 21.0, 100.0]])
        self._check(Lambda(F.softplus), (x,), atol=1e-7, rtol=2e-6)
        self._check(Lambda(F.softplus), (torch.randn(2, 3, 4) * 10,), rtol=2e-6)

    def test_softplus_is_one_ggml_softplus(self):
        # Both sides of the threshold of 20 and far tails.
        x = torch.cat([torch.linspace(-40, 40, 161), torch.randn(39)]).reshape(4, 50)
        ops = collections.Counter()
        self._check(Lambda(F.softplus), (x,), atol=1e-6, rtol=1e-6, graph_ops=ops)
        self.assertGreater(ops["SOFTPLUS"], 0)
        self.assertEqual((ops["LOG"], ops["EXP"], ops["CONCAT"]), (0, 0, 0))

    def test_softplus_with_other_beta_or_threshold(self):
        # Each side of thresholds below, at and above ggml's 20.
        x = torch.cat([torch.linspace(-40, 40, 161), torch.randn(39)]).reshape(4, 50)
        for kwargs in (
            {"beta": 2.0},
            {"beta": 0.5, "threshold": 30.0},
            {"threshold": 10.0},
            {"beta": -1.0},
        ):
            with self.subTest(**kwargs):
                ops = collections.Counter()
                self._check(
                    Lambda(lambda t, kw=kwargs: F.softplus(t, **kw)),
                    (x,),
                    atol=1e-6,
                    rtol=1e-6,
                    graph_ops=ops,
                )
                self.assertGreater(ops["SOFTPLUS"], 0)
                self.assertGreater(ops["GET_ROWS"], 0)

    def test_swiglu_arithmetic_with_shared_silu(self):
        def forward(x, up):
            silu = x * torch.sigmoid(x)
            return silu, silu * up

        ops = collections.Counter()
        self._check(
            Lambda(forward), (torch.randn(2, 3, 4), torch.randn(2, 3, 4)), graph_ops=ops
        )
        # x * sigmoid(x) becomes SILU, and SILU * up becomes SWIGLU.
        self.assertEqual(ops["SIGMOID"], 0)
        self.assertGreater(ops["SILU"], 0)
        self.assertGreater(ops["SWIGLU"], 0)

    def test_swiglu_and_silu_operand_orders(self):
        cases = {
            "up * silu(x)": (lambda x, up: up * F.silu(x), "SWIGLU", "SILU"),
            "sigmoid(x) * x": (
                lambda x, up: torch.sigmoid(x) * x + up,
                "SILU",
                "SIGMOID",
            ),
        }
        for name, (fn, fused, absent) in cases.items():
            with self.subTest(name):
                ops = collections.Counter()
                self._check(
                    Lambda(fn),
                    (torch.randn(2, 3, 4), torch.randn(2, 3, 4)),
                    graph_ops=ops,
                )
                self.assertGreater(ops[fused], 0)
                self.assertEqual(ops[absent], 0)

    def test_swiglu_broadcast_not_rewritten(self):
        ops = collections.Counter()
        self._check(
            Lambda(lambda x, up: (x * torch.sigmoid(x)) * up),
            (torch.randn(2, 3, 4), torch.randn(4)),
            graph_ops=ops,
        )
        self.assertEqual(ops["SWIGLU"], 0)

    def test_geglu_operand_orders(self):
        cases = {
            "gelu(x) * up": (
                lambda x, up: F.gelu(x, approximate="tanh") * up,
                "GEGLU",
                "GELU",
            ),
            "up * gelu(x)": (
                lambda x, up: up * F.gelu(x, approximate="tanh"),
                "GEGLU",
                "GELU",
            ),
            "up * erf gelu(x)": (lambda x, up: up * F.gelu(x), "GEGLU_ERF", "GELU_ERF"),
        }
        for name, (fn, fused, absent) in cases.items():
            with self.subTest(name):
                ops = collections.Counter()
                # ggml-cpu's tanh GELU reads an FP16 table, as in test_gelu_tanh.
                self._check(
                    Lambda(fn),
                    (torch.randn(2, 3, 4) * 3.0, torch.randn(2, 3, 4)),
                    atol=1e-3,
                    rtol=2e-3,
                    graph_ops=ops,
                )
                self.assertGreater(ops[fused], 0)
                self.assertEqual(ops[absent], 0)

    def test_geglu_with_shared_gelu(self):
        def forward(x, up):
            gelu = F.gelu(x, approximate="tanh")
            return gelu, gelu * up

        ops = collections.Counter()
        self._check(
            Lambda(forward),
            (torch.randn(2, 3, 4) * 3.0, torch.randn(2, 3, 4)),
            atol=1e-3,
            rtol=2e-3,
            graph_ops=ops,
        )
        self.assertGreater(ops["GELU"], 0)
        self.assertGreater(ops["GEGLU"], 0)

    def test_glu_of_a_buffer_written_after_the_activation(self):
        # The activation reads a buffer that is then written in place, so the
        # product must use the activation from before the write.
        class ActivationThenWrite(torch.nn.Module):
            def __init__(self, activation):
                super().__init__()
                self.activation = activation
                self.register_buffer("state", torch.randn(1, 2, 4, 8))

            def forward(self, rows, positions, up):
                gate = self.activation(self.state)
                self.state.index_copy_(2, positions, rows)
                return gate * up

        activations = {
            "gelu": lambda x: F.gelu(x, approximate="tanh"),
            "erf gelu": F.gelu,
            "silu": F.silu,
        }
        for name, activation in activations.items():
            with self.subTest(name):
                self._check(
                    ActivationThenWrite(activation),
                    (
                        torch.randn(1, 2, 1, 8) * 5.0,
                        torch.tensor([1]),
                        torch.randn(1, 2, 4, 8),
                    ),
                    atol=2e-3,
                    rtol=2e-3,
                )

    def test_sigmoid(self):
        self._check(Lambda(torch.sigmoid), (torch.randn(2, 3, 4),))

    def test_tanh(self):
        self._check(Lambda(torch.tanh), (torch.randn(1, 512) * 3.0,))

    def test_gelu(self):
        ops = collections.Counter()
        self._check(Lambda(F.gelu), (torch.randn(2, 3, 4) * 3.0,), graph_ops=ops)
        self.assertGreater(ops["GELU_ERF"], 0)

    def test_gelu_tanh(self):
        # ggml's GELU, as llama.cpp: ggml-cpu reads it from an F16 table, a few
        # 1e-4 off in F32, which GPUs evaluate in F32.
        gpu = os.environ.get("GGML_PTN_DEVICE", "cpu") == "gpu"
        ops = collections.Counter()
        self._check(
            Lambda(lambda x: F.gelu(x, approximate="tanh")),
            (torch.linspace(-12.0, 12.0, 96).reshape(2, 3, 16),),
            **(_GELU_TOL if gpu else {"atol": 1e-3, "rtol": 2e-3}),
            graph_ops=ops,
        )
        self.assertEqual(ops["GELU_ERF"], 0)
        self.assertGreater(ops["GELU"], 0)
        self.assertEqual(ops["TANH"], 0)

    def test_gelu_tanh_exact_for_testing(self):
        # With exact_gelu_for_testing the CPU builds it from tanh and
        # arithmetic; GPUs keep ggml's GELU, which is F32 already.
        gpu = os.environ.get("GGML_PTN_DEVICE", "cpu") == "gpu"
        ops = collections.Counter()
        self._check(
            Lambda(lambda x: F.gelu(x, approximate="tanh")),
            (torch.linspace(-12.0, 12.0, 96).reshape(2, 3, 16),),
            **_GELU_TOL,
            graph_ops=ops,
            exact_gelu_for_testing=True,
        )
        self.assertEqual(ops["GELU"] > 0, gpu)
        self.assertEqual(ops["TANH"] > 0, not gpu)

    def test_log(self):
        self._check(Lambda(torch.log), (torch.rand(2, 3, 4) + 0.1,))

    def test_exp(self):
        self._check(Lambda(torch.exp), (torch.randn(3, 1, 1).clamp(max=4.6),))

    def test_exp_half_and_bfloat16(self):
        for dtype, rtol in ((torch.float16, 1e-3), (torch.bfloat16, 8e-3)):
            with self.subTest(dtype=dtype):
                self._check(
                    Lambda(torch.exp),
                    (torch.randn(2, 3, 4, dtype=dtype),),
                    rtol=rtol,
                )

    def test_neg(self):
        self._check(Lambda(torch.neg), (torch.randn(2, 3, 4),))

    def test_relu(self):
        self._check(Lambda(F.relu), (torch.randn(2, 3, 4),))

    def test_hardtanh(self):
        # relu6 exports as hardtanh(0, 6).
        self._check(Lambda(F.relu6), (torch.randn(2, 3, 4) * 8.0,))
        with self.assertRaisesRegex(AssertionError, "floating-point"):
            self._check(Lambda(F.relu6), (torch.arange(6).reshape(2, 3) - 2,))

    def test_relu6_of_half_inputs(self):
        x = torch.randn(2, 3, 4) * 8.0
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check(Lambda(F.relu6), (x.to(dtype),), atol=0, rtol=0)

    def test_elu(self):
        self._check(Lambda(F.elu), (torch.randn(2, 3, 4) * 3.0,))
        # Eager's expm1 is within rtol of these; exp(x) - 1 is 19% off at -1e-7.
        special = torch.tensor(
            [[float("nan"), float("inf"), -float("inf"), 0.0, -1e-7, -1e-30]]
        )
        self._check(Lambda(F.elu), (special,), atol=0)

    def test_elu_in_place(self):
        # Exports as elu, not elu_, unlike ReLU6(inplace=True) in ConvBN.
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ELU(inplace=True))
        self._check(model, (torch.randn(2, 3),), **_MATMUL_TOL)

    def test_elu_half_and_bfloat16(self):
        for dtype, rtol in ((torch.float16, 1e-3), (torch.bfloat16, 8e-3)):
            with self.subTest(dtype=dtype):
                self._check(
                    Lambda(F.elu), (torch.randn(2, 3, 4, dtype=dtype),), rtol=rtol
                )

    def test_elu_rejects_other_parameters(self):
        for args in ((0.5, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 0.5)):
            with self.subTest(args=args):
                with self.assertRaisesRegex(AssertionError, "input_scale 1"):
                    self._check(
                        Lambda(
                            lambda x, args=args: torch.ops.aten.elu.default(x, *args)
                        ),
                        (torch.randn(2, 3),),
                    )

    def test_sqrt(self):
        x = torch.tensor(
            [[0.0, -0.0, 4.0, 2.0], [float("inf"), -1.0, float("nan"), 9.0]]
        )
        self._check(Lambda(torch.sqrt), (x,))
        self._check(Lambda(lambda x: x.transpose(0, 1).sqrt()), (x,))

    def test_sqrt_of_half(self):
        # ggml's sqrt kernels take half types, so no cast surrounds it.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                ops = collections.Counter()
                self._check(
                    Lambda(torch.sqrt),
                    (torch.rand(2, 3, dtype=dtype),),
                    atol=0,
                    rtol=0,
                    graph_ops=ops,
                )
                self.assertGreater(ops["SQRT"], 0)
                self.assertEqual(ops["CPY"], 0)

    def test_conv2d(self):
        self._check(
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            (torch.randn(2, 3, 17, 15),),
            **_MATMUL_TOL,
        )

    def test_conv2d_asymmetric(self):
        self._check(
            torch.nn.Conv2d(
                3, 8, (3, 5), stride=(2, 1), padding=(1, 2), dilation=(1, 2)
            ),
            (torch.randn(1, 3, 11, 13),),
            **_MATMUL_TOL,
        )

    def test_depthwise_conv2d_asymmetric(self):
        self._check(
            torch.nn.Conv2d(
                4, 4, (3, 5), stride=(1, 2), padding=(1, 2), dilation=(2, 1), groups=4
            ),
            (torch.randn(1, 4, 10, 12),),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_conv1d(self):
        self._check(
            torch.nn.Conv1d(4, 6, 3, stride=2, padding=2, dilation=2),
            (torch.randn(2, 4, 11),),
            **_MATMUL_TOL,
        )

    def test_depthwise_conv1d(self):
        self._check(
            torch.nn.Conv1d(4, 4, 3, padding=1, groups=4),
            (torch.randn(1, 4, 9),),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_conv_transpose1d_is_refused(self):
        # Running this as a forward conv would give the same output shape.
        with self.assertRaisesRegex(AssertionError, "non-transposed"):
            self._check(
                torch.nn.ConvTranspose1d(16, 16, 3, padding=1), (torch.randn(1, 16, 5),)
            )

    def test_conv2d_pointwise(self):
        self._check(
            torch.nn.Conv2d(16, 8, 1, bias=False),
            (torch.randn(1, 16, 7, 7),),
            **_MATMUL_TOL,
        )

    def test_conv2d_pointwise_batched(self):
        ops = collections.Counter()
        self._check(
            torch.nn.Conv2d(16, 8, 1),
            (torch.randn(2, 16, 5, 6),),
            graph_ops=ops,
            **_MATMUL_TOL,
        )
        # At stride 1 without padding the columns are a transpose copy.
        self.assertEqual(ops["IM2COL"], 0)

    def test_conv2d_pointwise_strided_padded(self):
        self._check(
            torch.nn.Conv2d(16, 8, 1, stride=(2, 3), padding=(1, 2)),
            (torch.randn(2, 16, 7, 8),),
            **_MATMUL_TOL,
        )

    def test_conv1d_pointwise_strided(self):
        self._check(
            torch.nn.Conv1d(6, 4, 1, stride=2),
            (torch.randn(2, 6, 9),),
            **_MATMUL_TOL,
        )

    def test_conv_bn_relu6(self):
        ops = collections.Counter()
        self._check(
            ConvBN(groups=1, bias=False),
            (torch.randn(1, 8, 9, 9),),
            graph_ops=ops,
            **_MATMUL_TOL,
        )
        # Absorbed into the convolution at load: no statistics math or scaling.
        self.assertEqual(ops["SQRT"] + ops["MUL"], 0)

    def test_depthwise_conv_bn_relu6(self):
        self._check(ConvBN(groups=8, bias=True), (torch.randn(1, 8, 9, 9),), atol=1e-4)

    def test_pointwise_conv_bn_relu6(self):
        ops = collections.Counter()
        self._check(
            ConvBN(groups=1, bias=True, kernel_size=1),
            (torch.randn(1, 8, 9, 9),),
            graph_ops=ops,
            **_MATMUL_TOL,
        )
        self.assertEqual(ops["SQRT"] + ops["MUL"], 0)

    def test_batch_norms_with_shared_convolution(self):
        ops = collections.Counter()
        self._check(
            SharedConvBN(), (torch.randn(1, 4, 5, 6),), graph_ops=ops, **_MATMUL_TOL
        )
        # AOT folds the shared-weight branch; two batch norm scales and two
        # convolutions remain in each graph.
        self.assertEqual(ops["SQRT"], 0)
        self.assertGreater(ops["MUL_MAT"], 0)
        self.assertEqual(ops["MUL"], ops["MUL_MAT"])

    def test_batch_norm_of_computed_statistics(self):
        # Statistics computed in the graph cannot fold at load.
        def forward(x, mean, var, weight, bias):
            return F.batch_norm(x, mean * 1.0, var * 1.0, weight, bias, training=False)

        ops = collections.Counter()
        self._check(
            Lambda(forward),
            (
                torch.randn(1, 4, 5, 6),
                torch.randn(4),
                torch.rand(4) + 0.5,
                torch.rand(4) + 0.5,
                torch.randn(4),
            ),
            graph_ops=ops,
        )
        self.assertGreater(ops["SQRT"], 0)

    def test_batch_norm_no_affine(self):
        norm = torch.nn.BatchNorm2d(4, affine=False)
        norm.running_mean.uniform_(-1.0, 1.0)
        norm.running_var.uniform_(0.5, 2.0)
        self._check(norm, (torch.randn(1, 4, 5, 6),), atol=1e-5)

    def test_batch_norm_statistics_written_by_another_method(self):
        class Update(torch.nn.Module):
            def __init__(self, bn):
                super().__init__()
                self.bn = bn

            def forward(self, delta):
                self.bn.running_mean.add_(delta)
                return self.bn.running_mean + 1

        model = ConvBN(groups=1, bias=True).eval()
        methods = {
            "predict": (model, (torch.randn(1, 8, 9, 9),)),
            "update": (Update(model.bn), (torch.randn(8),)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(
                {
                    name: torch.export.export(module, inputs)
                    for name, (module, inputs) in methods.items()
                }
            ).save(str(package))
            for name, (module, inputs) in methods.items():
                with self.subTest(method=name), torch.no_grad():
                    expected = copy.deepcopy(module)(*inputs)
                    ((raw, shape),) = _run_ggml(
                        package, [inputs], outdir, 1, method=name
                    )[0]
                    actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                    torch.testing.assert_close(
                        actual.reshape(shape), expected, atol=1e-4, rtol=1e-5
                    )

    def test_max_pool2d(self):
        self._check(
            torch.nn.MaxPool2d(3, stride=2, padding=1), (torch.randn(1, 4, 11, 9),)
        )

    def test_max_pool2d_asymmetric(self):
        self._check(
            torch.nn.MaxPool2d((2, 3), stride=(2, 1), padding=(1, 1)),
            (torch.randn(1, 4, 9, 11),),
        )

    def test_adaptive_avg_pool2d(self):
        self._check(
            torch.nn.AdaptiveAvgPool2d(1), (torch.randn(2, 4, 7, 5),), atol=1e-5
        )

    def test_mean_trailing_dims(self):
        self._check(Lambda(lambda x: x.mean(dim=(-1, -2))), (torch.randn(2, 3, 4, 5),))

    def test_sum_trailing_dims(self):
        self._check(
            Lambda(lambda x: x.sum(dim=-1, keepdim=True)), (torch.randn(2, 3, 16, 32),)
        )
        self._check(Lambda(lambda x: x.sum(dim=[2, 3])), (torch.randn(2, 3, 4, 5),))

    def test_amax_trailing_dims(self):
        self._check(
            Lambda(lambda x: x.amax(dim=-1, keepdim=True)), (torch.randn(2, 3, 16, 32),)
        )
        self._check(Lambda(lambda x: x - x.amax(dim=[0, 1])), (torch.randn(80, 300),))

    def _check_over_any_dims(self, fn):
        x = torch.randn(2, 3, 4, 5)
        for dims, keepdim in (((0,), False), ((1, 3), True), ((-3,), False)):
            with self.subTest(dims=dims, keepdim=keepdim):
                self._check(
                    Lambda(
                        lambda x, dims=dims, keepdim=keepdim: fn(
                            x, dims, keepdim=keepdim
                        )
                    ),
                    (x,),
                )

    def test_sum_over_any_dims(self):
        self._check_over_any_dims(torch.sum)

    def test_mean_over_any_dims(self):
        self._check_over_any_dims(torch.mean)

    def test_amax_over_any_dims(self):
        self._check_over_any_dims(torch.amax)

    def test_softmax_over_any_dim(self):
        x = torch.randn(2, 3, 5, 8) * 4.0
        for dim in (-1, 0, 1, -2):
            with self.subTest(dim=dim):
                self._check(
                    Lambda(lambda x, dim=dim: torch.softmax(x, dim)), (x,), atol=1e-6
                )

    def test_softmax_over_an_empty_dim_is_refused(self):
        with self.assertRaisesRegex(AssertionError, "empty dimension"):
            self._check(Lambda(lambda x: torch.softmax(x, -1)), (torch.randn(2, 0),))

    def test_softmax_last_dim(self):
        self._check(
            Lambda(lambda x: torch.softmax(x, dim=-1)),
            (torch.randn(2, 3, 5, 40) * 4.0,),
            atol=1e-6,
        )

    def test_linear(self):
        self._check(Linear(bias=False), (torch.randn(4, 32),), **_MATMUL_TOL)

    def test_linear_bias(self):
        self._check(Linear(bias=True), (torch.randn(2, 4, 32),), **_MATMUL_TOL)

    def test_mm(self):
        self._check(
            Lambda(torch.mm), (torch.randn(4, 8), torch.randn(8, 5)), **_MATMUL_TOL
        )

    def test_bmm(self):
        self._check(
            Lambda(torch.bmm),
            (torch.randn(2, 4, 8), torch.randn(2, 8, 5)),
            **_MATMUL_TOL,
        )

    def test_batched_experts_lower_to_mul_mat_id(self):
        model = BatchedExperts()
        for tokens in (1, 5):
            with self.subTest(tokens=tokens):
                ops = collections.Counter()
                self._check(
                    model, (torch.randn(tokens, 64),), graph_ops=ops, **_MATMUL_TOL
                )
                # Per built graph: the router's softmax and mul_mat, and one
                # mul_mat_id per expert projection.
                graphs = ops["SOFT_MAX"]
                self.assertGreater(graphs, 0)
                self.assertEqual(ops["MUL_MAT_ID"], 2 * graphs)
                self.assertEqual(ops["MUL_MAT"], graphs)

    @unittest.skipIf(
        _BACKEND in ("cuda", "hip"),
        "ggml's MMF kernel takes TF32 (CUDA) or XF32 (HIP) inputs for F32 experts",
    )
    def test_f32_experts_of_many_rows_match_eager(self):
        # 16 tokens with two experts each are 32 rows, past the 8 below which
        # HIP keeps its exact kernel.
        self._check(BatchedExperts(), (torch.randn(16, 64),))

    def test_repeated_tokens_feed_mul_mat_id_once(self):
        # Each token repeated k times in a row, as batched_mm does, is read
        # once by mul_mat_id; tokens tiled instead are repeated as written.
        weight = torch.randn(4, 16, 64) / 8
        ids = torch.tensor([3, 0, 1, 3, 2, 0])

        def interleaved(x, ids):
            rows = x.repeat_interleave(2, dim=0).unsqueeze(-1)
            return torch.bmm(weight[ids], rows).squeeze(-1)

        def tiled(x, ids):
            rows = x.repeat(2, 1).unsqueeze(-1)
            return torch.bmm(weight[ids], rows).squeeze(-1)

        for name, fn, repeats in (("interleaved", interleaved, 0), ("tiled", tiled, 1)):
            with self.subTest(name):
                ops = collections.Counter()
                self._check(
                    Lambda(fn), (torch.randn(3, 64), ids), graph_ops=ops, **_MATMUL_TOL
                )
                self.assertGreater(ops["MUL_MAT_ID"], 0)
                self.assertEqual(ops["REPEAT"] > 0, repeats > 0)

    def test_half_and_bfloat16_expert_matmuls(self):
        ids = torch.tensor([3, 0, 1, 3, 2, 0])
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                ops = collections.Counter()
                self._check(
                    ExpertMatmul(4, 16, 32, dtype),
                    (torch.randn(6, 32, dtype=dtype), ids),
                    atol=1e-2,
                    rtol=1e-2,
                    graph_ops=ops,
                )
                self.assertGreater(ops["MUL_MAT_ID"], 0)

    def test_other_expert_gathers_take_the_generic_path(self):
        # Transposed experts are the bmm's right operand, and the shared
        # gate_up experts are read by another op too, so those projections
        # gather their weights and run as mul_mat; the shared model's down
        # projection still runs as mul_mat_id.
        for name, model, mul_mat_ids in (
            ("transposed", BatchedExperts(transposed=True), 0),
            ("shared", BatchedExperts(shared=True), 1),
        ):
            with self.subTest(experts=name):
                ops = collections.Counter()
                self._check(model, (torch.randn(3, 64),), graph_ops=ops, **_MATMUL_TOL)
                graphs = ops["SOFT_MAX"]
                self.assertGreater(graphs, 0)
                self.assertEqual(ops["MUL_MAT_ID"], mul_mat_ids * graphs)
                self.assertEqual(ops["MUL_MAT"], (3 - mul_mat_ids) * graphs)

    def test_index_gathers_slices_of_any_rank(self):
        class Gather(torch.nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.table = torch.nn.Parameter(torch.randn(shape))

            def forward(self, ids):
                return self.table[ids]

        for shape in ((4,), (4, 5), (4, 3, 5)):
            with self.subTest(shape=shape):
                self._check(Gather(shape), (torch.tensor([[3, 0], [1, 3]]),))

    def test_constant_subgraphs_are_computed_once(self):
        class ConstantMath(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.log_decay = torch.nn.Parameter(torch.randn(16))
                self.weight = torch.nn.Parameter(torch.randn(16) * 0.1)

            def forward(self, x):
                decay = -torch.exp(self.log_decay.float())
                norm = F.rms_norm(x, (16,), eps=1e-6) * (self.weight + 1.0)
                return decay * x + norm

        model = ConstantMath().eval()
        x = torch.randn(3, 16)
        with torch.no_grad():
            expected = model(x)
        ep = torch.export.export(model, (x,))
        ops = {fold: collections.Counter() for fold in (True, False)}
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(ep).save(str(package))
            folded, unfolded = (
                _run_ggml(
                    package,
                    [(x,)],
                    outdir,
                    1,
                    graph_ops=ops[fold],
                    fold_constants=fold,
                )
                for fold in (True, False)
            )
        _assert_same_as_unfolded(self, folded, unfolded)
        (((raw, shape),),) = folded
        actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        torch.testing.assert_close(actual.reshape(shape), expected)
        self.assertGreater(ops[False]["EXP"], 0)
        for op in ("EXP", "NEG"):
            self.assertEqual(ops[True][op], 0, op)
        self.assertLess(ops[True].total(), ops[False].total())

    def test_matmul(self):
        self._check(
            Lambda(torch.matmul),
            (torch.randn(2, 4, 8), torch.randn(2, 8, 5)),
            **_MATMUL_TOL,
        )

    def test_matmul_broadcast_rhs(self):
        self._check(
            Lambda(torch.matmul),
            (torch.randn(2, 4, 8), torch.randn(8, 5)),
            **_MATMUL_TOL,
        )

    def test_matmul_reads_a_transposed_rhs_without_copies(self):
        ops = collections.Counter()
        self._check(
            Lambda(lambda a, b: a @ b.transpose(-1, -2)),
            (torch.randn(2, 4, 8), torch.randn(2, 5, 8)),
            graph_ops=ops,
            **_MATMUL_TOL,
        )
        self.assertEqual(ops["CONT"], 0)

    def test_matmul_outer_product(self):
        # Transposing the [1, 5] right operand gives a view that passes
        # ggml_is_contiguous with transposed strides, which ggml_mul_mat rejects.
        self._check(
            Lambda(torch.matmul),
            (torch.randn(2, 4, 1), torch.randn(2, 1, 5)),
            **_MATMUL_TOL,
        )

    def test_embedding(self):
        self._check(Embedding(), (torch.tensor([[1, 3, 9], [0, 2, 5]]),))

    def test_embedding_squeezed_indices(self):
        class SqueezedEmbedding(Embedding):
            def forward(self, indices):
                return self.embedding(indices.squeeze(0))

        self._check(SqueezedEmbedding(), (torch.tensor([[1, 3, 9]]),))

    def test_embedding_arange_indices(self):
        class PositionEmbedding(Embedding):
            def forward(self, x):
                return self.embedding(torch.arange(1, 10, 2)) + x

        self._check(PositionEmbedding(), (torch.randn(5, 8),))

    def test_embedding_arange_indices_rejects_out_of_bounds(self):
        class InvalidPositionEmbedding(Embedding):
            def forward(self, x):
                return self.embedding(torch.arange(-1, 4)) + x

        with self.assertRaisesRegex(AssertionError, "arange row index"):
            # Export does not execute the invalid gather; the runner must reject it.
            with tempfile.TemporaryDirectory() as tmp:
                outdir = Path(tmp)
                x = torch.randn(5, 8)
                package = outdir / "model.ptn"
                ep = torch.export.export(InvalidPositionEmbedding().eval(), (x,))
                to_native(ep).save(str(package))
                _run_ggml(package, [(x,)], outdir, 1)

    def test_repeated_row_gather_rechecks_indices(self):
        class RepeatedRowGather(Embedding):
            def forward(self, indices):
                return self.embedding.weight[indices.unsqueeze(0).repeat(2, 1)]

        model = RepeatedRowGather().eval()
        inputs = (torch.tensor([1, 3, 9]),)
        self._check(model, inputs)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(model, inputs)).save(str(package))
            for invalid in (-1, 10):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(AssertionError, "row index is outside"):
                        _run_ggml(
                            package,
                            [inputs, (torch.tensor([1, invalid, 3]),)],
                            outdir,
                            1,
                        )

    def test_repeat_tiles_non_singleton_dimensions(self):
        self._check(
            Lambda(lambda x: x.repeat(2, 3, 2)),
            (torch.arange(6, dtype=torch.float32).reshape(2, 3),),
        )

    def test_rms_norm(self):
        self._check(RMSNorm(), (torch.randn(2, 3, 16),))

    def test_layer_norm(self):
        norm = torch.nn.LayerNorm(16)
        torch.nn.init.uniform_(norm.weight, 0.5, 1.5)
        torch.nn.init.uniform_(norm.bias, -0.5, 0.5)
        self._check(norm, (torch.randn(2, 3, 16),))

    def test_layer_norm_without_affine(self):
        self._check(
            torch.nn.LayerNorm(16, elementwise_affine=False), (torch.randn(2, 3, 16),)
        )

    def test_layer_norm_half(self):
        norm = torch.nn.LayerNorm(16)
        torch.nn.init.uniform_(norm.weight, 0.5, 1.5)
        torch.nn.init.uniform_(norm.bias, -0.5, 0.5)
        x = torch.randn(2, 3, 16)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                # One rounding of the output; eager normalizes in F32 too.
                self._check(
                    copy.deepcopy(norm).to(dtype),
                    (x.to(dtype),),
                    atol=1e-2 if dtype == torch.bfloat16 else 2e-3,
                    rtol=1e-2 if dtype == torch.bfloat16 else 2e-3,
                )

    def test_layer_norm_rejects_used_mean(self):
        def forward(x):
            out, mean, _ = torch.native_layer_norm(x, [16], None, None, 1e-5)
            return out + mean

        with self.assertRaisesRegex(AssertionError, "lowers only its first output"):
            self._check(Lambda(forward), (torch.randn(2, 3, 16),))

    def test_reshape(self):
        self._check(Lambda(lambda x: x.reshape(4, 6)), (torch.randn(2, 3, 4),))

    def test_view_infer(self):
        self._check(Lambda(lambda x: x.view(-1, 2)), (torch.randn(2, 3, 4),))

    def test_permute(self):
        self._check(Lambda(lambda x: x.permute(2, 0, 1)), (torch.randn(2, 3, 4),))

    def test_permute_4d(self):
        self._check(Lambda(lambda x: x.permute(0, 2, 1, 3)), (torch.randn(2, 3, 4, 5),))

    def test_transpose(self):
        self._check(Lambda(lambda x: x.transpose(0, 2)), (torch.randn(2, 3, 4),))

    def test_slice(self):
        self._check(Lambda(lambda x: x[:, 1:3]), (torch.randn(2, 4, 5),))

    def test_slice_last_dim(self):
        self._check(Lambda(lambda x: x[..., 1:4]), (torch.randn(2, 4, 5),))

    def test_slice_negative(self):
        self._check(Lambda(lambda x: x[:, -3:-1]), (torch.randn(2, 4, 5),))

    def test_slice_int64(self):
        self._check(Lambda(lambda x: x[:, 1:]), (torch.tensor([[1, 2, 3], [4, 5, 6]]),))

    def test_select(self):
        self._check(Lambda(lambda x: x[:, 2]), (torch.randn(2, 4, 5),))

    def test_select_negative(self):
        self._check(Lambda(lambda x: x[:, -1]), (torch.randn(2, 4, 5),))

    def test_unsqueeze(self):
        self._check(Lambda(lambda x: x.unsqueeze(1)), (torch.randn(2, 3),))

    def test_squeeze(self):
        self._check(Lambda(lambda x: x.squeeze(1)), (torch.randn(2, 1, 3),))

    def test_expand(self):
        self._check(Lambda(lambda x: x.expand(2, 3, 4)), (torch.randn(1, 3, 1),))

    def test_expand_keep_dim(self):
        self._check(Lambda(lambda x: x.expand(-1, 3, 4)), (torch.randn(2, 3, 1),))

    def test_cat(self):
        self._check(
            Lambda(lambda x, y: torch.cat([x, y], dim=1)),
            (torch.randn(2, 3, 4), torch.randn(2, 5, 4)),
        )

    def test_contiguous_cat_runs_as_three_dims(self):
        # ggml-cuda launches one kernel per ne[3] slice of contiguous operands,
        # which therefore concatenate as [inner, mid, outer] views.
        for dim in range(4):
            with self.subTest(dim=dim):
                ops = collections.Counter()
                self._check(
                    Lambda(lambda x, y, dim=dim: torch.cat([x * 2, y * 3], dim)),
                    (torch.randn(3, 4, 5, 6), torch.randn(3, 4, 5, 6)),
                    graph_ops=ops,
                )
                self.assertEqual(ops["RESHAPE"] > 0, dim != 0)

    def test_cat_of_mixed_ggml_types(self):
        # The F16 embedding result is F32 in ggml; the F16 input stays F16.
        self._check(
            HalfEmbeddingCat(),
            (
                torch.randn(2, 3, 4, dtype=torch.float16),
                torch.tensor([[1, 3, 9], [0, 2, 5]]),
            ),
        )

    def test_cat_of_a_thousand_repeated_inputs(self):
        # One node and two values, but a concat per list entry.
        self._check(Lambda(lambda x: torch.cat([x] * 1000)), (torch.randn(3),))

    def test_cat_last_dim(self):
        self._check(
            Lambda(lambda x, y: torch.cat([x, y], dim=-1)),
            (torch.randn(2, 3, 4), torch.randn(2, 3, 2)),
        )

    def test_constant_pad(self):
        self._check(Lambda(lambda x: F.pad(x, (1, 2, 0, 3))), (torch.randn(2, 3, 4),))

    def test_constant_pad_rejects_nonzero_value(self):
        with self.assertRaisesRegex(AssertionError, "only pads with zeros"):
            self._check(
                Lambda(lambda x: F.pad(x, (1, 1), value=1.0)), (torch.randn(2, 3),)
            )

    def test_constant_pad_of_half(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._check(
                    Lambda(lambda x: F.pad(x, (1, 1))),
                    (torch.randn(2, 3, dtype=dtype),),
                    atol=0,
                    rtol=0,
                )

    def test_clone(self):
        self._check(Lambda(lambda x: x.clone() + 1.0 * x), (torch.randn(2, 3),))

    def test_to_half(self):
        self._check(Lambda(lambda x: x.to(torch.float16)), (torch.randn(2, 3),))

    def test_unary_rejects_integer_input(self):
        x = torch.arange(6).reshape(2, 3) - 2
        for name, fn in {"neg": torch.neg, "sigmoid": torch.sigmoid}.items():
            with self.subTest(op=name):
                with self.assertRaisesRegex(AssertionError, "floating-point"):
                    self._check(Lambda(fn), (x,))

    def test_rejects_add_alpha(self):
        with self.assertRaisesRegex(AssertionError, "alpha"):
            self._binary(lambda x, y: torch.add(x, y, alpha=2.0))

    def test_alias_of_intermediate(self):
        self._check(
            Lambda(lambda x, y: torch.ops.aten.alias.default(x + y) * 0.5),
            (torch.randn(2, 3), torch.randn(2, 3)),
        )

    def test_permute_of_size_one_dims(self):
        self._check(
            Lambda(
                lambda x: (x * 2.0).permute(1, 0, 2).contiguous() + x.transpose(0, 1)
            ),
            (torch.randn(4, 1, 3),),
        )

    def test_linear_8da4w(self):
        model = torch.nn.Linear(128, 16)
        _quantize_8da4w(model, group_size=64)
        # ggml quantizes activations in its own blocks, torchao per token.
        self._check(model, (torch.randn(3, 128),), atol=5e-2, rtol=5e-2)

    def test_linear_8da4w_3d_input(self):
        model = torch.nn.Linear(64, 16)
        _quantize_8da4w(model, group_size=32)
        self._check(model, (torch.randn(2, 3, 64),), atol=5e-2, rtol=5e-2)

    def test_linear_8da4w_rejects_group_16(self):
        model = torch.nn.Linear(64, 16)
        _quantize_8da4w(model, group_size=16)
        with self.assertRaisesRegex(AssertionError, "groups divisible by 32"):
            self._check(model, (torch.randn(3, 64),))

    def test_embedding_4bit(self):
        model = torch.nn.Embedding(10, 64)
        _quantize_8da4w(model, group_size=64)
        self._check(model, (torch.tensor([[1, 3, 9], [0, 2, 5]]),))

    def test_embedding_4bit_rejects_asymmetric(self):
        model = torch.nn.Embedding(10, 64)
        _quantize_8da4w(model, group_size=64, embedding_mapping=MappingType.ASYMMETRIC)
        with self.assertRaisesRegex(AssertionError, "zero points of zero"):
            self._check(model, (torch.tensor([[1, 3, 9]]),))

    def test_sdpa_bool_mask_gqa(self):
        scores = torch.rand(3, 8)
        scores[:, 0] = 0.0
        self._check(
            Lambda(
                lambda q, k, v, m: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=m.le(0.7), enable_gqa=True
                )
            ),
            (
                torch.randn(1, 4, 3, 64),
                torch.randn(1, 2, 8, 64),
                torch.randn(1, 2, 8, 64),
                scores,
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_sdpa_causal(self):
        self._check(
            Lambda(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
            ),
            (
                torch.randn(1, 2, 5, 32),
                torch.randn(1, 2, 5, 32),
                torch.randn(1, 2, 5, 32),
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_sdpa_float_mask(self):
        self._check(
            Lambda(
                lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m)
            ),
            (
                torch.randn(1, 2, 3, 32),
                torch.randn(1, 2, 8, 32),
                torch.randn(1, 2, 8, 32),
                torch.randn(3, 8),
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_sdpa_le_tensor_mask(self):
        self._check(
            Lambda(
                lambda q, k, v, m, n: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=m.le(n)
                )
            ),
            (
                torch.randn(1, 2, 3, 32),
                torch.randn(1, 2, 8, 32),
                torch.randn(1, 2, 8, 32),
                torch.arange(8.0).repeat(3, 1),
                torch.tensor([[2.0], [4.0], [7.0]]),
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_lt_tensor(self):
        self._check(
            Lambda(lambda x, y: (x < y).to(x.dtype)),
            (torch.randn(2, 3, 4), torch.randn(2, 3, 4)),
        )

    def test_lt_scalar(self):
        self._check(
            Lambda(lambda x: (x < 0.25).to(x.dtype)),
            (torch.tensor([[-1.0, 0.0, 0.25, 0.5]]),),
        )

    # Known ggml GPU numerics, listed in the README. Each test compares with eager
    # at the tolerance the CPU passes and fails on the backend that skips it.
    @unittest.skipIf(_BACKEND == "cuda", "ggml-cuda runs F32 cuBLAS GEMMs in TF32")
    def test_f32_linear_matches_eager(self):
        # TF32 keeps 10 mantissa bits, so 1 + 2^-12 reaches the GEMM as 1.
        linear = torch.nn.Linear(1024, 4, bias=False)
        with torch.no_grad():
            linear.weight.fill_(1.0)
        self._check(linear, (torch.full((512, 1024), 1.0 + 2.0**-12),))

    def _check_f32_linear_rows(self, rows):
        # TF32 and XF32 keep 10 mantissa bits, so 1 + 2^-12 reaches the MMA as 1.
        linear = torch.nn.Linear(1024, 64, bias=False)
        with torch.no_grad():
            linear.weight.fill_(1.0)
        self._check(linear, (torch.full((rows, 1024), 1.0 + 2.0**-12),))

    def test_f32_linear_of_up_to_three_rows_matches_eager(self):
        for rows in (1, 3):
            with self.subTest(rows=rows):
                self._check_f32_linear_rows(rows)

    @unittest.skipIf(
        _BACKEND in ("cuda", "hip"),
        "ggml's MMF kernel takes TF32 (CUDA) or XF32 (HIP) inputs for F32 weights "
        "against 4-16 rows",
    )
    def test_f32_linear_of_four_to_sixteen_rows_matches_eager(self):
        for rows in (4, 16):
            with self.subTest(rows=rows):
                self._check_f32_linear_rows(rows)

    @unittest.skipIf(_BACKEND == "cuda", "ggml-cuda runs F32 cuBLAS GEMMs in TF32")
    def test_f32_linear_of_seventeen_rows_matches_eager(self):
        self._check_f32_linear_rows(17)

    @unittest.skipIf(
        _BACKEND == "cuda",
        "ggml-cuda accumulates F16 GEMMs in F16 unless "
        "cuda_f32_accumulation_for_testing is set",
    )
    def test_half_linear_matches_eager(self):
        if _BACKEND == "hip" and not _hip_accumulates_half_gemms_in_f32():
            self.skipTest(
                "ggml-hip accumulates F16 GEMMs in F16 on AMD GPUs other than CDNA "
                "and RDNA4"
            )
        self._check_half_linear_sum()

    @unittest.skipIf(
        _BACKEND in ("cuda", "hip"),
        "ggml-cuda and ggml-hip multiply Q4_1 and Q5_1 block scales in fp16 for "
        "up to 8 activation rows",
    )
    def test_small_scale_q4_1_and_q5_1_linear_matches_eager(self):
        # Weight scales of 3.1e-4 times the exact 2^-10 int8 scale of +-127/1024
        # activations make products of about five fp16 subnormal steps.
        x = torch.full((8, 512), 127 / 1024)
        x[1::2, ::2] *= -1
        for ggml_type, d, m in (("q4_1", 3.1e-4, -2.3e-3), ("q5_1", 3.1e-4, -4.8e-3)):
            with self.subTest(ggml_type):
                block, block_bytes = GGML_QUANT_SIZES[
                    GGMLQuantizationType[ggml_type.upper()]
                ]
                raw = torch.randint(0, 256, (8, 512 // block, block_bytes))
                raw = raw.to(torch.uint8)
                raw[..., :4] = torch.tensor([d, m]).half().view(torch.uint8)
                model = torch.nn.Linear(512, 8, bias=False)
                model.weight = torch.nn.Parameter(
                    ExportableGGUFTensor.from_raw(
                        raw.reshape(8, -1), ggml_type, torch.float32
                    ),
                    requires_grad=False,
                )
                self._check(
                    model,
                    (x,),
                    atol=1e-6,
                    rtol=1e-5,
                    allow_activation_quant_for_weight_only=True,
                )

    @unittest.skipIf(
        _BACKEND == "cuda", "ggml-cuda builds tanh and GELU with fast-math tanhf"
    )
    def test_tanh_and_gelu_match_eager(self):
        x = torch.linspace(-3.0, 3.0, 193)
        self._check(Lambda(torch.tanh), (x,), atol=1e-6, rtol=1e-5)
        # ggml-cpu's default GELU reads an F16 table, as in llama.cpp.
        self._check(
            Lambda(lambda x: F.gelu(x, approximate="tanh")),
            (x,),
            atol=1e-6,
            rtol=1e-5,
            exact_gelu_for_testing=True,
        )

    @unittest.skipIf(
        _BACKEND == "hip", "ggml-hip rounds BF16 GEMM results before adding the bias"
    )
    def test_bfloat16_linear_rounds_once_with_its_bias(self):
        # 1 + 2^-8 + 2^-12 - 2^-9 rounds to 1 in BF16; rounding the product
        # first gives 1 + 2^-7, and adding the bias keeps it there.
        linear = torch.nn.Linear(3, 4).bfloat16()
        with torch.no_grad():
            linear.weight.fill_(1.0)
            linear.bias.fill_(-(2.0**-9))
        row = torch.tensor([1.0, 2.0**-8, 2.0**-12])
        self._check(linear, (row.repeat(64, 1).bfloat16(),))

    @unittest.skipIf(
        _BACKEND in ("cuda", "hip"),
        "ggml's GPU flash attention is a few 1e-5 off eager on inputs exact in F16",
    )
    def test_grouped_attention_of_half_values_matches_eager(self):
        # Six query heads per KV head of 256 and one query at position 200 of
        # 256 keys. The inputs are exact in F16, so only the kernel's arithmetic
        # can differ from eager.
        def attention(q, k, v, positions):
            mask = torch.arange(256) <= positions.view(-1, 1)
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask[None, None], enable_gqa=True
            )

        generator = torch.Generator().manual_seed(0)
        q, k, v = (
            torch.randn(1, heads, rows, 256, generator=generator).half().float()
            for heads, rows in ((24, 1), (4, 256), (4, 256))
        )
        self._check(Lambda(attention), (q, k, v, torch.tensor([200])))

    def _check_traced_integer_compare(self, fn):
        small = (torch.tensor([[7, 3, -5]]), torch.tensor([[6, 4, -5]]))
        # 2^24 + 1 rounds to 2^24 in F32, which would then compare equal.
        large = (torch.tensor([[16777217, 3]]), torch.tensor([[16777216, 4]]))
        self._check(Lambda(lambda a, b: fn(a, b).float()), small)
        with self.assertRaisesRegex(AssertionError, r"2\^24"):
            self._check(Lambda(lambda a, b: fn(a, b).float()), large)
        with self.assertRaisesRegex(AssertionError, r"2\^24"):
            self._check(Lambda(lambda a: fn(a, 16777217).float()), (small[0],))

    def _check_host_computed_integer_compare(self, fn):
        # Integers the host computes from inputs, as attention positions are,
        # pass when every step stays within 2^24.
        self._check(
            Lambda(lambda a, b: fn(a + a, b).float()),
            (torch.tensor([[1, 2]]), torch.tensor([[2, 2]])),
        )
        # x + 1 rounds to x = 2^24 in F32, where (x + 1) <= x holds.
        x = (torch.tensor([[16777216, 3]]),)
        with self.assertRaisesRegex(AssertionError, r"2\^24"):
            self._check(Lambda(lambda x: fn(x + 1, x).float()), x)
        with self.assertRaisesRegex(AssertionError, r"2\^24"):
            self._check(Lambda(lambda x: fn(x + 1 - 1, x - 1).float()), x)
        with self.assertRaisesRegex(AssertionError, "must be an integer"):
            self._check(
                Lambda(lambda a, b: fn(a * a, b).float()),
                (torch.tensor([[7, 3, -5]]), torch.tensor([[6, 4, -5]])),
            )

    def test_ne_of_integers_checks_that_f32_holds_them(self):
        # 2^24 + 1 rounds to 2^24 in F32.
        with self.assertRaisesRegex(AssertionError, r"2\^24"):
            self._check(
                Lambda(lambda a, b: torch.ne(a, b).float()),
                (torch.tensor([[16777217, 3]]), torch.tensor([[16777216, 4]])),
            )
        with self.assertRaisesRegex(AssertionError, r"2\^24"):
            self._check(
                Lambda(lambda a: torch.ne(a, 16777217).float()),
                (torch.tensor([[16777216, 4]]),),
            )

    def test_le_checks_traced_integers(self):
        self._check_traced_integer_compare(torch.le)

    def test_lt_checks_traced_integers(self):
        self._check_traced_integer_compare(torch.lt)

    def test_le_checks_host_computed_integers(self):
        self._check_host_computed_integer_compare(torch.le)

    def test_lt_checks_host_computed_integers(self):
        self._check_host_computed_integer_compare(torch.lt)

    def test_ne_checks_host_computed_integers(self):
        self._check_host_computed_integer_compare(torch.ne)

    def test_integer_compare_refuses_unbounded_operands_at_compile(self):
        def compile_ggml(model, inputs):
            # The runner refuses the surplus input of the one step only once
            # the method is compiled.
            with tempfile.TemporaryDirectory() as tmp:
                outdir = Path(tmp)
                package = outdir / "model.ptn"
                to_native(torch.export.export(model, inputs)).save(str(package))
                _run_ggml(package, [(*inputs, inputs[0])], outdir, 1)

        class BufferCompare(torch.nn.Module):
            def __init__(self, fn, values, mutate):
                super().__init__()
                self.fn = fn
                self.mutate = mutate
                self.register_buffer("buffer", torch.tensor(values))

            def forward(self, x):
                result = self.fn(self.buffer, x).float()
                if self.mutate:
                    self.buffer.add_(1)
                return result

        a, b = torch.tensor([[1, 2]]), torch.tensor([[2, 9]])
        for name, fn in {"le": torch.le, "lt": torch.lt, "ne": torch.ne}.items():
            cases = {
                "x * 3": (Lambda(lambda a, b, fn=fn: fn(a * 3, b).float()), (a, b)),
                "mutable buffer": (BufferCompare(fn, [[1, 2]], True), (b,)),
            }
            for case, (model, inputs) in cases.items():
                with self.subTest(op=name, case=case):
                    with self.assertRaisesRegex(AssertionError, "must be an integer"):
                        compile_ggml(model, inputs)
            with self.subTest(op=name, case="constant over 2^24"):
                with self.assertRaisesRegex(AssertionError, r"2\^24"):
                    compile_ggml(BufferCompare(fn, [[2**25, 2]], False), (b,))
            with self.subTest(op=name, case="adds and subtracts"):
                self._check(
                    Lambda(lambda a, b, fn=fn: fn(a + b - 3, b).float()), (a, b)
                )

    def test_integer_compare_rechecks_its_inputs_on_every_run(self):
        steps = [
            (torch.tensor([[1, 2]]), torch.tensor([[2, 2]])),
            (torch.tensor([[16777217, 2]]), torch.tensor([[16777216, 2]])),
        ]
        for model in (
            Lambda(lambda a, b: (a <= b).float()),
            Lambda(lambda a, b: (a + 1 <= b).float()),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                outdir = Path(tmp)
                package = outdir / "model.ptn"
                to_native(torch.export.export(model, steps[0])).save(str(package))
                # The second run replays the first run's graph.
                with self.assertRaisesRegex(AssertionError, r"2\^24"):
                    _run_ggml(package, steps, outdir, 1)

    def test_le_and_lt_round_to_half_types_as_eager(self):
        x = torch.tensor([[0.5, 0.25, 0.75]])
        cases = {
            "le": lambda x: (x <= 0.49999).float(),
            "lt": lambda x: (x < 0.50001).float(),
        }
        for dtype in (torch.float16, torch.bfloat16):
            for name, fn in cases.items():
                with self.subTest(op=name, dtype=dtype):
                    self._check(Lambda(fn), (x.to(dtype),), atol=0, rtol=0)

    def test_empty_tensors_are_refused(self):
        # Eager splits an empty dim into one empty piece.
        cases = {
            "split": lambda x: torch.split(x, 2)[0] * 2.0,
            "mul": lambda x: x * 2.0,
        }
        for name, fn in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(
                AssertionError, "without elements"
            ):
                self._check(Lambda(fn), (torch.randn(0, 2),))

    def test_split(self):
        x = torch.randn(4, 6, 5)
        cases = {
            "sizes last dim": lambda x: torch.split(x, [2, 1, 2], dim=-1),
            "sizes middle dim": lambda x: torch.split(x, [2, 3, 1], dim=1),
            "size outer dim": lambda x: torch.split(x, 3, dim=0),
            "one piece used": lambda x: torch.split(x, [2, 4], dim=1)[1] * 2.0,
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                self._check(Lambda(fn), (x,))

    def test_topk(self):
        # Distinct values: ties may come out in another order.
        x = torch.randperm(240).float().reshape(2, 3, 4, 10) / 7
        for largest in (True, False):
            with self.subTest(largest=largest):
                ops = collections.Counter()
                self._check(
                    Lambda(
                        lambda x, largest=largest: torch.topk(x, 4, largest=largest)
                    ),
                    (x,),
                    graph_ops=ops,
                )
                # llama.cpp's ggml_argsort_top_k: one argsort and the values'
                # gather per built graph, and no NaN key map.
                self.assertEqual(ops["ARGSORT"], ops["GET_ROWS"])
                self.assertEqual((ops["CONCAT"], ops["FILL"]), (0, 0))

    def test_topk_rejects_an_inner_dim(self):
        with self.assertRaisesRegex(AssertionError, "last dimension only"):
            self._check(Lambda(lambda x: torch.topk(x, 2, dim=0)), (torch.randn(4, 3),))

    def test_topk_ranks_nan_above_every_value_for_testing(self):
        # Only with exact_topk_for_testing: by default topk sorts as
        # llama.cpp, with no order for a row holding NaN.
        nan, inf = float("nan"), float("inf")
        x = torch.tensor(
            [
                [1.0, nan, -inf, 3.0, inf, 0.5],
                [inf, -2.0, nan, -inf, 4.0, 2.0],
                [nan, 0.0, -1.0, 5.0, -inf, inf],
            ]
        )
        for largest in (True, False):
            with self.subTest(largest=largest):
                self._check(
                    Lambda(
                        lambda x, largest=largest: torch.topk(x, 4, largest=largest)
                    ),
                    (x,),
                    atol=0,
                    rtol=0,
                    exact_topk_for_testing=True,
                )

    def test_topk_of_integers_checks_that_f32_holds_them(self):
        def topk(x):
            return torch.topk(x, 2).values

        self._check(Lambda(topk), (torch.tensor([[4, 9, -1]]),))
        with self.assertRaisesRegex(AssertionError, r"2\^24"):
            self._check(Lambda(topk), (torch.tensor([[16777217, 2, 1]]),))

    def test_symbolic_neg_slices_from_the_end(self):
        class Tail(torch.nn.Module):
            def forward(self, x, buffer):
                return buffer[:, -x.shape[1] :] + x

        model = Tail()
        steps = [(torch.randn(2, n), torch.randn(2, 8)) for n in (5, 1, 7)]
        # At n = 8 the slice would start at 0, which export specializes.
        ep = torch.export.export(
            model,
            steps[0],
            dynamic_shapes={"x": {1: Dim("n", min=1, max=7)}, "buffer": None},
        )
        self.assertIn(
            "<built-in function neg>",
            {str(n.target) for n in ep.graph.nodes if n.op == "call_function"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(ep).save(str(package))
            actual = _run_ggml(package, steps, outdir, 1, resize=True)
        for inputs, [(raw, shape)] in zip(steps, actual):
            expected = model(*inputs)
            self.assertEqual(shape, tuple(expected.shape))
            torch.testing.assert_close(
                torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape),
                expected,
            )

    def test_sdpa_mask_broadcasts_over_queries(self):
        # A [B, 1, 1, S] padding mask: every query reads the same row.
        keep = torch.rand(1, 1, 1, 8)
        keep[..., 0] = 0.0
        self._check(
            Lambda(
                lambda q, k, v, m: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=m.le(0.7)
                )
            ),
            (
                torch.randn(1, 2, 3, 32),
                torch.randn(1, 2, 8, 32),
                torch.randn(1, 2, 8, 32),
                keep,
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_attention_reads_head_transposes_without_copies(self):
        # The [B, T, H, D] -> [B, H, T, D] views of q, k and v, and the output
        # transposed back, are read through their strides.
        def forward(q, k, v):
            def heads(t):
                return t.view(1, 6, 2, 32).transpose(1, 2)

            out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
            return out.transpose(1, 2).reshape(1, 6, 64)

        ops = collections.Counter()
        self._check(
            Lambda(forward),
            tuple(torch.randn(1, 6, 64) for _ in range(3)),
            atol=1e-2,
            rtol=1e-2,
            graph_ops=ops,
        )
        self.assertEqual(ops["CONT"], 0)

    def test_cast_within_a_ggml_type_keeps_a_transpose(self):
        # int64 and int32 are both I32 in ggml, so the cast reads the
        # transpose as a view and the transpose back undoes it.
        ops = collections.Counter()
        self._check(
            Lambda(lambda x: x.transpose(0, 1).to(torch.int32).transpose(0, 1)),
            (torch.arange(6).reshape(2, 3),),
            graph_ops=ops,
        )
        self.assertEqual(ops["CONT"], 0)

    def test_readers_of_a_transpose_share_one_copy(self):
        def forward(x):
            t = x.transpose(1, 2)
            return t + 1.0, t * 2.0

        ops = collections.Counter()
        self._check(Lambda(forward), (torch.randn(2, 3, 4),), graph_ops=ops)
        # Each graph build copies the transpose once for both readers.
        self.assertEqual(ops["CONT"], ops["ADD"])

    def test_fp16_kv_attention_reads_the_f32_query(self):
        # Flash attention rounds an F32 query to the F16 keys' precision
        # itself, so the query's cast to the cache dtype is dropped: each
        # attention casts only k and v, and rounds its F32 result to the F16
        # output and back for .float().
        def forward(q, k, v):
            return F.scaled_dot_product_attention(q.half(), k.half(), v.half()).float()

        ops = collections.Counter()
        self._check(
            Lambda(forward),
            (
                torch.randn(1, 2, 3, 32),
                torch.randn(1, 2, 8, 32),
                torch.randn(1, 2, 8, 32),
            ),
            atol=1e-2,
            rtol=1e-2,
            graph_ops=ops,
        )
        self.assertEqual(ops["CPY"], 4 * ops["FLASH_ATTN_EXT"])

    def test_fp16_kv_attention_keeps_the_cast_of_a_written_query(self):
        # The query is cast from a buffer that is then written, so attention
        # must read the cast from before the write, not the buffer.
        class WrittenQuery(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.randn(1, 2, 3, 32))

            def forward(self, k, v, rows, positions):
                q = self.state.half()
                self.state.index_copy_(2, positions, rows)
                return F.scaled_dot_product_attention(q, k.half(), v.half()).float()

        model = WrittenQuery()
        inputs = (
            torch.randn(1, 2, 8, 32),
            torch.randn(1, 2, 8, 32),
            torch.randn(1, 2, 1, 32) * 4.0,
            torch.tensor([1]),
        )
        (actual,) = self._run_exported(model, inputs, inputs)
        with torch.no_grad():
            expected = model(*inputs)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_sdpa_reuses_computed_bool_mask(self):
        def forward(q, k, v, mask_input):
            mask = mask_input.le(0.7)
            first = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            second = F.scaled_dot_product_attention(q, k, v * 2.0, attn_mask=mask)
            return first + second

        ops = collections.Counter()
        self._check(
            Lambda(forward),
            (
                torch.randn(1, 2, 3, 32),
                torch.randn(1, 2, 8, 32),
                torch.randn(1, 2, 8, 32),
                torch.rand(1, 2, 3, 8),
            ),
            atol=1e-2,
            rtol=1e-2,
            graph_ops=ops,
        )
        # One F16 conversion of the shared mask serves both attention ops.
        self.assertEqual(ops["CPY"] * 2, ops["FLASH_ATTN_EXT"])

    def test_sdpa_reuses_bool_mask_through_views(self):
        def forward(q, k, v, mask_input):
            mask = mask_input.le(0.7)[..., :8]
            first = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            second = F.scaled_dot_product_attention(q, k, v * 2.0, attn_mask=mask)
            return first + second

        self._check(
            Lambda(forward),
            (
                torch.randn(1, 2, 3, 32),
                torch.randn(1, 2, 8, 32),
                torch.randn(1, 2, 8, 32),
                torch.rand(1, 2, 3, 9),
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_sdpa_reuses_product_of_bool_masks(self):
        def forward(q, k, v, a, b):
            mask = a.le(0.7) * b.le(0.7)
            first = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            second = F.scaled_dot_product_attention(q, k, v * 2.0, attn_mask=mask)
            return first + second

        a = torch.rand(1, 2, 3, 8)
        a[..., 0] = 0.0
        b = torch.rand(1, 2, 3, 8)
        b[..., 0] = 0.0
        ops = collections.Counter()
        self._check(
            Lambda(forward),
            (
                torch.randn(1, 2, 3, 32),
                torch.randn(1, 2, 8, 32),
                torch.randn(1, 2, 8, 32),
                a,
                b,
            ),
            atol=1e-2,
            rtol=1e-2,
            graph_ops=ops,
        )
        # One F16 conversion of the shared mask serves both attention ops.
        self.assertEqual(ops["CPY"] * 2, ops["FLASH_ATTN_EXT"])

    def test_sdpa_key_prefix_of_a_transposed_key(self):
        # When every query masks the keys past its position, attention covers
        # only the 256-key buckets of k and v up to the last position; k
        # arrives with its head dim strided. The last mask attends the keys
        # from each position on, so no prefix may be taken.
        masks = {
            "key <= position": lambda key, pos: key[None, :] <= pos[:, None],
            "key < position": lambda key, pos: key[None, :] < pos[:, None],
            "position <= key": lambda key, pos: pos[:, None] <= key[None, :],
        }
        tensors = (
            torch.randn(1, 2, 2, 32),
            torch.randn(1, 2, 32, 512),
            torch.randn(1, 2, 512, 32),
        )
        steps = [(*tensors, torch.tensor(p)) for p in ([3, 4], [300, 301])]
        for name, mask in masks.items():
            with self.subTest(mask=name):

                def forward(q, k_t, v, positions, mask=mask):
                    return F.scaled_dot_product_attention(
                        q,
                        k_t.transpose(2, 3),
                        v,
                        attn_mask=mask(torch.arange(512), positions),
                    )

                model = Lambda(forward).eval()
                ops = collections.Counter()
                with tempfile.TemporaryDirectory() as tmp:
                    outdir = Path(tmp)
                    package = outdir / "model.ptn"
                    to_native(torch.export.export(model, steps[0])).save(str(package))
                    actual = _run_ggml(package, steps, outdir, 1, graph_ops=ops)
                for step, [(raw, shape)] in zip(steps, actual):
                    with torch.no_grad():
                        expected = model(*step)
                    torch.testing.assert_close(
                        torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(
                            shape
                        ),
                        expected,
                        atol=1e-2,
                        rtol=1e-2,
                    )
                # The compile-time graph and the first step's; a prefix in the
                # first step makes the second step's wider extent build a third.
                prefix = name != "position <= key"
                self.assertEqual(ops["FLASH_ATTN_EXT"], 3 if prefix else 2)

    def test_sdpa_masks_over_caches_longer_than_the_host_computes(self):
        # The host bounds the keys' positions by the arange's arguments once,
        # and positions - window, which a graph input computes, on every run:
        # in the graph compile builds and in the run's.
        for window, checks in ((None, 0), (4, 2)):
            with self.subTest(window=window):
                ops = collections.Counter()
                self._check(
                    LongCacheMask(131072, window),
                    (torch.randn(1, 2, 3, 32), torch.tensor([7, 8, 9])),
                    graph_ops=ops,
                )
                self.assertEqual(ops["host integer checks"], checks)

    def test_sdpa_mask_positions_written_to_a_cache_need_no_per_run_check(self):
        # Every run checks the positions a cache is written at against its
        # rows, which bounds the query positions computed from them too.
        inputs = (
            torch.randn(1, 2, 3, 32),
            torch.randn(1, 2, 3, 32),
            torch.tensor([5, 6, 7]),
        )
        # Compile builds a graph, and so does the run.
        for rows, checks in ((None, 0), (torch.tensor([1, 2, 3]), 2)):
            with self.subTest(rows=rows):
                ops = collections.Counter()
                written = inputs if rows is None else (*inputs, rows)
                self._check(WrittenCacheMask(16), written, graph_ops=ops)
                self.assertEqual(ops["host integer checks"], checks)

    def test_cache_positions_past_the_cache_fail_before_any_compare(self):
        # The positions bounding the mask are the cache write's row indices,
        # rejected on the first run and on a replay, cached or not.
        def step(positions):
            return (torch.randn(1, 2, 3, 32), torch.randn(1, 2, 3, 32), positions)

        good, bad = step(torch.tensor([5, 6, 7])), step(torch.tensor([14, 15, 16]))
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            exported = torch.export.export(WrittenCacheMask(16), good)
            to_native(exported).save(str(package))
            for cached in (1, 0):
                for steps in ([bad], [good, bad]):
                    with self.subTest(cached=cached, steps=len(steps)):
                        with self.assertRaisesRegex(
                            AssertionError, r"row index is outside \[0, 16\)"
                        ):
                            _run_ggml(
                                package, steps, outdir, 1, max_cached_graphs=cached
                            )

    def test_compare_checks_shape_bounded_integers_once_per_graph(self):
        n = Dim("n", min=1, max=8)
        cases = {
            # arange(n) + 3 is bounded by n's size in each graph.
            "arange + 3 <= i": (lambda x, i: (torch.arange(x.shape[1]) + 3 <= i), 0),
            # i + 3 is bounded only by i's values in each run.
            "i + 3 <= arange": (lambda x, i: (i + 3 <= torch.arange(x.shape[1])), 1),
        }
        steps = [
            (torch.randn(1, 5), torch.tensor([[6]])),
            (torch.randn(1, 3), torch.tensor([[1]])),
        ]

        class Masked(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x, i):
                return x * self.fn(x, i).float()

        for name, (fn, checks) in cases.items():
            with self.subTest(case=name):
                model = Masked(fn)
                ops = collections.Counter()
                with tempfile.TemporaryDirectory() as tmp:
                    outdir = Path(tmp)
                    package = outdir / "model.ptn"
                    ep = torch.export.export(
                        model, steps[0], dynamic_shapes={"x": {1: n}, "i": None}
                    )
                    to_native(ep).save(str(package))
                    outputs = _run_ggml(
                        package, steps, outdir, 1, resize=True, graph_ops=ops
                    )
                # Compile builds a graph, then each shape its own.
                self.assertEqual(ops["host integer checks"], 3 * checks)
                for ((raw, shape),), inputs in zip(outputs, steps):
                    actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                    torch.testing.assert_close(actual.reshape(shape), model(*inputs))

    def test_sdpa_key_prefix_of_overflowing_positions(self):
        # The host folds positions + 1 for the key extent before the engine
        # range-checks the int64 input, so the fold must not overflow.
        def forward(q, k, v, positions):
            mask = torch.arange(512)[None, :] <= (positions + 1)[:, None]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        example = (
            torch.randn(1, 2, 2, 32),
            torch.randn(1, 2, 512, 32),
            torch.randn(1, 2, 512, 32),
            torch.tensor([3, 4]),
        )
        with self.assertRaisesRegex(AssertionError, "does not fit in int32"):
            self._run_exported(
                Lambda(forward), example, (*example[:3], torch.tensor([3, 2**63 - 1]))
            )

    def test_sdpa_key_prefix_of_masks_that_share_values(self):
        # A sliding-window mask, the causal mask and'ed with a window bound as a
        # product of bools, shares its causal comparison with a global layer's
        # mask. Both attend only keys up to the last position, so attention
        # covers the 256-key buckets up to it.
        def forward(q, k, v, positions):
            keys = torch.arange(512)
            causal = keys[None, :] <= positions[:, None]
            sliding = causal * ((positions[:, None] - 128) < keys[None, :])
            first = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            second = F.scaled_dot_product_attention(q, k, v * 2.0, attn_mask=sliding)
            return first + second

        tensors = (
            torch.randn(1, 2, 2, 64),
            torch.randn(1, 2, 512, 64),
            torch.randn(1, 2, 512, 64),
        )
        ops = collections.Counter()
        self._check_steps(
            Lambda(forward).eval(),
            [(*tensors, torch.tensor(p)) for p in ([3, 4], [300, 301])],
            graph_ops=ops,
        )
        # The compile-time graph and one per step: the first step attends one
        # bucket, so the second step's wider extent builds a third graph.
        self.assertEqual(ops["FLASH_ATTN_EXT"], 3 * 2)

    def test_sdpa_key_prefix_of_a_mask_group_is_its_widest(self):
        # Masks that share the causal comparison end in different buckets, so
        # their group is computed over the causal mask's wider prefix.
        def forward(q, k, v, positions):
            keys = torch.arange(1024)
            causal = keys[None, :] <= positions[:, None]
            early = causal * (keys[None, :] <= positions[:, None] - 300)
            first = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            second = F.scaled_dot_product_attention(q, k, v * 2.0, attn_mask=early)
            return first + second

        tensors = (
            torch.randn(1, 2, 2, 64),
            torch.randn(1, 2, 1024, 64),
            torch.randn(1, 2, 1024, 64),
        )
        self._check_steps(
            Lambda(forward).eval(),
            [(*tensors, torch.tensor(p)) for p in ([400, 401], [700, 701])],
        )

    def test_sdpa_key_prefix_of_a_mask_read_by_other_ops(self):
        # The causal comparison also feeds a sum over every key, so attention
        # covers all keys.
        def forward(q, k, v, positions):
            causal = torch.arange(512)[None, :] <= positions[:, None]
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            return attended + (causal * 1.0).sum(-1, keepdim=True)

        tensors = (
            torch.randn(1, 2, 2, 64),
            torch.randn(1, 2, 512, 64),
            torch.randn(1, 2, 512, 64),
        )
        ops = collections.Counter()
        self._check_steps(
            Lambda(forward).eval(),
            [(*tensors, torch.tensor(p)) for p in ([3, 4], [300, 301])],
            graph_ops=ops,
        )
        # The compile-time graph and one that both steps share.
        self.assertEqual(ops["FLASH_ATTN_EXT"], 2)

    def test_sdpa_key_prefix_of_constant_key_positions(self):
        # A static KV cache's key positions, held by a constant, evaluate on
        # the host.
        class ConstantKeys(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("key_positions", torch.arange(512))

            def forward(self, q, k, v, positions):
                mask = self.key_positions[None, :] <= positions[:, None]
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        tensors = (
            torch.randn(1, 2, 2, 64),
            torch.randn(1, 2, 512, 64),
            torch.randn(1, 2, 512, 64),
        )
        ops = collections.Counter()
        self._check_steps(
            ConstantKeys().eval(),
            [(*tensors, torch.tensor(p)) for p in ([3, 4], [300, 301])],
            graph_ops=ops,
        )
        # The compile-time graph and one per step: the second step's wider
        # extent builds a third.
        self.assertEqual(ops["FLASH_ATTN_EXT"], 3)

    def test_sdpa_causal_fewer_queries(self):
        self._check(
            Lambda(
                lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
            ),
            (
                torch.randn(1, 2, 3, 32),
                torch.randn(1, 2, 5, 32),
                torch.randn(1, 2, 5, 32),
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    def _check_methods_sharing_a_cache(self, persistent, names):
        """Static prefill and decode methods write one cache that a third reads,
        all in one context, as in eager where the modules share the tensor.
        `names` names the method of each step; prefill and decode steps after
        the first of each write the given positions."""
        initial = torch.randn(1, 2, 8, 4) if persistent else torch.zeros(1, 2, 8, 4)

        def methods(cache):
            return {
                "prefill": SharedCacheWrite(cache, persistent),
                "decode": SharedCacheWrite(cache, persistent),
                "read": SharedCacheRead(cache, persistent),
            }

        examples = {
            "prefill": (torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6])),
            "decode": (torch.randn(1, 2, 1, 4), torch.tensor([0])),
            "read": (torch.randn(1, 2, 8, 4),),
        }
        later = {
            "prefill": (torch.randn(1, 2, 3, 4), torch.tensor([2, 3, 4])),
            "decode": (torch.randn(1, 2, 1, 4), torch.tensor([7])),
            "read": (torch.randn(1, 2, 8, 4),),
        }
        steps = [
            (name, later[name] if name in names[:i] else examples[name])
            for i, name in enumerate(names)
        ]
        programs = {
            name: torch.export.export(module, examples[name])
            for name, module in methods(initial.clone()).items()
            if name in names
        }
        eager = methods(initial.clone())
        with torch.no_grad():
            expected = [eager[name](*inputs) for name, inputs in steps]
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(programs).save(str(package))
            outputs = _run_ggml(
                package,
                [inputs for _, inputs in steps],
                outdir,
                1,
                method=";".join(name for name, _ in steps),
            )
        for ((raw, shape),), reference in zip(outputs, expected):
            actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
            torch.testing.assert_close(actual.reshape(shape), reference)

    def test_methods_share_a_persistent_cache(self):
        self._check_methods_sharing_a_cache(
            True, ["read", "prefill", "read", "decode", "decode", "prefill", "read"]
        )

    # A method only reading a zero-initialized buffer exports it as a constant,
    # unlike the methods writing it, so the package is refused.
    def test_methods_share_a_zero_initialized_cache(self):
        self._check_methods_sharing_a_cache(
            False, ["decode", "prefill", "decode", "prefill"]
        )

    def test_index_put_cache_write(self):
        self._check(CacheWrite(), (torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6])))

    def test_write_back_reads_a_slice_without_copies(self):
        ops = collections.Counter()
        self._check(WindowWriteBack(), (torch.randn(1, 6, 3),), graph_ops=ops)
        self.assertEqual(ops["CONT"], 0)

    def test_elementwise_ops_read_slices_without_copies(self):
        ops = collections.Counter()
        self._check(
            Lambda(lambda x: x[..., 1:] * x[..., :-1] + torch.sigmoid(x[..., 1:])),
            (torch.randn(2, 3, 8),),
            graph_ops=ops,
        )
        self.assertEqual(ops["CONT"], 0)

    def test_cached_graph_replays_after_another_methods_split(self):
        # On a GPU, attention with head size 32 runs on the CPU backend, so
        # ggml's scheduler splits each graph and points the attention at
        # copies of its inputs that it owns. The context's one scheduler
        # splits "short" between two calls of "long", whose cached graph must
        # still compute what a fresh graph does.
        model = Lambda(
            lambda q, k, v: F.scaled_dot_product_attention(q * 2.0, k, v) + 1.0
        )
        kv = (torch.randn(1, 4, 16, 32), torch.randn(1, 4, 16, 32))
        examples = {
            "long": (torch.randn(1, 4, 8, 32), *kv),
            "short": (torch.randn(1, 4, 1, 32), *kv),
        }
        names = ["long", "short", "long", "short", "long"]
        steps = [examples[name] for name in names]
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "attention.ptn"
            to_native(
                {
                    name: torch.export.export(model, inputs)
                    for name, inputs in examples.items()
                }
            ).save(str(package))
            fresh, cached = (
                _run_ggml(
                    package,
                    steps,
                    outdir,
                    1,
                    method=";".join(names),
                    max_cached_graphs=graphs,
                )
                for graphs in (0, 2)
            )
        self.assertEqual(cached, fresh)
        with torch.no_grad():
            for ((raw, shape),), inputs in zip(cached, steps):
                actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                torch.testing.assert_close(
                    actual.reshape(shape), model(*inputs), **_MATMUL_TOL
                )

    def test_index_put_reads_transposed_values_without_copies(self):
        ops = collections.Counter()
        self._check(
            TransposedCacheWrite(),
            (torch.randn(1, 3, 2, 4), torch.tensor([5, 1, 6])),
            graph_ops=ops,
        )
        self.assertEqual(ops["CONT"], 0)

    def test_graph_with_as_many_leafs_as_nodes(self):
        # 2,100 additions of separate constants fit the default graph size of
        # 4,096 nodes, but the scheduler also hashes their 2,100 leafs.
        constants = [torch.randn(4) for _ in range(2100)]

        def forward(x):
            for constant in constants:
                x = x + constant
            return x

        self._check(Lambda(forward), (torch.randn(4),))

    def test_cached_graphs_refresh_inputs_and_persistent_cache(self):
        t = Dim("t", min=1, max=8)
        ep = torch.export.export(
            CacheWrite(),
            (torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6])),
            dynamic_shapes={"values": {2: t}, "positions": {0: t}},
        )
        # Alternating shapes evict each other with one cached graph and replay
        # with two.
        steps = [
            (torch.full((1, 2, 3, 4), 1.0), torch.tensor([5, 1, 6])),
            (torch.full((1, 2, 1, 4), 2.0), torch.tensor([3])),
            (torch.full((1, 2, 3, 4), 3.0), torch.tensor([6, 0, 7])),
            (torch.full((1, 2, 1, 4), 4.0), torch.tensor([1])),
            (torch.full((1, 2, 1, 4), 5.0), torch.tensor([2])),
        ]
        expected, cache = [], torch.zeros(1, 2, 8, 4)
        for values, positions in steps:
            cache.index_copy_(2, positions, values)
            expected.append(cache.clone())
        # Each graph build, the one at compile included, has one MUL (the
        # output's `cache * 1.0`).
        for max_cached_graphs, builds in ((0, 6), (1, 5), (2, 3)):
            with self.subTest(max_cached_graphs=max_cached_graphs):
                ops = collections.Counter()
                with tempfile.TemporaryDirectory() as tmp:
                    outdir = Path(tmp)
                    package = outdir / "model.ptn"
                    to_native(ep).save(str(package))
                    actual = _run_ggml(
                        package,
                        steps,
                        outdir,
                        1,
                        resize=True,
                        max_cached_graphs=max_cached_graphs,
                        graph_ops=ops,
                    )
                self.assertEqual(ops["MUL"], builds)
                for [(raw, shape)], reference in zip(actual, expected):
                    self.assertEqual(shape, tuple(reference.shape))
                    torch.testing.assert_close(
                        torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(
                            shape
                        ),
                        reference,
                        rtol=0,
                        atol=0,
                    )

    def test_cached_graph_rechecks_row_indices(self):
        inputs = (torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6]))
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(CacheWrite(), inputs)).save(str(package))
            with self.assertRaisesRegex(AssertionError, "outside"):
                _run_ggml(
                    package,
                    [inputs, (inputs[0], torch.tensor([5, 1, 8]))],
                    outdir,
                    1,
                )

    def test_fp16_cache_write_skips_cast_unless_source_may_change(self):
        class FreshSource(HalfCacheWrite):
            def forward(self, values, positions):
                self.cache.index_copy_(2, positions, (values * 2.0).to(torch.float16))
                return self.cache * 1.0

        class BufferSource(HalfCacheWrite):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.randn(1, 2, 3, 4))

            def forward(self, values, positions):
                self.cache.index_copy_(2, positions, self.state.to(torch.float16))
                return self.cache * 1.0, values * 2.0

        class WrittenSource(HalfCacheWrite):
            def forward(self, values, positions, row):
                rows = values * 2.0
                rows.index_copy_(2, row, values[:, :, :1])
                self.cache.index_copy_(2, positions, rows.to(torch.float16))
                return self.cache * 1.0

        values, positions = torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6])
        # Per graph build (two MULs each): the F32 -> F16 cast of the cache
        # rows when it is kept. Casts of constants, the buffer and the scalar
        # factors export lifts to tensors, are computed once, at compile.
        cases = {
            "fresh": (FreshSource(), (values, positions), 0),
            "buffer": (BufferSource(), (values, positions), 0),
            "written": (WrittenSource(), (values, positions, torch.tensor([0])), 1),
        }
        for source, (model, inputs, casts) in cases.items():
            with self.subTest(source=source):
                ops = collections.Counter()
                self._check(model, inputs, graph_ops=ops)
                self.assertEqual(ops["CPY"], casts * ops["MUL"] // 2)

    def test_reused_graph_fp16_cache_cast_and_write(self):
        model = HalfCacheWrite()
        steps = [
            (torch.full((1, 2, 3, 4), 1.0 / 3), torch.tensor([5, 1, 6])),
            (torch.full((1, 2, 3, 4), 2.0 / 7), torch.tensor([2, 3, 4])),
            (torch.full((1, 2, 3, 4), -3.0 / 7), torch.tensor([6, 0, 7])),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(model, steps[0])).save(str(package))
            actual = _run_ggml(package, steps, outdir, 1)
        expected = torch.zeros(1, 2, 8, 4, dtype=torch.float16)
        for (values, positions), [(raw, shape)] in zip(steps, actual):
            expected.index_copy_(2, positions, values.to(torch.float16))
            self.assertEqual(shape, tuple(expected.shape))
            torch.testing.assert_close(
                torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(shape),
                expected,
                rtol=0,
                atol=0,
            )

    @staticmethod
    def _run_cached_attention(f32_gpu_kv_cache_options):
        """A prefill of 3 positions, then one decode step each at 3 and 4, run
        once per given f32_gpu_kv_cache_for_testing value; also returns eager's
        outputs."""
        t = Dim("t", min=1, max=8)
        steps = []
        for start, count in ((0, 3), (3, 1), (4, 1)):
            steps.append(
                (
                    torch.randn(1, 2, count, 64),
                    torch.randn(1, 2, count, 64),
                    torch.randn(1, 2, count, 64),
                    torch.arange(start, start + count),
                )
            )
        model = CachedAttention()
        ep = torch.export.export(
            CachedAttention(),
            steps[0],
            dynamic_shapes=({2: t}, {2: t}, {2: t}, {0: t}),
        )
        expected = [model(*inputs) for inputs in steps]
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(ep).save(str(package))
            runs = {
                f32: _run_ggml(
                    package,
                    steps,
                    outdir,
                    1,
                    resize=True,
                    f32_gpu_kv_cache_for_testing=f32,
                )
                for f32 in f32_gpu_kv_cache_options
            }
        return runs, expected

    def test_attention_over_written_caches(self):
        # On CUDA and HIP the engine stores these caches in F16.
        runs, expected = self._run_cached_attention((False,))
        for [(raw, shape)], reference in zip(runs[False], expected):
            self.assertEqual(shape, tuple(reference.shape))
            torch.testing.assert_close(
                torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape),
                reference,
                atol=1e-2,
                rtol=1e-2,
            )

    @requires_option_tests
    def test_f32_gpu_kv_cache_matches_f16_storage(self):
        # Flash attention rounds the F32 caches the testing option keeps to F16
        # on every call, as the F16 rows of the default storage are rounded
        # once: the results are equal bit for bit.
        runs, _ = self._run_cached_attention((False, True))
        for [(raw, _)], [(kept_raw, _)] in zip(runs[False], runs[True]):
            self.assertEqual(raw, kept_raw)

    def test_index_put_rejects_out_of_range(self):
        inputs = (torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6]))
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(CacheWrite(), inputs)).save(str(package))
            with self.assertRaisesRegex(AssertionError, "outside"):
                _run_ggml(package, [(inputs[0], torch.tensor([5, 1, 8]))], outdir, 1)

    def test_clone_of_intermediate_then_index_put(self):
        def forward(x, values, index):
            out = (x * 2.0).clone()
            out.index_copy_(2, index, values)
            return out

        self._check(
            Lambda(forward),
            (torch.randn(1, 2, 4, 2), torch.randn(1, 2, 1, 2), torch.tensor([1])),
        )

    def test_clone_shares_unwritten_storage(self):
        ops = collections.Counter()
        self._check(
            Lambda(lambda x: (x * 2.0).clone() + x), (torch.randn(2, 3),), graph_ops=ops
        )
        self.assertEqual(ops["CONT"] + ops["CPY"], 0)

    def test_clone_of_buffer_keeps_value_from_before_write_back(self):
        class SnapshotThenCopy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 3))

            def forward(self, x):
                old = self.state.clone()
                self.state.copy_(x + 1.0)
                return old

        inputs = (torch.ones(2, 3),)
        (old,) = self._run_exported(SnapshotThenCopy(), inputs, inputs)
        torch.testing.assert_close(old, torch.zeros(2, 3))

    def test_read_before_index_put_sees_old_rows(self):
        inputs = (torch.ones(1, 2, 1, 2), torch.tensor([1]))
        for clone in (False, True):
            with self.subTest(clone=clone):
                (old,) = self._run_exported(SnapshotThenWrite(clone), inputs, inputs)
                torch.testing.assert_close(old, torch.zeros(1, 2, 4, 2))

    def test_check_keeps_eager_writes_out_of_package(self):
        inputs = (torch.ones(1, 2, 1, 2), torch.tensor([1]))
        self._check(SnapshotThenWrite(clone=False), inputs)

    def test_index_put_checks_cast_indices(self):
        inputs = (torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6]))
        self._check(CastIndexCacheWrite(), inputs)
        with self.assertRaisesRegex(AssertionError, "outside"):
            self._run_exported(
                CastIndexCacheWrite(), inputs, (inputs[0], torch.tensor([5, 1, 8]))
            )

    def test_sdpa_rejects_zero_kv_heads(self):
        with self.assertRaisesRegex(AssertionError, "incompatible"):
            self._check(
                Lambda(
                    lambda q, k, v: F.scaled_dot_product_attention(
                        q, k, v, enable_gqa=True
                    )
                ),
                (
                    torch.randn(1, 4, 3, 8),
                    torch.randn(1, 0, 3, 8),
                    torch.randn(1, 0, 3, 8),
                ),
            )

    def test_arange_rejects_bounds_beyond_exact_f32(self):
        with self.assertRaisesRegex(AssertionError, "arange"):
            self._check(
                Lambda(
                    lambda x: x + torch.arange(16777216, 16777219, dtype=torch.float32)
                ),
                (torch.randn(3),),
            )

    def test_cat_int64_of_mixed_ggml_types(self):
        # The int64 input is I32 in ggml; the int64 arange result is F32.
        self._check(
            Lambda(lambda x: torch.cat([x, torch.arange(1, 7, 2)])),
            (torch.tensor([7, 8, 9]),),
        )

    def test_arange(self):
        self._check(
            Lambda(lambda x: x + torch.arange(1, 7, 2, dtype=torch.float32)),
            (torch.randn(2, 3),),
        )

    def test_rope_reuses_computed_position_cast(self):
        config = LlamaConfig(hidden_size=256, num_attention_heads=4, head_dim=64)
        ops = collections.Counter()
        self._check(
            RopeQK(config),
            (torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64), torch.tensor([3])),
            atol=1e-4,
            rtol=1e-4,
            graph_ops=ops,
        )
        # Per graph build (one ARANGE each): a cast of `start` to F32 and one
        # I32 cast of the positions, shared by both ropes.
        self.assertEqual(ops["CPY"], 2 * ops["ARANGE"])

    def test_rope_reads_head_transposes_without_copies(self):
        config = LlamaConfig(hidden_size=256, num_attention_heads=4, head_dim=64)
        ops = collections.Counter()
        self._check(
            RopeOfPermute(config, (0, 2, 1, 3)),
            (torch.randn(1, 5, 4, 64), torch.tensor([[3, 4, 5, 6, 7]])),
            atol=1e-4,
            rtol=1e-4,
            graph_ops=ops,
        )
        self.assertEqual(ops["CONT"], 0)

    def test_rope_copies_a_permute_that_moves_the_last_dim(self):
        config = LlamaConfig(hidden_size=256, num_attention_heads=4, head_dim=64)
        self._check(
            RopeOfPermute(config, (0, 1, 3, 2)),
            (torch.randn(1, 4, 64, 5), torch.tensor([[3, 4, 5, 6, 7]])),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_rope_of_a_per_head_rms_norm(self):
        # A per-head RMS norm transposed to [B, H, T, D] for RoPE and back, at
        # positions that reshape a vector: the rope reads the norm and the
        # vector themselves, which ggml-cuda can fuse.
        class NormRope(torch.nn.Module):
            def __init__(self):
                super().__init__()
                config = LlamaConfig(
                    hidden_size=256, num_attention_heads=4, head_dim=64
                )
                self.rope = Rope(config)
                self.weight = torch.nn.Parameter(torch.rand(64) + 0.5)

            def forward(self, x, positions):
                h = F.rms_norm(x.unflatten(-1, (4, 64)), (64,), self.weight, eps=1e-6)
                rotated = self.rope(h.transpose(1, 2), positions.unsqueeze(0))
                return rotated.transpose(1, 2)

        for tokens in (1, 3):
            with self.subTest(tokens=tokens):
                self._check(
                    NormRope(),
                    (torch.randn(1, tokens, 256), torch.arange(3, 3 + tokens)),
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_rope_of_as_many_heads_as_tokens(self):
        # The rope's [D, H, T] permute of the input then has the input's shape
        # but not its order, so the rope must read the permute.
        config = LlamaConfig(hidden_size=256, num_attention_heads=4, head_dim=64)
        self._check(
            Rope(config),
            (torch.randn(1, 4, 4, 64), torch.tensor([[3, 9, 27, 81]])),
            atol=1e-4,
            rtol=1e-4,
        )

    def _check_half_linear_sum(self, **options):
        # Accumulated in F16, each 2^-8 term vanishes next to the running sum:
        # 1024 instead of eager's 1032.
        row = torch.cat([torch.ones(1024), torch.full((2048,), 2.0**-8)])
        linear = torch.nn.Linear(3072, 4, bias=False).half()
        with torch.no_grad():
            linear.weight.fill_(1.0)
        x = row.repeat(64, 1).half()
        with torch.no_grad():
            expected = linear(x)
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "model.ptn"
            to_native(torch.export.export(linear, (x,))).save(str(package))
            (((raw, shape),),) = _run_ggml(package, [(x,)], Path(tmp), 1, **options)
        actual = torch.frombuffer(bytearray(raw), dtype=torch.float16)
        torch.testing.assert_close(actual.reshape(shape), expected, atol=0.0, rtol=0.0)

    @requires_option_tests
    def test_half_linear_accumulates_in_f32_when_requested(self):
        self._check_half_linear_sum(cuda_f32_accumulation_for_testing=True)

    def test_cuda_f32_accumulation_is_a_no_op_on_the_cpu(self):
        if os.environ.get("GGML_PTN_DEVICE", "cpu") != "cpu":
            self.skipTest("CPU behavior")
        self._check_half_linear_sum(cuda_f32_accumulation_for_testing=True)

    def test_half_linear_result_is_rounded_before_the_next_op(self):
        # 256 * 256 * 2 = 131072 overflows F16, so eager reads inf and inf * 0
        # is NaN; the F32 accumulation would give 0.
        linear = _half_linear([[256.0, 256.0], [0.0, 0.0]])
        self._check(
            Lambda(lambda x: linear(x) * 0.0),
            (torch.tensor([[256.0, 256.0]]).half(),),
        )

    def test_rope_of_a_bf16_linear(self):
        class LinearRope(torch.nn.Module):
            def __init__(self, config: LlamaConfig):
                super().__init__()
                self.proj = torch.nn.Linear(64, 64, bias=False)
                self.rope = Rope(config)

            def forward(self, x, position_ids):
                return self.rope(self.proj(x), position_ids)

        config = LlamaConfig(hidden_size=256, num_attention_heads=4, head_dim=64)
        self._check(
            LinearRope(config).to(torch.bfloat16),
            (
                torch.randn(1, 4, 5, 64).to(torch.bfloat16),
                torch.tensor([[3, 4, 5, 6, 7]]),
            ),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_rope_of_a_bf16_input(self):
        config = LlamaConfig(hidden_size=256, num_attention_heads=4, head_dim=64)
        self._check(
            Rope(config).to(torch.bfloat16),
            (
                torch.randn(1, 4, 5, 64).to(torch.bfloat16),
                torch.tensor([[3, 4, 5, 6, 7]]),
            ),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_pad_and_max_pool_of_half_intermediates(self):
        # A half linear or embedding result is rounded to its dtype, so a
        # lowering that computes in F32 only must widen it again.
        torch.manual_seed(0)
        cases = {
            "pad of a linear": (
                torch.nn.Linear(16, 64),
                lambda y: F.pad(y, (1, 2)),
                torch.randn(4, 16),
            ),
            "max_pool2d of a linear": (
                torch.nn.Linear(16, 64),
                lambda y: F.max_pool2d(y.view(4, 1, 8, 8), 2),
                torch.randn(4, 16),
            ),
            "pad of an embedding": (
                torch.nn.Embedding(10, 8),
                lambda y: F.pad(y, (0, 3)),
                torch.tensor([[1, 7, 3]]),
            ),
        }
        for name, (module, fn, x) in cases.items():
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(case=name, dtype=dtype):
                    half = copy.deepcopy(module).to(dtype).requires_grad_(False)
                    inputs = (x.to(dtype) if x.is_floating_point() else x,)
                    self._check(
                        Lambda(lambda x, half=half, fn=fn: fn(half(x))),
                        inputs,
                        atol=1e-2,
                        rtol=1e-2,
                    )

    def test_reductions_and_sqrt_of_half_linears(self):
        # A half linear result is rounded to its dtype, so these lowerings,
        # which compute in F32 only, must widen it again.
        torch.manual_seed(0)
        linear = torch.nn.Linear(16, 8)
        with torch.no_grad():
            linear.weight.abs_()
            linear.bias.abs_()
        x = torch.rand(4, 16)
        cases = {
            "mean": lambda y: y.mean(-1),
            "sum": lambda y: y.sum(-1),
            "amax": lambda y: y.amax(-1),
            "sqrt": torch.sqrt,
            "softmax": lambda y: y.softmax(-1),
        }
        for name, fn in cases.items():
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(op=name, dtype=dtype):
                    half = copy.deepcopy(linear).to(dtype).requires_grad_(False)
                    self._check(
                        Lambda(lambda x, half=half, fn=fn: fn(half(x))),
                        (x.to(dtype),),
                        atol=1e-2,
                        rtol=1e-2,
                    )

    def test_layer_norm_and_hardtanh_of_half_intermediates(self):
        # A half linear or embedding result is rounded to its dtype, so
        # layer_norm, which computes in F32 only, and hardtanh, which ggml-cpu
        # clamps in F32 and F16 only, must widen it again.
        torch.manual_seed(0)
        norm = torch.nn.LayerNorm(64)
        torch.nn.init.uniform_(norm.weight, 0.5, 1.5)
        torch.nn.init.uniform_(norm.bias, -0.5, 0.5)
        linear = torch.nn.Linear(16, 64)
        x = torch.randn(4, 16)
        cases = {
            "layer_norm of a linear": (linear, norm, x),
            "layer_norm of an embedding": (
                torch.nn.Embedding(10, 64),
                norm,
                torch.tensor([[1, 7, 3]]),
            ),
            "hardtanh of a linear": (linear, torch.nn.Hardtanh(-0.5, 0.5), x),
            "relu6 of a linear": (linear, torch.nn.ReLU6(), x * 8),
        }
        for name, (first, second, x) in cases.items():
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(case=name, dtype=dtype):
                    model = torch.nn.Sequential(first, second)
                    half = copy.deepcopy(model).to(dtype).requires_grad_(False)
                    inputs = (x.to(dtype) if x.is_floating_point() else x,)
                    self._check(half, inputs, atol=1e-2, rtol=1e-2)

    def test_rope_llama3(self):
        config = LlamaConfig(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            rope_theta=500000.0,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
            },
        )
        self._check(
            Rope(config),
            (torch.randn(1, 4, 5, 64), torch.tensor([[3, 4, 5, 6, 7]])),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_rope_frequency_tables(self):
        # Two ropes hold equal inv_freq constants, which share one table; a
        # third holds its own.
        class Tables(torch.nn.Module):
            def __init__(self):
                super().__init__()
                config = LlamaConfig(
                    hidden_size=256, num_attention_heads=4, head_dim=64
                )
                self.a = Rope(config)
                self.b = Rope(config)
                config = LlamaConfig(
                    hidden_size=256,
                    num_attention_heads=4,
                    head_dim=64,
                    rope_theta=500000.0,
                )
                self.c = Rope(config)

            def forward(self, x, positions):
                return (
                    self.a(x, positions)
                    + 2 * self.b(x, positions)
                    + 3 * self.c(x, positions)
                )

        self._check(
            Tables(),
            (torch.randn(1, 4, 5, 64), torch.tensor([[3, 40, 200, 6, 100]])),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_rope_zero_frequency_leaves_dims_unrotated(self):
        # Partial RoPE written as zero inverse frequencies, e.g. Gemma 4's
        # proportional RoPE: those dims keep their values.
        config = LlamaConfig(hidden_size=256, num_attention_heads=4, head_dim=64)
        model = Rope(config)
        model.rotary.inv_freq[8:] = 0.0
        self._check(
            model,
            (torch.randn(1, 4, 5, 64), torch.tensor([[3, 4, 5, 6, 7]])),
            atol=1e-4,
            rtol=1e-4,
        )


class GgmlLlamaTest(unittest.TestCase):
    MAX_SEQ = 16

    def test_tiny_llama_8da4w(self):
        self._check_steps(self.MAX_SEQ, prefill=7, decode=2)

    def test_tiny_llama_attends_key_prefix(self):
        # A cache longer than the engine's 256-key extent bucket: attention
        # covers only the attended prefix, which grows into a second bucket
        # when position 256 is decoded. Random weights over a long context
        # leave near-tied logits, so picks are checked within tolerance.
        ops = collections.Counter()
        self._check_steps(600, prefill=255, decode=2, exact_pick=False, graph_ops=ops)
        # Two layers per graph. Besides the compile-time graph and the prefill,
        # each decode builds its own: the second one's prefix is longer.
        self.assertEqual(ops["FLASH_ATTN_EXT"], 2 * 4)

    def _check_steps(self, max_seq, prefill, decode, exact_pick=True, graph_ops=None):
        torch.manual_seed(0)
        model = TinyLlama(max_seq).eval()
        example = (torch.zeros(1, 6, dtype=torch.long), torch.arange(6))
        seq = Dim("seq", min=1, max=max_seq - 1)
        with torch.no_grad():
            ep = torch.export.export(
                model,
                example,
                dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                strict=True,
            )
        prompt = torch.randint(0, 512, (1, prefill + decode))
        steps = [(prompt[:, :prefill], torch.arange(prefill))] + [
            (prompt[:, i : i + 1], torch.tensor([i]))
            for i in range(prefill, prefill + decode)
        ]
        eager = copy.deepcopy(model)
        with torch.no_grad():
            expected = [eager(*inputs) for inputs in steps]
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "tiny_llama.ptn"
            to_native(ep).save(str(package))
            outputs = _run_ggml(
                package, steps, outdir, 1, resize=True, graph_ops=graph_ops
            )
        # ggml-cuda applies Q4_0's -8 offset through an fp16 sum of each Q8_1
        # block, where the CPU subtracts it in integers.
        gpu = os.environ.get("GGML_PTN_DEVICE", "cpu") == "gpu"
        tolerance = 15e-2 if gpu else 8e-2
        self.assertEqual(len(outputs), len(steps))
        for step, (step_outputs, reference) in enumerate(zip(outputs, expected)):
            ((raw, shape),) = step_outputs
            with self.subTest(step=step):
                self.assertEqual(shape, tuple(reference.shape))
                actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                actual = actual.reshape(shape)
                atol = tolerance * reference.abs().max().item()
                torch.testing.assert_close(actual, reference, atol=atol, rtol=0)
                if gpu or not exact_pick:
                    # Near-ties may flip on the GPU, so ggml's pick only has
                    # to be within tolerance of eager's best logit.
                    chosen = reference[0, -1, actual[0, -1].argmax()]
                    self.assertGreaterEqual(chosen, reference[0, -1].max() - atol)
                else:
                    self.assertEqual(
                        actual[0, -1].argmax().item(),
                        reference[0, -1].argmax().item(),
                    )


class GgmlQwen35Test(unittest.TestCase):
    """A tiny Qwen3.5, gated DeltaNet layers and full attention, exported as
    static one-token calls whose recurrent states are buffers."""

    def test_native_gdn_rule_calls_match_eager_across_a_restart(self):
        self._check(native=True)

    def test_prefill_and_one_token_calls_share_state(self):
        # A prompt as prefill calls of 8 and 5 tokens and one-token calls,
        # then a second sequence from position 0, each call on its method of
        # one package.
        model = self._model()
        tokens = torch.randint(0, 128, (19,))
        calls = [(0, 8), (8, 5), (13, 1), (14, 1), (0, 3), (3, 1)]
        steps, methods, start = [], [], 0
        for position, size in calls:
            steps.append(
                (
                    tokens[start : start + size].view(1, size),
                    torch.arange(position, position + size),
                )
            )
            methods.append("forward" if size == 1 else "prefill")
            start += size
        with native_gdn_rules(model), torch.no_grad():
            wrapper = LastTokenLM(model, 32).eval()
            one = (
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
            programs = {
                "forward": torch.export.export(wrapper, one, strict=True),
                "prefill": export_prefill(wrapper, 16),
            }
            eager = copy.deepcopy(wrapper)
            expected = [eager(*inputs) for inputs in steps]
        ops = {fold: collections.Counter() for fold in (True, False)}
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "tiny_qwen35.ptn"
            to_native(programs).save(str(package))
            outputs, unfolded = (
                _run_ggml(
                    package,
                    steps,
                    outdir,
                    1,
                    resize=True,
                    method=";".join(methods),
                    graph_ops=ops[fold],
                    fold_constants=fold,
                )
                for fold in (True, False)
            )
        # Every method computes its constants once, prefill as well as decode.
        _assert_same_as_unfolded(self, outputs, unfolded)
        self.assertLess(ops[True].total(), ops[False].total())
        self.assertEqual(len(outputs), len(steps))
        # GPUs multiply 8-row F32 linears in TF32 or XF32 (README: Known ggml
        # GPU numerics).
        tol = 1e-3 if _BACKEND in ("cuda", "hip") else 1e-4
        for step, (step_outputs, reference) in enumerate(zip(outputs, expected)):
            ((raw, shape),) = step_outputs
            with self.subTest(step=step, method=methods[step]):
                actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                torch.testing.assert_close(
                    actual.reshape(shape), reference, atol=tol, rtol=tol
                )

    def test_methods_load_one_weight_set(self):
        # forward and prefill, on 64 and 8 tokens, read every weight; the
        # context loads each once, and computes each result folded from them
        # once.
        model = self._model()
        tokens = torch.randint(0, 128, (73,))
        calls = [(0, 64), (0, 8), (8, 1)]
        steps, methods, start = [], [], 0
        for position, size in calls:
            steps.append(
                (
                    tokens[start : start + size].view(1, size),
                    torch.arange(position, position + size),
                )
            )
            methods.append("forward" if size == 1 else "prefill")
            start += size
        with native_gdn_rules(model), torch.no_grad():
            wrapper = LastTokenLM(model, 64).eval()
            one = (
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
            programs = {
                "forward": torch.export.export(wrapper, one, strict=True),
                "prefill": export_prefill(wrapper, 64),
            }
        constant_bytes = {fold: [] for fold in (True, False)}
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "tiny_qwen35.ptn"
            to_native(programs).save(str(package))
            folded, unfolded = (
                _run_ggml(
                    package,
                    steps,
                    outdir,
                    1,
                    resize=True,
                    method=";".join(methods),
                    fold_constants=fold,
                    constant_bytes=constant_bytes[fold],
                )
                for fold in (True, False)
            )
        _assert_same_as_unfolded(self, folded, unfolded)
        self.assertEqual(len(folded), len(steps))
        weights = sum(p.numel() * p.element_size() for p in model.parameters())
        for fold, after_compiles in constant_bytes.items():
            with self.subTest(fold=fold):
                self.assertEqual(len(after_compiles), 2)
                self.assertEqual(after_compiles[1], after_compiles[0])
                self.assertLess(after_compiles[0], 2 * weights)
        self.assertGreater(constant_bytes[True][0], constant_bytes[False][0])

    def test_methods_share_one_compute_buffer(self):
        # forward and prefill reserve the context's compute buffers at
        # compile; calls on either, at any size, allocate no more.
        model = self._model()
        tokens = torch.randint(0, 128, (140,))
        calls = [(0, 64), (64, 1), (0, 8), (8, 1), (9, 2), (0, 64)]
        steps, methods, start = [], [], 0
        for position, size in calls:
            steps.append(
                (
                    tokens[start : start + size].view(1, size),
                    torch.arange(position, position + size),
                )
            )
            methods.append("forward" if size == 1 else "prefill")
            start += size
        with native_gdn_rules(model), torch.no_grad():
            wrapper = LastTokenLM(model, 128).eval()
            one = (
                torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
            programs = {
                "forward": torch.export.export(wrapper, one, strict=True),
                "prefill": export_prefill(wrapper, 64),
            }
        memory = []
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "tiny_qwen35.ptn"
            to_native(programs).save(str(package))
            outputs = _run_ggml(
                package,
                steps,
                outdir,
                1,
                resize=True,
                method=";".join(methods),
                memory=memory,
            )
            # Cached graphs, laid out again in the shared buffers after the
            # other method's, must compute what fresh graphs do.
            fresh = _run_ggml(
                package,
                steps,
                outdir,
                1,
                resize=True,
                method=";".join(methods),
                max_cached_graphs=0,
            )
            cached = _run_ggml(
                package,
                steps,
                outdir,
                1,
                resize=True,
                method=";".join(methods),
                max_cached_graphs=4,
            )
        self.assertEqual(len(outputs), len(steps))
        self.assertEqual(cached, fresh)
        self.assertEqual([when for when, _, _ in memory][-1], "at exit")
        compiled = memory[-2]
        self.assertGreater(compiled[1], 0)
        self.assertEqual(memory[-1][1:], compiled[1:])

    def _model(self):
        torch.manual_seed(5)
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
        )
        model = AutoModelForCausalLM.from_config(config).eval()
        model.config._attn_implementation = "sdpa"
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        return model

    def _check(self, native: bool):
        model = self._model()
        example = (
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        )
        # Two sequences on one executable: the second restarts at position 0
        # with no reset call, so the graph must zero the states itself.
        tokens = torch.randint(0, 128, (12,))
        steps = [
            (tokens[i : i + 1].view(1, 1), torch.tensor([position]))
            for i, position in enumerate([*range(7), *range(5)])
        ]
        rules = native_gdn_rules(model) if native else contextlib.nullcontext()
        with rules, torch.no_grad():
            wrapper = LastTokenLM(model, 32).eval()
            ep = torch.export.export(wrapper, example, strict=True)
            targets = [n.target for n in ep.graph.nodes if n.op == "call_function"]
            self.assertEqual(targets.count(gated_delta_rule_op), 3 if native else 0)
            eager = copy.deepcopy(wrapper)
            expected = [eager(*inputs) for inputs in steps]
        ops = {fold: collections.Counter() for fold in (True, False)}
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "tiny_qwen35.ptn"
            to_native(ep).save(str(package))
            outputs, unfolded = (
                _run_ggml(
                    package,
                    steps,
                    outdir,
                    1,
                    graph_ops=ops[fold],
                    fold_constants=fold,
                )
                for fold in (True, False)
            )
        _assert_same_as_unfolded(self, outputs, unfolded)
        self.assertLess(ops[True].total(), ops[False].total())
        self.assertEqual(len(outputs), len(steps))
        for step, (step_outputs, reference) in enumerate(zip(outputs, expected)):
            ((raw, shape),) = step_outputs
            with self.subTest(step=step):
                actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                torch.testing.assert_close(
                    actual.reshape(shape), reference, atol=1e-4, rtol=1e-4
                )


class GgmlMoeTest(unittest.TestCase):
    """Tiny transformers MoE text models with batched_mm experts, whose expert
    projections run as ggml_mul_mat_id."""

    def _check_model(self, config):
        torch.manual_seed(3)
        model = AutoModelForCausalLM.from_config(
            config, experts_implementation="batched_mm"
        ).eval()
        model.config._attn_implementation = "sdpa"
        model.generation_config.use_cache = True
        model.generation_config.cache_implementation = "static"
        wrapper = LastTokenLM(model, 16).eval()
        inputs = (torch.randint(0, 64, (1, 3)), torch.arange(3))
        with torch.no_grad():
            ep = torch.export.export(wrapper, inputs, strict=True)
            expected = copy.deepcopy(wrapper)(*inputs)
        ops = collections.Counter()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "moe.ptn"
            to_native(ep).save(str(package))
            (((raw, shape),),) = _run_ggml(package, [inputs], outdir, 1, graph_ops=ops)
        actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        torch.testing.assert_close(
            actual.reshape(shape), expected, atol=1e-3, rtol=1e-3
        )
        self.assertGreater(ops["MUL_MAT_ID"], 0)

    def test_qwen3_5_moe(self):
        self._check_model(
            Qwen3_5MoeTextConfig(
                vocab_size=64,
                hidden_size=64,
                num_hidden_layers=1,
                layer_types=["full_attention"],
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=32,
                shared_expert_intermediate_size=32,
                max_position_embeddings=64,
                tie_word_embeddings=False,
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 1e7,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [2, 1, 1],
                    "mrope_interleaved": True,
                },
            )
        )

    def test_gemma4_moe(self):
        self._check_model(
            Gemma4TextConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=64,
                num_hidden_layers=1,
                layer_types=["full_attention"],
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                global_head_dim=16,
                enable_moe_block=True,
                num_experts=4,
                top_k_experts=2,
                moe_intermediate_size=32,
                max_position_embeddings=64,
            )
        )


class GgmlKeyExtentTest(unittest.TestCase):
    """Attention masks whose query positions the host evaluates bound the
    attended keys; any form it refuses attends the whole cache."""

    def _attention_ops(self, model) -> int:
        torch.manual_seed(0)
        seq = Dim("seq", min=1, max=599)
        with torch.no_grad():
            ep = torch.export.export(
                model,
                (torch.randn(1, 2, 6, 32), torch.arange(6)),
                dynamic_shapes={"q": {2: seq}, "cache_position": {0: seq}},
                strict=True,
            )
        steps = [(torch.randn(1, 2, 255, 32), torch.arange(255))] + [
            (torch.randn(1, 2, 1, 32), torch.tensor([p])) for p in (255, 256)
        ]
        with torch.no_grad():
            expected = [model(*inputs) for inputs in steps]
        ops = collections.Counter()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "attention.ptn"
            to_native(ep).save(str(package))
            outputs = _run_ggml(package, steps, outdir, 1, resize=True, graph_ops=ops)
        for step, (((raw, shape),), reference) in enumerate(zip(outputs, expected)):
            with self.subTest(step=step):
                actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                torch.testing.assert_close(
                    actual.reshape(shape), reference, atol=1e-2, rtol=1e-2
                )
        return ops["FLASH_ATTN_EXT"]

    def test_sliced_first_position_bounds_the_attended_keys(self):
        # Besides the compile-time graph and the prefill, each decode builds
        # its own: the second one's prefix crosses into a second bucket.
        self.assertEqual(self._attention_ops(SlicedQueryPositions(600)), 4)

    def test_slice_from_computed_start_attends_the_whole_cache(self):
        # The host evaluator reads a slice bound only as an int or an input
        # size, so it refuses the computed start: every call attends all keys
        # and the second decode replays the first one's graph.
        self.assertEqual(self._attention_ops(SlicedQueryPositions(600, last=True)), 3)


def _gated_delta_rule_inputs(tokens: int, heads: int = 4, dim: int = 16):
    generator = torch.Generator().manual_seed(tokens)
    qkv = [torch.randn(1, tokens, heads, dim, generator=generator) for _ in range(3)]
    g = -torch.rand(1, tokens, heads, generator=generator)
    beta = torch.rand(1, tokens, heads, generator=generator)
    return (*qkv, g, beta)


class _GatedDeltaRule(torch.nn.Module):
    def __init__(self, qk_l2norm: bool, snapshots: int = 1) -> None:
        super().__init__()
        self.qk_l2norm = qk_l2norm
        self.snapshots = snapshots

    def forward(self, q, k, v, g, beta, state):
        return gated_delta_rule_op(
            q, k, v, g, beta, state, self.qk_l2norm, self.snapshots
        )


class _StatefulGatedDeltaRule(torch.nn.Module):
    """The rule over a state buffer each call continues."""

    def __init__(self, heads: int = 4, dim: int = 16) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(1, heads, dim, dim), persistent=False)

    def forward(self, q, k, v, g, beta):
        out, state = gated_delta_rule_op(q, k, v, g, beta, self.state, True)
        self.state.copy_(state)
        return out


class GgmlGatedDeltaRuleTest(unittest.TestCase):
    """native::gated_delta_rule on ggml_gated_delta_net against its reference,
    transformers' recurrent rule: one token, a partial 64-token chunk, and one
    and a part, from a random state."""

    def _check(self, model, steps, num_outputs, atol=1e-4):
        with torch.no_grad():
            ep = torch.export.export(copy.deepcopy(model), steps[0])
            expected = [model(*inputs) for inputs in steps]
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "gated_delta_rule.ptn"
            to_native(ep).save(str(package))
            outputs = _run_ggml(package, steps, Path(tmp), num_outputs)
        for step, (got, want) in enumerate(zip(outputs, expected, strict=True)):
            want = want if isinstance(want, tuple) else (want,)
            for i, ((raw, shape), reference) in enumerate(zip(got, want, strict=True)):
                with self.subTest(step=step, output=i):
                    actual = torch.frombuffer(bytearray(raw), dtype=torch.float32)
                    torch.testing.assert_close(
                        actual.reshape(shape), reference, atol=atol, rtol=1e-4
                    )

    def test_matches_reference(self):
        for tokens, qk_l2norm in ((1, True), (8, True), (100, True), (3, False)):
            with self.subTest(tokens=tokens, qk_l2norm=qk_l2norm):
                state = torch.randn(1, 4, 16, 16, generator=torch.Generator())
                inputs = (*_gated_delta_rule_inputs(tokens), state * 0.1)
                self._check(_GatedDeltaRule(qk_l2norm), [inputs], 2)

    def test_snapshots_match_reference(self):
        # Fewer tokens than slots keep older input slots; more run the tokens
        # before the last S first.
        for tokens, snapshots in ((1, 3), (3, 3), (2, 4), (9, 3)):
            with self.subTest(tokens=tokens, snapshots=snapshots):
                states = torch.randn(
                    1, snapshots, 4, 16, 16, generator=torch.Generator()
                )
                inputs = (*_gated_delta_rule_inputs(tokens), states * 0.1)
                self._check(_GatedDeltaRule(True, snapshots), [inputs], 2)

    def test_l2_norm_adds_eps_to_the_sum_of_squares(self):
        # Query and key norms near sqrt(eps), where rsqrt(sum + eps) and
        # 1 / max(norm, eps) differ.
        q, k, v, g, beta = _gated_delta_rule_inputs(3)
        state = torch.zeros(1, 4, 16, 16)
        inputs = (q * 1e-3, k * 1e-3, v, g, beta, state)
        self._check(_GatedDeltaRule(True), [inputs], 2)

    def test_prompt_chunk_matches_reference_closely(self):
        # ggml-cuda multiplies prompt chunks in split BF16, high and low parts.
        state = torch.randn(1, 4, 16, 16, generator=torch.Generator()) * 0.1
        inputs = (*_gated_delta_rule_inputs(64), state)
        self._check(_GatedDeltaRule(True), [inputs], 2, atol=2e-6)

    def test_state_buffer_carries_across_calls(self):
        steps = [_gated_delta_rule_inputs(tokens) for tokens in (1, 1, 1)]
        self._check(_StatefulGatedDeltaRule(), steps, 1)


if __name__ == "__main__":
    unittest.main()
