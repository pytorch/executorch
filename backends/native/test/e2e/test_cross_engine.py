# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One PTN, exported once, run on every native engine this target links.

Each engine's PTN runner executes the same package; its outputs must match
eager PyTorch, and greedy Llama tokens must agree across engines.
"""

import copy
import os
import subprocess
import tempfile
import unittest
from importlib.resources import as_file, files
from pathlib import Path

import gguf
import torch
import torch.nn.functional as F
import torchvision
from executorch.backends.native.examples.hf_llm.export_hf_llm import (
    golden_samples,
    LastTokenLM,
    load_gguf,
)
from executorch.exir._serialize._ptn import read_ptn
from executorch.exir.native import to_native
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from torch.export import Dim
from transformers import Gemma2Config, Gemma2ForCausalLM, LlamaConfig, LlamaForCausalLM


_ENGINES = ("ggml_ptn_runner", "vulkan_ptn_runner")
_FP32_KV_ONLY = ("vulkan_ptn_runner",)
_MAX_SEQ = 16


def _run(
    runner: str,
    package: Path,
    steps,
    outdir: Path,
    resize: bool = False,
    flags: tuple[str, ...] = (),
):
    """Runs each step's inputs in order on one executable; returns each step's
    single float32 output."""
    input_steps, output_steps, shapes = [], [], []
    for step, inputs in enumerate(steps):
        paths = []
        for i, tensor in enumerate(inputs):
            path = outdir / f"{runner}_{step}_{i}.bin"
            path.write_bytes(tensor.contiguous().flatten().view(torch.uint8).numpy())
            paths.append(str(path))
        input_steps.append(",".join(paths))
        output_steps.append(str(outdir / f"{runner}_{step}_out.bin"))
        shapes.append(",".join("x".join(map(str, t.shape)) for t in inputs))
    command = [
        f"--package={package}",
        f"--inputs={';'.join(input_steps)}",
        f"--outputs={';'.join(output_steps)}",
    ]
    if resize:
        command.append(f"--input_sizes={';'.join(shapes)}")
    command.extend(flags)
    resources = files(__package__)
    with as_file(resources / runner) as binary, as_file(
        resources / "vulkan_driver"
    ) as driver:
        # The Vulkan runner loads SwiftShader as its Vulkan driver.
        env = {**os.environ, "LD_LIBRARY_PATH": str(driver)}
        result = subprocess.run(
            [str(binary), *command],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    if result.returncode != 0:
        raise AssertionError(f"{runner} failed:\n{result.stderr}")
    sizes = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    return [
        torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.float32)
        .reshape(size)
        .clone()
        for path, size in zip(output_steps, sizes, strict=True)
    ]


def _tiny_llama() -> LlamaForCausalLM:
    # 256-wide projections so every weight is a whole number of K-quant blocks.
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    model.config._attn_implementation = "sdpa"
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    return model


def _tiny_gemma2(**config) -> Gemma2ForCausalLM:
    # A sliding-window layer, then a full-attention one; the window is shorter
    # than the prompt.
    config = Gemma2Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        **config,
    )
    model = Gemma2ForCausalLM(config).eval().requires_grad_(False)
    model.config._attn_implementation = "sdpa"
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    return model


def _blocks(
    ggml_type: str,
    rows: int,
    cols: int,
    scales: dict[int, float],
    generator: torch.Generator,
) -> ExportableGGUFTensor:
    """Random `ggml_type` blocks whose fp16 scales, by byte offset in a block,
    give weights of HF-init magnitude, which keeps attention from saturating
    on random weights. Each block scales them all by 1/2, 1 or 2, so a decoder
    that reads another block's scales does not match."""
    block, block_bytes = gguf.GGML_QUANT_SIZES[
        gguf.GGMLQuantizationType[ggml_type.upper()]
    ]
    raw = torch.randint(
        0, 256, (rows, cols // block, block_bytes), generator=generator
    ).to(torch.uint8)
    factors = 2.0 ** torch.randint(-1, 2, (rows, cols // block, 1), generator=generator)
    for offset, scale in scales.items():
        fp16 = (factors * scale).to(torch.float16)
        raw[..., offset : offset + 2] = fp16.view(torch.uint8)
    return ExportableGGUFTensor.from_raw(
        raw.reshape(rows, -1), ggml_type, torch.float32
    )


def _q4_k(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 144-byte blocks: fp16 d and dmin, 12 bytes of 6-bit scales and mins, then
    # 128 bytes of nibbles.
    return _blocks("q4_k", rows, cols, {0: 2**-12, 2: 2**-9}, generator)


def _q5_k(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 176-byte blocks: Q4_K's plus 32 bytes of fifth bits, so half its d.
    return _blocks("q5_k", rows, cols, {0: 2**-13, 2: 2**-9}, generator)


def _q6_k(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 210-byte blocks: 128 low and 64 high bit bytes, 16 int8 scales, fp16 d.
    weight = _blocks("q6_k", rows, cols, {208: 2**-12}, generator)
    raw = weight.raw.reshape(rows, cols // 256, 210)
    scales = torch.randint(-8, 9, (rows, cols // 256, 16), generator=generator)
    raw[..., 192:208] = scales.to(torch.int8).view(torch.uint8)
    return weight


def _q8_0(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 34-byte blocks: fp16 d, then 32 int8 codes.
    return _blocks("q8_0", rows, cols, {0: 2**-11}, generator)


def _q3_k(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 110-byte blocks: 32 bytes of high bits, 64 of low bits, 12 of 6-bit
    # scales, then fp16 d.
    return _blocks("q3_k", rows, cols, {108: 2**-10}, generator)


def _q2_k(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 84-byte blocks: 16 bytes of 4-bit scales and mins, 64 of 2-bit codes,
    # then fp16 d and dmin; 1.5 d of dmin centers the weights.
    return _blocks("q2_k", rows, cols, {80: 2**-9, 82: 3 * 2**-10}, generator)


def _q4_0(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 18-byte blocks: fp16 d, then 32 4-bit codes offset by 8.
    return _blocks("q4_0", rows, cols, {0: 2**-8}, generator)


def _q5_0(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 22-byte blocks: fp16 d, 4 bytes of fifth bits, then 32 4-bit codes;
    # codes are offset by 16.
    return _blocks("q5_0", rows, cols, {0: 2**-9}, generator)


def _q5_1(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 24-byte blocks: fp16 d and m, 4 bytes of fifth bits, then 32 4-bit
    # codes; m = -15.5 d centers the weights.
    return _blocks("q5_1", rows, cols, {0: 2**-9, 2: -31 * 2**-10}, generator)


def _q4_1(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 20-byte blocks: fp16 d and m, then 32 4-bit codes; m = -7.5 d centers
    # the weights.
    return _blocks("q4_1", rows, cols, {0: 2**-8, 2: -15 * 2**-9}, generator)


# Synthetic weights of each GGUF type the tests cover.
_GGUF_WEIGHTS = {
    "q4_k": _q4_k,
    "q5_k": _q5_k,
    "q6_k": _q6_k,
    "q8_0": _q8_0,
    "q3_k": _q3_k,
    "q2_k": _q2_k,
    "q4_0": _q4_0,
    "q5_0": _q5_0,
    "q5_1": _q5_1,
    "q4_1": _q4_1,
}

# The GGUF type of each tiny-Llama weight by llama.cpp role, mixing types as
# "Q4_K_M" files do. A "blk.<i>.<role>" entry overrides the role in one layer.
_MIXED_TYPES = {
    "token_embd": "q6_k",
    "attn_q": "q4_k",
    "attn_k": "q5_k",
    "attn_v": "q6_k",
    "attn_output": "q4_k",
    "ffn_gate": "q5_k",
    "ffn_up": "q4_k",
    "ffn_down": "q6_k",
    "blk.1.attn_v": "q8_0",
    "blk.1.attn_q": "q3_k",
    "blk.1.ffn_up": "q2_k",
    "blk.1.attn_output": "q4_0",
    "blk.1.ffn_gate": "q5_0",
    "blk.1.attn_k": "q5_1",
    "blk.1.ffn_down": "q4_1",
}


def _gguf_weights(model: LlamaForCausalLM) -> list[tuple[str, torch.Tensor]]:
    """Synthetic GGUF tensors under llama.cpp names, typed by _MIXED_TYPES."""
    generator = torch.Generator().manual_seed(3)
    config = model.config
    hidden, inner = config.hidden_size, config.intermediate_size
    kv = config.num_key_value_heads * config.head_dim

    def norm() -> torch.Tensor:
        return torch.rand(hidden, generator=generator) + 0.5

    def weight(name: str, rows: int, cols: int) -> torch.Tensor:
        ggml_type = _MIXED_TYPES.get(name, _MIXED_TYPES[name.split(".")[-1]])
        return _GGUF_WEIGHTS[ggml_type](rows, cols, generator)

    pairs = [
        ("token_embd.weight", weight("token_embd", config.vocab_size, hidden)),
        ("output_norm.weight", norm()),
    ]
    for i in range(config.num_hidden_layers):
        pairs += [
            (f"blk.{i}.attn_norm.weight", norm()),
            (f"blk.{i}.attn_q.weight", weight(f"blk.{i}.attn_q", hidden, hidden)),
            (f"blk.{i}.attn_k.weight", weight(f"blk.{i}.attn_k", kv, hidden)),
            (f"blk.{i}.attn_v.weight", weight(f"blk.{i}.attn_v", kv, hidden)),
            (
                f"blk.{i}.attn_output.weight",
                weight(f"blk.{i}.attn_output", hidden, hidden),
            ),
            (f"blk.{i}.ffn_norm.weight", norm()),
            (f"blk.{i}.ffn_gate.weight", weight(f"blk.{i}.ffn_gate", inner, hidden)),
            (f"blk.{i}.ffn_up.weight", weight(f"blk.{i}.ffn_up", inner, hidden)),
            (f"blk.{i}.ffn_down.weight", weight(f"blk.{i}.ffn_down", hidden, inner)),
        ]
    return pairs


class _Lambda(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _write_gguf(path: Path, pairs: list[tuple[str, torch.Tensor]]) -> None:
    writer = gguf.GGUFWriter(str(path), "llama")
    for name, value in pairs:
        if isinstance(value, ExportableGGUFTensor):
            writer.add_tensor(
                name,
                value.raw.numpy(),
                raw_dtype=gguf.GGMLQuantizationType[value.ggml_type.upper()],
            )
        else:
            writer.add_tensor(name, value.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class _ViewRelu(torch.nn.Module):
    """relu over a reshaping view of a linear output: the in-place relu the
    native passes emit writes through a view nothing else reads."""

    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(16, 32)
        self.down = torch.nn.Linear(8, 4)

    def forward(self, x):
        y = self.up(x)
        return self.down(torch.relu(y.view(1, 4 * x.shape[1], 8)))


class _PaddingMask(torch.nn.Module):
    """Zeroes frames past `lengths`, with a mask built from unsqueezed,
    dynamically sized positions."""

    def forward(self, x, lengths):
        positions = torch.arange(x.shape[1])
        keep = positions[None, :] < lengths[:, None]
        return x * keep[:, :, None].to(x.dtype)


class _HeadMajor(torch.nn.Module):
    """Splits a dynamically sized sequence into heads and back, so the
    head-major transposes view a resized source; the in-place relu before the
    split puts the first transpose in an alias group that relu mutates."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)
        self.out = torch.nn.Linear(16, 16)

    def forward(self, x):
        seq = x.shape[1]
        heads = torch.relu(self.proj(x)).view(1, seq, 4, 4).transpose(1, 2)
        heads = heads * heads
        return self.out(heads.transpose(1, 2).reshape(1, seq, 16))


class _NegativeDimViews(torch.nn.Module):
    """permute, squeeze, unsqueeze and select with negative dims over a
    dynamically sized activation, which run as ET-VK copy kernels."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)
        self.out = torch.nn.Linear(16, 8)

    def forward(self, x):
        seq = x.shape[1]
        y = self.proj(x).view(1, seq, 4, 4).permute(0, 2, 1, -1)
        y = y.squeeze(-4).unsqueeze(-3).select(-3, 0)
        return self.out(y.permute(1, 0, -1).reshape(1, seq, 16))


class _TailPastLength(torch.nn.Module):
    """Sums the last n + 1 rows of a fixed buffer: the slice starts at
    -(n + 1), whose operand is not a tensor size."""

    def __init__(self):
        super().__init__()
        self.register_buffer("buffer", torch.randn(1, 12, 16))

    def forward(self, x):
        return self.buffer[:, -(x.shape[1] + 1) :].sum(1, keepdim=True) + x


class _TailOfBuffer(torch.nn.Module):
    """Adds the last n rows of a fixed buffer, for n the dynamic sequence
    length: the slice starts at the SymInt -n (operator.neg)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("buffer", torch.randn(1, 12, 16))

    def forward(self, x):
        return self.buffer[:, -x.shape[1] :] + x


class _SumThenIndex(torch.nn.Module):
    def __init__(self, dims=0):
        super().__init__()
        self.dims = dims

    def forward(self, x, index):
        return x.sum(self.dims)[index]


class _ProjectThenReduce(torch.nn.Module):
    """A linear, then a reduction that drops the dims it reduces."""

    def __init__(self, reduce):
        super().__init__()
        self.proj = torch.nn.Linear(16, 32)
        self.reduce = reduce

    def forward(self, x):
        return self.reduce(self.proj(x).view(1, x.shape[1], 4, 8))


class _RmsNormCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(5) + 0.5)

    def forward(self, x):
        y = F.rms_norm(x, (5,), self.weight, eps=1e-6)
        return torch.cat([y, y * 0.5], 1)


class _ExpandBesideLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 16)

    def forward(self, x):
        return x.expand(5, 2, 3) + self.linear(x)[..., :3]


class _SumAfterLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x).sum(-1, keepdim=True)


class _RepeatIntoRmsNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(4) + 0.5)

    def forward(self, x):
        return F.rms_norm(x.repeat(1, 2), (4,), self.weight, eps=1e-6)


class _PoolCloneLinear(torch.nn.Module):
    """A clone between a max pool, which ET-VK runs on channels-packed
    textures only, and a linear, which wants width-packed input: the clone
    converts between the two."""

    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(2)
        self.proj = torch.nn.Linear(4, 8)

    def forward(self, x):
        return self.proj(self.pool(x).clone())


class _NormLinear(torch.nn.Module):
    """layer_norm of a graph input feeding a linear, which wants its input
    width-packed while the graph input arrives in the default layout."""

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(16)
        self.proj = torch.nn.Linear(16, 8)

    def forward(self, x):
        return self.proj(self.norm(x))


class _GeluSigmoid(torch.nn.Module):
    """GELU and sigmoid after linears, which the native passes emit in place."""

    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(16, 32)
        self.down = torch.nn.Linear(32, 8)

    def forward(self, x):
        return torch.sigmoid(self.down(torch.nn.functional.gelu(self.up(x))))


class _ConvBetweenLinears(torch.nn.Module):
    """A conv1d over the sequence between linears, whose neighbours want
    width-packed activations while ET-VK's convolution reads and writes
    channels-packed ones."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)
        self.conv = torch.nn.Conv1d(16, 24, kernel_size=3, padding=1)
        self.out = torch.nn.Linear(24, 8)

    def forward(self, x):
        y = self.conv(self.proj(x).transpose(1, 2))
        return self.out(y.transpose(1, 2))


class _StackFrames(torch.nn.Module):
    """Pads, then stacks every 8 frames into one before a linear. The linear
    pulls width-packing back through the reshape onto the pad's output, while
    the pad reads a graph input in the default layout."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8 * 16, 8)

    def forward(self, x):
        # One extra frame of zeros along the dynamic dim, as when padding
        # features up to a whole number of frames.
        x = torch.nn.functional.pad(x, (0, 0, 0, 8))
        return self.proj(x.reshape(1, x.shape[1] // 8, 8 * 16))


class _PadThenMean(torch.nn.Module):
    """A pad between a linear, which writes width-packed output, and a mean
    over the last two dims, which constrains the packed dim it reads."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16)

    def forward(self, x):
        x = torch.nn.functional.pad(self.proj(x), (0, 0, 0, 8))
        return x.mean((-2, -1), keepdim=True)


class CrossEngineTest(unittest.TestCase):
    def _check_dynamic(
        self, model, steps, dynamic_shapes, atol_fraction=1e-4, engines=_ENGINES
    ):
        """Exports `model` once with `dynamic_shapes`, then runs every step, each
        at its own sizes, on one executable per engine against eager."""
        model = model.eval().requires_grad_(False)
        with torch.no_grad():
            expected = [model(*inputs) for inputs in steps]
            ep = torch.export.export(
                model, steps[0], dynamic_shapes=dynamic_shapes, strict=True
            )
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(ep).save(str(package))
            self._check_engines(
                package, steps, expected, outdir, atol_fraction, engines
            )

    def test_head_major_transposes_of_dynamic_sequence(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        self._check_dynamic(_HeadMajor(), steps, ({1: seq},))

    def test_negative_dim_views_of_dynamic_sequence(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        self._check_dynamic(_NegativeDimViews(), steps, ({1: seq},))

    def test_slice_from_the_end_by_dynamic_length(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        # At 12 the slice would start at 0, which export specializes.
        seq = Dim("seq", min=2, max=11)
        self._check_dynamic(_TailOfBuffer(), steps, ({1: seq},))

    def test_symbolic_neg_bounds_on_vulkan(self):
        # native-vk negates in int32: only a size bounded within int32 is
        # accepted, and each run checks input sizes against their bounds.
        x = torch.randn(1, 5, 16)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            computed = outdir / "computed.ptn"
            seq = Dim("seq", min=2, max=10)
            to_native(
                torch.export.export(_TailPastLength(), (x,), dynamic_shapes=({1: seq},))
            ).save(str(computed))
            with self.assertRaisesRegex(AssertionError, "symbolic neg"):
                _run("vulkan_ptn_runner", computed, [(x,)], outdir)
            size = outdir / "size.ptn"
            seq = Dim("seq", min=2, max=11)
            to_native(
                torch.export.export(_TailOfBuffer(), (x,), dynamic_shapes=({1: seq},))
            ).save(str(size))
            with self.assertRaisesRegex(AssertionError, "bound"):
                _run(
                    "vulkan_ptn_runner",
                    size,
                    [(torch.randn(1, 12, 16),)],
                    outdir,
                    resize=True,
                )

    def test_two_dim_reduction_feeding_advanced_indexing(self):
        # native-vk only: the index makes it squeeze dims 1 and 3 out of a
        # buffer, once per input size.
        torch.manual_seed(0)
        steps = [(torch.randn(1, 3, s, 5), torch.tensor([0, 0])) for s in (4, 2, 6)]
        seq = Dim("seq", min=2, max=8)
        self._check_dynamic(
            _SumThenIndex((1, 3)),
            steps,
            ({2: seq}, None),
            atol_fraction=1e-5,
            engines=("vulkan_ptn_runner",),
        )

    def test_reductions_without_keepdim(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        reductions = {
            "sum": lambda y: y.sum(-1),
            "mean": lambda y: y.mean(-1),
            "amax": lambda y: y.amax(-1),
            "sum over the last two dims": lambda y: y.sum((-2, -1)),
            "sum over heads": lambda y: y.sum(2),
        }
        for name, reduce in reductions.items():
            with self.subTest(reduction=name):
                self._check_dynamic(
                    _ProjectThenReduce(reduce), steps, ({1: seq},), atol_fraction=1e-5
                )

    def test_reduction_feeding_advanced_indexing(self):
        # On native-vk the index makes the reduction's output a buffer, while
        # its texture input reduces into a texture.
        torch.manual_seed(0)
        steps = [(torch.randn(2, 3, 4), torch.tensor([0, 2]))]
        self._check_dynamic(_SumThenIndex(), steps, None, atol_fraction=1e-5)

    def test_rank_one_reductions_over_every_dim(self):
        x = torch.randn(8)
        cases = {
            "amax": lambda x: torch.amax(x * 2.0),
            "amin": lambda x: torch.amin(x * 2.0),
            "mean": lambda x: torch.mean(x * 2.0, dim=None),
        }
        for name, fn in cases.items():
            with self.subTest(op=name), tempfile.TemporaryDirectory() as tmp:
                outdir = Path(tmp)
                package = outdir / "model.ptn"
                to_native(torch.export.export(_Lambda(fn), (x,))).save(str(package))
                (out,) = _run("vulkan_ptn_runner", package, [(x,)], outdir)
                expected = fn(x)
                self.assertEqual(out.shape, expected.shape)
                torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_negative_dims(self):
        torch.manual_seed(0)
        steps = [(torch.randn(2, 3, 4, 8),)]
        ops = {
            "softmax": lambda x: F.softmax(x, -1),
            "cat": lambda x: torch.cat([x, x * 2.0], -3),
            "narrow": lambda x: x.narrow(-2, 1, 2) * 1.0,
            "select": lambda x: x.select(-3, 1) * 1.0,
            "sum keepdim": lambda x: x.sum((-1, -2), keepdim=True),
            "mean": lambda x: x.mean(-1),
            "amax": lambda x: x.amax(-1),
            "transpose": lambda x: x.transpose(-1, -2) * 1.0,
            "permute": lambda x: x.permute(0, -1, -3, -2) * 1.0,
            "split": lambda x: torch.split(x, [1, 3], -2)[1] * 1.0,
            "unbind": lambda x: torch.unbind(x, -3)[2] * 1.0,
            "chunk": lambda x: torch.chunk(x, 2, -1)[0] * 1.0,
            "glu": lambda x: F.glu(x, -1),
        }
        for name, fn in ops.items():
            with self.subTest(op=name):
                self._check_dynamic(_Lambda(fn), steps, None, atol_fraction=1e-5)

    def test_ops_beside_width_packed_ops(self):
        # Linear and rms_norm make their neighbors width-packed, while a cat,
        # expand, reduction or repeat reads and writes one packing.
        torch.manual_seed(0)
        cases = {
            "cat after rms_norm": (_RmsNormCat(), torch.randn(8, 4, 5)),
            "expand beside linear": (_ExpandBesideLinear(), torch.randn(5, 2, 1)),
            "sum after linear": (_SumAfterLinear(), torch.randn(1, 4)),
            "repeat into rms_norm": (_RepeatIntoRmsNorm(), torch.randn(1, 2)),
        }
        for name, (model, x) in cases.items():
            with self.subTest(case=name):
                self._check_dynamic(model, [(x,)], None)

    def test_linear_of_a_rank_4_input(self):
        torch.manual_seed(0)
        self._check_dynamic(torch.nn.Linear(5, 8), [(torch.randn(5, 2, 5, 5),)], None)

    def test_clone_between_pool_and_linear(self):
        torch.manual_seed(0)
        self._check_dynamic(_PoolCloneLinear(), [(torch.randn(1, 4, 8, 8),)], None)

    def test_layer_norm_of_graph_input_into_linear(self):
        torch.manual_seed(0)
        model = _NormLinear()
        torch.nn.init.uniform_(model.norm.weight, 0.5, 1.5)
        torch.nn.init.uniform_(model.norm.bias, -0.5, 0.5)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        self._check_dynamic(model, steps, ({1: seq},))

    def test_inplace_gelu_and_sigmoid(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3)]
        seq = Dim("seq", min=2, max=12)
        self._check_dynamic(_GeluSigmoid(), steps, ({1: seq},))

    def test_conv1d_between_linears(self):
        torch.manual_seed(0)
        # Shrinking from 5 to 3 leaves the input texture longer than the input.
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        for dynamic_shapes in (({1: seq},), None):
            self._check_dynamic(
                _ConvBetweenLinears(),
                steps if dynamic_shapes else steps[:1],
                dynamic_shapes,
            )

    def test_transposed_conv1d_is_refused(self):
        # ET-VK runs every rank-3 convolution as a forward conv1d.
        model = _ConvBetweenLinears().eval().requires_grad_(False)
        model.conv = torch.nn.ConvTranspose1d(16, 24, kernel_size=3, padding=1)
        x = torch.randn(1, 5, 16)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(model, (x,))).save(str(package))
            with self.assertRaisesRegex(AssertionError, "transposed 1D convolution"):
                _run("vulkan_ptn_runner", package, [(x,)], outdir)

    def test_padded_frame_stacking_into_linear(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, 8 * s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        self._check_dynamic(_StackFrames(), steps, ({1: 8 * seq},))

    def test_pad_between_linear_and_mean(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        for dynamic_shapes in (({1: seq},), None):
            self._check_dynamic(
                _PadThenMean(), steps if dynamic_shapes else steps[:1], dynamic_shapes
            )

    def test_padding_mask_of_dynamic_positions(self):
        torch.manual_seed(0)
        steps = [
            (torch.randn(1, s, 8), torch.tensor([n]))
            for s, n in ((6, 4), (3, 3), (9, 2))
        ]
        seq = Dim("seq", min=2, max=12)
        self._check_dynamic(_PaddingMask(), steps, ({1: seq}, {}))

    def test_inplace_relu_of_dynamic_view(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        self._check_dynamic(_ViewRelu(), steps, ({1: seq},))

    def _check_engines(
        self, package, steps, expected, outdir, atol_fraction, engines, flags=()
    ):
        tokens = {}
        for runner in engines:
            with self.subTest(engine=runner):
                outputs = _run(runner, package, steps, outdir, resize=True, flags=flags)
                self.assertEqual(len(outputs), len(expected))
                for step, (actual, reference) in enumerate(zip(outputs, expected)):
                    self.assertEqual(actual.shape, reference.shape)
                    torch.testing.assert_close(
                        actual,
                        reference,
                        atol=atol_fraction * reference.abs().max().item(),
                        rtol=0,
                        msg=lambda m, step=step: f"step {step}: {m}",
                    )
                tokens[runner] = [o[0, -1].argmax().item() for o in outputs]
        return tokens

    def test_mobilenet_v2(self):
        torch.manual_seed(0)
        model = torchvision.models.mobilenet_v2(num_classes=16).eval()
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            expected = model(image)
            ep = torch.export.export(model, (image,))
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "mobilenet_v2.ptn"
            to_native(ep).save(str(package))
            for runner in _ENGINES:
                with self.subTest(engine=runner):
                    (actual,) = _run(runner, package, [(image,)], outdir)
                    torch.testing.assert_close(
                        actual, expected, atol=1e-3 * expected.abs().max(), rtol=0
                    )

    def _check_llama(
        self,
        model,
        kv_cache_dtype,
        atol_fraction: float,
        engines=_ENGINES,
        max_seq: int = _MAX_SEQ,
        flags: tuple[str, ...] = (),
        engine_overrides: dict[str, tuple[float, tuple[str, ...]]] | None = None,
    ):
        """Exports once and runs the one saved package on every engine, with
        `engine_overrides[runner]` = (atol_fraction, flags) for that runner."""
        prompt = torch.randint(0, model.config.vocab_size, (1, 7))
        eager = LastTokenLM(copy.deepcopy(model), max_seq, kv_cache_dtype).eval()
        wrapper = LastTokenLM(model, max_seq, kv_cache_dtype).eval()
        steps = [(prompt, torch.arange(7))]
        expected = []
        with torch.no_grad():
            for position in range(7, 10):
                expected.append(eager(*steps[-1]))
                next_id = expected[-1][0, -1].argmax().reshape(1, 1)
                steps.append((next_id, torch.tensor([position])))
            steps.pop()
            seq = Dim("seq", min=1, max=max_seq - 1)
            ep = torch.export.export(
                wrapper,
                steps[0],
                dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                strict=True,
            )
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "tiny_llama.ptn"
            to_native(ep).save(str(package))
            tokens = {}
            for runner in engines:
                atol, runner_flags = (engine_overrides or {}).get(
                    runner, (atol_fraction, flags)
                )
                tokens |= self._check_engines(
                    package, steps, expected, outdir, atol, (runner,), runner_flags
                )
        greedy = [e[0, -1].argmax().item() for e in expected]
        for runner, chosen in tokens.items():
            self.assertEqual(chosen, greedy, runner)

    def test_tiny_llama(self):
        torch.manual_seed(0)
        self._check_llama(_tiny_llama(), torch.float32, atol_fraction=1e-3)

    def test_tiny_llama_gguf(self):
        # A file of every type in _MIXED_TYPES, read back from disk.
        torch.manual_seed(0)
        model = _tiny_llama()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.gguf"
            _write_gguf(path, _gguf_weights(model))
            load_gguf(model, str(path))
        self.assertEqual(
            {
                p.ggml_type
                for p in model.parameters()
                if isinstance(p, ExportableGGUFTensor)
            },
            set(_MIXED_TYPES.values()),
        )
        # Both engines run one unchanged package. native-vk decodes the GGUF
        # weights exactly to fp32. ggml, once allowed to, requantizes each
        # block of activations to int8 for every GGUF matmul, and with these
        # types that rounding reaches 6% of the largest logit. Its bound is set
        # above that, so on ggml this test catches only gross faults, such as a
        # wrong codec or role or a doubled scale; test_ggml_ops checks each
        # type exactly.
        torch.manual_seed(0)
        self._check_llama(
            model,
            torch.float32,
            atol_fraction=1e-3,
            engines=("ggml_ptn_runner", "vulkan_ptn_runner"),
            engine_overrides={
                "ggml_ptn_runner": (
                    1e-1,
                    ("--allow_activation_quant_for_weight_only",),
                )
            },
        )

    def test_tiny_llama_gguf_dequantized_twin_matches_its_golden_samples(self):
        torch.manual_seed(0)
        model = _tiny_llama()
        pairs = _gguf_weights(model)
        load_gguf(model, "tiny.gguf", "llama", lambda _: iter(pairs), dequantize=True)
        wrapper = LastTokenLM(model, _MAX_SEQ).eval()
        seq = Dim("seq", min=1, max=_MAX_SEQ - 1)
        with torch.no_grad():
            ep = torch.export.export(
                wrapper,
                (torch.tensor([[3, 4, 5, 6, 7, 8, 9]]), torch.arange(7)),
                dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
                strict=True,
            )
        golden = golden_samples(wrapper, [[3, 4, 5, 6, 7, 8, 9], [200, 17]], 3)
        self.assertEqual(golden.tol, (1e-3, 1e-3))
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "twin.ptn"
            to_native(ep).save(str(package))
            _, constants = read_ptn(str(package))
            self.assertNotIn(torch.uint8, {t.dtype for t in constants.values()})
            atol, rtol = golden.tol
            for i, sample in enumerate(golden.samples):
                steps = [tuple(call.inputs) for call in sample.calls]
                outputs = _run("ggml_ptn_runner", package, steps, outdir, resize=True)
                for step, (got, call) in enumerate(
                    zip(outputs, sample.calls, strict=True)
                ):
                    torch.testing.assert_close(
                        got,
                        call.outputs[0],
                        atol=atol,
                        rtol=rtol,
                        msg=lambda m, i=i, step=step: f"sample {i} call {step}: {m}",
                    )

    def _check_gguf_dynamic_tokens(self, ggml_type: str):
        """A linear and an embedding of synthetic `ggml_type` weights, each
        exported once with a dynamic token count and run at 9, 1 and 33 tokens
        from the one saved package on every engine."""
        generator = torch.Generator().manual_seed(0)
        linear = torch.nn.Linear(256, 8, bias=False)
        linear.weight = torch.nn.Parameter(
            _GGUF_WEIGHTS[ggml_type](8, 256, generator), requires_grad=False
        )
        embedding = torch.nn.Embedding(8, 256)
        embedding.weight = torch.nn.Parameter(
            _GGUF_WEIGHTS[ggml_type](8, 256, generator), requires_grad=False
        )
        # Activations of +-127/1024, alternating by token, survive ggml's int8
        # requantization exactly, and native-vk decodes the weights exactly.
        rows = torch.full((2, 256), 127 / 1024)
        rows[1, ::2] *= -1
        lengths = (9, 1, 33)
        tokens = Dim("tokens", min=1, max=64)
        flags = {"ggml_ptn_runner": ("--allow_activation_quant_for_weight_only",)}
        for name, model, inputs in (
            ("linear", linear, [rows[torch.arange(n) % 2] for n in lengths]),
            ("embedding", embedding, [torch.arange(n) % 8 for n in lengths]),
        ):
            steps = [(x,) for x in inputs]
            with torch.no_grad():
                expected = [model(x) for x in inputs]
                ep = torch.export.export(
                    model, steps[0], dynamic_shapes={"input": {0: tokens}}
                )
            with tempfile.TemporaryDirectory() as tmp:
                outdir = Path(tmp)
                package = outdir / f"{name}.ptn"
                to_native(ep).save(str(package))
                for runner in _ENGINES:
                    with self.subTest(case=name, engine=runner):
                        outputs = _run(
                            runner,
                            package,
                            steps,
                            outdir,
                            resize=True,
                            flags=flags.get(runner, ()),
                        )
                        self.assertEqual(len(outputs), len(expected))
                        for actual, reference in zip(outputs, expected):
                            torch.testing.assert_close(
                                actual, reference, atol=1e-5, rtol=1e-5
                            )

    def test_gguf_q4_k_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q4_k")

    def test_gguf_q5_k_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q5_k")

    def test_gguf_q6_k_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q6_k")

    def test_gguf_q8_0_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q8_0")

    def test_gguf_q3_k_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q3_k")

    def test_gguf_q2_k_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q2_k")

    def test_gguf_q4_0_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q4_0")

    def test_gguf_q5_0_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q5_0")

    def test_gguf_q5_1_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q5_1")

    def test_gguf_q4_1_dynamic_tokens(self):
        self._check_gguf_dynamic_tokens("q4_1")

    def test_tiny_llama_fp16_kv(self):
        torch.manual_seed(0)
        engines = tuple(e for e in _ENGINES if e not in _FP32_KV_ONLY)
        self._check_llama(
            _tiny_llama(), torch.float16, atol_fraction=2e-2, engines=engines
        )

    def test_tiny_llama_fp16_kv_long_cache(self):
        # A cache longer than a ggml attention key-extent bucket (256), so
        # the engine attends only the prefix the mask leaves open.
        torch.manual_seed(0)
        engines = tuple(e for e in _ENGINES if e not in _FP32_KV_ONLY)
        self._check_llama(
            _tiny_llama(),
            torch.float16,
            atol_fraction=2e-2,
            engines=engines,
            max_seq=600,
        )

    def test_tiny_gemma2_sliding_window(self):
        torch.manual_seed(0)
        self._check_llama(
            _tiny_gemma2(attn_logit_softcapping=None),
            torch.float32,
            atol_fraction=1e-3,
            engines=("ggml_ptn_runner",),
        )

    def test_tiny_gemma2_attention_softcapping(self):
        torch.manual_seed(0)
        self._check_llama(
            _tiny_gemma2(attn_logit_softcapping=1.0),
            torch.float32,
            atol_fraction=1e-3,
            engines=("ggml_ptn_runner",),
        )

    # native-vk: Gemma2's aten.rms_norm.default is not lowered, so loading fails.
    @unittest.expectedFailure
    def test_tiny_gemma2_sliding_window_vulkan(self):
        torch.manual_seed(0)
        self._check_llama(
            _tiny_gemma2(attn_logit_softcapping=None),
            torch.float32,
            atol_fraction=1e-3,
            engines=("vulkan_ptn_runner",),
        )

    @unittest.skip(
        "native-vk: LowerHFAttention only rewrites HF attention over FP32 KV "
        "caches, so an FP16-KV PTN fails at the unsqueeze of cache_position"
    )
    def test_tiny_llama_fp16_kv_vulkan(self):
        torch.manual_seed(0)
        self._check_llama(
            _tiny_llama(), torch.float16, atol_fraction=2e-2, engines=_FP32_KV_ONLY
        )


if __name__ == "__main__":
    unittest.main()
