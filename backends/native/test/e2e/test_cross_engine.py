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

import torch
import torchvision
from executorch.backends.native.examples.hf_llm.export_hf_llm import (
    LastTokenLM,
    load_gguf,
)
from executorch.exir.native import to_native
from executorch.extension.llm.export.gguf import ExportableGGUFTensor
from torch.export import Dim
from transformers import LlamaConfig, LlamaForCausalLM


_ENGINES = ("ggml_ptn_runner", "vulkan_ptn_runner")
_FP32_KV_ONLY = ("vulkan_ptn_runner",)
_MAX_SEQ = 16


def _run(runner: str, package: Path, steps, outdir: Path, resize: bool = False):
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
        for path, size in zip(output_steps, sizes)
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


def _q4_k(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 144-byte blocks: fp16 d and dmin, 12 bytes of 6-bit scales and mins, then
    # 128 bytes of nibbles. d and dmin give weights of HF-init magnitude, which
    # keeps attention from saturating on random weights.
    raw = torch.randint(0, 256, (rows, cols // 256, 144), generator=generator)
    raw = raw.to(torch.uint8)
    raw[..., 0:2] = torch.tensor([0x00, 0x0C], dtype=torch.uint8)  # d = 2^-12
    raw[..., 2:4] = torch.tensor([0x00, 0x18], dtype=torch.uint8)  # dmin = 2^-9
    return ExportableGGUFTensor.from_raw(raw.reshape(rows, -1), "q4_k", torch.float32)


def _q6_k(rows: int, cols: int, generator: torch.Generator) -> ExportableGGUFTensor:
    # 210-byte blocks: 128 low and 64 high bit bytes, 16 int8 scales, fp16 d.
    raw = torch.randint(0, 256, (rows, cols // 256, 210), generator=generator)
    raw = raw.to(torch.uint8)
    scales = torch.randint(-8, 9, (rows, cols // 256, 16), generator=generator)
    raw[..., 192:208] = scales.to(torch.int8).view(torch.uint8)
    raw[..., 208:210] = torch.tensor([0x00, 0x0C], dtype=torch.uint8)  # d = 2^-12
    return ExportableGGUFTensor.from_raw(raw.reshape(rows, -1), "q6_k", torch.float32)


def _gguf_weights(model: LlamaForCausalLM) -> list[tuple[str, torch.Tensor]]:
    """Synthetic Q4_K_M-style GGUF tensors under llama.cpp names."""
    generator = torch.Generator().manual_seed(3)
    config = model.config
    hidden, inner = config.hidden_size, config.intermediate_size
    kv = config.num_key_value_heads * config.head_dim

    def norm() -> torch.Tensor:
        return torch.rand(hidden, generator=generator) + 0.5

    pairs = [
        ("token_embd.weight", _q6_k(config.vocab_size, hidden, generator)),
        ("output_norm.weight", norm()),
    ]
    for i in range(config.num_hidden_layers):
        pairs += [
            (f"blk.{i}.attn_norm.weight", norm()),
            (f"blk.{i}.attn_q.weight", _q4_k(hidden, hidden, generator)),
            (f"blk.{i}.attn_k.weight", _q4_k(kv, hidden, generator)),
            (f"blk.{i}.attn_v.weight", _q6_k(kv, hidden, generator)),
            (f"blk.{i}.attn_output.weight", _q4_k(hidden, hidden, generator)),
            (f"blk.{i}.ffn_norm.weight", norm()),
            (f"blk.{i}.ffn_gate.weight", _q4_k(inner, hidden, generator)),
            (f"blk.{i}.ffn_up.weight", _q4_k(inner, hidden, generator)),
            (f"blk.{i}.ffn_down.weight", _q6_k(hidden, inner, generator)),
        ]
    return pairs


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
        # ggml does not lower aten.native_layer_norm yet.
        self._check_dynamic(model, steps, ({1: seq},), engines=("vulkan_ptn_runner",))

    def test_inplace_gelu_and_sigmoid(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3)]
        seq = Dim("seq", min=2, max=12)
        # ggml does not lower aten.gelu yet.
        self._check_dynamic(
            _GeluSigmoid(), steps, ({1: seq},), engines=("vulkan_ptn_runner",)
        )

    def test_padding_mask_of_dynamic_positions(self):
        torch.manual_seed(0)
        steps = [
            (torch.randn(1, s, 8), torch.tensor([n]))
            for s, n in ((6, 4), (3, 3), (9, 2))
        ]
        seq = Dim("seq", min=2, max=12)
        # ggml does not lower aten.lt yet.
        self._check_dynamic(
            _PaddingMask(), steps, ({1: seq}, {}), engines=("vulkan_ptn_runner",)
        )

    def test_inplace_relu_of_dynamic_view(self):
        torch.manual_seed(0)
        steps = [(torch.randn(1, s, 16),) for s in (5, 3, 9)]
        seq = Dim("seq", min=2, max=12)
        self._check_dynamic(_ViewRelu(), steps, ({1: seq},))

    def _check_engines(self, package, steps, expected, outdir, atol_fraction, engines):
        tokens = {}
        for runner in engines:
            with self.subTest(engine=runner):
                outputs = _run(runner, package, steps, outdir, resize=True)
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
        self, model, kv_cache_dtype, atol_fraction: float, engines=_ENGINES
    ):
        prompt = torch.randint(0, model.config.vocab_size, (1, 7))
        eager = LastTokenLM(copy.deepcopy(model), _MAX_SEQ, kv_cache_dtype).eval()
        wrapper = LastTokenLM(model, _MAX_SEQ, kv_cache_dtype).eval()
        steps = [(prompt, torch.arange(7))]
        expected = []
        with torch.no_grad():
            for position in range(7, 10):
                expected.append(eager(*steps[-1]))
                next_id = expected[-1][0, -1].argmax().reshape(1, 1)
                steps.append((next_id, torch.tensor([position])))
            steps.pop()
            seq = Dim("seq", min=1, max=_MAX_SEQ - 1)
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
            tokens = self._check_engines(
                package, steps, expected, outdir, atol_fraction, engines
            )
        greedy = [e[0, -1].argmax().item() for e in expected]
        for runner, chosen in tokens.items():
            self.assertEqual(chosen, greedy, runner)

    def test_tiny_llama(self):
        torch.manual_seed(0)
        self._check_llama(_tiny_llama(), torch.float32, atol_fraction=1e-3)

    def test_tiny_llama_gguf(self):
        torch.manual_seed(0)
        model = _tiny_llama()
        pairs = _gguf_weights(model)
        load_gguf(model, "tiny.gguf", "llama", lambda _: iter(pairs))
        # ggml requantizes activations to Q8_K for its K-quant matmuls, while
        # native-vk decodes the K-quant weights exactly to fp32. Re-seeding
        # gives both engines the same prompt.
        for runner, atol_fraction in (
            ("ggml_ptn_runner", 5e-2),
            ("vulkan_ptn_runner", 1e-3),
        ):
            torch.manual_seed(0)
            self._check_llama(
                copy.deepcopy(model), torch.float32, atol_fraction, engines=(runner,)
            )

    def test_tiny_llama_fp16_kv(self):
        torch.manual_seed(0)
        engines = tuple(e for e in _ENGINES if e not in _FP32_KV_ONLY)
        self._check_llama(
            _tiny_llama(), torch.float16, atol_fraction=2e-2, engines=engines
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
