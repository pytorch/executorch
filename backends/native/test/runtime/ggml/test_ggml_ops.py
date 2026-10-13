# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export single-operator modules to .ptn and check the ggml CPU engine
against eager PyTorch."""

import copy
import subprocess
import tempfile
import unittest
from importlib.resources import as_file, files
from pathlib import Path

import torch
import torch.nn.functional as F
from executorch.exir.native import to_native
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    MappingType,
    quantize_,
)
from torchao.quantization.granularity import PerGroup
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
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


class CacheWrite(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1, 2, 8, 4))

    def forward(self, values, positions):
        self.cache.index_copy_(2, positions, values)
        return self.cache * 1.0


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


def _run_ggml(package: Path, inputs, outdir: Path, num_outputs: int):
    input_paths = []
    for i, tensor in enumerate(inputs):
        path = outdir / f"input{i}.bin"
        path.write_bytes(tensor.contiguous().flatten().view(torch.uint8).numpy())
        input_paths.append(str(path))
    output_paths = [str(outdir / f"output{i}.bin") for i in range(num_outputs)]
    with as_file(files(__package__) / "ggml_ptn_runner") as runner:
        result = subprocess.run(
            [
                str(runner),
                f"--package={package}",
                f"--inputs={','.join(input_paths)}",
                f"--outputs={','.join(output_paths)}",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        raise AssertionError(f"ggml_ptn_runner failed:\n{result.stderr}")
    shapes = [tuple(map(int, line.split())) for line in result.stdout.splitlines()]
    return [
        (Path(path).read_bytes(), shape) for path, shape in zip(output_paths, shapes)
    ]


class GgmlOpsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _check(self, model, inputs, atol=1e-5, rtol=1e-5):
        model = model.eval()
        # Eager writes buffers in place; the package must keep them as constructed.
        with torch.no_grad():
            expected = copy.deepcopy(model)(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(model, inputs)).save(str(package))
            outputs = _run_ggml(package, inputs, outdir, len(expected))
        self.assertEqual(len(outputs), len(expected))
        for (raw, shape), reference in zip(outputs, expected):
            self.assertEqual(shape, tuple(reference.shape))
            actual = torch.frombuffer(bytearray(raw), dtype=reference.dtype)
            torch.testing.assert_close(
                actual.reshape(shape), reference, atol=atol, rtol=rtol
            )

    def _run_exported(self, model, example, inputs):
        """Exports `model` at `example`, before any eager call writes its
        buffers, and returns its float outputs on `inputs`."""
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(model, example)).save(str(package))
            outputs = _run_ggml(package, inputs, outdir, 1)
        return [
            torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
            for raw, shape in outputs
        ]

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

    def test_sub_scalar(self):
        self._check(Lambda(lambda x: x - 2.0), (torch.randn(2, 3, 4),))

    def test_add_scalar_permuted(self):
        self._check(Lambda(lambda x: x.permute(1, 0) + 1.0), (torch.randn(2, 3),))

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

    def test_silu(self):
        self._check(Lambda(F.silu), (torch.randn(2, 3, 4),))

    def test_sigmoid(self):
        self._check(Lambda(torch.sigmoid), (torch.randn(2, 3, 4),))

    def test_neg(self):
        self._check(Lambda(torch.neg), (torch.randn(2, 3, 4),))

    def test_linear(self):
        self._check(Linear(bias=False), (torch.randn(4, 32),), atol=1e-4)

    def test_linear_bias(self):
        self._check(Linear(bias=True), (torch.randn(2, 4, 32),), atol=1e-4)

    def test_mm(self):
        self._check(Lambda(torch.mm), (torch.randn(4, 8), torch.randn(8, 5)), atol=1e-4)

    def test_bmm(self):
        self._check(
            Lambda(torch.bmm), (torch.randn(2, 4, 8), torch.randn(2, 8, 5)), atol=1e-4
        )

    def test_matmul(self):
        self._check(
            Lambda(torch.matmul),
            (torch.randn(2, 4, 8), torch.randn(2, 8, 5)),
            atol=1e-4,
        )

    def test_matmul_broadcast_rhs(self):
        self._check(
            Lambda(torch.matmul), (torch.randn(2, 4, 8), torch.randn(8, 5)), atol=1e-4
        )

    def test_embedding(self):
        self._check(Embedding(), (torch.tensor([[1, 3, 9], [0, 2, 5]]),))

    def test_embedding_squeezed_indices(self):
        class SqueezedEmbedding(Embedding):
            def forward(self, indices):
                return self.embedding(indices.squeeze(0))

        self._check(SqueezedEmbedding(), (torch.tensor([[1, 3, 9]]),))

    def test_rms_norm(self):
        self._check(RMSNorm(), (torch.randn(2, 3, 16),))

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

    def test_cat_of_mixed_ggml_types(self):
        # The F16 embedding result is F32 in ggml; the F16 input stays F16.
        self._check(
            HalfEmbeddingCat(),
            (
                torch.randn(2, 3, 4, dtype=torch.float16),
                torch.tensor([[1, 3, 9], [0, 2, 5]]),
            ),
        )

    def test_cat_last_dim(self):
        self._check(
            Lambda(lambda x, y: torch.cat([x, y], dim=-1)),
            (torch.randn(2, 3, 4), torch.randn(2, 3, 2)),
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
        # Only the Q4_0 fp16 scales round, so this fails on a nibble swap.
        self._check(
            model, (torch.tensor([[1, 3, 9], [0, 2, 5]]),), atol=2e-3, rtol=1e-3
        )

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

    def test_index_put_cache_write(self):
        self._check(CacheWrite(), (torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6])))

    def test_index_put_rejects_out_of_range(self):
        inputs = (torch.randn(1, 2, 3, 4), torch.tensor([5, 1, 6]))
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            package = outdir / "model.ptn"
            to_native(torch.export.export(CacheWrite(), inputs)).save(str(package))
            with self.assertRaisesRegex(AssertionError, "outside"):
                _run_ggml(package, (inputs[0], torch.tensor([5, 1, 8])), outdir, 1)

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


if __name__ == "__main__":
    unittest.main()
