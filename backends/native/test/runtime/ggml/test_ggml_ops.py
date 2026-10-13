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

    def test_add_scalar(self):
        self._check(Lambda(lambda x: x + 2.0), (torch.randn(2, 3, 4),))

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


if __name__ == "__main__":
    unittest.main()
