# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""triton::linear (triton/kernels/gemm): its layout ABI, the accuracy of its
strict FP32 schedule against an FP64 reference and cuBLAS, and an AOTI export
that routes linears to it.

Accuracy gate, per call: with ``scale = max(1, max|reference|)``, the largest
absolute error must stay within ``max(1e-6 * scale, 3 * cuBLAS's)``.

    python -m pytest backends/cuda/tests/test_gemm_linear.py -v
"""

import glob
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

_PASS_LOGGER = "executorch.backends.cuda.triton.replacement_pass"


class _GemmTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        # Registers triton::linear.
        import executorch.backends.cuda.triton.kernels.gemm.linear  # noqa: F401

        torch.manual_seed(0)

    def linear(self, x, w, b=None, strides=None, config=None):
        from executorch.backends.cuda.triton.kernels.gemm.linear import (
            force_linear_candidate,
            tile,
        )

        with force_linear_candidate("ieee_tile", config or tile.CONFIGS[0]):
            return torch.ops.triton.linear(x, w, b, strides)

    def check(self, out, x, w, b=None):
        """``out`` within the accuracy gate."""
        reference = F.linear(x.double(), w.double(), None if b is None else b.double())
        cublas = (F.linear(x, w, b).double() - reference).abs().max().item()
        error = (out.double() - reference).abs().max().item()
        scale = max(1.0, reference.abs().max().item())
        self.assertLessEqual(error, max(1e-6 * scale, 3 * cublas), (error, cublas))


class LayoutTest(_GemmTest):
    def test_every_admitted_layout_matches_its_output_strides(self):
        w = torch.randn(48, 40, device="cuda")
        b = torch.randn(48, device="cuda")
        wide = torch.randn(9, 7, 50, device="cuda")
        cases = {
            "2-D": (torch.randn(6, 40, device="cuda"), None),
            "higher rank": (torch.randn(2, 3, 5, 40, device="cuda"), None),
            "1-D input": (torch.randn(40, device="cuda"), None),
            "transposed input": (torch.randn(40, 6, device="cuda").t(), None),
            "sliced input with an offset": (wide[1:, 2, 3:43], None),
            "singleton dims with odd strides": (
                torch.randn(1, 6, 1, 40, device="cuda").as_strided(
                    (1, 6, 1, 40), (7, 40, 3, 1)
                ),
                None,
            ),
            "transposed output": (torch.randn(6, 40, device="cuda"), [1, 6]),
            "permuted output": (torch.randn(2, 3, 40, device="cuda"), [3, 1, 6]),
        }
        for label, (x, strides) in cases.items():
            for bias in (None, b):
                with self.subTest(label, bias=bias is not None):
                    out = self.linear(x, w, bias, strides)
                    self.assertEqual(tuple(out.shape), (*x.shape[:-1], 48))
                    if strides is not None:
                        self.assertEqual(list(out.stride()), strides)
                    self.check(out, x, w, bias)
                    fake = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
                    with fake:
                        meta = torch.ops.triton.linear(x, w, bias, strides)
                    self.assertEqual(meta.stride(), out.stride())

    def test_weight_strides_are_followed(self):
        x = torch.randn(5, 40, device="cuda")
        w = torch.randn(40, 24, device="cuda").t()
        self.check(self.linear(x, w), x, w)

    def test_layouts_one_step_outside_are_refused(self):
        from executorch.backends.cuda.triton.kernels.gemm.linear import layout_reason

        w = torch.randn(32, 64, device="cuda")
        x3 = torch.randn(2, 3, 64, device="cuda")
        cases = {
            "the leading dims of input": (x3.transpose(0, 1), w, None, [64, 32, 1]),
            "the leading dims of output strides": (x3, w, None, [32, 64, 1]),
            "not a dense layout": (x3, w, None, [192, 64, 1]),
            "differ in K": (torch.randn(2, 63, device="cuda"), w, None, [32, 1]),
            "needs float32": (x3.half(), w.half(), None, [96, 32, 1]),
            "is not a contiguous [32] tensor": (
                x3,
                w,
                torch.randn(64, device="cuda")[::2],
                [96, 32, 1],
            ),
            "empty input": (torch.randn(0, 64, device="cuda"), w, None, [32, 1]),
        }
        for expected, (x, weight, bias, strides) in cases.items():
            with self.subTest(expected):
                self.assertIn(expected, layout_reason(x, weight, bias, strides) or "")
                with self.assertRaisesRegex(ValueError, "triton::linear cannot run"):
                    torch.ops.triton.linear(x, weight, bias, strides)


class AccuracyTest(_GemmTest):
    """Every config on calls across the row classes, and the cases where
    summation order shows."""

    def test_every_config(self):
        from executorch.backends.cuda.triton.kernels.gemm import tile

        for m, n, k in ((1, 256, 2048), (5, 300, 77), (64, 128, 512), (65, 96, 1000)):
            x = torch.randn(m, k, device="cuda")
            w = torch.randn(n, k, device="cuda") / k**0.5
            b = torch.randn(n, device="cuda")
            for config in tile.CONFIGS:
                with self.subTest(m=m, n=n, k=k, config=str(config)):
                    self.check(self.linear(x, w, b, config=config), x, w, b)

    def test_autotuned_long_reductions(self):
        for m, n, k in ((1, 512, 16384), (4, 256, 8192), (300, 128, 4096)):
            x = torch.randn(m, k, device="cuda")
            w = torch.randn(n, k, device="cuda") / k**0.5
            b = torch.randn(n, device="cuda")
            with self.subTest(m=m, n=n, k=k):
                self.check(torch.ops.triton.linear(x, w, b), x, w, b)

    def test_cancellation_mixed_magnitudes_and_zeros(self):
        k = 1024
        x = torch.randn(8, k, device="cuda")
        w = torch.randn(64, k, device="cuda")
        cancelling = torch.cat([x[:, : k // 2], -x[:, : k // 2]], dim=1)
        magnitudes = x * torch.logspace(-6, 6, k, device="cuda")
        b = -F.linear(x, w)[0]
        for label, (xi, bi) in {
            "cancelling sums": (cancelling, None),
            "mixed magnitudes": (magnitudes, None),
            "bias cancelling the first row": (x, b),
            "zeros": (torch.zeros_like(x), None),
        }.items():
            with self.subTest(label):
                self.check(self.linear(xi, w, bi), xi, w, bi)

    def test_nan_and_inf_propagate(self):
        x = torch.randn(4, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        x[0, 3] = float("nan")
        x[1, 5] = float("inf")
        out = self.linear(x, w)
        reference = F.linear(x, w)
        self.assertTrue(torch.equal(out.isnan(), reference.isnan()))
        self.assertTrue(torch.equal(out.isinf(), reference.isinf()))

    def test_repeated_launches_are_bitwise_identical(self):
        x = torch.randn(33, 1000, device="cuda")
        w = torch.randn(70, 1000, device="cuda")
        first = self.linear(x, w)
        self.assertTrue(torch.equal(first, self.linear(x, w)))

    def test_64_bit_indexing(self):
        k, n = 64, 32
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            self.skipTest("needs 12 GiB of free device memory")
        storage = torch.randn(2**31 + k, device="cuda")
        # Two rows 2^31 elements apart: offsets past the int32 range.
        x = storage.as_strided((2, k), (2**31, 1))
        w = torch.randn(n, k, device="cuda")
        out = self.linear(x, w)
        self.check(out, x.contiguous(), w)
        del storage


class _Linears(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(64, 96)
        self.down = nn.Linear(96, 32, bias=False)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


class LoweringTest(_GemmTest):
    def test_aoti_export_runs_the_routed_linears(self):
        """With the static choice taking every call, both lowering APIs route
        both linears, and the program matches eager (outside fbcode)."""
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.backends.cuda.tests.routing_test_utils import (
            static_routing_compile_specs,
        )
        from executorch.backends.cuda.triton.kernels.gemm import routing
        from executorch.exir import (
            EdgeCompileConfig,
            to_edge,
            to_edge_transform_and_lower,
        )
        from executorch.export.utils import is_fbcode

        model = _Linears().cuda().eval()
        x = torch.randn(2, 5, 64, device="cuda")
        with torch.no_grad():
            program = torch.export.export(model, (x,), strict=True)
            expected = model(x)
        config = EdgeCompileConfig(_check_ir_validity=False)
        for api in ("to_edge_transform_and_lower", "to_edge_to_backend"):
            partitioner = CudaPartitioner(static_routing_compile_specs())
            with self.subTest(api), mock.patch.object(
                routing, "worth_routing", return_value=None
            ), self.assertLogs(_PASS_LOGGER, "INFO") as logs:
                if api == "to_edge_to_backend":
                    edge = to_edge(program, compile_config=config).to_backend(
                        partitioner
                    )
                else:
                    edge = to_edge_transform_and_lower(
                        program, partitioner=[partitioner], compile_config=config
                    )
                self.assertIn(
                    "Replaced 2 nodes with Triton kernels", "\n".join(logs.output)
                )
                if is_fbcode():
                    # Running the program is skipped in fbcode.
                    continue
                self.assertClose(edge, x, expected)

    def assertClose(self, edge, x, expected):
        from executorch.runtime import Runtime

        program = edge.to_executorch()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pte")
            with open(path, "wb") as f:
                program.write_to_file(f)
            program.write_tensor_data_to_file(directory)
            data_path = glob.glob(os.path.join(directory, "*.ptd"))[0]
            method = Runtime.get().load_program(path, data_path=data_path)
            (out,) = method.load_method("forward").execute([x.cpu()])
        torch.testing.assert_close(out, expected.cpu(), atol=1e-5, rtol=1e-5)
