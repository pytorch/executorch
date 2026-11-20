# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""triton::linear (triton/kernels/gemm): its layout ABI, the accuracy of
every schedule, precision and K split against an FP64 reference and cuBLAS,
the bounds on split-K, and AOTI exports that route linears to it.

Accuracy gate, per call: with ``scale = max(1, max|reference|)``, the largest
absolute error must stay within ``max(1e-6 * scale, 3 * cuBLAS's)``.

    python -m pytest backends/cuda/tests/test_gemm_linear.py -v
"""

import glob
import importlib
import itertools
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

_PASS_LOGGER = "executorch.backends.cuda.triton.replacement_pass"
_TILES = ("ieee_tile", "tf32x3_tile")
# The package exports the op under the module's name.
_LINEAR_MODULE = "executorch.backends.cuda.triton.kernels.gemm.linear"


class _GemmTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        # Registers triton::linear.
        import executorch.backends.cuda.triton.kernels.gemm.linear  # noqa: F401

        torch.manual_seed(0)

    def linear(
        self, x, w, b=None, strides=None, config=None, schedule="ieee_tile", split=1
    ):
        from executorch.backends.cuda.triton.kernels.gemm import gemv, tile
        from executorch.backends.cuda.triton.kernels.gemm.linear import (
            force_linear_candidate,
        )

        configs = gemv.CONFIGS if schedule == "gemv" else tile.CONFIGS
        with force_linear_candidate(schedule, config or configs[0], split):
            return torch.ops.triton.linear(x, w, b, strides)

    def tiles(self):
        """The tile schedules this GPU runs."""
        return _TILES if torch.cuda.get_device_capability()[0] >= 8 else _TILES[:1]

    def schedules(self, x):
        """The schedules that run ``x``'s rows."""
        rows = x.numel() // x.shape[-1]
        return (*self.tiles(), "gemv") if rows <= 4 else self.tiles()

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
        cases["a single row"] = (torch.randn(1, 40, device="cuda"), None)
        cases["transposed rows"] = (torch.randn(40, 3, device="cuda").t(), [1, 3])
        for label, (x, strides) in cases.items():
            # The precisions address memory alike; one stands for both.
            schedules = [s for s in self.schedules(x) if s != "tf32x3_tile"]
            for bias, schedule in itertools.product((None, b), schedules):
                with self.subTest(label, bias=bias is not None, schedule=schedule):
                    out = self.linear(x, w, bias, strides, schedule=schedule)
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

    def test_gemv_loads_vectors_only_from_aligned_rows_and_bases(self):
        from executorch.backends.cuda.triton.kernels.gemm import gemv
        from executorch.backends.cuda.triton.kernels.gemm.linear import linear_layout

        k, n = 256, 96
        w = torch.randn(n, k, device="cuda")
        b = torch.randn(n, device="cuda")
        storage = torch.randn(4 * (k + 8) + 8, device="cuda")
        weights = torch.randn(n * k + 3, device="cuda")

        def rows(offset, stride):
            return storage[offset : offset + 4 * stride].view(4, stride)[:, :k]

        cases = {"aligned rows at offset 4": (rows(4, k + 8), w, 4)}
        cases["an odd padded row stride"] = (rows(0, k + 1), w, 1)
        for offset in (1, 2, 3):
            cases[f"input offset {offset}"] = (rows(offset, k + 8), w, 1)
            weight = weights[offset : offset + n * k].view(n, k)
            cases[f"weight offset {offset}"] = (rows(0, k), weight, 1)
        for label, (x, weight, vec) in cases.items():
            with self.subTest(label):
                layout = linear_layout(x, weight, (n, 1))
                self.assertEqual(gemv.vector_width(x, weight, layout), vec)
                self.check(self.linear(x, weight, b, schedule="gemv"), x, weight, b)

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
    """Every config, precision and split on calls across the row classes, and
    the cases where summation order shows."""

    def test_every_config(self):
        """Whole in FP32 FMAs on calls across the row classes, and split in
        two in tf32x3 (which shares the split's stores) on a ragged one."""
        from executorch.backends.cuda.triton.kernels.gemm import tile

        runs = [
            (call, "ieee_tile", 1)
            for call in ((1, 256, 2048), (5, 300, 77), (65, 96, 1000))
        ]
        runs += [((65, 96, 1000), t, 2) for t in self.tiles()[1:]]
        for (m, n, k), schedule, split in runs:
            x = torch.randn(m, k, device="cuda")
            w = torch.randn(n, k, device="cuda") / k**0.5
            b = torch.randn(n, device="cuda")
            for config in tile.CONFIGS:
                with self.subTest(
                    m=m, n=n, k=k, schedule=schedule, split=split, config=str(config)
                ):
                    out = self.linear(
                        x, w, b, config=config, schedule=schedule, split=split
                    )
                    self.check(out, x, w, b)

    def test_every_split(self):
        """Every split, including slices past the end of K that store zeros."""
        from executorch.backends.cuda.triton.kernels.gemm.linear import SPLITS

        # 2100 is 66 blocks of 32: 16 slices of 5 blocks leave the last two empty.
        m, n, k = 70, 130, 2100
        x = torch.randn(m, k, device="cuda")
        w = torch.randn(n, k, device="cuda") / k**0.5
        b = torch.randn(n, device="cuda")
        for schedule, split in itertools.product(self.tiles(), SPLITS):
            with self.subTest(m=m, n=n, k=k, schedule=schedule, split=split):
                out = self.linear(x, w, b, schedule=schedule, split=split)
                self.check(out, x, w, b)

    def test_split_reduction_adds_the_bias_once_into_any_output_layout(self):
        w = torch.randn(48, 512, device="cuda")
        b = torch.randn(48, device="cuda")
        wide = torch.randn(9, 7, 520, device="cuda")
        cases = {
            "contiguous": (torch.randn(6, 512, device="cuda"), None),
            "transposed output": (torch.randn(6, 512, device="cuda"), [1, 6]),
            "permuted output": (torch.randn(2, 3, 512, device="cuda"), [3, 1, 6]),
            "sliced input with an offset": (wide[1:, 2, 3:515], None),
        }
        for (label, (x, strides)), schedule in itertools.product(
            cases.items(), self.tiles()
        ):
            with self.subTest(label, schedule=schedule):
                out = self.linear(x, w, b, strides, schedule=schedule, split=4)
                if strides is not None:
                    self.assertEqual(list(out.stride()), strides)
                self.check(out, x, w, b)
                zeros = self.linear(
                    torch.zeros_like(x), w, b, strides, schedule=schedule, split=4
                )
                self.assertTrue(torch.equal(zeros, b.expand_as(zeros)))

    def test_splits_and_schedules_past_their_bounds_are_refused(self):
        linear = importlib.import_module(_LINEAR_MODULE)

        x = torch.randn(8, 512, device="cuda")
        w = torch.randn(64, 512, device="cuda")
        cases = {
            "split_k=8 needs K >= 1024, got K=512": ("ieee_tile", 8, {}),
            "needs a 2x8x64 workspace, more than 1000 elements": (
                "tf32x3_tile",
                2,
                {"MAX_WORKSPACE_ELEMENTS": 1000},
            ),
            "tf32x3_tile needs sm_80": (
                "tf32x3_tile",
                1,
                {"device_supports": lambda capability, device: capability < 80},
            ),
            "gemv does not split K": ("gemv", 2, {}),
            "unknown schedule fp16_tile": ("fp16_tile", 1, {}),
        }
        for expected, (schedule, split, patches) in cases.items():
            with self.subTest(expected), mock.patch.dict(vars(linear), patches):
                with self.assertRaisesRegex(ValueError, expected):
                    self.linear(
                        x[:1] if schedule == "gemv" else x,
                        w,
                        schedule=schedule,
                        split=split,
                    )

    def test_every_gemv_config(self):
        from executorch.backends.cuda.triton.kernels.gemm import gemv

        for m, n, k in ((1, 256, 2048), (2, 300, 77), (3, 129, 1000), (4, 1024, 4096)):
            x = torch.randn(m, k, device="cuda")
            w = torch.randn(n, k, device="cuda") / k**0.5
            b = torch.randn(n, device="cuda")
            for config in gemv.CONFIGS:
                with self.subTest(m=m, n=n, k=k, config=str(config)):
                    out = self.linear(x, w, b, config=config, schedule="gemv")
                    self.check(out, x, w, b)

    def test_gemv_streams_at_most_four_rows(self):
        x = torch.randn(5, 64, device="cuda")
        w = torch.randn(32, 64, device="cuda")
        with self.assertRaisesRegex(ValueError, "at most 4 rows, not 5"):
            self.linear(x, w, schedule="gemv")

    def test_timed_schedule_choice(self):
        """The compile times every tile schedule with the splits K and the
        workspace bound allow, and the matrix-vector schedule for a few rows."""
        from executorch.backends.cuda.autotune.launch_params import (
            autotune_launch_params,
            clear_launch_param_cache,
        )

        linear = importlib.import_module(_LINEAR_MODULE)

        w = torch.randn(512, 1024, device="cuda")
        # 1024 / 128 allows splits up to 8; 5 rows allow a 2x5x512 workspace.
        for rows, splits, workspace in ((1, (1, 2, 4, 8), 2**26), (5, (1, 2), 5120)):
            legal = {(t, s) for t in self.tiles() for s in splits}
            if rows == 1:
                legal.add(("gemv", 1))
            clear_launch_param_cache()
            x = torch.randn(rows, 1024, device="cuda")
            with self.subTest(
                rows=rows
            ), autotune_launch_params() as stats, mock.patch.object(
                linear, "MAX_WORKSPACE_ELEMENTS", workspace
            ):
                out = torch.ops.triton.linear(x, w, None)
                (pick,) = stats.picks
                self.assertEqual(set(pick.times_us), legal)
                self.assertIn(pick.choice, legal)
                self.check(out, x, w)

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
        reference = F.linear(x, w)
        for schedule in self.schedules(x):
            with self.subTest(schedule):
                out = self.linear(x, w, schedule=schedule)
                self.assertTrue(torch.equal(out.isnan(), reference.isnan()))
                self.assertTrue(torch.equal(out.isinf(), reference.isinf()))

    def test_repeated_launches_are_bitwise_identical(self):
        w = torch.randn(70, 1000, device="cuda")
        for x in (
            torch.randn(33, 1000, device="cuda"),
            torch.randn(3, 1000, device="cuda"),
        ):
            for schedule in self.schedules(x):
                for split in (1,) if schedule == "gemv" else (1, 4):
                    with self.subTest(rows=x.shape[0], schedule=schedule, split=split):
                        first = self.linear(x, w, schedule=schedule, split=split)
                        again = self.linear(x, w, schedule=schedule, split=split)
                        self.assertTrue(torch.equal(first, again))

    def test_64_bit_indexing(self):
        k, n = 256, 32
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            self.skipTest("needs 12 GiB of free device memory")
        storage = torch.randn(2**31 + k, device="cuda")
        # Two rows 2^31 elements apart: offsets past the int32 range.
        x = storage.as_strided((2, k), (2**31, 1))
        w = torch.randn(n, k, device="cuda")
        for schedule in self.schedules(x):
            for split in (1,) if schedule == "gemv" else (1, 2):
                with self.subTest(schedule, split=split):
                    out = self.linear(x, w, schedule=schedule, split=split)
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
        both linears, of many rows (the tile) and of one (the matrix-vector
        schedule), and the program matches eager (outside fbcode)."""
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
        config = EdgeCompileConfig(_check_ir_validity=False)
        for api, shape in itertools.product(
            ("to_edge_transform_and_lower", "to_edge_to_backend"), ((2, 5, 64), (1, 64))
        ):
            x = torch.randn(*shape, device="cuda")
            with torch.no_grad():
                program = torch.export.export(model, (x,), strict=True)
                expected = model(x)
            partitioner = CudaPartitioner(static_routing_compile_specs())
            with self.subTest(api, shape=shape), mock.patch.object(
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

    def test_split_k_workspace_under_cuda_graph_replay(self):
        """A forced split allocates its workspace inside the program; the
        method, captured into a CUDA graph and replayed on new inputs, matches
        eager (outside fbcode)."""
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.backends.cuda.tests.routing_test_utils import (
            static_routing_compile_specs,
        )
        from executorch.backends.cuda.triton.kernels.gemm import routing, tile
        from executorch.backends.cuda.triton.kernels.gemm.linear import (
            force_linear_candidate,
        )
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
        from executorch.export.utils import is_fbcode

        model = nn.Linear(512, 96).cuda().eval()
        # Three warmup calls, the capture call, then replays.
        inputs = [torch.randn(2, 5, 512, device="cuda") for _ in range(6)]
        with torch.no_grad():
            program = torch.export.export(model, (inputs[0],), strict=True)
        with mock.patch.object(
            routing, "worth_routing", return_value=None
        ), force_linear_candidate(self.tiles()[-1], split=4), mock.patch.object(
            tile, "run", wraps=tile.run
        ) as run:
            edge = to_edge_transform_and_lower(
                program,
                partitioner=[CudaPartitioner(static_routing_compile_specs())],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
        # tile.run(x, weight, bias, out, layout, precision, split, config)
        self.assertEqual({call.args[6] for call in run.call_args_list}, {4})
        if is_fbcode():
            # Running the program is skipped in fbcode.
            return
        from executorch.runtime import Runtime

        et_program = edge.to_executorch()
        runtime = Runtime.get()
        option = "enable_cuda_graph_for_method"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pte")
            with open(path, "wb") as f:
                et_program.write_to_file(f)
            et_program.write_tensor_data_to_file(directory)
            data_path = glob.glob(os.path.join(directory, "*.ptd"))[0]
            runtime.backend_registry.set_option("CudaBackend", {option: "forward"})
            try:
                method = runtime.load_program(path, data_path=data_path).load_method(
                    "forward"
                )
            finally:
                # The option is process wide.
                runtime.backend_registry.set_option("CudaBackend", {option: ""})
            for i, x in enumerate(inputs):
                (out,) = method.execute([x.cpu()])
                with self.subTest(call=i), torch.no_grad():
                    torch.testing.assert_close(
                        out, model(x).cpu(), atol=1e-5, rtol=1e-5
                    )

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
