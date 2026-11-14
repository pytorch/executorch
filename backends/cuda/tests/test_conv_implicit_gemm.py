# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""triton::convolution on dense convolutions: every schedule and K order of
the implicit-GEMM template against an FP64 reference."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

_TOLERANCE = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}
_TF32_TOLERANCE = 5e-3


def _reference(x, w, b, stride, before, after):
    pad = [p for pair in zip(reversed(before), reversed(after)) for p in pair]
    conv = F.conv2d if x.dim() == 4 else F.conv3d
    bias = None if b is None else b.double()
    return conv(F.pad(x.double(), pad), w.double(), bias, stride)


def _channels_last(t):
    return t.to(
        memory_format=torch.channels_last if t.dim() == 4 else torch.channels_last_3d
    )


class ImplicitGemmScheduleTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.tensor_cores = torch.cuda.get_device_capability() >= (8, 0)

    def convolution(self, x, w, b, stride, before, after):
        from executorch.backends.cuda.triton.kernels.conv.conv import convolution

        rank = x.dim() - 2
        return convolution(
            x,
            w,
            b,
            list(stride),
            list(before),
            list(after),
            [1] * rank,
            False,
            [0] * rank,
            1,
        )

    def check(self, x, w, b, stride, before, after, every_config=False):
        """Runs every schedule and K order on the call, with the first config of
        each or every config, and checks the result and the layout; the
        tensor-core schedule runs FP32 operands in TF32."""
        from executorch.backends.cuda.triton.kernels.conv.conv import (
            force_conv_candidate,
        )
        from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            ConvCall,
            ConvParams,
            out_shape_and_stride,
        )

        rank = x.dim() - 2
        params = ConvParams(
            tuple(stride),
            tuple(before),
            tuple(after),
            (1,) * rank,
            False,
            (0,) * rank,
            1,
        )
        call = ConvCall(x, w, b, params)
        tf32 = mock.patch.object(implicit_gemm, "conv_allows_tf32", return_value=True)
        self.assertIsNone(implicit_gemm.supports(call))
        expected = _reference(x, w, b, stride, before, after)
        layout = out_shape_and_stride(call)
        schedules = ["igemm_ffma"] + (["igemm_tc"] if self.tensor_cores else [])
        for (schedule, _), configs in implicit_gemm.CONFIGS.items():
            if schedule not in schedules:
                continue
            with tf32:
                self.assertIsNone(implicit_gemm.schedule_supports(schedule, call))
            tolerance = _TOLERANCE[x.dtype]
            if schedule == "igemm_tc" and x.dtype == torch.float32:
                tolerance = _TF32_TOLERANCE
            for config in configs if every_config else configs[:1]:
                with tf32, force_conv_candidate(schedule, config):
                    out = self.convolution(x, w, b, stride, before, after)
                self.assertEqual((tuple(out.shape), out.stride()), layout)
                torch.testing.assert_close(
                    out.double(),
                    expected.to(x.dtype).double(),
                    atol=tolerance * max(1.0, expected.abs().max().item()),
                    rtol=tolerance,
                    equal_nan=True,
                    msg=lambda m, s=schedule, c=config: f"{s} {c}: {m}",
                )

    def tensors(self, shape, out_channels, kernel, dtype=torch.float32, bias=True):
        channels = shape[1]
        x = torch.randn(shape, device="cuda", dtype=dtype)
        fan_in = channels * torch.Size(kernel).numel()
        w = torch.randn(out_channels, channels, *kernel, device="cuda", dtype=dtype) / (
            fan_in**0.5
        )
        b = torch.randn(out_channels, device="cuda", dtype=dtype) if bias else None
        return x, w, b

    def test_every_config_of_every_schedule(self):
        for shape, out_channels, kernel, stride, padding in (
            ((2, 32, 13, 11), 40, (3, 3), (1, 1), (1, 1)),
            ((2, 3, 19, 17), 24, (7, 7), (2, 2), (3, 3)),
        ):
            x, w, b = self.tensors(shape, out_channels, kernel)
            with self.subTest(shape=shape):
                self.check(x, w, b, stride, padding, padding, every_config=True)

    def check_geometries(self, dtype, layouts=("contiguous", "channels_last")):
        """Every schedule on each geometry in ``dtype`` and each of ``layouts``:
        a contiguous or channels-last input, or a channels-last weight."""
        geometries = (
            # input, C_out, kernel, stride, before, after: representative
            # geometries at small sizes, then edge cases.
            ((6, 3, 2, 14, 14), 40, (2, 14, 14), (2, 14, 14), (0,) * 3, (0,) * 3),
            ((1, 24, 3, 6, 10), 33, (1, 2, 2), (1, 2, 2), (0,) * 3, (0,) * 3),
            ((1, 3, 56, 42), 72, (14, 14), (14, 14), (0, 0), (0, 0)),
            ((2, 3, 32, 48), 48, (16, 16), (16, 16), (0, 0), (0, 0)),
            ((3, 3, 29, 31), 40, (7, 7), (4, 4), (3, 3), (3, 3)),
            ((2, 3, 30, 26), 64, (7, 7), (2, 2), (3, 3), (3, 3)),
            ((2, 32, 11, 13), 32, (3, 3), (1, 1), (1, 1), (1, 1)),
            ((1, 48, 12, 9), 64, (3, 3), (2, 2), (1, 1), (1, 1)),
            ((2, 40, 7, 9), 24, (1, 1), (1, 1), (0, 0), (0, 0)),
            ((2, 40, 7, 9), 24, (1, 1), (2, 2), (0, 0), (0, 0)),
            ((1, 3, 41, 39), 32, (7, 7), (2, 2), (0, 0), (0, 0)),
            ((2, 3, 46, 36), 24, (7, 7), (1, 1), (0, 0), (0, 0)),
            ((1, 32, 39, 39), 32, (5, 5), (2, 2), (0, 0), (0, 0)),
            ((1, 32, 8, 8), 32, (3, 3), (2, 2), (0, 0), (0, 0)),
            ((1, 5, 10, 12), 7, (3, 5), (2, 1), (0, 2), (1, 0)),
            ((1, 20, 9, 7), 3, (2, 4), (1, 3), (2, 0), (0, 5)),
            ((1, 17, 4, 9, 8), 18, (3, 3, 3), (1, 2, 1), (2, 1, 1), (0, 1, 1)),
            ((1, 2, 1, 1), 1, (1, 1), (1, 1), (0, 0), (0, 0)),
        )
        cases = [
            (geometry, layout, layout == "weight_cl" or i % 2 == 0)
            for i, geometry in enumerate(geometries)
            for layout in layouts
        ]
        for (shape, c_out, kernel, stride, before, after), layout, bias in cases:
            x, w, b = self.tensors(shape, c_out, kernel, dtype=dtype, bias=bias)
            if layout == "channels_last":
                x = _channels_last(x)
            if layout == "weight_cl":
                w = _channels_last(w)
            with self.subTest(shape=shape, kernel=kernel, stride=stride, layout=layout):
                self.check(x, w, b, stride, before, after)

    def test_geometries_fp32(self):
        self.check_geometries(torch.float32, ("contiguous",))

    def test_geometries_fp32_channels_last(self):
        self.check_geometries(torch.float32, ("channels_last",))

    def test_geometries_fp32_channels_last_weight(self):
        self.check_geometries(torch.float32, ("weight_cl",))

    def test_geometries_fp16(self):
        self.check_geometries(torch.float16)

    def test_geometries_bf16(self):
        self.check_geometries(torch.bfloat16)

    def test_sliced_inputs_with_storage_offset(self):
        base = torch.randn(2, 20, 15, 18, device="cuda")
        _, w, b = self.tensors((1, 16, 1, 1), 24, (3, 3))
        for x in (
            base[:, 2:18, 1:-1, 2:],
            _channels_last(base)[:, 4:, 3:, :-1],
            base[:, 2:18, :, ::2],
        ):
            with self.subTest(stride=x.stride(), offset=x.storage_offset()):
                self.assertGreater(x.storage_offset(), 0)
                self.check(x, w, b, (1, 1), (1, 1), (1, 1))

    def test_bias_is_added_in_float32(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x, w, _ = self.tensors((1, 6, 6, 6), 8, (3, 3), dtype=dtype)
            conv = _reference(x, w, None, (1, 1), (1, 1), (1, 1))
            finfo = torch.finfo(dtype)
            biases = {
                "zero": torch.zeros(8),
                "tiny": torch.full((8,), finfo.tiny),
                "huge": torch.full((8,), finfo.max / 2),
                "near max": torch.full((8,), finfo.max * 0.999),
                "cancels": -conv[0, :, 3, 3].float().cpu(),
                "nan": torch.full((8,), float("nan")),
                "inf": torch.tensor([float("inf"), -float("inf")] * 4),
            }
            for name, bias in biases.items():
                with self.subTest(dtype=dtype, bias=name):
                    self.check(x, w, bias.to("cuda", dtype), (1, 1), (1, 1), (1, 1))

    def test_capability_and_tf32_gate_the_tensor_core_schedule(self):
        from executorch.backends.cuda.triton.kernels.conv.conv import (
            force_conv_candidate,
        )
        from executorch.backends.cuda.triton.kernels.conv.core import implicit_gemm
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            ConvCall,
            ConvParams,
            Schedule,
        )

        self.assertEqual(implicit_gemm.SCHEDULES[0], Schedule("igemm_ffma", 0))
        params = ConvParams((1, 1), (1, 1), (1, 1), (1, 1), False, (0, 0), 1)
        x, w, b = self.tensors((1, 8, 6, 6), 8, (3, 3))
        fp32, bf16 = (
            ConvCall(x, w, b, params),
            ConvCall(x.bfloat16(), w.bfloat16(), b.bfloat16(), params),
        )
        expected = {
            # (export GPU capability, TF32 allowed): (FP32, BF16) run igemm_tc
            ((7, 5), False): (False, False),
            ((7, 5), True): (False, False),
            ((8, 0), False): (False, True),
            ((8, 6), True): (True, True),
        }
        for (capability, tf32), allowed in expected.items():
            with self.subTest(capability=capability, tf32=tf32), mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ), mock.patch.object(implicit_gemm, "conv_allows_tf32", return_value=tf32):
                for call in (fp32, bf16):
                    self.assertIsNone(
                        implicit_gemm.schedule_supports("igemm_ffma", call)
                    )
                got = tuple(
                    implicit_gemm.schedule_supports("igemm_tc", call) is None
                    for call in (fp32, bf16)
                )
                self.assertEqual(got, allowed)
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 0)
        ), mock.patch.object(
            implicit_gemm, "conv_allows_tf32", return_value=False
        ), force_conv_candidate(
            "igemm_tc"
        ), self.assertRaisesRegex(
            ValueError, "TF32"
        ):
            self.convolution(x, w, b, (1, 1), (1, 1), (1, 1))

    def test_timed_schedule_choice(self):
        from executorch.backends.cuda.autotune.launch_params import (
            autotune_launch_params,
            clear_launch_param_cache,
        )

        if not self.tensor_cores:
            self.skipTest("needs tensor cores")
        clear_launch_param_cache()
        x, w, b = self.tensors((2, 16, 20, 20), 32, (3, 3), dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), autotune_launch_params() as stats:
            out = self.convolution(x, w, b, (1, 1), (1, 1), (1, 1))
        stream.synchronize()
        self.assertEqual(stats.measured, 1)
        self.assertEqual(set(stats.picks[0].times_us), {"igemm_ffma", "igemm_tc"})
        torch.testing.assert_close(
            out.double(),
            _reference(x, w, b, (1, 1), (1, 1), (1, 1)),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_64_bit_indexing(self):
        from executorch.backends.cuda.triton.kernels.conv.conv import (
            force_conv_candidate,
        )

        # 2^31 + 2^17 input elements: offsets past the int32 range.
        shape = (1, 2, 32769, 32768)
        if torch.cuda.mem_get_info()[0] < 16 * 2**30:
            self.skipTest("needs 16 GiB of free device memory")
        w = torch.randn(4, 2, 3, 3, device="cuda", dtype=torch.float16)
        for memory_format in (torch.contiguous_format, torch.channels_last):
            x = torch.randn(shape, device="cuda", dtype=torch.float16)
            x = x.contiguous(memory_format=memory_format)
            with self.subTest(memory_format=memory_format), force_conv_candidate(
                "igemm_ffma"
            ):
                out = self.convolution(x, w, None, (1, 1), (1, 1), (1, 1))
                top = _reference(x[:, :, :4], w, None, (1, 1), (1, 1), (0, 1))
                bottom = _reference(x[:, :, -4:], w, None, (1, 1), (0, 1), (1, 1))
                for got, expected in (
                    (out[:, :, :3], top[:, :, :3]),
                    (out[:, :, -3:], bottom),
                ):
                    torch.testing.assert_close(
                        got, expected.to(torch.float16), atol=3e-3, rtol=3e-3
                    )
            del x, out


class Tf32AuthorityTest(unittest.TestCase):
    def test_follows_the_inductor_conv_template_setting(self):
        from executorch.backends.cuda.triton.kernels.conv.core import common
        from torch._dynamo import device_interface

        for allowed in (True, False):
            interface = SimpleNamespace(conv_allow_tf32=lambda allowed=allowed: allowed)
            with self.subTest(allowed=allowed), mock.patch.object(
                device_interface, "get_interface_for_device", return_value=interface
            ):
                self.assertEqual(common.conv_allows_tf32(), allowed)
