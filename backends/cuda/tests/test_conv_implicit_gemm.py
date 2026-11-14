# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""triton::convolution on dense convolutions: every schedule and K order of
the implicit-GEMM template against an FP64 reference, and models lowered
through the CUDA backend."""

import dataclasses
import glob
import itertools
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

_PASS_LOGGER = "executorch.backends.cuda.triton.replacement_pass"
_TOLERANCE = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}
_TF32_TOLERANCE = 5e-3


def _reference(x, w, b, stride, before, after, output_padding=None):
    """The convolution in FP64; transposed if ``output_padding`` is given, as
    the full transposed convolution cropped by ``before`` and ``after``."""
    bias = None if b is None else b.double()
    if output_padding is not None:
        conv = {3: F.conv_transpose1d, 4: F.conv_transpose2d, 5: F.conv_transpose3d}
        full = conv[x.dim()](x.double(), w.double(), bias, stride, 0, output_padding)
        crop = [slice(s, n - a) for s, n, a in zip(before, full.shape[2:], after)]
        return full[(..., *crop)]
    pad = [p for pair in zip(reversed(before), reversed(after)) for p in pair]
    conv = {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}[x.dim()]
    return conv(F.pad(x.double(), pad), w.double(), bias, stride)


def _dense_strides(shape, order):
    """Dense strides of ``shape`` with the dims in ``order``, outermost first."""
    strides, running = [0] * len(shape), 1
    for d in reversed(order):
        strides[d] = running
        running *= shape[d]
    return strides


def _channels_last(t):
    if t.dim() == 3:
        return t.transpose(1, 2).contiguous().transpose(1, 2)
    return t.to(
        memory_format=torch.channels_last if t.dim() == 4 else torch.channels_last_3d
    )


class ImplicitGemmScheduleTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        torch.manual_seed(0)
        self.tensor_cores = torch.cuda.get_device_capability() >= (8, 0)

    def convolution(
        self,
        x,
        w,
        b,
        stride,
        before,
        after,
        output_strides=None,
        output_padding=None,
    ):
        """triton::convolution; transposed if ``output_padding`` is given."""
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
            output_padding is not None,
            [0] * rank if output_padding is None else list(output_padding),
            1,
            output_strides,
        )

    def check(
        self,
        x,
        w,
        b,
        stride,
        before,
        after,
        every_config=False,
        output_strides=None,
        output_padding=None,
    ):
        """Runs every schedule and K order on the call, with the first config of
        each or every config, and checks the result and the layout (the fake
        implementation's, and ``output_strides`` if given); the tensor-core
        schedule runs FP32 operands in TF32. Transposed if ``output_padding`` is
        given."""
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
            output_padding is not None,
            (0,) * rank if output_padding is None else tuple(output_padding),
            1,
        )
        call = ConvCall(
            x, w, b, params, None if output_strides is None else tuple(output_strides)
        )
        tf32 = mock.patch.object(implicit_gemm, "conv_allows_tf32", return_value=True)
        self.assertIsNone(implicit_gemm.supports(call))
        expected = _reference(x, w, b, stride, before, after, output_padding)
        layout = out_shape_and_stride(call)
        with FakeTensorMode() as mode:
            fake = self.convolution(
                *(None if t is None else mode.from_tensor(t) for t in (x, w, b)),
                stride,
                before,
                after,
                output_strides,
                output_padding,
            )
        self.assertEqual((tuple(fake.shape), fake.stride()), layout)
        if output_strides is not None:
            self.assertEqual(layout[1], tuple(output_strides))
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
                    out = self.convolution(
                        x, w, b, stride, before, after, output_strides, output_padding
                    )
                self.assertEqual((tuple(out.shape), out.stride()), layout)
                torch.testing.assert_close(
                    out.double(),
                    expected.to(x.dtype).double(),
                    atol=tolerance * max(1.0, expected.abs().max().item()),
                    rtol=tolerance,
                    equal_nan=True,
                    msg=lambda m, s=schedule, c=config: f"{s} {c}: {m}",
                )

    def tensors(
        self,
        shape,
        out_channels,
        kernel,
        dtype=torch.float32,
        bias=True,
        transposed=False,
    ):
        channels = shape[1]
        x = torch.randn(shape, device="cuda", dtype=dtype)
        fan_in = channels * torch.Size(kernel).numel()
        w_shape = (channels, out_channels) if transposed else (out_channels, channels)
        w = torch.randn(*w_shape, *kernel, device="cuda", dtype=dtype) / fan_in**0.5
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
            # 1-D: k5, k3 s2 and k1 convolutions over short sequences, and an
            # asymmetric pad.
            ((1, 64, 16), 96, (5,), (1,), (2,), (2,)),
            ((2, 32, 8), 32, (3,), (2,), (1,), (1,)),
            ((1, 48, 4), 64, (1,), (1,), (0,), (0,)),
            ((1, 5, 13), 7, (4,), (3,), (2,), (1,)),
            # 3-D: a causal 3x3x3 on an explicitly padded input, a
            # temporal (3,1,1) s(2,1,1) and a 1x1x1.
            ((1, 24, 3, 10, 12), 16, (3, 3, 3), (1, 1, 1), (0,) * 3, (0,) * 3),
            ((2, 32, 9, 4, 5), 32, (3, 1, 1), (2, 1, 1), (0,) * 3, (0,) * 3),
            ((1, 16, 2, 5, 6), 24, (1, 1, 1), (1, 1, 1), (0,) * 3, (0,) * 3),
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

    def test_output_layouts(self):
        # Every schedule writes the output strides the graph recorded, whatever
        # the input layout; dims of size 1 keep their recorded strides.
        for shape, c_out, kernel, stride, padding, orders in (
            (
                (2, 3, 30, 26),
                40,
                (7, 7),
                (2, 2),
                (3, 3),
                [(0, 1, 2, 3), (0, 2, 3, 1), (1, 0, 2, 3), (2, 3, 1, 0)],
            ),
            ((1, 3, 56, 14), 48, (14, 14), (14, 14), (0, 0), [(0, 1, 2, 3)]),
            (
                (2, 3, 2, 28, 32),
                32,
                (2, 16, 16),
                (2, 16, 16),
                (0, 0, 0),
                [(0, 1, 2, 3, 4), (0, 2, 3, 4, 1), (1, 0, 2, 3, 4)],
            ),
            ((2, 16, 12), 24, (5,), (1,), (2,), [(0, 1, 2), (0, 2, 1), (2, 1, 0)]),
        ):
            x, w, b = self.tensors(shape, c_out, kernel)
            out_shape = (
                shape[0],
                c_out,
                *(
                    (e + 2 * p - k) // s + 1
                    for e, k, s, p in zip(shape[2:], kernel, stride, padding)
                ),
            )
            inputs = (x, _channels_last(x))
            for x, order in itertools.product(inputs, orders):
                strides = [
                    12345 if n == 1 else s
                    for n, s in zip(out_shape, _dense_strides(out_shape, order))
                ]
                with self.subTest(shape=shape, input=x.stride(), strides=strides):
                    self.check(
                        x, w, b, stride, padding, padding, output_strides=strides
                    )

    def test_transposed(self):
        geometries = (
            # input, C_out, kernel, stride, before, after, output_padding: a
            # kernel-4 stride-2 1-D upsample, then edge cases; an asymmetric crop
            # only the canonical op can express; 16+ channels walk K tap-major.
            ((1, 32, 8), 16, (4,), (2,), (1,), (1,), (0,)),
            ((2, 8, 5, 6), 12, (3, 3), (2, 2), (1, 1), (1, 1), (1, 1)),
            (
                (1, 6, 3, 4, 5),
                10,
                (2, 3, 3),
                (2, 1, 3),
                (0, 1, 1),
                (0, 1, 1),
                (1, 0, 2),
            ),
            ((1, 20, 7), 9, (5,), (3,), (3,), (1,), (0,)),
            ((1, 4, 9), 4, (3,), (1,), (1,), (1,), (0,)),
            ((2, 24, 6, 7), 16, (4, 4), (2, 2), (1, 1), (1, 1), (0, 0)),
        )
        cases = itertools.product(
            geometries,
            (torch.float32, torch.float16, torch.bfloat16),
            ("contiguous", "channels_last"),
        )
        for geometry, dtype, layout in cases:
            shape, c_out, kernel, stride, before, after, output_padding = geometry
            x, w, b = self.tensors(shape, c_out, kernel, dtype=dtype, transposed=True)
            if layout == "channels_last":
                x = _channels_last(x)
            with self.subTest(shape=shape, kernel=kernel, dtype=dtype, layout=layout):
                self.check(
                    x, w, b, stride, before, after, output_padding=output_padding
                )
        x, w, b = self.tensors((1, 32, 8), 16, (4,), transposed=True)
        with self.subTest(every_config=True):
            self.check(x, w, b, (2,), (1,), (1,), True, output_padding=(0,))
        with self.subTest(output_strides=(256, 1, 16)):
            self.check(
                x,
                w,
                b,
                (2,),
                (1,),
                (1,),
                output_strides=(256, 1, 16),
                output_padding=(0,),
            )

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


class _KernelEqualsStride(nn.Module):
    """A 2-D and a 3-D convolution with kernel == stride."""

    def __init__(self, dtype):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 64, 14, stride=14, padding="valid", dtype=dtype)
        self.conv3d = nn.Conv3d(
            3, 64, (2, 14, 14), stride=(2, 14, 14), bias=False, dtype=dtype
        )

    def forward(self, x, y):
        return self.conv2d(x), self.conv3d(y.view(-1, 3, 2, 14, 14))


class _ThreeChannelBlock(nn.Module):
    """3-input-channel 7x7 stride-2 convolutions, padded and unpadded, and
    the 1x1 convolutions left to Inductor."""

    def __init__(self, dtype):
        super().__init__()
        self.padded = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False, dtype=dtype)
        self.unpadded = nn.Conv2d(3, 32, 7, stride=2, dtype=dtype)
        self.down = nn.Conv2d(32, 64, 1, stride=2, bias=False, dtype=dtype)
        self.proj = nn.Conv2d(64, 16, 1, dtype=dtype)

    def forward(self, x):
        return self.proj(self.down(F.relu(self.padded(x)))), self.unpadded(x)


class _Conv1dBlock(nn.Module):
    """1-D convolutions: k5 p2, a k1 residual and a k3 s2 p1 downsample; and a
    "same"-padded even kernel."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(16, 32, 5, padding=2)
        self.res = nn.Conv1d(16, 32, 1)
        self.same = nn.Conv1d(32, 32, 4, padding="same")
        self.down = nn.Conv1d(32, 32, 3, stride=2, padding=1)

    def forward(self, x):
        return self.down(self.same(F.mish(self.conv(x)) + self.res(x)))


class _ExplicitPadConv3dBlock(nn.Module):
    """3-D convolutions as models write them: a causal
    pad and a 3x3x3, a 1x1x1 shortcut, and a (3,1,1) s(2,1,1) time
    convolution."""

    def __init__(self, dtype):
        super().__init__()
        self.causal = nn.Conv3d(12, 16, 3, dtype=dtype)
        self.shortcut = nn.Conv3d(12, 16, 1, dtype=dtype)
        self.time = nn.Conv3d(16, 16, (3, 1, 1), stride=(2, 1, 1), dtype=dtype)

    def forward(self, x):
        y = self.causal(F.pad(x, (1, 1, 1, 1, 2, 0))) + self.shortcut(x)
        return self.time(y)


class _Upsample(nn.Module):
    """A transposed 1-D upsample, and a transposed
    3-D one."""

    def __init__(self):
        super().__init__()
        self.up1d = nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1)
        self.up3d = nn.ConvTranspose3d(8, 4, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x, y):
        return self.up1d(x), self.up3d(y)


class ImplicitGemmLoweringTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")

    def lower_and_run(
        self, model, inputs, routed, tolerance, route="optional", to_backend=False
    ):
        """Lowers ``model`` with to_edge_transform_and_lower, or with to_edge
        and to_backend, checks that ``routed`` convolutions take the template
        on ``route``, and runs the program against eager (outside fbcode)."""
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import (
            EdgeCompileConfig,
            to_edge,
            to_edge_transform_and_lower,
        )

        with torch.no_grad():
            program = torch.export.export(model, inputs, strict=True)
        partitioner = CudaPartitioner(
            [CudaBackend.generate_method_name_compile_spec("forward")]
        )
        config = EdgeCompileConfig(_check_ir_validity=False)
        with self.assertLogs(_PASS_LOGGER, "INFO") as logs:
            if to_backend:
                edge = to_edge(program, compile_config=config).to_backend(partitioner)
            else:
                edge = to_edge_transform_and_lower(
                    program, partitioner=[partitioner], compile_config=config
                )
        output = "\n".join(logs.output)
        self.assertEqual(output.count(f"({route}, implicit_gemm template)"), routed)
        self.assertIn(f"Replaced {routed} nodes with Triton kernels", output)

        from executorch.export.utils import is_fbcode

        if is_fbcode():
            # Running the program is skipped in fbcode.
            return output
        from executorch.runtime import Runtime

        et_program = edge.to_executorch()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pte")
            with open(path, "wb") as f:
                et_program.write_to_file(f)
            et_program.write_tensor_data_to_file(directory)
            data_path = glob.glob(os.path.join(directory, "*.ptd"))[0]
            method = Runtime.get().load_program(path, data_path=data_path)
            outputs = method.load_method("forward").execute([i.cpu() for i in inputs])
        with torch.no_grad(), torch.backends.cudnn.flags(
            enabled=True, allow_tf32=False
        ):
            expected = model(*inputs)
        if isinstance(expected, torch.Tensor):
            expected = (expected,)
        for got, want in zip(outputs, expected, strict=True):
            torch.testing.assert_close(
                got.float(), want.float().cpu(), atol=tolerance, rtol=tolerance
            )
        return output

    def fp32_tolerance(self):
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            conv_allows_tf32,
        )

        return _TF32_TOLERANCE if conv_allows_tf32() else 1e-4

    def test_kernel_equals_stride_lowers_and_runs(self):
        for dtype, tolerance in (
            (torch.float32, self.fp32_tolerance()),
            (torch.bfloat16, 3e-2),
        ):
            torch.manual_seed(0)
            model = _KernelEqualsStride(dtype).to("cuda").eval()
            inputs = (
                torch.randn(1, 3, 224, 224, device="cuda", dtype=dtype),
                torch.randn(256, 3 * 2 * 14 * 14, device="cuda", dtype=dtype),
            )
            with self.subTest(dtype=dtype):
                self.lower_and_run(model, inputs, routed=2, tolerance=tolerance)

    def test_three_channel_block_lowers_and_runs(self):
        for dtype, tolerance in (
            (torch.float32, self.fp32_tolerance()),
            (torch.bfloat16, 3e-2),
        ):
            torch.manual_seed(0)
            model = _ThreeChannelBlock(dtype).to("cuda").eval()
            inputs = (torch.randn(2, 3, 84, 84, device="cuda", dtype=dtype),)
            with self.subTest(dtype=dtype):
                output = self.lower_and_run(
                    model, inputs, routed=2, tolerance=tolerance
                )
                # The 1x1 convolutions keep Inductor's lowering.
                self.assertEqual(output.count("Skipping conv replacement"), 2)

    def test_conv1d_block_lowers_and_runs(self):
        torch.manual_seed(0)
        model = _Conv1dBlock().to("cuda").eval()
        inputs = (torch.randn(1, 16, 16, device="cuda"),)
        # to_edge keeps aten.convolution, which Inductor cannot lower in 1-D:
        # the template is required. to_edge_transform_and_lower keeps
        # aten.conv1d, which the backend lowers as a 2-D convolution, except
        # for the "same" overload.
        self.lower_and_run(model, inputs, 4, 1e-4, route="required", to_backend=True)
        output = self.lower_and_run(
            model, inputs, 1, self.fp32_tolerance(), route="required"
        )
        self.assertEqual(output.count("Skipping conv replacement"), 3)

    def test_conv3d_block_lowers_and_runs(self):
        from executorch.backends.cuda.triton.kernels.conv import routing

        torch.manual_seed(0)
        model = _ExplicitPadConv3dBlock(torch.bfloat16).to("cuda").eval()
        inputs = (torch.randn(1, 12, 5, 16, 16, device="cuda", dtype=torch.bfloat16),)
        # Outside the benchmarked range: Inductor lowers all three.
        self.lower_and_run(model, inputs, 0, 3e-2)
        # Every one is correct through the template; the pad stays separate.
        take_all = tuple(
            dataclasses.replace(t, worth_routing=lambda call: None)
            for t in routing.TEMPLATES
        )
        with mock.patch.object(routing, "TEMPLATES", take_all):
            self.lower_and_run(model, inputs, 3, 3e-2)

    def test_transposed_lowers_and_runs(self):
        torch.manual_seed(0)
        model = _Upsample().to("cuda").eval()
        inputs = (
            torch.randn(1, 32, 8, device="cuda"),
            torch.randn(1, 8, 3, 4, 5, device="cuda"),
        )
        # Inductor has no transposed template on CUDA. to_edge keeps both as
        # aten.convolution; to_edge_transform_and_lower keeps conv_transpose1d,
        # which the backend rewrites as a convolution, and conv_transpose3d.
        self.lower_and_run(model, inputs, 2, 1e-4, route="required", to_backend=True)
        output = self.lower_and_run(
            model, inputs, 1, self.fp32_tolerance(), route="required"
        )
        self.assertIn("transposed convolution is not benchmarked faster", output)
