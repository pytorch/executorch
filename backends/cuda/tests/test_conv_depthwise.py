# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""triton::convolution on 2-D depthwise convolutions and grouped convolutions
with few channels per group: every schedule of the direct-stencil template
against an FP64 reference, and models lowered through the CUDA backend."""

import glob
import itertools
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

_PASS_LOGGER = "executorch.backends.cuda.triton.replacement_pass"
_TOLERANCE = {torch.float32: 2e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}


def _dense_strides(shape, order):
    """Dense strides of ``shape`` with the dims in ``order``, outermost first."""
    strides, running = [0] * len(shape), 1
    for d in reversed(order):
        strides[d] = running
        running *= shape[d]
    return strides


def _reference(x, w, b, stride, before, after):
    padded = F.pad(x.double(), [before[1], after[1], before[0], after[0]])
    bias = None if b is None else b.double()
    return F.conv2d(padded, w.double(), bias, stride, 0, 1, x.shape[1] // w.shape[1])


class DepthwiseScheduleTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        torch.manual_seed(0)

    def convolution(self, x, w, b, stride, before, after, output_strides=None):
        from executorch.backends.cuda.triton.kernels.conv.conv import convolution

        return convolution(
            x,
            w,
            b,
            list(stride),
            list(before),
            list(after),
            [1, 1],
            False,
            [0, 0],
            x.shape[1] // w.shape[1],
            output_strides,
        )

    def check(
        self, x, w, b, stride, before, after, every_config=False, output_strides=None
    ):
        """Runs every schedule that supports the call, with its first config or
        every config, and checks the result and the layout (the fake
        implementation's, and ``output_strides`` if given)."""
        from executorch.backends.cuda.triton.kernels.conv.conv import (
            force_conv_candidate,
        )
        from executorch.backends.cuda.triton.kernels.conv.core import direct
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            ConvCall,
            ConvParams,
            out_shape_and_stride,
        )

        params = ConvParams(
            tuple(stride),
            tuple(before),
            tuple(after),
            (1, 1),
            False,
            (0, 0),
            x.shape[1] // w.shape[1],
        )
        call = ConvCall(
            x, w, b, params, None if output_strides is None else tuple(output_strides)
        )
        self.assertIsNone(direct.supports(call))
        expected = _reference(x, w, b, stride, before, after).to(x.dtype)
        layout = out_shape_and_stride(call)
        with FakeTensorMode() as mode:
            fake = self.convolution(
                *(None if t is None else mode.from_tensor(t) for t in (x, w, b)),
                stride,
                before,
                after,
                output_strides,
            )
        self.assertEqual((tuple(fake.shape), fake.stride()), layout)
        if output_strides is not None:
            self.assertEqual(layout[1], tuple(output_strides))
        ran = []
        for schedule, all_configs in (
            ("dw_flat", direct.FLAT_CONFIGS),
            ("dw_row8", direct.ROW8_CONFIGS),
            ("dw_mult", direct.MULT_CONFIGS),
            ("grp_plane", direct.PLANE_CONFIGS),
            ("grp_reduce", direct.REDUCE_CONFIGS),
        ):
            if direct.schedule_supports(schedule, call) is not None:
                with force_conv_candidate(schedule), self.assertRaises(ValueError):
                    self.convolution(x, w, b, stride, before, after, output_strides)
                continue
            for config in all_configs if every_config else all_configs[:1]:
                with force_conv_candidate(schedule, config):
                    out = self.convolution(
                        x, w, b, stride, before, after, output_strides
                    )
                self.assertEqual((tuple(out.shape), out.stride()), layout)
                tolerance = _TOLERANCE[x.dtype]
                torch.testing.assert_close(
                    out.double(),
                    expected.double(),
                    atol=tolerance * max(1.0, expected.abs().max().item()),
                    rtol=tolerance,
                    equal_nan=True,
                    msg=lambda m, s=schedule, c=config: f"{s} {c}: {m}",
                )
                ran.append(schedule)
        self.assertIn("dw_flat", ran)
        return ran

    def tensors(
        self,
        n,
        c,
        h,
        w,
        kh,
        kw,
        dtype=torch.float32,
        bias=True,
        group_channels=1,
        multiplier=1,
    ):
        """Input, weight and bias of a convolution with ``group_channels`` input
        and ``multiplier`` output channels per group."""
        out_channels = c // group_channels * multiplier
        x = torch.randn(n, c, h, w, device="cuda", dtype=dtype)
        weight = torch.randn(
            out_channels, group_channels, kh, kw, device="cuda", dtype=dtype
        ) / (group_channels * kh * kw) ** 0.5
        b = torch.randn(out_channels, device="cuda", dtype=dtype) if bias else None
        return x, weight, b

    def test_every_config_of_every_schedule(self):
        for layout in ("nchw", "channels_last"):
            x, w, b = self.tensors(2, 40, 13, 19, 3, 3)
            if layout == "channels_last":
                x = x.to(memory_format=torch.channels_last)
            with self.subTest(layout=layout):
                ran = self.check(x, w, b, (1, 1), (1, 1), (1, 1), every_config=True)
                self.assertEqual("dw_row8" in ran, layout == "channels_last")

    def test_geometry_dtypes_and_layouts(self):
        geometries = (
            # (n, c, h, w), (kh, kw), stride, before, after
            ((2, 48, 17, 23), (3, 3), (1, 1), (1, 1), (1, 1)),
            ((1, 7, 9, 31), (5, 3), (2, 1), (2, 1), (2, 1)),
            ((3, 33, 11, 10), (7, 7), (1, 2), (3, 3), (3, 3)),
            ((1, 16, 8, 9), (9, 9), (1, 1), (4, 4), (4, 4)),
            ((2, 5, 6, 7), (1, 9), (3, 1), (0, 4), (0, 4)),
            ((1, 24, 10, 12), (4, 4), (1, 1), (1, 1), (2, 2)),
            ((1, 12, 9, 9), (3, 3), (2, 2), (0, 0), (0, 0)),
            ((1, 3, 1, 1), (1, 1), (1, 1), (0, 0), (0, 0)),
            ((1, 64, 2, 2), (3, 3), (2, 2), (1, 0), (0, 1)),
        )
        cases = [
            (geometry, dtype, layout, i % 2 == 0)
            for i, geometry in enumerate(geometries)
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
            for layout in ("nchw", "channels_last")
        ]
        cases += [
            (geometry, torch.float32, "weight_cl", True) for geometry in geometries
        ]
        for (shape, kernel, stride, before, after), dtype, layout, bias in cases:
            x, w, b = self.tensors(*shape, *kernel, dtype=dtype, bias=bias)
            if layout == "channels_last":
                x = x.to(memory_format=torch.channels_last)
            if layout == "weight_cl":
                w = w.to(memory_format=torch.channels_last)
            with self.subTest(
                shape=shape, kernel=kernel, stride=stride, dtype=dtype, layout=layout
            ):
                self.check(x, w, b, stride, before, after)

    def test_grouped_geometry_dtypes_and_layouts(self):
        geometries = (
            # (n, c, h, w), (kh, kw), stride, before, after, (group channels,
            # multiplier)
            ((2, 16, 13, 17), (3, 3), (1, 1), (1, 1), (1, 1), (1, 2)),
            ((1, 12, 11, 9), (5, 5), (2, 2), (2, 2), (2, 2), (1, 4)),
            ((2, 16, 10, 12), (3, 3), (1, 1), (1, 1), (1, 1), (2, 1)),
            ((1, 24, 15, 13), (7, 7), (1, 1), (3, 3), (3, 3), (3, 2)),
            ((1, 32, 9, 11), (3, 3), (2, 2), (1, 1), (1, 1), (4, 1)),
            ((2, 32, 12, 10), (5, 5), (1, 2), (2, 2), (2, 2), (8, 1)),
            ((1, 16, 14, 14), (7, 7), (2, 2), (3, 3), (3, 3), (8, 2)),
            ((1, 16, 9, 8), (4, 4), (1, 1), (1, 1), (2, 2), (4, 4)),
        )
        cases = [
            (geometry, dtype, layout, i % 2 == 0)
            for i, geometry in enumerate(geometries)
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
            for layout in ("nchw", "channels_last")
        ]
        cases += [
            (geometry, torch.float32, "weight_cl", True) for geometry in geometries
        ]
        for geometry, dtype, layout, bias in cases:
            shape, kernel, stride, before, after, (group_channels, multiplier) = (
                geometry
            )
            x, w, b = self.tensors(
                *shape,
                *kernel,
                dtype=dtype,
                bias=bias,
                group_channels=group_channels,
                multiplier=multiplier,
            )
            if layout == "channels_last":
                x = x.to(memory_format=torch.channels_last)
            if layout == "weight_cl":
                w = w.to(memory_format=torch.channels_last)
            with self.subTest(
                shape=shape,
                kernel=kernel,
                group_channels=group_channels,
                multiplier=multiplier,
                dtype=dtype,
                layout=layout,
            ):
                self.check(x, w, b, stride, before, after)

    def test_every_config_of_every_schedule_grouped(self):
        for (group_channels, multiplier), layout in (
            ((3, 2), "nchw"),
            ((3, 2), "channels_last"),
            ((1, 3), "nchw"),
            ((1, 3), "channels_last"),
        ):
            x, w, b = self.tensors(
                2,
                24,
                11,
                13,
                3,
                3,
                group_channels=group_channels,
                multiplier=multiplier,
            )
            if layout == "channels_last":
                x = x.to(memory_format=torch.channels_last)
            with self.subTest(group_channels=group_channels, layout=layout):
                ran = self.check(x, w, b, (1, 1), (1, 1), (1, 1), every_config=True)
                self.assertEqual("dw_row8" in ran, layout == "channels_last")
                self.assertEqual("dw_mult" in ran, group_channels == 1)
                self.assertIn("grp_plane", ran)
                self.assertIn("grp_reduce", ran)

    def test_high_channel_multipliers(self):
        from executorch.backends.cuda.autotune.launch_params import (
            autotune_launch_params,
            clear_launch_param_cache,
        )
        from executorch.backends.cuda.triton.kernels.conv.core import direct

        x, w, b = self.tensors(1, 3, 9, 11, 3, 3, multiplier=direct.MAX_MULT_MULTIPLIER)
        ran = self.check(x, w, b, (1, 1), (1, 1), (1, 1), every_config=True)
        self.assertIn("dw_mult", ran)
        # 4096 output channels from two input channels: dw_mult declines the
        # multiplier and the timed choice runs the other schedules.
        x, w, b = self.tensors(1, 2, 12, 12, 3, 3, multiplier=2048)
        self.assertNotIn("dw_mult", self.check(x, w, b, (1, 1), (1, 1), (1, 1)))
        clear_launch_param_cache()
        with autotune_launch_params() as stats:
            out = self.convolution(x, w, b, (1, 1), (1, 1), (1, 1))
        self.assertEqual(stats.measured, 1)
        self.assertEqual(
            set(stats.picks[0].times_us), {"dw_flat", "grp_plane", "grp_reduce"}
        )
        torch.testing.assert_close(
            out.double(),
            _reference(x, w, b, (1, 1), (1, 1), (1, 1)),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_unsupported_groupings(self):
        from executorch.backends.cuda.triton.kernels.conv.core import direct
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            ConvCall,
            ConvParams,
        )

        x = torch.randn(1, 18, 8, 8, device="cuda")
        for groups, out_channels, reason in (
            (1, 4, "not a depthwise convolution"),
            (2, 4, "not a depthwise convolution"),
        ):
            w = torch.randn(out_channels, 18 // groups, 3, 3, device="cuda")
            call = ConvCall(
                x,
                w,
                None,
                ConvParams((1, 1), (1, 1), (1, 1), (1, 1), False, (0, 0), groups),
            )
            with self.subTest(groups=groups):
                self.assertIn(reason, direct.supports(call))
    def test_output_layouts(self):
        # Every schedule writes the output strides the graph recorded, whatever
        # the input layout; dims of size 1 keep their recorded strides.
        for shape, kernel, stride, padding in (
            ((2, 24, 11, 13), (3, 3), (1, 1), (1, 1)),
            ((1, 16, 9, 1), (5, 1), (2, 1), (2, 0)),
        ):
            x, w, b = self.tensors(*shape, *kernel)
            out_shape = (
                shape[0],
                shape[1],
                (shape[2] + 2 * padding[0] - kernel[0]) // stride[0] + 1,
                (shape[3] + 2 * padding[1] - kernel[1]) // stride[1] + 1,
            )
            orders = [(0, 1, 2, 3), (0, 2, 3, 1), (1, 0, 2, 3), (2, 3, 1, 0)]
            inputs = (x, x.to(memory_format=torch.channels_last))
            for x, order in itertools.product(inputs, orders):
                strides = [
                    12345 if n == 1 else s
                    for n, s in zip(out_shape, _dense_strides(out_shape, order))
                ]
                with self.subTest(shape=shape, input=x.stride(), strides=strides):
                    self.check(
                        x, w, b, stride, padding, padding, output_strides=strides
                    )

    def test_sliced_inputs_with_storage_offset(self):
        base = torch.randn(2, 20, 15, 18, device="cuda")
        _, w, b = self.tensors(1, 16, 1, 1, 3, 3)
        for x in (
            base[:, 2:18, 1:-1, 2:],
            base.to(memory_format=torch.channels_last)[:, 4:, 3:, :-1],
            base[:, 2:18, :, ::2],
        ):
            with self.subTest(stride=x.stride(), offset=x.storage_offset()):
                self.assertGreater(x.storage_offset(), 0)
                self.check(x, w, b, (1, 1), (1, 1), (1, 1))

    def test_bias_is_added_in_float32(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x, w, _ = self.tensors(1, 8, 6, 6, 3, 3, dtype=dtype)
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

    def test_every_gpu_runs_the_base_schedule(self):
        from executorch.backends.cuda.triton.kernels.conv.core import direct
        from executorch.backends.cuda.triton.kernels.conv.core.common import (
            ConvCall,
            ConvParams,
        )

        self.assertEqual(direct.SCHEDULES[0], direct.Schedule("dw_flat", 0))
        x, w, b = self.tensors(1, 8, 6, 6, 3, 3)
        call = ConvCall(
            x, w, b, ConvParams((1, 1), (1, 1), (1, 1), (1, 1), False, (0, 0), 8)
        )
        for capability in ((6, 0), (7, 5), (8, 0), (9, 0)):
            with mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                self.assertIsNone(direct.schedule_supports("dw_flat", call))

    def test_timed_schedule_choice(self):
        from executorch.backends.cuda.autotune.launch_params import (
            autotune_launch_params,
            clear_launch_param_cache,
        )

        clear_launch_param_cache()
        x, w, b = self.tensors(2, 64, 28, 28, 3, 3)
        x = x.to(memory_format=torch.channels_last)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), autotune_launch_params() as stats:
            out = self.convolution(x, w, b, (1, 1), (1, 1), (1, 1))
        stream.synchronize()
        self.assertEqual(stats.measured, 1)
        self.assertEqual(set(stats.picks[0].times_us), {"dw_flat", "dw_row8"})
        torch.testing.assert_close(
            out.double(),
            _reference(x, w, b, (1, 1), (1, 1), (1, 1)),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_64_bit_indexing(self):
        from executorch.backends.cuda.triton.kernels.conv.conv import (
            force_conv_candidate,
        )

        # 2^31 + 2^17 elements: offsets past the int32 range.
        shape = (1, 2, 32769, 32768)
        if torch.cuda.mem_get_info()[0] < 12 * 2**30:
            self.skipTest("needs 12 GiB of free device memory")
        w = torch.randn(2, 1, 3, 3, device="cuda", dtype=torch.float16)
        for schedule, memory_format in (
            ("dw_flat", torch.contiguous_format),
            ("dw_row8", torch.channels_last),
        ):
            x = torch.randn(shape, device="cuda", dtype=torch.float16)
            x = x.contiguous(memory_format=memory_format)
            with self.subTest(schedule=schedule), force_conv_candidate(schedule):
                out = self.convolution(x, w, None, (1, 1), (1, 1), (1, 1))
                top = _reference(x[:, :, :4], w, None, (1, 1), (1, 1), (0, 1))
                bottom = _reference(x[:, :, -4:], w, None, (1, 1), (0, 1), (1, 1))
                for got, expected in (
                    (out[:, :, :3], top[:, :, :3]),
                    (out[:, :, -3:], bottom),
                ):
                    torch.testing.assert_close(
                        got, expected.to(torch.float16), atol=2e-3, rtol=2e-3
                    )
            del x, out


class _DepthwiseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw3 = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.dw7 = nn.Conv2d(32, 32, 7, padding="same", groups=32, bias=False)
        self.pw = nn.Conv2d(32, 16, 1)

    def forward(self, x):
        return self.pw(F.gelu(self.dw7(self.dw3(x))))


class _GroupedBlock(nn.Module):
    """Grouped convolutions with 2 and 4 input channels per group and a
    depthwise convolution with channel multiplier 2."""

    def __init__(self):
        super().__init__()
        self.g2 = nn.Conv2d(32, 32, 3, padding=1, groups=16)
        self.dw_m2 = nn.Conv2d(32, 64, 5, padding=2, groups=32, bias=False)
        self.g4 = nn.Conv2d(64, 16, 7, stride=2, padding=3, groups=16)

    def forward(self, x):
        return self.g4(F.gelu(self.dw_m2(self.g2(x))))


class DepthwiseLoweringTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")

    def test_depthwise_block_lowers_and_runs(self):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        torch.manual_seed(0)
        model = _DepthwiseBlock().cuda().eval()
        inputs = (torch.randn(2, 32, 20, 24, device="cuda"),)
        with torch.no_grad():
            program = torch.export.export(model, inputs, strict=True)
        with self.assertLogs(_PASS_LOGGER, "INFO") as logs:
            edge = to_edge_transform_and_lower(
                program,
                partitioner=[
                    CudaPartitioner(
                        [CudaBackend.generate_method_name_compile_spec("forward")]
                    )
                ],
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            )
        output = "\n".join(logs.output)
        self.assertEqual(output.count("(optional, direct template)"), 2)
        self.assertIn("Skipping conv replacement", output)
        self.assertIn("Replaced 2 nodes with Triton kernels", output)

        from executorch.export.utils import is_fbcode

        if is_fbcode():
            # Running the program is skipped in fbcode.
            return
        from executorch.runtime import Runtime

        et_program = edge.to_executorch()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.pte")
            with open(path, "wb") as f:
                et_program.write_to_file(f)
            et_program.write_tensor_data_to_file(directory)
            data_path = glob.glob(os.path.join(directory, "*.ptd"))[0]
            method = Runtime.get().load_program(path, data_path=data_path)
            (output,) = method.load_method("forward").execute([inputs[0].cpu()])
        with torch.no_grad():
            expected = model(*inputs)
        torch.testing.assert_close(output, expected.cpu(), atol=1e-4, rtol=1e-4)

    def test_grouped_block_lowers_and_runs_with_both_apis(self):
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
        from executorch.exir import (
            EdgeCompileConfig,
            to_edge,
            to_edge_transform_and_lower,
        )
        from executorch.export.utils import is_fbcode

        torch.manual_seed(0)
        model = _GroupedBlock().cuda().eval()
        inputs = (torch.randn(2, 32, 20, 24, device="cuda"),)
        with torch.no_grad():
            program = torch.export.export(model, inputs, strict=True)
            expected = model(*inputs)
        config = EdgeCompileConfig(_check_ir_validity=False)
        for api in ("to_edge_transform_and_lower", "to_edge_to_backend"):
            with self.subTest(api=api):
                partitioner = CudaPartitioner(
                    [CudaBackend.generate_method_name_compile_spec("forward")]
                )
                with self.assertLogs(_PASS_LOGGER, "INFO") as logs:
                    if api == "to_edge_transform_and_lower":
                        edge = to_edge_transform_and_lower(
                            program, partitioner=[partitioner], compile_config=config
                        )
                    else:
                        edge = to_edge(program, compile_config=config).to_backend(
                            partitioner
                        )
                output = "\n".join(logs.output)
                self.assertEqual(output.count("(optional, direct template)"), 3)
                self.assertIn("Replaced 3 nodes with Triton kernels", output)
                if is_fbcode():
                    # Running the program is skipped in fbcode.
                    continue
                from executorch.runtime import Runtime

                et_program = edge.to_executorch()
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "model.pte")
                    with open(path, "wb") as f:
                        et_program.write_to_file(f)
                    et_program.write_tensor_data_to_file(directory)
                    data_path = glob.glob(os.path.join(directory, "*.ptd"))[0]
                    method = Runtime.get().load_program(path, data_path=data_path)
                    (out,) = method.load_method("forward").execute([inputs[0].cpu()])
                torch.testing.assert_close(out, expected.cpu(), atol=1e-4, rtol=1e-4)
