# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Binding of every conv overload to triton::convolution's arguments, and the
output layout helper, checked against ATen's own fake-tensor convolution.

No GPU needed: the tensors are fake CUDA tensors.
"""

import itertools
import unittest

import torch
from executorch.backends.cuda.triton.kernels.conv.core.common import (
    bind_conv,
    ConvBindingError,
    ConvCall,
    ConvParams,
    out_shape_and_stride,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode

aten = torch.ops.aten


def _channels_last(t):
    """``t`` with channels-last strides (for 3-D tensors: channel stride 1)."""
    if t.dim() == 3:
        return t.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    return t.contiguous(
        memory_format=torch.channels_last if t.dim() == 4 else torch.channels_last_3d
    )


class ConvBindingTest(unittest.TestCase):
    def setUp(self):
        self.mode = FakeTensorMode()

    def empty(self, *shape, dtype=torch.float32):
        with self.mode:
            return torch.empty(*shape, dtype=dtype, device="cuda")

    def assertBinds(self, op, args, kwargs, params):
        """``op`` binds to ``params`` and its output matches ATen's."""
        input, weight, bias, bound = bind_conv(op, args, kwargs)
        self.assertIs(input, args[0])
        self.assertEqual(bound, params)
        with self.mode:
            expected = op(*args, **kwargs)
        call = ConvCall(input, weight, bias, bound)
        self.assertEqual(
            out_shape_and_stride(call),
            (tuple(expected.shape), tuple(expected.stride())),
        )
        edge_op = getattr(
            getattr(exir_ops.edge.aten, op._schema.name.removeprefix("aten::")),
            op._overloadname,
        )
        self.assertEqual(bind_conv(edge_op, args, kwargs)[3], params)

    def test_omitted_defaults(self):
        for rank, op in enumerate((aten.conv1d, aten.conv2d, aten.conv3d), 1):
            x = self.empty(2, 4, *([9] * rank))
            w = self.empty(6, 4, *([3] * rank))
            ones, zeros = (1,) * rank, (0,) * rank
            with self.subTest(rank=rank):
                self.assertBinds(
                    op.default,
                    (x, w),
                    {},
                    ConvParams(ones, zeros, zeros, ones, False, zeros, 1),
                )

    def test_keywords_and_single_values(self):
        x, w, b = self.empty(1, 4, 11, 13), self.empty(8, 2, 3, 5), self.empty(8)
        self.assertBinds(
            aten.conv2d.default,
            (x, w),
            {"bias": b, "stride": [2], "padding": [1, 2], "groups": 2},
            ConvParams((2, 2), (1, 2), (1, 2), (1, 1), False, (0, 0), 2),
        )
        self.assertBinds(
            aten.conv2d.default,
            (x, self.empty(8, 4, 3, 5), None, [1, 2]),
            {"dilation": [2]},
            ConvParams((1, 2), (0, 0), (0, 0), (2, 2), False, (0, 0), 1),
        )

    def test_padding_modes(self):
        x = self.empty(1, 3, 10, 12)
        cases = (
            ((4, 4), [1, 1], "same", (1, 1), (2, 2)),
            ((3, 4), [2, 1], "same", (2, 1), (2, 2)),
            ((2, 5), [3, 1], "same", (1, 2), (2, 2)),
            ((3, 3), [1, 1], "valid", (0, 0), (0, 0)),
        )
        for kernel, dilation, mode, before, after in cases:
            w = self.empty(5, 3, *kernel)
            with self.subTest(kernel=kernel, dilation=dilation, mode=mode):
                self.assertBinds(
                    aten.conv2d.padding,
                    (x, w, None, [1, 1], mode, dilation),
                    {},
                    ConvParams(
                        (1, 1), before, after, tuple(dilation), False, (0, 0), 1
                    ),
                )
        self.assertBinds(
            aten.conv1d.padding,
            (self.empty(2, 3, 7), self.empty(3, 3, 2)),
            {"padding": "same"},
            ConvParams((1,), (0,), (1,), (1,), False, (0,), 1),
        )
        self.assertBinds(
            aten.conv3d.padding,
            (self.empty(1, 2, 5, 6, 7), self.empty(4, 1, 2, 3, 4)),
            {"padding": "same", "groups": 2},
            ConvParams((1,) * 3, (0, 1, 1), (1, 1, 2), (1,) * 3, False, (0,) * 3, 2),
        )

    def test_same_padding_needs_unit_stride(self):
        x, w = self.empty(1, 3, 10, 12), self.empty(5, 3, 3, 3)
        with self.assertRaisesRegex(ConvBindingError, "stride 1"):
            bind_conv(aten.conv2d.padding, (x, w, None, [2, 2], "same"), {})

    def test_transposed_argument_order(self):
        # conv_transpose takes output_padding before groups and dilation, and an
        # [in, out / groups, k] weight.
        self.assertBinds(
            aten.conv_transpose1d.default,
            (self.empty(2, 4, 9), self.empty(4, 3, 4), None, [2], [1], [1], 2, [1]),
            {},
            ConvParams((2,), (1,), (1,), (1,), True, (1,), 2),
        )
        self.assertBinds(
            aten.conv_transpose2d.input,
            (self.empty(1, 6, 5, 7), self.empty(6, 2, 3, 3), self.empty(6)),
            {"stride": [2, 3], "output_padding": [1, 2], "groups": 3, "dilation": [1]},
            ConvParams((2, 3), (0, 0), (0, 0), (1, 1), True, (1, 2), 3),
        )
        self.assertBinds(
            aten.conv_transpose3d.input,
            (self.empty(1, 2, 3, 4, 5), self.empty(2, 3, 2, 2, 2), None, [2], [1]),
            {"dilation": [1, 2, 1]},
            ConvParams((2,) * 3, (1,) * 3, (1,) * 3, (1, 2, 1), True, (0,) * 3, 1),
        )

    def test_convolution(self):
        # A forward convolution ignores output_padding.
        x, w = self.empty(2, 4, 8, 8), self.empty(4, 2, 3, 3)
        for transposed in (False, True):
            with self.subTest(transposed=transposed):
                self.assertBinds(
                    aten.convolution.default,
                    (x, w, None, [2, 1], [1, 0], [1, 1], transposed, [1, 0], 2),
                    {},
                    ConvParams(
                        (2, 1),
                        (1, 0),
                        (1, 0),
                        (1, 1),
                        transposed,
                        (1, 0) if transposed else (0, 0),
                        2,
                    ),
                )

    def test_rejects(self):
        x, w = self.empty(4, 8, 8), self.empty(4, 4, 3, 3)
        with self.assertRaisesRegex(ConvBindingError, "unbatched"):
            bind_conv(aten.conv2d.default, (x, w), {})
        with self.assertRaisesRegex(ConvBindingError, "not a conv"):
            bind_conv(aten.add.Tensor, (x, w), {})
        with self.assertRaisesRegex(ConvBindingError, "missing"):
            bind_conv(aten.convolution.default, (x, w, None), {})
        with self.assertRaisesRegex(ConvBindingError, "entries"):
            bind_conv(
                aten.conv2d.default, (self.empty(1, 4, 8, 8), w, None, [1, 1, 1]), {}
            )


class CanonicalArgumentsTest(unittest.TestCase):
    """A transposed convolution needs 0 <= output_padding < max(stride,
    dilation) on every axis, a forward one output_padding 0; both binding and
    direct calls of the canonical op check it."""

    def setUp(self):
        self.mode = FakeTensorMode()

    def empty(self, *shape):
        with self.mode:
            return torch.empty(*shape, device="cuda")

    def cases(self, rank):
        """(stride, dilation, output_padding, valid) on the last axis."""
        return (
            (2, 1, 1, True),
            (2, 1, 2, False),
            (2, 1, -1, False),
            # At or past the stride but below the dilation.
            (1, 3, 2, True),
            (1, 3, 3, False),
            (3, 2, 2, True),
        )

    def test_transposed_binding(self):
        ops = (
            aten.conv_transpose1d.default,
            aten.conv_transpose2d.input,
            aten.conv_transpose3d.input,
        )
        for rank, op in enumerate(ops, 1):
            x, w = self.empty(1, 4, *([6] * rank)), self.empty(4, 2, *([3] * rank))
            for stride, dilation, output_padding, valid in self.cases(rank):
                ones = [1] * (rank - 1)
                args = (
                    x,
                    w,
                    None,
                    ones + [stride],
                    [0] * rank,
                    [0] * (rank - 1) + [output_padding],
                    1,
                    ones + [dilation],
                )
                convolution_args = (*args[:4], [0] * rank, args[7], True, args[5], 1)
                with self.subTest(
                    rank=rank, stride=stride, output_padding=output_padding
                ):
                    for target, call_args in (
                        (op, args),
                        (aten.convolution.default, convolution_args),
                    ):
                        if valid:
                            params = bind_conv(target, call_args, {})[3]
                            self.assertEqual(params.output_padding[-1], output_padding)
                        else:
                            with self.assertRaisesRegex(
                                ConvBindingError, "output_padding"
                            ):
                                bind_conv(target, call_args, {})

    def test_canonical_op(self):
        import executorch.backends.cuda.triton.kernels  # noqa: F401

        for rank in (1, 2, 3):
            x, w = self.empty(1, 4, *([6] * rank)), self.empty(4, 2, *([3] * rank))
            for stride, dilation, output_padding, valid in self.cases(rank):
                ones = [1] * (rank - 1)
                args = (
                    x,
                    w,
                    None,
                    ones + [stride],
                    [0] * rank,
                    [0] * rank,
                    ones + [dilation],
                    True,
                    [0] * (rank - 1) + [output_padding],
                    1,
                )
                with self.subTest(
                    rank=rank, stride=stride, output_padding=output_padding
                ):
                    with self.mode:
                        if valid:
                            out = torch.ops.triton.convolution(*args)
                            self.assertEqual(out.dim(), rank + 2)
                        else:
                            with self.assertRaisesRegex(ValueError, "output_padding"):
                                torch.ops.triton.convolution(*args)
        x, w = self.empty(1, 4, 6, 6), self.empty(4, 4, 3, 3)
        forward = (x, w, None, [1, 1], [0, 0], [0, 0], [1, 1], False, [0, 1], 1)
        with self.mode, self.assertRaisesRegex(ValueError, "output_padding"):
            torch.ops.triton.convolution(*forward)


class OutShapeAndStrideTest(unittest.TestCase):
    """``out_shape_and_stride`` against FakeTensor ``aten.convolution`` on CUDA."""

    LAYOUTS = ("contiguous", "input_cl", "weight_cl", "both_cl")

    def check(self, x_shape, w_shape, params, layout, offset=False):
        mode = FakeTensorMode()
        with mode:
            x = torch.empty(x_shape, device="cuda")
            w = torch.empty(w_shape, device="cuda")
            if offset:
                x = torch.empty(x.shape[0], x.shape[1] + 1, *x.shape[2:], device="cuda")
                x = x[:, 1:]
            if layout in ("input_cl", "both_cl"):
                x = _channels_last(x)
            if layout in ("weight_cl", "both_cl"):
                w = _channels_last(w)
            expected = aten.convolution.default(
                x,
                w,
                None,
                list(params.stride),
                list(params.padding_before),
                list(params.dilation),
                params.transposed,
                list(params.output_padding),
                params.groups,
            )
        got = out_shape_and_stride(ConvCall(x, w, None, params))
        self.assertEqual(got, (tuple(expected.shape), tuple(expected.stride())))

    def test_matches_aten(self):
        shapes = {
            1: [(2, 4, 9), (1, 1, 7), (3, 4, 1)],
            2: [(2, 4, 9, 7), (1, 4, 1, 1), (2, 1, 5, 6), (1, 4, 1, 6), (2, 4, 6, 1)],
            3: [(2, 4, 5, 6, 7), (1, 4, 1, 1, 1), (1, 1, 3, 1, 4)],
        }
        for rank, transposed, layout, offset in itertools.product(
            (1, 2, 3), (False, True), self.LAYOUTS, (False, True)
        ):
            for x_shape in shapes[rank]:
                channels = x_shape[1]
                groups = 2 if channels % 2 == 0 else 1
                kernel = (3,) + (1, 2)[: rank - 1]
                w_shape = (
                    (channels, 3, *kernel)
                    if transposed
                    else (6, channels // groups, *kernel)
                )
                params = ConvParams(
                    (2,) * rank,
                    (1,) * rank,
                    (1,) * rank,
                    (1,) * rank,
                    transposed,
                    (1,) * rank if transposed else (0,) * rank,
                    groups,
                )
                with self.subTest(
                    x=x_shape, transposed=transposed, layout=layout, offset=offset
                ):
                    self.check(x_shape, w_shape, params, layout, offset)

    def test_depthwise_and_dilated(self):
        for layout in self.LAYOUTS:
            with self.subTest(layout=layout):
                self.check(
                    (2, 8, 9, 9),
                    (8, 1, 3, 3),
                    ConvParams((1, 1), (1, 1), (1, 1), (1, 1), False, (0, 0), 8),
                    layout,
                )
                self.check(
                    (2, 8, 9, 9),
                    (16, 1, 3, 3),
                    ConvParams((1, 1), (2, 2), (2, 2), (2, 2), False, (0, 0), 8),
                    layout,
                )

    def test_large_output_padding_is_contiguous(self):
        # output_padding >= stride (allowed below the dilation) leaves cuDNN.
        for layout in self.LAYOUTS:
            with self.subTest(layout=layout):
                self.check(
                    (1, 4, 5, 5),
                    (4, 2, 3, 3),
                    ConvParams((1, 1), (0, 0), (0, 0), (2, 2), True, (1, 1), 1),
                    layout,
                )

    def test_asymmetric_forward_matches_explicit_pad(self):
        mode = FakeTensorMode()
        with mode:
            x = torch.empty(1, 3, 10, 11, device="cuda")
            w = torch.empty(5, 3, 4, 3, device="cuda")
            padded = aten.constant_pad_nd.default(x, [0, 2, 3, 1])
            expected = aten.convolution.default(
                padded, w, None, [2, 1], [0, 0], [1, 1], False, [0, 0], 1
            )
        params = ConvParams((2, 1), (3, 0), (1, 2), (1, 1), False, (0, 0), 1)
        got = out_shape_and_stride(ConvCall(x, w, None, params))
        self.assertEqual(got, (tuple(expected.shape), tuple(expected.stride())))

    def test_asymmetric_transposed_crops_the_full_output(self):
        # Full transposed length (in - 1) * s + d * (k - 1) + 1, plus
        # output_padding, minus before and after.
        mode = FakeTensorMode()
        with mode:
            x = torch.empty(2, 4, 6, 5, device="cuda")
            w = torch.empty(4, 3, 3, 4, device="cuda")
        params = ConvParams((2, 3), (1, 0), (2, 3), (1, 2), True, (1, 2), 1)
        shape, _ = out_shape_and_stride(ConvCall(x, w, None, params))
        full = [(6 - 1) * 2 + 1 * (3 - 1) + 1, (5 - 1) * 3 + 2 * (4 - 1) + 1]
        self.assertEqual(shape, (2, 3, full[0] + 1 - 1 - 2, full[1] + 2 - 0 - 3))
