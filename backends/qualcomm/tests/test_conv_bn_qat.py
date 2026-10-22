# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from torch.fx import Node
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e.utils import _is_bn_node, _is_conv_or_conv_transpose_node

_DQ_PER_CHANNEL = torch.ops.quantized_decomposed.dequantize_per_channel.default
_Q_PER_TENSOR = torch.ops.quantized_decomposed.quantize_per_tensor.default


class ConvBnNoBias(torch.nn.Module):
    """conv(bias=False) -> bn -> relu, the shape that broke OSS QNN QAT."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBnWithBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvNoBn(torch.nn.Module):
    """A conv with no batchnorm must be left to the per-node annotator."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=True)

    def forward(self, x):
        return torch.relu(self.conv(x))


class DepthwiseSeparableConvBn(torch.nn.Module):
    """The `conv_dw` block: two bias-free conv+bn+relu pairs back to back."""

    def __init__(self):
        super().__init__()
        self.dw = torch.nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.pw = torch.nn.Conv2d(8, 16, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(16)

    def forward(self, x):
        x = torch.relu(self.bn1(self.dw(x)))
        return torch.relu(self.bn2(self.pw(x)))


class ConvBnSamePadding(torch.nn.Module):
    """`padding="same"` exports to `aten.conv2d.padding`, which torchao cannot fold."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding="same", bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1dBn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 8, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm1d(8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvTranspose2dBn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _qat_convert(module: torch.nn.Module, example_inputs):
    quantizer = QnnQuantizer()
    quantizer.set_default_quant_config(
        QuantDtype.use_8a8w, is_qat=True, is_conv_per_channel=True
    )
    exported = torch.export.export(module, example_inputs, strict=True).module()
    prepared = prepare_qat_pt2e(exported, quantizer)
    prepared(*example_inputs)
    return convert_pt2e(prepared)


class TestConvBnQat(unittest.TestCase):
    """The two defects behind `0x7532` on the OSS QNN QAT export.

    1. batchnorm never folded, because an observer on the conv output landed
       inside torchao's QAT conv-bn fusion and stopped `_fold_conv_bn_qat` from
       matching;
    2. the folded bias was left unquantized on convs declared `bias=False`,
       which QNN rejects on a per-channel-quantized conv.
    """

    def _assert_no_batch_norm(self, converted):
        surviving = [n for n in converted.graph.nodes if _is_bn_node(n)]
        self.assertEqual(surviving, [], f"batchnorm survived convert_pt2e: {surviving}")

    def _assert_conv_outputs_quantized(self, converted):
        """QNN lowers conv and relu to separate ops, so the conv's own output has
        to be quantized. Putting the partition's output qspec on a trailing relu
        instead leaves the conv emitting float32 and QNN rejects it with
        `0x232 != 0x408`.
        """
        convs = [n for n in converted.graph.nodes if _is_conv_or_conv_transpose_node(n)]
        self.assertGreater(len(convs), 0)
        for conv in convs:
            users = list(conv.users)
            self.assertEqual(
                [u.target for u in users],
                [_Q_PER_TENSOR] * len(users),
                f"{conv.name} output is not quantized; its users are "
                f"{[(u.op, u.target) for u in users]}",
            )

    def _assert_conv_biases_quantized(self, converted):
        convs = [n for n in converted.graph.nodes if _is_conv_or_conv_transpose_node(n)]
        self.assertGreater(len(convs), 0)
        for conv in convs:
            if len(conv.args) < 3:
                continue
            bias = conv.args[2]
            if not isinstance(bias, Node):
                continue
            self.assertEqual(
                bias.target,
                _DQ_PER_CHANNEL,
                f"{conv.name} bias is {bias.op}:{bias.target}, expected a "
                "dequantize_per_channel; QNN rejects a float bias on a "
                "per-channel-quantized conv",
            )

    def test_conv_bn_relu_without_bias_folds_and_quantizes_bias(self):
        example_inputs = (torch.randn(1, 3, 16, 16),)
        converted = _qat_convert(ConvBnNoBias().eval(), example_inputs)
        self._assert_no_batch_norm(converted)
        self._assert_conv_biases_quantized(converted)
        self._assert_conv_outputs_quantized(converted)

    def test_conv_bn_with_bias_folds(self):
        example_inputs = (torch.randn(1, 3, 16, 16),)
        converted = _qat_convert(ConvBnWithBias().eval(), example_inputs)
        self._assert_no_batch_norm(converted)
        self._assert_conv_biases_quantized(converted)
        self._assert_conv_outputs_quantized(converted)

    def test_depthwise_separable_conv_bn_folds(self):
        example_inputs = (torch.randn(1, 8, 16, 16),)
        converted = _qat_convert(DepthwiseSeparableConvBn().eval(), example_inputs)
        self._assert_no_batch_norm(converted)
        self._assert_conv_biases_quantized(converted)
        self._assert_conv_outputs_quantized(converted)

    def test_conv1d_bn_folds(self):
        example_inputs = (torch.randn(1, 3, 16),)
        converted = _qat_convert(Conv1dBn().eval(), example_inputs)
        self._assert_no_batch_norm(converted)
        self._assert_conv_outputs_quantized(converted)

    def test_conv_transpose2d_bn_folds(self):
        example_inputs = (torch.randn(1, 3, 16, 16),)
        converted = _qat_convert(ConvTranspose2dBn().eval(), example_inputs)
        self._assert_no_batch_norm(converted)
        self._assert_conv_outputs_quantized(converted)

    def test_unfoldable_conv_overload_keeps_quantized_output(self):
        """A conv the fold cannot match must keep its own output qspec.

        `padding="same"` exports to `aten.conv2d.padding`. torchao's
        `_fuse_conv_bn_qat` / `_fold_conv_bn_qat` trace their patterns from
        `F.conv1d`/`F.conv2d`/`F.conv_transpose{1,2}d`, so only those four
        overloads can ever match and BatchNorm survives here.

        That is a torchao limitation this pass cannot fix, but it must not make it
        worse: if the partition pass claims the conv it drops the conv's output
        qspec expecting the fold to move the bn output observer onto it, and the
        conv is left emitting float32 into a quantized graph -- which QNN rejects
        with `0x232 != 0x408`. So the conv must stay quantized even when the fold
        misses.
        """
        example_inputs = (torch.randn(1, 3, 16, 16),)
        converted = _qat_convert(ConvBnSamePadding().eval(), example_inputs)
        self._assert_conv_outputs_quantized(converted)

    def test_conv_without_bn_is_still_annotated(self):
        """The partition pass must not starve the per-node conv annotator."""
        example_inputs = (torch.randn(1, 3, 16, 16),)
        converted = _qat_convert(ConvNoBn().eval(), example_inputs)
        convs = [n for n in converted.graph.nodes if _is_conv_or_conv_transpose_node(n)]
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0].args[1].target, _DQ_PER_CHANNEL)

    def test_ptq_annotation_is_unchanged(self):
        """PTQ already worked; the partition pass is gated off for it."""
        from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

        example_inputs = (torch.randn(1, 3, 16, 16),)
        quantizer = QnnQuantizer()
        quantizer.set_default_quant_config(
            QuantDtype.use_8a8w, is_qat=False, is_conv_per_channel=True
        )
        exported = torch.export.export(
            ConvBnNoBias().eval(), example_inputs, strict=True
        ).module()
        prepared = prepare_pt2e(exported, quantizer)
        prepared(*example_inputs)
        converted = convert_pt2e(prepared)
        # PTQ folds batchnorm outside the quantizer; assert we did not regress it.
        self._assert_no_batch_norm(converted)
