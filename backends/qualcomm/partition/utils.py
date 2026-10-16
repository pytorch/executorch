# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional

import torch

from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC

from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops

logger = logging.getLogger(__name__)


def generate_qnn_executorch_option(
    compiler_specs: List[CompileSpec],
) -> bytes:
    qnn_compile_spec_buffer = None

    for compiler_spec in compiler_specs:
        if compiler_spec.key == QCOM_QNN_COMPILE_SPEC:
            qnn_compile_spec_buffer = compiler_spec.value
        else:
            raise ValueError(f"unknown compiler spec key value: {compiler_spec.key}")

    if qnn_compile_spec_buffer is None:
        raise ValueError(
            f"QNN compile spec (key={QCOM_QNN_COMPILE_SPEC}) not found in compiler_specs"
        )

    return qnn_compile_spec_buffer


def estimate_conv_vtcm_working_set(
    in_channels: int,
    spatial: List[int],
    paddings: List[int],
    kernels: List[int],
    dilations: List[int],
    weight_numel: int,
    byte_width: int = 2,
) -> int:
    """
    Proxy for the VTCM (TCM) working set of a single convolution, in bytes:
        (in_channels * prod(receptive_field) + weight_numel) * byte_width
    The weights stay resident for the whole operator, and the smallest useful
    tile is one output pixel, whose input window is the dilated receptive field
    `1 + dilation * (kernel - 1)` per spatial dimension, clamped to the padded
    input. `byte_width` is the element width, i.e. 2 for fp16.

    Over-estimates, because it ignores that the tiler can also split along `cin`
    and `cout`: it exceeds VTCM for every shape measured in #23096 that fails on
    device, but also for one that runs (cin 960, dilation 24). Callers must treat
    an over-VTCM result as a hint only, never as grounds for rejecting an op.
    """
    footprint = in_channels
    for size, pad, kernel, dilation in zip(spatial, paddings, kernels, dilations):
        footprint *= min(size + 2 * pad, 1 + dilation * (kernel - 1))
    return (footprint + weight_numel) * byte_width


def warn_if_dilated_conv_may_not_fit_vtcm(
    node: torch.fx.Node,
    vtcm_size_in_mb: int,
    phase: str = "QnnPartitioner",
) -> Optional[int]:
    """
    Log a warning if `node` is a dilated convolution that may not be tileable
    into VTCM. Returns the estimate when it warns, otherwise None.

    On the fp16 path HTP does not reject a dilated convolution it failed to tile
    into VTCM -- the quantized path does, at finalize, with `not sufficiently
    tiled to fit in TCM`. fp16 instead emits a context binary that cannot
    execute, and the DSP stops responding with a transport error (err 1003 /
    1007 / 1011) after a fixed watchdog timeout. Nothing in the export log points
    at the cause, so this names the operator likely responsible.

    Nothing is un-delegated: the estimate over-estimates on at least one shape
    that runs, so acting on it would move working models to CPU.

    TODO: this is a workaround for a backend defect, not a fix. Remove it once
    HTP performs the TCM-fit check on the fp16 path. See
    https://github.com/pytorch/executorch/issues/23096
    """
    if node.target is not exir_ops.edge.aten.convolution.default:
        return None

    vtcm_bytes = vtcm_size_in_mb * 1024 * 1024
    if vtcm_bytes == 0:
        return None

    try:
        dilations = node.args[5]
        if node.args[8] != 1 or not any(d > 1 for d in dilations):
            # Grouped convolutions divide the channel work up already, and an
            # undilated convolution tiles down to the kernel size.
            return None

        paddings = node.args[4]
        input_shape = node.args[0].meta["val"].shape
        # (cout, cin / groups, *kernel)
        weight = node.args[1].meta["val"]
        estimate = estimate_conv_vtcm_working_set(
            in_channels=int(input_shape[1]),
            spatial=[int(s) for s in input_shape[2:]],
            paddings=list(paddings),
            kernels=[int(k) for k in weight.shape[2:]],
            dilations=list(dilations),
            weight_numel=weight.numel(),
        )
    except (AttributeError, IndexError, KeyError, TypeError):
        # Not a shape we can reason about (e.g. dynamic shapes, or a weight fed
        # by something other than a tensor). Stay silent rather than guess.
        return None

    if estimate <= vtcm_bytes:
        return None

    logger.warning(
        f"[{phase}] {node.name} | dilation={list(dilations)} convolution needs "
        f"at least {estimate} bytes of VTCM for its weights and the receptive "
        f"field of a single output pixel, but this SoC has {vtcm_bytes} bytes, "
        "so HTP may fail to tile it. Unlike the quantized path, fp16 does not "
        "reject such a graph at compile time -- it emits a context binary that "
        "stops the DSP at execute (err 1003/1007/1011 after a fixed ~10 s "
        "timeout). This operator is still being delegated, since the estimate "
        "is approximate. If you do hit that failure, splitting this convolution "
        "along its input channels and summing the results is mathematically "
        "identical and does run. See pytorch/executorch#23096."
    )
    return estimate


# Logic to determine whether to skip decompose and has higher priority than get_skip_decomp_table()
def filter_fn(node: torch.fx.Node) -> bool:
    # QNN does not support int32/int64 IO for the following OPs.
    potential_i32_i64_io_ops = [
        torch.ops.aten.stack.default,
        torch.ops.aten.unbind.int,
    ]
    if node.target in potential_i32_i64_io_ops and node.meta["val"].dtype in [
        torch.int32,
        torch.int64,
    ]:
        return False
    return True


def get_skip_decomp_table() -> List[torch._ops.OperatorBase]:
    do_not_decompose = [
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.channel_shuffle.default,
        torch.ops.aten.col2im.default,
        torch.ops.aten.convolution_backward.default,
        torch.ops.aten.elu.default,
        torch.ops.aten.floor_divide.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardswish.default,
        torch.ops.aten.im2col.default,
        torch.ops.aten.instance_norm.default,
        torch.ops.aten.leaky_relu.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.pixel_shuffle.default,
        torch.ops.aten.pixel_unshuffle.default,
        torch.ops.aten.prelu.default,
        torch.ops.aten.reflection_pad1d.default,
        torch.ops.aten.reflection_pad2d.default,
        torch.ops.aten.rms_norm.default,
        torch.ops.aten._safe_softmax.default,
        torch.ops.aten.scatter.src,
        torch.ops.aten.scatter_add.default,
        torch.ops.aten.scatter_reduce.two,
        torch.ops.aten.stack.default,
        torch.ops.aten.upsample_bicubic2d.vec,
        # This request is ignored because it is in a blocklist. Refer to exir/program/_program.py
        torch.ops.aten.unbind.int,
        torch.ops.torchao.quantize_affine.default,
        torch.ops.torchao.dequantize_affine.default,
        torch.ops.quantized_decomposed.quantize_per_channel_group.default,
        torch.ops.quantized_decomposed.dequantize_per_channel_group.default,
    ]
    return do_not_decompose
