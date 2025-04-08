# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from executorch.backends.qualcomm.utils.constants import QCOM_QNN_COMPILE_SPEC

from executorch.exir.backend.compile_spec_schema import CompileSpec


def generate_qnn_executorch_option(
    compiler_specs: List[CompileSpec],
) -> bytes:
    for compiler_spec in compiler_specs:
        if compiler_spec.key == QCOM_QNN_COMPILE_SPEC:
            qnn_compile_spec_buffer = compiler_spec.value
        else:
            raise ValueError(f"unknown compiler spec key value: {compiler_spec.key}")
    return qnn_compile_spec_buffer


def get_skip_decomp_table() -> List[torch._ops.OperatorBase]:
    do_not_decompose = [
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.elu.default,
        torch.ops.aten.hardsigmoid.default,
        torch.ops.aten.hardswish.default,
        torch.ops.aten.instance_norm.default,
        torch.ops.aten.leaky_relu.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.pixel_shuffle.default,
        torch.ops.aten.pixel_unshuffle.default,
        torch.ops.aten.prelu.default,
        torch.ops.aten.rms_norm.default,
        torch.ops.aten._safe_softmax.default,
        torch.ops.aten.stack.default,
        # This request is ignored because it is in a blocklist. Refer to exir/program/_program.py
        # torch.ops.aten.unbind.int,
        torch.ops.pt2e_quant.quantize_affine.default,
        torch.ops.pt2e_quant.dequantize_affine.default,
    ]
    return do_not_decompose
