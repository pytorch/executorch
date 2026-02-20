# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from executorch.backends.transforms.utils import is_param_node
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.dialects._ops import ops as exir_ops

from torch.export.exported_program import ExportedProgram


def get_compile_spec(
    compile_specs: List[CompileSpec], spec_name: str, required=False
) -> CompileSpec:
    for spec in compile_specs:
        if spec_name == spec.key:
            return spec
    assert not required, f"Require {spec_name} but it doesn't exist."


def is_graph_input(exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
    return node.op == "placeholder" and not is_param_node(exported_program, node)


def is_graph_output(node: torch.fx.Node) -> bool:
    # skip getitem node
    for user in node.users.keys():
        if user.op == "output" or (
            user.target.__name__ == "getitem" and is_graph_output(user)
        ):
            return True
    return False


def _quantize_per_tensor(
    in_tensor: torch.Tensor,
    scales: List[float],
    zeropoints: List[int],
    dtype: torch.dtype,
    qrange: Optional[Tuple[int, int]],
):
    assert (
        len(scales) == 1
    ), "For per-tensor quantization, there should be only one scale/zeropoint"
    return exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
        in_tensor,
        torch.Tensor(scales),
        torch.Tensor(zeropoints),
        qrange[0],
        qrange[1],
        dtype,
    )


def _quantize_per_channel(
    in_tensor: torch.Tensor,
    scales: List[float],
    zeropoints: List[int],
    dtype: torch.dtype,
    qrange: Optional[Tuple[int, int]],
    axis: Optional[int],  # Only for per-channel
):
    assert (
        len(scales) == in_tensor.shape[axis]
    ), "Shape not match for quant params and input tensor"
    return exir_ops.edge.quantized_decomposed.quantize_per_channel.default(
        in_tensor,
        torch.Tensor(scales),
        torch.Tensor(zeropoints),
        axis,
        qrange[0],
        qrange[1],
        dtype,
    )


def quantize_tensor(
    in_tensor: torch.Tensor,
    scales: List[float],
    zeropoints: List[int],
    dtype: torch.dtype,
    qrange: Optional[Tuple[int, int]] = None,
    axis: Optional[int] = None,  # Only for per-channel
) -> torch.Tensor:
    """
    To quantize constant tensor by executorch OPs. If `axis` not set, we quantize the tensor by per tensor.
    If `axis` was set, we do per-channel quantize.

    :param in_tensor: The tensor to be quantized
    :param scales: List of scales. For per-tensor quantization, it should contain only one element
    :param zeropoints: List of zeropoints. For per-tensor quantization, it should contain only one element
    :param dtype: The output dtype
    :param qrange: The quantization range (qmin, qmax).
        If not set, we will get the maximum range of the dtype by `torch.iinfo`
    :param axis: We do per-channel quantize by which axis.
        Only when this parameter set, we do per-channel quantization
    :type in_tensor: torch.Tensor
    :type scalse: List[float]
    :type zeropoints: List[int]
    :type dtype: torch.dtype
    :type qrange: Optional[Tuple[int,int]]
    :type axis: Optional[int]
    :return: The quantized tensor
    """
    assert len(scales) == len(
        zeropoints
    ), "scales should have same shape with zeropoints"
    if not qrange:
        qrange = (torch.iinfo(dtype).min, torch.iinfo(dtype).max)

    if axis is not None:
        return _quantize_per_channel(in_tensor, scales, zeropoints, dtype, qrange, axis)
    return _quantize_per_tensor(
        in_tensor,
        scales,
        zeropoints,
        dtype,
        qrange,
    )
