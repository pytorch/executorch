# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
from typing import final, List

import torch

from executorch.backends.qnnpack.serialization.qnnpack_graph_schema import (
    ConstTensor,
    QNNDynamicLinear,
)
from executorch.backends.qnnpack.serialization.qnnpack_graph_serialize import (
    convert_to_flatbuffer,
)

from executorch.backends.transforms import get_shape

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)

from executorch.exir.dialects._ops import ops as exir_ops
from torch._export.exported_program import ExportedProgram

T_Mm = exir_ops.edge.aten.mm.default
T_Addmm = exir_ops.edge.aten.addmm.default
T_Linear = exir_ops.edge.aten.linear.default


def _copy_buffer(storage: torch.UntypedStorage) -> bytes:
    array_type = ctypes.c_char * storage.nbytes()
    array = ctypes.cast(
        storage.data_ptr(),
        ctypes.POINTER(array_type),
    ).contents
    return bytes(array)


@final
class QnnpackBackend(BackendDetails):
    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:

        for node in edge_program.graph.nodes:
            # TODO(maxren): Follow this up by removing addm and mm nodes
            if node.op == "call_function":
                # Finding the linear node
                if node.target in (T_Mm, T_Addmm, T_Linear):
                    # Padding with 16 bytes of 0
                    padding = b"\0" * 16
                    bias_tensor = None
                    if node.target == T_Addmm:
                        op_input = node.args[1]
                        weight = node.args[2]
                        # For linear node, bias is known
                        bias_tensor = getattr(
                            edge_program.graph_module, node.args[0].target
                        ).contiguous()
                        # t_defualt node -> dequant node
                        weight_dequant = weight.args[0]
                    elif node.target == T_Mm:
                        op_input = node.args[0]
                        weight = node.args[1]
                        # t_defualt node -> dequant node
                        weight_dequant = weight.args[0]
                    elif node.target == T_Linear:
                        op_input = node.args[0]
                        weight_dequant = node.args[1]
                        if len(node.args) > 2:
                            bias_tensor = getattr(
                                edge_program.graph_module, node.args[2].target
                            ).contiguous()
                    else:
                        raise RuntimeError(
                            "Node %s not supported", node.target.__name__
                        )

                    weight_shape = get_shape(weight_dequant)
                    output_channels = weight_shape[0]
                    if bias_tensor is None:
                        bias_tensor = torch.zeros(output_channels)
                    # input
                    input_shape = get_shape(op_input)

                    # bias
                    op_bias = ConstTensor(
                        shape=list(bias_tensor.shape),  # should be 1d
                        buffer=_copy_buffer(bias_tensor.untyped_storage()) + padding,
                    )

                    # deqaunt node -> quant node
                    weight_quant = weight_dequant.args[0]
                    # quant node -> tensor_constant
                    weight_const = getattr(
                        edge_program.graph_module, weight_quant.args[0].target
                    )
                    if (
                        weight_quant.target.__name__
                        == "quantized_decomposed.quantize_per_channel.default"
                    ):
                        # scale and zero_point are tensors
                        weight_scale = weight_quant.args[1]
                        scale_tensor = getattr(
                            edge_program.graph_module, weight_scale.target
                        )
                        weight_zeropoint = weight_quant.args[2]
                        zp_tensor = (
                            getattr(edge_program.graph_module, weight_zeropoint.target)
                            + 128
                        )
                        axis = weight_quant.args[3]
                        # requantize weight to uint8
                        requantized_weight_tensor = weight_quant.target(
                            weight_const,
                            scale_tensor,
                            zp_tensor,
                            axis,
                            0,
                            255,
                            torch.uint8,
                        )
                    elif (
                        weight_quant.target.__name__
                        == "quantized_decomposed.quantize_per_tensor.default"
                    ):
                        scale = weight_quant.args[1]
                        zeropoint = weight_quant.args[2] + 128
                        scale_tensor = torch.FloatTensor([scale] * output_channels)
                        zp_tensor = torch.IntTensor([zeropoint] * output_channels)
                        requantized_weight_tensor = weight_quant.target(
                            weight_const,
                            scale,
                            zeropoint,
                            0,
                            255,
                            torch.uint8,
                        )
                    else:
                        raise RuntimeError("Not Supported Quantization")

                    # Prep Tensors for Serializing Data
                    zp_tensor = zp_tensor.type(torch.uint8).contiguous()
                    scale_tensor = scale_tensor.contiguous()
                    requantized_weight_tensor = requantized_weight_tensor.contiguous()
                    # Weights as Tensor
                    op_weight = ConstTensor(
                        # Right now we are serializing shape which is not in
                        # congruence with weight layout. This is just wrong.
                        # However, not changing it here since this is a BC breaking
                        # change. Lets follow this up so as to make the following
                        # line look like
                        # shape=weight_val.shape,
                        # TODO(maxren)
                        shape=[weight_shape[1], weight_shape[0]],
                        buffer=_copy_buffer(requantized_weight_tensor.untyped_storage())
                        + padding,
                    )
                    # Weight's Scales as Tensor
                    weight_scale = ConstTensor(
                        shape=[output_channels],
                        buffer=_copy_buffer(scale_tensor.untyped_storage()) + padding,
                    )
                    # Weight's Zeropoints as Tensor
                    weight_zp = ConstTensor(
                        shape=[output_channels],
                        buffer=_copy_buffer(zp_tensor.untyped_storage()) + padding,
                    )

                    dynamic_linear = QNNDynamicLinear(
                        input_shape=input_shape,
                        bias=op_bias,
                        weights=op_weight,
                        weights_zero_point=weight_zp,
                        weights_scale=weight_scale,
                    )

                    return PreprocessResult(
                        processed_bytes=convert_to_flatbuffer(dynamic_linear),
                    )

        raise RuntimeError("QNNPACK preprocess failed to lower the partitioned graph")
        return PreprocessResult(processed_bytes=b"")
