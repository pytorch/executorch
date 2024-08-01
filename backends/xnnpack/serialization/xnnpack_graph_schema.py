# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Please refer to executorch/backends/xnnpack/serialization/schema.fbs for the schema definitions
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Union


# Generic node data class with one input and one output
@dataclass
class XNNNode1x1:
    input_id: int
    output_id: int
    flags: int


# Generic node data class with two inputs and one output
@dataclass
class XNNNode2x1:
    input1_id: int
    input2_id: int
    output_id: int
    flags: int


# Generic node data class for concatenation node
@dataclass
class XNNCat:
    axis: int
    input1_id: int
    input2_id: int
    input3_id: int
    input4_id: int
    output_id: int
    flags: int


# Generic node data class for convolution type nodes
@dataclass
class XNNNodeConv:
    padding_top: int
    padding_right: int
    padding_bottom: int
    padding_left: int
    kernel_height: int
    kernel_width: int
    subsampling_height: int
    subsampling_width: int
    dilation_height: int
    dilation_width: int
    group_input_channels: int
    group_output_channels: int
    groups: int
    adjustment_height: int
    adjustment_width: int
    input1_id: int
    filter_id: int
    bias_id: int
    output_id: int
    flags: int


@dataclass
class XNNPooling2D:
    padding_top: int
    padding_right: int
    padding_bottom: int
    padding_left: int
    pooling_height: int
    pooling_width: int
    stride_height: int
    stride_width: int
    dilation_height: int
    dilation_width: int
    input_id: int
    output_id: int
    flags: int


# Node data class for average pooling 2d
@dataclass
class XNNAvgPooling2d(XNNPooling2D):
    pass


@dataclass
class XNNMaxPooling2d(XNNPooling2D):
    pass


@dataclass
class XNNConv2d(XNNNodeConv):
    pass


@dataclass
class XNNAdd(XNNNode2x1):
    pass


@dataclass
class XNNGlobalAvgPooling2d(XNNNode1x1):
    pass


@dataclass
class XNNDiv(XNNNode2x1):
    pass


@dataclass
class XNNMultiply(XNNNode2x1):
    pass


@dataclass
class XNNMinimum(XNNNode2x1):
    pass


@dataclass
class XNNSubtract(XNNNode2x1):
    pass


@dataclass
class XNNSoftmax(XNNNode1x1):
    pass


@dataclass
class XNNSigmoid(XNNNode1x1):
    pass


@dataclass
class XNNFloor(XNNNode1x1):
    pass


@dataclass
class XNNConvert(XNNNode1x1):
    pass


@dataclass
class XNNNegate(XNNNode1x1):
    pass


@dataclass
class XNNAbs(XNNNode1x1):
    pass


@dataclass
class XNNConcatenate2(XNNCat):
    pass


@dataclass
class XNNConcatenate3(XNNCat):
    pass


@dataclass
class XNNConcatenate4(XNNCat):
    pass


@dataclass
class XNNBatchMatrixMultiply(XNNNode2x1):
    pass


@dataclass
class XNNStaticTranspose:
    num_dims: int
    perm: List[int]
    input_id: int
    output_id: int
    flags: int


@dataclass
class XNNStaticSlice:
    num_dims: int
    offsets: List[int]
    sizes: List[int]
    input_id: int
    output_id: int
    flags: int


@dataclass
class XNNClamp(XNNNode1x1):
    pass


@dataclass
class XNNStaticResizeBilinear2D:
    new_height: int
    new_width: int
    input_id: int
    output_id: int
    flags: int


@dataclass
class XNNStaticConstantPad:
    pre_paddings: List[int]
    post_paddings: List[int]
    padding_value: float
    input_id: int
    output_id: int
    flags: int


@dataclass
class XNNDepthwiseConv2d(XNNNodeConv):
    pass


@dataclass
class XNNArgMaxPooling2d:
    padding_top: int
    padding_right: int
    padding_bottom: int
    padding_left: int
    pooling_height: int
    pooling_width: int
    input_id: int
    output_value_id: int
    output_index_id: int
    flags: int


# this class such that Python can infer the XNodeUnion Type. If there is only type in Union, like
# Union[XNNAdd], python will infer it's XNNAdd type instead of Union type. After we add more operators
# this one can be removed.
@dataclass
class XNNFullyConnected:  # aten::Linear
    input1_id: int
    filter_id: int
    bias_id: int
    output_id: int
    flags: int


@dataclass
class XNNStaticReshape:
    num_dims: int
    new_shape: List[int]
    input_id: int
    output_id: int
    flags: int


@dataclass
class XNNSquareRoot(XNNNode1x1):
    pass


@dataclass
class XNNCeiling(XNNNode1x1):
    pass


@dataclass
class XNNHardswish(XNNNode1x1):
    pass


@dataclass
class XNNSquare(XNNNode1x1):
    pass


@dataclass
class XNNLeakyReLU:
    negative_slope: float
    input_id: int
    output_id: int
    flags: int


@dataclass
class XNNMaximum(XNNNode2x1):
    pass


@dataclass
class XNNELU:
    alpha: float
    input_id: int
    output_id: int
    flags: int


@dataclass
class XNNPReLU(XNNNode2x1):
    pass


@dataclass
class XNNScaledDotProductAttention:
    query_id: int
    key_id: int
    value_id: int
    scale_id: int
    mask_id: int
    output_id: int
    flags: int


XNodeUnion = Union[
    XNNAdd,
    XNNFullyConnected,
    XNNSoftmax,
    XNNSigmoid,
    XNNStaticTranspose,
    XNNClamp,
    XNNConv2d,
    XNNDiv,
    XNNStaticResizeBilinear2D,
    XNNStaticConstantPad,
    XNNAvgPooling2d,
    XNNMinimum,
    XNNDepthwiseConv2d,
    XNNMaxPooling2d,
    XNNMultiply,
    XNNSubtract,
    XNNFloor,
    XNNConvert,
    XNNGlobalAvgPooling2d,
    XNNStaticReshape,
    XNNArgMaxPooling2d,
    XNNSquareRoot,
    XNNCeiling,
    XNNHardswish,
    XNNLeakyReLU,
    XNNMaximum,
    XNNNegate,
    XNNSquare,
    XNNELU,
    XNNAbs,
    XNNPReLU,
    XNNConcatenate2,
    XNNConcatenate3,
    XNNConcatenate4,
    XNNStaticSlice,
    XNNScaledDotProductAttention,
    XNNBatchMatrixMultiply,
]


@dataclass
class OutputMinMax:
    output_min: Union[float, str]
    output_max: Union[float, str]


@dataclass
class XNode:
    xnode_union: "XNodeUnion"
    debug_handle: int
    output_min_max: Optional[OutputMinMax] = None


class XNNDatatype(IntEnum):
    xnn_datatype_invalid = 0
    xnn_datatype_fp32 = 1
    xnn_datatype_fp16 = 2
    xnn_datatype_qint8 = 3
    xnn_datatype_quint8 = 4
    xnn_datatype_qint32 = 5
    xnn_datatype_qcint8 = 6
    xnn_datatype_qcint32 = 7
    xnn_datatype_qcint4 = 8
    xnn_datatype_qdint8 = 9
    xnn_datatype_qbint4 = 10


@dataclass
class PerChannelQuant:
    scale: List[float]
    channel_dim: int


@dataclass
class PerChannelGroupQuant:
    scale: List[float]
    channel_dim: int
    group_size: int = 1


@dataclass
class PerTokenDynamicQuant:
    num_nonbatch_dims: int


@dataclass
class PerTensorQuant:
    scale: float
    zero_point: int


XNNQuantParams = Union[
    PerChannelQuant, PerTensorQuant, PerTokenDynamicQuant, PerChannelGroupQuant
]


@dataclass
class XNNTensorValue:
    datatype: XNNDatatype
    num_dims: int
    dims: List[int]
    constant_buffer_idx: int
    external_id: int
    flags: int
    id_out: int


@dataclass
class XNNQuantizedTensorValue:
    tensor_value: XNNTensorValue
    quant_params: "XNNQuantParams"


XValueUnion = Union[
    XNNTensorValue,
    XNNQuantizedTensorValue,
]


@dataclass
class XValue:
    xvalue_union: "XValueUnion"


@dataclass
class ConstantDataOffset:
    offset: int
    size: int


@dataclass
class XNNGraph:
    version: str
    xnodes: List[XNode]
    xvalues: List[XValue]

    num_externs: int
    input_ids: List[int]
    output_ids: List[int]

    constant_data: List[ConstantDataOffset]
