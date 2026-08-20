#
#  Copyright (c) 2023 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

"""
Please refer to executorch/backends/apple/mps/serialization/schema.fbs for the schema definitions
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Union


class MPSDataType(IntEnum):
    mps_data_type_invalid = 0
    mps_data_type_float16 = 1
    mps_data_type_float32 = 2
    mps_data_type_float64 = 3
    mps_data_type_bfloat16 = 4

    # Signed integers.
    mps_data_type_int4 = 5
    mps_data_type_int8 = 6
    mps_data_type_int16 = 7
    mps_data_type_int32 = 8
    mps_data_type_int64 = 9

    # Unsigned integers. range: [0, UTYPE_MAX]
    mps_data_type_uint4 = 10
    mps_data_type_uint8 = 11
    mps_data_type_uint16 = 12
    mps_data_type_uint32 = 13
    mps_data_type_uint64 = 14

    mps_data_type_bool = 15

    mps_data_type_complex_float16 = 16
    mps_data_type_complex_float32 = 17


class OpType(IntEnum):
    mps_graph = 0
    metal_kernel = 1


@dataclass
class MPSNode1x1:
    input1_id: int
    output_id: int


@dataclass
class MPSNode2x1:
    input1_id: int
    input2_id: int
    output_id: int


@dataclass
class MPSDivNode2x1(MPSNode2x1):
    rounding_mode: str = None


@dataclass
class MPSNode3x1:
    input1_id: int
    input2_id: int
    input3_id: int
    output_id: int


@dataclass
class MPSDequantizeNode(MPSNode1x1):
    scales_id: int
    zero_points_id: int


@dataclass
class MPSConv(MPSNode3x1):
    stride_x: int = 0
    stride_y: int = 0
    dilation_x: int = 0
    dilation_y: int = 0
    groups: int = 0
    padding_left: int = 0
    padding_right: int = 0
    padding_top: int = 0
    padding_bottom: int = 0


@dataclass
class MPSPooling2D:
    input1_id: int
    kernel_height: int
    kernel_width: int
    stride_height: int
    stride_width: int
    padding_left: int
    padding_right: int
    padding_top: int
    padding_bottom: int
    dilation_height: int
    dilation_width: int
    ceil_mode: bool
    output1_id: int
    output2_id: int = -1
    count_include_pad: bool = True
    divisor_override: int = 0


@dataclass
class MPSMinMax:
    min_value: Union[float, str] = "-inf"
    max_value: Union[float, str] = "inf"


##
## Activation ops
##
@dataclass
class MPSHardTanh(MPSNode1x1):
    min_value: float = 0.0
    max_value: float = 0.0


@dataclass
class MPSReLU(MPSNode1x1):
    pass


@dataclass
class MPSGELU(MPSNode1x1):
    approximate: str = "none"


@dataclass
class MPSLeakyReLU(MPSNode1x1):
    negative_slope: float = 0.01


@dataclass
class MPSSoftmax(MPSNode1x1):
    dim: int = 0
    half_to_float: bool = False


@dataclass
class MPSLogSoftmax(MPSNode1x1):
    dim: int = 0
    half_to_float: bool = False


##
## Binary ops
##
@dataclass
class MPSAdd(MPSNode2x1):
    alpha: float = 1.0


@dataclass
class MPSSub(MPSNode2x1):
    alpha: float = 1.0


@dataclass
class MPSMul(MPSNode2x1):
    pass


@dataclass
class MPSDiv(MPSDivNode2x1):
    pass


@dataclass
class MPSFmod(MPSDivNode2x1):
    pass


@dataclass
class MPSRemainder(MPSNode2x1):
    pass


@dataclass
class MPSMin(MPSNode2x1):
    pass


@dataclass
class MPSMax(MPSNode2x1):
    pass


@dataclass
class MPSPow(MPSNode2x1):
    pass


@dataclass
class MPSAtan2(MPSNode2x1):
    pass


@dataclass
class MPSBitwiseAnd(MPSNode2x1):
    pass


@dataclass
class MPSBitwiseOr(MPSNode2x1):
    pass


@dataclass
class MPSBitwiseXor(MPSNode2x1):
    pass


@dataclass
class MPSMinimum(MPSNode2x1):
    pass


##
## Unary ops
##
@dataclass
class MPSExp(MPSNode1x1):
    pass


@dataclass
class MPSExp2(MPSNode1x1):
    pass


@dataclass
class MPSReciprocal(MPSNode1x1):
    pass


@dataclass
class MPSSqrt(MPSNode1x1):
    pass


@dataclass
class MPSNeg(MPSNode1x1):
    pass


@dataclass
class MPSLog(MPSNode1x1):
    pass


@dataclass
class MPSLog10(MPSNode1x1):
    pass


@dataclass
class MPSLog2(MPSNode1x1):
    pass


@dataclass
class MPSErf(MPSNode1x1):
    pass


@dataclass
class MPSFloor(MPSNode1x1):
    pass


@dataclass
class MPSCeil(MPSNode1x1):
    pass


@dataclass
class MPSRsqrt(MPSNode1x1):
    pass


@dataclass
class MPSSigmoid(MPSNode1x1):
    pass


@dataclass
class MPSSin(MPSNode1x1):
    pass


@dataclass
class MPSSign(MPSNode1x1):
    pass


@dataclass
class MPSCos(MPSNode1x1):
    pass


@dataclass
class MPSTan(MPSNode1x1):
    pass


@dataclass
class MPSAbs(MPSNode1x1):
    pass


@dataclass
class MPSAsin(MPSNode1x1):
    pass


@dataclass
class MPSAcos(MPSNode1x1):
    pass


@dataclass
class MPSAtan(MPSNode1x1):
    pass


@dataclass
class MPSSinh(MPSNode1x1):
    pass


@dataclass
class MPSCosh(MPSNode1x1):
    pass


@dataclass
class MPSTanh(MPSNode1x1):
    pass


@dataclass
class MPSAsinh(MPSNode1x1):
    pass


@dataclass
class MPSAcosh(MPSNode1x1):
    pass


@dataclass
class MPSAtanh(MPSNode1x1):
    pass


@dataclass
class MPSBitwiseNot(MPSNode1x1):
    pass


@dataclass
class MPSIsnan(MPSNode1x1):
    pass


@dataclass
class MPSIsinf(MPSNode1x1):
    pass


@dataclass
class MPSRound(MPSNode1x1):
    pass


@dataclass
class MPSLogicalNot(MPSNode1x1):
    pass


@dataclass
class MPSBitwise(MPSNode1x1):
    pass


##
## Linear algebra ops
##
@dataclass
class MPSMatMul(MPSNode2x1):
    pass


@dataclass
class MPSAddmm(MPSNode3x1):
    beta: float = 1.0
    alpha: float = 1.0


##
## Constant ops
##
@dataclass
class MPSFull:
    output_id: int
    shape: List[int]
    fill_value: float
    dtype: MPSDataType


@dataclass
class MPSFullLike(MPSNode1x1):
    fill_value: Union[float, str] = 0.0
    dtype: MPSDataType = MPSDataType.mps_data_type_float32


##
## Clamp ops
##
@dataclass
class MPSClamp(MPSNode1x1):
    pass


@dataclass
class MPSWhere(MPSNode3x1):
    pass


##
## Reduce ops
##
@dataclass
class MPSMean(MPSNode1x1):
    num_dims: int = 0
    dims: List[int] = field(default_factory=list)
    keep_dims: bool = False


##
## Indexing ops
##
@dataclass
class MPSIndexSelect(MPSNode1x1):
    dim: int = 0
    index_id: int = -1


@dataclass
class MPSEmbedding(MPSNode2x1):
    padding_idx: int = -1
    scale_grad_by_freq: bool = False
    sparse: bool = False


@dataclass
class MPSIndexTensor(MPSNode1x1):
    indices_id: List[int] = field(default_factory=list)


@dataclass
class MPSIndexPut(MPSNode1x1):
    indices_id: List[int] = field(default_factory=list)
    values_shape: List[int] = field(default_factory=list)
    values_id: int = -1


@dataclass
class MPSScatter(MPSNode1x1):
    dim: int = 0
    idx_id: int = -1
    src_id: int = -1


##
## Shape ops
##
@dataclass
class MPSPermute(MPSNode1x1):
    num_dims: int = 0
    perm: List[int] = field(default_factory=list)


@dataclass
class MPSView(MPSNode1x1):
    num_dims: int = 0
    shape: List[int] = field(default_factory=list)


@dataclass
class MPSExpand(MPSNode1x1):
    num_dims: int = 0
    shape: List[int] = field(default_factory=list)


@dataclass
class MPSCat:
    input_ids: List[int]
    output_id: int
    dim: int


@dataclass
class MPSSqueeze(MPSNode1x1):
    dims: List[int] = field(default_factory=list)


@dataclass
class MPSUnsqueeze(MPSNode1x1):
    dim: int = 0


@dataclass
class MPSSelect(MPSNode1x1):
    dim: int = 0
    index: int = 0


@dataclass
class MPSSlice(MPSNode1x1):
    dim: int = 0
    start: int = -1
    end: int = -1
    step: int = 1


@dataclass
class MPSPixelShuffle(MPSNode1x1):
    upscale_factor: int = 1


@dataclass
class MPSSplitWithSizes:
    input1_id: int
    output_ids: List[int]
    split_sizes: List[int]
    dim: int


@dataclass
class MPSCast(MPSNode1x1):
    dtype: MPSDataType


##
## Convolution ops
##


@dataclass
class MPSConv2D(MPSConv):
    pass


@dataclass
class MPSDepthwiseConv2D(MPSConv):
    pass


##
## Comparison Ops
##
class MPSEq(MPSNode2x1):
    pass


class MPSNe(MPSNode2x1):
    pass


class MPSGe(MPSNode2x1):
    pass


class MPSGt(MPSNode2x1):
    pass


class MPSLe(MPSNode2x1):
    pass


class MPSLt(MPSNode2x1):
    pass


##
## Normalization op
##
@dataclass
class MPSBatchNorm:
    input_id: int
    mean_id: int
    var_id: int
    weight_id: int
    bias_id: int
    momentum: float
    epsilon: float
    output1_id: int
    output2_id: int
    output3_id: int


@dataclass
class MPSLayerNorm:
    input1_id: int
    normalized_shape: List[int]
    weight_id: int
    bias_id: int
    eps: float
    output1_id: int
    output2_id: int
    output3_id: int


##
## Pooling ops
##


@dataclass
class MPSMaxPool2DWithIndices(MPSPooling2D):
    pass


@dataclass
class MPSAvgPool2D(MPSPooling2D):
    pass


##
## Pad ops
##
@dataclass
class MPSConstantPadND(MPSNode1x1):
    pad: List[int] = field(default_factory=list)
    value: float = 0.0


##
## Range ops
##
@dataclass
class MPSArange:
    output_id: int
    start: float
    end: float
    step: float
    dtype: MPSDataType


##
## Quant - Dequant ops
##
@dataclass
class MPSDequantizePerChannelGroup(MPSDequantizeNode):
    quant_min: int
    quant_max: int
    dtype: MPSDataType
    group_size: int
    output_dtype: MPSDataType


MPSNodeUnion = Union[
    # Activation ops
    MPSHardTanh,
    MPSReLU,
    MPSGELU,
    MPSLeakyReLU,
    MPSSoftmax,
    # Binary ops
    MPSAdd,
    MPSSub,
    MPSMul,
    MPSDiv,
    MPSMin,
    MPSMax,
    MPSPow,
    MPSRemainder,
    MPSAtan2,
    MPSBitwiseAnd,
    MPSBitwiseOr,
    MPSBitwiseXor,
    MPSMinimum,
    # Unary ops
    MPSExp,
    MPSExp2,
    MPSReciprocal,
    MPSSqrt,
    MPSNeg,
    MPSLog,
    MPSLog10,
    MPSLog2,
    MPSErf,
    MPSFloor,
    MPSCeil,
    MPSRsqrt,
    MPSSigmoid,
    MPSSin,
    MPSSign,
    MPSCos,
    MPSTan,
    MPSAbs,
    MPSAsin,
    MPSAcos,
    MPSAtan,
    MPSSinh,
    MPSCosh,
    MPSTanh,
    MPSAsinh,
    MPSAcosh,
    MPSAtanh,
    MPSBitwiseNot,
    MPSIsnan,
    MPSIsinf,
    MPSRound,
    MPSLogicalNot,
    # Linear algebra ops
    MPSMatMul,
    MPSAddmm,
    # Constant ops
    MPSFull,
    MPSFullLike,
    # Clamp ops
    MPSClamp,
    MPSWhere,
    # Reduce ops
    MPSMean,
    # Indexing ops
    MPSIndexSelect,
    MPSEmbedding,
    MPSIndexTensor,
    MPSIndexPut,
    MPSScatter,
    # Shape ops
    MPSPermute,
    MPSView,
    MPSExpand,
    MPSCat,
    MPSSqueeze,
    MPSUnsqueeze,
    MPSSelect,
    MPSSlice,
    MPSPixelShuffle,
    MPSSplitWithSizes,
    MPSCast,
    # Convolution ops
    MPSConv2D,
    MPSDepthwiseConv2D,
    # Comparison ops
    MPSEq,
    MPSNe,
    MPSGe,
    MPSGt,
    MPSLe,
    MPSLt,
    # Normalization ops
    MPSBatchNorm,
    MPSLayerNorm,
    # Pooling ops
    MPSMaxPool2DWithIndices,
    MPSAvgPool2D,
    # Pad ops
    MPSConstantPadND,
    # Range ops
    MPSArange,
    # Quant-Dequant ops
    MPSDequantizePerChannelGroup,
]


@dataclass
class MPSNode:
    mpsnode_union: "MPSNodeUnion"
    min_max: Optional[MPSMinMax] = None


@dataclass
class Buffer:
    storage: bytes


@dataclass
class MPSTensor:
    datatype: MPSDataType
    num_dims: int
    dims: List[int]
    constant_buffer_size: int
    constant_buffer: Buffer  # deprecated
    segment_offset: int = 0


@dataclass
class DataSegment:
    offset: int
    size: int


@dataclass
class MPSGraph:
    version: str
    mps_nodes: List[MPSNode]
    mps_values: List[MPSTensor]
    input_ids: List[int]
    output_ids: List[int]
    constant_ids: List[int]
    graph_type: OpType
    constant_segment: DataSegment
