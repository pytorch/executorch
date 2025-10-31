# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import IntEnum, unique

QNN_OP_PACKAGE_NAME_QTI_AISW = "qti.aisw"

# Below constants should be same as those in QNN headers.
# Maybe someday we should expose these constants by pybind
# instead of replicating them here.


@dataclass(init=False, frozen=True)
class OpArgmax:
    op_name: str = "Argmax"
    param_axis: str = "axis"
    param_keep_dims: str = "keep_dims"


@dataclass(init=False, frozen=True)
class OpArgmin:
    op_name: str = "Argmin"
    param_axis: str = "axis"
    param_keep_dims: str = "keep_dims"


@dataclass(init=False, frozen=True)
class OpBatchnorm:
    op_name: str = "Batchnorm"


@dataclass(init=False, frozen=True)
class OpCast:
    op_name: str = "Cast"


@dataclass(init=False, frozen=True)
class OpConcat:
    op_name: str = "Concat"
    param_axis: str = "axis"


@dataclass(init=False, frozen=True)
class OpContextLoader:
    namespace: str = "qaisw"
    meta_ctx_bin: str = "qnn_context_binary"


@dataclass(init=False, frozen=True)
class OpConv2d:
    op_name: str = "Conv2d"
    param_stride: str = "stride"
    param_pad_amount: str = "pad_amount"
    param_group: str = "group"
    param_dilation: str = "dilation"


@dataclass(init=False, frozen=True)
class OpConv3d:
    op_name: str = "Conv3d"
    param_stride: str = "stride"
    param_pad_amount: str = "pad_amount"
    param_group: str = "group"
    param_dilation: str = "dilation"


@dataclass(init=False, frozen=True)
class OpConvert:
    op_name: str = "Convert"


@dataclass(init=False, frozen=True)
class OpCumulativeSum:
    op_name = "CumulativeSum"
    param_axis = "axis"
    param_exclusive = "exclusive"
    param_reverse = "reverse"


@dataclass(init=False, frozen=True)
class OpDepthToSpace:
    op_name: str = "DepthToSpace"
    param_block_size: str = "block_size"
    param_mode: str = "mode"

    @unique
    class Mode(IntEnum):
        DCR = 0
        CRD = 1


@dataclass(init=False, frozen=True)
class OpDepthWiseConv2d:
    op_name: str = "DepthWiseConv2d"
    param_stride: str = "stride"
    param_pad_amount: str = "pad_amount"
    param_dilation: str = "dilation"


@dataclass(init=False, frozen=True)
class OpDequantize:
    op_name: str = "Dequantize"


@dataclass(init=False, frozen=True)
class OpElementWiseAbs:
    op_name: str = "ElementWiseAbs"


@dataclass(init=False, frozen=True)
class OpElementWiseAdd:
    op_name: str = "ElementWiseAdd"


@dataclass(init=False, frozen=True)
class OpElementWiseAnd:
    op_name: str = "ElementWiseAnd"


@dataclass(init=False, frozen=True)
class OpElementWiseAsin:
    op_name: str = "ElementWiseAsin"


@dataclass(init=False, frozen=True)
class OpElementWiseAtan:
    op_name: str = "ElementWiseAtan"


@dataclass(init=False, frozen=True)
class OpElementWiseBinary:
    op_name: str = "ElementWiseBinary"
    param_operation: str = "operation"


@dataclass(init=False, frozen=True)
class OpElementWiseCeil:
    op_name = "ElementWiseCeil"


@dataclass(init=False, frozen=True)
class OpElementWiseCos:
    op_name: str = "ElementWiseCos"


@dataclass(init=False, frozen=True)
class OpElementWiseDivide:
    op_name: str = "ElementWiseDivide"


@dataclass(init=False, frozen=True)
class OpElementWiseExp:
    op_name: str = "ElementWiseExp"


@dataclass(init=False, frozen=True)
class OpElementWiseEqual:
    op_name: str = "ElementWiseEqual"


@dataclass(init=False, frozen=True)
class OpElementWiseFloor:
    op_name: str = "ElementWiseFloor"


@dataclass(init=False, frozen=True)
class OpElementWiseGreater:
    op_name: str = "ElementWiseGreater"


@dataclass(init=False, frozen=True)
class OpElementWiseGreaterEqual:
    op_name: str = "ElementWiseGreaterEqual"


@dataclass(init=False, frozen=True)
class OpElementWiseLess:
    op_name: str = "ElementWiseLess"


@dataclass(init=False, frozen=True)
class OpElementWiseLessEqual:
    op_name: str = "ElementWiseLessEqual"


@dataclass(init=False, frozen=True)
class OpElementWiseLog:
    op_name: str = "ElementWiseLog"


@dataclass(init=False, frozen=True)
class OpElementWiseMaximum:
    op_name: str = "ElementWiseMaximum"


@dataclass(init=False, frozen=True)
class OpElementWiseMinimum:
    op_name: str = "ElementWiseMinimum"


@dataclass(init=False, frozen=True)
class OpElementWiseMultiply:
    op_name: str = "ElementWiseMultiply"


@dataclass(init=False, frozen=True)
class OpElementWiseNeg:
    op_name: str = "ElementWiseNeg"


@dataclass(init=False, frozen=True)
class OpElementWiseNeuron:
    op_name: str = "ElementWiseNeuron"
    param_operation: str = "operation"
    param_alpha: str = "alpha"
    param_beta: str = "beta"


@dataclass(init=False, frozen=True)
class OpElementWiseNot:
    op_name: str = "ElementWiseNot"


@dataclass(init=False, frozen=True)
class OpElementWiseNotEqual:
    op_name: str = "ElementWiseNotEqual"


@dataclass(init=False, frozen=True)
class OpElementWiseOr:
    op_name: str = "ElementWiseOr"


@dataclass(init=False, frozen=True)
class OpElementWisePower:
    op_name: str = "ElementWisePower"


@dataclass(init=False, frozen=True)
class OpElementWiseRound:
    op_name: str = "ElementWiseRound"


@dataclass(init=False, frozen=True)
class OpElementWiseRsqrt:
    op_name: str = "ElementWiseRsqrt"


@dataclass(init=False, frozen=True)
class OpElementWiseSin:
    op_name: str = "ElementWiseSin"


@dataclass(init=False, frozen=True)
class OpElementWiseSelect:
    op_name = "ElementWiseSelect"


@dataclass(init=False, frozen=True)
class OpElementWiseSign:
    op_name: str = "ElementWiseSign"


@dataclass(init=False, frozen=True)
class OpElementWiseSquareRoot:
    op_name = "ElementWiseSquareRoot"


@dataclass(init=False, frozen=True)
class OpElementWiseSubtract:
    op_name = "ElementWiseSubtract"


@dataclass(init=False, frozen=True)
class OpElementWiseXor:
    op_name: str = "ElementWiseXor"


@dataclass(init=False, frozen=True)
class OpElu:
    op_name: str = "Elu"
    param_alpha: str = "alpha"


@dataclass(init=False, frozen=True)
class OpExpandDims:
    op_name: str = "ExpandDims"
    param_axis: str = "axis"


@dataclass(init=False, frozen=True)
class OpFullyConnected:
    op_name: str = "FullyConnected"
    param_keep_dims: str = "keep_dims"


@dataclass(init=False, frozen=True)
class OpGather:
    op_name: str = "Gather"
    param_axis: str = "axis"


@dataclass(init=False, frozen=True)
class OpGatherElements:
    op_name: str = "GatherElements"
    param_axis: str = "axis"


@dataclass(init=False, frozen=True)
class OpGatherND:
    op_name: str = "GatherNd"
    param_batch_dims: str = "batch_dims"


@dataclass(init=False, frozen=True)
class OpGelu:
    op_name: str = "Gelu"


class OpGroupNorm:
    op_name: str = "GroupNorm"
    param_epsilon = "epsilon"
    param_group = "group"


@dataclass(init=False, frozen=True)
class OpHardSwish:
    op_name: str = "HardSwish"


@dataclass(init=False, frozen=True)
class OpInstanceNorm:
    op_name: str = "InstanceNorm"
    param_epsilon = "epsilon"
    param_mode = "mode"
    param_normalize_variance = "normalize_variance"
    param_region = "region"


@dataclass(init=False, frozen=True)
class OpLayerNorm:
    op_name: str = "LayerNorm"
    param_epsilon = "epsilon"
    param_axes = "axes"


@dataclass(init=False, frozen=True)
class OpLogSoftmax:
    op_name: str = "LogSoftmax"
    param_axis: str = "axis"
    param_beta: str = "beta"


@dataclass(init=False, frozen=True)
class OpMatMul:
    op_name: str = "MatMul"
    param_transpose_in0: str = "transpose_in0"
    param_transpose_in1: str = "transpose_in1"


@dataclass(init=False, frozen=True)
class OpPack:
    op_name: str = "Pack"
    param_axis: str = "axis"


@dataclass(init=False, frozen=True)
class OpPad:
    op_name: str = "Pad"
    param_scheme: str = "scheme"
    param_pad_amount: str = "pad_amount"
    param_pad_constant_value: str = "pad_constant_value"

    @unique
    class Scheme(IntEnum):
        CONSTANT = 0
        MIRROR_SYMMETRIC = 1
        MIRROR_REFLECT = 2
        EDGE = 3


@dataclass(init=False, frozen=True)
class OpPoolAvg2d:
    op_name: str = "PoolAvg2d"
    param_filter_size: str = "filter_size"
    param_stride: str = "stride"
    param_pad_amount: str = "pad_amount"
    param_count_pad_for_edges: str = "count_pad_for_edges"
    param_rounding_mode: str = "rounding_mode"

    @unique
    class RoundingMode(IntEnum):
        FLOOR = 0
        CEIL = 1


@dataclass(init=False, frozen=True)
class OpPoolAvg3d:
    op_name: str = "PoolAvg3d"
    param_filter_size: str = "filter_size"
    param_stride: str = "stride"
    param_pad_amount: str = "pad_amount"
    param_count_pad_for_edges: str = "count_pad_for_edges"
    param_rounding_mode: str = "rounding_mode"

    @unique
    class RoundingMode(IntEnum):
        FLOOR = 0
        CEIL = 1


@dataclass(init=False, frozen=True)
class OpPoolMax2d:
    op_name: str = "PoolMax2d"
    param_filter_size: str = "filter_size"
    param_stride: str = "stride"
    param_pad_amount: str = "pad_amount"
    param_rounding_mode: str = "rounding_mode"

    @unique
    class RoundingMode(IntEnum):
        FLOOR = 0
        CEIL = 1


@dataclass(init=False, frozen=True)
class OpPRelu:
    op_name: str = "Prelu"


@dataclass(init=False, frozen=True)
class OpQuantize:
    op_name: str = "Quantize"


@dataclass(init=False, frozen=True)
class OpReduceMax:
    op_name: str = "ReduceMax"
    param_axes: str = "axes"
    param_keep_dims: str = "keep_dims"


@dataclass(init=False, frozen=True)
class OpReduceMean:
    op_name: str = "ReduceMean"
    param_axes: str = "axes"
    param_keep_dims: str = "keep_dims"


@dataclass(init=False, frozen=True)
class OpReduceMin:
    op_name: str = "ReduceMin"
    param_axes: str = "axes"
    param_keep_dims: str = "keep_dims"


@dataclass(init=False, frozen=True)
class OpReduceSum:
    op_name: str = "ReduceSum"
    param_axes: str = "axes"
    param_keep_dims: str = "keep_dims"


@dataclass(init=False, frozen=True)
class OpRelu:
    op_name: str = "Relu"


@dataclass(init=False, frozen=True)
class OpReluMinMax:
    op_name: str = "ReluMinMax"
    param_min_value: str = "min_value"
    param_max_value: str = "max_value"


@dataclass(init=False, frozen=True)
class OpReshape:
    op_name: str = "Reshape"


@dataclass(init=False, frozen=True)
class OpResize:
    op_name: str = "Resize"
    param_exclude_outside: str = "exclude_outside"
    param_transformation_mode: str = "transformation_mode"
    param_interpolation_mode: str = "interpolation_mode"
    param_nearest_mode: str = "nearest_mode"
    param_cubic_coeff: str = "cubic_coeff"


@dataclass(init=False, frozen=True)
class OpResizeBilinear:
    op_name: str = "ResizeBilinear"
    param_align_corners: str = "align_corners"
    param_half_pixel_centers: str = "half_pixel_centers"


@dataclass(init=False, frozen=True)
class OpResizeNearestNeighbor:
    op_name: str = "ResizeNearestNeighbor"
    param_align_corners: str = "align_corners"
    param_half_pixel_centers: str = "half_pixel_centers"


@dataclass(init=False, frozen=True)
class OpRmsNorm:
    op_name: str = "RmsNorm"
    param_epsilon: str = "epsilon"
    param_axes: str = "axes"


@dataclass(init=False, frozen=True)
class OpScatterNd:
    op_name: str = "ScatterNd"
    param_reduction: str = "reduction"


@dataclass(init=False, frozen=True)
class OpSigmoid:
    op_name: str = "Sigmoid"


@dataclass(init=False, frozen=True)
class OpSoftmax:
    op_name: str = "Softmax"
    param_axis: str = "axis"
    param_beta: str = "beta"


@dataclass(init=False, frozen=True)
class OpSpaceToDepth:
    op_name: str = "SpaceToDepth"
    param_block_size: str = "block_size"
    param_mode: str = "mode"

    @unique
    class Mode(IntEnum):
        DCR = 0
        CRD = 1


class OpSplit:
    op_name: str = "Split"
    param_axis: str = "axis"
    param_split_index: str = "split_index"


@dataclass(init=False, frozen=True)
class OpSqueeze:
    op_name: str = "Squeeze"


@dataclass(init=False, frozen=True)
class OpStridedSlice:
    op_name: str = "StridedSlice"
    param_ranges: str = "ranges"
    param_begin_mask: str = "begin_mask"
    param_end_mask: str = "end_mask"
    param_shrink_axes: str = "shrink_axes"
    param_new_axes_mask: str = "new_axes_mask"


@dataclass(init=False, frozen=True)
class OpTanh:
    op_name: str = "Tanh"


@dataclass(init=False, frozen=True)
class OpTile:
    op_name: str = "Tile"
    param_multiples: str = "multiples"


@dataclass(init=False, frozen=True)
class OpTopK:
    op_name: str = "TopK"
    param_k: str = "k"
    param_largest: str = "largest"


@dataclass(init=False, frozen=True)
class OpTranspose:
    op_name: str = "Transpose"
    param_perm: str = "perm"


@dataclass(init=False, frozen=True)
class OpTransposeConv2d:
    op_name: str = "TransposeConv2d"
    param_stride: str = "stride"
    param_pad_amount: str = "pad_amount"
    param_group: str = "group"
    param_output_padding: str = "output_padding"


@dataclass(init=False, frozen=True)
class OpTransposeConv3d:
    op_name: str = "TransposeConv3d"
    param_stride: str = "stride"
    param_pad_amount: str = "pad_amount"
    param_group: str = "group"
    param_output_padding: str = "output_padding"


@dataclass(init=False, frozen=True)
class OpUnpack:
    op_name: str = "UnPack"
    param_axis: str = "axis"
