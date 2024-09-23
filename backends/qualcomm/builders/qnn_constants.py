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
class OpConvert:
    op_name: str = "Convert"


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
class OpElementWiseAdd:
    op_name: str = "ElementWiseAdd"


@dataclass(init=False, frozen=True)
class OpElementWiseCeil:
    op_name = "ElementWiseCeil"


@dataclass(init=False, frozen=True)
class OpElementWiseDivide:
    op_name: str = "ElementWiseDivide"


@dataclass(init=False, frozen=True)
class OpElementWiseMultiply:
    op_name: str = "ElementWiseMultiply"


@dataclass(init=False, frozen=True)
class OpElementWiseNeuron:
    op_name: str = "ElementWiseNeuron"
    param_operation: str = "operation"
    param_alpha: str = "alpha"
    param_beta: str = "beta"


@dataclass(init=False, frozen=True)
class OpElementWisePower:
    op_name: str = "ElementWisePower"


@dataclass(init=False, frozen=True)
class OpElementWiseRsqrt:
    op_name: str = "ElementWiseRsqrt"


@dataclass(init=False, frozen=True)
class OpElementWiseSubtract:
    op_name = "ElementWiseSubtract"


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
class OpGatherND:
    op_name: str = "GatherNd"
    param_batch_dims: str = "batch_dims"


@dataclass(init=False, frozen=True)
class OpGelu:
    op_name: str = "Gelu"


@dataclass(init=False, frozen=True)
class OpHardSwish:
    op_name: str = "HardSwish"


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
class OpReduceMean:
    op_name: str = "ReduceMean"
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
class OpSqrt:
    op_name: str = "ElementWiseSquareRoot"


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
class OpTranspose:
    op_name: str = "Transpose"
    param_perm: str = "perm"
