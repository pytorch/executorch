# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import executorch.backends.vulkan.custom_ops_lib  # noqa
import executorch.backends.vulkan.utils as utils
import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload

######################
## OpFeatures class ##
######################


def allow_node(node: torch.fx.Node) -> bool:
    return True


class OpFeatures:
    __slots__ = [
        # Dtype-related slots:
        "inputs_dtypes",  # DtypeSetList for input tensor dtype constraints
        "outputs_dtypes",  # DtypeSetList for output tensor dtype constraints
        # Storage-related slots:
        # Sets of possible (storage types, memory layouts) to use for the input tensor(s)
        "inputs_storage",
        # Sets of possible (storage types, memory layouts) to use for the output tensor(s)
        "outputs_storage",
        # bool indicating if the operator has a resize function, which allows it to
        # support models with dynamic shape
        "supports_resize",
        # bool indicating if the operator supports tensors with more than 4 dimensions
        "supports_highdim",
        # bool indicating if the operator handles its own prepacking. If this is True,
        # then the insert_prepack_nodes pass will not insert prepack nodes for the args
        # of the op.
        "supports_prepacking",
        # Optional check function used during partitioning to determine if a node's
        # inputs are supported by the operator implementation.
        "are_node_inputs_supported_fn",
        # Optional function to determine valid representation sets for input and outputs
        # once a node's actual inputs are known.
        "pick_io_storage_fn",
    ]

    def __init__(
        self,
        inputs_dtypes: Optional[Union[utils.DtypeSet, List[utils.DtypeSet]]] = None,
        outputs_dtypes: Optional[Union[utils.DtypeSet, List[utils.DtypeSet]]] = None,
        inputs_storage: Optional[
            Union[utils.TensorRepSet, List[utils.TensorRepSet]]
        ] = None,
        outputs_storage: Optional[
            Union[utils.TensorRepSet, List[utils.TensorRepSet]]
        ] = None,
        supports_resize: bool = False,
        supports_highdim: bool = False,
        supports_prepacking: bool = False,
        are_node_inputs_supported_fn: Optional[Callable] = allow_node,
        pick_io_storage_fn: Optional[Callable] = None,
    ):
        # Dtype initialization
        self.inputs_dtypes: utils.DtypeSetList = utils.DtypeSetList(
            inputs_dtypes if inputs_dtypes is not None else utils.ALL_T
        )
        self.outputs_dtypes: utils.DtypeSetList = utils.DtypeSetList(
            outputs_dtypes if outputs_dtypes is not None else self.inputs_dtypes[0]
        )

        # Storage initialization
        self.inputs_storage: utils.TensorRepSetList = utils.TensorRepSetList(
            inputs_storage if inputs_storage is not None else []
        )
        self.outputs_storage: utils.TensorRepSetList = utils.TensorRepSetList(
            outputs_storage if outputs_storage is not None else []
        )

        # If output storage is not set, assume that it is derived from the first input
        if self.outputs_storage.any_is_empty():
            self.outputs_storage = utils.TensorRepSetList(self.inputs_storage[0])

        self.supports_resize = supports_resize
        self.supports_highdim = supports_highdim
        self.supports_prepacking = supports_prepacking

        self.are_node_inputs_supported_fn = are_node_inputs_supported_fn
        self.pick_io_storage_fn = pick_io_storage_fn

    def check_dtypes(self, node: torch.fx.Node) -> Tuple[bool, str]:
        """
        Check if all tensor inputs/outputs have dtypes supported by this operator.
        Returns (is_valid, reason_string).
        """
        return utils.check_node_dtypes(
            node,
            self.inputs_dtypes,
            self.outputs_dtypes,
        )

    def make_op_repsets(
        self,
        op_node: torch.fx.Node,
        texture_limits: utils.ImageExtents = utils.DEFAULT_TEXTURE_LIMITS,
    ) -> utils.OpRepSets:
        inputs_storage = self.inputs_storage
        outputs_storage = self.outputs_storage
        if self.pick_io_storage_fn is not None:
            i_storage, o_storage = self.pick_io_storage_fn(op_node)
            inputs_storage = utils.TensorRepSetList(i_storage)
            outputs_storage = utils.TensorRepSetList(o_storage)

        return utils.OpRepSets(inputs_storage, outputs_storage, op_node, texture_limits)


#######################
## Operator Registry ##
#######################

OpKey = Union[str, torch._ops.OpOverload, EdgeOpOverload]

vulkan_supported_ops: Dict[OpKey, OpFeatures] = {}


def update_features(aten_op):
    def features_decorator(fn: Callable):
        def update_features_impl(op: OpKey):
            if op in vulkan_supported_ops:
                raise RuntimeError(f"[Vulkan delegate] duplicate registration of {op}!")
            vulkan_supported_ops[op] = fn()

        if isinstance(aten_op, list):
            for op in aten_op:
                update_features_impl(op)
        else:
            update_features_impl(aten_op)

        return fn

    return features_decorator


# =============================================================================
# Ephemeral Operators (no C++ dispatch - handled symbolically)
# =============================================================================


@update_features(
    [
        operator.getitem,
        # Symbolic integer ops (SymIntOps.cpp - symbolic handling)
        torch.ops.aten.sym_size.int,
        operator.add,
        operator.sub,
        operator.lt,
        operator.gt,
        operator.ge,
        operator.le,
        operator.eq,
        # Guard and assert ops
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
    ]
)
def register_ephemeral_ops():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
    )


# =============================================================================
# UnaryOp.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.abs.default,
        exir_ops.edge.aten.cos.default,
        exir_ops.edge.aten.exp.default,
        exir_ops.edge.aten.gelu.default,
        exir_ops.edge.aten.hardshrink.default,
        exir_ops.edge.aten.hardtanh.default,
        exir_ops.edge.aten.neg.default,
        exir_ops.edge.aten.relu.default,
        exir_ops.edge.aten.sigmoid.default,
        exir_ops.edge.aten.sin.default,
        exir_ops.edge.aten.sqrt.default,
        exir_ops.edge.aten.rsqrt.default,
        exir_ops.edge.aten.tanh.default,
        exir_ops.edge.aten.round.default,
        exir_ops.edge.aten.leaky_relu.default,
    ]
)
def register_unaryop_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
    )


@update_features(exir_ops.edge.aten.clamp.default)
def register_clamp():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_T,
        supports_resize=True,
    )


# =============================================================================
# BinaryOp.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.minimum.default,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.div.Tensor_mode,
        exir_ops.edge.aten.pow.Tensor_Tensor,
        exir_ops.edge.aten.eq.Tensor,
        exir_ops.edge.aten.lt.Tensor,
        exir_ops.edge.aten.le.Tensor,
        exir_ops.edge.aten.gt.Tensor,
        exir_ops.edge.aten.ge.Tensor,
    ]
)
def register_binaryop_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# BinaryScalarOp.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.pow.Tensor_Scalar)
def register_pow_tensor_scalar():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# ToCopy.cpp
# =============================================================================


@update_features(exir_ops.edge.aten._to_copy.default)
def register_to_copy():
    def check_to_copy_node(node: torch.fx.Node) -> bool:
        # Only single-arg _to_copy is supported
        return len(node.args) == 1

    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_T,
        outputs_dtypes=utils.FP_T,
        supports_resize=True,
        are_node_inputs_supported_fn=check_to_copy_node,
    )


# =============================================================================
# Softmax.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten._log_softmax.default,
        exir_ops.edge.aten._softmax.default,
    ]
)
def register_softmax_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
    )


# =============================================================================
# MatMul.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.bmm.default,
        exir_ops.edge.aten.mm.default,
    ]
)
def register_matmul_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_prepacking=True,
    )


# =============================================================================
# Linear.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.addmm.default,
        exir_ops.edge.aten.linear.default,
    ]
)
def register_linear_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_prepacking=True,
    )


# =============================================================================
# QuantizedLinearQCSNW.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten._weight_int8pack_mm.default,
        exir_ops.edge.et_vk.linear_qcs4w.default,
    ]
)
def register_quantizedlinearqcsnw_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_prepacking=True,
    )


# =============================================================================
# QuantizedLinear.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.et_vk.linear_q8ta_q8csw.default,
        exir_ops.edge.et_vk.linear_q4gsw.default,
    ]
)
def register_quantizedlinear_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        inputs_dtypes=utils.FP_T,
        supports_prepacking=True,
    )


@update_features(exir_ops.edge.et_vk.linear_dq8ca_q4gsw.default)
def register_linear_dq8ca_q4gsw():
    return OpFeatures(
        inputs_storage=[
            utils.CONTIGUOUS_ANY,  # input
            utils.WIDTH_PACKED_TEXTURE,  # input_scale
            utils.WIDTH_PACKED_TEXTURE,  # input_zero_point
            utils.NO_STORAGE,  # weight (prepacked)
            utils.NO_STORAGE,  # weight_sums (prepacked)
            utils.NO_STORAGE,  # weight_scales (prepacked)
            utils.NO_STORAGE,  # group_size (scalar)
            utils.NO_STORAGE,  # bias (prepacked)
        ],
        inputs_dtypes=utils.FP_T,
        supports_prepacking=True,
    )


# =============================================================================
# QuantizeDequantize.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_token.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_token.default,
    ]
)
def register_quantizedequantize_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_BUFFER,
        supports_resize=True,
    )


@update_features(
    [
        exir_ops.edge.torchao.quantize_affine.default,
        exir_ops.edge.torchao.dequantize_affine.default,
    ]
)
def register_torchao_quantize_dequantize():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_BUFFER,
        supports_resize=True,
    )


# =============================================================================
# Q8taQuantizeDequantize.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
    ]
)
def register_quantize_per_tensor():
    return OpFeatures(
        inputs_storage=[
            utils.CHANNELS_PACKED_TEXTURE_OR_CONTIGUOUS_BUFFER,
        ],
        outputs_storage=[
            utils.PACKED_INT8_BUFFER,
        ],
    )


@update_features(
    [
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
    ]
)
def register_dequantize_per_tensor():
    return OpFeatures(
        inputs_storage=[
            utils.PACKED_INT8_BUFFER,
        ],
        outputs_storage=[
            utils.CHANNELS_PACKED_TEXTURE_OR_CONTIGUOUS_BUFFER,
        ],
    )


# =============================================================================
# ChooseQParams.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.quantized_decomposed.choose_qparams.tensor,
        exir_ops.edge.quantized_decomposed.choose_qparams_per_token_asymmetric.default,
    ]
)
def register_chooseqparams_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_BUFFER,
        supports_resize=True,
    )


@update_features(exir_ops.edge.torchao.choose_qparams_affine.default)
def register_torchao_choose_qparams_affine():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        outputs_storage=[
            utils.WIDTH_PACKED_TEXTURE,  # scales
            utils.WIDTH_PACKED_TEXTURE,  # zero_points
        ],
        supports_resize=True,
    )


# =============================================================================
# Q8taBinary.cpp
# =============================================================================


@update_features(exir_ops.edge.et_vk.q8ta_add.default)
def register_q8ta_add():
    return OpFeatures(
        inputs_storage=utils.PACKED_INT8_BUFFER,
        supports_resize=False,
    )


def get_dims_reduced(node: torch.fx.Node) -> Union[int, List[int]]:
    ndim = utils.ndim_of(node.args[0])
    assert ndim is not None
    dims_reduced = None
    if len(node.args) >= 2:
        dims_reduced = node.args[1]

    # If dim_list is None, return a list containing all the dims of the tensor
    if dims_reduced is None:
        dims_reduced = list(range(ndim))

        # Special case for reducing tensors with shape [1, N] - this is equivalent to
        # reducing the last dim.
        if utils.is_unsqueezed_vector(node) and ndim == 2:
            dims_reduced = 1

    if isinstance(dims_reduced, (list, tuple)) and len(dims_reduced) == 1:
        dims_reduced = dims_reduced[0]

    assert isinstance(dims_reduced, (int, list, tuple))
    return utils.normalize_dims(dims_reduced, ndim)


def get_keepdim_setting(node: torch.fx.Node) -> bool:
    for arg in node.args:
        if isinstance(arg, bool):
            return arg

    # Assume false by default
    return False


def is_reduce_node_supported_by_per_row_impl(node: torch.fx.Node) -> bool:
    """
    Checks if a reduction node is supported by the Vulkan backend's reduce per row
    special case implementation.
    """
    input_ndim = utils.ndim_of(node.args[0])
    assert input_ndim is not None
    dims_reduced = get_dims_reduced(node)

    return dims_reduced == input_ndim - 1


def is_reduce_node_supported_by_general_impl(node: torch.fx.Node) -> bool:
    dims_reduced = get_dims_reduced(node)
    # Only 1D and 2D reductions are supported at the moment.
    if isinstance(dims_reduced, (list, tuple)) and len(dims_reduced) > 2:
        return False

    keepdim = get_keepdim_setting(node)
    # keepdim = False is not supported yet for general implementation
    if isinstance(keepdim, bool) and not keepdim:
        return False

    return True


def is_reduce_node_supported(node: torch.fx.Node) -> bool:
    return is_reduce_node_supported_by_per_row_impl(
        node
    ) or is_reduce_node_supported_by_general_impl(node)


def pick_storage_for_reduce(node: torch.fx.Node):
    inputs_storage = utils.NO_STORAGE
    outputs_storage = utils.NO_STORAGE

    ndim = utils.ndim_of(node.args[0])
    dim_list = get_dims_reduced(node)

    if is_reduce_node_supported_by_general_impl(node):
        inputs_storage = inputs_storage.make_union(utils.ANY_TEXTURE)
        outputs_storage = inputs_storage

    # For 1D reductions of the last dim, a special reduce per row case is implemented
    # for buffer backed tensors.
    if is_reduce_node_supported_by_per_row_impl(node):
        inputs_storage = inputs_storage.make_union(utils.CONTIGUOUS_BUFFER)
        outputs_storage = inputs_storage
        return inputs_storage, outputs_storage

    # For 2D reductions, the packed dimension cannot be one of the reduced dims
    if isinstance(dim_list, (list, tuple)) and len(dim_list) == 2:
        # pyre-ignore[6]
        reduce_dim1_whcn = utils.nchw_dim_to_whcn_dim(dim_list[0], ndim)
        # pyre-ignore[6]
        reduce_dim2_whcn = utils.nchw_dim_to_whcn_dim(dim_list[1], ndim)

        possible_packed_dims = {0, 1, 2}
        possible_packed_dims.discard(reduce_dim1_whcn)
        possible_packed_dims.discard(reduce_dim2_whcn)

        packed_dim = possible_packed_dims.pop()
        assert packed_dim in [0, 1, 2]

        if packed_dim == 0:
            inputs_storage = utils.WIDTH_PACKED_TEXTURE
            outputs_storage = utils.WIDTH_PACKED_TEXTURE
        elif packed_dim == 1:
            inputs_storage = utils.HEIGHT_PACKED_TEXTURE
            outputs_storage = utils.HEIGHT_PACKED_TEXTURE
        else:
            inputs_storage = utils.CHANNELS_PACKED_TEXTURE
            outputs_storage = utils.CHANNELS_PACKED_TEXTURE

    return inputs_storage, outputs_storage


# =============================================================================
# Reduce.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.sum.dim_IntList,
        exir_ops.edge.aten.amax.default,
        exir_ops.edge.aten.amin.default,
    ]
)
def register_reduce_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_highdim=True,
        are_node_inputs_supported_fn=is_reduce_node_supported,
        pick_io_storage_fn=pick_storage_for_reduce,
    )


# =============================================================================
# ArgReduce.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.argmax.default,
        exir_ops.edge.aten.argmin.default,
    ]
)
def register_argreduce_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_highdim=True,
        are_node_inputs_supported_fn=is_reduce_node_supported,
        pick_io_storage_fn=pick_storage_for_reduce,
    )


# =============================================================================
# Pool.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.max_pool2d.default,
    ]
)
def register_pool_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
    )


@update_features(exir_ops.edge.aten.max_pool2d_with_indices.default)
def register_max_pool2d_with_indices():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_T,
        outputs_dtypes=[utils.FP_T, utils.INT_T],
        supports_resize=True,
    )


# =============================================================================
# Convolution.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.et_vk.conv_with_clamp.default,
    ]
)
def register_convolution_cpp_ops():
    def check_conv_node(node: torch.fx.Node) -> bool:
        x = node.args[0]
        assert isinstance(x, torch.fx.Node)
        x_shape = x.meta["val"].size()
        # 4-D input implies 2D convolution
        if len(x_shape) == 4:
            batches = x.meta["val"].size()[0]
            if batches != 1:
                return False
        # 3-D input implies 1D convolution
        if len(x_shape) == 3:
            transpose = node.args[6]
            # Transposed 1D convolution is not supported yet
            if transpose:
                return False

        return True

    return OpFeatures(
        inputs_storage=[
            utils.CHANNELS_PACKED_TEXTURE,  # input
            utils.NO_STORAGE,  # weight (prepacked)
            utils.NO_STORAGE,  # bias (prepacked)
            utils.NO_STORAGE,  # stride (non tensor)
            utils.NO_STORAGE,  # padding (non tensor)
            utils.NO_STORAGE,  # dilation (non tensor)
            utils.NO_STORAGE,  # transposed (non tensor)
            utils.NO_STORAGE,  # output_padding (non tensor)
            utils.NO_STORAGE,  # groups (non tensor)
            utils.NO_STORAGE,  # output_min (non tensor)
            utils.NO_STORAGE,  # output_max (non tensor)
        ],
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_prepacking=True,
        are_node_inputs_supported_fn=check_conv_node,
    )


# =============================================================================
# Q8taConv2d.cpp, Q8taConv2dPW.cpp, Q8taConv2dDW.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.et_vk.q8ta_conv2d_pw.default,
    ]
)
def register_q8ta_conv_pw_op():
    return OpFeatures(
        inputs_storage=[
            utils.PACKED_INT8_4W4C_BUFFER,  # input
            utils.NO_STORAGE,  # input_scale (non tensor)
            utils.NO_STORAGE,  # input_zero_point (non tensor)
            utils.NO_STORAGE,  # weight (prepacked)
            utils.NO_STORAGE,  # weight_sums (prepacked)
            utils.NO_STORAGE,  # weight_scales (prepacked)
            utils.NO_STORAGE,  # output_scale (non tensor)
            utils.NO_STORAGE,  # output_zero_point (non tensor)
            utils.NO_STORAGE,  # bias (prepacked)
            utils.NO_STORAGE,  # kernel_size (non tensor)
            utils.NO_STORAGE,  # stride (non tensor)
            utils.NO_STORAGE,  # padding (non tensor)
            utils.NO_STORAGE,  # dilation (non tensor)
            utils.NO_STORAGE,  # groups (non tensor)
            utils.NO_STORAGE,  # original OC count (non tensor)
        ],
        outputs_storage=[
            utils.PACKED_INT8_CHANNELS_PACKED_BUFFER,
        ],
        supports_resize=False,
        supports_prepacking=True,
    )


@update_features(
    [
        exir_ops.edge.et_vk.q8ta_conv2d.default,
        exir_ops.edge.et_vk.q8ta_conv2d_dw.default,
    ]
)
def register_q8ta_conv2d_ops():
    return OpFeatures(
        inputs_storage=[
            utils.PACKED_INT8_4C1W_BUFFER,  # input
            utils.NO_STORAGE,  # input_scale (non tensor)
            utils.NO_STORAGE,  # input_zero_point (non tensor)
            utils.NO_STORAGE,  # weight (prepacked)
            utils.NO_STORAGE,  # weight_sums (prepacked)
            utils.NO_STORAGE,  # weight_scales (prepacked)
            utils.NO_STORAGE,  # output_scale (non tensor)
            utils.NO_STORAGE,  # output_zero_point (non tensor)
            utils.NO_STORAGE,  # bias (prepacked)
            utils.NO_STORAGE,  # kernel_size (non tensor)
            utils.NO_STORAGE,  # stride (non tensor)
            utils.NO_STORAGE,  # padding (non tensor)
            utils.NO_STORAGE,  # dilation (non tensor)
            utils.NO_STORAGE,  # groups (non tensor)
            utils.NO_STORAGE,  # original OC count (non tensor)
        ],
        outputs_storage=[
            utils.PACKED_INT8_CHANNELS_PACKED_BUFFER,
        ],
        supports_resize=False,
        supports_prepacking=True,
    )


# =============================================================================
# SDPA.cpp
# =============================================================================


@update_features("llama::sdpa_with_kv_cache")
def register_sdpa_with_kv_cache():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_prepacking=True,
    )


@update_features(
    [
        "llama::update_cache",
        "llama::custom_sdpa",
    ]
)
def register_sdpa_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
    )


# =============================================================================
# RotaryEmbedding.cpp
# =============================================================================


@update_features(exir_ops.edge.et_vk.apply_rotary_emb.default)
def register_apply_rotary_emb():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        inputs_dtypes=utils.FP_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Permute.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.permute.default)
def register_permute():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
    )


@update_features(exir_ops.edge.aten.permute_copy.default)
def register_permute_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# View.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.view_copy.default)
def register_view_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


@update_features(exir_ops.edge.dim_order_ops._to_dim_order_copy.default)
def register_to_dim_order_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_BUFFER,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Squeeze.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.squeeze_copy.dims)
def register_squeeze_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Unsqueeze.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.unsqueeze_copy.default)
def register_unsqueeze_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Clone.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.clone.default)
def register_clone():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


@update_features(exir_ops.edge.dim_order_ops._clone_dim_order.default)
def register_clone_dim_order():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Gather.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.gather.default)
def register_gather():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Expand.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.expand_copy.default)
def register_expand_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_BUFFER,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=False,
        supports_highdim=True,
    )


# =============================================================================
# Concat.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.cat.default)
def register_cat():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_T,
        supports_resize=True,
    )


# =============================================================================
# Select.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.select_copy.int)
def register_select_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Slice.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.slice_copy.Tensor)
def register_slice_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Split.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.split_with_sizes_copy.default)
def register_split_with_sizes_copy():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
        supports_resize=True,
        supports_highdim=True,
    )


# =============================================================================
# Transpose.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.t_copy.default)
def register_t_copy():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
    )


# =============================================================================
# Flip.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.flip.default)
def register_flip():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
    )


# =============================================================================
# IndexSelect.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.index_select.default)
def register_index_select():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
    )


# =============================================================================
# Arange.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.arange.start_step)
def register_arange():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_INT_T,
    )


# =============================================================================
# Pad.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.constant_pad_nd.default)
def register_constant_pad_nd():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
    )


# =============================================================================
# Full.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.full.default,
        exir_ops.edge.aten.full_like.default,
        exir_ops.edge.aten.ones.default,
        exir_ops.edge.aten.ones_like.default,
        exir_ops.edge.aten.zeros.default,
        exir_ops.edge.aten.zeros_like.default,
    ]
)
def register_full_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
    )


# =============================================================================
# ScalarTensor.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.scalar_tensor.default)
def register_scalar_tensor():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_INT_T,
    )


# =============================================================================
# Upsample.cpp
# =============================================================================


@update_features(
    [
        exir_ops.edge.aten.upsample_nearest2d.vec,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
    ]
)
def register_upsample_cpp_ops():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_T,
    )


# =============================================================================
# GridPriors.cpp
# =============================================================================


@update_features(exir_ops.edge.et_vk.grid_priors.default)
def register_grid_priors():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_T,
    )


# =============================================================================
# Repeat.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.repeat.default)
def register_repeat():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        inputs_dtypes=utils.FP_INT_BOOL_T,
    )


# =============================================================================
# Embedding.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.embedding.default)
def register_embedding():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=[utils.FP_T, utils.INT_T],
        supports_prepacking=True,
        supports_resize=True,
    )


# =============================================================================
# BatchNorm.cpp
# =============================================================================


@update_features(exir_ops.edge.aten._native_batch_norm_legit_no_training.default)
def register_native_batch_norm_legit_no_training():
    def check_batch_norm_node(node: torch.fx.Node) -> bool:
        x = node.args[0]
        if not isinstance(x, torch.fx.Node):
            return False
        x_val = x.meta.get("val", None)
        if x_val is None:
            return False
        x_shape = x_val.size()
        # Only support 4-D input tensors since this is a restriction enforced by the
        # operator implementation.
        # TODO(ssjia): Add shape agnostic support for batch norm
        return len(x_shape) == 4

    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_T,
        supports_prepacking=True,
        supports_resize=True,
        are_node_inputs_supported_fn=check_batch_norm_node,
    )


# =============================================================================
# GroupNorm.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.native_group_norm.default)
def register_native_group_norm():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        inputs_dtypes=utils.FP_T,
        outputs_storage=[
            utils.CHANNELS_PACKED_TEXTURE,
            utils.CONTIGUOUS_BUFFER,
            utils.CONTIGUOUS_BUFFER,
        ],
        supports_prepacking=True,
    )


# =============================================================================
# NativeLayerNorm.cpp
# =============================================================================


@update_features(exir_ops.edge.aten.native_layer_norm.default)
def register_native_layer_norm():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        inputs_dtypes=utils.FP_T,
        supports_prepacking=True,
        supports_resize=True,
    )


#######################
## Utility functions ##
#######################


def has_impl(target: Any) -> bool:
    if not isinstance(target, str):
        if target not in vulkan_supported_ops:
            return target.name() in vulkan_supported_ops
        return target in vulkan_supported_ops
    else:
        return target in vulkan_supported_ops


def get_op_features(target: Any) -> OpFeatures:
    if not isinstance(target, str):
        if target not in vulkan_supported_ops:
            # Try the op's name
            return vulkan_supported_ops[target.name()]

        return vulkan_supported_ops[target]
    else:
        return vulkan_supported_ops[target]


def handles_own_prepacking(target: OpKey) -> bool:
    return get_op_features(target).supports_prepacking
