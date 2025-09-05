# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator

from typing import Any, Callable, Dict, List, Optional, Union

import executorch.backends.vulkan.custom_ops_lib  # noqa

import executorch.backends.vulkan.utils as utils

import torch

from executorch.backends.vulkan.serialization.vulkan_graph_schema import VkMemoryLayout

from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch._subclasses.fake_tensor import FakeTensor

######################
## OpFeatures class ##
######################


def allow_node(node: torch.fx.Node) -> bool:
    return True


class OpFeatures:
    __slots__ = [
        # Sets of possible (storage types, memory layouts) to use for the input tensor(s)
        "inputs_storage",
        # Sets of possible (storage types, memory layouts) to use for the output tensor(s)
        "outputs_storage",
        # bool indicating if the operator has a resize function, which allows it to
        # support models with dynamic shape
        "supports_resize",
        # bool indicating if the operator handles its own prepacking. If this is True,
        # then the insert_prepack_nodes pass will not insert prepack nodes for the args
        # of the op.
        "supports_prepacking",
        # Optional check function used during partitioning to determine if a node's
        # inputs are supported by the operator implementation.
        "are_node_inputs_supported_fn",
    ]

    def __init__(
        self,
        inputs_storage: Optional[
            Union[utils.TensorRepSet, List[utils.TensorRepSet]]
        ] = None,
        outputs_storage: Optional[
            Union[utils.TensorRepSet, List[utils.TensorRepSet]]
        ] = None,
        supports_resize: bool = False,
        supports_prepacking: bool = False,
        are_node_inputs_supported_fn: Optional[Callable] = allow_node,
    ):
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
        self.supports_prepacking = supports_prepacking

        self.are_node_inputs_supported_fn = are_node_inputs_supported_fn

    def make_op_repsets(
        self,
        op_node: torch.fx.Node,
        texture_limits: utils.ImageExtents = utils.DEFAULT_TEXTURE_LIMITS,
    ) -> utils.OpRepSets:
        return utils.OpRepSets(
            self.inputs_storage, self.outputs_storage, op_node, texture_limits
        )


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


@update_features(
    [
        operator.getitem,
        # Symbolic integer ops
        torch.ops.aten.sym_size.int,
        operator.add,
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
def register_ephemeral_op():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
    )


@update_features(
    [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_token.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_token.default,
    ]
)
def register_quantization_op():
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
def register_affine_quantization_op():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_BUFFER,
        supports_resize=True,
    )


@update_features(
    [
        exir_ops.edge.torchao.choose_qparams_affine.default,
        exir_ops.edge.quantized_decomposed.choose_qparams.tensor,
        exir_ops.edge.quantized_decomposed.choose_qparams_per_token_asymmetric.default,
    ]
)
def register_torchao_quantization_op():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_BUFFER,
        supports_resize=True,
    )


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
def register_binary_op():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
    )


@update_features(
    [
        exir_ops.edge.aten.abs.default,
        exir_ops.edge.aten.clamp.default,
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
def register_unary_op():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
    )


@update_features(exir_ops.edge.aten._to_copy.default)
def register_to_copy_op():
    def check_to_copy_node(node: torch.fx.Node) -> bool:
        float_dtypes = [torch.float16, torch.float32]

        if len(node.args) != 1:
            return False

        in_arg = node.args[0]
        if not isinstance(in_arg, torch.fx.Node):
            return False

        in_tensor = in_arg.meta.get("val", None)
        out_tensor = node.meta.get("val", None)

        if isinstance(in_tensor, FakeTensor) and isinstance(out_tensor, FakeTensor):
            if out_tensor.dtype in float_dtypes and in_tensor.dtype in float_dtypes:
                return True

        return False

    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
        are_node_inputs_supported_fn=check_to_copy_node,
    )


@update_features(exir_ops.edge.dim_order_ops._to_dim_order_copy.default)
def register_to_copy_dim_order_op():
    # Currently there is no "real" implementation for to_dim_order_copy, but it can be
    # removed as long as the operator is not changing the dtype, i.e. the operator call
    # is modifying the dim order only. Therefore, check that the input and output dtypes
    # are the same, if so the operator is safe to remove.
    def check_dim_order_copy_node(node: torch.fx.Node) -> bool:
        in_arg = node.args[0]
        if not isinstance(in_arg, torch.fx.Node):
            return False

        in_tensor = in_arg.meta.get("val", None)
        out_tensor = node.meta.get("val", None)

        if in_tensor.dtype != out_tensor.dtype:
            return False

        return True

    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
        are_node_inputs_supported_fn=check_dim_order_copy_node,
    )


@update_features(
    [
        exir_ops.edge.aten.bmm.default,
        exir_ops.edge.aten.mm.default,
        exir_ops.edge.aten.addmm.default,
        exir_ops.edge.aten.linear.default,
    ]
)
def register_mm_op():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        supports_resize=True,
        supports_prepacking=True,
    )


@update_features(
    [
        exir_ops.edge.aten._weight_int8pack_mm.default,
        exir_ops.edge.et_vk.linear_qcs4w.default,
    ]
)
def register_int8_mm_op():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        supports_resize=True,
        supports_prepacking=True,
    )


@update_features(
    [
        exir_ops.edge.et_vk.linear_q8ta_q8csw.default,
    ]
)
def register_qa_qw_linear():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        supports_prepacking=True,
        supports_resize=False,
    )


@update_features(
    [
        exir_ops.edge.et_vk.linear_weight_int4.default,
    ]
)
def register_int4_mm_op():
    return OpFeatures(
        inputs_storage=utils.CONTIGUOUS_ANY,
        supports_resize=True,
        supports_prepacking=True,
    )


@update_features(
    [
        exir_ops.edge.et_vk.linear_qta8a_qga4w.default,
    ]
)
def register_dqlinear_op():
    return OpFeatures(
        inputs_storage=[
            utils.CONTIGUOUS_ANY,  # input
            utils.CONTIGUOUS_BUFFER,  # mat1 scales
            utils.CONTIGUOUS_BUFFER,  # mat1 zeros
            utils.NO_STORAGE,  # weight (prepacked)
            utils.NO_STORAGE,  # group size (non tensor)
            utils.CONTIGUOUS_BUFFER,  # mat2 scales
            utils.CONTIGUOUS_BUFFER,  # mat2 zeros
        ],
        supports_resize=True,
        supports_prepacking=True,
    )


@update_features(
    [
        exir_ops.edge.aten._log_softmax.default,
        exir_ops.edge.aten._softmax.default,
    ]
)
def register_softmax_op():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        supports_resize=True,
    )


@update_features(
    [
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.sum.dim_IntList,
        exir_ops.edge.aten.amax.default,
        exir_ops.edge.aten.amin.default,
    ]
)
def register_reduce_op():
    def check_reduce_node(node: torch.fx.Node) -> bool:
        dim_list = node.args[1]
        if isinstance(dim_list, list) and len(dim_list) > 2:
            return False

        if isinstance(dim_list, list) and len(dim_list) == 2:
            # Try to get the memory layout for this node
            try:
                memory_layout = utils.get_node_memory_layout(node)

                # If we have memory layout information, check if any dimension in dim_list corresponds to a packed dimension
                if (
                    memory_layout is not None
                    and memory_layout != VkMemoryLayout.DEFAULT_LAYOUT
                ):
                    # For now only default layout is supported for 2D reduction.
                    # Because we can't determine if the input is NCHW or NHWC here,
                    # assume the reduction dimension is packed so we cannot support it.
                    return False
            except (AssertionError, KeyError, AttributeError):
                # If we can't get memory layout information, we'll assume the dims aren't packed
                pass

        def try_find_keepdim_arg(node: torch.fx.Node) -> bool:
            for arg in node.args:
                if isinstance(arg, bool):
                    return arg

            # Assume false by default
            return False

        keepdim = try_find_keepdim_arg(node)
        if isinstance(keepdim, bool) and not keepdim:
            return False

        return True

    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        supports_resize=True,
        are_node_inputs_supported_fn=check_reduce_node,
    )


@update_features(
    [
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.max_pool2d.default,
        exir_ops.edge.aten.max_pool2d_with_indices.default,
    ]
)
def register_2d_pool_op():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        supports_resize=True,
    )


@update_features(
    [
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.et_vk.conv_with_clamp.default,
    ]
)
def register_convolution_op():
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
        supports_resize=True,
        supports_prepacking=True,
    )


@update_features(
    [
        exir_ops.edge.et_vk.conv2d_q8ta_q8csw.default,
    ]
)
def register_quantized_conv_op():
    return OpFeatures(
        inputs_storage=[
            utils.CHANNELS_PACKED_TEXTURE,  # input
            utils.NO_STORAGE,  # input_scale (non tensor)
            utils.NO_STORAGE,  # input_zero_point (non tensor)
            utils.NO_STORAGE,  # weight (prepacked)
            utils.NO_STORAGE,  # weight_sums (prepacked)
            utils.NO_STORAGE,  # weight_scales (prepacked)
            utils.NO_STORAGE,  # bias (prepacked)
            utils.NO_STORAGE,  # kernel_size (non tensor)
            utils.NO_STORAGE,  # stride (non tensor)
            utils.NO_STORAGE,  # padding (non tensor)
            utils.NO_STORAGE,  # dilation (non tensor)
            utils.NO_STORAGE,  # groups (non tensor)
            utils.NO_STORAGE,  # original OC count (non tensor)
        ],
        supports_resize=False,
        supports_prepacking=True,
    )


@update_features("llama::sdpa_with_kv_cache")
def register_sdpa_with_kv_cache_op():
    return OpFeatures(
        inputs_storage=utils.WIDTH_PACKED_TEXTURE,
        supports_resize=True,
        supports_prepacking=True,
    )


@update_features(
    [
        "llama::update_cache",
        "llama::custom_sdpa",
    ]
)
def register_sdpa_ops():
    return OpFeatures(
        inputs_storage=utils.WIDTH_PACKED_TEXTURE,
        supports_resize=True,
    )


@update_features(exir_ops.edge.et_vk.apply_rotary_emb.default)
def register_rotary_emb_op():
    return OpFeatures(
        inputs_storage=utils.WIDTH_PACKED_TEXTURE,
        supports_resize=True,
    )


@update_features(
    [
        exir_ops.edge.aten.permute.default,
    ]
)
def register_view_ops():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        supports_resize=True,
    )


@update_features(
    [
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.unsqueeze_copy.default,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten.permute_copy.default,
    ]
)
def register_view_ops_with_buffer_meta():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
    )


@update_features(exir_ops.edge.aten.expand_copy.default)
def register_expand():
    return OpFeatures(inputs_storage=utils.ANY_BUFFER, supports_resize=False)


# Fully featured transfer operators (i.e. operators that copy data from the input
# tensor(s) to the output tensor(s)), which have memory layout agnostic implementations
# for both texture and buffer storage types.
@update_features(exir_ops.edge.aten.cat.default)
def register_cat_op():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
    )


# Fully featured transfer operators (i.e. operators that copy data from the input
# tensor(s) to the output tensor(s)), which have memory layout agnostic implementations
# for both texture and buffer storage types.
@update_features(
    [
        exir_ops.edge.aten.select_copy.int,
        exir_ops.edge.aten.slice_copy.Tensor,
    ]
)
def register_transfer_ops():
    return OpFeatures(
        inputs_storage=utils.ANY_STORAGE,
        supports_resize=True,
    )


# Ops ported from PyTorch Vulkan backend. These ops commonly support channels
# packed tensors only and do not have a resize function.
@update_features(
    [
        # Shape Manipulation
        exir_ops.edge.aten.t_copy.default,
        # Indexing and lookup
        exir_ops.edge.aten.flip.default,
        exir_ops.edge.aten.index_select.default,
        # Tensor creation
        exir_ops.edge.aten.arange.start_step,
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.full.default,
        exir_ops.edge.aten.full_like.default,
        exir_ops.edge.aten.ones.default,
        exir_ops.edge.aten.ones_like.default,
        exir_ops.edge.aten.scalar_tensor.default,
        exir_ops.edge.aten.upsample_nearest2d.vec,
        exir_ops.edge.aten.upsample_bilinear2d.vec,
        exir_ops.edge.aten.zeros.default,
        exir_ops.edge.aten.zeros_like.default,
        exir_ops.edge.et_vk.grid_priors.default,
    ]
)
def register_ported_op():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
    )


# Ops ported from PyTorch Vulkan backend. These ops are in a separate registry because they support all packed dimensions
@update_features(
    [
        # Tensor combination
        exir_ops.edge.aten.repeat.default,
        exir_ops.edge.aten.split_with_sizes_copy.default,
        exir_ops.edge.aten.split.Tensor,
    ]
)
def register_ported_op_all_packed_dims():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
    )


# Ported ops that support their own prepacking.
@update_features(
    [
        exir_ops.edge.aten.embedding.default,
        exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
    ]
)
def register_ported_ops_with_prepacking():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        supports_prepacking=True,
    )


@update_features(
    [
        exir_ops.edge.aten.native_group_norm.default,
    ]
)
def register_native_group_norm():
    return OpFeatures(
        inputs_storage=utils.CHANNELS_PACKED_TEXTURE,
        outputs_storage=[
            utils.CHANNELS_PACKED_TEXTURE,
            utils.CONTIGUOUS_BUFFER,
            utils.CONTIGUOUS_BUFFER,
        ],
        supports_prepacking=True,
    )


# Ported ops that support their own prepacking.
@update_features(
    [
        exir_ops.edge.aten.native_layer_norm.default,
    ]
)
def register_ported_ops_with_prepacking_all_dims():
    return OpFeatures(
        inputs_storage=utils.ANY_TEXTURE,
        supports_prepacking=True,
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
