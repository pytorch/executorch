# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import operator

from typing import Callable, Dict, Optional, Set, Union

import executorch.backends.vulkan.custom_ops_lib  # noqa

import torch

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)

from executorch.backends.vulkan.utils import (
    all_memory_layouts,
    all_packed_dims,
    PackedDim,
)
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from torch._subclasses.fake_tensor import FakeTensor

######################
## OpFeatures class ##
######################


def allow_node(node: torch.fx.Node) -> bool:
    return True


class TextureImplFeatures:
    __slots__ = [
        "valid_packed_dims",
        "uses_axis_map",
    ]

    def __init__(
        self,
        uses_axis_map: bool = False,
        valid_packed_dims: Optional[Set[PackedDim]] = None,
    ):
        self.uses_axis_map: bool = uses_axis_map
        self.valid_packed_dims = set()
        if valid_packed_dims is not None:
            self.valid_packed_dims = valid_packed_dims

    def valid_memory_layouts(self) -> Set[VkMemoryLayout]:
        """
        Derive the set of memory layouts supported by the texture implementation based
        on the valid packed dimensions.
        """
        layouts = set()

        if PackedDim.WIDTH in self.valid_packed_dims:
            layouts.add(VkMemoryLayout.TENSOR_WIDTH_PACKED)

        if PackedDim.HEIGHT in self.valid_packed_dims:
            layouts.add(VkMemoryLayout.TENSOR_HEIGHT_PACKED)

        if PackedDim.CHANNELS in self.valid_packed_dims:
            layouts.add(VkMemoryLayout.TENSOR_CHANNELS_PACKED)

        return layouts


class OpFeatures:
    __slots__ = [
        # None or TextureImplFeatures to specify implementation details of the texture
        # based operator implementation.
        "texture_impl",
        # bool indicating if the operator has a buffer based implementation.
        "buffer_impl",
        # bool indicating if the operator has a resize function, which allows it to
        # support dynamic shape tensors.
        "resize_fn",
        # Optimal
        "optimal_storage",
        "optimal_layout",
        # bool indicating if the operator handles its own prepacking. If this is True,
        # then the insert_prepack_nodes pass will not insert prepack nodes for the args
        # of the op.
        "handles_own_prepacking",
        # Optional dictionary to specify a custom function to calculate the required
        # image extents for a particular argument index.
        "skip_limits_check",
        # Optional check function used during partitioning to determine if a node's
        # inputs are supported by the operator implementation.
        "check_node_fn",
    ]

    def __init__(
        self,
        texture_impl: Optional[TextureImplFeatures] = None,
        buffer_impl: bool = False,
        resize_fn: bool = False,
        optimal_storage: Optional[VkStorageType] = None,
        optimal_layout: Optional[VkMemoryLayout] = None,
        handles_own_prepacking: bool = False,
        skip_limits_check: Optional[Set[int]] = None,
        check_node_fn: Optional[Callable] = None,
    ):
        self.texture_impl: Optional[TextureImplFeatures] = texture_impl
        self.buffer_impl: bool = buffer_impl
        self.resize_fn: bool = resize_fn
        self.optimal_storage: Optional[VkStorageType] = optimal_storage
        self.optimal_layout: Optional[VkMemoryLayout] = optimal_layout
        self.handles_own_prepacking: bool = handles_own_prepacking

        self.skip_limits_check: Set[int] = set()
        if skip_limits_check is not None:
            self.skip_limits_check = skip_limits_check

        self.check_node_fn: Callable = allow_node
        if check_node_fn is not None:
            self.check_node_fn = check_node_fn

    def propose_storage_type(self) -> Optional[VkStorageType]:
        """
        Propose a storage type that should be used for this operator. A proposal can be
        made if one of the following is true:
        1. The operator specifies an optimal storage type
        2. Only one storage type is supported.

        If both storage types are supported and no optimal storage type is specified,
        then None is returned to indicate that there is no preference in storage type.
        """
        if self.optimal_storage is not None:
            return self.optimal_storage

        if self.texture_impl is not None and not self.buffer_impl:
            return VkStorageType.TEXTURE_3D
        elif self.buffer_impl and self.texture_impl is None:
            return VkStorageType.BUFFER

        return None

    def supported_storage_types(self) -> Set[VkStorageType]:
        """
        Return the set of storage types supported by this operator.
        """
        storage_types = set()
        if self.texture_impl is not None:
            storage_types.add(VkStorageType.TEXTURE_3D)
        if self.buffer_impl:
            storage_types.add(VkStorageType.BUFFER)

        return storage_types

    def propose_memory_layout(self, storage: VkStorageType) -> Optional[VkMemoryLayout]:
        """
        Given a storage type as a precondition, propose a memory layout that should be
        used for this operator. A proposal can be made if one of the following is true:
        1. The operator specifies an optimal memory layout
        2. Only one memory layout is supported.

        If multiple memory layouts are supported and no optimal memory layout is
        specified then return None to indicate that the "best" memory layout for the
        operator is ambiguous.
        """
        if self.optimal_layout is not None:
            return self.optimal_layout

        if storage == VkStorageType.TEXTURE_3D:
            assert self.texture_impl is not None
            possible_layouts = self.texture_impl.valid_memory_layouts()
            if len(possible_layouts) == 1:
                return next(iter(possible_layouts))

        return None

    def supported_memory_layouts(self, storage: VkStorageType) -> Set[VkMemoryLayout]:
        """
        Return the set of memory layouts supported by this operator for a given storage
        type.
        """
        if storage == VkStorageType.TEXTURE_3D:
            assert self.texture_impl is not None
            return self.texture_impl.valid_memory_layouts()
        else:
            return all_memory_layouts


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
            vulkan_supported_ops[op] = OpFeatures()
            vulkan_supported_ops[op] = fn(vulkan_supported_ops[op])

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
        # Quantization related ops will be fused via graph passes
        exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.tensor,
        exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    ]
)
def register_ephemeral_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        uses_axis_map=True,
        valid_packed_dims=all_packed_dims,
    )
    features.buffer_impl = True
    features.resize_fn = True
    return features


@update_features(
    [
        exir_ops.edge.aten.add.Tensor,
        exir_ops.edge.aten.sub.Tensor,
        exir_ops.edge.aten.minimum.default,
        exir_ops.edge.aten.mul.Tensor,
        exir_ops.edge.aten.div.Tensor,
        exir_ops.edge.aten.div.Tensor_mode,
        exir_ops.edge.aten.pow.Tensor_Tensor,
    ]
)
def register_binary_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        uses_axis_map=True,
        valid_packed_dims=all_packed_dims,
    )
    features.resize_fn = True
    return features


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
    ]
)
def register_unary_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        uses_axis_map=True,
        valid_packed_dims=all_packed_dims,
    )
    features.buffer_impl = True
    features.resize_fn = True
    return features


@update_features(exir_ops.edge.aten._to_copy.default)
def register_to_copy_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        uses_axis_map=True,
        valid_packed_dims=all_packed_dims,
    )
    features.resize_fn = True

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

    features.check_node_fn = check_to_copy_node

    return features


@update_features(exir_ops.edge.dim_order_ops._to_dim_order_copy.default)
def register_to_copy_dim_order_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        uses_axis_map=True,
        valid_packed_dims=all_packed_dims,
    )
    features.buffer_impl = True
    features.resize_fn = True

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

    features.check_node_fn = check_dim_order_copy_node

    return features


@update_features(
    [
        exir_ops.edge.aten.bmm.default,
        exir_ops.edge.aten.mm.default,
        exir_ops.edge.aten.addmm.default,
        exir_ops.edge.aten.linear.default,
    ]
)
def register_mm_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        uses_axis_map=True,
        valid_packed_dims={
            PackedDim.WIDTH,
            PackedDim.CHANNELS,
        },
    )
    features.buffer_impl = True
    features.resize_fn = True
    features.optimal_storage = VkStorageType.TEXTURE_3D
    features.optimal_layout = VkMemoryLayout.TENSOR_WIDTH_PACKED
    features.handles_own_prepacking = True
    return features


@update_features(exir_ops.edge.aten._weight_int8pack_mm.default)
def register_int8_mm_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        uses_axis_map=False,
        valid_packed_dims={PackedDim.WIDTH},
    )
    features.buffer_impl = True
    features.resize_fn = True
    features.optimal_storage = VkStorageType.TEXTURE_3D
    features.optimal_layout = VkMemoryLayout.TENSOR_WIDTH_PACKED
    features.handles_own_prepacking = True
    return features


@update_features(exir_ops.edge.et_vk.linear_weight_int4.default)
def register_int4_mm_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        uses_axis_map=False,
        valid_packed_dims={PackedDim.WIDTH},
    )
    features.resize_fn = True
    features.optimal_storage = VkStorageType.TEXTURE_3D
    features.optimal_layout = VkMemoryLayout.TENSOR_WIDTH_PACKED
    features.handles_own_prepacking = True
    return features


@update_features(
    [
        exir_ops.edge.aten._log_softmax.default,
        exir_ops.edge.aten._softmax.default,
    ]
)
def register_softmax_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims=all_packed_dims,
    )
    features.resize_fn = True
    return features


@update_features(
    [
        exir_ops.edge.aten.mean.dim,
        exir_ops.edge.aten.sum.dim_IntList,
        exir_ops.edge.aten.amax.default,
        exir_ops.edge.aten.amin.default,
    ]
)
def register_reduce_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims=all_packed_dims,
    )
    features.resize_fn = True

    def check_reduce_node(node: torch.fx.Node) -> bool:
        dim_list = node.args[1]
        if isinstance(dim_list, list) and len(dim_list) != 1:
            return False

        keepdim = node.args[2]
        if isinstance(keepdim, bool) and not keepdim:
            return False

        return True

    features.check_node_fn = check_reduce_node
    return features


@update_features(
    [
        exir_ops.edge.aten.avg_pool2d.default,
        exir_ops.edge.aten.max_pool2d_with_indices.default,
    ]
)
def register_2d_pool_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims={PackedDim.CHANNELS},
    )
    features.resize_fn = True
    return features


@update_features(
    [
        exir_ops.edge.aten.convolution.default,
        exir_ops.edge.et_vk.conv_with_clamp.default,
    ]
)
def register_convolution_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims={PackedDim.CHANNELS},
    )
    features.resize_fn = True
    features.optimal_storage = VkStorageType.TEXTURE_3D
    features.optimal_layout = VkMemoryLayout.TENSOR_CHANNELS_PACKED
    features.handles_own_prepacking = True
    features.skip_limits_check = {1, 2}
    return features


@update_features("llama::sdpa_with_kv_cache")
def register_sdpa_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims={PackedDim.WIDTH},
    )
    features.resize_fn = True
    features.optimal_storage = VkStorageType.TEXTURE_3D
    features.optimal_layout = VkMemoryLayout.TENSOR_WIDTH_PACKED
    features.handles_own_prepacking = True
    return features


@update_features(exir_ops.edge.et_vk.apply_rotary_emb.default)
def register_rotary_emb_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims={PackedDim.WIDTH},
    )
    features.resize_fn = True
    return features


@update_features(exir_ops.edge.aten.view_copy.default)
def register_view_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims=all_packed_dims,
    )
    features.resize_fn = True
    return features


# Ops ported from PyTorch Vulkan backend. These ops commonly support channels
# packed tensors only and do not have a resize function.
@update_features(
    [
        # Shape Manipulation
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.unsqueeze_copy.default,
        exir_ops.edge.aten.permute_copy.default,
        exir_ops.edge.aten.t_copy.default,
        # Indexing and lookup
        exir_ops.edge.aten.flip.default,
        exir_ops.edge.aten.index_select.default,
        exir_ops.edge.aten.select_copy.int,
        exir_ops.edge.aten.slice_copy.Tensor,
        # Tensor combination
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.split_with_sizes_copy.default,
        exir_ops.edge.aten.split.Tensor,
        exir_ops.edge.aten.repeat.default,
        # Tensor creation
        exir_ops.edge.aten.arange.start_step,
        exir_ops.edge.aten.clone.default,
        exir_ops.edge.aten.constant_pad_nd.default,
        exir_ops.edge.aten.full.default,
        exir_ops.edge.aten.full_like.default,
        exir_ops.edge.aten.ones.default,
        exir_ops.edge.aten.ones_like.default,
        exir_ops.edge.aten.upsample_nearest2d.vec,
        exir_ops.edge.aten.zeros.default,
        exir_ops.edge.aten.zeros_like.default,
        exir_ops.edge.et_vk.grid_priors.default,
    ]
)
def register_ported_op(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims={PackedDim.CHANNELS},
    )
    return features


# Ported ops that support their own prepacking.
@update_features(
    [
        exir_ops.edge.aten.embedding.default,
        exir_ops.edge.aten._native_batch_norm_legit_no_training.default,
        exir_ops.edge.aten.native_layer_norm.default,
    ]
)
def register_ported_ops_with_prepacking(features: OpFeatures):
    features.texture_impl = TextureImplFeatures(
        valid_packed_dims={PackedDim.CHANNELS},
    )
    features.handles_own_prepacking = True
    return features


#######################
## Utility functions ##
#######################


def has_impl(target: OpKey) -> bool:
    if not isinstance(target, str):
        if target not in vulkan_supported_ops:
            return target.name() in vulkan_supported_ops
        return target in vulkan_supported_ops
    else:
        return target in vulkan_supported_ops


def get_op_features(target: OpKey) -> OpFeatures:
    if not isinstance(target, str):
        if target not in vulkan_supported_ops:
            # Try the op's name
            return vulkan_supported_ops[target.name()]

        return vulkan_supported_ops[target]
    else:
        return vulkan_supported_ops[target]


def handles_own_prepacking(target: OpKey) -> bool:
    return get_op_features(target).handles_own_prepacking
