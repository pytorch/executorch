# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import IntEnum
from typing import Optional, Set, Tuple

import torch

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)

from executorch.exir.tensor import TensorSpec

from torch._export.utils import is_buffer, is_param

from torch._subclasses.fake_tensor import FakeTensor

from torch.export import ExportedProgram

##
## Node type determination
##


def is_get_attr_node(node: torch.fx.Node) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "get_attr"


def is_constant(program: ExportedProgram, node: torch.fx.Node) -> bool:
    return node.name in program.graph_signature.inputs_to_lifted_tensor_constants


def is_param_node(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Check if the given node is a parameter within the exported program
    """
    return (
        is_get_attr_node(node)
        or is_param(program, node)
        or is_buffer(program, node)
        or is_constant(program, node)
    )


def is_symint_node(node: torch.fx.Node) -> bool:
    """
    Returns true if the given node produces a SymInt value
    """
    if "val" not in node.meta:
        return False

    if isinstance(node.meta["val"], torch.SymInt):
        return True

    return False


def is_tensor_node(node: torch.fx.Node) -> bool:
    """
    Returns true if the given node produces a tensor value, or a collection of tensor values
    """
    # All nodes with tensor values are tagged by the SpecPropPass transform
    if "spec" in node.meta:
        return True

    if "val" not in node.meta:
        return False

    if isinstance(node.meta["val"], FakeTensor):
        return True

    if isinstance(node.meta["val"], list) or isinstance(node.meta["val"], tuple):
        return all(isinstance(x, FakeTensor) for x in node.meta["val"])

    return False


##
## Memory Layout, Storage Type Determination
##

ImageExtents = Tuple[int, int, int]

DEFAULT_TEXTURE_LIMITS = (16384, 16384, 2048)
DEFAULT_BUFFER_LIMIT = 128 * (1024 * 1024)


class PackedDim(IntEnum):
    WIDTH = 0
    HEIGHT = 1
    CHANNELS = 2


all_packed_dims: Set[PackedDim] = {
    PackedDim.WIDTH,
    PackedDim.HEIGHT,
    PackedDim.CHANNELS,
}

all_storage_types: Set[VkStorageType] = {
    VkStorageType.BUFFER,
    VkStorageType.TEXTURE_3D,
}

all_memory_layouts: Set[VkMemoryLayout] = {
    VkMemoryLayout.TENSOR_WIDTH_PACKED,
    VkMemoryLayout.TENSOR_HEIGHT_PACKED,
    VkMemoryLayout.TENSOR_CHANNELS_PACKED,
}


def within_buffer_limit(node: torch.fx.Node, buffer_limit: int) -> int:
    """
    Checks whether the tensors produced by the given node can fit within the device's
    GPU buffer limit, which represents the maximum number of elements that can be stored
    in a GPU buffer.
    """
    assert is_tensor_node(node)

    if isinstance(node.meta["val"], FakeTensor):
        return node.meta["val"].numel() < buffer_limit
    elif isinstance(node.meta["val"], list) or isinstance(node.meta["val"], tuple):
        return all(x.numel() < buffer_limit for x in node.meta["val"])
    else:
        raise RuntimeError(f"Cannot get numel for val of type {type(node.meta['val'])}")


def required_image_extents(sizes: torch.Size, layout: VkMemoryLayout) -> ImageExtents:
    """
    Calculate the image extents that will be used to represent a tensor with the given sizes
    and memory layout in the Vulkan Delegate.
    """
    width = sizes[-1] if len(sizes) >= 1 else 1
    height = sizes[-2] if len(sizes) >= 2 else 1
    channels = sizes[-3] if len(sizes) >= 3 else 1
    batch = sizes[0] if len(sizes) >= 4 else 1

    if layout == VkMemoryLayout.TENSOR_WIDTH_PACKED:
        width = (width + 3) // 4
    elif layout == VkMemoryLayout.TENSOR_HEIGHT_PACKED:
        height = (height + 3) // 4
    elif layout == VkMemoryLayout.TENSOR_CHANNELS_PACKED:
        channels = (channels + 3) // 4
    else:
        raise RuntimeError(f"Unsupported memory layout {layout}")

    return width, height, channels * batch


def extents_are_valid(extents: ImageExtents, limits: ImageExtents) -> bool:
    return all(extents[i] <= limits[i] for i in range(len(extents)))


def valid_texture_memory_layouts(
    tensor_sizes: torch.Size, texture_limits: ImageExtents
) -> Set[VkMemoryLayout]:
    """
    Given tensor sizes, determine the set of memory layouts which will prodice a texture
    that can fit within the specified device limits.
    """
    valid_layouts = set()
    for layout in list(all_memory_layouts):
        extents = required_image_extents(tensor_sizes, layout)
        if extents_are_valid(extents, texture_limits):
            valid_layouts.add(layout)

    return valid_layouts


def possible_node_memory_layouts(
    node: torch.fx.Node, texture_limits: ImageExtents
) -> Set[VkMemoryLayout]:
    """
    Given a node, determine the set of memory layouts which can be used to represent all
    tensors involved in the computation.
    """
    assert is_tensor_node(node)
    if isinstance(node.meta["val"], FakeTensor):
        return valid_texture_memory_layouts(node.meta["val"].shape, texture_limits)
    valid_layouts = set()
    if isinstance(node.meta["val"], list) or isinstance(node.meta["val"], tuple):
        for fake_tensor in node.meta["val"]:
            valid_layouts = valid_layouts.union(
                valid_texture_memory_layouts(fake_tensor.shape, texture_limits)
            )

    return valid_layouts


##
## TensorSpec Utils
##


def set_node_spec_attr(node: torch.fx.Node, attr: str, value):
    assert "spec" in node.meta
    spec = node.meta["spec"]
    if isinstance(spec, TensorSpec):
        setattr(spec, attr, value)
    elif isinstance(spec, (list, tuple)):
        for s in spec:
            assert isinstance(s, TensorSpec)
            setattr(s, attr, value)
    else:
        raise RuntimeError(f"Cannot set attr for spec of type {type(spec)}")


def get_node_spec_attr(node: torch.fx.Node, attr: str, return_first: bool = True):
    assert "spec" in node.meta
    spec = node.meta["spec"]
    if isinstance(spec, TensorSpec):
        return getattr(spec, attr) if hasattr(spec, attr) else None
    elif isinstance(spec, (list, tuple)):
        if return_first:
            return getattr(spec[0], attr) if hasattr(spec[0], attr) else None
        else:
            return [getattr(s, attr) if hasattr(s, attr) else None for s in spec]
    else:
        raise RuntimeError(f"Cannot get attr for spec of type {type(spec)}")


def get_node_storage_type(node: torch.fx.Node) -> Optional[VkStorageType]:
    return get_node_spec_attr(node, "vk_storage_type")


def get_node_memory_layout(node: torch.fx.Node) -> Optional[VkMemoryLayout]:
    return get_node_spec_attr(node, "vk_memory_layout")
