# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Any, List, Optional, Set, Tuple, Union

import torch

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)

from executorch.exir.backend.canonical_partitioners.config_partitioner import (
    format_target_name,
)

from executorch.exir.dialects.edge._ops import EdgeOpOverload

from executorch.exir.tensor import TensorSpec

from torch._export.utils import is_buffer, is_lifted_tensor_constant, is_param

from torch._subclasses.fake_tensor import FakeTensor, FakeTensorConverter

from torch.export import ExportedProgram

from torch.export.exported_program import InputKind
from torch.export.graph_signature import TensorArgument

TorchOpType = Union[EdgeOpOverload, torch._ops.OpOverload, str]

_DQ_OPS = {
    "dequantize_per_tensor.tensor",
    "dequantize_per_tensor.default",
    "dequantize_per_channel.default",
    "dequantize_per_channel_group.default",
    "dequantize_per_token.default",
    "dequantize_affine.default",
}

_Q_OPS = {
    "quantize_per_tensor.tensor",
    "quantize_per_tensor.default",
    "quantize_per_channel.default",
    "quantize_per_token.default",
    "quantize_affine.default",
}

##
## Node type determination
##

# Convenience type
MaybeNodeList = Union[torch.fx.Node, List[torch.fx.Node], Tuple[torch.fx.Node]]


def is_torch_op_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False

    if isinstance(node.target, EdgeOpOverload):
        return True
    if isinstance(node.target, torch._ops.OpOverload):
        return True

    return False


def is_dequant_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    return node_name in _DQ_OPS


def is_quant_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    return node_name in _Q_OPS


def is_choose_qparams_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    return "choose_qparams" in node_name


def is_dequant_per_channel_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    return node_name == "dequantize_per_channel.default"


def is_view_copy_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    return "view_copy" in node_name


def is_linear_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    node_name = format_target_name(node.target.__name__)  # pyre-ignore
    return node_name == "linear.default"


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
        or is_lifted_tensor_constant(program, node)
    )


def is_mutable_buffer_node(
    node: torch.fx.Node, exported_program: ExportedProgram
) -> bool:
    if node.target not in exported_program.graph_signature.inputs_to_buffers:
        return False
    buf = exported_program.graph_signature.inputs_to_buffers[node.target]
    return buf in exported_program.graph_signature.buffers_to_mutate.values()


def is_symint_node(node: torch.fx.Node) -> bool:
    """
    Returns true if the given node produces a SymInt value
    """
    if "val" not in node.meta:
        return False

    if isinstance(node.meta["val"], torch.SymInt):
        return True

    return False


def is_single_tensor_node(node: torch.fx.Node) -> bool:
    """
    Returns true if the given node produces a single tensor value
    """
    if "val" not in node.meta:
        return False

    if isinstance(node.meta["val"], FakeTensor):
        return True

    return False


def is_tensor_collection_node(node: Any) -> bool:
    """
    Returns true if the given node produces a collection of tensor values
    """
    if not isinstance(node, torch.fx.Node):
        return False

    if "val" not in node.meta:
        return False

    if isinstance(node.meta["val"], list) or isinstance(node.meta["val"], tuple):
        return all(isinstance(x, FakeTensor) for x in node.meta["val"])

    return False


def is_tensor_node(node: Any) -> bool:
    """
    Returns true if the given node produces a tensor value, or a collection of tensor values
    """
    if not isinstance(node, torch.fx.Node):
        return False

    if "val" not in node.meta:
        return False

    if isinstance(node.meta["val"], FakeTensor):
        return True

    if isinstance(node.meta["val"], list) or isinstance(node.meta["val"], tuple):
        return all(isinstance(x, FakeTensor) for x in node.meta["val"])

    return False


def is_tensor_arg_node(node: Any) -> bool:
    if isinstance(node, torch.fx.Node):
        return is_tensor_node(node)
    elif isinstance(node, (list, tuple)):
        return all(is_tensor_node(n) for n in node)

    return False


def num_tensor_arg_nodes(node: torch.fx.Node) -> int:
    """
    For a given node, return the number of argument nodes that are associated with
    tensors.
    """
    count = 0
    for arg_node in node.args:
        if not isinstance(arg_node, torch.fx.Node):
            continue
        if is_tensor_node(arg_node):
            count += 1

    return count


def num_tensors_in_node(node: torch.fx.Node) -> int:
    """
    Returns the number of tensors associated a given node
    """
    if "val" not in node.meta:
        return 0

    if isinstance(node.meta["val"], FakeTensor):
        return 1

    if isinstance(node.meta["val"], list) or isinstance(node.meta["val"], tuple):
        if all(isinstance(x, FakeTensor) for x in node.meta["val"]):
            return len(node.meta["val"])

    return 0


def tensor_node_is_bool(node: torch.fx.Node) -> bool:
    """
    Returns true if a given node contains a tensor with bool dtype
    """
    if isinstance(node.meta["val"], FakeTensor):
        return node.meta["val"].dtype == torch.bool
    if isinstance(node.meta["val"], list) or isinstance(node.meta["val"], tuple):
        for fake_tensor in node.meta["val"]:
            if isinstance(fake_tensor, FakeTensor):
                if fake_tensor.dtype == torch.bool:
                    return True
    return False


def get_primary_arg_idx(self, node: torch.fx.Node) -> Optional[int]:
    primary_arg_idx: Optional[int] = None
    for i, arg_node in enumerate(node.args):
        if self.is_non_constant_tensor_node(arg_node):
            return i

    return primary_arg_idx


def node_comes_from_any_nn_module_in_set(
    node,
    nn_module_typenames: Set[str],
) -> bool:
    if isinstance(node, (list, tuple)):
        return all(
            node_comes_from_any_nn_module_in_set(n, nn_module_typenames) for n in node
        )

    if not isinstance(node, torch.fx.Node):
        return False

    nn_module_stack = node.meta.get("nn_module_stack", None)
    if nn_module_stack is None:
        return False

    for _, packed in nn_module_stack.items():
        _, typename = packed
        for partial_name in nn_module_typenames:
            if partial_name in typename:
                return True

    return False


def get_tensor_name(exp_prog: ExportedProgram, node: torch.fx.Node) -> str:
    if node is None:
        return ""
    if is_param(exp_prog, node):
        return exp_prog.graph_signature.inputs_to_parameters[node.name]
    elif is_buffer(exp_prog, node):
        return exp_prog.graph_signature.inputs_to_buffers[node.name]
    elif is_lifted_tensor_constant(exp_prog, node):
        return exp_prog.graph_signature.inputs_to_lifted_tensor_constants[node.name]
    else:
        assert isinstance(node.target, str)
        return node.target

    return ""


def find_dequant_user(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """
    Search the direct users of the given node and return the first one that is a
    dequantization op. Returns None if no dequantization op is found.
    """
    for user in node.users:
        if is_dequant_node(user):
            return user
    return None


def find_quant_user(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """
    Search the direct users of the given node and return the first one that is a
    quantization op. Returns None if no quantization op is found.
    """
    for user in node.users:
        if is_quant_node(user):
            return user

    return None


##
## Memory Layout, Storage Type Determination
##

ImageExtents = Tuple[int, int, int]

DEFAULT_TEXTURE_LIMITS = (16384, 16384, 2048)
DEFAULT_BUFFER_LIMIT = 128 * (1024 * 1024)

all_storage_types: Set[VkStorageType] = {
    VkStorageType.BUFFER,
    VkStorageType.TEXTURE_3D,
}

all_memory_layouts: Set[VkMemoryLayout] = {
    VkMemoryLayout.TENSOR_WIDTH_PACKED,
    VkMemoryLayout.TENSOR_HEIGHT_PACKED,
    VkMemoryLayout.TENSOR_CHANNELS_PACKED,
    VkMemoryLayout.PACKED_INT8_4W4C,
    VkMemoryLayout.PACKED_INT8_4H4W,
}

MemoryLayoutSet = Set[VkMemoryLayout]
MemoryLayoutSetList = Union[MemoryLayoutSet, List[MemoryLayoutSet]]


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


def tensor_node_is_high_dim(node: torch.fx.Node) -> bool:
    """
    Returns true if a given node contains a tensor with more than 4 dimensions
    """
    if isinstance(node.meta["val"], FakeTensor):
        return len(node.meta["val"].shape) > 4
    if isinstance(node.meta["val"], list) or isinstance(node.meta["val"], tuple):
        for fake_tensor in node.meta["val"]:
            if isinstance(fake_tensor, FakeTensor):
                if len(fake_tensor.shape) > 4:
                    return True
    return False


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
    elif layout == VkMemoryLayout.PACKED_INT8_4W4C:
        width = (width + 3) // 4
        channels = (channels + 3) // 4
    elif layout == VkMemoryLayout.PACKED_INT8_4H4W:
        height = (height + 3) // 4
        width = (width + 3) // 4
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


class TensorRepr:
    """
    This class is a wrapper around a pair of VkStorageType and VkMemoryLayout which
    describes how a tensor should be represented in the Vulkan Delegate.
    """

    def __init__(self, storage_type: VkStorageType, memory_layout: VkMemoryLayout):
        self.storage_type = storage_type
        self.memory_layout = memory_layout

    def __str__(self) -> str:
        return f"TensorRepr({self.storage_type}, {self.memory_layout})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorRepr):
            return NotImplemented
        return (
            self.storage_type == other.storage_type
            and self.memory_layout == other.memory_layout
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


class TensorReprList:
    """
    This class is a wrapper around a list of TensorRepr instances that automatically
    applies a "broadcasting" mechanism. The broadcasting mechanism allows for a single
    underlying TensorRepr to be used to represent multiple tensors.
    """

    def __init__(self, tensor_reprs: Union[TensorRepr, List[TensorRepr]]):
        self.vals: List[TensorRepr] = (
            tensor_reprs if isinstance(tensor_reprs, list) else [tensor_reprs]
        )

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, idx: int) -> TensorRepr:
        if idx > 0 and len(self) == 1:
            return self.vals[0]
        else:
            return self.vals[idx]

    def __setitem__(self, idx: int, val: TensorRepr) -> None:
        if idx > 0 and len(self) == 1:
            self.vals[0] = val
        else:
            self.vals[idx] = val

    def __str__(self) -> str:
        return f"[{', '.join(str(ts) for ts in self.vals)}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorReprList):
            return NotImplemented

        if len(self) == len(other):
            for self_val, other_val in zip(self.vals, other.vals):
                if self_val != other_val:
                    return False

            return True

        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def append(self, val: TensorRepr) -> None:
        self.vals.append(val)

    def storage_type(self, idx: int = 0) -> VkStorageType:
        return self.vals[idx].storage_type

    def memory_layout(self, idx: int = 0) -> VkMemoryLayout:
        return self.vals[idx].memory_layout


class TensorRepSet:
    """
    This class describes the possible set of representations (i.e. TensorRepr) that may
    be used to represent a tensor. This set is determined by the implementation of the
    operator that the tensor participates in as well as the texture extents of the GPU.
    """

    def __init__(
        self,
        buffer_memory_layouts: Set[VkMemoryLayout],
        texture_memory_layouts: Set[VkMemoryLayout],
    ):
        self.valid_buffer_layouts = buffer_memory_layouts
        self.valid_texture_layouts = texture_memory_layouts

    def __str__(self) -> str:
        buffer_layouts = ", ".join(layout.name for layout in self.valid_buffer_layouts)
        texture_layouts = ", ".join(
            layout.name for layout in self.valid_texture_layouts
        )
        return f"TensorRepSet(Buffer Layouts: [{buffer_layouts}], Texture Layouts: [{texture_layouts}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorRepSet):
            return NotImplemented
        return (
            self.valid_buffer_layouts == other.valid_buffer_layouts
            and self.valid_texture_layouts == other.valid_texture_layouts
        )

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def is_empty(self) -> bool:
        """
        A TensorRepSet is "empty" if there are no valid representations of the tensor.
        """
        return (
            len(self.valid_buffer_layouts) == 0 and len(self.valid_texture_layouts) == 0
        )

    def make_intersect(self, other: "TensorRepSet") -> "TensorRepSet":
        """
        Merge this TensorRepr with another TensorRepr, returning a new TensorRepr
        with the intersection of the two.
        """
        return TensorRepSet(
            self.valid_buffer_layouts & other.valid_buffer_layouts,
            self.valid_texture_layouts & other.valid_texture_layouts,
        )

    def is_compatible(self, storage: TensorRepr) -> bool:
        """
        Check if this TensorRepr is compatible with the given TensorRepSet.
        """
        if storage.storage_type == VkStorageType.BUFFER:
            return storage.memory_layout in self.valid_buffer_layouts
        elif storage.storage_type == VkStorageType.TEXTURE_3D:
            return storage.memory_layout in self.valid_texture_layouts
        else:
            raise RuntimeError(f"Unsupported storage type {storage.storage_type}")

    def any_in_common(self, other: "TensorRepSet") -> bool:
        """
        Check if this TensorRepr has any representations in common with another
        TensorRepr.
        """
        return (
            len(self.valid_buffer_layouts & other.valid_buffer_layouts) > 0
            or len(self.valid_texture_layouts & other.valid_texture_layouts) > 0
        )

    def texture_is_valid(self):
        return len(self.valid_texture_layouts) > 0

    def buffer_is_valid(self):
        return len(self.valid_buffer_layouts) > 0

    def first_valid_buffer_layout(self):
        return list(self.valid_buffer_layouts)[0]

    def first_valid_texture_layout(self):
        return list(self.valid_texture_layouts)[0]

    def make_tensor_repr(self) -> TensorRepr:
        """
        Pick a representation (i.e. TensorRepr) from the set of possible representations.
        If there are multiple valid representations, then:
        1. Prefer texture storage over buffer storage
        2. Pick the first available memory layout.
        """
        if self.is_empty():
            # An empty repset typically means that it is associated with a weight tensor
            # or non tensor argument. In this case, just return default storage and
            # layout as placeholder.
            return TensorRepr(
                VkStorageType.DEFAULT_STORAGE, VkMemoryLayout.DEFAULT_LAYOUT
            )

        if self.texture_is_valid():
            return TensorRepr(
                VkStorageType.TEXTURE_3D, self.first_valid_texture_layout()
            )

        else:
            return TensorRepr(VkStorageType.BUFFER, self.first_valid_buffer_layout())

    def is_constrained(self) -> bool:
        """
        A "constrained" RepSet is one that has either:
        1. A single valid texture memory layout, and no valid buffer memory layouts
        2. No valid texture memory layouts, and a single valid buffer memory layout
        3. Is empty

        In this case, it is unambiguous which representation should be used for the
        tensor.
        """
        if self.is_empty():
            return True
        elif (
            len(self.valid_texture_layouts) == 1 and len(self.valid_buffer_layouts) == 0
        ):
            return True
        elif (
            len(self.valid_texture_layouts) == 0 and len(self.valid_buffer_layouts) == 1
        ):
            return True
        else:
            return False

    def is_ambiguous(self) -> bool:
        """
        An "ambiguous" RepSet is one that is not constrained.
        """
        return not self.is_constrained()


def make_tensor_repset(tensor_repr: TensorRepr) -> TensorRepSet:
    """
    Given a TensorRepr, return a TensorRepSet that contains only that TensorRepr
    """
    if tensor_repr.storage_type == VkStorageType.BUFFER:
        return TensorRepSet({tensor_repr.memory_layout}, set())
    elif tensor_repr.storage_type == VkStorageType.TEXTURE_3D:
        return TensorRepSet(set(), {tensor_repr.memory_layout})
    else:
        raise RuntimeError(f"Unsupported storage type {tensor_repr.storage_type}")


def make_filtered_tensor_repset(
    tensor_val: FakeTensor,
    tensor_repset: TensorRepSet,
    texture_limits: ImageExtents,
) -> TensorRepSet:
    """
    `tensor_val` represents an actual tensor participating in some operator computation.

    `tensor_repset` represents the set of valid tensor representations that may be used
    for that tensor that is supported by the op implementation.

    `texture_limits` represents the maximum texture sizes that is supported by the GPU.

    Given the above, return a new TensorRepSet that contains only texture layouts that
    can be used to produce a valid image texture for the given tensor (i.e. fits within
    texture limits).
    """
    valid_texture_layouts = set()
    for memory_layout in tensor_repset.valid_texture_layouts:
        extents = required_image_extents(tensor_val.shape, memory_layout)
        if extents_are_valid(extents, texture_limits):
            valid_texture_layouts.add(memory_layout)

    # High dimensional tensors require buffer storage
    if len(tensor_val.shape) > 4:
        return TensorRepSet(tensor_repset.valid_buffer_layouts, set())

    # Bool tensors are currently not supported
    if tensor_val.dtype == torch.bool:
        return NO_STORAGE

    return TensorRepSet(tensor_repset.valid_buffer_layouts, valid_texture_layouts)


## Convenience TensorRepSet definitions

PACKED_INT8_4W4C_BUFFER = TensorRepSet({VkMemoryLayout.PACKED_INT8_4W4C}, set())

CONTIGUOUS_ANY = TensorRepSet(
    {VkMemoryLayout.TENSOR_WIDTH_PACKED}, {VkMemoryLayout.TENSOR_WIDTH_PACKED}
)
CONTIGUOUS_BUFFER = TensorRepSet({VkMemoryLayout.TENSOR_WIDTH_PACKED}, set())

WIDTH_PACKED_TEXTURE = TensorRepSet(set(), {VkMemoryLayout.TENSOR_WIDTH_PACKED})
CHANNELS_PACKED_TEXTURE = TensorRepSet(set(), {VkMemoryLayout.TENSOR_CHANNELS_PACKED})

ANY_TEXTURE = TensorRepSet(set(), all_memory_layouts)
ANY_BUFFER = TensorRepSet(all_memory_layouts, set())

ANY_STORAGE = TensorRepSet(all_memory_layouts, all_memory_layouts)
NO_STORAGE = TensorRepSet(set(), set())


class TensorRepSetList:
    """
    This class is a wrapper around a list of TensorRepSet instances that automatically
    applies a "broadcasting" mechanism. The broadcasting mechanism allows for a single
    underlying TensorRepSet to be used for multiple tensors.
    """

    def __init__(
        self,
        tensor_repsets: Union[TensorRepSet, List[TensorRepSet]],
    ):
        self.vals: List[TensorRepSet] = (
            tensor_repsets if isinstance(tensor_repsets, list) else [tensor_repsets]
        )

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, idx: int) -> TensorRepSet:
        if idx > 0 and len(self) == 1:
            return self.vals[0]
        else:
            return self.vals[idx]

    def __setitem__(self, idx: int, val: TensorRepSet) -> None:
        if idx > 0 and len(self.vals) == 1:
            self.vals[0] = val
        else:
            self.vals[idx] = val

    def __str__(self) -> str:
        return f"[{', '.join(str(ts) for ts in self.vals)}]"

    def append(self, val: TensorRepSet) -> None:
        return self.vals.append(val)

    def any_is_empty(self) -> bool:
        if len(self.vals) == 0:
            return True

        return any(tensor_repr.is_empty() for tensor_repr in self.vals)


class OpRepSets:
    """
    This class is responsible for representing and managing the set of valid tensor
    representations that may be used for all input and output tensors of an operator.
    It is also responsible for maintaining synchronization rules between tensors
    participating in the computation.

    Currently, three synchronization rules exist:
    1. All input tensors must use the same representation (e.g. binary ops)
    2. The "primary" input and output tensors must use the same representation
       (e.g. group norm; the output is a tuple of out, mean, rstd; out must be the same
       representation as the first input x, but mean and rstd may use different
       representations as out)
    3. All output tensors must use the same representation (e.g. choose qparams)

    Note that "primary" input and output tensor refers to the first non-weight input
    tensor and the first output tensor. Note that Some operators (such as arange) do not
    have any tensor inputs.

    Currently, the above three synchronization rules are sufficient to describe the
    representation requirements of all ET-VK operators.

    This class also provides utilities to constrain the repsets; when applying the
    constraints, the synchronization rules will be maintained.
    """

    def __init__(  # noqa: C901
        self,
        inputs_repsets: TensorRepSetList,
        outputs_repsets: TensorRepSetList,
        op_node: torch.fx.Node,
        texture_limits: ImageExtents,
    ):
        self.op_node = op_node

        # inputs_repset_list is received from the operator registration. If a different
        # repset is defined for each input tensor, then assume that the input tensor
        # representations do not need to be synchronized.
        if len(inputs_repsets) > 1:
            self.sync_args_repr = False
        # Otherwise, default to True
        else:
            self.sync_args_repr = True

        # outputs_repset_list is received from the operator registration. If a different
        # repset is defined for each output tensor, then assume that the output tensor
        # representations do not need to be synchronized.
        if len(outputs_repsets) > 1:
            self.sync_outs_repr = False
        else:
            self.sync_outs_repr = True

        # Try to determine the index of the "primary" argument, i.e. the first non
        # constant tensor argument. For the vast majority of operators with tensor
        # arguments, this will be the first argument.
        self.primary_arg_idx: Optional[int] = None
        for i, arg_node in enumerate(self.op_node.args):
            arg_node_repset = inputs_repsets[i]
            if not is_tensor_arg_node(arg_node):
                continue
            if arg_node_repset is None:
                continue
            if arg_node_repset.is_empty():
                continue

            self.primary_arg_idx = i
            break

        # If the repset of the primary input and the primary output are the same, then
        # assume they need to be the same.
        self.sync_primary_io_repr = self.primary_arg_idx is not None
        if self.primary_arg_idx is not None:
            if inputs_repsets[self.primary_arg_idx] != outputs_repsets[0]:
                self.sync_primary_io_repr = False

        # Now, go through the arguments of the operator and create a filtered repset
        # for each based on the actual tensor value.
        args_repset_list = TensorRepSetList([])
        common_arg_repset = ANY_STORAGE
        for i, arg_node in enumerate(op_node.args):
            arg_repset = inputs_repsets[i]

            # Use ANY_STORAGE for non-tensor nodes so they don't cause the op repsets to
            # appear empty
            if not is_tensor_arg_node(arg_node):
                args_repset_list.append(ANY_STORAGE)
            # NO_STORAGE is used to denote that an input is either a non tensor arg or
            # a weight tensor that is not prepacked. Similar to the above, use
            # ANY_STORAGE in this case.
            elif arg_repset.is_empty():
                args_repset_list.append(ANY_STORAGE)
            else:
                assert not arg_repset.is_empty()

                arg_repset = self.make_valid_tensor_repset_for_arg(
                    arg_repset, arg_node, texture_limits
                )

                args_repset_list.append(arg_repset)
                common_arg_repset = common_arg_repset.make_intersect(arg_repset)

        # Repeat for output tensors.
        outs_repset_list = TensorRepSetList([])
        common_out_repset = ANY_STORAGE
        if num_tensors_in_node(op_node) == 1:
            common_out_repset = make_filtered_tensor_repset(
                op_node.meta["val"], outputs_repsets[0], texture_limits
            )
            outs_repset_list.append(common_out_repset)
        # Multiple output tensors
        else:
            for i, val in enumerate(op_node.meta["val"]):
                assert isinstance(val, FakeTensor)
                out_repset = make_filtered_tensor_repset(
                    val, outputs_repsets[i], texture_limits
                )

                outs_repset_list.append(out_repset)
                common_out_repset = common_out_repset.make_intersect(out_repset)

        # Apply synchronization rules; if either all inputs/outputs must use the same
        # representation, then only use a single underlying repset.
        if self.sync_args_repr:
            args_repset_list = TensorRepSetList([common_arg_repset])

        if self.sync_outs_repr:
            outs_repset_list = TensorRepSetList([common_out_repset])

        # Finally, apply synchronization rules that sync inputs and outputs. If input
        # or output repsets are updated, then maintain synchronization rules.
        if self.sync_primary_io_repr:
            assert self.primary_arg_idx is not None

            primary_in_repset = args_repset_list[self.primary_arg_idx]
            primary_out_repset = outs_repset_list[0]

            primary_repset = primary_in_repset.make_intersect(primary_out_repset)

            if self.sync_args_repr:
                args_repset_list = TensorRepSetList([primary_repset])
            else:
                assert self.primary_arg_idx is not None
                args_repset_list[self.primary_arg_idx] = primary_repset

            if self.sync_outs_repr:
                outs_repset_list = TensorRepSetList([primary_repset])
            else:
                assert self.primary_arg_idx is not None
                outs_repset_list[0] = primary_repset

        # Save the resulting repsets
        self.args_repset_list = args_repset_list
        self.outs_repset_list = outs_repset_list

        # Check that synchronization rules are respected.
        self.assert_sync_contraints()

    def __str__(self) -> str:
        return f"OpRepSets(ins={self.args_repset_list}, outs={self.outs_repset_list})"

    def make_valid_tensor_repset_for_node_list_arg(
        self,
        arg_repsets: TensorRepSet,
        arg_node: List[torch.fx.Node],
        texture_limits: ImageExtents,
    ) -> TensorRepSet:
        """
        Wrapper around make_filtered_tensor_repset for a list of nodes. This will happen
        for the cat operator, where the first argument is a list of nodes.
        """
        # For variable length args, assume that they all need to use the same representation
        # only one repset should be defined
        common_tensor_repsets = arg_repsets

        for n in arg_node:
            assert isinstance(n, torch.fx.Node)
            common_tensor_repsets = common_tensor_repsets.make_intersect(
                make_filtered_tensor_repset(
                    n.meta["val"], common_tensor_repsets, texture_limits
                )
            )

        return common_tensor_repsets

    def make_valid_tensor_repset_for_arg(
        self, arg_repsets: TensorRepSet, arg_node: Any, texture_limits: ImageExtents
    ) -> TensorRepSet:
        """
        Helper function to call make_filtered_tensor_repset
        """
        if isinstance(arg_node, torch.fx.Node) and is_single_tensor_node(arg_node):
            return make_filtered_tensor_repset(
                arg_node.meta["val"], arg_repsets, texture_limits
            )
        elif isinstance(arg_node, list) and all(
            is_single_tensor_node(n) for n in arg_node
        ):
            return self.make_valid_tensor_repset_for_node_list_arg(
                arg_repsets, arg_node, texture_limits
            )
        # Special case for getitem; return the repset of the particular val in the
        # list of tensors that is being extracted.
        elif (
            self.op_node.target == operator.getitem and arg_node == self.op_node.args[0]
        ):
            idx = self.op_node.args[1]
            assert isinstance(idx, int)
            return make_filtered_tensor_repset(
                arg_node.meta["val"][idx], arg_repsets, texture_limits
            )

        raise NotImplementedError(f"Unhandled node type {arg_node}")

    def assert_sync_contraints(self) -> None:
        if self.sync_args_repr:
            assert len(self.args_repset_list) == 1

        if self.sync_outs_repr:
            assert len(self.outs_repset_list) == 1

        if self.sync_primary_io_repr:
            assert (
                self.args_repset_list[self.primary_arg_idx] == self.outs_repset_list[0]
            )

    def any_is_empty(self) -> bool:
        return (
            self.args_repset_list.any_is_empty() or self.outs_repset_list.any_is_empty()
        )

    def get_arg_repset(self, i: int):
        return self.args_repset_list[i]

    def get_out_repset(self, i: int):
        return self.outs_repset_list[i]

    def try_constrain_with_arg_repset(
        self, arg_i: int, source_repset: TensorRepSet
    ) -> bool:
        """
        Attempt to constrain the repsets of the tensors participating in this operator
        based on an "existing" repset of an argument. The existing repset can have two
        sources:
        * A representation may have been determined for the argument already from a
          prior operator
        * The output repset of the operator which produces the argument

        If the existing repset of the argument is compatible with the current operator,
        then constrain the repsets of this operator and apply synchronization rules.

        This process tries to minimize the number of transition nodes that will need to
        be inserted by tag_memory_meta_pass.py by maintaining existing representations
        for as long as possible.
        """
        arg_current_repset = self.args_repset_list[arg_i]

        if arg_current_repset == source_repset:
            return False

        if not arg_current_repset.any_in_common(source_repset):
            return False

        if self.sync_primary_io_repr:
            if not self.get_out_repset(0).any_in_common(source_repset):
                return False

        # If this point is reached, then it is possible to constrain
        self.args_repset_list[arg_i] = arg_current_repset.make_intersect(source_repset)
        if self.sync_primary_io_repr and (
            arg_i == self.primary_arg_idx or self.sync_args_repr
        ):
            self.outs_repset_list[0] = arg_current_repset.make_intersect(source_repset)

        self.assert_sync_contraints()
        return True

    def pick_representations(self) -> Tuple[TensorReprList, TensorReprList]:
        """
        For each tensor participating in the op, pick a representation for it among the
        possible represetntation sets.
        """
        args_repr_list = TensorReprList([])
        outs_repr_list = TensorReprList([])

        for i in range(len(self.op_node.args)):
            arg_repset = self.args_repset_list[i]
            args_repr_list.append(arg_repset.make_tensor_repr())

        for i in range(num_tensors_in_node(self.op_node)):
            out_repset = self.outs_repset_list[i]
            outs_repr_list.append(out_repset.make_tensor_repr())

        return args_repr_list, outs_repr_list


##
## TensorSpec Utils
##


def has_node_spec_attr(node: torch.fx.Node, attr: str) -> bool:
    return "spec" in node.meta and hasattr(node.meta["spec"], attr)


def set_node_spec_attr(node: torch.fx.Node, attr: str, value):
    assert "spec" in node.meta
    spec = node.meta["spec"]
    if isinstance(spec, TensorSpec):
        setattr(spec, attr, value)
    elif isinstance(spec, (list, tuple)):
        # Special case if value is a list/tuple of the same length as the
        # collection of tensors in the node. In this case, treat the value list
        # as a list of values to set indivudually for each tensor in the node
        if isinstance(value, (list, tuple)) and len(spec) == len(value):
            assert len(spec) == len(value)
            for s, v in zip(spec, value):
                assert isinstance(s, TensorSpec)
                setattr(s, attr, v)
        # Otherwise, set the attribute to value for all tensors in the list
        else:
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


def has_node_repr(node) -> bool:
    if isinstance(node, (list, tuple)):
        return all(has_node_spec_attr(n, "etvk_node_repr") for n in node)
    else:
        return has_node_spec_attr(node, "etvk_node_repr")


def set_node_repr(node: torch.fx.Node, node_repr: Union[TensorRepr, TensorReprList]):
    if isinstance(node_repr, TensorReprList):
        # Convert to a regular list so taht `set_node_spec_attr` can attach each entry
        # to a separate TensorSpec
        node_repr_list = [node_repr[i] for i in range(num_tensors_in_node(node))]
        set_node_spec_attr(node, "etvk_node_repr", node_repr_list)
    else:
        set_node_spec_attr(node, "etvk_node_repr", node_repr)


def get_node_repr(node) -> Union[TensorRepr, TensorReprList]:
    if isinstance(node, (list, tuple)):
        raise NotImplementedError("get_node_repr not implemented for list of nodes")
    else:
        return get_node_spec_attr(node, "etvk_node_repr", False)


##
## Graph Pattern Matching
##


def maybe_skip_q_dq_arg_chain(
    arg: torch.fx.node.Argument,
) -> Tuple[Optional[torch.fx.Node], Optional[torch.fx.Node], Optional[torch.fx.Node]]:
    """
    Check if the given node argument is part of a Quantize/Dequantize chain produced by
    the quant workflow. If so, return the source tensor that is the input to the Q/DQ
    chain and the quantize/dequantize nodes in the chain. Otherwise, return the argument
    as is and None, None
    """
    if not isinstance(arg, torch.fx.Node):
        return None, None, None

    # If the arg is a view copy node, check if the original node is a dequant node
    if is_dequant_node(arg) or (
        is_view_copy_node(arg) and is_dequant_node(arg.args[0])  # pyre-ignore[6]
    ):
        dequant_node = arg
        if is_view_copy_node(arg):
            dequant_node = arg.args[0]

        quant_node = dequant_node.args[0]  # pyre-ignore[16]
        assert isinstance(quant_node, torch.fx.Node)
        source_arg = quant_node.args[0]
        assert isinstance(source_arg, torch.fx.Node)
        assert isinstance(dequant_node, torch.fx.Node)
        return source_arg, quant_node, dequant_node
    else:
        return arg, None, None


def trace_args_until_placeholder(
    node: torch.fx.node.Argument, max_search_depth: int = 4
) -> Tuple[Optional[torch.fx.Node], List[torch.fx.Node]]:
    """
    Trace through node.args[0] of a given initial node until a placeholder node is found
    then return it and the list of nodes traversed. If no placeholder node is found,
    returns None and an empty list.
    """
    cur_node = node
    search_depth = 0

    if not isinstance(cur_node, torch.fx.Node):
        return None, []

    traversed = [cur_node]
    while cur_node.op != "placeholder" and search_depth < max_search_depth:
        # Break if cur_node has no args
        if len(cur_node.args) == 0:
            break

        cur_node = cur_node.args[0]
        if not isinstance(cur_node, torch.fx.Node):
            break
        traversed.append(cur_node)
        search_depth += 1

    if not isinstance(cur_node, torch.fx.Node):
        return None, []
    if cur_node.op != "placeholder":
        return None, []

    assert isinstance(cur_node, torch.fx.Node)
    return cur_node, traversed


def is_in_4bit_range(tensor: torch.Tensor) -> bool:
    """
    Check if the given tensor is in the range of 4-bit quantization and is of integer type.
    """
    if tensor.dtype not in (torch.int8, torch.uint8):
        return False

    return tensor.min().item() >= -8 and tensor.max().item() <= 7


def is_in_8bit_range(tensor: torch.Tensor) -> bool:
    """
    Check if the given tensor is in the range of 4-bit quantization and is of integer type.
    """
    if tensor.dtype not in (torch.int8, torch.uint8):
        return False

    return tensor.min().item() >= -128 and tensor.max().item() <= 127


##
## Misc
##


def nchw_dim_to_whcn_dim(nchw_dim: int, ndim: int) -> int:
    # Handle negative indices for nchw_dim
    if nchw_dim < 0:
        nchw_dim += ndim

    assert nchw_dim >= 0 and nchw_dim < ndim
    whcn_dim = (ndim - 1) - nchw_dim
    return whcn_dim


def get_tensor_val_str(tensor_val: FakeTensor) -> str:
    return f"{tensor_val.dtype}: {tensor_val.shape}"


def get_node_val_str(node: torch.fx.Node) -> str:
    if is_single_tensor_node(node):
        assert isinstance(node.meta["val"], FakeTensor)
        return get_tensor_val_str(node.meta["val"])
    elif is_tensor_collection_node(node):
        assert isinstance(node.meta["val"], (list, tuple))
        return f"[{', '.join(get_tensor_val_str(t) for t in node.meta['val'])}]"
    else:
        if "val" not in node.meta:
            return str(node)
        return str(node.meta["val"])


def get_arg_node_val_str(arg_node: Any) -> str:
    if isinstance(arg_node, torch.fx.Node):
        return get_node_val_str(arg_node)
    elif isinstance(arg_node, (list, tuple)):
        return f"[{', '.join(get_arg_node_val_str(n) for n in arg_node)}]"
    else:
        return str(arg_node)


def node_io_str(node: torch.fx.Node) -> str:
    target = node.target
    if isinstance(target, EdgeOpOverload):
        assert isinstance(target, EdgeOpOverload)
        target_name = target.__name__
    elif isinstance(target, torch._ops.OpOverload):
        assert isinstance(target, torch._ops.OpOverload)
        target_name = target.name()
    else:
        target_name = str(target)

    out_str = f"{get_node_val_str(node)} = {target_name}("
    for arg in node.args:
        out_str += get_arg_node_val_str(arg) + ", "

    out_str += " ...)"
    return out_str


def update_program_state_dict(
    program: ExportedProgram,
    buffer_name: str,
    updated_tensor: torch.Tensor,
) -> None:
    target_name = None
    kind = None
    # Iterate over all the tensors in the graph signature, and find
    # the one corresponding to the parameter/buffer name
    for input_ in program.graph_signature.input_specs:
        if (
            input_.kind in (InputKind.BUFFER, InputKind.PARAMETER)
            and isinstance(input_.arg, TensorArgument)
            and input_.arg.name == buffer_name
        ):
            kind = input_.kind
            target_name = input_.target
            break

    # Assert that we found the parameter/buffer
    assert (
        target_name is not None
    ), f"could not find {buffer_name} in source program signature"
    assert target_name in program.state_dict, f"could not find {target_name}"

    if kind == InputKind.PARAMETER:
        updated_tensor = torch.nn.Parameter(updated_tensor, requires_grad=False)

    # Finally, overwrite the current tensor with updated tensor
    program.state_dict[target_name] = updated_tensor


def align_width_and_update_state_dict(
    ep: ExportedProgram,
    node: torch.fx.Node,
    cur_tensor: torch.Tensor,
    align_to: int = 4,
    force_update: bool = False,
) -> torch.Tensor:
    """
    Align the width of the given tensor to the given alignment value and update the
    state dict of the program with the aligned tensor.
    """
    added_padding = False
    cur_width = cur_tensor.shape[-1]
    # Only align the width of the tensor if it is not already aligned
    if cur_width % align_to != 0:
        num_padding = align_to - (cur_width % align_to)
        # Align the width of the tensor to the given alignment value
        aligned_tensor = torch.nn.functional.pad(
            cur_tensor, (0, num_padding)
        ).contiguous()
        added_padding = True
    else:
        aligned_tensor = cur_tensor

    if added_padding or force_update:
        update_program_state_dict(ep, node.name, aligned_tensor)
        # FakeTensor needs to match updated tensor
        cur_fake_tensor = node.meta["val"]
        node.meta["val"] = FakeTensorConverter().from_real_tensor(
            cur_fake_tensor.fake_mode,
            aligned_tensor,
        )

    return aligned_tensor
