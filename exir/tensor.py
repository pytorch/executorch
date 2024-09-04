# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[6]
# pyre-ignore-all-errors[16]
from __future__ import annotations

import copy

import math
import typing
from typing import Dict, List, Optional, Tuple, Union

import executorch.exir.schema as schema
import torch
from executorch.exir.error import internal_assert
from executorch.exir.schema import ScalarType, TensorShapeDynamism
from executorch.exir.sym_util import eval_shape


class AddressSpaceOverflowException(Exception):
    pass


def num_bytes_from_shape_and_dtype(shape: torch.Size, dtype: torch.dtype) -> int:
    """
    Assume the tensor is a contiguous one.
    """

    return math.prod(shape) * torch._utils._element_size(dtype)


def contiguous_stride_from_shape(shape: torch.Size) -> Tuple[int]:
    strides = []
    accum = 1
    for sz in reversed(shape):
        strides.append(accum)
        # For sizes[i] == 0, treat it as 1 to be consistent with core Pytorch
        # This preserves the PT equivalent behavior for dims with 0 elements
        if isinstance(sz, int):
            if sz != 0:
                accum *= sz
        else:
            # Unbacked symints may error on the != 0 check
            accum *= sz
    return tuple(reversed(strides))


def dim_order_from_stride(stride: Tuple[int]) -> Tuple[bytes]:
    """
    Dimension order represents how dimensions are laid out in memory,
    starting from the outer-most to the inner-most dimension.
    Thus, the conversion from strides is done by sorting the strides
    from larger to smaller since the dimension with the largest stride
    is the outer-most and the dimension with the smallest stride is the inner-most.
    For example, tensor with sizes = (3, 5, 2) and strides = (5, 1, 15), implies
    dimension order of (2, 0, 1). Dimension order of (2, 0, 1) can be obtained
    by sorting strides from large to smaller.

    When strides do not convey dimension order unambiguously, dimension order
    returned is dependent on stability of sort. In python same key elements are kept
    in original order. Thus when strides = (4, 3, 1, 1) returned value is (0, 1, 2, 3)
    Another example is: sizes = (1, 3, 1, 1) with strides = (3, 1, 3, 3), returned
    value is (0, 2, 3, 1)
    """
    for _, s in enumerate(stride):
        if s == 0:
            raise ValueError("0 in strides is not supported for ExecuTorch.")
    sorted_dims = [
        i[0] for i in sorted(enumerate(stride), key=lambda x: x[1], reverse=True)
    ]
    return tuple(typing.cast(Tuple[bytes], sorted_dims))


def stride_from_dim_order(sizes: List[int], dim_order: List[bytes]) -> List[int]:
    """
    Converts dim order to stride using sizes
    e.g. if sizes = (2, 3, 4) and dim_order = (0, 1, 2) then strides = (12, 4, 1)
    while for the same size if dim_order = (0, 2, 1) then strides = (12, 1, 3)
    See executorch/runtime/core/exec_aten/util/dim_order_util.h for details
    Args:
        sizes (Tuple[int]): sizes of the tensor
        dim_order (Tuple[bytes]): dim order of the tensor
    Returns:
        Tuple[int]: stride
    """
    if len(sizes) == 0:
        return []
    strides = copy.deepcopy(sizes)
    ndim = len(sizes)
    strides[dim_order[ndim - 1]] = 1
    for i in range(ndim - 2, -1, -1):
        if sizes[dim_order[i + 1]] == 0:
            strides[dim_order[i]] = strides[dim_order[i + 1]]
        else:
            strides[dim_order[i]] = sizes[dim_order[i + 1]] * strides[dim_order[i + 1]]
    return strides


def calculate_aligned_num_bytes(num: int, alignment: int) -> int:
    return math.ceil(num / alignment) * alignment


def determine_tensor_dynanism(shape: torch.Size) -> TensorShapeDynamism:
    if all(isinstance(s, int) for s in shape):
        return TensorShapeDynamism.STATIC
    else:
        try:
            _ = eval_shape(shape)
            return TensorShapeDynamism.DYNAMIC_BOUND
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            return TensorShapeDynamism.DYNAMIC_UNBOUND


ALIGNMENT = 16


class TensorSpec:
    """
    Captures the metadata for a given Tensor (ex. scalar type, storage, etc.).
    """

    def __init__(
        self,
        dtype: torch.dtype,
        shape: torch.Size,
        layout: torch.layout = torch.strided,
        is_sparse: bool = False,
        const: bool = False,
        requires_grad: bool = False,
    ) -> None:
        self.scalar_type = dtype
        self.const = const
        self.alignment: int = ALIGNMENT
        self.storage: Optional[torch.UntypedStorage] = None
        # convert to list making it easier to handle type checking
        self.shape: List[int] = list(shape)
        self.stride: Tuple[int] = contiguous_stride_from_shape(shape)
        self.dim_order: Tuple[bytes] = dim_order_from_stride(self.stride)
        self.requires_grad = requires_grad
        self.layout = layout
        self.is_sparse = is_sparse
        self.init_mem_planning_fields()
        self.shape_dynamism: TensorShapeDynamism = determine_tensor_dynanism(self.shape)

    @property
    def allocated_memory(self) -> int:
        nbytes = num_bytes_from_shape_and_dtype(self.shape, self.dtype)
        return calculate_aligned_num_bytes(nbytes, self.alignment)

    def realign(self, new_alignment: int) -> int:
        self.alignment = new_alignment
        return self.allocated_memory

    def nbytes(self) -> int:
        return num_bytes_from_shape_and_dtype(self.shape, self.dtype)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, const: bool = False) -> TensorSpec:
        if const:
            # for non-contigous tensors, convert to a contiguous one
            tensor = tensor.contiguous()
            # Weights cannot be views during emission or serialization
            if tensor.nbytes != tensor.untyped_storage().nbytes():
                tensor = tensor.clone()

        spec = cls(
            dtype=tensor.dtype,
            shape=tensor.shape,
            layout=tensor.layout,
            const=const,
            is_sparse=tensor.is_sparse,
        )
        spec.stride = tensor.stride()
        spec.dim_order = dim_order_from_stride(spec.stride)
        spec.requires_grad = tensor.requires_grad
        spec.storage = tensor.untyped_storage() if const else None

        return spec

    def init_mem_planning_fields(self) -> None:
        self.lifetime = [None, None]
        self.mem_id = None
        self.mem_obj_id = None
        self.mem_offset = None

    @property
    def dtype(self) -> torch.dtype:
        return self.scalar_type

    @property
    def is_dynamic_shape_tensor(self) -> bool:
        return self.shape_dynamism != schema.TensorShapeDynamism.STATIC

    @property
    def is_static_shape_tensor(self) -> bool:
        return self.shape_dynamism == TensorShapeDynamism.STATIC

    @property
    def is_upper_bound_tensor(self) -> bool:
        return self.shape_dynamism == TensorShapeDynamism.DYNAMIC_BOUND

    @property
    def is_dynamic_unbound_tensor(self) -> bool:
        return self.shape_dynamism == TensorShapeDynamism.DYNAMIC_UNBOUND

    def debug(self) -> str:
        return (
            f"TensorSpec(id={id(self)}, const={self.const}, scalar_type={self.scalar_type}"
            + f", allocated_memory={self.allocated_memory}, mem_id={self.mem_id}"
            + f", mem_offset={self.mem_offset}, lifetime={self.lifetime}"
            + f", shape_dynamism={self.shape_dynamism}"
            + (f", shape={self.shape}")
            + ")"
        )

    def __repr__(self) -> str:
        """
        Round-trippable printing function
        """
        return (
            f"TensorSpec(dtype={self.scalar_type}, shape={self.shape}"
            + f", layout={self.layout}"
            + f", is_sparse={self.is_sparse}"
            + f", shape_dynamism={self.shape_dynamism}"
            + f", const={self.const}, requires_grad={self.requires_grad}"
            + ")"
        )


def memory_format_enum(memory_format: torch.memory_format) -> int:
    internal_assert(
        isinstance(memory_format, torch.memory_format),
        "We only support torch.memory_format",
    )
    table = {
        torch.contiguous_format: 0,
        torch.preserve_format: 1,
    }
    return table[memory_format]


scalar_type_table: Dict[torch.dtype, ScalarType] = {
    torch.uint8: ScalarType.BYTE,
    torch.int8: ScalarType.CHAR,
    torch.int16: ScalarType.SHORT,
    torch.int32: ScalarType.INT,
    torch.int64: ScalarType.LONG,
    torch.half: ScalarType.HALF,
    torch.float: ScalarType.FLOAT,
    torch.double: ScalarType.DOUBLE,
    torch.complex32: ScalarType.COMPLEX32,
    torch.complex64: ScalarType.COMPLEX64,
    torch.complex128: ScalarType.COMPLEX128,
    torch.bool: ScalarType.BOOL,
    torch.qint8: ScalarType.QINT8,
    torch.quint8: ScalarType.QUINT8,
    torch.qint32: ScalarType.QINT32,
    torch.bfloat16: ScalarType.BFLOAT16,
    torch.quint4x2: ScalarType.QUINT4x2,
    torch.uint16: ScalarType.Bits16,
}


enum_to_scalar_map: Dict[ScalarType, torch.dtype] = {
    scalar_type_table[key]: key for key in scalar_type_table
}


def scalar_type_enum(dtype: torch.dtype) -> ScalarType:
    # TODO (zhengxu) single source of truth from c10/core/ScalarType.h.
    internal_assert(
        isinstance(dtype, torch.dtype), "We only support dtypes defined in Pytorch Core"
    )
    return scalar_type_table[dtype]


def get_scalar_type(enum: ScalarType) -> torch.dtype:
    return enum_to_scalar_map[enum]


def layout_enum(layout: torch.layout) -> int:
    # TODO single source of truth.
    table = {
        torch.strided: 0,
        torch.sparse_coo: 1,
    }
    return table[layout]


def make_allocation_info(mem_id: int, mem_offset: int) -> schema.AllocationDetails:
    """
    Creates the allocation_details object for creating tensors
    """
    if mem_offset < 0:
        raise ValueError(f"mem_offset {mem_offset} must not be negative")
    memory_offset_low = mem_offset & ((1 << 32) - 1)
    memory_offset_high = mem_offset >> 32
    if memory_offset_high >= 1 << 32:
        raise AddressSpaceOverflowException(
            f"mem_offset {mem_offset} does not fit in 64 bits"
        )

    allocation_info = schema.AllocationDetails(
        memory_id=mem_id,
        memory_offset_low=memory_offset_low,
        memory_offset_high=memory_offset_high,
    )
    return allocation_info


def make_tensor_value(
    data_buffer_idx: int,
    allocation_info: Optional[schema.AllocationDetails],
    spec: TensorSpec,
) -> schema.Tensor:
    """
    Converts the normal torch tensor to a flatbuffer tensor.
    """

    def to_list(
        x: Union[torch.Size, int, List[int], Tuple[int]]
    ) -> Union[List[int], List[torch.Size]]:
        if isinstance(x, torch.Size) or isinstance(x, tuple):
            return list(x)
        elif isinstance(x, int):
            return [x]
        else:
            return x

    tensor_size = to_list(spec.shape)
    tensor_dim_order = to_list(spec.dim_order)

    flatbuffer_tensor = schema.Tensor(
        scalar_type=scalar_type_enum(spec.scalar_type),
        # The runtime currently only supports tensors with offsets of zero.
        storage_offset=0,
        sizes=tensor_size,
        dim_order=tensor_dim_order,
        requires_grad=spec.requires_grad,
        data_buffer_idx=data_buffer_idx,
        allocation_info=allocation_info,
        layout=layout_enum(spec.layout),
        shape_dynamism=spec.shape_dynamism,
    )
    return flatbuffer_tensor


def check_spec(tensor: torch.Tensor, spec: TensorSpec) -> None:
    internal_assert(
        tensor.is_sparse == spec.is_sparse,
        f"Tensor attribute 'is_sparse' is expected to be equal to '{spec.is_sparse}', actually got: '{tensor.is_sparse}'",
    )
    internal_assert(
        tensor.shape == spec.shape,
        f"Tensor attribute 'shape' is expected to be equal to '{spec.shape}', actually got: '{tensor.shape}'",
    )
    internal_assert(
        tensor.dtype == spec.dtype,
        f"Tensor attribute 'dtype' is expected to be equal to '{spec.dtype}', actually got: '{tensor.dtype}'",
    )
