# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide PyTorch-to-TOSA mapping helpers.

Use these utilities to translate PyTorch dtypes and FX node metadata into the
TOSA serializer types and shapes used during initial compilation.

"""

import operator
from enum import Enum
from typing import Any, Optional, Sequence

import torch
import tosa_serializer as ts
from executorch.backends.arm.tosa.specification import TosaSpecification

TOSA_TENSOR_NAME_META = "tosa_tensor_name"

UNSUPPORTED_DTYPES = (
    torch.float64,
    torch.double,
    torch.complex64,
    torch.cfloat,
    torch.complex128,
    torch.cdouble,
    torch.uint8,
    torch.int64,
    torch.long,
)


class TosaSpecialDtype(Enum):
    """Special TOSA dtypes not natively expressed in PyTorch."""

    INT48 = ts.DType.INT48
    INT4 = ts.DType.INT4

    def get_tosa_dtype(self) -> ts.DType:
        """Return the underlying ``ts.DType`` enumerant.

        Returns:
            ts.DType: Serializer dtype associated with the enum entry.

        """
        return self.value

    @staticmethod
    def meta_key() -> str:
        """Return the FX ``meta`` key that stores special dtypes.

        Returns:
            str: Metadata key used to encode :class:`TosaSpecialDtype`.

        """
        return "tosa_special_dtype"

    def max(self):
        match self:
            case self.INT4:
                return 7
            case self.INT48:
                return 2**47 - 1
            case _:
                raise ValueError(f"Unrecognized TosaSpecialDtype {self}.")

    def min(self):
        match self:
            case self.INT4:
                return -7
            case self.INT48:
                return -(2**47)
            case _:
                raise ValueError(f"Unrecognized TosaSpecialDtype {self}.")


def map_dtype(data_type: torch.dtype, tosa_spec: TosaSpecification) -> Any:
    """Map a ``torch.dtype`` to a ``ts.DType``.

    Args:
        data_type (torch.dtype): PyTorch dtype to convert.
        tosa_spec (TosaSpecification): Active spec (reserved for future checks).

    Returns:
        ts.DType: Matching serializer dtype.

    Raises:
        ValueError: If the dtype is unsupported or unknown.

    """
    if data_type in UNSUPPORTED_DTYPES:
        raise ValueError(f"Unsupported type: {data_type}")

    dtype_map = {
        torch.float32: ts.DType.FP32,
        torch.float: ts.DType.FP32,
        torch.float16: ts.DType.FP16,
        torch.half: ts.DType.FP16,
        torch.bfloat16: ts.DType.BF16,
        torch.int8: ts.DType.INT8,
        torch.int16: ts.DType.INT16,
        torch.short: ts.DType.INT16,
        torch.int32: ts.DType.INT32,
        torch.int: ts.DType.INT32,
        torch.bool: ts.DType.BOOL,
    }
    if data_type not in dtype_map:
        raise ValueError(f"Unknown type: {data_type}")
    return dtype_map[data_type]


# Returns the shape and type of a node
# TODO: other types, can be
# SymInt, FakeTensor, a List[Union[FakeTensor, SymInt]], or None
def extract_tensor_meta(meta, tosa_spec: TosaSpecification):
    """Extract dtype, shape, and dimension order from FX metadata.

    Args:
        meta (dict): FX node ``meta`` containing a ``val`` FakeTensor (or tuple).
        tosa_spec (TosaSpecification): Active TOSA spec for dtype mapping.

    Returns:
        tuple[ts.DType, tuple[int, ...], tuple[int, ...]]: Tuple containing
        tensor dtype, shape, and dimension order.

    Raises:
        ValueError: If ``meta['val']`` is not a ``FakeTensor``.

    """
    if meta.get("val") is None:
        raise ValueError("Expected node.meta['val'] to be set to a FakeTensor")
    val = meta["val"]
    if type(val) is tuple:
        # TODO: should use first concrete representation
        val = val[0]

    if not isinstance(val, torch._subclasses.fake_tensor.FakeTensor):
        raise ValueError(
            f"Expected first value in node.meta['val'] to be FakeTensor, got {val.__class__}"
        )
    dtype = map_dtype(val.dtype, tosa_spec)
    shape = tuple(val.size())

    if meta.get("tosa_dim_order") is not None:
        dim_order = meta["tosa_dim_order"]
    else:
        dim_order = tuple(range(len(shape)))
    return (dtype, shape, dim_order)


class TosaArg:
    """Capture and normalize TOSA operator arguments.

    Use this to convert FX nodes, sequences, and numeric literals into a
    consistent structure suitable for TOSA serialization.

    Attributes:
        name (str): Node name when argument is a ``torch.fx.Node``; empty
            otherwise.
        dtype (ts.DType | None): Inferred dtype when available.
        shape (tuple[int, ...] | None): Inferred shape when available.
        dim_order (tuple[int, ...] | None): Dimension order, defaulting to
            ``range(len(shape))``.
        special (list | None): Captured list when the argument is a sequence.
        number (float | int | None): Captured numeric value when provided.
        tosa_spec (TosaSpecification): Active specification used for mapping.
        multiple_output_name (list[str]): Output node names when node has multiple outputs; empty otherwise.
    """

    def __process_node(self, argument: torch.fx.Node):
        """Parse a ``torch.fx.Node`` and populate tensor attributes.

        Args:
            argument (torch.fx.Node): FX node to inspect.

        """
        suffix = argument.meta.get(TOSA_TENSOR_NAME_META, "")
        self.name = argument.name + suffix

        if "val" in argument.meta:
            output_dtype, self.shape, self.dim_order = extract_tensor_meta(
                argument.meta, self.tosa_spec
            )
            # Handle special case of types not representable in torch (i.e. i48_t)
            if special_type := argument.meta.get(TosaSpecialDtype.meta_key(), None):
                output_dtype = special_type.get_tosa_dtype()

            self.dtype = output_dtype

        # If all users of the node are getitems, node visitors should connect the output of this node directly to the getitem tensors.
        # Add a new attribute 'multiple_output_names' instead of making 'name' a list to avoid ambiguity regarding the type of 'name'.
        # Make name of the output is the first getitem since we in most cases only handle that output.
        users = list(argument.users)
        if len(users) > 0 and all(user.target == operator.getitem for user in users):
            self.multiple_output_names: list = [user.name + suffix for user in users]
            self.name = self.multiple_output_names[0]
        else:
            self.multiple_output_names = []

        if not self.__validate():
            raise ValueError(
                f"{self.tosa_spec} doesn't support tensor {self.__repr__()}"
            )

    def __process_list(self, argument):
        """Capture a sequence argument as ``special``.

        Args:
            argument (Sequence[Any]): Sequence to store.

        """
        self.special: list = list(argument)

    def __process_number(self, argument: float | int):
        """Capture a numeric argument as ``number``.

        Args:
            argument (float | int): Numeric value.

        """
        self.number: float | int = argument

    def __validate(self) -> bool:
        match getattr(self, "dtype", None):
            case ts.DType.FP32:
                if not self.tosa_spec.support_float():
                    return False
            case ts.DType.INT4:
                if not self.tosa_spec.support_extension("int4"):
                    return False

        return True

    def __init__(
        self, argument: Any, tosa_spec: Optional[TosaSpecification] = None
    ) -> None:
        """Initialize the argument wrapper and populate fields.

        Args:
            argument (Any): One of ``torch.fx.Node``, ``Sequence``, ``int``,
                ``float``, ``torch.dtype``, or ``None``.
            tosa_spec (Optional[TosaSpecification]): Active specification;
                required for metadata extraction.

        Raises:
            ValueError: If ``tosa_spec`` is missing or has the wrong type.
            RuntimeError: If ``argument`` is of an unsupported type.

        """
        if tosa_spec is None:
            raise ValueError("tosa_spec is None")
        elif not isinstance(tosa_spec, TosaSpecification):
            raise ValueError(
                f"Expected tosa_spec to be a TosaSpecification, but got {tosa_spec}"
            )
        self.tosa_spec = tosa_spec

        if isinstance(argument, torch.fx.Node):
            self.__process_node(argument)
            return
        if isinstance(argument, Sequence):
            self.__process_list(argument)
            return
        if isinstance(argument, (int, float)):
            self.__process_number(argument)
            return
        if isinstance(argument, torch.dtype):
            # Dtype is parsed from fake tensor
            return

        if argument is None:
            self.name = ""
            self.dtype = None
            self.shape = None
            self.dim_order = None
            return

        raise RuntimeError(
            f"Unhandled node input argument: {argument}, of type {type(argument)}"
        )

    def __repr__(self):
        """Return a compact representation of populated attributes.

        Returns:
            str: Readable list of set attributes.

        """
        attrs = []
        if hasattr(self, "name"):
            if self.name is not None:
                attrs.append(f"name={self.name!r}")
            if self.dtype is not None:
                attrs.append(f"dtype={ts.DTypeNames[self.dtype]}")
            if self.shape is not None:
                attrs.append(f"shape={self.shape!r}")
            if self.dim_order is not None:
                attrs.append(f"dim_order={self.dim_order!r}")
        if hasattr(self, "special") and self.special is not None:
            attrs.append(f"special={self.special!r}")
        if hasattr(self, "number") and self.number is not None:
            attrs.append(f"number={self.number!r}")
        if hasattr(self, "tosa_spec") and self.tosa_spec is not None:
            attrs.append(f"tosa_spec={self.tosa_spec!r}")
        if hasattr(self, "multiple_output_names"):
            attrs.append(f"names={self.multiple_output_names!r}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"
