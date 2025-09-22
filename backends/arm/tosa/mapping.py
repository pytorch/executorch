# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""Provide PyTorch-to-TOSA mapping helpers.

Use these utilities to translate PyTorch dtypes and FX node metadata into
the TOSA serializer types and shapes used during initial compilation.

"""

from typing import Any, Optional, Sequence

import serializer.tosa_serializer as ts  # type: ignore

import torch
from executorch.backends.arm.tosa.specification import TosaSpecification

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


def map_dtype(data_type: torch.dtype, tosa_spec: TosaSpecification) -> Any:
    """Map a ``torch.dtype`` to a ``ts.DType``.

    Args:
        data_type (torch.dtype): PyTorch dtype to convert.
        tosa_spec (TosaSpecification): Active spec (reserved for future checks).

    Returns:
        Any: Matching ``ts.DType`` enum value.

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
        tuple: ``(dtype, shape, dim_order)`` where ``dtype`` is ``ts.DType``,
        ``shape`` is ``Tuple[int, ...]``, and ``dim_order`` is ``Tuple[int, ...]``.

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
        name (str): Node name when argument is a ``torch.fx.Node``; empty otherwise.
        dtype (ts.DType | None): Inferred dtype when available.
        shape (tuple[int, ...] | None): Inferred shape when available.
        dim_order (tuple[int, ...] | None): Dimension order, defaulting to ``range(len(shape))``.
        special (list | None): Captured list when the argument is a sequence.
        number (float | int | None): Captured numeric value when given.
        tosa_spec (TosaSpecification): Active specification used for mapping.

    """

    def __process_node(self, argument: torch.fx.Node):
        """Parse a ``torch.fx.Node`` and populate tensor attributes.

        Args:
            argument (torch.fx.Node): FX node to inspect.

        """
        self.name: str = argument.name
        self.dtype, self.shape, self.dim_order = extract_tensor_meta(
            argument.meta, self.tosa_spec
        )

    def __process_list(self, argument):
        """Capture a sequence argument as ``special``.

        Args:
            argument (Sequence): Sequence to store.

        """
        self.special: list = list(argument)

    def __process_number(self, argument: float | int):
        """Capture a numeric argument as ``number``.

        Args:
            argument (float | int): Numeric value.

        """
        self.number: float | int = argument

    def __init__(
        self, argument: Any, tosa_spec: Optional[TosaSpecification] = None
    ) -> None:
        """Initialize the argument wrapper and populate fields.

        Args:
            argument (Any): One of ``torch.fx.Node``, ``Sequence``, ``int``, ``float``, ``torch.dtype``, or ``None``.
            tosa_spec (Optional[TosaSpecification]): Active specification; required.

        Raises:
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
        return f"{self.__class__.__name__}({', '.join(attrs)})"
