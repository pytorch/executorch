#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MLX Op Handlers - registered handlers for converting ATen/custom ops to MLX.

This module contains all the op handler functions registered with the MLXOpRegistry.
Each handler converts a specific PyTorch operation to the corresponding MLX graph node.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from executorch.backends.mlx.builder.op_helpers import (
    emit_lifted_constant,
    emit_quantized_biases,
    parse_dequant_node,
    to_mlx_qparams,
    torch_dtype_to_scalar_type,
)
from executorch.backends.mlx.builder.op_registry import REGISTRY
from executorch.backends.mlx.builder.program_builder import MLXProgramBuilder
from executorch.backends.mlx.builder.slot_manager import IdType, Slot
from executorch.backends.mlx.serialization.mlx_graph_schema import (
    AbsNode,
    AddIntNode,
    AddmmNode,
    AddNode,
    AllNode,
    AnyNode,
    ARangeNode,
    ArccoshNode,
    ArccosNode,
    ArcsinhNode,
    ArcsinNode,
    ArctanhNode,
    ArctanNode,
    ArgmaxNode,
    ArgminNode,
    ArgPartitionNode,
    ArgsortNode,
    AsStridedNode,
    AsTypeNode,
    Atan2Node,
    BroadcastToNode,
    CeilNode,
    ClipNode,
    ConcatenateNode,
    ContiguousNode,
    Conv1DNode,
    Conv2DNode,
    Conv3DNode,
    ConvTranspose1DNode,
    ConvTranspose2DNode,
    ConvTranspose3DNode,
    CoshNode,
    CosNode,
    CumsumNode,
    DequantizeNode,
    DivideNode,
    EqualNode,
    ErfNode,
    ExpandDimsNode,
    Expm1Node,
    ExpNode,
    FloatOrVid,
    FloorDivideIntNode,
    FloorDivideNode,
    FloorNode,
    FullLikeNode,
    FullNode,
    GatherNode,
    GeluNode,
    GreaterEqualNode,
    GreaterNode,
    IdCopyNode,
    IntOrVid,
    IntOrVidOrTid,
    ItemIntNode,
    LayerNormNode,
    LessEqualNode,
    LessNode,
    Log10Node,
    Log1pNode,
    Log2Node,
    LogAddExpNode,
    LogicalAndNode,
    LogicalNotNode,
    LogicalOrNode,
    LogNode,
    LogSumExpNode,
    MaximumNode,
    MaxNode,
    MeanNode,
    MinimumNode,
    MinNode,
    ModIntNode,
    MultiplyIntNode,
    MultiplyNode,
    NegNode,
    NotEqualNode,
    PadNode,
    PartitionNode,
    PowerNode,
    ProdNode,
    ReciprocalNode,
    RemainderNode,
    RepeatNode,
    ReshapeNode,
    RMSNormNode,
    RopeNode,
    RoundNode,
    RsqrtNode,
    ScatterAddNode,
    SigmoidNode,
    SignNode,
    SiluNode,
    SinhNode,
    SinNode,
    SliceNode,
    SliceUpdateNode,
    SoftmaxNode,
    SortNode,
    SplitNode,
    SqrtNode,
    SquareNode,
    SqueezeNode,
    StackNode,
    StdNode,
    SubtractIntNode,
    SubtractNode,
    SumNode,
    SymSizeNode,
    TakeAlongAxisNode,
    TakeNode,
    TanhNode,
    TanNode,
    TileNode,
    TransposeNode,
    TrilNode,
    TriuNode,
    VarNode,
    VidOrTid,
    WhereNode,
)

# The coding style is for handlers to register against aten targets
# The corresponding edge ops are automatically registered
# For ops that are not in aten (e.g., dim order ops), directly register on exir_ops
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.node import Node


def require_static_int(value: Any, param_name: str, op_name: str) -> None:
    """
    Validate that a parameter is a static integer (not a Slot/SymInt).

    Raises NotImplementedError if the value is dynamic.

    Args:
        value: The parameter value to check
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if isinstance(value, Slot) or not isinstance(value, int):
        raise NotImplementedError(
            f"{op_name} with dynamic {param_name} is not supported. "
            f"{param_name} requires a static int32 value, but got {value} (type={type(value).__name__})."
        )


def require_static_float(value: Any, param_name: str, op_name: str) -> None:
    """
    Validate that a parameter is a static float (not a Slot/SymFloat).

    Raises NotImplementedError if the value is dynamic.

    Args:
        value: The parameter value to check
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if isinstance(value, Slot) or not isinstance(value, (int, float)):
        raise NotImplementedError(
            f"{op_name} with dynamic {param_name} is not supported. "
            f"{param_name} requires a static float value, but got {value} (type={type(value).__name__})."
        )


def require_static_ints(
    values: Union[List[Any], Any], param_name: str, op_name: str
) -> None:
    """
    Validate that all values in a list are static integers (not Slots/SymInts).

    Raises NotImplementedError if any value is dynamic.

    Args:
        values: List of values to check, or a single value
        param_name: Name of the parameter (for error message)
        op_name: Name of the operation (for error message)
    """
    if not isinstance(values, list):
        values = [values]

    for v in values:
        require_static_int(v, param_name, op_name)


def require_args(
    args: List[Any],
    min_count: int,
    max_count: int,
    op_name: str,
) -> None:
    """
    Validate that args count is within expected range.

    Raises ValueError if the count is outside the expected range.

    Args:
        args: The handler args list
        min_count: Minimum number of args expected
        max_count: Maximum number of args expected
        op_name: Name of the operation (for error message)
    """
    if not (min_count <= len(args) <= max_count):
        if min_count == max_count:
            raise ValueError(f"{op_name}: expected {min_count} args, got {len(args)}")
        raise ValueError(
            f"{op_name}: expected {min_count}-{max_count} args, got {len(args)}"
        )


def require_kwargs(
    kwargs: Dict[str, Any],
    allowed: Set[str],
    op_name: str,
) -> None:
    """
    Validate that only allowed kwargs are present.

    Raises ValueError if unexpected kwargs are found.

    Args:
        kwargs: The handler kwargs dict
        allowed: Set of allowed kwarg names
        op_name: Name of the operation (for error message)
    """
    unexpected = set(kwargs.keys()) - allowed
    if unexpected:
        raise ValueError(f"{op_name}: unexpected kwargs: {unexpected}")


def require_contiguous_format(
    *,
    layout=None,
    memory_format=None,
    dim_order=None,
    op_name: str,
) -> None:
    """
    Validate that layout/memory_format/dim_order specify contiguous format.

    MLX only supports contiguous (strided) tensors. Raises ValueError if
    sparse layouts or non-contiguous memory formats are requested.

    Args:
        layout: The torch layout (e.g., torch.strided, torch.sparse_coo)
        memory_format: The torch memory format (e.g., torch.contiguous_format,
            torch.channels_last)
        dim_order: The dimension order (list of ints, identity = contiguous)
        op_name: Name of the operation (for error message)
    """
    if layout is not None and layout != torch.strided:
        raise ValueError(f"{op_name}: only strided layout supported, got {layout}")

    if memory_format is not None and memory_format not in (
        torch.contiguous_format,
        torch.preserve_format,
    ):
        raise ValueError(
            f"{op_name}: only contiguous memory format supported, got {memory_format}"
        )

    if dim_order is not None:
        if list(dim_order) != list(range(len(dim_order))):
            raise ValueError(
                f"{op_name}: only contiguous dim_order supported, got {dim_order}"
            )


def is_static_value(value: Any) -> bool:
    """
    Check if a value is static (not a Slot/SymInt).

    Returns:
        True if the value is a static scalar (int, float, bool), False otherwise
    """
    return not isinstance(value, Slot)


def used_getitem_indices(n: Node) -> Set[int]:
    """Return the set of getitem indices actually consumed downstream.

    Only includes indices where the getitem node has at least one user.
    """
    return {
        user.args[1]
        for user in n.users
        if user.target == operator.getitem and len(user.users) > 0
    }


def normalize_reduction_dim(
    args: List[Any], start_idx: int = 1
) -> Tuple[Optional[List[int]], bool]:
    """
    Normalize dim argument for reduction operations.

    Extracts and normalizes the dim argument from handler args, returning a list of axes
    and the keepdim flag. Handles both list-based dims (e.g., sum.dim_IntList) and
    single int dims (e.g., prod.dim_int).

    Args:
        args: The handler args list
        start_idx: Index where the dim argument starts (default 1, after self)

    Returns:
        Tuple of (axes, keepdim) where:
        - axes: List of dimension indices, or empty list for reduce-all
        - keepdim: Boolean keepdim flag (default False)
    """
    if len(args) > start_idx and isinstance(args[start_idx], (list, tuple)):
        dim = list(args[start_idx])
        keepdim = args[start_idx + 1] if len(args) > start_idx + 1 else False
    elif len(args) > start_idx and isinstance(args[start_idx], int):
        dim = [args[start_idx]]
        keepdim = args[start_idx + 1] if len(args) > start_idx + 1 else False
    else:
        dim = []
        keepdim = False

    return dim, keepdim


_UNARY_OPS: List[Tuple[Any, Any, str]] = [
    # Activations
    (torch.ops.aten.silu.default, SiluNode, "aten.silu"),
    (torch.ops.aten.sigmoid.default, SigmoidNode, "aten.sigmoid"),
    (torch.ops.aten.tanh.default, TanhNode, "aten.tanh"),
    # Reciprocal square root
    (torch.ops.aten.rsqrt.default, RsqrtNode, "aten.rsqrt"),
    # Rounding
    (torch.ops.aten.floor.default, FloorNode, "aten.floor"),
    (torch.ops.aten.ceil.default, CeilNode, "aten.ceil"),
    # Powers / roots
    (torch.ops.aten.square.default, SquareNode, "aten.square"),
    (torch.ops.aten.exp.default, ExpNode, "aten.exp"),
    (torch.ops.aten.sqrt.default, SqrtNode, "aten.sqrt"),
    (torch.ops.aten.reciprocal.default, ReciprocalNode, "aten.reciprocal"),
    # Trigonometric
    (torch.ops.aten.sin.default, SinNode, "aten.sin"),
    (torch.ops.aten.cos.default, CosNode, "aten.cos"),
    (torch.ops.aten.tan.default, TanNode, "aten.tan"),
    (torch.ops.aten.asin.default, ArcsinNode, "aten.asin"),
    (torch.ops.aten.acos.default, ArccosNode, "aten.acos"),
    (torch.ops.aten.atan.default, ArctanNode, "aten.atan"),
    # Hyperbolic
    (torch.ops.aten.sinh.default, SinhNode, "aten.sinh"),
    (torch.ops.aten.cosh.default, CoshNode, "aten.cosh"),
    (torch.ops.aten.asinh.default, ArcsinhNode, "aten.asinh"),
    (torch.ops.aten.acosh.default, ArccoshNode, "aten.acosh"),
    (torch.ops.aten.atanh.default, ArctanhNode, "aten.atanh"),
    # Logarithmic
    (torch.ops.aten.log.default, LogNode, "aten.log"),
    (torch.ops.aten.log2.default, Log2Node, "aten.log2"),
    (torch.ops.aten.log10.default, Log10Node, "aten.log10"),
    (torch.ops.aten.log1p.default, Log1pNode, "aten.log1p"),
    # Special
    (torch.ops.aten.erf.default, ErfNode, "aten.erf"),
    (torch.ops.aten.expm1.default, Expm1Node, "aten.expm1"),
    # Sign / magnitude
    (torch.ops.aten.abs.default, AbsNode, "aten.abs"),
    (torch.ops.aten.neg.default, NegNode, "aten.neg"),
    (torch.ops.aten.sign.default, SignNode, "aten.sign"),
    # Logical
    (torch.ops.aten.logical_not.default, LogicalNotNode, "aten.logical_not"),
]


def _make_unary_handler(node_cls: Any, op_name: str):
    """Create a handler for a simple unary op: x → node_cls(x, out)."""

    def handler(P: MLXProgramBuilder, n: Node) -> Slot:
        args = P.args(n)
        require_args(args, 1, 1, op_name)
        require_kwargs(P.kwargs(n), set(), op_name)
        x = args[0]
        out = P.make_or_get_slot(n)
        P.emit(node_cls(x=P.slot_to_tid(x), out=P.slot_to_tid(out)))
        return out

    handler.__name__ = f"_{op_name.replace('.', '_')}_handler"
    handler.__doc__ = f"Handle {op_name} (table-driven unary op)."
    return handler


for _target, _node_cls, _op_name in _UNARY_OPS:
    REGISTRY.register(target=[_target])(_make_unary_handler(_node_cls, _op_name))


_BINARY_OPS: List[Tuple[List[Any], Any, str, bool]] = [
    (
        [torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar],
        MultiplyNode,
        "aten.mul",
        True,
    ),
    (
        [torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar],
        DivideNode,
        "aten.div",
        True,
    ),
    (
        [torch.ops.aten.remainder.Tensor, torch.ops.aten.remainder.Scalar],
        RemainderNode,
        "aten.remainder",
        True,
    ),
    (
        [torch.ops.aten.pow.Tensor_Tensor, torch.ops.aten.pow.Tensor_Scalar],
        PowerNode,
        "aten.pow",
        True,
    ),
    (
        [torch.ops.aten.floor_divide.default],
        FloorDivideNode,
        "aten.floor_divide",
        False,
    ),
    ([torch.ops.aten.maximum.default], MaximumNode, "aten.maximum", False),
    ([torch.ops.aten.minimum.default], MinimumNode, "aten.minimum", False),
    ([torch.ops.aten.atan2.default], Atan2Node, "aten.atan2", False),
    ([torch.ops.aten.logaddexp.default], LogAddExpNode, "aten.logaddexp", False),
    ([torch.ops.aten.logical_or.default], LogicalOrNode, "aten.logical_or", False),
    (
        [torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Scalar],
        LessNode,
        "aten.lt",
        True,
    ),
    (
        [torch.ops.aten.le.Tensor, torch.ops.aten.le.Scalar],
        LessEqualNode,
        "aten.le",
        True,
    ),
    (
        [torch.ops.aten.gt.Tensor, torch.ops.aten.gt.Scalar],
        GreaterNode,
        "aten.gt",
        True,
    ),
    (
        [torch.ops.aten.ge.Tensor, torch.ops.aten.ge.Scalar],
        GreaterEqualNode,
        "aten.ge",
        True,
    ),
    (
        [torch.ops.aten.eq.Tensor, torch.ops.aten.eq.Scalar],
        EqualNode,
        "aten.eq",
        True,
    ),
    (
        [torch.ops.aten.ne.Tensor, torch.ops.aten.ne.Scalar],
        NotEqualNode,
        "aten.ne",
        True,
    ),
]


def _make_binary_handler(node_cls: Any, op_name: str, lift_b: bool):
    """Create a handler for a binary op: (a, b) -> node_cls(a, b, out).

    When lift_b is True, scalar b values are lifted to 0-D constant tensors
    via emit_lifted_constant, using a's dtype.
    """

    def handler(P: MLXProgramBuilder, n: Node) -> Slot:
        args = P.args(n)
        require_args(args, 2, 2, op_name)
        require_kwargs(P.kwargs(n), set(), op_name)
        a, b = args[0], args[1]
        if lift_b and (not isinstance(b, Slot) or b.id_type != IdType.Tensor):
            input_meta = n.args[0].meta.get("val")
            dtype = input_meta.dtype if input_meta is not None else torch.float32
            b = emit_lifted_constant(P, b, dtype)
        out = P.make_or_get_slot(n)
        P.emit(node_cls(a=P.slot_to_tid(a), b=P.slot_to_tid(b), out=P.slot_to_tid(out)))
        return out

    handler.__name__ = f"_{op_name.replace('.', '_')}_handler"
    handler.__doc__ = f"Handle {op_name} (table-driven binary op)."
    return handler


for _targets, _node_cls, _op_name, _lift_b in _BINARY_OPS:
    REGISTRY.register(target=_targets)(
        _make_binary_handler(_node_cls, _op_name, _lift_b)
    )


_SCALAR_INT_OPS: List[Tuple[Any, Any, str]] = [
    (operator.add, AddIntNode, "operator.add"),
    (operator.sub, SubtractIntNode, "operator.sub"),
    (operator.mul, MultiplyIntNode, "operator.mul"),
    (operator.floordiv, FloorDivideIntNode, "operator.floordiv"),
    (operator.mod, ModIntNode, "operator.mod"),
]


def _make_scalar_int_handler(node_cls: Any, op_name: str):
    """Create a handler for a scalar int op: (a, b) -> node_cls(a, b, out)."""

    def handler(P: MLXProgramBuilder, n: Node) -> Slot:
        args = P.args(n)
        require_args(args, 2, 2, op_name)
        require_kwargs(P.kwargs(n), set(), op_name)
        a, b = args
        out = P.make_or_get_slot(n)
        P.emit(
            node_cls(
                a=P.to_int_or_vid(a),
                b=P.to_int_or_vid(b),
                out=P.slot_to_vid(out),
            )
        )
        return out

    handler.__name__ = f"_{op_name.replace('.', '_')}_handler"
    handler.__doc__ = f"Handle {op_name} (table-driven scalar int op)."
    return handler


for _target, _node_cls, _op_name in _SCALAR_INT_OPS:
    REGISTRY.register(target=[_target])(_make_scalar_int_handler(_node_cls, _op_name))


_REDUCTION_OPS: List[Tuple[List[Any], Any, str, int]] = [
    (
        [torch.ops.aten.sum.dim_IntList, torch.ops.aten.sum.default],
        SumNode,
        "aten.sum",
        4,
    ),
    ([torch.ops.aten.mean.dim, torch.ops.aten.mean.default], MeanNode, "aten.mean", 4),
    (
        [torch.ops.aten.prod.dim_int, torch.ops.aten.prod.default],
        ProdNode,
        "aten.prod",
        4,
    ),
    ([torch.ops.aten.amax.default], MaxNode, "aten.amax", 3),
    ([torch.ops.aten.amin.default], MinNode, "aten.amin", 3),
    ([torch.ops.aten.any.dim, torch.ops.aten.any.default], AnyNode, "aten.any", 3),
    ([torch.ops.aten.all.dim, torch.ops.aten.all.default], AllNode, "aten.all", 3),
]


def _make_reduction_handler(node_cls: Any, op_name: str, max_args: int):
    """Create a handler for a reduction op: x -> node_cls(x, out, axes, keepdims)."""

    def handler(P: MLXProgramBuilder, n: Node) -> Slot:
        args = P.args(n)
        require_args(args, 1, max_args, op_name)
        require_kwargs(P.kwargs(n), set(), op_name)
        x = args[0]
        axes, keepdim = normalize_reduction_dim(args)
        out = P.make_or_get_slot(n)
        P.emit(
            node_cls(
                x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=axes, keepdims=keepdim
            )
        )
        return out

    handler.__name__ = f"_{op_name.replace('.', '_')}_handler"
    handler.__doc__ = f"Handle {op_name} (table-driven reduction op)."
    return handler


for _targets, _node_cls, _op_name, _max_args in _REDUCTION_OPS:
    REGISTRY.register(target=_targets)(
        _make_reduction_handler(_node_cls, _op_name, _max_args)
    )


_FULL_OPS: List[Tuple[List[Any], str, Optional[float]]] = [
    ([torch.ops.aten.full.default], "aten.full", None),
    ([torch.ops.aten.zeros.default], "aten.zeros", 0.0),
    ([torch.ops.aten.ones.default], "aten.ones", 1.0),
]


def _make_full_handler(op_name: str, fixed_fill: Optional[float]):
    """Create a handler for full/zeros/ones: shape -> FullNode(shape, v, dtype)."""

    has_fill_arg = fixed_fill is None
    n_args = 2 if has_fill_arg else 1

    def handler(P: MLXProgramBuilder, n: Node) -> Slot:
        args = P.args(n)
        require_args(args, n_args, n_args, op_name)
        kwargs = P.kwargs(n)
        require_kwargs(kwargs, {"dtype", "layout", "device", "pin_memory"}, op_name)
        require_contiguous_format(layout=kwargs.get("layout"), op_name=op_name)

        shape = args[0]
        shape_iovs = [P.to_int_or_vid(d) for d in shape]
        v = (
            P.to_float_or_vid(args[1])
            if has_fill_arg
            else FloatOrVid.from_literal(fixed_fill)
        )
        dtype = n.kwargs.get("dtype")
        if dtype is None:
            dtype = torch.float32

        out = P.make_or_get_slot(n)
        P.emit(
            FullNode(
                out=P.slot_to_tid(out),
                shape=shape_iovs,
                v=v,
                scalar_type=torch_dtype_to_scalar_type(dtype),
            )
        )
        return out

    handler.__name__ = f"_{op_name.replace('.', '_')}_handler"
    handler.__doc__ = f"Handle {op_name} (table-driven full op)."
    return handler


for _targets, _op_name, _fixed_fill in _FULL_OPS:
    REGISTRY.register(target=_targets)(_make_full_handler(_op_name, _fixed_fill))


_FULL_LIKE_OPS: List[Tuple[List[Any], str, Optional[float]]] = [
    ([torch.ops.aten.full_like.default], "aten.full_like", None),
    ([torch.ops.aten.zeros_like.default], "aten.zeros_like", 0.0),
    ([torch.ops.aten.ones_like.default], "aten.ones_like", 1.0),
]


def _make_full_like_handler(op_name: str, fixed_fill: Optional[float]):
    """Create a handler for full_like/zeros_like/ones_like: x -> FullLikeNode(x, v, dtype)."""

    has_fill_arg = fixed_fill is None
    n_args = 2 if has_fill_arg else 1

    def handler(P: MLXProgramBuilder, n: Node) -> Slot:
        args = P.args(n)
        require_args(args, n_args, n_args, op_name)
        kwargs = P.kwargs(n)
        require_kwargs(
            kwargs,
            {"dtype", "layout", "device", "pin_memory", "memory_format"},
            op_name,
        )
        require_contiguous_format(
            layout=kwargs.get("layout"),
            memory_format=kwargs.get("memory_format"),
            op_name=op_name,
        )

        x = args[0]
        v = (
            P.to_float_or_vid(args[1])
            if has_fill_arg
            else FloatOrVid.from_literal(fixed_fill)
        )
        dtype = n.kwargs.get("dtype")

        out = P.make_or_get_slot(n)
        P.emit(
            FullLikeNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
                v=v,
                scalar_type=(
                    torch_dtype_to_scalar_type(dtype) if dtype is not None else None
                ),
            )
        )
        return out

    handler.__name__ = f"_{op_name.replace('.', '_')}_handler"
    handler.__doc__ = f"Handle {op_name} (table-driven full_like op)."
    return handler


for _targets, _op_name, _fixed_fill in _FULL_LIKE_OPS:
    REGISTRY.register(target=_targets)(_make_full_like_handler(_op_name, _fixed_fill))


@REGISTRY.register(target=["NOOP", torch.ops.aten._assert_scalar.default])
def _noop_handler(P: MLXProgramBuilder, n: Node) -> None:
    """No-op handler for nodes that don't emit any MLX instructions."""
    return None


# Handler for auto_functionalized_v2 higher-order op
# This handles mutating ops that have been functionalized
@REGISTRY.register(target=[torch.ops.higher_order.auto_functionalized_v2])
def _auto_functionalized_v2_handler(P: MLXProgramBuilder, n: Node):
    """
    Handler for auto_functionalized_v2 higher-order op.

    auto_functionalized_v2 wraps mutating ops after functionalization.
    It returns a tuple of (token, mutated_values...).

    This handler emits the actual lowering instructions and returns a tuple
    of slots that getitem can index into.
    """
    if len(n.args) < 1:
        raise ValueError(
            f"auto_functionalized_v2 requires at least 1 arg, got {len(n.args)}"
        )

    wrapped_op = n.args[0]

    # Unknown wrapped op - not supported
    raise NotImplementedError(
        f"auto_functionalized_v2 wrapping '{wrapped_op}' is not supported."
    )


@REGISTRY.register(target=[torch.ops.aten.linear.default])
def _linear_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 3, "aten.linear")
    require_kwargs(P.kwargs(n), set(), "aten.linear")
    x, w = args[0], args[1]
    b = args[2] if len(args) > 2 else None
    out = P.make_or_get_slot(n)

    # Transpose weight: linear(x, w) = x @ w.T
    _, w_t = P.make_tmp_slot()
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(w),
            out=P.slot_to_tid(w_t),
            perm=[1, 0],
        )
    )

    P.emit(
        AddmmNode(
            mat1=P.slot_to_tid(x),
            mat2=P.slot_to_tid(w_t),
            out=P.slot_to_tid(out),
            bias=P.slot_to_tid(b) if b else None,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.addmm.default])
def _addmm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle addmm: self + (mat1 @ mat2).

    addmm(self, mat1, mat2, *, beta=1, alpha=1) computes:
        beta * self + alpha * (mat1 @ mat2)

    This is typically the result of decomposing linear(x, w, b) in Edge IR:
        permute(w) -> addmm(b, x, permuted_w)

    For the common case where beta=1 and alpha=1, this is equivalent to:
        mat1 @ mat2 + self

    We use AddmmNode which calls matmul directly (no transposition needed).
    """
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 3, 3, "aten.addmm")
    require_kwargs(kwargs, {"beta", "alpha"}, "aten.addmm")
    bias, mat1, mat2 = args[0], args[1], args[2]

    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)

    out = P.make_or_get_slot(n)

    # Emit AddmmNode with alpha and beta parameters
    P.emit(
        AddmmNode(
            mat1=P.slot_to_tid(mat1),
            mat2=P.slot_to_tid(mat2),
            out=P.slot_to_tid(out),
            bias=P.slot_to_tid(bias),
            alpha=float(alpha),
            beta=float(beta),
        )
    )
    return out


@REGISTRY.register(
    target=[
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.matmul.default,
    ]
)
def _mm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle mm/bmm/matmul: matrix multiplication without bias.

    All three ops compute matrix products with different dimension expectations:
    - mm: 2D x 2D
    - bmm: 3D x 3D (batched)
    - matmul: arbitrary dimensions (NumPy semantics)

    MLX's matmul handles all cases, so we emit AddmmNode with bias=None.
    """
    args = P.args(n)
    require_args(args, 2, 2, "aten.mm/bmm/matmul")
    require_kwargs(P.kwargs(n), set(), "aten.mm/bmm/matmul")
    mat1, mat2 = args[0], args[1]

    out = P.make_or_get_slot(n)

    P.emit(
        AddmmNode(
            mat1=P.slot_to_tid(mat1),
            mat2=P.slot_to_tid(mat2),
            out=P.slot_to_tid(out),
            bias=None,
        )
    )
    return out


@REGISTRY.register(
    target=[
        torch.ops.aten.view.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.reshape.default,
    ]
)
def _view_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.view")
    require_kwargs(P.kwargs(n), set(), "aten.view")
    x, shape = args
    out = P.make_or_get_slot(n)

    shape_iovs = [P.to_int_or_vid(s) for s in shape]
    P.emit(
        ReshapeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            shape=shape_iovs,
        )
    )
    return out


@REGISTRY.register(
    target=[
        torch.ops.aten.clone.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.alias_copy.default,
    ]
)
def _clone_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten.clone")
    require_kwargs(kwargs, {"memory_format"}, "aten.clone")
    require_contiguous_format(
        memory_format=kwargs.get("memory_format"),
        op_name="aten.clone",
    )
    (x,) = args
    out = P.make_or_get_slot(n)
    P.emit(
        ContiguousNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.copy.default])
def _copy_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.copy - copy data from src to self.

    Schema: aten::copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
    In functionalized Edge IR, this returns a copy of src (args[1]).
    """
    args = P.args(n)
    require_args(args, 2, 2, "aten.copy")
    require_kwargs(P.kwargs(n), {"non_blocking"}, "aten.copy")
    src = args[1]
    out = P.make_or_get_slot(n)
    P.emit(
        ContiguousNode(
            x=P.slot_to_tid(src),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[exir_ops.edge.dim_order_ops._clone_dim_order.default])
def _dim_order_clone_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    # dim_order_ops._clone_dim_order(Tensor self, *, bool non_blocking=False, int[]? dim_order=None) -> Tensor
    # This is essentially a contiguous/clone operation for memory layout
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "dim_order_ops._clone_dim_order")
    require_kwargs(
        kwargs, {"non_blocking", "dim_order"}, "dim_order_ops._clone_dim_order"
    )
    require_contiguous_format(
        dim_order=kwargs.get("dim_order"),
        op_name="dim_order_ops._clone_dim_order",
    )
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(
        ContiguousNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
        )
    )
    return out


# Handle Edge IR's dim_order_ops._to_dim_order_copy (dtype conversion)
# This is what x.to(dtype) becomes after to_edge() transformation
@REGISTRY.register(target=[exir_ops.edge.dim_order_ops._to_dim_order_copy.default])
def _dim_order_copy_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    # dim_order_ops._to_dim_order_copy(Tensor self, *, ScalarType? dtype=None, ...)
    # If dtype is specified, this is a dtype conversion (use AsTypeNode)
    # If dtype is None/same, this is just a memory layout copy (use ContiguousNode)
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "dim_order_ops._to_dim_order_copy")
    require_kwargs(
        kwargs,
        {"dtype", "device", "layout", "non_blocking", "dim_order"},
        "dim_order_ops._to_dim_order_copy",
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        dim_order=kwargs.get("dim_order"),
        op_name="dim_order_ops._to_dim_order_copy",
    )
    x = args[0]
    out = P.make_or_get_slot(n)

    dtype = kwargs.get("dtype")
    if dtype is not None:
        # Dtype conversion
        P.emit(
            AsTypeNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
                scalar_type=torch_dtype_to_scalar_type(dtype),
            )
        )
    else:
        # No dtype change, just memory layout (contiguous)
        P.emit(
            ContiguousNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
            )
        )
    return out


@REGISTRY.register(target=[torch.ops.aten._to_copy.default])
def _to_copy_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten._to_copy - lower-level dtype/device conversion."""
    # aten._to_copy(Tensor self, *, ScalarType? dtype=None, ...)
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten._to_copy")
    require_kwargs(
        kwargs, {"dtype", "device", "layout", "memory_format"}, "aten._to_copy"
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        memory_format=kwargs.get("memory_format"),
        op_name="aten._to_copy",
    )
    x = args[0]
    out = P.make_or_get_slot(n)

    dtype = kwargs.get("dtype")
    if dtype is not None:
        # Dtype conversion
        P.emit(
            AsTypeNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
                scalar_type=torch_dtype_to_scalar_type(dtype),
            )
        )
    else:
        # No dtype change, just copy (use contiguous)
        P.emit(
            ContiguousNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(out),
            )
        )
    return out


@REGISTRY.register(target=[torch.ops.aten.embedding.default])
def _embedding_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 3, "aten.embedding")
    # "padding_idx", "scale_grad_by_freq", "sparse" are training only args
    # and ignored
    require_kwargs(
        P.kwargs(n), {"padding_idx", "scale_grad_by_freq", "sparse"}, "aten.embedding"
    )
    w, x = args[0], args[1]
    # padding_idx (args[2] if present) is ignored - only affects gradients
    out = P.make_or_get_slot(n)
    P.emit(
        TakeNode(
            x=P.slot_to_tid(w),
            index=IntOrVidOrTid.from_tid(P.slot_to_tid(x)),
            out=P.slot_to_tid(out),
            axis=0,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar])
def _add_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.add.Tensor: a + alpha * b."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.add.Tensor")
    require_kwargs(P.kwargs(n), {"alpha"}, "aten.add.Tensor")
    a, b = args
    input_meta = n.args[0].meta.get("val")
    dtype = input_meta.dtype if input_meta is not None else torch.float32
    if not isinstance(b, Slot):
        b = emit_lifted_constant(P, b, dtype)
    alpha = P.kwargs(n).get("alpha", 1)
    if alpha != 1:
        alpha_slot = emit_lifted_constant(P, alpha, dtype)
        _, tmp = P.make_tmp_slot()
        P.emit(
            MultiplyNode(
                a=P.slot_to_tid(b),
                b=P.slot_to_tid(alpha_slot),
                out=P.slot_to_tid(tmp),
            )
        )
        b = tmp
    out = P.make_or_get_slot(n)
    P.emit(
        AddNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.div.Tensor_mode])
def _div_tensor_mode_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.div.Tensor_mode with rounding mode."""
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 2, 2, "aten.div.Tensor_mode")
    require_kwargs(kwargs, {"rounding_mode"}, "aten.div.Tensor_mode")
    out = P.make_or_get_slot(n)
    a = args[0]
    b = args[1]
    rounding_mode = kwargs.get("rounding_mode", None)

    # Handle scalar b by creating a constant tensor
    if not isinstance(b, Slot):
        b = P.make_or_get_constant(
            f"_scalar_{b}", torch.tensor([b], dtype=n.meta["val"].dtype)
        )

    # Handle scalar a
    if not isinstance(a, Slot):
        a = P.make_or_get_constant(
            f"_scalar_{a}", torch.tensor([a], dtype=n.meta["val"].dtype)
        )

    if rounding_mode == "trunc":
        raise NotImplementedError(
            "aten.div.Tensor_mode with rounding_mode='trunc' is not supported. "
            "MLX does not have a truncate operation."
        )
    elif rounding_mode == "floor":
        P.emit(
            FloorDivideNode(
                a=P.slot_to_tid(a),
                b=P.slot_to_tid(b),
                out=P.slot_to_tid(out),
            )
        )
    else:
        # rounding_mode is None - true division
        P.emit(
            DivideNode(
                a=P.slot_to_tid(a),
                b=P.slot_to_tid(b),
                out=P.slot_to_tid(out),
            )
        )
    return out


@REGISTRY.register(target=[torch.ops.aten._softmax.default])
def _softmax_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle softmax: computes softmax along the specified dimension.

    aten._softmax(self, dim, half_to_float) computes:
        softmax(self, axis=dim)

    The half_to_float parameter is for type conversion and is ignored for MLX.
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten._softmax")
    require_kwargs(P.kwargs(n), set(), "aten._softmax")
    x, dim, _ = args[0], args[1], args[2]  # half_to_float is unused for MLX

    out = P.make_or_get_slot(n)

    # Emit SoftmaxNode with the specified axis
    P.emit(
        SoftmaxNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=dim,
            precise=False,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.gelu.default])
def _gelu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten.gelu")
    require_kwargs(kwargs, {"approximate"}, "aten.gelu")
    (x,) = args
    # GELU approximate mode: 'none' (default) or 'tanh'
    approximate = kwargs.get("approximate", "none")
    out = P.make_or_get_slot(n)
    P.emit(
        GeluNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            approximate=approximate,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.permute.default, torch.ops.aten.permute_copy.default]
)
def _permute_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.permute")
    require_kwargs(P.kwargs(n), set(), "aten.permute")
    x, dims = args
    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            perm=list(dims),
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.transpose.int, torch.ops.aten.transpose_copy.int]
)
def _transpose_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 3, 3, "aten.transpose")
    require_kwargs(P.kwargs(n), set(), "aten.transpose")
    x, dim0, dim1 = args
    perm = list(range(len(n.meta["val"].shape)))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            perm=perm,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor]
)
def _slice_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 4, 5, "aten.slice")
    require_kwargs(P.kwargs(n), set(), "aten.slice")
    x, dim, start, stop = args[0], args[1], args[2], args[3]
    step = args[4] if len(args) > 4 else 1
    if start is None:
        start = 0
    require_static_int(step, "step", "aten.slice")
    assert step >= 1, f"aten.slice: step must be >= 1, got {step}"
    out = P.make_or_get_slot(n)
    P.emit(
        SliceNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=P.to_int_or_vid(dim),
            start=P.to_int_or_vid(start),
            stop=P.to_int_or_vid(stop),
            step=step,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.narrow.default])
def _narrow_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """
    Handle narrow(input, dim, start, length) -> slice(input, dim, start, start+length).

    This is needed for KV cache updates with dynamic positions where narrow
    is preferred over slice syntax for better torch.export compatibility.
    """
    args = P.args(n)
    require_args(args, 4, 4, "aten.narrow")
    require_kwargs(P.kwargs(n), set(), "aten.narrow")
    x, dim, start, length = args
    out = P.make_or_get_slot(n)

    # Convert narrow (start, length) to slice (start, end)
    # The end is start + length
    start_iov = P.to_int_or_vid(start)
    length_iov = P.to_int_or_vid(length)

    # For stop = start + length, we need to emit an ADD_SCALAR if either is a Vid
    if isinstance(start_iov, IntOrVid) and start_iov.vid is not None:
        # start is a Vid, need to add at runtime
        if isinstance(length_iov, IntOrVid) and length_iov.vid is not None:
            # Both are Vids - emit add to compute stop
            _, stop_slot = P.make_tmp_value_slot()
            stop_vid = P.slot_to_vid(stop_slot)
            P.emit(
                AddIntNode(
                    a=start_iov.vid,
                    b=length_iov.vid,
                    out=stop_vid,
                )
            )
            stop_iov = IntOrVid(int64=None, vid=stop_vid)
        else:
            # start is Vid, length is int - emit add scalar
            _, stop_slot = P.make_tmp_value_slot()
            stop_vid = P.slot_to_vid(stop_slot)
            P.emit(
                AddIntNode(
                    a=start_iov.vid,
                    b=(
                        length_iov.int64
                        if isinstance(length_iov, IntOrVid)
                        else length_iov
                    ),
                    out=stop_vid,
                )
            )
            stop_iov = IntOrVid(int64=None, vid=stop_vid)
    elif isinstance(length_iov, IntOrVid) and length_iov.vid is not None:
        # length is Vid, start is int - emit add scalar
        start_val = start_iov.int64 if isinstance(start_iov, IntOrVid) else start_iov
        _, stop_slot = P.make_tmp_value_slot()
        stop_vid = P.slot_to_vid(stop_slot)
        P.emit(
            AddIntNode(
                a=length_iov.vid,
                b=start_val,
                out=stop_vid,
            )
        )
        stop_iov = IntOrVid(int64=None, vid=stop_vid)
    else:
        # Both are concrete ints
        start_val = start_iov.int64 if isinstance(start_iov, IntOrVid) else start_iov
        length_val = (
            length_iov.int64 if isinstance(length_iov, IntOrVid) else length_iov
        )
        stop_iov = IntOrVid(int64=start_val + length_val, vid=None)

    P.emit(
        SliceNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=P.to_int_or_vid(dim),
            start=start_iov,
            stop=stop_iov,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.unsqueeze.default, torch.ops.aten.unsqueeze_copy.default]
)
def _unsqueeze_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.unsqueeze")
    require_kwargs(P.kwargs(n), set(), "aten.unsqueeze")
    x, dim = args
    out = P.make_or_get_slot(n)
    P.emit(
        ExpandDimsNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.squeeze.dims, torch.ops.aten.squeeze_copy.dims]
)
def _squeeze_dims_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle squeeze operation for specific dimensions.

    Removes dimensions of size 1 from the tensor at specified positions.
    """
    args = P.args(n)
    require_args(args, 2, 2, "aten.squeeze.dims")
    require_kwargs(P.kwargs(n), set(), "aten.squeeze.dims")
    x, dims = args
    out = P.make_or_get_slot(n)

    dims_list = list(dims) if dims is not None else None

    P.emit(
        SqueezeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            dims=dims_list,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.squeeze.default, torch.ops.aten.squeeze_copy.default]
)
def _squeeze_default_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle squeeze operation without specified dimensions.

    Removes all dimensions of size 1 from the tensor.
    """
    args = P.args(n)
    require_args(args, 1, 1, "aten.squeeze.default")
    require_kwargs(P.kwargs(n), set(), "aten.squeeze.default")
    (x,) = args
    out = P.make_or_get_slot(n)

    P.emit(
        SqueezeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            dims=None,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.cat.default])
def _cat_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle concatenation of a list of tensors.

    Concatenates tensors along a specified dimension.
    All tensors must have the same shape except in the concatenating dimension.
    """
    args = P.args(n)
    require_args(args, 1, 2, "aten.cat")
    require_kwargs(P.kwargs(n), set(), "aten.cat")
    # aten.cat.default signature: cat(Tensor[] tensors, int dim=0) -> Tensor
    # args can be (tensors_list,) or (tensors_list, dim)
    tensors_list = args[0]
    dim = args[1] if len(args) > 1 else 0

    out = P.make_or_get_slot(n)

    # Convert list of tensor slots to list of Tids
    tensor_tids = [P.slot_to_tid(t) for t in tensors_list]

    # dim is typically an int
    axis = dim if dim is not None else 0

    P.emit(
        ConcatenateNode(
            tensors=tensor_tids,
            out=P.slot_to_tid(out),
            axis=axis,
        )
    )
    return out


@REGISTRY.register(
    target=[
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split_with_sizes_copy.default,
    ]
)
def _split_with_sizes_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle split_with_sizes operation.

    Splits a tensor into chunks with specified sizes along a dimension.
    Returns a tuple of output slots that getitem can extract from.

    PyTorch: split_with_sizes(x, [2, 3, 4], dim=1)
    MLX: split(x, indices=[2, 5], axis=1)  # indices are cumulative positions
    """
    args = P.args(n)
    require_args(args, 2, 3, "aten.split_with_sizes")
    require_kwargs(P.kwargs(n), set(), "aten.split_with_sizes")
    x = args[0]
    sizes = args[1]
    dim = args[2] if len(args) > 2 else 0  # dim has default value of 0

    # Convert sizes to IntOrVid (supports both static ints and dynamic Vids)
    sizes_int_or_vid = [P.to_int_or_vid(s) for s in sizes]

    axis = dim if dim is not None else 0

    # Create output slots for multi-output operation
    # make_or_get_slots automatically creates slots based on node.meta["val"]
    output_slots = P.make_or_get_slots(n)

    # Emit SplitNode with all output slots
    P.emit(
        SplitNode(
            x=P.slot_to_tid(x),
            outs=[P.slot_to_tid(s) for s in output_slots],
            sizes=sizes_int_or_vid,
            axis=axis,
        )
    )

    # Return tuple of slots - getitem will extract individual elements
    return output_slots


@REGISTRY.register(target=[torch.ops.mlx.gather_mm.default])
def _gather_mm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle mlx::gather_mm — fused gather + matmul for MoE experts."""
    from executorch.backends.mlx.serialization.mlx_graph_schema import GatherMmNode

    args = P.args(n)
    kwargs = P.kwargs(n)

    a = args[0]
    b = args[1]
    rhs_indices = args[2] if len(args) > 2 else kwargs.get("rhs_indices")
    lhs_indices = args[3] if len(args) > 3 else kwargs.get("lhs_indices")
    sorted_indices = args[4] if len(args) > 4 else kwargs.get("sorted_indices", False)

    out = P.make_or_get_slot(n)
    P.emit(
        GatherMmNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
            lhs_indices=P.slot_to_tid(lhs_indices) if lhs_indices is not None else None,
            rhs_indices=P.slot_to_tid(rhs_indices) if rhs_indices is not None else None,
            sorted_indices=sorted_indices,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.mlx.gather_qmm.default])
def _gather_qmm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle mlx::gather_qmm — fused gather + dequant + matmul for quantized MoE experts.

    Converts TorchAO quantization format to MLX format (unsigned + biases)
    and emits a GatherQmmNode.
    """
    from executorch.backends.mlx.serialization.mlx_graph_schema import GatherQmmNode

    args = P.args(n)
    kwargs = P.kwargs(n)

    x = args[0]
    w_node = n.args[1]  # Need the original node for constant lookup
    scales_node = n.args[2]
    biases_node = n.args[3] if len(n.args) > 3 else n.kwargs.get("biases")
    rhs_indices = args[4] if len(args) > 4 else kwargs.get("rhs_indices")
    lhs_indices = args[5] if len(args) > 5 else kwargs.get("lhs_indices")
    transpose = args[6] if len(args) > 6 else kwargs.get("transpose", True)
    group_size = args[7] if len(args) > 7 else kwargs.get("group_size", 32)
    bits = args[8] if len(args) > 8 else kwargs.get("bits", 4)
    mode = args[9] if len(args) > 9 else kwargs.get("mode", "affine")
    sorted_indices = args[10] if len(args) > 10 else kwargs.get("sorted_indices", False)

    # Convert quantized weights to MLX format
    w_target, w_data = P.get_placeholder_target_and_tensor(w_node)
    _, scale_data = P.get_placeholder_target_and_tensor(scales_node)
    zp_target = None
    zp_data = None
    if biases_node is not None:
        zp_target, zp_data = P.get_placeholder_target_and_tensor(biases_node)

    # Reshape 3D [E, out, in] to 2D for to_mlx_qparams, then reshape back
    orig_shape = w_data.shape
    E, out_dim = orig_shape[0], orig_shape[1]
    w_2d = w_data.reshape(E * out_dim, -1)
    s_2d = scale_data.reshape(E * out_dim, -1)
    zp_2d = (
        zp_data.reshape(E * out_dim, -1)
        if zp_data is not None
        else torch.zeros_like(s_2d, dtype=torch.int8)
    )

    Q, B = to_mlx_qparams(w_2d, s_2d, zp_2d, bits)
    Q = Q.reshape(E, out_dim, -1)
    B = B.reshape(E, out_dim, -1)

    packed_slot = P.make_or_get_constant(f"{w_target}_to_packed", Q)
    scale_slot = P.slot_map([scales_node])[0]
    biases_slot = P.make_or_get_constant(f"{zp_target or w_target}_to_biases", B)

    out = P.make_or_get_slot(n)
    P.emit(
        GatherQmmNode(
            x=P.slot_to_tid(x),
            w=P.slot_to_tid(packed_slot),
            scales=P.slot_to_tid(scale_slot),
            out=P.slot_to_tid(out),
            biases=P.slot_to_tid(biases_slot),
            lhs_indices=P.slot_to_tid(lhs_indices) if lhs_indices is not None else None,
            rhs_indices=P.slot_to_tid(rhs_indices) if rhs_indices is not None else None,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=sorted_indices,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.split.Tensor, torch.ops.aten.split_copy.Tensor]
)
def _split_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle split operation with uniform chunk size.

    Splits a tensor into chunks of a given size along a dimension.
    The last chunk may be smaller if the dimension does not divide evenly.

    PyTorch: split(x, split_size, dim=0)

    We pass [split_size] to the interpreter, which computes the actual
    chunk sizes based on the tensor dimension.
    """
    args = P.args(n)
    require_args(args, 2, 3, "aten.split")
    require_kwargs(P.kwargs(n), set(), "aten.split")
    x = args[0]
    split_size = args[1]
    dim = args[2] if len(args) > 2 else 0

    axis = dim if dim is not None else 0
    if axis < 0:
        x_meta = n.args[0].meta.get("val")
        if x_meta is None:
            raise RuntimeError("split: missing tensor metadata for negative axis")
        axis += len(x_meta.shape)

    # Create output slots for multi-output operation
    output_slots = P.make_or_get_slots(n)

    # Emit SplitNode - interpreter computes actual chunk sizes from split_size
    P.emit(
        SplitNode(
            x=P.slot_to_tid(x),
            outs=[P.slot_to_tid(s) for s in output_slots],
            sizes=[P.to_int_or_vid(split_size)],
            axis=axis,
        )
    )

    return output_slots


@REGISTRY.register(target=[torch.ops.aten.repeat.default])
def _repeat_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.repeat")
    require_kwargs(P.kwargs(n), set(), "aten.repeat")
    x, reps = args

    # Convert reps to IntOrVid (supports both static ints and dynamic Vids)
    reps_int_or_vid = [P.to_int_or_vid(r) for r in reps]

    out = P.make_or_get_slot(n)
    P.emit(
        TileNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            reps=reps_int_or_vid,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.index.Tensor])
def _index_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.index.Tensor")
    require_kwargs(P.kwargs(n), set(), "aten.index.Tensor")
    x, idx_list = args
    if not isinstance(idx_list, list) or len(idx_list) == 0:
        raise ValueError(
            f"aten.index.Tensor requires a list of index tensors, "
            f"got {type(idx_list)}"
        )

    x_meta = n.args[0].meta.get("val")
    x_ndim = len(x_meta.shape) if x_meta is not None else None

    # Filter out None indices and track which axes they correspond to
    non_none = [(i, idx) for i, idx in enumerate(idx_list) if idx is not None]

    if len(non_none) == 0:
        raise ValueError("aten.index.Tensor: all indices are None")

    if len(non_none) == 1:
        axis, idx = non_none[0]
        idx_meta = n.args[1][axis].meta.get("val")
        ndim_match = (
            x_meta is not None
            and idx_meta is not None
            and len(x_meta.shape) == len(idx_meta.shape)
        )
        out = P.make_or_get_slot(n)
        if ndim_match:
            # Same ndim: use TakeAlongAxisNode (element-wise gather)
            P.emit(
                TakeAlongAxisNode(
                    x=P.slot_to_tid(x),
                    indices=P.slot_to_tid(idx),
                    out=P.slot_to_tid(out),
                    axis=axis,
                )
            )
        else:
            # Different ndim (e.g. 1D indices into 3D tensor): use TakeNode
            P.emit(
                TakeNode(
                    x=P.slot_to_tid(x),
                    index=IntOrVidOrTid.from_tid(P.slot_to_tid(idx)),
                    out=P.slot_to_tid(out),
                    axis=axis,
                )
            )
        return out

    # Multi-index: use GatherNode (maps to mlx::gather)
    if x_meta is None or x_ndim is None:
        raise ValueError(
            "aten.index.Tensor with multiple indices requires input shape metadata"
        )

    indices = [P.slot_to_tid(idx) for _, idx in non_none]
    axes = [i for i, _ in non_none]

    # slice_sizes: 1 for indexed axes, full dim size for non-indexed axes
    # Use int() to handle SymInt values from dynamic shapes
    indexed_axes = set(axes)
    slice_sizes = []
    for dim in range(x_ndim):
        if dim in indexed_axes:
            slice_sizes.append(1)
        else:
            dim_size = x_meta.shape[dim]
            if not isinstance(dim_size, int):
                raise ValueError(
                    f"aten.index.Tensor: non-indexed dimension {dim} has dynamic size "
                    f"{dim_size}, which is not supported with multi-index gather"
                )
            slice_sizes.append(dim_size)

    # Emit gather — output shape is broadcast(indices).shape + slice_sizes
    _, gather_slot = P.make_tmp_slot()
    P.emit(
        GatherNode(
            x=P.slot_to_tid(x),
            indices=indices,
            out=P.slot_to_tid(gather_slot),
            axes=axes,
            slice_sizes=slice_sizes,
        )
    )

    # Reshape to match aten.index.Tensor output shape, which strips the
    # trailing dimensions introduced by gather's slice_sizes
    out_meta = n.meta.get("val")
    if out_meta is None:
        raise ValueError(
            "aten.index.Tensor: output shape metadata required for reshape after gather"
        )
    out_shape = [P.to_int_or_vid(int(d)) for d in out_meta.shape]

    out = P.make_or_get_slot(n)
    P.emit(
        ReshapeNode(
            x=P.slot_to_tid(gather_slot),
            out=P.slot_to_tid(out),
            shape=out_shape,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.index_select.default])
def _index_select_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.index_select: select elements along an axis using a 1D index tensor.

    index_select(input, dim, index) returns input.take(index, axis=dim).
    Unlike select (which takes a scalar index and removes the dim),
    index_select takes a tensor of indices and preserves the dim.
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten.index_select")
    require_kwargs(P.kwargs(n), set(), "aten.index_select")
    x, dim, indices = args
    out = P.make_or_get_slot(n)
    P.emit(
        TakeNode(
            x=P.slot_to_tid(x),
            index=IntOrVidOrTid.from_tid(P.slot_to_tid(indices)),
            out=P.slot_to_tid(out),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.slice_scatter.default])
def _slice_scatter_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.slice_scatter: return a copy of self with self[dim][start:end:step] = src."""
    args = P.args(n)
    require_args(args, 2, 6, "aten.slice_scatter")
    require_kwargs(P.kwargs(n), set(), "aten.slice_scatter")
    self_tensor = args[0]
    src = args[1]
    dim = args[2] if len(args) > 2 else 0
    start = args[3] if len(args) > 3 else 0
    end = args[4] if len(args) > 4 else None
    step = args[5] if len(args) > 5 else 1

    # If end is None, default to dim size
    if end is None:
        input_meta = n.args[0].meta.get("val")
        if input_meta is not None:
            end = input_meta.shape[dim]
        else:
            raise ValueError(
                "aten.slice_scatter: end=None requires input shape metadata"
            )

    require_static_int(step, "step", "aten.slice_scatter")
    assert step >= 1, f"aten.slice_scatter: step must be >= 1, got {step}"

    out = P.make_or_get_slot(n)
    P.emit(
        SliceUpdateNode(
            dst=P.slot_to_tid(self_tensor),
            update=P.slot_to_tid(src),
            out=P.slot_to_tid(out),
            axis=P.to_int_or_vid(dim),
            start=P.to_int_or_vid(start),
            stop=P.to_int_or_vid(end),
            step=step,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.scatter_add.default])
def _scatter_add_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.scatter_add: accumulate src into self at index positions along dim.

    scatter_add(self, dim, index, src) -> Tensor

    Maps to mlx::scatter_add(a, indices, updates, axis).
    """
    args = P.args(n)
    require_args(args, 4, 4, "aten.scatter_add")
    require_kwargs(P.kwargs(n), set(), "aten.scatter_add")
    x, dim, indices, src = args
    out = P.make_or_get_slot(n)
    P.emit(
        ScatterAddNode(
            x=P.slot_to_tid(x),
            indices=P.slot_to_tid(indices),
            updates=P.slot_to_tid(src),
            out=P.slot_to_tid(out),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.select.int, torch.ops.aten.select_copy.int])
def _select_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """
    Handle aten.select_copy.int - select a single index along a dimension.

    select_copy(input, dim, index) returns input[..., index, ...] where the
    indexing happens at dimension `dim`. The selected dimension is removed.

    Maps to MLX's take(array, int index, axis) which also removes the dimension.
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten.select_copy.int")
    require_kwargs(P.kwargs(n), set(), "aten.select_copy.int")
    x, dim, index = args
    out = P.make_or_get_slot(n)
    P.emit(
        TakeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            index=P.to_int_or_vid_or_tid(index),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.sym_size.int])
def _sym_size_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 2, "aten.sym_size.int")
    require_kwargs(P.kwargs(n), set(), "aten.sym_size.int")
    a, dim = args
    out = P.make_or_get_slot(n)
    P.emit(
        SymSizeNode(
            a=P.slot_to_tid(a),
            dim=dim,
            out=P.slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.item.default])
def _item_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    if not isinstance(n.meta["val"], torch.SymInt):
        raise ValueError("item only supported if it returns a SymInt")
    args = P.args(n)
    require_args(args, 1, 1, "aten.item")
    require_kwargs(P.kwargs(n), set(), "aten.item")
    (x,) = args
    out = P.make_or_get_slot(n)
    P.emit(
        ItemIntNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_vid(out),
        )
    )
    return out


@REGISTRY.register(target=[operator.getitem])
def _getitem_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """
    Handle getitem(tuple, idx) - extracts element from a tuple of slots.

    The source tuple comes from ops that return multiple values (like
    auto_functionalized_v2). Those handlers return tuples of slots,
    and we just ID_COPY the selected element to a new output slot.
    """
    args = P.args(n)
    require_args(args, 2, 2, "operator.getitem")
    require_kwargs(P.kwargs(n), set(), "operator.getitem")
    a, idx = args
    out = P.make_or_get_slot(n)
    P.emit(
        IdCopyNode(
            x=P.slot_to_tid(a[idx]),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.layer_norm.default])
def _layer_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 5, "aten.layer_norm")
    require_kwargs(P.kwargs(n), set(), "aten.layer_norm")
    x, shape = args[0:2]
    if len(shape) > 1:
        raise ValueError(
            "LayerNorm is only supported when normalizing over the last dimension"
        )
    w = args[2] if len(args) > 2 else None
    bias = args[3] if len(args) > 3 else None
    eps = args[4] if len(args) > 4 else 1e-5

    out = P.make_or_get_slot(n)
    P.emit(
        LayerNormNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            weight=P.slot_to_tid(w) if w else None,
            bias=P.slot_to_tid(bias) if bias else None,
            eps=eps,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.native_layer_norm.default])
def _native_layer_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle native_layer_norm which returns (output, mean, rstd).

    Only the normalized output (index 0) is computed via fast::layer_norm;
    mean and rstd (indices 1 and 2) are needed only for backward.
    """
    # Verify mean/rstd outputs are unused — we only compute the normalized output.
    unsupported = used_getitem_indices(n) & {1, 2}
    if unsupported:
        raise ValueError(
            f"native_layer_norm outputs {unsupported} (mean/rstd) are used, "
            "but only the normalized output (index 0) is supported"
        )

    args = P.args(n)
    require_args(args, 2, 5, "aten.native_layer_norm")
    require_kwargs(P.kwargs(n), set(), "aten.native_layer_norm")
    x, shape = args[0:2]
    if len(shape) > 1:
        raise ValueError(
            "LayerNorm is only supported when normalizing over the last dimension"
        )
    w = args[2] if len(args) > 2 else None
    bias = args[3] if len(args) > 3 else None
    eps = args[4] if len(args) > 4 else 1e-5

    # native_layer_norm returns (output, mean, rstd) — allocate all 3 slots
    output_slots = P.make_or_get_slots(n)

    P.emit(
        LayerNormNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(output_slots[0]),
            weight=P.slot_to_tid(w) if w else None,
            bias=P.slot_to_tid(bias) if bias else None,
            eps=eps,
        )
    )
    return output_slots


@REGISTRY.register(target=[torch.ops.aten.arange.default])
def _arange_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle arange with just stop, or (start, stop) or (start, stop, step).

    Supports both static (literal int) and dynamic (Slot from item()) values.
    """
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 3, "aten.arange")
    require_kwargs(kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.arange")
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.arange",
    )
    if len(args) == 1:
        start = 0
        stop = args[0]
    else:
        start, stop = args[0:2]
    step = args[2] if len(args) > 2 else 1

    # arange defaults to int64 when dtype is not specified (like torch.arange)
    dtype = kwargs.get("dtype", torch.int64)
    scalar_type_val = torch_dtype_to_scalar_type(dtype)

    out = P.make_or_get_slot(n)
    P.emit(
        ARangeNode(
            out=P.slot_to_tid(out),
            start=P.to_int_or_vid(start),
            stop=P.to_int_or_vid(stop),
            step=P.to_int_or_vid(step),
            scalar_type=scalar_type_val,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.arange.start_step])
def _arange_start_step_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle arange with start, end, and step arguments.

    Supports both static (literal int) and dynamic (Slot from item()) start/stop/step.
    """
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 2, 3, "aten.arange.start_step")
    require_kwargs(
        kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.arange.start_step"
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.arange.start_step",
    )
    start = args[0]
    stop = args[1]
    step = args[2] if len(args) > 2 else 1

    # arange defaults to int64 when dtype is not specified (like torch.arange)
    dtype = kwargs.get("dtype", torch.int64)
    scalar_type_val = torch_dtype_to_scalar_type(dtype)

    out = P.make_or_get_slot(n)
    P.emit(
        ARangeNode(
            out=P.slot_to_tid(out),
            start=P.to_int_or_vid(start),
            stop=P.to_int_or_vid(stop),
            step=P.to_int_or_vid(step),
            scalar_type=scalar_type_val,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.rms_norm.default])
def _aten_rms_norm_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 2, 4, "aten.rms_norm")
    require_kwargs(P.kwargs(n), set(), "aten.rms_norm")
    x, normalized_shape = args[0], args[1]
    if len(normalized_shape) > 1:
        raise ValueError(
            "RMSNorm is only supported when normalizing over the last dimension"
        )
    w = args[2] if len(args) > 2 else None
    eps = args[3] if len(args) > 3 else 1e-5
    out = P.make_or_get_slot(n)
    P.emit(
        RMSNormNode(
            x=P.slot_to_tid(x),
            weight=P.slot_to_tid(w) if w else None,
            out=P.slot_to_tid(out),
            eps=eps,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.mlx.rope.default])
def _rope_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    args = P.args(n)
    require_args(args, 3, 7, "mlx.rope")
    require_kwargs(P.kwargs(n), set(), "mlx.rope")
    x, dims, pos = args[0], args[1], args[2]
    traditional = args[3] if len(args) > 3 else False
    base = args[4] if len(args) > 4 else 500000.0
    scale = args[5] if len(args) > 5 else 1.0
    freqs = args[6] if len(args) > 6 else None
    out = P.make_or_get_slot(n)

    # pos must be a Slot (SymInt) from input_pos.item() during tracing
    # The schema supports both Vid (scalar) and Tid (tensor) for offset
    if not isinstance(pos, Slot):
        raise ValueError(
            f"RopeNode.offset must be a SymInt (traced via tensor.item()), got {type(pos)}. "
            "Make sure input_pos is a tensor and you call input_pos.item() to get a SymInt."
        )

    P.emit(
        RopeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            dims=dims,
            offset=VidOrTid.from_vid(P.slot_to_vid(pos)),
            freqs=P.slot_to_tid(freqs) if freqs else None,
            traditional=traditional,
            base=base,
            scale=scale,
        )
    )

    return out


def _emit_channel_last_weight(P: MLXProgramBuilder, w_node: Node, perm: list) -> Slot:
    """Get convolution weight in channel-last format.

    If the weight is a placeholder (static parameter), permute at compile time
    and store as a constant.  If it comes from another node (e.g. dequantize
    output), emit a runtime TransposeNode instead.
    """
    if w_node.op == "placeholder":
        w_target, w_tensor = P.get_placeholder_target_and_tensor(w_node)
        return P.make_or_get_constant(
            f"{w_target}_channel_last", w_tensor.permute(perm).contiguous()
        )
    else:
        w_slot = P.slot_map([w_node])[0]
        _, w = P.make_tmp_slot()
        P.emit(
            TransposeNode(
                x=P.slot_to_tid(w_slot),
                out=P.slot_to_tid(w),
                perm=perm,
            )
        )
        return w


def _emit_conv_transpose_weight(
    P: MLXProgramBuilder, w_node: Node, groups: int, ndim: int
) -> Slot:
    """Get conv_transpose weight in MLX format, handling grouped convolutions.

    PyTorch conv_transpose weight shape: [C_in, C_out/G, *K]
    MLX expects:                         [C_out, *K, C_in/G]

    For groups=1, a simple permute suffices (C_in==C_in/G, C_out/G==C_out).
    For groups>1, we need reshape-permute-reshape to rearrange the group dim:
      [C_in, C_out/G, *K] -> [G, C_in/G, C_out/G, *K]
                           -> [G, C_out/G, *K, C_in/G]
                           -> [C_out, *K, C_in/G]
    """
    if groups == 1:
        # Simple permute: [C_in, C_out, *K] -> [C_out, *K, C_in]
        # e.g. 1D: [1, 2, 0], 2D: [1, 2, 3, 0], 3D: [1, 2, 3, 4, 0]
        perm = list(range(1, ndim + 2)) + [0]
        return _emit_channel_last_weight(P, w_node, perm)

    # Grouped: need reshape-permute-reshape at compile time
    if w_node.op != "placeholder":
        raise ValueError(
            f"conv_transpose with groups > 1 requires static weights, "
            f"got dynamic weight from {w_node.op}"
        )

    w_target, w_tensor = P.get_placeholder_target_and_tensor(w_node)
    c_in = w_tensor.shape[0]
    c_out_per_g = w_tensor.shape[1]
    kernel_shape = list(w_tensor.shape[2:])
    c_in_per_g = c_in // groups

    # [C_in, C_out/G, *K] -> [G, C_in/G, C_out/G, *K]
    w = w_tensor.reshape([groups, c_in_per_g, c_out_per_g] + kernel_shape)
    # [G, C_in/G, C_out/G, *K] -> [G, C_out/G, *K, C_in/G]
    # perm: [0, 2, 3, ..., ndim+1, 1]
    perm = [0, 2] + list(range(3, ndim + 3)) + [1]
    w = w.permute(perm).contiguous()
    # [G, C_out/G, *K, C_in/G] -> [C_out, *K, C_in/G]
    c_out = groups * c_out_per_g
    w = w.reshape([c_out] + kernel_shape + [c_in_per_g])

    return P.make_or_get_constant(f"{w_target}_channel_last", w)


def _emit_conv_bias(
    P: MLXProgramBuilder, bias: Optional[Slot], tmp: Slot, ndim: int
) -> None:
    """Reshape conv bias to channel-last broadcast shape and add to tmp in-place.

    After the convolution the activation is in channel-last layout, so the bias
    (shape ``[C_out]``) must be reshaped to ``[1, …, 1, -1]`` with *ndim*
    dimensions before being added.  Does nothing when *bias* is ``None``.
    """
    if bias is None:
        return
    _, tmp2 = P.make_tmp_slot()
    shape = [IntOrVid.from_literal(1)] * (ndim - 1) + [IntOrVid.from_literal(-1)]
    P.emit(
        ReshapeNode(
            x=P.slot_to_tid(bias),
            out=P.slot_to_tid(tmp2),
            shape=shape,
        )
    )
    P.emit(
        AddNode(
            a=P.slot_to_tid(tmp),
            b=P.slot_to_tid(tmp2),
            out=P.slot_to_tid(tmp),
        )
    )


def _emit_conv(
    P: MLXProgramBuilder,
    n: Node,
    x_node: Node,
    w_node: Node,
    bias_node,
    stride: list,
    padding: list,
    dilation: list,
    groups: int,
    ndim: int,
) -> Slot:
    """Shared logic for regular convolution emission.

    Handles weight transform, input/output transposition, bias, and node emission
    for all spatial dimensions (1D, 2D, 3D).

    Weight: [C_out, C_in/G, *K] -> [C_out, *K, C_in/G]
    Input: (N, C, *spatial) -> (N, *spatial, C)
    Output: (N, *spatial, C) -> (N, C, *spatial)
    """
    if ndim == 3 and groups != 1:
        raise ValueError(
            "conv3d with groups != 1 is not supported by MLX. " f"Got groups={groups}."
        )

    # Permutation: channels-first [N, C, *spatial] <-> channels-last [N, *spatial, C]
    ch_first_to_last = [0] + list(range(2, ndim + 2)) + [1]
    ch_last_to_first = [0, ndim + 1] + list(range(1, ndim + 1))

    # Weight: [C_out, C_in/G, *K] -> [C_out, *K, C_in/G] (same permutation)
    w = _emit_channel_last_weight(P, w_node, ch_first_to_last)

    x, bias = P.slot_map([x_node, bias_node])

    _, tmp = P.make_tmp_slot()
    P.emit(
        TransposeNode(x=P.slot_to_tid(x), out=P.slot_to_tid(tmp), perm=ch_first_to_last)
    )

    if ndim == 1:
        P.emit(
            Conv1DNode(
                x=P.slot_to_tid(tmp),
                w=P.slot_to_tid(w),
                out=P.slot_to_tid(tmp),
                stride=stride[0],
                padding=padding[0],
                dilation=dilation[0],
                groups=groups,
            )
        )
    elif ndim == 2:
        P.emit(
            Conv2DNode(
                x=P.slot_to_tid(tmp),
                w=P.slot_to_tid(w),
                out=P.slot_to_tid(tmp),
                stride_h=stride[0],
                stride_w=stride[1],
                padding_h=padding[0],
                padding_w=padding[1],
                dilation_h=dilation[0],
                dilation_w=dilation[1],
                groups=groups,
            )
        )
    elif ndim == 3:
        P.emit(
            Conv3DNode(
                x=P.slot_to_tid(tmp),
                w=P.slot_to_tid(w),
                out=P.slot_to_tid(tmp),
                stride_d=stride[0],
                stride_h=stride[1],
                stride_w=stride[2],
                padding_d=padding[0],
                padding_h=padding[1],
                padding_w=padding[2],
                dilation_d=dilation[0],
                dilation_h=dilation[1],
                dilation_w=dilation[2],
                groups=groups,
            )
        )

    _emit_conv_bias(P, bias, tmp, ndim=ndim + 2)

    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(tmp), out=P.slot_to_tid(out), perm=ch_last_to_first
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.conv1d.default])
def _conv1d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.conv1d: (input, weight, bias, stride, padding, dilation, groups)."""
    require_args(n.args, 2, 7, "aten.conv1d")
    require_kwargs(P.kwargs(n), set(), "aten.conv1d")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    groups = n.args[6] if len(n.args) > 6 else 1
    stride = _normalize_conv_param(n.args[3] if len(n.args) > 3 else 1, 1, 1)
    padding = _normalize_conv_param(n.args[4] if len(n.args) > 4 else 0, 1, 0)
    dilation = _normalize_conv_param(n.args[5] if len(n.args) > 5 else 1, 1, 1)
    return _emit_conv(
        P, n, x_node, w_node, bias_node, stride, padding, dilation, groups, ndim=1
    )


@REGISTRY.register(target=[torch.ops.aten.conv2d.default])
def _conv2d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.conv2d: (input, weight, bias, stride, padding, dilation, groups)."""
    require_args(n.args, 2, 7, "aten.conv2d")
    require_kwargs(P.kwargs(n), set(), "aten.conv2d")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    groups = n.args[6] if len(n.args) > 6 else 1
    stride = _normalize_conv_param(n.args[3] if len(n.args) > 3 else [1, 1], 2, 1)
    padding = _normalize_conv_param(n.args[4] if len(n.args) > 4 else [0, 0], 2, 0)
    dilation = _normalize_conv_param(n.args[5] if len(n.args) > 5 else [1, 1], 2, 1)
    return _emit_conv(
        P, n, x_node, w_node, bias_node, stride, padding, dilation, groups, ndim=2
    )


@REGISTRY.register(target=[torch.ops.aten.conv3d.default])
def _conv3d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.conv3d: (input, weight, bias, stride, padding, dilation, groups)."""
    require_args(n.args, 2, 7, "aten.conv3d")
    require_kwargs(P.kwargs(n), set(), "aten.conv3d")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    groups = n.args[6] if len(n.args) > 6 else 1
    stride = _normalize_conv_param(n.args[3] if len(n.args) > 3 else [1, 1, 1], 3, 1)
    padding = _normalize_conv_param(n.args[4] if len(n.args) > 4 else [0, 0, 0], 3, 0)
    dilation = _normalize_conv_param(n.args[5] if len(n.args) > 5 else [1, 1, 1], 3, 1)
    return _emit_conv(
        P, n, x_node, w_node, bias_node, stride, padding, dilation, groups, ndim=3
    )


def _emit_conv_transpose(
    P: MLXProgramBuilder,
    n: Node,
    x_node: Node,
    w_node: Node,
    bias_node,
    stride: list,
    padding: list,
    dilation: list,
    output_padding: list,
    groups: int,
    ndim: int,
) -> Slot:
    """Shared logic for transposed convolution emission.

    Handles weight transform, input/output transposition, bias, and node emission
    for all spatial dimensions. Called by both the specific conv_transpose handlers
    and the unified aten.convolution.default handler.
    """
    if ndim == 3 and groups != 1:
        raise ValueError(
            "conv_transpose with groups != 1 is not supported for 3D by MLX"
        )

    w = _emit_conv_transpose_weight(P, w_node, groups, ndim=ndim)
    x, bias = P.slot_map([x_node, bias_node])

    # Transpose input: channels-first -> channels-last
    ch_first_to_last = list(range(ndim + 2))
    ch_first_to_last = [0] + list(range(2, ndim + 2)) + [1]
    ch_last_to_first = [0, ndim + 1] + list(range(1, ndim + 1))

    _, tmp = P.make_tmp_slot()
    P.emit(
        TransposeNode(x=P.slot_to_tid(x), out=P.slot_to_tid(tmp), perm=ch_first_to_last)
    )

    if ndim == 1:
        P.emit(
            ConvTranspose1DNode(
                x=P.slot_to_tid(tmp),
                w=P.slot_to_tid(w),
                out=P.slot_to_tid(tmp),
                stride=stride[0],
                padding=padding[0],
                dilation=dilation[0],
                output_padding=output_padding[0],
                groups=groups,
            )
        )
    elif ndim == 2:
        P.emit(
            ConvTranspose2DNode(
                x=P.slot_to_tid(tmp),
                w=P.slot_to_tid(w),
                out=P.slot_to_tid(tmp),
                stride_h=stride[0],
                stride_w=stride[1],
                padding_h=padding[0],
                padding_w=padding[1],
                dilation_h=dilation[0],
                dilation_w=dilation[1],
                output_padding_h=output_padding[0],
                output_padding_w=output_padding[1],
                groups=groups,
            )
        )
    elif ndim == 3:
        P.emit(
            ConvTranspose3DNode(
                x=P.slot_to_tid(tmp),
                w=P.slot_to_tid(w),
                out=P.slot_to_tid(tmp),
                stride_d=stride[0],
                stride_h=stride[1],
                stride_w=stride[2],
                padding_d=padding[0],
                padding_h=padding[1],
                padding_w=padding[2],
                dilation_d=dilation[0],
                dilation_h=dilation[1],
                dilation_w=dilation[2],
                output_padding_d=output_padding[0],
                output_padding_h=output_padding[1],
                output_padding_w=output_padding[2],
                groups=groups,
            )
        )

    _emit_conv_bias(P, bias, tmp, ndim=ndim + 2)

    out = P.make_or_get_slot(n)
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(tmp), out=P.slot_to_tid(out), perm=ch_last_to_first
        )
    )
    return out


def _normalize_conv_param(val, ndim, default=0):
    """Normalize a conv parameter (stride/padding/etc.) to a list of length ndim."""
    if isinstance(val, int):
        return [val] * ndim
    if isinstance(val, list):
        if len(val) == 1:
            return val * ndim
        return val
    return [default] * ndim


@REGISTRY.register(target=[torch.ops.aten.convolution.default])
def _convolution_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.convolution.default — the unified convolution op.

    Args layout: convolution(input, weight, bias, stride, padding, dilation,
                             transposed, output_padding, groups)

    This op appears when PyTorch doesn't decompose to specific conv ops
    (e.g. grouped conv_transpose).
    """
    raw_args = n.args
    x_node, w_node = raw_args[0], raw_args[1]
    bias_node = raw_args[2] if len(raw_args) > 2 else None
    transposed = raw_args[6] if len(raw_args) > 6 else False
    groups = raw_args[8] if len(raw_args) > 8 else 1

    if not transposed:
        raise ValueError(
            "aten.convolution with transposed=False: use aten.conv{1,2,3}d instead"
        )

    x_meta = x_node.meta.get("val")
    if x_meta is None:
        raise ValueError("aten.convolution: input shape metadata required")
    ndim = len(x_meta.shape) - 2

    stride = _normalize_conv_param(raw_args[3] if len(raw_args) > 3 else 1, ndim, 1)
    padding = _normalize_conv_param(raw_args[4] if len(raw_args) > 4 else 0, ndim, 0)
    dilation = _normalize_conv_param(raw_args[5] if len(raw_args) > 5 else 1, ndim, 1)
    output_padding = _normalize_conv_param(
        raw_args[7] if len(raw_args) > 7 else 0, ndim, 0
    )

    return _emit_conv_transpose(
        P,
        n,
        x_node,
        w_node,
        bias_node,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        ndim,
    )


@REGISTRY.register(target=[torch.ops.aten.conv_transpose1d.default])
def _conv_transpose1d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.conv_transpose1d: (input, weight, bias, stride, padding, output_padding, groups, dilation)."""
    require_args(n.args, 2, 8, "aten.conv_transpose1d")
    require_kwargs(P.kwargs(n), set(), "aten.conv_transpose1d")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    groups = n.args[6] if len(n.args) > 6 else 1

    stride = _normalize_conv_param(n.args[3] if len(n.args) > 3 else 1, 1, 1)
    padding = _normalize_conv_param(n.args[4] if len(n.args) > 4 else 0, 1, 0)
    output_padding = _normalize_conv_param(n.args[5] if len(n.args) > 5 else 0, 1, 0)
    dilation = _normalize_conv_param(n.args[7] if len(n.args) > 7 else 1, 1, 1)

    return _emit_conv_transpose(
        P,
        n,
        x_node,
        w_node,
        bias_node,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        ndim=1,
    )


@REGISTRY.register(target=[torch.ops.aten.conv_transpose2d.input])
def _conv_transpose2d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.conv_transpose2d: (input, weight, bias, stride, padding, output_padding, groups, dilation)."""
    require_args(n.args, 2, 8, "aten.conv_transpose2d")
    require_kwargs(P.kwargs(n), set(), "aten.conv_transpose2d")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    groups = n.args[6] if len(n.args) > 6 else 1

    stride = _normalize_conv_param(n.args[3] if len(n.args) > 3 else [1, 1], 2, 1)
    padding = _normalize_conv_param(n.args[4] if len(n.args) > 4 else [0, 0], 2, 0)
    output_padding = _normalize_conv_param(
        n.args[5] if len(n.args) > 5 else [0, 0], 2, 0
    )
    dilation = _normalize_conv_param(n.args[7] if len(n.args) > 7 else [1, 1], 2, 1)

    return _emit_conv_transpose(
        P,
        n,
        x_node,
        w_node,
        bias_node,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        ndim=2,
    )


@REGISTRY.register(target=[torch.ops.aten.conv_transpose3d.input])
def _conv_transpose3d_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.conv_transpose3d: (input, weight, bias, stride, padding, output_padding, groups, dilation)."""
    require_args(n.args, 2, 8, "aten.conv_transpose3d")
    require_kwargs(P.kwargs(n), set(), "aten.conv_transpose3d")
    x_node, w_node = n.args[0:2]
    bias_node = n.args[2] if len(n.args) > 2 else None
    groups = n.args[6] if len(n.args) > 6 else 1

    stride = _normalize_conv_param(n.args[3] if len(n.args) > 3 else [1, 1, 1], 3, 1)
    padding = _normalize_conv_param(n.args[4] if len(n.args) > 4 else [0, 0, 0], 3, 0)
    output_padding = _normalize_conv_param(
        n.args[5] if len(n.args) > 5 else [0, 0, 0], 3, 0
    )
    dilation = _normalize_conv_param(n.args[7] if len(n.args) > 7 else [1, 1, 1], 3, 1)

    return _emit_conv_transpose(
        P,
        n,
        x_node,
        w_node,
        bias_node,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        ndim=3,
    )


@REGISTRY.register(target=[torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar])
def _sub_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.sub.Tensor: a - alpha * b."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.sub.Tensor")
    require_kwargs(P.kwargs(n), {"alpha"}, "aten.sub.Tensor")
    a, b = args
    input_meta = n.args[0].meta.get("val")
    dtype = input_meta.dtype if input_meta is not None else torch.float32
    if not isinstance(b, Slot):
        b = emit_lifted_constant(P, b, dtype)
    alpha = P.kwargs(n).get("alpha", 1)
    if alpha != 1:
        alpha_slot = emit_lifted_constant(P, alpha, dtype)
        _, tmp = P.make_tmp_slot()
        P.emit(
            MultiplyNode(
                a=P.slot_to_tid(b),
                b=P.slot_to_tid(alpha_slot),
                out=P.slot_to_tid(tmp),
            )
        )
        b = tmp
    out = P.make_or_get_slot(n)
    P.emit(
        SubtractNode(
            a=P.slot_to_tid(a),
            b=P.slot_to_tid(b),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.relu.default])
def _relu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.relu.default - rectified linear unit.

    ReLU(x) = max(x, 0), implemented using MaximumNode with a scalar zero.
    Uses broadcasting in maximum operation for efficiency.
    """
    args = P.args(n)
    require_args(args, 1, 1, "aten.relu")
    require_kwargs(P.kwargs(n), set(), "aten.relu")
    (x,) = args  # x is already a Slot

    x_meta = n.args[0].meta.get("val")
    if x_meta is None:
        raise ValueError("Input tensor metadata not found for relu")
    dtype = x_meta.dtype

    zero_slot = emit_lifted_constant(P, 0.0, dtype)

    out = P.make_or_get_slot(n)
    P.emit(
        MaximumNode(
            a=P.slot_to_tid(x),
            b=P.slot_to_tid(zero_slot),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten._log_softmax.default])
def _log_softmax_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten._log_softmax.default - log of softmax.

    LogSoftmax(x, dim) = x - logsumexp(x, dim, keepdims=True)

    This is numerically stable because it avoids computing softmax
    (which can underflow to 0) followed by log (which gives -inf for 0).
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten._log_softmax")
    require_kwargs(P.kwargs(n), set(), "aten._log_softmax")
    x, dim, _half_to_float = args  # x is already a Slot

    # Create temporary slot for logsumexp output
    _, logsumexp_slot = P.make_tmp_slot()

    # Emit LogSumExpNode with keepdims=True
    P.emit(
        LogSumExpNode(
            x=P.slot_to_tid(x),
            axes=[dim],
            keepdims=True,
            out=P.slot_to_tid(logsumexp_slot),
        )
    )

    # Emit SubtractNode: x - logsumexp(x)
    out = P.make_or_get_slot(n)
    P.emit(
        SubtractNode(
            a=P.slot_to_tid(x),
            b=P.slot_to_tid(logsumexp_slot),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.constant_pad_nd.default])
def _constant_pad_nd_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.constant_pad_nd - pad with a constant value.

    PyTorch pad format: [left_0, right_0, left_1, right_1, ...]
    MLX pad_width format: [(before_0, after_0), (before_1, after_1), ...]

    Note: PyTorch pads in reverse order (last dimensions first).
    """
    args = P.args(n)
    require_args(args, 2, 3, "aten.constant_pad_nd")
    require_kwargs(P.kwargs(n), set(), "aten.constant_pad_nd")
    x_node, pad = args[0], args[1]
    value = args[2] if len(args) > 2 else 0

    if not isinstance(value, (int, float)):
        raise ValueError(
            f"aten.constant_pad_nd: constant value must be a scalar, got {type(value)}"
        )

    # Convert PyTorch pad format to MLX pad_width format
    # PyTorch: [left_D, right_D, left_D-1, right_D-1, ...]
    # MLX: [(before_0, after_0), (before_1, after_1), ..., (before_D, after_D)]
    if len(pad) % 2 != 0:
        raise ValueError(
            f"aten.constant_pad_nd: pad length must be even, got {len(pad)}"
        )

    x = P.slot_map([x_node])[0]
    x_meta = n.args[0].meta.get("val")
    if x_meta is None:
        raise ValueError("Input tensor metadata not found for constant_pad_nd")

    ndim = len(x_meta.shape)
    num_pad_dims = len(pad) // 2

    if num_pad_dims > ndim:
        raise ValueError(
            f"aten.constant_pad_nd: trying to pad {num_pad_dims} dimensions "
            f"but input has only {ndim} dimensions"
        )

    # Build MLX pad_width: start with zeros for non-padded dims
    pad_width = []
    for _ in range(ndim - num_pad_dims):
        pad_width.extend([0, 0])  # No padding for these dimensions

    # Add padding for the padded dimensions (reverse order)
    for i in range(num_pad_dims - 1, -1, -1):
        left = pad[i * 2]
        right = pad[i * 2 + 1]
        pad_width.extend([left, right])

    out = P.make_or_get_slot(n)
    P.emit(
        PadNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            pad_width=[P.to_int_or_vid(v) for v in pad_width],
            mode="constant",
            constant_value=float(value),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.clamp.default, torch.ops.aten.clamp.Tensor])
def _clamp_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.clamp - clamp values to [min, max] range.

    clamp(input, min=None, max=None) -> Tensor

    Clamps all elements in input into the range [min, max].
    If min is None, there is no lower bound. If max is None, there is no upper bound.
    """
    args = P.args(n)
    require_args(args, 1, 3, "aten.clamp")
    require_kwargs(P.kwargs(n), set(), "aten.clamp")

    x = args[0]
    min_val = args[1] if len(args) > 1 else None
    max_val = args[2] if len(args) > 2 else None

    x_meta = n.args[0].meta.get("val")
    if x_meta is None:
        raise ValueError("Input tensor metadata not found for clamp")
    dtype = x_meta.dtype

    out = P.make_or_get_slot(n)

    # Lift scalar bounds to 0-D constant tensors
    a_min_tid = None
    a_max_tid = None
    if min_val is not None:
        if isinstance(min_val, Slot) and min_val.id_type == IdType.Tensor:
            a_min_tid = P.slot_to_tid(min_val)
        else:
            a_min_tid = P.slot_to_tid(emit_lifted_constant(P, float(min_val), dtype))
    if max_val is not None:
        if isinstance(max_val, Slot) and max_val.id_type == IdType.Tensor:
            a_max_tid = P.slot_to_tid(max_val)
        else:
            a_max_tid = P.slot_to_tid(emit_lifted_constant(P, float(max_val), dtype))

    P.emit(
        ClipNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            a_min=a_min_tid,
            a_max=a_max_tid,
        )
    )
    return out


@REGISTRY.register(
    target=[torch.ops.aten.expand.default, torch.ops.aten.expand_copy.default]
)
def _expand_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle expand: broadcasts dimensions of size 1 to larger sizes."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.expand")
    require_kwargs(P.kwargs(n), set(), "aten.expand")
    x, size = args
    out = P.make_or_get_slot(n)

    shape_iovs = [P.to_int_or_vid(s) for s in size]
    P.emit(
        BroadcastToNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            shape=shape_iovs,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten._native_batch_norm_legit_no_training.default])
def _native_batch_norm_legit_no_training_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle batch norm inference (no training).

    Formula: output = (input - mean) / sqrt(var + eps) * weight + bias

    Args:
        input: [N, C, ...] tensor
        weight: [C] gamma parameter
        bias: [C] beta parameter
        running_mean: [C]
        running_var: [C]
        momentum: float (unused in inference)
        eps: float

    Returns:
        Tuple of (output, empty, empty) - save_mean and save_invstd are empty for no_training
    """
    args = P.args(n)
    require_args(args, 7, 7, "aten._native_batch_norm_legit_no_training")
    require_kwargs(P.kwargs(n), set(), "aten._native_batch_norm_legit_no_training")
    x = args[0]
    weight = args[1]  # gamma [C] - optional (None if affine=False)
    bias = args[2]  # beta [C] - optional (None if affine=False)
    mean = args[3]  # running_mean [C]
    var = args[4]  # running_var [C]
    # momentum = args[5] - not used in inference
    eps = args[6]  # epsilon

    # Get output slots (3 outputs: normalized, save_mean, save_invstd)
    output_slots = P.make_or_get_slots(n)
    out = output_slots[0]  # Main output

    # Get input ndim to determine reshape dimensions
    # For BatchNorm1d: input is [N, C, L] -> reshape params to [1, C, 1]
    # For BatchNorm2d: input is [N, C, H, W] -> reshape params to [1, C, 1, 1]
    input_node = n.args[0]
    input_ndim = len(input_node.meta["val"].shape)

    # Validate input dimensionality (only 3D and 4D supported)
    if input_ndim not in (3, 4):
        raise NotImplementedError(
            f"MLX batch norm handler only supports 3D (BatchNorm1d) and 4D (BatchNorm2d) inputs. "
            f"Got {input_ndim}D input."
        )

    def reshape_for_broadcast(slot, name_suffix):
        """Reshape a [C] tensor for broadcasting with input."""
        _, reshaped = P.make_tmp_slot()
        # Build shape: [1, -1] + [1] * (ndim - 2)
        shape = [P.to_int_or_vid(1), P.to_int_or_vid(-1)]
        for _ in range(input_ndim - 2):
            shape.append(P.to_int_or_vid(1))
        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(slot),
                shape=shape,
                out=P.slot_to_tid(reshaped),
            )
        )
        return reshaped

    mean_reshaped = reshape_for_broadcast(mean, "mean")
    var_reshaped = reshape_for_broadcast(var, "var")

    # Step 1: x_centered = x - mean
    _, tmp_centered = P.make_tmp_slot()
    P.emit(
        SubtractNode(
            a=P.slot_to_tid(x),
            b=P.slot_to_tid(mean_reshaped),
            out=P.slot_to_tid(tmp_centered),
        )
    )

    # Step 2: var_eps = var + eps
    eps_slot = emit_lifted_constant(P, float(eps), torch.float32)
    _, tmp_var_eps = P.make_tmp_slot()
    P.emit(
        AddNode(
            a=P.slot_to_tid(var_reshaped),
            b=P.slot_to_tid(eps_slot),
            out=P.slot_to_tid(tmp_var_eps),
        )
    )

    # Step 3: inv_std = rsqrt(var_eps)
    _, tmp_inv_std = P.make_tmp_slot()
    P.emit(RsqrtNode(x=P.slot_to_tid(tmp_var_eps), out=P.slot_to_tid(tmp_inv_std)))

    # Step 4: x_normalized = x_centered * inv_std
    _, tmp_normalized = P.make_tmp_slot()
    P.emit(
        MultiplyNode(
            a=P.slot_to_tid(tmp_centered),
            b=P.slot_to_tid(tmp_inv_std),
            out=P.slot_to_tid(tmp_normalized),
        )
    )

    # Step 5: x_scaled = x_normalized * weight (skip if weight is None, i.e. affine=False)
    if weight is not None:
        weight_reshaped = reshape_for_broadcast(weight, "weight")
        _, tmp_scaled = P.make_tmp_slot()
        P.emit(
            MultiplyNode(
                a=P.slot_to_tid(tmp_normalized),
                b=P.slot_to_tid(weight_reshaped),
                out=P.slot_to_tid(tmp_scaled),
            )
        )
        current_result = tmp_scaled
    else:
        current_result = tmp_normalized

    # Step 6: out = current_result + bias (skip if bias is None, i.e. affine=False)
    if bias is not None:
        bias_reshaped = reshape_for_broadcast(bias, "bias")
        P.emit(
            AddNode(
                a=P.slot_to_tid(current_result),
                b=P.slot_to_tid(bias_reshaped),
                out=P.slot_to_tid(out),
            )
        )
    else:
        # No bias - just copy the result to output
        P.emit(
            IdCopyNode(
                x=P.slot_to_tid(current_result),
                out=P.slot_to_tid(out),
            )
        )

    # For no_training mode, outputs 1 and 2 (save_mean, save_invstd) are empty
    # They should already be allocated by make_or_get_slots but we don't write to them
    # PyTorch returns empty tensors for these in no_training mode

    return output_slots


@REGISTRY.register(target=[torch.ops.aten.where.self])
def _where_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle where: select from x or y according to condition.

    where(condition, x, y) returns elements from x where condition is True,
    and elements from y where condition is False.
    """
    args = P.args(n)
    require_args(args, 3, 3, "aten.where")
    require_kwargs(P.kwargs(n), set(), "aten.where")
    condition, x, y = args
    out = P.make_or_get_slot(n)

    P.emit(
        WhereNode(
            condition=P.slot_to_tid(condition),
            x=P.slot_to_tid(x),
            y=P.slot_to_tid(y),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.bitwise_not.default])
def _bitwise_not_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.bitwise_not - for boolean tensors, dispatch to logical_not."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.bitwise_not")
    require_kwargs(P.kwargs(n), set(), "aten.bitwise_not")
    x_meta = n.args[0].meta.get("val")

    if x_meta is not None and x_meta.dtype == torch.bool:
        # For boolean tensors, bitwise_not is equivalent to logical_not
        out = P.make_or_get_slot(n)
        P.emit(
            LogicalNotNode(
                x=P.slot_to_tid(args[0]),
                out=P.slot_to_tid(out),
            )
        )
        return out
    else:
        raise NotImplementedError(
            f"aten.bitwise_not is only supported for boolean tensors. "
            f"Got dtype={x_meta.dtype if x_meta else 'unknown'}"
        )


@REGISTRY.register(
    target=[torch.ops.aten.logical_and.default, torch.ops.aten.bitwise_and.Tensor]
)
def _logical_and_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.logical_and / aten.bitwise_and on bool tensors."""
    args = P.args(n)
    require_args(args, 2, 2, "aten.logical_and/bitwise_and")
    require_kwargs(P.kwargs(n), set(), "aten.logical_and/bitwise_and")

    # bitwise_and is only equivalent to logical_and for bool tensors.
    if n.target == torch.ops.aten.bitwise_and.Tensor:
        dtype = n.args[0].meta.get("val", None)
        if dtype is not None and hasattr(dtype, "dtype") and dtype.dtype != torch.bool:
            raise ValueError(
                f"aten.bitwise_and on non-bool dtype {dtype.dtype} is not supported; "
                "only bool tensors can be lowered via LogicalAndNode"
            )
    out = P.make_or_get_slot(n)
    P.emit(
        LogicalAndNode(
            a=P.slot_to_tid(args[0]),
            b=P.slot_to_tid(args[1]),
            out=P.slot_to_tid(out),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.scalar_tensor.default])
def _scalar_tensor_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """This is equivalent to torch.full([], scalar, dtype=dtype)."""
    args = P.args(n)
    kwargs = P.kwargs(n)
    require_args(args, 1, 1, "aten.scalar_tensor")
    require_kwargs(
        kwargs, {"dtype", "layout", "device", "pin_memory"}, "aten.scalar_tensor"
    )
    require_contiguous_format(
        layout=kwargs.get("layout"),
        op_name="aten.scalar_tensor",
    )
    scalar_value = args[0]

    out = P.make_or_get_slot(n)

    # Get dtype from kwargs, default to float32
    dtype = n.kwargs.get("dtype")
    if dtype is None:
        # Infer dtype from scalar type
        if isinstance(scalar_value, bool):
            dtype = torch.bool
        elif isinstance(scalar_value, int):
            dtype = torch.int64
        else:
            dtype = torch.float32

    P.emit(
        FullNode(
            out=P.slot_to_tid(out),
            shape=[],  # 0-D tensor (scalar)
            v=P.to_float_or_vid(scalar_value),
            scalar_type=torch_dtype_to_scalar_type(dtype),
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.tril.default])
def _tril_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.tril - extract lower triangular part of matrix.

    tril(input, diagonal=0) -> Tensor

    Returns the lower triangular part of the matrix, with all elements above
    the diagonal set to zero. The diagonal parameter controls which diagonal
    to consider: 0 = main diagonal, positive = above main, negative = below main.
    """
    args = P.args(n)
    require_args(args, 1, 2, "aten.tril")
    require_kwargs(P.kwargs(n), set(), "aten.tril")
    x = args[0]
    diagonal = args[1] if len(args) > 1 else 0

    out = P.make_or_get_slot(n)
    P.emit(
        TrilNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            k=diagonal,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.triu.default])
def _triu_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.triu - extract upper triangular part of matrix.

    triu(input, diagonal=0) -> Tensor

    Returns the upper triangular part of the matrix, with all elements below
    the diagonal set to zero. The diagonal parameter controls which diagonal
    to consider: 0 = main diagonal, positive = above main, negative = below main.
    """
    args = P.args(n)
    require_args(args, 1, 2, "aten.triu")
    require_kwargs(P.kwargs(n), set(), "aten.triu")
    x = args[0]
    diagonal = args[1] if len(args) > 1 else 0

    out = P.make_or_get_slot(n)
    P.emit(
        TriuNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            k=diagonal,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.round.default])
def _round_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.round - round elements to nearest integer.

    Note: round.decimals variant is not supported as it's not in Core ATen.
    """
    args = P.args(n)
    require_args(args, 1, 1, "aten.round")
    require_kwargs(P.kwargs(n), set(), "aten.round")
    x = args[0]
    out = P.make_or_get_slot(n)
    P.emit(RoundNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), decimals=0))
    return out


@REGISTRY.register(target=[torch.ops.aten.logsumexp.default])
def _logsumexp_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.logsumexp - log(sum(exp(x))) along axes."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.logsumexp")
    require_kwargs(P.kwargs(n), set(), "aten.logsumexp")
    x = args[0]
    dim = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False

    # Normalize dim to list
    if dim is None:
        axes = []
    elif isinstance(dim, int):
        axes = [dim]
    else:
        axes = list(dim)

    out = P.make_or_get_slot(n)
    P.emit(
        LogSumExpNode(
            x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=axes, keepdims=keepdim
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.var.correction, torch.ops.aten.var.dim])
def _var_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.var - variance of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 2, "aten.var")
    require_kwargs(P.kwargs(n), {"correction", "keepdim"}, "aten.var")
    x = args[0]
    axes, _ = normalize_reduction_dim(args)

    # Get correction/ddof and keepdim from kwargs
    correction = n.kwargs.get("correction", None)
    keepdim = n.kwargs.get("keepdim", False)
    ddof = int(correction) if correction is not None else 1

    out = P.make_or_get_slot(n)
    P.emit(
        VarNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axes=axes,
            keepdims=keepdim,
            ddof=ddof,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.std.correction])
def _std_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.std - standard deviation of elements along axes."""
    args = P.args(n)
    require_args(args, 1, 2, "aten.std")
    require_kwargs(P.kwargs(n), {"correction", "keepdim"}, "aten.std")
    x = args[0]
    axes, _ = normalize_reduction_dim(args)

    correction = n.kwargs.get("correction", None)
    keepdim = n.kwargs.get("keepdim", False)
    ddof = int(correction) if correction is not None else 1

    out = P.make_or_get_slot(n)
    P.emit(
        StdNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axes=axes,
            keepdims=keepdim,
            ddof=ddof,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.max.default])
def _max_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.max.default - global max (reduce all axes)."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.max")
    require_kwargs(P.kwargs(n), set(), "aten.max")
    x = args[0]

    out = P.make_or_get_slot(n)
    P.emit(MaxNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=[], keepdims=False))
    return out


@REGISTRY.register(target=[torch.ops.aten.min.default])
def _min_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.min.default - global min (reduce all axes)."""
    args = P.args(n)
    require_args(args, 1, 1, "aten.min")
    require_kwargs(P.kwargs(n), set(), "aten.min")
    x = args[0]

    out = P.make_or_get_slot(n)
    P.emit(MinNode(x=P.slot_to_tid(x), out=P.slot_to_tid(out), axes=[], keepdims=False))
    return out


@REGISTRY.register(target=[torch.ops.aten.argmax.default])
def _argmax_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.argmax - index of max element along axis."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.argmax")
    require_kwargs(P.kwargs(n), set(), "aten.argmax")
    x = args[0]
    dim = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False

    out = P.make_or_get_slot(n)

    if dim is None:
        # argmax without dim: flatten tensor to 1D, then argmax over axis 0
        # Result is a scalar index into the flattened tensor
        _, flat_slot = P.make_tmp_slot()

        # Get total number of elements from input shape
        x_meta = n.args[0].meta.get("val")
        if x_meta is None:
            raise ValueError("Input tensor metadata not found for argmax")
        numel = x_meta.numel()

        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(flat_slot),
                shape=[P.to_int_or_vid(numel)],
            )
        )
        P.emit(
            ArgmaxNode(
                x=P.slot_to_tid(flat_slot),
                out=P.slot_to_tid(out),
                axis=0,
                keepdims=False,
            )
        )
    else:
        P.emit(
            ArgmaxNode(
                x=P.slot_to_tid(x), out=P.slot_to_tid(out), axis=dim, keepdims=keepdim
            )
        )
    return out


@REGISTRY.register(target=[torch.ops.aten.argmin.default])
def _argmin_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.argmin - index of min element along axis."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.argmin")
    require_kwargs(P.kwargs(n), set(), "aten.argmin")
    x = args[0]
    dim = args[1] if len(args) > 1 else None
    keepdim = args[2] if len(args) > 2 else False

    out = P.make_or_get_slot(n)

    if dim is None:
        # argmin without dim: flatten tensor to 1D, then argmin over axis 0
        # Result is a scalar index into the flattened tensor
        _, flat_slot = P.make_tmp_slot()

        # Get total number of elements from input shape
        x_meta = n.args[0].meta.get("val")
        if x_meta is None:
            raise ValueError("Input tensor metadata not found for argmin")
        numel = x_meta.numel()

        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(flat_slot),
                shape=[P.to_int_or_vid(numel)],
            )
        )
        P.emit(
            ArgminNode(
                x=P.slot_to_tid(flat_slot),
                out=P.slot_to_tid(out),
                axis=0,
                keepdims=False,
            )
        )
    else:
        P.emit(
            ArgminNode(
                x=P.slot_to_tid(x), out=P.slot_to_tid(out), axis=dim, keepdims=keepdim
            )
        )
    return out


def _parse_pool_args(args, ndim, op_name, is_avg_pool=False):  # noqa: C901
    """Parse pooling op arguments, normalizing scalars to lists.

    ATen pooling signatures:
      max_pool{N}d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
      avg_pool{N}d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    Extra args beyond (input, kernel_size, stride, padding) are accepted only
    when they match safe defaults:
      max_pool: dilation=1, ceil_mode=False
      avg_pool: ceil_mode=False, count_include_pad=True, divisor_override=None

    Returns (kernel_size, stride, padding) as lists of length ndim.
    """
    if is_avg_pool:
        require_args(args, 2, 7, op_name)
        # args[4] = ceil_mode (must be False)
        if len(args) > 4 and args[4]:
            raise ValueError(f"{op_name}: ceil_mode=True is not supported.")
        # args[5] = count_include_pad (must be True)
        if len(args) > 5 and not args[5]:
            raise ValueError(f"{op_name}: count_include_pad=False is not supported.")
        # args[6] = divisor_override (must be None)
        if len(args) > 6 and args[6] is not None:
            raise ValueError(f"{op_name}: divisor_override is not supported.")
    else:
        require_args(args, 2, 6, op_name)
        # args[4] = dilation (must be 1)
        if len(args) > 4:
            dilation = args[4]
            if isinstance(dilation, list):
                if any(d != 1 for d in dilation):
                    raise ValueError(
                        f"{op_name}: dilation != 1 is not supported, got {dilation}."
                    )
            elif dilation != 1:
                raise ValueError(
                    f"{op_name}: dilation != 1 is not supported, got {dilation}."
                )
        # args[5] = ceil_mode (must be False)
        if len(args) > 5 and args[5]:
            raise ValueError(f"{op_name}: ceil_mode=True is not supported.")

    kernel_size = args[1]
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim

    stride = args[2] if len(args) > 2 and args[2] else kernel_size
    if isinstance(stride, int):
        stride = [stride] * ndim
    if not stride:  # empty list means default to kernel_size
        stride = list(kernel_size)

    padding = args[3] if len(args) > 3 else [0] * ndim
    if isinstance(padding, int):
        padding = [padding] * ndim

    return list(kernel_size), list(stride), list(padding)


def _emit_pool_nd(
    P: MLXProgramBuilder,
    n: Node,
    ndim: int,
    reduce_node_cls: type,
    padding_value: float,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
) -> Slot:
    """Emit IR nodes for N-dimensional pooling.

    Decomposes pooling into:
      Transpose (channels-first -> channels-last)
      -> Pad (if needed)
      -> Reshape+Transpose (fast path) or AsStrided (general path)
      -> Max/Mean reduction over kernel dims
      -> Transpose (channels-last -> channels-first)

    Works for 1D, 2D, and 3D pooling uniformly.

    Args:
        P: Program builder.
        n: FX graph node for the pooling op.
        ndim: Spatial dimensionality (1, 2, or 3).
        reduce_node_cls: MaxNode or MeanNode.
        padding_value: Padding fill value (-inf for max, 0 for avg).
        kernel_size: Kernel size per spatial dim, length ndim.
        stride: Stride per spatial dim, length ndim.
        padding: Padding per spatial dim, length ndim.

    Returns:
        Output Slot with shape [N, C, *out_spatial].
    """
    x_node = P.args(n)[0]
    (x,) = P.slot_map([x_node])
    x_meta = n.args[0].meta["val"]
    shape = list(x_meta.shape)  # [N, C, *spatial]

    N = shape[0]
    C = shape[1]
    spatial = shape[2:]  # length == ndim

    # 1. Transpose: channels-first [N, C, *spatial] -> channels-last [N, *spatial, C]
    to_cl = [0] + list(range(2, ndim + 2)) + [1]
    _, cur = P.make_tmp_slot()
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(cur),
            perm=to_cl,
        )
    )

    # 2. Pad spatial dims if needed
    spatial_padded = [s + 2 * p for s, p in zip(spatial, padding)]
    if any(p > 0 for p in padding):
        pad_width = [0, 0]  # batch dim: no pad
        for p in padding:
            pad_width += [p, p]
        pad_width += [0, 0]  # channel dim: no pad
        P.emit(
            PadNode(
                x=P.slot_to_tid(cur),
                out=P.slot_to_tid(cur),
                pad_width=[P.to_int_or_vid(v) for v in pad_width],
                mode="constant",
                constant_value=padding_value,
            )
        )

    # 3. Sliding windows -> [N, *out_spatial, *kernel_size, C]
    out_spatial = [
        (sp - k) // s + 1 for sp, k, s in zip(spatial_padded, kernel_size, stride)
    ]

    can_fast_path = all(
        k == s and sp % k == 0 for k, s, sp in zip(kernel_size, stride, spatial_padded)
    )

    if can_fast_path:
        # Fast path: reshape + transpose (no AsStridedNode needed).
        # [N, *spatial_padded, C]
        #   -> reshape [N, sp0//k0, k0, sp1//k1, k1, ..., C]
        #   -> transpose to gather output-spatial dims, then kernel dims, then C
        reshape_shape = [N]
        for sp, k in zip(spatial_padded, kernel_size):
            reshape_shape += [sp // k, k]
        reshape_shape += [C]

        P.emit(
            ReshapeNode(
                x=P.slot_to_tid(cur),
                out=P.slot_to_tid(cur),
                shape=[IntOrVid.from_literal(d) for d in reshape_shape],
            )
        )

        # Transpose: gather output-spatial (odd indices), then kernel (even indices after batch)
        # Reshaped tensor axes: [0=batch, 1=out0, 2=k0, 3=out1, 4=k1, ..., last=C]
        last = 2 * ndim + 1
        out_spatial_axes = list(range(1, last, 2))  # [1, 3, 5, ...]
        kernel_axes = list(range(2, last, 2))  # [2, 4, 6, ...]
        perm = [0] + out_spatial_axes + kernel_axes + [last]

        P.emit(
            TransposeNode(
                x=P.slot_to_tid(cur),
                out=P.slot_to_tid(cur),
                perm=perm,
            )
        )
    else:
        # General path: as_strided to create sliding window view.
        # Input layout: [N, *spatial_padded, C] (channels-last, row-major)
        dims = [N] + spatial_padded + [C]
        elem_strides = []
        acc = 1
        for d in reversed(dims):
            elem_strides.append(acc)
            acc *= d
        elem_strides.reverse()

        # as_strided shape: [N, *out_spatial, *kernel_size, C]
        as_shape = [N] + out_spatial + kernel_size + [C]

        # as_strided strides:
        #   batch:          elem_strides[0]
        #   out_spatial[i]: elem_strides[i+1] * stride[i]  (skip by pool stride)
        #   kernel[i]:      elem_strides[i+1]               (consecutive rows/cols)
        #   channel:        1
        as_strides = [elem_strides[0]]
        for i in range(ndim):
            as_strides.append(elem_strides[i + 1] * stride[i])
        for i in range(ndim):
            as_strides.append(elem_strides[i + 1])
        as_strides.append(1)

        P.emit(
            AsStridedNode(
                x=P.slot_to_tid(cur),
                out=P.slot_to_tid(cur),
                shape=[IntOrVid.from_literal(d) for d in as_shape],
                strides=[IntOrVid.from_literal(d) for d in as_strides],
                offset=0,
            )
        )

    # 4. Reduce over kernel dims (axes [ndim+1 .. 2*ndim])
    reduce_axes = list(range(ndim + 1, 2 * ndim + 1))
    _, reduced = P.make_tmp_slot()
    P.emit(
        reduce_node_cls(
            x=P.slot_to_tid(cur),
            out=P.slot_to_tid(reduced),
            axes=reduce_axes,
            keepdims=False,
        )
    )

    # 5. Transpose: channels-last [N, *out_spatial, C] -> channels-first [N, C, *out_spatial]
    to_cf = [0, ndim + 1] + list(range(1, ndim + 1))
    output_slots = P.make_or_get_slots(n)
    out = output_slots[0]
    P.emit(
        TransposeNode(
            x=P.slot_to_tid(reduced),
            out=P.slot_to_tid(out),
            perm=to_cf,
        )
    )
    return out


_POOL_OPS: List[Tuple[Any, int, type, float, str, bool]] = [
    # (target, ndim, reduce_cls, pad_value, op_name, returns_indices)
    (
        torch.ops.aten.max_pool1d.default,
        1,
        MaxNode,
        float("-inf"),
        "aten.max_pool1d",
        False,
    ),
    (
        torch.ops.aten.max_pool1d_with_indices.default,
        1,
        MaxNode,
        float("-inf"),
        "aten.max_pool1d_with_indices",
        True,
    ),
    (
        torch.ops.aten.max_pool2d_with_indices.default,
        2,
        MaxNode,
        float("-inf"),
        "aten.max_pool2d_with_indices",
        True,
    ),
    (
        torch.ops.aten.max_pool3d_with_indices.default,
        3,
        MaxNode,
        float("-inf"),
        "aten.max_pool3d_with_indices",
        True,
    ),
    (torch.ops.aten.avg_pool1d.default, 1, MeanNode, 0.0, "aten.avg_pool1d", False),
    (torch.ops.aten.avg_pool2d.default, 2, MeanNode, 0.0, "aten.avg_pool2d", False),
    (torch.ops.aten.avg_pool3d.default, 3, MeanNode, 0.0, "aten.avg_pool3d", False),
]


def _make_pool_handler(
    ndim: int,
    reduce_node_cls: type,
    padding_value: float,
    op_name: str,
    returns_indices: bool,
):
    """Create a handler for an N-dimensional pooling op."""

    is_avg = reduce_node_cls is MeanNode

    def handler(P: MLXProgramBuilder, n: Node) -> Slot:
        args = P.args(n)
        kernel_size, stride, padding = _parse_pool_args(
            args, ndim, op_name, is_avg_pool=is_avg
        )
        result = _emit_pool_nd(
            P, n, ndim, reduce_node_cls, padding_value, kernel_size, stride, padding
        )
        if not returns_indices:
            return result

    handler.__name__ = f"_{op_name.replace('.', '_')}_handler"
    handler.__doc__ = f"Handle {op_name} (table-driven pool op)."
    return handler


for _target, _ndim, _cls, _pad, _name, _indices in _POOL_OPS:
    REGISTRY.register(target=[_target])(
        _make_pool_handler(_ndim, _cls, _pad, _name, _indices)
    )


@REGISTRY.register(target=[torch.ops.torchao.dequantize_affine.default])
def _dequantize_affine_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle standalone torchao.dequantize_affine (not fused with linear/embedding).

    MLX's dequantize always operates along the last axis.  When the quantized
    dimension is not last (e.g. Conv2d with block_size=[1,32,1,1]), we permute
    the constant weight/scale/zero_point tensors at compile time so the
    quantized dim becomes last, emit the DequantizeNode, then emit a
    TransposeNode with the inverse permutation to restore the original layout.
    """
    parsed = parse_dequant_node(n)
    if parsed is None:
        raise NotImplementedError(
            f"dequantize_affine: unsupported quantization config at {n}"
        )
    (
        qdata_node,
        scale_node,
        zero_point_node,
        group_size,
        bits,
        out_dtype,
        quantized_dim,
    ) = parsed

    qdata_target, qdata = P.get_placeholder_target_and_tensor(qdata_node)
    zero_point_target, zero_point = P.get_placeholder_target_and_tensor(zero_point_node)
    scale_target, scale = P.get_placeholder_target_and_tensor(scale_node)

    if out_dtype is None:
        out_dtype = scale_node.meta["val"].dtype
    out_scalar_type = torch_dtype_to_scalar_type(out_dtype)

    ndim = qdata.ndim
    needs_permute = quantized_dim != ndim - 1

    if needs_permute:
        perm = list(range(ndim))
        perm.remove(quantized_dim)
        perm.append(quantized_dim)
        qdata = qdata.permute(perm).contiguous()
        scale = scale.permute(perm).contiguous()
        zero_point = zero_point.permute(perm).contiguous()

    # to_mlx_qparams expects 2D tensors; flatten N-D to 2D for packing,
    # then restore the (possibly permuted) leading dimensions afterward.
    permuted_shape = qdata.shape
    qdata_2d = qdata.reshape(-1, qdata.shape[-1])
    scale_2d = scale.reshape(-1, scale.shape[-1])
    zero_point_2d = zero_point.reshape(-1, zero_point.shape[-1])

    Q, B = to_mlx_qparams(qdata_2d, scale_2d, zero_point_2d, bits)

    leading_dims = permuted_shape[:-1]
    Q = Q.reshape(*leading_dims, Q.shape[-1])
    scale_nd = scale_2d.reshape(*leading_dims, scale_2d.shape[-1])
    if B is not None:
        B = B.reshape(*leading_dims, B.shape[-1])

    w = P.make_or_get_constant(f"{qdata_target}_to_packed", Q)
    scale_const = P.make_or_get_constant(f"{scale_target}_scale", scale_nd)
    biases = emit_quantized_biases(
        P, zero_point_target, scale, zero_point, bits, B, scale_const
    )

    if needs_permute:
        _, dequant_tmp = P.make_tmp_slot()
    else:
        dequant_tmp = P.make_or_get_slot(n)

    P.emit(
        DequantizeNode(
            w=P.slot_to_tid(w),
            scales=P.slot_to_tid(scale_const),
            out=P.slot_to_tid(dequant_tmp),
            biases=P.slot_to_tid(biases),
            group_size=group_size,
            bits=bits,
            mode="affine",
            dtype=out_scalar_type,
        )
    )

    if needs_permute:
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out = P.make_or_get_slot(n)
        P.emit(
            TransposeNode(
                x=P.slot_to_tid(dequant_tmp),
                out=P.slot_to_tid(out),
                perm=inv_perm,
            )
        )
    else:
        out = dequant_tmp

    return out


@REGISTRY.register(target=[torch.ops.aten.cumsum.default])
def _cumsum_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.cumsum - cumulative sum along an axis."""
    args = P.args(n)
    require_args(args, 2, 3, "aten.cumsum")
    require_kwargs(P.kwargs(n), {"dtype"}, "aten.cumsum")
    x = args[0]
    dim = args[1]

    out = P.make_or_get_slot(n)
    P.emit(
        CumsumNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.stack.default])
def _stack_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.stack - stack tensors along a new axis."""
    args = P.args(n)
    require_args(args, 1, 2, "aten.stack")
    require_kwargs(P.kwargs(n), set(), "aten.stack")
    tensors_list = args[0]
    dim = args[1] if len(args) > 1 else 0

    out = P.make_or_get_slot(n)
    tensor_tids = [P.slot_to_tid(t) for t in tensors_list]
    P.emit(
        StackNode(
            tensors=tensor_tids,
            out=P.slot_to_tid(out),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.repeat_interleave.self_int])
def _repeat_interleave_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.repeat_interleave - repeat each element along an axis."""
    args = P.args(n)
    require_args(args, 2, 4, "aten.repeat_interleave")
    require_kwargs(P.kwargs(n), {"output_size"}, "aten.repeat_interleave")
    x = args[0]
    repeats = args[1]
    dim = args[2] if len(args) > 2 else 0

    out = P.make_or_get_slot(n)
    P.emit(
        RepeatNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            repeats=P.to_int_or_vid(repeats),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.sort.default])
def _sort_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.sort - sort elements along an axis.

    Returns (values, indices) as a tuple of output slots.
    """
    args = P.args(n)
    require_args(args, 1, 3, "aten.sort")
    require_kwargs(P.kwargs(n), set(), "aten.sort")
    x = args[0]
    dim = args[1] if len(args) > 1 else -1

    # torch.sort returns (values, indices) - 2 outputs
    output_slots = P.make_or_get_slots(n)
    values_slot, indices_slot = output_slots

    used = used_getitem_indices(n)

    if 0 in used:
        P.emit(
            SortNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(values_slot),
                axis=dim,
            )
        )
    if 1 in used:
        P.emit(
            ArgsortNode(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(indices_slot),
                axis=dim,
            )
        )

    return output_slots


@REGISTRY.register(target=[torch.ops.aten.argsort.default])
def _argsort_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.argsort - indices that sort elements along an axis."""
    args = P.args(n)
    require_args(args, 1, 3, "aten.argsort")
    require_kwargs(P.kwargs(n), set(), "aten.argsort")
    x = args[0]
    dim = args[1] if len(args) > 1 else -1

    out = P.make_or_get_slot(n)
    P.emit(
        ArgsortNode(
            x=P.slot_to_tid(x),
            out=P.slot_to_tid(out),
            axis=dim,
        )
    )
    return out


@REGISTRY.register(target=[torch.ops.aten.topk.default])
def _topk_handler(P: MLXProgramBuilder, n: Node) -> Slot:
    """Handle aten.topk - top-k elements along an axis.

    Decomposes into: partition → slice → sort → reverse (for values)
                     argpartition → slice → gather → argsort → reverse → reorder (for indices)

    torch.topk returns (values, indices) sorted descending.
    """
    args = P.args(n)
    require_args(args, 2, 5, "aten.topk")
    require_kwargs(P.kwargs(n), set(), "aten.topk")
    x = args[0]
    k = args[1]
    dim = args[2] if len(args) > 2 else -1

    output_slots = P.make_or_get_slots(n)
    values_slot, indices_slot = output_slots

    used = used_getitem_indices(n)

    # Get dim size from input metadata for forward slice stop
    x_meta = n.args[0].meta.get("val")
    if x_meta is None:
        raise ValueError("Input tensor metadata not found for topk")
    norm_axis = dim if dim >= 0 else dim + len(x_meta.shape)
    dim_size = x_meta.shape[norm_axis]

    # Compute -k for partition index and forward slice start
    if isinstance(k, int):
        neg_k = P.to_int_or_vid(-k)
        # Reverse slice: start=k-1, stop=-(k+1) on the k-sized sliced tensor
        rev_start = P.to_int_or_vid(k - 1)
        rev_stop = P.to_int_or_vid(-(k + 1))
    else:
        # k is dynamic — emit neg_k = k * -1 at runtime
        _, neg_k_slot = P.make_tmp_value_slot()
        P.emit(
            MultiplyIntNode(
                a=P.to_int_or_vid(k),
                b=IntOrVid.from_literal(-1),
                out=P.slot_to_vid(neg_k_slot),
            )
        )
        neg_k = P.to_int_or_vid(neg_k_slot)
        # rev_start = k - 1
        _, rev_start_slot = P.make_tmp_value_slot()
        P.emit(
            AddIntNode(
                a=P.to_int_or_vid(k),
                b=IntOrVid.from_literal(-1),
                out=P.slot_to_vid(rev_start_slot),
            )
        )
        rev_start = P.to_int_or_vid(rev_start_slot)
        # rev_stop = -(k + 1) = neg_k - 1
        _, rev_stop_slot = P.make_tmp_value_slot()
        P.emit(
            AddIntNode(
                a=neg_k,
                b=IntOrVid.from_literal(-1),
                out=P.slot_to_vid(rev_stop_slot),
            )
        )
        rev_stop = P.to_int_or_vid(rev_stop_slot)

    stop_val = P.to_int_or_vid(dim_size)

    def emit_partition_and_slice(node_cls):
        """Emit partition/argpartition → slice last k elements."""
        _, part_tmp = P.make_tmp_slot()
        P.emit(
            node_cls(
                x=P.slot_to_tid(x),
                out=P.slot_to_tid(part_tmp),
                kth=neg_k,
                axis=dim,
            )
        )
        _, slice_tmp = P.make_tmp_slot()
        P.emit(
            SliceNode(
                x=P.slot_to_tid(part_tmp),
                out=P.slot_to_tid(slice_tmp),
                axis=P.to_int_or_vid(dim),
                start=neg_k,
                stop=stop_val,
                step=1,
            )
        )
        return slice_tmp

    def emit_reverse(in_slot, out_slot):
        """Reverse a tensor along dim using slice with step=-1."""
        P.emit(
            SliceNode(
                x=P.slot_to_tid(in_slot),
                out=P.slot_to_tid(out_slot),
                axis=P.to_int_or_vid(dim),
                start=rev_start,
                stop=rev_stop,
                step=-1,
            )
        )

    if 0 in used:
        # partition → slice last k → sort ascending → reverse to descending
        slice_tmp = emit_partition_and_slice(PartitionNode)
        _, sort_tmp = P.make_tmp_slot()
        P.emit(
            SortNode(
                x=P.slot_to_tid(slice_tmp),
                out=P.slot_to_tid(sort_tmp),
                axis=dim,
            )
        )
        emit_reverse(sort_tmp, values_slot)

    if 1 in used:
        # argpartition → slice last k → gather values → argsort → reverse → reorder
        idx_slice_tmp = emit_partition_and_slice(ArgPartitionNode)
        # Gather original values at the partitioned indices
        _, gathered_tmp = P.make_tmp_slot()
        P.emit(
            TakeAlongAxisNode(
                x=P.slot_to_tid(x),
                indices=P.slot_to_tid(idx_slice_tmp),
                out=P.slot_to_tid(gathered_tmp),
                axis=dim,
            )
        )
        # Argsort gathered values ascending → reverse → descending order
        _, order_tmp = P.make_tmp_slot()
        P.emit(
            ArgsortNode(
                x=P.slot_to_tid(gathered_tmp),
                out=P.slot_to_tid(order_tmp),
                axis=dim,
            )
        )
        _, rev_order_tmp = P.make_tmp_slot()
        emit_reverse(order_tmp, rev_order_tmp)
        # Apply descending order to indices
        P.emit(
            TakeAlongAxisNode(
                x=P.slot_to_tid(idx_slice_tmp),
                indices=P.slot_to_tid(rev_order_tmp),
                out=P.slot_to_tid(indices_slot),
                axis=dim,
            )
        )

    return output_slots
