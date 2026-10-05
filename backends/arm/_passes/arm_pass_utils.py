# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
import operator
import traceback
from inspect import isclass
from typing import cast, Optional, Sequence

import torch
import torch.fx
from executorch.backends.arm.common.debug import get_node_debug_info
from executorch.backends.arm.common.type import ensure_type
from executorch.backends.arm.tosa.mapping import TosaSpecialDtype
from executorch.exir import ExportedProgram
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import NodeMetadata
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind

_Dim = int | torch.SymInt


# Meta key marking a value as *carried quantized state*: a tensor whose
# quantization is shared across invocations (e.g. recurrent state fed back frame
# to frame). Such a value's quantization must stay stable, so requantization-
# perturbing rewrites (e.g. FoldScalarMulIntoConvPass) must not fold into a
# convolution whose output reaches it. The marker is model-supplied because
# functional carried state has no structural signal before quantization runs.
CARRIED_QUANT_STATE_META_KEY = "carried_quant_state"


def mark_carried_quant_state(node: torch.fx.Node) -> None:
    """Mark ``node``'s output as carried quantized state (see the meta key
    doc).
    """
    node.meta[CARRIED_QUANT_STATE_META_KEY] = True


def is_carried_quant_state(node: torch.fx.Node) -> bool:
    """True if ``node`` was marked as carried quantized state."""
    return bool(node.meta.get(CARRIED_QUANT_STATE_META_KEY, False))


def is_submodule_node(node: torch.fx.Node):
    if node.op not in ("get_attr", "placeholder"):
        return False
    owning_module = node.graph.owning_module
    if owning_module is None or not isinstance(node.target, str):
        return False
    try:
        owning_module.get_submodule(node.target)
    except AttributeError:
        return False
    return True


def is_get_attr_node(node: torch.fx.Node) -> bool:
    """Returns true if the given node is a get attr node for a tensor of the
    model.
    """
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "get_attr"
        and not is_submodule_node(node)
    )


def get_getitem_users(
    source_node: torch.fx.Node, max_users: int
) -> dict[int, torch.fx.Node | None]:
    getitem_users: dict[int, torch.fx.Node | None] = {i: None for i in range(max_users)}
    for user in source_node.users:
        if user.target == operator.getitem:
            getitem_users[cast(int, user.args[1])] = user

    return getitem_users


def is_param_node(exp_prog: ExportedProgram, node: torch.fx.Node) -> bool:
    return (
        is_get_attr_node(node)
        or is_param(exp_prog, node)
        or is_buffer(exp_prog, node)
        or is_lifted_tensor_constant(exp_prog, node)
    )


def get_constant_placeholder_kind(
    exp_prog: ExportedProgram, node: torch.fx.Node
) -> InputKind:
    if is_param(exp_prog, node):
        return InputKind.PARAMETER
    if is_buffer(exp_prog, node):
        return InputKind.BUFFER
    if is_lifted_tensor_constant(exp_prog, node):
        return InputKind.CONSTANT_TENSOR

    raise RuntimeError("Node is neither PARAMETER, BUFFER nor CONSTANT_TENSOR")


def is_persistent_buffer(exp_prog: ExportedProgram, node: torch.fx.Node) -> bool | None:
    if is_buffer(exp_prog, node):
        buffer_name = exp_prog.graph_signature.inputs_to_buffers[node.name]
        if buffer_name in exp_prog.graph_signature.non_persistent_buffers:
            return False
        else:
            return True

    return None


def get_param_tensor(
    exp_prog: ExportedProgram, node: torch.fx.Node
) -> Optional[torch.Tensor]:
    if node is None:
        return None
    elif is_param(exp_prog, node):
        return get_param(exp_prog, node)
    elif is_buffer(exp_prog, node):
        return get_buffer(exp_prog, node)
    elif is_lifted_tensor_constant(exp_prog, node):
        return get_lifted_tensor_constant(exp_prog, node)
    elif is_get_attr_node(node):
        target_node = ensure_type(str, node.target)
        # This is a hack to support both lifted and unlifted graph
        try:
            return getattr(node.graph.owning_module, target_node)
        except AttributeError:
            return getattr(exp_prog.graph_module, target_node)
    raise RuntimeError(f"unsupported param type, {node.op}.")


def expand_around_channel(param: Sequence[int] | int, spatial_rank: int) -> list[int]:
    """Expand a scalar or 1-D parameter around the channel dimension into a
    broadcastable shape while preserving the channel location.
    """
    if isinstance(param, int):
        return [param] * spatial_rank

    param_list = list(param)
    if len(param_list) == 1 and spatial_rank > 1:
        param_list = param_list * spatial_rank
    return param_list


def create_node(
    graph: torch.fx.Graph,
    op_target: OpOverload | EdgeOpOverload,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    quantize: bool = False,
    q_params: Optional[tuple] = None,
    from_node: Optional[torch.fx.Node] = None,
    inherit_qparams: bool = False,
):
    """Adds a node to 'graph'.

    graph.inserting_before/after() should be used before the call to decide
    where to insert the node. If quantize is true and q_params is not None, a q
    dq pair is inserted after the newly created node.

    """

    node = graph.create_node(
        "call_function",
        op_target,
        args=args,
        kwargs=kwargs or {},
    )

    new_meta = {}
    if from_node:
        keys = from_node.meta.keys()
        for key in keys:
            new_meta[key] = from_node.meta[key]
        if not inherit_qparams:
            if "input_qparams" in new_meta:
                new_meta["input_qparams"] = {}
            if "output_qparams" in new_meta:
                new_meta["output_qparams"] = {}
    elif inherit_qparams:
        raise ValueError("inherit_qparams is only valid when from_node is given")

    old_stack_trace = new_meta.get("stack_trace", "")
    new_meta["stack_trace"] = f"{old_stack_trace}\n{traceback.format_stack()[-2]}"
    node.meta = new_meta

    if quantize and q_params:
        return insert_q_dq_pair(graph, node, q_params, from_node)
    return node


def create_shape_node(
    graph: torch.fx.Graph,
    op_target: EdgeOpOverload,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    from_node: Optional[torch.fx.Node] = None,
):
    """Adds a shape node to 'graph'.

    graph.inserting_before/after() should be used before the call to decide
    where to insert the node.

    """
    node = create_node(
        graph=graph,
        op_target=op_target,
        args=args,
        kwargs=kwargs,
        from_node=from_node,
    )
    node.meta[TosaSpecialDtype.meta_key()] = TosaSpecialDtype.SHAPE
    return node


def insert_q_dq_pair(
    graph: torch.fx.Graph,
    anchor: torch.fx.Node,
    q_params: tuple,
    from_node: Optional[torch.fx.Node] = None,
):
    """Inserts a q dq node pair after the node 'anchor'."""

    with graph.inserting_after(anchor):
        q = create_node(
            graph=graph,
            op_target=exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
            args=(),  # We add the argument last
            from_node=from_node if from_node else anchor,
        )
        q.meta = anchor.meta
    with graph.inserting_after(q):
        dq = create_node(
            graph=graph,
            op_target=exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
            args=(q,) + q_params,
            from_node=from_node if from_node else anchor,
        )
        dq.meta = q.meta
    anchor.replace_all_uses_with(dq)
    # We add this last so the replace all uses above does not replace the quantized
    # node's first use
    q.args = (anchor,) + q_params
    return dq


def meta_without_qparams(meta: NodeMetadata) -> NodeMetadata:
    """Return a copy of NodeMetadata with input/output qparams cleared."""
    plain_meta_dict = dict(meta.data)
    plain_meta_dict["input_qparams"] = {}
    plain_meta_dict["output_qparams"] = {}
    return NodeMetadata(plain_meta_dict)


def insert_scalar(
    graph: torch.fx.Graph,
    value: int | float,
    meta: NodeMetadata | dict,
    from_node: torch.fx.Node,
    is_tfa_pass: bool = False,
) -> torch.fx.Node | int | float:
    """Insert an `aten.full` scalar node for direct graph-rewrite passes."""

    if is_tfa_pass:
        return value

    kwargs = {}
    val = None
    if "val" in meta:
        val = meta["val"]
        if isinstance(val, tuple):
            val = val[0]
        kwargs = {"device": val.device, "dtype": val.dtype}

    scalar = create_node(
        graph=graph,
        op_target=exir_ops.edge.aten.full.default,
        args=((1,), value),
        kwargs=kwargs,
        from_node=from_node,
    )
    if val is not None:
        scalar.meta["val"] = torch.full((1,), value, **kwargs)
    return scalar


def _get_aten_target(node: torch.fx.Node):
    """Return the underlying ATen target for ATen and Edge operators."""
    target = node.target
    if isinstance(target, EdgeOpOverload):
        target = target._op
    return target


def _tensor_constant_to_float(value) -> float | None:
    """Return a scalar tensor constant as float, or None if not scalar."""
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        return None

    try:
        item = value.detach().cpu().item()
    except Exception:
        return None

    if isinstance(item, bool) or not isinstance(item, (int, float)):
        return None

    return float(item)


def _get_attr_value(node: torch.fx.Node):
    """Resolve a get_attr node from its completed owning GraphModule."""
    if node.op != "get_attr" or not isinstance(node.target, str):
        return None

    owning_module = node.graph.owning_module
    if owning_module is None:
        return None

    value = owning_module
    try:
        for part in node.target.split("."):
            value = getattr(value, part)
    except AttributeError:
        return None

    return value


def _get_constant_scalar_value(value) -> float | None:
    """Resolve a scalar value only when it is provably compile-time constant.

    Runtime placeholders and arbitrary FX tensor nodes are deliberately
    rejected. A small set of value-preserving wrappers and constant-only
    multiplication is accepted so positivity proofs survive normal TFA
    canonicalization, including DecomposeAddSubAlphaPass followed by
    ScalarsToAttributePass.

    """
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, torch.fx.Node):
        return None

    # Materialized constants created by ScalarsToAttributePass are get_attr nodes
    # on the completed input GraphModule.
    attr_value = _get_attr_value(value)
    scalar = _tensor_constant_to_float(attr_value)
    if scalar is not None:
        return scalar

    # Export can also retain a real compile-time value on a FakeTensor constant.
    meta_value = value.meta.get("val")
    constant = getattr(meta_value, "constant", None)
    scalar = _tensor_constant_to_float(constant)
    if scalar is not None:
        return scalar

    if value.op != "call_function" or not value.args:
        return None

    target = _get_aten_target(value)

    constant_passthrough_ops = {
        torch.ops.aten.lift_fresh_copy.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.detach_.default,
        torch.ops.aten.alias.default,
    }
    if target in constant_passthrough_ops:
        return _get_constant_scalar_value(value.args[0])

    # DecomposeAddSubAlphaPass can turn a positive scalar contribution into a
    # constant-only mul node. Evaluate only when *both* inputs are compile-time
    # scalars; never use runtime tensor values for the proof.
    constant_mul_ops = {
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
    }
    if target in constant_mul_ops and len(value.args) >= 2:
        lhs = _get_constant_scalar_value(value.args[0])
        rhs = _get_constant_scalar_value(value.args[1])
        if lhs is not None and rhs is not None:
            try:
                return lhs * rhs
            except (OverflowError, ValueError):
                return None

    return None


def _get_tensor_dtype(value) -> torch.dtype | None:
    """Return the dtype recorded for a tensor node, when available."""
    if not isinstance(value, torch.fx.Node):
        return None

    meta_value = value.meta.get("val")
    if isinstance(meta_value, torch.Tensor):
        return meta_value.dtype

    tensor_meta = value.meta.get("tensor_meta")
    dtype = getattr(tensor_meta, "dtype", None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _cast_finite_scalar_to_dtype(
    scalar: float, dtype: torch.dtype | None
) -> float | None:
    """Cast a finite scalar to dtype and return its finite value.

    This makes positivity proofs reflect the actual tensor dtype. In particular,
    a mathematically positive Python scalar that underflows to zero in float32
    must not be used to justify log(base).

    """
    if not math.isfinite(scalar):
        return None

    if dtype is None:
        return scalar

    try:
        cast_value = torch.tensor(scalar, dtype=dtype).item()
    except (RuntimeError, TypeError, OverflowError, ValueError):
        return None

    if isinstance(cast_value, bool) or not isinstance(cast_value, (int, float)):
        return None

    cast_value = float(cast_value)
    return cast_value if math.isfinite(cast_value) else None


def _is_positive_scalar(value, dtype: torch.dtype | None = None) -> bool:
    """Return True for finite compile-time scalar constants > 0 in dtype."""
    scalar = _get_constant_scalar_value(value)
    if scalar is None:
        return False
    cast_value = _cast_finite_scalar_to_dtype(scalar, dtype)
    return cast_value is not None and cast_value > 0.0


def _is_non_negative_scalar(value, dtype: torch.dtype | None = None) -> bool:
    """Return True for finite compile-time scalar constants >= 0 in dtype."""
    scalar = _get_constant_scalar_value(value)
    if scalar is None:
        return False
    cast_value = _cast_finite_scalar_to_dtype(scalar, dtype)
    return cast_value is not None and cast_value >= 0.0


def _is_positive_scaled_scalar(value, scale: float, dtype: torch.dtype | None) -> bool:
    """Return True when scale * value stays finite and > 0 in dtype."""
    scalar = _get_constant_scalar_value(value)
    if scalar is None:
        return False

    cast_scalar = _cast_finite_scalar_to_dtype(scalar, dtype)
    cast_scale = _cast_finite_scalar_to_dtype(scale, dtype)
    if (
        cast_scalar is None
        or cast_scale is None
        or cast_scalar <= 0.0
        or cast_scale <= 0.0
    ):
        return False

    cast_product = _cast_finite_scalar_to_dtype(cast_scalar * cast_scale, dtype)
    return cast_product is not None and cast_product > 0.0


def _get_lower_bound(node: torch.fx.Node):
    """Return the lower bound argument of a clamp-like node, if present."""
    if len(node.args) > 1:
        return node.args[1]
    return node.kwargs.get("min")


def _get_upper_bound(node: torch.fx.Node):
    """Return the upper bound argument of aten.clamp, if present."""
    if len(node.args) > 2:
        return node.args[2]
    return node.kwargs.get("max")


def _is_clamp_min_target(target) -> bool:
    return target == torch.ops.aten.clamp_min.default


def _is_clamp_target(target) -> bool:
    return target == torch.ops.aten.clamp.default


def _clamp_proves_non_negative(node: torch.fx.Node) -> bool:
    """Return True only when every effective clamp bound preserves >= 0."""
    target = _get_aten_target(node)
    dtype = _get_tensor_dtype(node.args[0]) if node.args else None
    lower = _get_lower_bound(node)

    if _is_clamp_min_target(target):
        return _is_non_negative_scalar(lower, dtype)

    if not _is_clamp_target(target) or not _is_non_negative_scalar(lower, dtype):
        return False

    upper = _get_upper_bound(node)
    # max=None means there is no upper bound. A dynamic/unresolved upper bound is
    # rejected because it could force the result negative.
    if upper is None:
        return True
    return _is_non_negative_scalar(upper, dtype)


def _clamp_proves_strictly_positive(node: torch.fx.Node) -> bool:
    """Return True only when every effective clamp bound preserves > 0."""
    target = _get_aten_target(node)
    dtype = _get_tensor_dtype(node.args[0]) if node.args else None
    lower = _get_lower_bound(node)

    if _is_clamp_min_target(target):
        return _is_positive_scalar(lower, dtype)

    if not _is_clamp_target(target) or not _is_positive_scalar(lower, dtype):
        return False

    upper = _get_upper_bound(node)
    if upper is None:
        return True
    return _is_positive_scalar(upper, dtype)


def _is_non_negative_tensor_node(node) -> bool:
    """Return True when graph structure proves a tensor is non-negative."""
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False

    target = _get_aten_target(node)

    if target == torch.ops.aten.abs.default:
        return True

    if target in (
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
    ):
        return True

    if _is_clamp_min_target(target) or _is_clamp_target(target):
        return _clamp_proves_non_negative(node)

    return False


def _get_numeric_add_alpha(node: torch.fx.Node) -> float | None:
    """Return aten.add.Tensor's finite numeric alpha, or None if unknown."""
    alpha = node.kwargs.get("alpha", 1)

    # Dynamic, non-numeric, bool, NaN and infinity cannot participate in a
    # compile-time positivity proof.
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        return None

    alpha = float(alpha)
    return alpha if math.isfinite(alpha) else None


def _is_non_negative_plus_positive_scalar(node: torch.fx.Node) -> bool:
    """Recognize add expressions that are provably strictly positive.

    aten.add.Tensor computes lhs + alpha * rhs. Supported proofs are:

    nonnegative_tensor + alpha * positive_scalar positive_scalar + alpha *
    nonnegative_tensor

    Scalar signs are checked *after conversion to the tensor dtype* so values
    that underflow to zero cannot incorrectly establish strict positivity.

    """
    if len(node.args) < 2:
        return False

    alpha = _get_numeric_add_alpha(node)
    if alpha is None:
        return False

    lhs = node.args[0]
    rhs = node.args[1]

    if isinstance(lhs, torch.fx.Node) and _is_non_negative_tensor_node(lhs):
        dtype = _get_tensor_dtype(lhs)
        if _is_positive_scaled_scalar(rhs, alpha, dtype):
            return True

    if isinstance(rhs, torch.fx.Node) and _is_non_negative_tensor_node(rhs):
        dtype = _get_tensor_dtype(rhs)
        cast_alpha = _cast_finite_scalar_to_dtype(alpha, dtype)
        if (
            cast_alpha is not None
            and cast_alpha >= 0.0
            and _is_positive_scalar(lhs, dtype)
        ):
            return True

    return False


POW_LOG_POSITIVE_LOWER_BOUND_META = "pow_log_positive_lower_bound"


def get_strictly_positive_lower_bound(  # noqa: C901
    node: torch.fx.Node,
) -> float | None:
    """Return a conservative source-dtype lower bound proving ``node > 0``.

    This mirrors the structural cases accepted by
    ``is_strictly_positive_tensor_node`` and intentionally does not use
    calibration/example values. The bound is carried to the generated LOG so
    that the INT lowering can re-check it once activation qparams are known.

    """
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return None

    target = _get_aten_target(node)

    if _is_clamp_min_target(target) or _is_clamp_target(target):
        if not node.args:
            return None

        dtype = _get_tensor_dtype(node.args[0])
        lower_scalar = _get_constant_scalar_value(_get_lower_bound(node))
        if lower_scalar is None:
            return None
        lower = _cast_finite_scalar_to_dtype(lower_scalar, dtype)
        if lower is None or lower <= 0.0:
            return None

        if _is_clamp_min_target(target):
            return lower

        upper_arg = _get_upper_bound(node)
        if upper_arg is None:
            return lower

        upper_scalar = _get_constant_scalar_value(upper_arg)
        if upper_scalar is None:
            return None
        upper = _cast_finite_scalar_to_dtype(upper_scalar, dtype)
        if upper is None or upper <= 0.0:
            return None

        # If min > max, torch.clamp returns max. Requiring both bounds to be
        # positive makes min(lower, upper) a valid conservative lower bound.
        return min(lower, upper)

    if target != torch.ops.aten.add.Tensor or len(node.args) < 2:
        return None

    alpha = _get_numeric_add_alpha(node)
    if alpha is None:
        return None

    lhs = node.args[0]
    rhs = node.args[1]

    # nonnegative_tensor + alpha * positive_scalar
    if isinstance(lhs, torch.fx.Node) and _is_non_negative_tensor_node(lhs):
        dtype = _get_tensor_dtype(lhs)
        scalar = _get_constant_scalar_value(rhs)
        if scalar is not None:
            cast_scalar = _cast_finite_scalar_to_dtype(scalar, dtype)
            cast_alpha = _cast_finite_scalar_to_dtype(alpha, dtype)
            if (
                cast_scalar is not None
                and cast_alpha is not None
                and cast_scalar > 0.0
                and cast_alpha > 0.0
            ):
                product = _cast_finite_scalar_to_dtype(cast_scalar * cast_alpha, dtype)
                if product is not None and product > 0.0:
                    return product

    # positive_scalar + alpha * nonnegative_tensor
    if isinstance(rhs, torch.fx.Node) and _is_non_negative_tensor_node(rhs):
        dtype = _get_tensor_dtype(rhs)
        cast_alpha = _cast_finite_scalar_to_dtype(alpha, dtype)
        scalar = _get_constant_scalar_value(lhs)
        if cast_alpha is not None and cast_alpha >= 0.0 and scalar is not None:
            positive = _cast_finite_scalar_to_dtype(scalar, dtype)
            if positive is not None and positive > 0.0:
                return positive

    return None


def is_strictly_positive_tensor_node(node: torch.fx.Node) -> bool:
    """Return True when graph structure proves a tensor is strictly positive.

    Do not infer positivity from calibration/example values. Those values do not
    constrain runtime inputs. The proof is intentionally conservative because it
    gates pow(x, y) -> exp(y * log(x)), which is invalid for non-positive bases.

    """
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False

    target = _get_aten_target(node)

    if _is_clamp_min_target(target) or _is_clamp_target(target):
        return _clamp_proves_strictly_positive(node)

    # nonnegative_tensor + positive_scalar > 0, including constant-only scalar
    # expressions introduced by TFA canonicalization.
    if target == torch.ops.aten.add.Tensor:
        return _is_non_negative_plus_positive_scalar(node)

    return False


def get_first_fake_tensor(node: torch.fx.Node) -> FakeTensor:
    """Returns a FakeTensor from the meta field of 'node'.

    If the node contains many fake tensors, return the first one.

    """
    if isinstance(
        node.meta["val"], (Sequence, torch.fx.immutable_collections.immutable_list)
    ):
        fake_tensor = node.meta["val"][0]
    else:
        fake_tensor = node.meta["val"]

    if not isinstance(fake_tensor, FakeTensor):
        raise TypeError(
            f'Expected a FakeTensor in meta["val"] of node {node}, but got '
            f"{type(fake_tensor).__name__}\n"
            f"{get_node_debug_info(node)}"
        )

    return fake_tensor


def get_node_arg(args: list | dict, key: int | str | type, default_value=None):
    """Help-function for getting a value from node.args/ kwargs, three cases:

    1. By position in node.args - Returns arg at given position or default_value if index is one out of bounds
    2. By key in node.kwargs - Returns kwarg with given key or default_value if it deos not exist
    3. By type in node.args - Returns first arg of args of given type. Useful for cases where arg postions may differ but types are unique.

    """
    if isinstance(key, int):
        if 0 <= key < len(args):
            return args[key]
        elif key == len(args):
            if default_value is not None:
                return default_value
            else:
                raise RuntimeError(f"No defult value given for index {key}")
        else:
            raise RuntimeError(
                f"Out of bounds index {key} for getting value in args (of size {len(args)})"
            )
    elif isinstance(key, str):
        return args.get(key, default_value)  # type: ignore[union-attr]
    elif isclass(key):
        for arg in args:
            if isinstance(arg, key):
                return arg
        if default_value is not None:
            return default_value
        else:
            raise RuntimeError(f"No arg of type {key}")
    else:
        raise RuntimeError("Invalid type")


def set_node_arg(node: torch.fx.Node, i: int | str, value):
    """Help-function for setting a value in node.args/ kwargs.

    If the index is one larger than the list size, the value is instead appended
    to the list.

    """
    if isinstance(i, int):
        if 0 <= i < len(node.args):
            args = list(node.args)
            args[i] = value
            node.args = tuple(args)
            return
        elif i == len(node.args):
            node.args = node.args + (value,)
        else:
            raise RuntimeError(
                f"Out of bounds index {i} for setting value in {node} args (of size {len(node.args)})"
            )
    elif isinstance(i, str):
        kwargs = dict(node.kwargs)
        kwargs[i] = value
        node.kwargs = kwargs
    else:
        raise RuntimeError("Invalid type")


def to_2tuple(value):
    """Normalizes scalars, and 1-element sequences to a tuple of length 2."""
    if isinstance(value, int):
        return (value, value)
    if len(value) == 1:
        return (value[0], value[0])
    return tuple(value)


def permute_fake_tensor_metadata(
    fake_tensor: FakeTensor, permute_dims: tuple[int, ...]
) -> FakeTensor:
    permuted_shape = tuple(fake_tensor.shape[dim] for dim in permute_dims)
    meta_tensor = torch.empty(
        permuted_shape,
        dtype=fake_tensor.dtype,
        device="meta",
        requires_grad=fake_tensor.requires_grad,
    )
    return FakeTensor(fake_tensor.fake_mode, meta_tensor, fake_tensor.fake_device)
