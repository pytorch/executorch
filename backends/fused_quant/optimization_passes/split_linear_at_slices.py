# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import NamedTuple, Optional

import torch
from executorch.backends.fused_quant.graph_utils import (
    add_constant,
    get_constant,
    get_fqn,
    get_input_kind,
)
from executorch.backends.transforms.permute_pass_utils import get_arg, set_arg
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import constant_prop_pass
from torch import fx
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind

logger: logging.Logger = logging.getLogger(__name__)


def _is_output_dim_slice(
    slice_node: fx.Node, linear_out_features: int
) -> Optional[tuple[int, int, int]]:
    """Check if a slice_copy is on the output (last) dimension of a linear.

    Returns (start, end, step) if it's an output-dim slice, None otherwise.
    """
    dim = get_arg(slice_node, "dim", int)
    start = get_arg(slice_node, "start")
    end = get_arg(slice_node, "end")
    step = get_arg(slice_node, "step", int)
    if start is None:
        start = 0
    if end is None:
        end = linear_out_features

    assert isinstance(start, int) and isinstance(end, int), (
        f"Symbolic slice bounds not supported: start={start!r}, end={end!r}"
    )

    input_node = slice_node.args[0]
    assert isinstance(input_node, fx.Node)
    input_shape = input_node.meta["val"].shape

    ndim = len(input_shape)
    if dim < 0:
        dim = ndim + dim

    if dim != ndim - 1:
        return None

    if start < 0:
        start = max(0, start + linear_out_features)
    else:
        start = min(start, linear_out_features)

    if end < 0:
        end = max(0, end + linear_out_features)
    else:
        end = min(end, linear_out_features)

    return (start, end, step)


# Tensor args of fused_quant.linear that index the OUTPUT channels on dim 0, so
# each must be sliced to follow an output-dim split. The weight itself is handled
# separately (it is mandatory and always sliced). Note this includes the *bias*
# qparams (bias_scale/bias_zero_point): if a bias is per-output-channel quantized,
# its scale/zp ride on dim 0 just like the weight's and must be sliced too --
# slicing the bias tensor while leaving full-rank bias qparams would be a silent
# miscompile. Per-tensor / broadcast instances of any of these are reused
# unchanged; see _plan_output_qparam.
#
# The output qparams (out_scale/out_zero_point) are deliberately NOT here: they
# index the output tensor's *last* dim, not dim 0, so they need a different
# slicing axis. They are per-tensor in the current flow; _try_split bails (rather
# than silently mis-slice) if one is ever per-channel -- see _OUTPUT_QPARAMS.
_PER_OUTPUT_DIM0_ARGS: tuple[str, ...] = (
    "bias",
    "weight_scale",
    "weight_zero_point",
    "bias_scale",
    "bias_zero_point",
)

# Output requantization qparams. These index the output's last dim, not dim 0, so
# the dim-0 slicing above does not apply; per-channel output requant is not
# handled by the split, so _try_split bails when one of these is non-scalar.
_OUTPUT_QPARAMS: tuple[str, ...] = ("out_scale", "out_zero_point")


# A planned per-output tensor arg (bias / weight scale / zero_point / bias
# scale / zero_point): (do_slice, tensor, kind).
#   do_slice False -> reuse the original node unchanged (it broadcasts over every
#                     output slice); tensor/kind are None.
#   do_slice True  -> slice `tensor` along dim 0 per output slice, re-adding the
#                     result as a constant of input kind `kind`.
_QParamPlan = tuple[bool, Optional[torch.Tensor], Optional[InputKind]]


def _plan_output_qparam(
    ep: ExportedProgram,
    qparam_node: Optional[fx.Node],
    out_features: int,
) -> Optional[_QParamPlan]:
    """Decide -- WITHOUT mutating the graph -- how a per-output tensor arg (bias /
    weight scale / weight zero_point) is carried onto each split linear.

    Returns:
      - ``(False, None, None)`` when there is nothing to slice: the arg is absent,
        a scalar, or broadcasts over the output dim (its leading dim is not the
        output channels). Every split linear then reuses the original node, which
        broadcasts correctly over any output slice.
      - ``(True, tensor, kind)`` when the arg varies along the output dim and its
        constant value is available, so it can be sliced per output slice.
      - ``None`` to signal the whole split must be ABORTED: the arg varies along
        the output dim but its constant value is unavailable (e.g. a per-channel
        scale still behind an unfolded ``view_copy``). Slicing it correctly is
        impossible, and reusing the full-rank value on a smaller linear would be a
        silent miscompile -- so the caller must bail before touching the graph
        rather than proceed.

    Granularity is read from ``meta["val"]`` (always present), so the slice-vs-
    reuse decision never depends on the constant value being resolvable; the value
    is only needed when we actually slice.
    """
    if qparam_node is None:
        return (False, None, None)
    val = qparam_node.meta["val"]
    if val.ndim == 0 or val.shape[0] != out_features:
        # Scalar / broadcast over the output dim -> reuse unchanged.
        return (False, None, None)
    tensor = get_constant(ep, qparam_node)
    if tensor is None:
        # Per-output but the constant value isn't available -> cannot slice.
        return None
    kind = get_input_kind(ep, qparam_node)
    assert kind is not None
    return (True, tensor, kind)


def _apply_output_qparam(
    ep: ExportedProgram,
    qparam_node: Optional[fx.Node],
    plan: _QParamPlan,
    slice_range: tuple[int, int, int],
    linear_node: fx.Node,
) -> Optional[fx.Node]:
    """Apply a plan from ``_plan_output_qparam`` to one output slice.

    Returns the new sliced constant node, or ``None`` to reuse the original node
    (the plan determined the arg broadcasts / is absent).
    """
    do_slice, tensor, kind = plan
    if not do_slice:
        return None
    assert qparam_node is not None and tensor is not None and kind is not None
    start, end, step = slice_range
    fqn = get_fqn(ep, qparam_node)
    assert fqn is not None
    return add_constant(
        ep,
        f"{fqn.replace('.', '_')}_slice_{start}_{end}_{step}",
        tensor[start:end:step],
        linear_node,
        kind=kind,
    )


class _SplitPlan(NamedTuple):
    """A fully validated split -- everything needed to apply it, computed without
    mutating the graph (see ``_plan_split`` / ``_apply_split``)."""

    users: list[fx.Node]
    slice_ranges: list[tuple[int, int, int]]
    weight_tensor: torch.Tensor
    weight_kind: InputKind
    # Per-output dim-0 arg nodes and their plans, keyed by _PER_OUTPUT_DIM0_ARGS.
    arg_nodes: dict[str, Optional[fx.Node]]
    arg_plans: dict[str, _QParamPlan]


class SplitLinearAtSlices(ExportedProgramPassBase):
    """Split a fused_quant.linear into smaller linears when its output is
    sliced on the output dimension.

    Matches: fused_quant.linear → (slice_copy[dim=-1], slice_copy[dim=-1], ...)

    When all users of the linear are output-dimension slices and the total
    sliced output features does not exceed the original (no extra compute),
    the linear is split into one smaller linear per slice. Each gets a
    compile-time sliced copy of the weight (and bias/weight_scale/weight_zp
    if present).

    The split is fully validated before the graph is modified: weight, bias, and
    any per-output-channel weight scale / zero_point must all be resolvable to
    constants that can be sliced. If any of them varies along the output dim but
    can't be resolved, ``_try_split`` bails without touching the graph, so we
    never leave a sub-linear with a sliced weight but a mismatched full-rank
    qparam. (The ``ConstantFold`` pass run before the optimization passes bakes
    the per-channel scale/zp ``view_copy`` fusion inserts, so these resolve in the
    normal pipeline.)
    """

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        graph = exported_program.graph_module.graph
        modified = False

        for linear_node in list(
            graph.find_nodes(
                op="call_function",
                target=exir_ops.edge.fused_quant.linear.default,
            )
        ):
            if self._try_split(exported_program, linear_node):
                modified = True

        if modified:
            exported_program = constant_prop_pass(exported_program)

        return ExportedProgramPassResult(exported_program, modified)

    def _try_split(self, ep: ExportedProgram, linear_node: fx.Node) -> bool:
        plan = self._plan_split(ep, linear_node)
        if plan is None:
            return False
        self._apply_split(ep, linear_node, plan)
        logger.info(
            "Split linear %s into %d smaller linears at output slices",
            linear_node.name,
            len(plan.users),
        )
        return True

    def _plan_split(
        self, ep: ExportedProgram, linear_node: fx.Node
    ) -> Optional[_SplitPlan]:
        """Validate the split and compute everything needed to apply it, WITHOUT
        mutating the graph. Returns ``None`` (the linear can't be safely split) if
        any user is not an output-dim slice, the weight isn't a resolvable
        constant, the slices over-cover the output, the output qparams are
        per-channel, or any per-output dim-0 qparam varies along the output dim but
        can't be sliced -- so the caller bails before touching the graph rather
        than half-rewrite it."""
        users = list(linear_node.users.keys())
        if not all(u.target == exir_ops.edge.aten.slice_copy.Tensor for u in users):
            return None

        weight_node = get_arg(linear_node, "weight", fx.Node)
        weight_tensor = get_constant(ep, weight_node)
        if weight_tensor is None:
            return None
        weight_kind = get_input_kind(ep, weight_node)
        assert weight_kind is not None
        out_features = weight_tensor.shape[0]

        # Every user must be an output-dim slice, together covering no more than
        # out_features (no extra compute).
        slice_ranges: list[tuple[int, int, int]] = []
        for user in users:
            r = _is_output_dim_slice(user, out_features)
            if r is None:
                return None
            slice_ranges.append(r)
        if sum(len(range(s, e, st)) for s, e, st in slice_ranges) > out_features:
            return None

        # Per-channel OUTPUT requant (out_scale/out_zero_point index the output's
        # last dim -- a different axis than the dim-0 weight/bias qparams) isn't
        # handled; bail rather than reuse a full-rank value on a smaller linear.
        for name in _OUTPUT_QPARAMS:
            node = get_arg(linear_node, name, Optional[fx.Node])
            if node is not None and node.meta["val"].numel() != 1:
                return None

        # Plan every per-output dim-0 tensor arg. A None plan means the arg varies
        # along the output dim but can't be sliced -> the whole split aborts.
        arg_nodes: dict[str, Optional[fx.Node]] = {
            name: get_arg(linear_node, name, Optional[fx.Node])
            for name in _PER_OUTPUT_DIM0_ARGS
        }
        arg_plans: dict[str, _QParamPlan] = {}
        for name, node in arg_nodes.items():
            arg_plan = _plan_output_qparam(ep, node, out_features)
            if arg_plan is None:
                return None
            arg_plans[name] = arg_plan

        return _SplitPlan(
            users=users,
            slice_ranges=slice_ranges,
            weight_tensor=weight_tensor,
            weight_kind=weight_kind,
            arg_nodes=arg_nodes,
            arg_plans=arg_plans,
        )

    def _apply_split(
        self, ep: ExportedProgram, linear_node: fx.Node, plan: _SplitPlan
    ) -> None:
        """Apply a validated ``_SplitPlan``: replace each output slice with its own
        smaller linear carrying sliced weight / bias / qparams."""
        graph = ep.graph_module.graph
        weight_node = get_arg(linear_node, "weight", fx.Node)
        weight_fqn = get_fqn(ep, weight_node)
        assert weight_fqn is not None
        for user, (start, end, step) in zip(plan.users, plan.slice_ranges):
            new_weight_node = add_constant(
                ep,
                f"{weight_fqn.replace('.', '_')}_slice_{start}_{end}_{step}",
                plan.weight_tensor[start:end:step],
                linear_node,
                kind=plan.weight_kind,
            )
            new_arg_nodes = {
                name: _apply_output_qparam(
                    ep,
                    plan.arg_nodes[name],
                    plan.arg_plans[name],
                    (start, end, step),
                    linear_node,
                )
                for name in _PER_OUTPUT_DIM0_ARGS
            }

            with graph.inserting_before(user):
                new_linear = graph.call_function(
                    exir_ops.edge.fused_quant.linear.default,
                    args=linear_node.args,
                )
                new_linear.meta["val"] = user.meta["val"]

            set_arg(new_linear, "weight", new_weight_node)
            for name, new_node in new_arg_nodes.items():
                if new_node is not None:
                    set_arg(new_linear, name, new_node)

            user.replace_all_uses_with(new_linear)
