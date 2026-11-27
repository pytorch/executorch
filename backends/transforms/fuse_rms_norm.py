# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse FP32 RMSNorm decompositions with optional backend-controlled cast folding."""

import math
from dataclasses import dataclass
from functools import partial
from typing import Mapping, Optional, TypeGuard

import torch
from executorch.backends.transforms.utils import create_constant_placeholder
from executorch.exir._program_utils import _get_updated_graph_signature
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes.constant_prop_pass import get_constant_placeholder_dict
from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.fx import Node


def _match_target(node: object, op: torch._ops.OpOverload) -> bool:
    if not isinstance(node, Node) or node.op != "call_function":
        return False
    target = node.target
    if isinstance(target, EdgeOpOverload):
        target = target._op
    return target == op


def _match_op(
    node: object,
    op: torch._ops.OpOverload,
    nargs: tuple[int, int],
    allowed_kwargs: tuple[str, ...] = (),
    *,
    single_user: bool = True,
) -> TypeGuard[Node]:
    return (
        isinstance(node, Node)
        and _match_target(node, op)
        and nargs[0] <= len(node.args) <= nargs[1]
        and not set(node.kwargs).difference(allowed_kwargs)
        and (not single_user or len(node.users) == 1)
    )


def _scalar_constant(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, Node) or "lifted_tensor_constant" not in value.name:
        return None
    tensor = value.meta.get("val")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 0:
        return None
    try:
        return float(tensor.item())
    except (RuntimeError, TypeError, ValueError):
        return None


@dataclass
class _RMSNormMatch:
    body: list[Node]
    input_node: Node
    eps: float
    weight_node: Optional[Node] = None
    weight_cast_dtype: Optional[torch.dtype] = None


def _is_inverse_root(node: object) -> TypeGuard[Node]:
    return _match_op(node, torch.ops.aten.rsqrt.default, (1, 1)) or (
        _match_op(node, torch.ops.aten.pow.Tensor_Scalar, (2, 2))
        and node.args[1] == -0.5
    )


def _match_rms_norm_core(head: Node) -> Optional[_RMSNormMatch]:  # noqa: C901
    if not _match_op(head, torch.ops.aten.mul.Tensor, (2, 2), single_user=False):
        return None

    for inverse_idx in (0, 1):
        inverse = head.args[inverse_idx]
        if not _is_inverse_root(inverse):
            continue
        add = inverse.args[0]
        if not _match_op(add, torch.ops.aten.add.Tensor, (2, 2), ("alpha",)):
            continue
        for mean_idx in (0, 1):
            mean = add.args[mean_idx]
            if not _match_op(
                mean, torch.ops.aten.mean.dim, (2, 3), ("dtype", "keepdim")
            ):
                continue
            square = mean.args[0]
            if not _match_op(square, torch.ops.aten.pow.Tensor_Scalar, (2, 2)):
                continue
            input_node = square.args[0]
            if (
                not isinstance(input_node, Node)
                or input_node is not head.args[1 - inverse_idx]
                or square.args[1] != 2
            ):
                continue
            input_meta = input_node.meta.get("val")
            # Match FP32 arithmetic first, independently of the cast policy.
            if (
                not isinstance(input_meta, torch.Tensor)
                or input_meta.ndim == 0
                or not isinstance(input_meta.shape[-1], int)
            ):
                continue
            body = [square, mean, add, inverse]
            if any(
                not isinstance(n.meta.get("val"), torch.Tensor)
                or n.meta["val"].dtype != torch.float32
                for n in (input_node, head, *body)
            ):
                continue
            dims = mean.args[1]
            keepdim = (
                mean.args[2]
                if len(mean.args) == 3
                else mean.kwargs.get("keepdim", False)
            )
            if (
                not isinstance(dims, (list, tuple))
                or len(dims) != 1
                or dims[0] not in (-1, input_meta.ndim - 1)
                or keepdim is not True
                or mean.kwargs.get("dtype") not in (None, torch.float32)
                or add.kwargs.get("alpha", 1) != 1
            ):
                continue
            eps_value = _scalar_constant(add.args[1 - mean_idx])
            if eps_value is None or not math.isfinite(eps_value) or eps_value < 0:
                continue
            return _RMSNormMatch(
                body=body,
                input_node=input_node,
                eps=eps_value,
            )
    return None


def _match_weighted_rms_norm(head: Node) -> Optional[_RMSNormMatch]:
    if not _match_op(head, torch.ops.aten.mul.Tensor, (2, 2), single_user=False):
        return None

    for weight_idx, norm_idx in ((0, 1), (1, 0)):
        weight, norm = head.args[weight_idx], head.args[norm_idx]
        if not isinstance(weight, Node) or not isinstance(norm, Node):
            continue
        if len(norm.users) != 1:
            continue
        core = _match_rms_norm_core(norm)
        if core is None:
            continue
        weight_meta = weight.meta.get("val")
        output_meta = head.meta.get("val")
        input_meta = core.input_node.meta["val"]
        if (
            not isinstance(weight_meta, torch.Tensor)
            or weight_meta.ndim != 1
            or weight_meta.dtype != torch.float32
            or not isinstance(output_meta, torch.Tensor)
            or output_meta.dtype != torch.float32
            or not isinstance(weight_meta.shape[0], int)
            or not isinstance(input_meta.shape[-1], int)
            or weight_meta.shape[0] != input_meta.shape[-1]
        ):
            continue
        return _RMSNormMatch(
            body=[*core.body, norm],
            input_node=core.input_node,
            weight_node=weight,
            eps=core.eps,
        )
    return None


def _cast_source(node: Node) -> Optional[Node]:
    if not node.args or not isinstance(node.args[0], Node):
        return None
    source = node.args[0]
    before, after = source.meta.get("val"), node.meta.get("val")
    if not isinstance(before, torch.Tensor) or not isinstance(after, torch.Tensor):
        return None
    if before.device != after.device or before.layout != after.layout:
        return None
    if _match_target(node, torch.ops.aten._to_copy.default):
        # type_as exports explicit device/layout even when they do not change.
        if (
            len(node.args) != 1
            or set(node.kwargs) - {"dtype", "device", "layout", "non_blocking"}
            or node.kwargs.get("dtype") != after.dtype
            or node.kwargs.get("device") not in (None, before.device)
            or node.kwargs.get("layout") not in (None, before.layout)
            or node.kwargs.get("non_blocking", False)
        ):
            return None
    elif _match_target(node, torch.ops.aten.to.dtype):
        if len(node.args) != 2 or node.kwargs or node.args[1] != after.dtype:
            return None
    elif _match_target(node, torch.ops.aten.type_as.default):
        if len(node.args) != 2 or node.kwargs:
            return None
    else:
        return None
    return source


def _match_casted_rms_norm(
    head: Node, *, allow_lossy_weight_casts: bool = False
) -> Optional[_RMSNormMatch]:
    norm = _cast_source(head)
    if (
        norm is None
        or not _match_target(norm, torch.ops.aten.rms_norm.default)
        or len(norm.args) != 4
        or norm.kwargs
        or set(norm.users) != {head}
        or norm.meta["val"].dtype != torch.float32
    ):
        return None
    input_cast, normalized_shape, weight_cast, eps = norm.args
    if not isinstance(input_cast, Node):
        return None
    input_node = _cast_source(input_cast)
    if input_node is None or set(input_cast.users) != {norm}:
        return None
    input_meta = input_node.meta["val"]
    dtype = input_meta.dtype
    if (
        dtype not in (torch.float16, torch.bfloat16)
        or input_cast.meta["val"].dtype != torch.float32
        or head.meta["val"].dtype != dtype
        or input_meta.ndim == 0
        or not isinstance(input_meta.shape[-1], int)
        or not isinstance(normalized_shape, (list, tuple))
        or list(normalized_shape) != [input_meta.shape[-1]]
        or not isinstance(eps, (int, float))
    ):
        return None
    weight_node = None
    weight_cast_dtype = None
    casts = [input_cast]
    if weight_cast is not None:
        if not isinstance(weight_cast, Node) or set(weight_cast.users) != {norm}:
            return None
        weight_meta = weight_cast.meta.get("val")
        if (
            not isinstance(weight_meta, torch.Tensor)
            or weight_meta.dtype != torch.float32
            or weight_meta.ndim != 1
            or not isinstance(weight_meta.shape[0], int)
            or weight_meta.shape[0] != input_meta.shape[-1]
        ):
            return None
        weight_node = _cast_source(weight_cast)
        if (
            weight_node is not None
            and weight_node.meta["val"].dtype == dtype
            and weight_node.meta["val"].ndim == 1
            and isinstance(weight_node.meta["val"].shape[0], int)
            and weight_node.meta["val"].shape[0] == input_meta.shape[-1]
        ):
            if weight_cast is not input_cast:
                casts.append(weight_cast)
        elif (
            allow_lossy_weight_casts
            and weight_meta.device == input_meta.device
            and weight_meta.layout == input_meta.layout
        ):
            # Round the complete FP32 scale, not the inputs to its expression.
            weight_node = weight_cast
            weight_cast_dtype = dtype
        else:
            return None
    return _RMSNormMatch(
        body=[*casts, norm],
        input_node=input_node,
        weight_node=weight_node,
        weight_cast_dtype=weight_cast_dtype,
        eps=eps,
    )


def _frozen_fp32_weight(
    node: object, frozen: Mapping[Node, torch.Tensor]
) -> Optional[torch.Tensor]:
    if not isinstance(node, Node):
        return None
    meta = node.meta.get("val")
    if not isinstance(meta, torch.Tensor) or meta.dtype != torch.float32:
        return None
    source = node if node in frozen else _cast_source(node)
    if source is None:
        return None
    value = frozen.get(source)
    if (
        not isinstance(value, torch.Tensor)
        or isinstance(value, FakeTensor)
        or value.device.type == "meta"
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or value.ndim != 1
        or value.device != meta.device
        or value.layout != meta.layout
        or tuple(value.shape) != tuple(meta.shape)
        or (source is node and value.dtype != torch.float32)
    ):
        return None
    return value.float()


def _frozen_rmsnorm_weight(
    node: Node, frozen: Mapping[Node, torch.Tensor]
) -> Optional[torch.Tensor]:
    """Evaluate only a frozen FP32 weight, optionally followed by FP32 add-one."""
    value = _frozen_fp32_weight(node, frozen)
    if value is not None:
        return value
    if (
        not any(
            _match_target(node, target)
            for target in (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
        )
        or len(node.args) != 2
        or set(node.kwargs) - {"alpha"}
        or node.kwargs.get("alpha", 1) != 1
        or node.meta["val"].dtype != torch.float32
    ):
        return None
    for index in (0, 1):
        value = _frozen_fp32_weight(node.args[index], frozen)
        if value is None:
            continue
        one = node.args[1 - index]
        if isinstance(one, Node):
            one = frozen.get(one)
            if (
                not isinstance(one, torch.Tensor)
                or isinstance(one, FakeTensor)
                or one.device.type == "meta"
                or one.numel() != 1
            ):
                continue
            one = one.item()
        if isinstance(one, (int, float)) and one == 1:
            # Round only after the addition, never before it.
            return value + 1.0
    return None


def _prepare_frozen_weights(
    exported_program: ExportedProgram, norm_weight_casts: list[tuple[Node, Node]]
) -> None:
    """Materialize supported weights only for casts introduced by this fusion."""
    graph = exported_program.graph
    # Raw exports can still mutate buffers without listing signature mutations.
    # Do not infer frozen storage until all in-place operators are gone.
    for node in graph.nodes:
        target = node.target
        if isinstance(target, EdgeOpOverload):
            target = target._op
        if isinstance(target, torch._ops.OpOverload) and target._schema.is_mutable:
            return
    frozen = get_constant_placeholder_dict(exported_program)
    # Include FX's reserved names, storage keys, and specs, not just live nodes.
    names = set(graph._graph_namespace._used_names)
    names.update(exported_program.state_dict)
    names.update(exported_program.constants)
    for node in graph.nodes:
        names.add(node.meta.get("requested_name"))
        if isinstance(node.target, str):
            names.add(node.target)
    for spec in (
        *exported_program.graph_signature.input_specs,
        *exported_program.graph_signature.output_specs,
    ):
        names.add(spec.target)
        names.add(getattr(spec.arg, "name", None))
    # user_inputs contains literal values for ConstantArguments, not their names.
    user_input_names = {
        spec.arg.name
        for spec in exported_program.graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    }
    first_input = next(
        (
            node
            for node in graph.nodes
            if node.op != "placeholder" or node.name in user_input_names
        ),
        None,
    )
    for norm, weight_cast in norm_weight_casts:
        source = _cast_source(weight_cast)
        if source is None or weight_cast.meta["val"].dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            continue
        with unset_fake_temporarily(), torch.no_grad():
            value = _frozen_rmsnorm_weight(source, frozen)
            if value is None:
                continue
            data = value.to(weight_cast.meta["val"].dtype).detach().clone()
        index = 0
        name = "_rmsnorm_weight_0"
        while name in names:
            index += 1
            name = f"_rmsnorm_weight_{index}"
        names.add(name)
        with graph.inserting_before(first_input):
            constant = create_constant_placeholder(
                exported_program, graph, name, InputKind.CONSTANT_TENSOR, data
            )
        norm.replace_input_with(weight_cast, constant)


class FuseRMSNormPass(ExportedProgramPassBase):
    """Fuse weighted and scale-free FP32 RMSNorm decompositions into aten.rms_norm.

    Supports rsqrt and pow(-0.5) inverse roots on the last dimension. Leading
    dimensions may be dynamic; the normalized width must be static. Boundary
    casts are retained by default, preserving the FP32 computation and output
    dtype even for low-precision model inputs.

    Set fold_dtype_casts=True only for backends whose RMSNorm kernel internally
    accumulates in FP32 for FP16/BF16 inputs. This also opts into the kernel's
    low-precision scaling/rounding behavior, which need not be bitwise identical
    to the decomposed norm. Only private, matching dtype boundaries are folded.
    Backends must preserve aten.rms_norm from decomposition when lowering.

    With fold_dtype_casts=True, allow_lossy_weight_casts=True additionally permits
    rounding an FP32 effective weight (such as weight.float() + 1) to the input
    dtype before scaling. Supported frozen weights are prepared as derived
    constants without changing the original storage. FP32 expressions are
    evaluated before rounding; runtime or unsupported weight expressions retain
    their computation followed by the cast. Graphs with mutable operators skip
    frozen-weight preparation.

    Operates on the full ExportedProgram to access lifted weights and maintain
    input/output signatures. Call exported_program.module() after transformation
    when an executable GraphModule is needed.
    """

    def __init__(
        self,
        *,
        fold_dtype_casts: bool = False,
        allow_lossy_weight_casts: bool = False,
    ) -> None:
        super().__init__()
        if allow_lossy_weight_casts and not fold_dtype_casts:
            raise ValueError("allow_lossy_weight_casts requires fold_dtype_casts=True")
        self.fold_dtype_casts = fold_dtype_casts
        self.allow_lossy_weight_casts = allow_lossy_weight_casts

    def call(self, exported_program: ExportedProgram) -> ExportedProgramPassResult:
        """Fuse norms, prepare supported frozen weights, and update the program."""
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        norm_weight_casts: list[tuple[Node, Node]] = []
        matchers = [_match_weighted_rms_norm, _match_rms_norm_core]
        if self.fold_dtype_casts:
            matchers.append(
                partial(
                    _match_casted_rms_norm,
                    allow_lossy_weight_casts=self.allow_lossy_weight_casts,
                )
            )
        for match_pattern in matchers:
            for node in list(graph.nodes):
                match = match_pattern(node)
                if match is None:
                    continue

                input_meta = match.input_node.meta["val"]
                with graph.inserting_before(node):
                    weight_node = match.weight_node
                    if match.weight_cast_dtype is not None:
                        assert weight_node is not None
                        target = (
                            exir_ops.edge.aten._to_copy.default
                            if isinstance(node.target, EdgeOpOverload)
                            else torch.ops.aten._to_copy.default
                        )
                        source_weight = weight_node
                        weight_node = graph.call_function(
                            target,
                            args=(source_weight,),
                            kwargs={"dtype": match.weight_cast_dtype},
                        )
                        weight_node.meta = source_weight.meta.copy()
                        weight_node.meta["val"] = source_weight.meta["val"].to(
                            match.weight_cast_dtype
                        )
                        weight_node.meta.pop("spec", None)
                        weight_node.meta.pop("tensor_meta", None)
                    rms_norm_node = graph.call_function(
                        torch.ops.aten.rms_norm.default,
                        args=(
                            match.input_node,
                            [input_meta.shape[-1]],
                            weight_node,
                            match.eps,
                        ),
                    )
                    rms_norm_node.meta = node.meta.copy()
                    if match.weight_cast_dtype is not None:
                        assert weight_node is not None
                        norm_weight_casts.append((rms_norm_node, weight_node))

                node.replace_all_uses_with(rms_norm_node)
                graph.erase_node(node)
                for body_node in reversed(match.body):
                    if not body_node.users:
                        graph.erase_node(body_node)
                modified = True

        if modified:
            if norm_weight_casts:
                _prepare_frozen_weights(exported_program, norm_weight_casts)
            # Keep original placeholders/storage even when their expression is dead.
            graph.eliminate_dead_code()
            exported_program._graph_signature = _get_updated_graph_signature(
                exported_program.graph_signature, graph_module
            )
            graph.lint()
            graph_module.recompile()

        return ExportedProgramPassResult(exported_program, modified)
