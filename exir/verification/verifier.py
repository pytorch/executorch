# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import operator
from typing import Any, List, Optional, Tuple, Type

import torch
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.verification.arg_validator import (
    EdgeOpArgValidator,
    RunHigherOrderOperatorError,
)

from torch._export.verifier import (
    _check_has_fake_tensor,
    ATenDialectVerifier,
    SpecViolationError,
    Verifier,
)
from torch._ops import OpOverload
from torch._subclasses import FakeTensor
from torch.fx import GraphModule


ALLOWED_META_KEYS = {"spec", "stack_trace"}


def _check_tensors_are_contiguous(gm: GraphModule) -> None:
    # Tensors be of contiguous format
    for name, param in itertools.chain(gm.named_parameters(), gm.named_buffers()):
        if isinstance(param, torch.Tensor):
            if not param.is_contiguous():
                raise SpecViolationError(
                    f"Tensors in Aten dialect must be contiguous, {name} is not contiguous"
                )


class EXIRATenDialectVerifier(ATenDialectVerifier):
    def allowed_getattr_types(self) -> Tuple[Type[Any], ...]:
        return (torch.fx.GraphModule, LoweredBackendModule, torch.Tensor)


def _get_inputs(graph_module: GraphModule) -> List[Optional[FakeTensor]]:
    def extract_input(node: torch.fx.Node) -> Optional[FakeTensor]:
        if "val" in node.meta:
            return node.meta["val"]

        if len(node.users) == 0:
            return None

        # TODO(ycao): `val` should always exist after we enable shape environment
        # serialization and deserialization.
        raise ExportError(
            ExportErrorType.VIOLATION_OF_SPEC,
            f"Cannot construct an input for graph module: {graph_module}.",
        )

    return [
        extract_input(node)
        for node in graph_module.graph.nodes
        if node.op == "placeholder"
    ]


def _check_tensor_args_matching_op_allowed_dtype(gm: GraphModule) -> None:
    validator = EdgeOpArgValidator(gm)
    inputs = _get_inputs(gm)
    try:
        validator.run(*inputs)
    except RunHigherOrderOperatorError:
        # NB: ignore higher order operator in the graph.
        # If we lower a graph module to delegate and then compose it with some other graph module, retrace it,
        # if we also turn on edge ops and validator (_check_ir_validity=True), we will run
        # into RunHigherOrderOperatorError. The only thing we can do right now is to ignore this error, since
        # by definition it's still a valid Edge dialect. This is not ideal because it ignores possible invalidity
        # later in the graph.
        return

    if validator.violating_ops:
        raise SpecViolationError(
            f"These operators are taking Tensor inputs with mismatched dtypes: {validator.violating_ops}"
        )


class EXIREdgeDialectVerifier(Verifier):
    def __init__(self, check_edge_ops: bool = True) -> None:
        self.check_edge_ops = check_edge_ops

        if self.check_edge_ops:
            self.check_valid_op = self.check_valid_edge_op
        else:
            self.check_valid_op = self.check_valid_aten_op

    def allowed_getattr_types(self) -> Tuple[Type[Any], ...]:
        return (torch.fx.GraphModule, LoweredBackendModule, torch.Tensor)

    def check_valid_edge_op(self, op):
        if op in [operator.getitem]:
            return

        if isinstance(op, OpOverload) and not isinstance(op, EdgeOpOverload):
            raise SpecViolationError(
                "Operator {}.{} is not an Edge operator.".format(
                    op.__module__, op.__name__
                )
            )

    def check_valid_aten_op(self, op) -> None:
        super().check_valid_op(op)

        if isinstance(op, OpOverload):
            if (
                torch.Tag.core not in op.tags  # type: ignore[attr-defined]
                and torch.Tag.view_copy not in op.tags  # type: ignore[attr-defined]
            ):
                # NOTE(qihan): whether view_copy operators are marked as canonical is still under
                #            discussion.
                raise SpecViolationError(
                    "Operator {}.{} is not Aten Canonical.".format(
                        op.__module__, op.__name__
                    )
                )

    def check_additional(self, gm: GraphModule) -> None:
        if self.check_edge_ops:
            _check_tensors_are_contiguous(gm)
            _check_tensor_args_matching_op_allowed_dtype(gm)

        # Additionally, edge dialect's operator must have same input dtype
        for n in gm.graph.nodes:
            if n.op == "call_function" and isinstance(n.target, OpOverload):
                _check_has_fake_tensor(n)
                dtypes = set()
                for arg in n.args:
                    if isinstance(arg, torch.Tensor):
                        dtypes.add(arg.dtype)
                    if isinstance(arg, torch.fx.Node):
                        if arg.meta.get("val", None) is None:
                            raise SpecViolationError(
                                f"No metadata 'val' for node {arg}"
                            )
                        dtypes.add(arg.meta["val"].dtype)
                if len(dtypes) > 1:
                    raise SpecViolationError(
                        "Operators of Edge dialect in should work on tensors of same dtype"
                    )
