from typing import List, Optional

import torch
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.verification.arg_validator import (
    EdgeOpArgValidator,
    RunHigherOrderOperatorError,
)

from torch._export.verifier import (
    _check_has_fake_tensor,
    _check_tensors_are_contiguous,
    ATenDialectVerifier,
    SpecViolationError,
    Verifier,
)
from torch._ops import OpOverload
from torch._subclasses import FakeTensor
from torch.fx import GraphModule


ALLOWED_META_KEYS = {"spec", "stack_trace"}
VALID_BUILTIN_FUNCS = [
    executorch_call_delegate,
]


class EXIRATenDialectVerifier(ATenDialectVerifier):
    def valid_builtin_funcs(self):
        builtin_funcs = super().valid_builtin_funcs()
        builtin_funcs.extend(VALID_BUILTIN_FUNCS)
        return builtin_funcs

    # TODO(angelayi): Delete this function when we migrate all tests to
    # pt2_mode=True because right now old tracer does not add ["val"] metadata
    def check_valid(self, gm: GraphModule) -> None:  # noqa: C901

        for node in gm.graph.nodes:
            if node.op in {"call_module", "call_method"}:
                raise SpecViolationError(
                    "call_module is not valid: got a class '{}' ".format(node.target),
                )

            if node.op == "call_function":
                if node.target not in self.valid_builtin_funcs():
                    self.check_valid_op(node.target)


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
        # if we also turn on edge ops and validator (_use_edge_ops=True, _check_ir_validity=True), we will run
        # into RunHigherOrderOperatorError. The only thing we can do right now is to ignore this error, since
        # by definition it's still a valid Edge dialect. This is not ideal because it ignores possible invalidity
        # later in the graph.
        return

    if validator.violating_ops:
        raise SpecViolationError(
            f"These operators are taking Tensor inputs with mismatched dtypes: {validator.violating_ops}"
        )


class EXIREdgeDialectVerifier(Verifier):
    def __init__(self, check_edge_ops: bool = False) -> None:
        self.check_edge_ops = check_edge_ops

    def valid_builtin_funcs(self):
        builtin_funcs = super().valid_builtin_funcs()
        builtin_funcs.extend(VALID_BUILTIN_FUNCS)
        return builtin_funcs

    def check_valid_edge_op(self, op):
        if isinstance(op, OpOverload) and not isinstance(op, EdgeOpOverload):
            raise SpecViolationError(
                "Operator {}.{} is not an Edge operator.".format(
                    op.__module__, op.__name__
                )
            )

    def check_valid_aten_op(self, op) -> None:
        super().check_valid_op(op)

        op_name = op.name if hasattr(op, "name") else op.__name__

        if not isinstance(op, OpOverload):
            raise SpecViolationError(
                "Operator '{}' is not a registered Op".format(op_name),
            )

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

    def check_valid(self, gm: GraphModule) -> None:
        if self.check_edge_ops:
            self.check_valid_op = self.check_valid_edge_op
            super().check_valid(gm)
            _check_tensors_are_contiguous(gm)
            _check_tensor_args_matching_op_allowed_dtype(gm)
        else:
            self.check_valid_op = self.check_valid_aten_op

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
