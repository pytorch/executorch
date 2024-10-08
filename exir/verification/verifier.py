# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import operator
import types
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple, Type

import torch
from executorch.exir.capture._config import EdgeCompileConfig
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.verification.arg_validator import (
    EdgeOpArgValidator,
    RunHigherOrderOperatorError,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._export.utils import _detect_fake_mode_from_gm

from torch._export.verifier import SpecViolationError, Verifier
from torch._ops import OpOverload
from torch._subclasses import FakeTensor
from torch.export.exported_program import ExportedProgram
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


def _check_valid_dim_order_ops(op, use_dim_order) -> None:
    if use_dim_order:
        if op in (torch.ops.aten._to_copy.default,):
            raise SpecViolationError(f"{op} should not be used in dim_order mode")
    else:  # not using dim_order
        if op.namespace in ("dim_order_ops",):
            raise SpecViolationError(f"{op} should not be used in non-dim_order mode")


class EXIRATenDialectVerifierBase(Verifier):
    dialect = "OLD_EXIR_ATEN_DISABLED"

    def allowed_getattr_types(self) -> Tuple[Type[Any], ...]:
        return (
            torch.fx.GraphModule,
            LoweredBackendModule,
            torch.Tensor,
            torch.ScriptObject,
        )

    def allowed_op_types(self):
        return super().allowed_op_types() + (torch._ops.OpOverloadPacket,)

    def __call__(self, *args, **kwargs):
        if hasattr(self, "_check_graph_module"):
            return self._check_graph_module(*args, **kwargs)
        elif hasattr(self, "check_valid"):
            return self.check_valid(*args, **kwargs)
        else:
            raise RuntimeError("")


def EXIRATenDialectVerifier(  # noqa: C901
    edge_compile_config: Optional[EdgeCompileConfig] = None,
    class_only: bool = False,
    exception_list: Optional[List[torch._ops.OpOverload]] = None,
):
    """
    Returns a verifier class that runs ATen dialect specific checks on the graph module.
    """
    # merge the exception list from edge_compile_config and exception_list
    if edge_compile_config and edge_compile_config._core_aten_ops_exception_list:
        exception_list = edge_compile_config._core_aten_ops_exception_list + (
            exception_list or []
        )

    class _EXIRATenDialectVerifier(EXIRATenDialectVerifierBase):
        dialect = "OLD_EXIR_ATEN"

        def __init__(self) -> None:
            super().__init__()
            # Note: here we are using the exception list passed from EXIRATenDialectVerifier function!
            self._exception_list = exception_list if exception_list else []

        def _get_exception_list(self) -> List[torch._ops.OpOverload]:
            exception_list = [
                torch.ops.aten.mkldnn_rnn_layer.default,
                torch.ops.aten._upsample_bilinear2d_aa.default,
                torch.ops.aten.quantize_per_tensor.default,
                torch.ops.aten.dequantize.self,
                torch.ops.aten.max.default,  # TODO(T188268054)
                torch.ops.aten.min.default,  # TODO(T188268054)
                torch.ops.aten.full_like.default,  # TODO(T183507359)
            ]
            exception_list += self._exception_list

            return exception_list

        def check_valid_op(self, op):
            if isinstance(op, OpOverload):
                # TODO These special ops should be removable easily.
                if op.namespace != "aten" or op in self._get_exception_list():
                    return
                if torch.Tag.core not in op.tags and torch.Tag.view_copy not in op.tags:
                    # NOTE(qihan): whether view_copy operators are marked as canonical is still under
                    #            discussion.
                    raise SpecViolationError(
                        f"Operator {op.__module__}.{op.__name__} is not Aten Canonical."
                    )

    ret = _EXIRATenDialectVerifier
    if not class_only:
        ret = ret()
    return ret


def get_aten_verifier(config: EdgeCompileConfig):
    return (
        EXIRATenDialectVerifier(
            class_only=True, exception_list=config._core_aten_ops_exception_list
        )
        if config._check_ir_validity
        else EXIRATenDialectVerifierBase
    )


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
    fake_mode = _detect_fake_mode_from_gm(gm) or nullcontext()
    try:
        with enable_python_dispatcher(), fake_mode:
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


def EXIREdgeDialectVerifier(  # noqa: C901
    edge_compile_config: Optional[EdgeCompileConfig] = None,
    class_only: bool = False,
    exception_list: Optional[List[torch._ops.OpOverload]] = None,
):
    # merge the exception list from edge_compile_config and exception_list
    if edge_compile_config and edge_compile_config._core_aten_ops_exception_list:
        exception_list = edge_compile_config._core_aten_ops_exception_list + (
            exception_list or []
        )

    class _EXIREdgeDialectVerifier(Verifier):
        dialect = "EDGE"

        def __init__(self) -> None:
            _edge_compile_config = edge_compile_config or EdgeCompileConfig()

            self.enable = _edge_compile_config._check_ir_validity
            self.check_edge_ops = _edge_compile_config._use_edge_ops
            self.use_dim_order = not _edge_compile_config._skip_dim_order

            self.aten_op_verifier = EXIRATenDialectVerifier(
                exception_list=exception_list
            )
            self.check_valid_aten_op = self.aten_op_verifier.check_valid_op

            if self.check_edge_ops:
                self.check_valid_op = self.check_valid_edge_op
            else:
                self.check_valid_op = self.check_valid_aten_op
            self._exception_list = exception_list if exception_list else []

        def allowed_getattr_types(self) -> Tuple[Type[Any], ...]:
            return (
                torch.fx.GraphModule,
                LoweredBackendModule,
                torch.Tensor,
                torch.ScriptObject,
            )

        def allowed_op_types(self):
            return super().allowed_op_types() + (EdgeOpOverload, types.FunctionType)

        def check_valid_edge_op(self, op):
            if not self.enable:
                return
            if (
                op
                in [
                    operator.getitem,
                    torch.ops.aten.sym_size.int,
                    torch.ops.aten.scalar_tensor.default,
                    torch.ops.aten._assert_async.msg,
                    torch.ops.aten._assert_scalar.default,
                ]
                + self._exception_list
            ):
                return

            if isinstance(op, OpOverload) and not isinstance(op, EdgeOpOverload):
                raise SpecViolationError(
                    "Operator {}.{} is not an Edge operator.".format(
                        op.__module__, op.__name__
                    )
                )
            if isinstance(op, EdgeOpOverload):
                _check_valid_dim_order_ops(op._op, self.use_dim_order)
                self.check_valid_aten_op(op._op)

            if isinstance(op, types.FunctionType):
                assert op.__name__ in ("alloc",)

        def check_additional(self, gm: GraphModule) -> None:
            if not self.enable:
                return
            if self.check_edge_ops:
                _check_tensors_are_contiguous(gm)
                _check_tensor_args_matching_op_allowed_dtype(gm)

        def check_valid_op(self, op):
            if isinstance(op, OpOverload):
                # TODO These special ops should be removable easily.
                if op.namespace in (
                    "quantized_decomposed",
                    "boltnn_nimble",
                    "nimble",
                    "quantized",
                    "dim_order_ops",
                ) or op in (
                    torch.ops.aten.mkldnn_rnn_layer.default,
                    torch.ops.aten._upsample_bilinear2d_aa.default,
                    torch.ops.aten.quantize_per_tensor.default,
                    torch.ops.aten.dequantize.self,
                    torch.ops.aten.max.default,
                    torch.ops.aten.full_like.default,  # TODO(T183507359)
                ):
                    return
                if torch.Tag.core not in op.tags and torch.Tag.view_copy not in op.tags:
                    # NOTE(qihan): whether view_copy operators are marked as canonical is still under
                    #            discussion.
                    raise SpecViolationError(
                        f"Operator {op.__module__}.{op.__name__} is not Aten Canonical."
                    )

        def is_valid(self, gm: GraphModule) -> bool:
            try:
                self(gm)
                return True
            except SpecViolationError:
                return False

        def __call__(self, ep_or_gm):
            if not self.enable:
                return
            gm = ep_or_gm
            if isinstance(gm, ExportedProgram):
                gm = ep_or_gm.graph_module
            if hasattr(self, "_check_graph_module"):
                return self._check_graph_module(gm)
            elif hasattr(self, "check_valid"):
                return self.check_valid(gm)
            else:
                raise RuntimeError("")

    ret = _EXIREdgeDialectVerifier
    if not class_only:
        ret = ret()
    return ret


EXIREdgeDialectVerifier()
