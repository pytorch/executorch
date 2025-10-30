# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-unsafe

import itertools
import logging
import operator
import types
from contextlib import nullcontext
from typing import Any, List, Optional, Tuple, Type

import torch
from executorch.exir.capture._config import EdgeCompileConfig
from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.error import ExportError, ExportErrorType
from executorch.exir.lowered_backend_module import LoweredBackendModule
from executorch.exir.passes.dim_order_ops_registry import DimOrderOpsMap
from executorch.exir.passes.executorch_prim_ops_registry import _EXECUTORCH_SYM_OPS
from executorch.exir.passes.replace_aten_with_edge_pass import DISALLOW_LIST
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
        if op in DimOrderOpsMap:
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
    core_aten_ops_exception_list: Optional[List[torch._ops.OpOverload]] = None,
    preserve_ops: Optional[List[torch._ops.OpOverload]] = None,
):
    """
    Returns a verifier class that runs ATen dialect specific checks on the graph module.
    """
    _core_aten_ops_exception_list = core_aten_ops_exception_list or []
    _preserve_ops = preserve_ops or []
    # merge the exception list from edge_compile_config and exception_list
    if edge_compile_config:
        if edge_compile_config._core_aten_ops_exception_list:
            _core_aten_ops_exception_list.extend(
                edge_compile_config._core_aten_ops_exception_list
            )
        if edge_compile_config.preserve_ops:
            _preserve_ops.extend(edge_compile_config.preserve_ops)

    class _EXIRATenDialectVerifier(EXIRATenDialectVerifierBase):
        dialect = "OLD_EXIR_ATEN"

        def __init__(self) -> None:
            super().__init__()
            # Note: here we are using the exception list passed from EXIRATenDialectVerifier function!
            self._core_aten_ops_exception_list = _core_aten_ops_exception_list
            self._preserve_ops = _preserve_ops

        def _get_core_aten_ops_exception_list(self) -> List[torch._ops.OpOverload]:
            exception_list = (
                [
                    torch.ops.aten.mkldnn_rnn_layer.default,
                    torch.ops.aten._upsample_bilinear2d_aa.default,
                    torch.ops.aten.quantize_per_tensor.default,
                    torch.ops.aten.dequantize.self,
                    torch.ops.aten.max.default,  # TODO(T188268054)
                    torch.ops.aten.min.default,  # TODO(T188268054)
                    torch.ops.aten.full_like.default,  # TODO(T183507359)
                ]
                + list(_EXECUTORCH_SYM_OPS)
                + DISALLOW_LIST
                + self._core_aten_ops_exception_list
            )

            return exception_list

        def check_valid_op(self, op):
            if isinstance(op, OpOverload):
                # TODO These special ops should be removable easily.
                if (
                    op.namespace != "aten"
                    or op in self._get_core_aten_ops_exception_list()
                ):
                    return
                if op in self._preserve_ops:
                    if op.namespace != "aten":
                        raise RuntimeError(
                            f"Only preserve aten ops. Received op {op} with namespace {op.namespace}."
                        )
                    # Preserved ops should not include mutation or view,
                    # which may affect memory planning.
                    if op.is_view:
                        raise RuntimeError(
                            f"Cannot preserve operator {op} because it is a view."
                        )
                    if op._schema.is_mutable:
                        logging.warning(
                            f"Preserving mutation ops like {op} is a no-op because run_decomposition functionalizes it and prevents it from showing up."
                        )

                    return
                if torch.Tag.core not in op.tags and torch.Tag.view_copy not in op.tags:
                    # NOTE(qihan): whether view_copy operators are marked as canonical is still under
                    #            discussion.
                    raise SpecViolationError(
                        f"""
Operator {op.__module__}.{op.__name__} is not in Core ATen opset (https://pytorch.org/docs/stable/torch.compiler_ir.html#core-aten-ir)."
There are a few things to try:
1. You can proceed with `to_edge(compile_config=EdgeCompileConfig(_core_aten_ops_exception_list=[torch.ops.{str(op)}]))`.
   Please make sure that the backend(s) you are planning to lower to is able to handle {str(op)}, or you have a corresponding kernel linked to your runtime.

2. Sometimes inference and training gives slightly different op set. Try adding `with torch.no_grad():` context manager if you are export for inference only.

3. If the error persists after 2, this is likely caused by torch.export() + core ATen decomposition producing unexpected operators for your model.
   If you believe this operator should be included into core ATen opset, please create an issue in https://github.com/pytorch/pytorch/issues and add `module: core aten` tag.
                        """
                    )

    ret = _EXIRATenDialectVerifier
    if not class_only:
        ret = ret()
    return ret


def get_aten_verifier(config: EdgeCompileConfig):
    return (
        EXIRATenDialectVerifier(
            class_only=True,
            core_aten_ops_exception_list=config._core_aten_ops_exception_list,
            preserve_ops=config.preserve_ops,
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
        error_msg = ""
        for op, node in validator.violating_ops.items():
            # error_msg += f"#####################################################\n"
            error_msg += f"\nOperator: {op} with args: {node[0]}\n"
            error_msg += f"stack trace: {node[1].stack_trace}\n"
            # error_msg += f"#####################################################\n"
        raise SpecViolationError(
            f"These operators are taking Tensor inputs with mismatched dtypes:\n{error_msg}"
            "Please make sure the dtypes of the Tensor inputs are the same as the dtypes of the corresponding outputs."
        )


def EXIREdgeDialectVerifier(  # noqa: C901
    edge_compile_config: Optional[EdgeCompileConfig] = None,
    class_only: bool = False,
    core_aten_ops_exception_list: Optional[List[torch._ops.OpOverload]] = None,
    preserve_ops: Optional[List[torch._ops.OpOverload]] = None,
):
    _core_aten_ops_exception_list = core_aten_ops_exception_list or []
    _preserve_ops = preserve_ops or []
    # merge the exception list from edge_compile_config and exception_list
    if edge_compile_config:
        if edge_compile_config._core_aten_ops_exception_list:
            _core_aten_ops_exception_list.extend(
                edge_compile_config._core_aten_ops_exception_list
            )
        if edge_compile_config.preserve_ops:
            _preserve_ops.extend(edge_compile_config.preserve_ops)

    class _EXIREdgeDialectVerifier(Verifier):
        dialect = "EDGE"

        def __init__(self) -> None:
            _edge_compile_config = edge_compile_config or EdgeCompileConfig()

            self.enable = _edge_compile_config._check_ir_validity
            self.check_edge_ops = _edge_compile_config._use_edge_ops
            self.use_dim_order = not _edge_compile_config._skip_dim_order

            self._core_aten_ops_exception_list = _core_aten_ops_exception_list
            self._preserve_ops = _preserve_ops

            self.aten_op_verifier = EXIRATenDialectVerifier(
                core_aten_ops_exception_list=_core_aten_ops_exception_list,
                preserve_ops=_preserve_ops,
            )
            self.check_valid_aten_op = self.aten_op_verifier.check_valid_op

            if self.check_edge_ops:
                self.check_valid_op = self.check_valid_edge_op
            else:
                self.check_valid_op = self.check_valid_aten_op

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
                in [operator.getitem]
                + DISALLOW_LIST
                + list(_EXECUTORCH_SYM_OPS)
                + self._core_aten_ops_exception_list
            ):
                return

            if isinstance(op, OpOverload) and not isinstance(op, EdgeOpOverload):
                raise SpecViolationError(
                    "Operator {}.{} is not an Edge operator.".format(
                        op.__module__, op.__name__
                    )
                )
            if isinstance(op, EdgeOpOverload):
                _check_valid_dim_order_ops(op, self.use_dim_order)
                self.check_valid_aten_op(op._op)

            if isinstance(op, types.FunctionType):
                assert op.__name__ in ("alloc",)

        def check_additional(self, gm: GraphModule) -> None:
            if not self.enable:
                return
            if self.check_edge_ops:
                _check_tensors_are_contiguous(gm)
                _check_tensor_args_matching_op_allowed_dtype(gm)

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
