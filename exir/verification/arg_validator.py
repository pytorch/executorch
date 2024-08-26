# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from executorch.exir.dialects.edge._ops import EdgeDialectFunctionSchema, EdgeOpOverload
from executorch.exir.emit._emitter import _Argument, _Target
from executorch.exir.error import ExportError, InternalError
from torch._ops import HigherOrderOperator


class RunHigherOrderOperatorError(Exception):
    """
    Raised when an we try to run delegate or other HigherOrderOperator in a graph module.
    E.g., %executorch_call_delegate : [#users=1] = call_function[
        target=torch.ops.higher_order.executorch_call_delegate](args = (%lowered_module_0, %arg0_1), kwargs = {})
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# pyre-ignore[13]: Attribute `node` is never initialized.
class EdgeOpArgValidator(torch.fx.Interpreter):
    """
    Validate whether all the Tensor arguments passed to an operator are valid in terms of allowed dtype.
    Expecting all the operators are EdgeOpOverload which contains the allowed dtype information.
    Violating operators are being kept in self.violating_ops
    """

    node: torch.fx.Node

    def __init__(self, graph_module: torch.fx.GraphModule) -> None:
        super().__init__(graph_module)
        self.violating_ops: Dict[EdgeOpOverload, Dict[str, Optional[torch.dtype]]] = (
            defaultdict(dict)
        )

    def run_node(self, n: torch.fx.Node) -> None:
        self.node = n
        try:
            ret = super().run_node(n)
        except Exception as e:
            if isinstance(e, (InternalError, ExportError, RunHigherOrderOperatorError)):
                raise e
            else:
                raise InternalError(str(e)) from e
        return ret

    def _get_kernel_arg(self, schema_arg, schema_arg_idx, args, kwargs):
        if schema_arg.name in kwargs:
            kernel_arg = kwargs[schema_arg.name]
        elif not schema_arg.kwarg_only and schema_arg_idx < len(args):
            kernel_arg = args[schema_arg_idx]
        else:
            kernel_arg = schema_arg.default_value

        return kernel_arg

    def call_function(  # noqa: C901  # pyre-fixme[14]
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> Any:
        """
        Go through all the node.target and validate their Tensor arguments are having the allowed dtypes.
        """
        if not isinstance(target, EdgeOpOverload) or not isinstance(
            target._schema, EdgeDialectFunctionSchema
        ):
            if isinstance(target, HigherOrderOperator):
                raise RunHigherOrderOperatorError("Can't run delegate")
            return super().call_function(target, args, kwargs)  # pyre-fixme[6]

        # TODO(gasoonjia): Update Optional[torch.dtype] to a concrete class to support mixed dtypes in tensorlist.
        tensor_arg_types: Dict[str, Optional[torch.dtype]] = {}
        for i, schema_arg in enumerate(target._schema.arguments):
            if (
                isinstance(schema_arg.type, torch.TensorType)
                or schema_arg.type == torch.OptionalType.ofTensor()
            ):
                kernel_arg = self._get_kernel_arg(schema_arg, i, args, kwargs)
                if not isinstance(kernel_arg, torch.Tensor):
                    continue
                tensor_arg_types[schema_arg.name] = kernel_arg.dtype
            elif schema_arg.type == torch.ListType.ofTensors():
                kernel_arg = self._get_kernel_arg(schema_arg, i, args, kwargs)
                if not isinstance(kernel_arg, list) or not all(
                    isinstance(kernel_arg[i], torch.Tensor)
                    for i in range(len(kernel_arg))
                ):
                    continue
                if len(kernel_arg):
                    tensor_arg_types[schema_arg.name] = kernel_arg[0].dtype
                else:
                    # If kernel_arg is an empty list, treat its type as None.
                    # FunctionDtypeConstraint.validate will take None as any legal dtype.
                    tensor_arg_types[schema_arg.name] = None

        ret_index = 0
        kernel_rets = self.node.meta["val"]
        ret_iter = iter(
            kernel_rets if isinstance(kernel_rets, Sequence) else [kernel_rets]
        )
        for schema_ret in target._schema.returns:
            name = schema_ret.name if schema_ret.name else f"__ret_{ret_index}"
            kernel_ret = next(ret_iter)
            # Return value should not be in OptionalTensor type, so only check torch.TensorType here.
            if isinstance(schema_ret.type, torch.TensorType) and isinstance(
                kernel_ret, torch.Tensor
            ):
                tensor_arg_types[name] = kernel_ret.dtype
                ret_index += 1
            elif schema_ret.type == torch.ListType.ofTensors() and all(
                isinstance(kernel_ret[i], torch.Tensor) for i in range(len(kernel_ret))
            ):
                if len(kernel_ret):
                    tensor_arg_types[name] = kernel_ret[0].dtype
                else:
                    tensor_arg_types[name] = None
                ret_index += 1

        valid = target._schema.dtype_constraint.validate(tensor_arg_types)
        if not valid:
            self.violating_ops[target] = tensor_arg_types
        return super().call_function(target, args, kwargs)  # pyre-fixme[6]
