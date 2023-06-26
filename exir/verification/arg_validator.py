# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from typing import Any, Dict, List, Sequence, Tuple

import torch
from executorch.exir.dialects.edge._ops import EdgeDialectFunctionSchema, EdgeOpOverload
from executorch.exir.emit._emitter import _Argument, _Target
from executorch.exir.error import ExportError, InternalError

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
        self.violating_ops: List[Tuple[EdgeOpOverload, Dict[str, torch.dtype]]] = []

    def run_node(self, n: torch.fx.Node) -> None:
        self.node = n
        try:
            ret = super().run_node(n)
        except Exception as e:
            if isinstance(e, (InternalError, ExportError)):
                raise e
            else:
                raise InternalError(str(e)) from e
        return ret

    def call_function(
        self, target: _Target, args: Tuple[_Argument, ...], kwargs: Dict[str, _Argument]
    ) -> Any:
        """
        Go through all the node.target and validate their Tensor arguments are having the allowed dtypes.
        """
        if not isinstance(target, EdgeOpOverload) or not isinstance(
            target._schema, EdgeDialectFunctionSchema
        ):
            return False
        tensor_arg_types: Dict[str, torch.dtype] = {}
        for i, schema_arg in enumerate(target._schema.arguments):
            if not isinstance(schema_arg.type, torch.TensorType):
                continue
            if schema_arg.name in kwargs:
                kernel_arg = kwargs[schema_arg.name]
            elif not schema_arg.kwarg_only and i < len(args):
                kernel_arg = args[i]
            else:
                kernel_arg = schema_arg.default_value
            if not isinstance(kernel_arg, torch.Tensor):
                continue
            tensor_arg_types[schema_arg.name] = kernel_arg.dtype
        ret_index = 0
        kernel_rets = self.node.meta["val"]
        ret_iter = iter(
            kernel_rets if isinstance(kernel_rets, Sequence) else [kernel_rets]
        )
        for schema_ret in target._schema.returns:
            if isinstance(schema_ret.type, torch.TensorType):
                name = schema_ret.name if schema_ret.name else f"__ret_{ret_index}"
                kernel_ret = next(ret_iter)
                if not isinstance(kernel_ret, torch.Tensor):
                    continue
                tensor_arg_types[name] = kernel_ret.dtype
                ret_index += 1

        valid = target._schema.dtype_constraint.validate(tensor_arg_types)
        if not valid:
            self.violating_ops.append((target, tensor_arg_types))
        return valid
