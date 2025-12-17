# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from numbers import Number
from types import BuiltinFunctionType, BuiltinMethodType
from typing import Dict

import torch
from executorch.backends.qualcomm._passes.utils import is_float_tensor
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.passes import dead_code_elimination_pass
from torch import fx
from torch.ops import aten as aten
from torchao.quantization.pt2e.utils import get_new_attr_name_with_prefix


@dataclass(frozen=True)
class TensorConstant:
    tensor: torch.Tensor
    name: str


@dataclass(frozen=True)
class TensorOpInfo:
    target: torch._ops.OpOverload
    use_schema_args: bool
    use_self_dtype: bool


SCALAR_OPS = {
    aten.eq.Scalar: TensorOpInfo(aten.eq.Tensor, False, False),
    aten.ge.Scalar: TensorOpInfo(aten.ge.Tensor, False, False),
    aten.gt.Scalar: TensorOpInfo(aten.gt.Tensor, False, False),
    aten.le.Scalar: TensorOpInfo(aten.le.Tensor, False, False),
    aten.lt.Scalar: TensorOpInfo(aten.lt.Tensor, False, False),
    aten.ne.Scalar: TensorOpInfo(aten.ne.Tensor, False, False),
    aten.add.Scalar: TensorOpInfo(aten.add.Tensor, False, False),
    aten.add_.Scalar: TensorOpInfo(aten.add_.Tensor, False, False),
    # For below cases, refer to LiftAddTensor Model in UT for sample
    aten.add.Tensor: TensorOpInfo(aten.add.Tensor, False, False),
    aten.div.Scalar: TensorOpInfo(aten.div.Tensor, False, False),
    aten.mul.Scalar: TensorOpInfo(aten.mul.Tensor, False, False),
    aten.rsub.Scalar: TensorOpInfo(aten.rsub.Tensor, False, False),
    aten.sub.Scalar: TensorOpInfo(aten.sub.Tensor, False, False),
    aten.sub.Tensor: TensorOpInfo(aten.sub.Tensor, False, False),
    aten.pow.Tensor_Scalar: TensorOpInfo(aten.pow.Tensor_Tensor, False, False),
    # The scalar number arg[1] is missing when using default. Result in a corner case to deal
    aten.leaky_relu.default: TensorOpInfo(aten.prelu.default, True, False),
    aten.leaky_relu_.default: TensorOpInfo(aten.prelu.default, True, False),
    aten.where.ScalarSelf: TensorOpInfo(aten.where.self, False, True),
    aten.where.ScalarOther: TensorOpInfo(aten.where.self, False, True),
    aten.where.Scalar: TensorOpInfo(aten.where.self, False, True),
    aten.masked_fill.Scalar: TensorOpInfo(aten.masked_fill.Tensor, False, False),
    aten.masked_fill_.Scalar: TensorOpInfo(aten.masked_fill.Tensor, False, False),
    aten.bitwise_xor.Scalar: TensorOpInfo(aten.bitwise_xor.Tensor, False, False),
}


SKIP_LIFT_OPS = {
    aten.full_like.default,
    aten.full.default,
    aten.arange.start_step,
    aten.arange.default,
    aten.scalar_tensor.default,
    aten.elu.default,
    aten.hardtanh.default,
}


class LiftConstantScalarOperands(ExportPass):
    """
    Lift constant scalar so that we can use observer of quantizer.
    For floating point model, lift constant scalar to avoid
    creating temporary tensors for scalar node in the operation builder
    """

    def __init__(self):
        super(LiftConstantScalarOperands, self).__init__()

    def _build_tensor_constant(
        self, gm: torch.fx.GraphModule, node: fx.Node, const_val
    ) -> TensorConstant:
        # For dtype, in some cases, we cannot use node.args[0] as scalar dtype.
        # Ex: Where op args[0] can be bool, however, we probably want args[1] and args[2] to be dtype same as node.meta["val"] instead of bool type
        tensor = torch.tensor(
            const_val,
            dtype=(
                node.args[0].meta["val"].dtype
                if not is_float_tensor(node)
                and (info := SCALAR_OPS.get(node.target))
                and not info.use_self_dtype
                else node.meta["val"].dtype
            ),
            device=node.meta["val"].device,
        )
        name = get_new_attr_name_with_prefix("_tensor_constant_")(gm)
        tensor_constant = TensorConstant(tensor, name)
        return tensor_constant

    def _register_tensor(
        self, gm: torch.fx.GraphModule, node: fx.Node, tensor_constant: TensorConstant
    ) -> fx.Node:
        gm.register_buffer(tensor_constant.name, tensor_constant.tensor)

        fake_mode = node.meta["val"].fake_mode
        with gm.graph.inserting_before(node):
            get_attr_node = gm.graph.get_attr(tensor_constant.name)
            get_attr_node.meta["val"] = fake_mode.from_tensor(tensor_constant.tensor)
        return get_attr_node

    def _update_node(self, node: fx.Node, tensor_args: Dict) -> None:
        new_args = list(node.args)
        if (info := SCALAR_OPS.get(node.target)) and info.use_schema_args:
            new_args += [None] * max(
                0, (len(node.target._schema.arguments) - len(new_args))
            )

        for k, v in tensor_args.items():
            new_args[k] = v
        node.args = tuple(new_args)
        node.target = SCALAR_OPS.get(node.target, node).target

    def _create_tensor_args(
        self, node: fx.Node, gm: torch.fx.graph_module
    ) -> Dict[int, TensorConstant]:
        tensor_args = {}
        for i, arg in enumerate(node.args):
            schema = node.target._schema.arguments[i]
            is_tensor_arg_got_num = isinstance(
                schema.type, torch.TensorType
            ) and isinstance(arg, Number)

            is_scalar_arg = (
                isinstance(schema.type, torch.NumberType) and node.target in SCALAR_OPS
            )

            # This is for showing warning of new-coming op
            is_arg_num_type = (
                isinstance(schema.type, torch.NumberType)
                and node.target not in SCALAR_OPS
            )

            if is_tensor_arg_got_num or is_scalar_arg:
                tensor_constant = self._build_tensor_constant(gm, node, arg)
                tensor_constant_node = self._register_tensor(gm, node, tensor_constant)
                tensor_args[i] = tensor_constant_node

            elif is_arg_num_type:
                print(
                    f"[WARNING] the {i} th arg of node {node} is NumberType, might need to lift"
                )

        if (info := SCALAR_OPS.get(node.target)) and info.use_schema_args:
            schema_args = list(node.target._schema.arguments)
            for i, sa in enumerate(schema_args):
                if isinstance(sa.type, torch.NumberType) and i not in tensor_args:
                    tensor_constant = self._build_tensor_constant(
                        gm, node, sa.default_value
                    )
                    tensor_constant_node = self._register_tensor(
                        gm, node, tensor_constant
                    )
                    tensor_args[i] = tensor_constant_node
        return tensor_args

    def _lift(self, gm: torch.fx.GraphModule) -> None:
        for n in gm.graph.nodes:
            if (
                n.op != "call_function"
                or isinstance(n.target, (BuiltinMethodType, BuiltinFunctionType))
                or n.target in SKIP_LIFT_OPS
            ):
                continue

            if tensor_args := self._create_tensor_args(n, gm):
                self._update_node(n, tensor_args)

    def call(self, graph_module: torch.fx.GraphModule):
        self._lift(graph_module)
        graph_module.recompile()
        dead_code_elimination_pass(graph_module)
        return PassResult(graph_module, True)
