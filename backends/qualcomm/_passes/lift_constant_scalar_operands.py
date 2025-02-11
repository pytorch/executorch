# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch import fx
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix

COMPARE_SCALAR_OPS = {
    torch.ops.aten.gt.Scalar: torch.ops.aten.gt.Tensor,
    torch.ops.aten.lt.Scalar: torch.ops.aten.lt.Tensor,
    torch.ops.aten.ge.Scalar: torch.ops.aten.ge.Tensor,
    torch.ops.aten.le.Scalar: torch.ops.aten.le.Tensor,
    torch.ops.aten.eq.Scalar: torch.ops.aten.eq.Tensor,
    torch.ops.aten.ne.Scalar: torch.ops.aten.ne.Tensor,
}


def _not_float_tensor(node: fx.Node) -> bool:
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True
    return node.meta["val"].dtype != torch.float32


def _not_bool_tensor(node: fx.Node) -> bool:
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True
    return node.meta["val"].dtype != torch.bool


def lift_constant_scalar_operands(gm: torch.fx.GraphModule) -> None:
    # If the node is add(input, constant) and constant is a scalar, we can lift the constant 
    # and the annotation for quantization will insert q/dq nodes around the lifted constant.
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor,
            torch.ops.aten.rsub.Scalar,
            torch.ops.aten.add_.Scalar,
        ) + tuple(COMPARE_SCALAR_OPS.keys()):
            continue
        const_arg = None
        non_const_arg = None
        for arg in n.args:
            if isinstance(arg, torch.fx.Node):
                non_const_arg = arg
            else:
                const_arg = arg
        if non_const_arg is None or const_arg is None:
            continue

        if _not_float_tensor(n) and _not_bool_tensor(n):
            continue

        if not _not_float_tensor(n):
            tensor_constant = torch.tensor(
                [const_arg],
                dtype=n.meta["val"].dtype,
                device=n.meta["val"].device,
            )
        else:
            tensor_constant = torch.tensor(
                [const_arg],
                dtype=n.args[0].meta["val"].dtype,
                device=n.meta["val"].device,
            )
        tensor_constant_name = get_new_attr_name_with_prefix("_tensor_constant_")(gm)
        gm.register_buffer(tensor_constant_name, tensor_constant)

        fake_mode = n.meta["val"].fake_mode
        with gm.graph.inserting_before(n):
            get_attr_node = gm.graph.get_attr(tensor_constant_name)
            get_attr_node.meta["val"] = fake_mode.from_tensor(tensor_constant)

        if n.target == torch.ops.aten.rsub.Scalar:
            n.args = (get_attr_node, non_const_arg) + n.args[2:]
            n.target = torch.ops.aten.sub.Tensor
        else:
            n.args = (non_const_arg, get_attr_node) + n.args[2:]

        if n.target == torch.ops.aten.add_.Scalar:
            n.args = (non_const_arg, get_attr_node) + n.args[2:]
            n.target = torch.ops.aten.add.Tensor

        if n.target in tuple(COMPARE_SCALAR_OPS.keys()):
            n.args = (non_const_arg, get_attr_node) + n.args[2:]
            n.target = COMPARE_SCALAR_OPS[n.target]

    gm.recompile()
