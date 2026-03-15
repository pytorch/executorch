import logging
from typing import List, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.node import _get_qualified_name
from torch_tensorrt.dynamo.conversion.converter_utils import args_bounds_check

# dead-code elimination, linting, and recompilation for graph, in-place
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def replace_complex_placeholder_to_tuple(
    gm: torch.fx.GraphModule,
    inputListindices: List[int],
) -> torch.fx.GraphModule:
    modified_graph = False
    input_arg_list = [f"arg{inputListIndex}_1" for inputListIndex in inputListindices]
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.target in input_arg_list:
            from torch._subclasses.fake_tensor import FakeTensorMode

            node_shape = node.meta["val"].size()
            new_node_shape = node_shape + (2,)
            new_node_dtype = None
            if node.meta["val"].dtype == torch.complex64:
                new_node_dtype = torch.float32
            else:
                new_node_dtype = torch.float64
            fake_mode = FakeTensorMode()

            real_tensor = torch.empty(new_node_shape, dtype=new_node_dtype)
            with FakeTensorMode() as fake_mode:
                new_placeholder_tuple = fake_mode.from_tensor(real_tensor)
            node.meta["val"] = new_placeholder_tuple
            modified_graph = True
            # propagate the meta data change for the downstream ops
            # TODO:to check if this is required in all cases
            propogate_complex_num_shape_change_till_complex_mul(gm, node, fake_mode)

    # If graph was modified, clean it up
    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Graph after fusing wait_tensor and distributed op tensor:\n{gm.graph}"
        )

    return gm


def infer_slice_shape(node: torch.fx.Node) -> Tuple[int, ...]:
    input_shape = node.args[0].meta["val"].shape
    slice_args = node.args
    dim = slice_args[1]
    start = slice_args[2]
    end = slice_args[3]
    step = args_bounds_check(slice_args, 4, replacement=1)
    new_shape = list(input_shape)
    new_shape[dim] = (end - start + step - 1) // step
    return tuple(new_shape)


def infer_reshape_shape(node: torch.fx.Node) -> torch.fx.node.Argument:
    return node.args[1]


shape_inference_funcs = {
    "torch.ops.aten.slice.Tensor": infer_slice_shape,
    "torch.ops.aten.reshape.default": infer_reshape_shape,
}


# Please note this function is for the use case of Llama model
# with complex placeholder->reshape->slice->complex mul
# Hence mul is the terminating op
def propogate_complex_num_shape_change_till_complex_mul(
    node: torch.fx.Node, start_node: torch.fx.Node, fake_mode: FakeTensorMode
) -> None:
    visited_nodes = set()
    stack = [start_node]
    while stack:
        node = stack.pop()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        update_node_meta(node, fake_mode)
        for user in node.users:
            if (
                user.op == "call_function"
                and _get_qualified_name(user.target) == "torch.ops.aten.mul.Tensor"
            ):
                continue
            stack.append(user)


def update_node_meta(node: torch.fx.Node, fake_mode: FakeTensorMode) -> None:
    op_name = node.name
    op_target = node.target

    if node.op == "call_function":
        op_target = _get_qualified_name(node.target)

    if op_target in shape_inference_funcs:
        new_shape = shape_inference_funcs[op_target](node)
        real_tensor = torch.empty(new_shape, dtype=node.meta["val"].dtype)
        node.meta["val"] = fake_mode.from_tensor(real_tensor)
    else:
        print("No shape for the inference function", {op_name})
