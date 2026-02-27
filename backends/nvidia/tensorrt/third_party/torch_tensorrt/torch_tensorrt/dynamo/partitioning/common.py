import logging
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo.utils import contains_sym_int, extract_var_range_info

logger = logging.getLogger(__name__)


def construct_dynamic_input(
    input_shape: torch.Size,
    input_dtype: torch.dtype,
    name: str = "",
    is_shape_tensor: bool = False,
) -> Input:
    """
    Constructs a torch_tensorrt.Input based on a symbolic input
    Args:
        input_shape: A symbolic shape / regular shape of a tensor (which can have a  mix of SymInt nodes and static values)
    Returns:
        A dynamic shaped torch_tensorrt.Input which has the properties of the symbolic shaped input.
    """
    min_shape = []
    opt_shape = []
    max_shape = []
    for dim in input_shape:
        if isinstance(dim, torch.SymInt):
            min_max_opt = extract_var_range_info(dim)
            min_shape.append(min_max_opt["min"])
            # if opt not exist, set it to the mean of min and max
            opt_shape.append(
                min_max_opt.get("opt", int(min_max_opt["min"] + min_max_opt["max"] / 2))
            )
            max_shape.append(min_max_opt["max"])
        else:
            min_shape.append(dim)
            opt_shape.append(dim)
            max_shape.append(dim)

    return Input(
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        dtype=input_dtype,
        name=name,
        is_shape_tensor=is_shape_tensor,
    )


def get_input(
    input_shape: torch.Size,
    dtype: torch.dtype,
    name: str = "",
    is_shape_tensor: bool = False,
) -> Input:
    """
    Based on type of dimensions in the input_shape, construct regular or dynamic shaped inputs
    """
    if contains_sym_int(input_shape):
        return construct_dynamic_input(
            input_shape, dtype, name=name, is_shape_tensor=is_shape_tensor
        )
    else:
        return Input(
            shape=input_shape, dtype=dtype, name=name, is_shape_tensor=is_shape_tensor
        )


def construct_submodule_inputs(module: torch.fx.GraphModule) -> Sequence[Input]:
    """
    Construct torch_tensorrt Inputs based on the module inputs.
    The module inputs will have meta data which has the shape and dtype info
    Args:
        module: Input FX GraphModule
    Returns:
        Sequence of torch_tensorrt.Input's representing inputs to given module
    """
    with unset_fake_temporarily():
        torchtrt_inputs = []
        module_inputs = [
            node for node in module.graph.nodes if node.op == "placeholder"
        ]
        for input in module_inputs:
            if input.meta:
                if "val" in input.meta:
                    input_meta = input.meta["val"]
                    if isinstance(input_meta, (FakeTensor, torch.Tensor)):
                        input_shape = input_meta.size()
                        torchtrt_inputs.append(
                            get_input(input_shape, input_meta.dtype, name=input.name)
                        )
                    elif isinstance(input_meta, torch.SymInt):
                        # Assuming sym_integers | shape inputs always have torch.int64 dtype
                        torchtrt_inputs.append(
                            get_input(
                                [input_meta],
                                torch.int64,
                                name=input.name,
                                is_shape_tensor=True,
                            )
                        )
                    elif isinstance(input_meta, torch.SymFloat):
                        torchtrt_inputs.append(
                            get_input(
                                [1],
                                torch.float32,
                                name=input.name,
                                is_shape_tensor=False,  # Only SymInt inputs are treated as shape tensors
                            )
                        )
                    else:
                        raise ValueError(
                            f"The meta val for input node {input.target} is of type : {type(input_meta)}. Supported types: torch.Tensor|FakeTensor|torch.SymInt"
                        )

                elif "tensor_meta" in input.meta:
                    input_meta = input.meta["tensor_meta"]
                    input_shape = input_meta.shape
                    torchtrt_inputs.append(
                        get_input(input_shape, input_meta.dtype, name=input.name)
                    )
                else:
                    raise AssertionError(
                        f"Input {input.name} does not contain val and tensor_meta fields in the metadata. Please ensure you have exported the graph correctly"
                    )
            else:
                raise AssertionError(
                    f"Input {input.name} does not contain metadata. Please ensure you have exported the graph correctly"
                )

        return torchtrt_inputs


def run_shape_analysis(
    parent_module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    kwarg_inputs: Optional[dict[str, Any]] = None,
) -> Tuple[Dict[Any, Sequence[Any]], Dict[Any, Sequence[Any]]]:
    submod_inputs_shape_map: Dict[Any, Sequence[Any]] = {}
    submod_outputs_shape_map: Dict[Any, Sequence[Any]] = {}
    sub_inputs: Sequence[torch.Tensor] = []
    sub_outputs: Sequence[torch.Tensor] = []

    # Register a hook to capture IO shapes for submodules
    def get_submodule_io(
        self: Any, inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]
    ) -> None:
        nonlocal sub_inputs, sub_outputs
        sub_inputs = inputs
        sub_outputs = outputs
        return

    if kwarg_inputs is None:
        kwarg_inputs = {}
    # Iterate through submodules (both Torch and TRT) and store IO shapes
    for name, _ in parent_module.named_children():
        submodule = getattr(parent_module, name)
        handle = submodule.register_forward_hook(get_submodule_io)
        parent_module(*inputs, **kwarg_inputs)
        handle.remove()
        submod_inputs_shape_map[name] = (
            [input.shape for input in sub_inputs]
            if isinstance(sub_inputs, (tuple, list))
            else [sub_inputs.shape]
        )
        submod_outputs_shape_map[name] = (
            [output.shape for output in sub_outputs]
            if isinstance(sub_outputs, (tuple, list))
            else [sub_outputs.shape]
        )

    return submod_inputs_shape_map, submod_outputs_shape_map


def get_graph_converter_support(
    graph_module: torch.fx.GraphModule,
    torch_executed_ops: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Helper function to get converter support overview pre-partitioning

    Args:
        graph_module: FX GraphModule to determine support for
        verbose: Bool representing whether to print operator support
        torch_executed_ops: Collection of operations to run in Torch, regardless of converter coverage
    Returns:
        The number of supported call_function nodes in the graph
    """
    from ._global_partitioner import TorchTensorRTOperatorSupport

    # Instantiate operator support object and module dictionary
    op_support = TorchTensorRTOperatorSupport(torch_executed_ops=torch_executed_ops)
    module_dict = dict(graph_module.named_modules())

    number_of_supported_nodes = 0
    total_functional_nodes = 0

    # Iterate over all nodes in the graph, enumerating call_function nodes
    for node in graph_module.graph.nodes:
        if node.op == "call_function":
            total_functional_nodes += 1

            if op_support.is_node_supported(module_dict, node):
                number_of_supported_nodes += 1

    # Print node support overview prior to partitioning
    op_support.print_support_overview(print_node_support=True)

    return number_of_supported_nodes, total_functional_nodes
