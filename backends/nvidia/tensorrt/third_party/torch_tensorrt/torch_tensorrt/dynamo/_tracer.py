from __future__ import annotations

import logging
from inspect import signature
from typing import Any, Optional, Tuple, Union

import torch
from torch.export import Dim, export
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo.utils import get_torch_inputs, to_torch_device

logger = logging.getLogger(__name__)


def trace(
    mod: torch.nn.Module | torch.fx.GraphModule,
    inputs: Optional[Tuple[Any, ...]] = None,
    *,
    arg_inputs: Optional[Tuple[Any, ...]] = None,
    kwarg_inputs: Optional[dict[Any, Any]] = None,
    **kwargs: Any,
) -> torch.export.ExportedProgram:
    """Exports a ``torch.export.ExportedProgram`` from a ``torch.nn.Module`` or ``torch.fx.GraphModule`` specifically targeting being compiled with Torch-TensorRT

    Exports a ``torch.export.ExportedProgram`` from either a ``torch.nn.Module`` or torch.fx.GraphModule``. Runs specific operator decompositions geared towards
    compilation by Torch-TensorRT's dynamo frontend.

    Arguments:
        mod (torch.nn.Module | torch.fx.GraphModule): Source module to later be compiled by Torch-TensorRT's dynamo fronted
        inputs (Tuple[Any, ...]): List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]
    Keyword Arguments:
        arg_inputs (Tuple[Any, ...]): Same as inputs. Alias for better understanding with kwarg_inputs.
        kwarg_inputs (dict[Any, ...]): Optional, kwarg inputs to the module forward function.
        device (Union(torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch.device("cuda:0")

        debug (bool): Enable debuggable engine
        enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the graph easier to convert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
        **kwargs: Any,
    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT
    """

    # Set log level at the top of compilation (torch_tensorrt.dynamo)
    if arg_inputs is None and inputs is None:
        raise AssertionError("'arg_inputs' and 'inputs' should not both be None.")

    elif arg_inputs is not None and inputs is not None:
        raise AssertionError(
            "'arg_inputs' and 'inputs' should not be used at the same time."
        )
    arg_inputs = inputs or arg_inputs

    if kwarg_inputs is None:
        kwarg_inputs = {}

    device = to_torch_device(kwargs.get("device", default_device()))
    torch_arg_inputs = get_torch_inputs(arg_inputs, device)
    torch_kwarg_inputs = get_torch_inputs(kwarg_inputs, device)
    # Constructing dynamic shape list as a nested dict
    dynamic_shapes = get_dynamic_shapes_args(mod, arg_inputs)
    dynamic_shapes.update(get_dynamic_shapes_kwargs(kwarg_inputs))
    exp_program = export(
        mod,
        tuple(torch_arg_inputs),
        kwargs=torch_kwarg_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=kwargs.get("strict", False),
    )

    return exp_program


def get_dynamic_shapes_kwargs(inputs: Any) -> Union[dict[str, Any], list[Any]]:
    if isinstance(inputs, dict):
        dynamic_shapes_kwarg = {}
        for k, v in inputs.items():
            dynamic_shapes_kwarg[k] = get_dynamic_shapes_kwargs(v)
        return dynamic_shapes_kwarg

    elif isinstance(inputs, Input):
        return get_dynamic_shapes(inputs)

    elif isinstance(inputs, (list, tuple)):
        dynamic_shapes = []
        for input in inputs:
            dynamic_shapes.append(get_dynamic_shapes(input))
        return dynamic_shapes

    raise TypeError(f"Unknown type {type(inputs)}.")


def get_dynamic_shapes_args(mod: torch.nn.Module, inputs: Any) -> dict[str, Any]:
    # dynamic_shape is a dict and cannot work without keys. Here we use position argument name
    # in forward function as the name
    args = list(signature(mod.forward).parameters.keys())
    dynamic_shapes = {}
    for input, input_name in zip(inputs, args[: len(inputs)]):
        dynamic_shapes[input_name] = get_dynamic_shapes(input)
    return dynamic_shapes


def get_dynamic_shapes(input: Input) -> dict[Any, Any]:
    if not isinstance(input, Input):
        # If the input is torch.Tensor, no dynamic is needed. Return empty dict
        return {}
    else:
        dynamic_dims = {}
        if input.shape_mode == Input._ShapeMode.DYNAMIC:
            min_shape = input.shape["min_shape"]
            opt_shape = input.shape["opt_shape"]
            max_shape = input.shape["max_shape"]
            assert len(min_shape) == len(opt_shape) == len(max_shape)
            for dim in range(len(min_shape)):
                if min_shape[dim] == opt_shape[dim] == max_shape[dim]:
                    continue
                else:
                    dynamic_dims[dim] = Dim(
                        input.name + "_" + str(dim),
                        min=min_shape[dim],
                        max=max_shape[dim],
                    )
        return dynamic_dims
