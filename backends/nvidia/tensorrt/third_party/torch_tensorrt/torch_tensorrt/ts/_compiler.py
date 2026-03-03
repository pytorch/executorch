from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch_tensorrt._C.ts as _C
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.ts._compile_spec import _parse_compile_spec, _parse_device


def compile(
    module: torch.jit.ScriptModule,
    inputs: Optional[Sequence[Input | torch.Tensor]] = None,
    input_signature: Optional[Tuple[Input | torch.Tensor | Sequence[Any]]] = None,
    device: Device = Device._current_device(),
    disable_tf32: bool = False,
    sparse_weights: bool = False,
    enabled_precisions: Optional[Set[Union[torch.dtype, dtype]]] = None,
    refit: bool = False,
    debug: bool = False,
    capability: EngineCapability = EngineCapability.STANDARD,
    num_avg_timing_iters: int = 1,
    workspace_size: int = 0,
    dla_sram_size: int = 1048576,
    dla_local_dram_size: int = 1073741824,
    dla_global_dram_size: int = 536870912,
    truncate_long_and_double: bool = False,
    require_full_compilation: bool = False,
    min_block_size: int = 3,
    torch_executed_ops: Optional[List[str]] = None,
    torch_executed_modules: Optional[List[str]] = None,
    allow_shape_tensors: bool = False,
) -> torch.jit.ScriptModule:
    """Compile a TorchScript module for NVIDIA GPUs using TensorRT

    Takes a existing TorchScript module and a set of settings to configure the compiler
    and will convert methods to JIT Graphs which call equivalent TensorRT engines

    Converts specifically the forward method of a TorchScript Module

    Arguments:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
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

        input_signature Union(List, Tuple, torch_tensorrt.Input, torch.Tensor): A formatted collection of input specifications for the module. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum to select device type. **This API should be considered beta-level stable and may change in the future** ::

                input_signature=([
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                ], torch.randn((1, 3, 224, 244))) # Use an example tensor and let torch_tensorrt infer settings for input #3
        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        refit (bool): Enable refitting
        debug (bool): Enable debuggable engine
        capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        truncate_long_and_double (bool): Truncate weights provided in int64 or double (float64) to int32 and float32
        require_full_compilation (bool): Require modules to be compiled end to end or return an error as opposed to returning a hybrid graph where operations that cannot be run in TensorRT are run in PyTorch
        min_block_size (int): The minimum number of contiguous TensorRT convertible operations in order to run a set of operations in TensorRT
        torch_executed_ops (List[str]): List of aten operators that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
        torch_executed_modules (List[str]): List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
        allow_shape_tensors: (Experimental) Allow aten::size to output shape tensors using IShapeLayer in TensorRT

    Returns:
        torch.jit.ScriptModule: Compiled TorchScript Module, when run it will execute via TensorRT
    """

    warnings.warn(
        'The torchscript frontend for Torch-TensorRT has been deprecated, please plan on porting to the dynamo frontend (torch_tensorrt.compile(..., ir="dynamo"). Torchscript will continue to be a supported deployment format via post compilation torchscript tracing, see: https://pytorch.org/TensorRT/user_guide/saving_models.html for more details',
        DeprecationWarning,
        stacklevel=2,
    )

    input_list = list(inputs) if inputs is not None else []
    enabled_precisions_set = (
        enabled_precisions if enabled_precisions is not None else set()
    )
    torch_executed_module_list = (
        torch_executed_modules if torch_executed_modules is not None else []
    )
    torch_executed_op_list = (
        torch_executed_ops if torch_executed_ops is not None else []
    )

    if isinstance(module, torch.jit.ScriptFunction):
        raise TypeError(
            "torch.jit.ScriptFunction currently is not directly supported, wrap the function in a module to compile"
        )

    if require_full_compilation and (
        len(torch_executed_module_list) > 0 or len(torch_executed_op_list) > 0
    ):
        raise ValueError(
            f"require_full_compilation is enabled however the list of modules and ops to run in torch is not empty. Found: torch_executed_ops: {torch_executed_ops}, torch_executed_modules: {torch_executed_modules}"
        )

    spec = {
        "inputs": input_list,
        "input_signature": input_signature,
        "device": device,
        "disable_tf32": disable_tf32,  # Force FP32 layers to use traditional as FP32 format
        "sparse_weights": sparse_weights,  # Enable sparsity for convolution and fully connected layers.
        "enabled_precisions": enabled_precisions_set,  # Enabling FP16 kernels
        "refit": refit,  # enable refit
        "debug": debug,  # enable debuggable engine
        "capability": capability,  # Restrict kernel selection to safe gpu kernels or safe dla kernels
        "num_avg_timing_iters": num_avg_timing_iters,  # Number of averaging timing iterations used to select kernels
        "workspace_size": workspace_size,  # Maximum size of workspace given to TensorRT
        "dla_sram_size": dla_sram_size,
        "dla_local_dram_size": dla_local_dram_size,
        "dla_global_dram_size": dla_global_dram_size,
        "truncate_long_and_double": truncate_long_and_double,
        "torch_fallback": {
            "enabled": not require_full_compilation,
            "forced_fallback_ops": torch_executed_op_list,
            "forced_fallback_modules": torch_executed_module_list,
            "min_block_size": min_block_size,
        },
        "allow_shape_tensors": allow_shape_tensors,
    }

    compiled_cpp_mod = _C.compile_graph(module._c, _parse_compile_spec(spec))
    compiled_module: torch.jit.ScriptModule = torch.jit._recursive.wrap_cpp_module(
        compiled_cpp_mod
    )
    return compiled_module


def convert_method_to_trt_engine(
    module: torch.jit.ScriptModule,
    method_name: str = "forward",
    inputs: Optional[Sequence[Input | torch.Tensor]] = None,
    device: Device = Device._current_device(),
    disable_tf32: bool = False,
    sparse_weights: bool = False,
    enabled_precisions: Optional[Set[Union[torch.dtype, dtype]]] = None,
    refit: bool = False,
    debug: bool = False,
    capability: EngineCapability = EngineCapability.STANDARD,
    num_avg_timing_iters: int = 1,
    workspace_size: int = 0,
    dla_sram_size: int = 1048576,
    dla_local_dram_size: int = 1073741824,
    dla_global_dram_size: int = 536870912,
    truncate_long_and_double: int = False,
    allow_shape_tensors: bool = False,
) -> bytes:
    """Convert a TorchScript module method to a serialized TensorRT engine

    Converts a specified method of a module to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``

    Keyword Args:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
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

        method_name (str): Name of method to convert
        input_signature Union(List, Tuple, torch_tensorrt.Input, torch.Tensor): A formatted collection of input specifications for the module. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum to select device type. **This API should be considered beta-level stable and may change in the future** ::

                input_signature=([
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                ], torch.randn((1, 3, 224, 244))) # Use an example tensor and let torch_tensorrt infer settings for input #3

        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        refit (bool): Enable refitting
        debug (bool): Enable debuggable engine
        capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        truncate_long_and_double (bool): Truncate weights provided in int64 or double (float64) to int32 and float32
        allow_shape_tensors: (Experimental) Allow aten::size to output shape tensors using IShapeLayer in TensorRT

    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    warnings.warn(
        'The torchscript frontend for Torch-TensorRT has been deprecated, please plan on porting to the dynamo frontend (torch_tensorrt.convert_method_to_trt_engine(..., ir="dynamo"). Torchscript will continue to be a supported deployment format via post compilation torchscript tracing, see: https://pytorch.org/TensorRT/user_guide/saving_models.html for more details',
        DeprecationWarning,
        stacklevel=2,
    )

    input_list = list(inputs) if inputs is not None else []
    enabled_precisions_set = (
        enabled_precisions if enabled_precisions is not None else {torch.float}
    )

    if isinstance(module, torch.jit.ScriptFunction):
        raise TypeError(
            "torch.jit.ScriptFunctions currently are not directly supported, wrap the function in a module to compile"
        )

    compile_spec = {
        "inputs": input_list,
        "device": device,
        "disable_tf32": disable_tf32,  # Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        "sparse_weights": sparse_weights,  # Enable sparsity for convolution and fully connected layers.
        "enabled_precisions": enabled_precisions_set,  # Enabling FP16 kernels
        "refit": refit,  # enable refit
        "debug": debug,  # enable debuggable engine
        "capability": capability,  # Restrict kernel selection to safe gpu kernels or safe dla kernels
        "num_avg_timing_iters": num_avg_timing_iters,  # Number of averaging timing iterations used to select kernels
        "workspace_size": workspace_size,  # Maximum size of workspace given to TensorRT
        "truncate_long_and_double": truncate_long_and_double,
        "allow_shape_tensors": allow_shape_tensors,
    }

    engine_str = _C.convert_graph_to_trt_engine(
        module._c, method_name, _parse_compile_spec(compile_spec)
    )

    import io

    with io.BytesIO() as engine_bytes:
        engine_bytes.write(engine_str)
        engine_bytearray = engine_bytes.getvalue()

    return engine_bytearray


def embed_engine_in_new_module(
    serialized_engine: bytes,
    input_binding_names: Optional[List[str]] = None,
    output_binding_names: Optional[List[str]] = None,
    device: Device = Device._current_device(),
) -> torch.jit.ScriptModule:
    """Takes a pre-built serialized TensorRT engine and embeds it within a TorchScript module

    Takes a pre-built serialied TensorRT engine (as bytes) and embeds it within a TorchScript module.
    Registers the forward method to execute the TensorRT engine with the function signature:

        forward(Tensor[]) -> Tensor[]

    TensorRT bindings either be explicitly specified using ``[in/out]put_binding_names`` or have names with the following format:
      - [symbol].[index in input / output array]
      ex.
      - [x.0, x.1, x.2] -> [y.0]

    Module can be save with engine embedded with torch.jit.save and moved / loaded according to torch_tensorrt portability rules

    Arguments:
        serialized_engine (bytearray): Serialized TensorRT engine from either torch_tensorrt or TensorRT APIs

    Keyword Arguments:
        input_binding_names (List[str]): List of names of TensorRT bindings in order to be passed to the encompassing PyTorch module
        output_binding_names (List[str]): List of names of TensorRT bindings in order that should be returned from the encompassing PyTorch module
        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device to run engine on. Must be compatible with engine provided. Default: Current active device
    Returns:
        torch.jit.ScriptModule: New TorchScript module with engine embedded
    """
    input_binding_name_list = (
        input_binding_names if input_binding_names is not None else []
    )
    output_binding_name_list = (
        output_binding_names if output_binding_names is not None else []
    )
    cpp_mod = _C.embed_engine_in_new_module(
        serialized_engine,
        _parse_device(device),
        input_binding_name_list,
        output_binding_name_list,
    )
    wrapped_mod: torch.jit.ScriptModule = torch.jit._recursive.wrap_cpp_module(cpp_mod)
    return wrapped_mod


def check_method_op_support(
    module: torch.jit.ScriptModule, method_name: str = "forward"
) -> bool:
    """Checks to see if a method is fully supported by torch_tensorrt

    Checks if a method of a TorchScript module can be compiled by torch_tensorrt, if not, a list of operators
    that are not supported are printed out and the function returns false, else true.

    Arguments:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``
        method_name (str): Name of method to check

    Returns:
        bool: True if supported Method
    """
    supported: bool = _C.check_method_op_support(module._c, method_name)
    return supported
