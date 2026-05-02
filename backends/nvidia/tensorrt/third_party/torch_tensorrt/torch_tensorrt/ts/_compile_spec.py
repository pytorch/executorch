from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Union

import tensorrt as trt
import torch
import torch_tensorrt._C.ts as _ts_C
from torch_tensorrt import _C
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import DeviceType, EngineCapability, dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.ts._Device import TorchScriptDevice
from torch_tensorrt.ts._Input import TorchScriptInput
from torch_tensorrt.ts.logging import Level, log


def _internal_input_to_torch_class_input(i: _C.Input) -> torch.classes.tensorrt._Input:
    clone = torch.classes.tensorrt._Input()
    clone._set_min(i.min)
    clone._set_opt(i.opt)
    clone._set_max(i.max)
    clone._set_dtype(i.dtype)
    clone._set_tensor_domain(i.tensor_domain)
    clone._set_format(i.format)
    clone._set_input_is_dynamic(i.input_is_dynamic)
    clone._set_explicit_set_dtype(i._explicit_set_dtype)
    return clone


def _supported_input_size_type(input_size: Any) -> bool:
    if isinstance(input_size, torch.Size):
        return True
    elif isinstance(input_size, tuple):
        return True
    elif isinstance(input_size, list):
        return True
    else:
        raise TypeError(
            "Input sizes for inputs are required to be a List, tuple or torch.Size or a Dict of three sizes (min, opt, max), found type: "
            + str(type(input_size))
        )


def _parse_op_precision(precision: Any) -> _C.dtype:
    return dtype._from(precision).to(_C.dtype)


def _parse_enabled_precisions(precisions: Any) -> Set[_C.dtype]:
    parsed_precisions = set()
    if any(isinstance(precisions, type) for type in [list, tuple, set]):
        for p in precisions:
            parsed_precisions.add(_parse_op_precision(p))
    else:
        parsed_precisions.add(_parse_op_precision(precisions))
    return parsed_precisions


def _parse_device_type(device: Any) -> _C.DeviceType:
    return DeviceType._from(device).to(_C.DeviceType)


def _parse_device(device_info: Any) -> _C.Device:
    if isinstance(device_info, dict):
        info = _C.Device()
        if "device_type" not in device_info:
            raise KeyError("Device type is required parameter")
        else:
            info.device_type = _parse_device_type(device_info["device_type"])

        if "gpu_id" in device_info:
            assert isinstance(device_info["gpu_id"], int)
            info.gpu_id = device_info["gpu_id"]

        if "dla_core" in device_info:
            assert isinstance(device_info["dla_core"], int)
            info.dla_core = device_info["dla_core"]

        if "allow_gpu_fallback" in device_info:
            assert isinstance(device_info["allow_gpu_fallback"], bool)
            info.allow_gpu_fallback = device_info["allow_gpu_fallback"]

        return info
    elif isinstance(device_info, Device):
        return TorchScriptDevice._from(device_info)._to_internal()
    elif isinstance(device_info, TorchScriptDevice):
        return device_info._to_internal()
    elif isinstance(device_info, torch.device):
        return TorchScriptDevice._from(device_info)._to_internal()
    else:
        raise ValueError(
            "Unsupported data for device specification. Expected either a dict, torch_tensorrt.Device or torch.Device"
        )


def _parse_torch_fallback(fallback_info: Dict[str, Any]) -> _ts_C.TorchFallback:
    info = _ts_C.TorchFallback()
    if "enabled" not in fallback_info:
        raise KeyError("Enabled is required parameter")
    else:
        assert isinstance(fallback_info["enabled"], bool)
        info.enabled = fallback_info["enabled"]
    if "min_block_size" in fallback_info:
        assert isinstance(fallback_info["min_block_size"], int)
        info.min_block_size = fallback_info["min_block_size"]

    if "forced_fallback_ops" in fallback_info:
        assert isinstance(fallback_info["forced_fallback_ops"], list)
        info.forced_fallback_operators = fallback_info["forced_fallback_ops"]

    if "forced_fallback_modules" in fallback_info:
        assert isinstance(fallback_info["forced_fallback_modules"], list)
        info.forced_fallback_modules = fallback_info["forced_fallback_modules"]

    return info


def _parse_input_signature(input_signature: Any, depth: int = 0) -> Any:
    if depth > 2:
        raise AssertionError(
            "Input nesting depth exceeds max supported depth, use 1 level: [A, B], or 2 level: [A, (B, C)]"
        )

    if isinstance(input_signature, tuple):
        input_list = []
        for item in input_signature:
            input = _parse_input_signature(item, depth + 1)
            input_list.append(input)
        return tuple(input_list)
    elif isinstance(input_signature, list):
        input_list = []
        for item in input_signature:
            input = _parse_input_signature(item, depth + 1)
            input_list.append(input)
        return input_list
    elif isinstance(input_signature, (Input, torch.Tensor)):
        i = (
            Input.from_tensor(input_signature)
            if isinstance(input_signature, torch.Tensor)
            else input_signature
        )

        if not i.dtype.try_to(trt.DataType, use_default=True):
            raise TypeError(
                "Using non-TRT input types ({}) with input_signature is not currently ".format(
                    i.dtype
                )
                + "supported. Please specify inputs individually to use "
                + "non-TRT types."
            )

        ts_i = i
        if i.shape_mode == Input._ShapeMode.STATIC:
            ts_i = TorchScriptInput(shape=i.shape, dtype=i.dtype, format=i.format)
        elif i.shape_mode == Input._ShapeMode.DYNAMIC:
            if isinstance(i.shape, dict):
                ts_i = TorchScriptInput(
                    min_shape=i.shape["min_shape"],
                    opt_shape=i.shape["opt_shape"],
                    max_shape=i.shape["max_shape"],
                    dtype=i.dtype,
                    format=i.format,
                )
            else:
                raise ValueError(
                    f"Input set as dynamic, expected dictionary of shapes but found {i.shape}"
                )
        else:
            raise ValueError(
                "Invalid shape mode detected for input while parsing the input_signature"
            )

        clone = _internal_input_to_torch_class_input(ts_i._to_internal())
        return clone
    else:
        raise KeyError(
            "Input signature contains an unsupported type {}".format(
                type(input_signature)
            )
        )


def _parse_compile_spec(compile_spec_: Dict[str, Any]) -> _ts_C.CompileSpec:
    # TODO: Use deepcopy to support partial compilation of collections
    compile_spec = deepcopy(compile_spec_)
    info = _ts_C.CompileSpec()

    if len(compile_spec["inputs"]) > 0:
        if not all(
            isinstance(i, (torch.Tensor, Input)) for i in compile_spec["inputs"]
        ):
            raise KeyError(
                "Input specs should be either torch_tensorrt.Input or torch.Tensor, found types: {}".format(
                    [type(i) for i in compile_spec["inputs"]]
                )
            )

        inputs = [
            Input.from_tensor(i) if isinstance(i, torch.Tensor) else i
            for i in compile_spec["inputs"]
        ]
        ts_inputs = []
        for i in inputs:
            if i.shape_mode == Input._ShapeMode.STATIC:
                ts_inputs.append(
                    TorchScriptInput(
                        shape=i.shape,
                        dtype=i.dtype.to(_C.dtype),
                        format=i.format.to(_C.TensorFormat),
                    )._to_internal()
                )
            elif i.shape_mode == Input._ShapeMode.DYNAMIC:
                ts_inputs.append(
                    TorchScriptInput(
                        min_shape=i.shape["min_shape"],
                        opt_shape=i.shape["opt_shape"],
                        max_shape=i.shape["max_shape"],
                        dtype=i.dtype.to(_C.dtype),
                        format=i.format.to(_C.TensorFormat),
                    )._to_internal()
                )
        info.inputs = ts_inputs

    elif compile_spec["input_signature"] is not None:
        log(
            Level.Warning,
            "Input signature parsing is an experimental feature, behavior and APIs may change",
        )
        signature = _parse_input_signature(compile_spec["input_signature"])
        info.input_signature = _C.InputSignature(signature)  # py_object

    else:
        raise KeyError(
            'Module input definitions are required to compile module. Provide a list of torch_tensorrt.Input keyed to "inputs" in the compile spec'
        )

    if "enabled_precisions" in compile_spec:
        info.enabled_precisions = _parse_enabled_precisions(
            compile_spec["enabled_precisions"]
        )

    if "sparse_weights" in compile_spec:
        assert isinstance(compile_spec["sparse_weights"], bool)
        info.sparse_weights = compile_spec["sparse_weights"]

    if "disable_tf32" in compile_spec:
        assert isinstance(compile_spec["disable_tf32"], bool)
        info.disable_tf32 = compile_spec["disable_tf32"]

    if "refit" in compile_spec:
        assert isinstance(compile_spec["refit"], bool)
        info.refit = compile_spec["refit"]

    if "debug" in compile_spec:
        assert isinstance(compile_spec["debug"], bool)
        info.debug = compile_spec["debug"]

    if "allow_shape_tensors" in compile_spec:
        assert isinstance(compile_spec["allow_shape_tensors"], bool)
        info.allow_shape_tensors = compile_spec["allow_shape_tensors"]

    if "device" in compile_spec:
        info.device = _parse_device(compile_spec["device"])

    if "capability" in compile_spec:
        capability = EngineCapability._from(compile_spec["capability"]).to(
            _C.EngineCapability
        )
        info.capability = capability

    if "num_avg_timing_iters" in compile_spec:
        assert type(compile_spec["num_avg_timing_iters"]) is int
        info.num_avg_timing_iters = compile_spec["num_avg_timing_iters"]

    if "workspace_size" in compile_spec:
        assert type(compile_spec["workspace_size"]) is int
        info.workspace_size = compile_spec["workspace_size"]

    if "dla_sram_size" in compile_spec:
        assert type(compile_spec["dla_sram_size"]) is int
        info.dla_sram_size = compile_spec["dla_sram_size"]

    if "dla_local_dram_size" in compile_spec:
        assert type(compile_spec["dla_local_dram_size"]) is int
        info.dla_local_dram_size = compile_spec["dla_local_dram_size"]

    if "dla_global_dram_size" in compile_spec:
        assert type(compile_spec["dla_global_dram_size"]) is int
        info.dla_global_dram_size = compile_spec["dla_global_dram_size"]

    if "truncate_long_and_double" in compile_spec:
        assert type(compile_spec["truncate_long_and_double"]) is bool
        info.truncate_long_and_double = compile_spec["truncate_long_and_double"]

    if "torch_fallback" in compile_spec:
        info.torch_fallback = _parse_torch_fallback(compile_spec["torch_fallback"])

    log(Level.Debug, str(info))

    return info


def TensorRTCompileSpec(
    inputs: Optional[List[torch.Tensor | Input]] = None,
    input_signature: Optional[Any] = None,
    device: Optional[torch.device | Device] = None,
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
    allow_shape_tensors: bool = False,
) -> torch.classes.tensorrt.CompileSpec:
    """Utility to create a formatted spec dictionary for using the PyTorch TensorRT backend

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
        truncate_long_and_double (bool): Truncate weights provided in int64 or double (float64) to int32 and float32
        allow_shape_tensors: (Experimental) Allow aten::size to output shape tensors using IShapeLayer in TensorRT

      Returns:
        torch.classes.tensorrt.CompileSpec: List of methods and formatted spec objects to be provided to ``torch._C._jit_to_tensorrt``
    """

    compile_spec = {
        "inputs": inputs if inputs is not None else [],
        # "input_signature": input_signature,
        "device": Device._current_device() if device is None else device,
        "disable_tf32": disable_tf32,  # Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        "sparse_weights": sparse_weights,  # Enable sparsity for convolution and fully connected layers.
        "enabled_precisions": (
            enabled_precisions if enabled_precisions is not None else set()
        ),  # Enabling FP16 kernels
        "refit": refit,  # enable refit
        "debug": debug,  # enable debuggable engine
        "capability": capability,  # Restrict kernel selection to safe gpu kernels or safe dla kernels
        "num_avg_timing_iters": num_avg_timing_iters,  # Number of averaging timing iterations used to select kernels
        "workspace_size": workspace_size,  # Maximum size of workspace given to TensorRT
        "dla_sram_size": dla_sram_size,  # Fast software managed RAM used by DLA to communicate within a layer.
        "dla_local_dram_size": dla_local_dram_size,  # Host RAM used by DLA to share intermediate tensor data across operations
        "dla_global_dram_size": dla_global_dram_size,  # Host RAM used by DLA to store weights and metadata for execution
        "truncate_long_and_double": truncate_long_and_double,
        "allow_shape_tensors": allow_shape_tensors,
    }

    parsed_spec = _parse_compile_spec(compile_spec)

    backend_spec = torch.classes.tensorrt.CompileSpec()

    if input_signature is not None:
        raise ValueError(
            "Input signature parsing is not currently supported in the TorchScript backend integration"
        )

    for i in parsed_spec.inputs:
        clone = _internal_input_to_torch_class_input(i)
        backend_spec._append_input(clone)

    d = torch.classes.tensorrt._Device()
    d._set_device_type(int(parsed_spec.device.device_type))
    d._set_gpu_id(parsed_spec.device.gpu_id)
    d._set_dla_core(parsed_spec.device.dla_core)
    d._set_allow_gpu_fallback(parsed_spec.device.allow_gpu_fallback)

    if parsed_spec.torch_fallback.enabled:
        raise RuntimeError(
            "Partial module compilation is not currently supported via the PyTorch TensorRT backend. If you need partial compilation, use torch_tensorrt.compile"
        )

    torch_fallback = torch.classes.tensorrt._TorchFallback()
    torch_fallback._set_enabled(parsed_spec.torch_fallback.enabled)
    torch_fallback._set_min_block_size(parsed_spec.torch_fallback.min_block_size)
    torch_fallback._set_forced_fallback_operators(
        parsed_spec.torch_fallback.forced_fallback_operators
    )
    torch_fallback._set_forced_fallback_modules(
        parsed_spec.torch_fallback.forced_fallback_modules
    )

    backend_spec._set_device(d)
    backend_spec._set_torch_fallback(torch_fallback)
    backend_spec._set_precisions([int(i) for i in parsed_spec.enabled_precisions])

    backend_spec._set_disable_tf32(parsed_spec.disable_tf32)
    backend_spec._set_refit(parsed_spec.refit)
    backend_spec._set_debug(parsed_spec.debug)
    backend_spec._set_refit(parsed_spec.refit)
    backend_spec._set_capability(int(parsed_spec.capability))
    backend_spec._set_num_avg_timing_iters(parsed_spec.num_avg_timing_iters)
    backend_spec._set_workspace_size(parsed_spec.workspace_size)
    backend_spec._set_dla_sram_size(parsed_spec.dla_sram_size)
    backend_spec._set_dla_local_dram_size(parsed_spec.dla_local_dram_size)
    backend_spec._set_dla_global_dram_size(parsed_spec.dla_global_dram_size)
    backend_spec._set_truncate_long_and_double(parsed_spec.truncate_long_and_double)
    backend_spec._set_allow_shape_tensors(parsed_spec.allow_shape_tensors)

    return backend_spec
