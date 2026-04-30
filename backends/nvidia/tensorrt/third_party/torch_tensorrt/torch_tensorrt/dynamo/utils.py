from __future__ import annotations

import ctypes
import gc
import logging
import os
import platform
import warnings
from dataclasses import fields, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import psutil
import sympy
import tensorrt as trt
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt._Input import Input
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings

from packaging import version

from .types import TRTDataType

logger = logging.getLogger(__name__)

COSINE_THRESHOLD = 0.99
DYNAMIC_DIM = -1
RTOL = 5e-3
ATOL = 5e-3
CPU_DEVICE = "cpu"


class Frameworks(Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    TRT = "trt"


DataTypeEquivalence: Dict[
    TRTDataType, Dict[Frameworks, Union[TRTDataType, np.dtype, torch.dtype]]
] = {
    trt.int8: {
        Frameworks.NUMPY: np.int8,
        Frameworks.TORCH: torch.int8,
        Frameworks.TRT: trt.int8,
    },
    trt.int32: {
        Frameworks.NUMPY: np.int32,
        Frameworks.TORCH: torch.int32,
        Frameworks.TRT: trt.int32,
    },
    trt.int64: {
        Frameworks.NUMPY: np.int64,
        Frameworks.TORCH: torch.int64,
        Frameworks.TRT: trt.int64,
    },
    trt.float16: {
        Frameworks.NUMPY: np.float16,
        Frameworks.TORCH: torch.float16,
        Frameworks.TRT: trt.float16,
    },
    trt.float32: {
        Frameworks.NUMPY: np.float32,
        Frameworks.TORCH: torch.float32,
        Frameworks.TRT: trt.float32,
    },
    trt.bool: {
        Frameworks.NUMPY: bool,
        Frameworks.TORCH: torch.bool,
        Frameworks.TRT: trt.bool,
    },
}

if is_tensorrt_version_supported("7.0"):
    DataTypeEquivalence[trt.bool] = {
        Frameworks.NUMPY: np.bool_,
        Frameworks.TORCH: torch.bool,
        Frameworks.TRT: trt.bool,
    }


def unified_dtype_converter(
    dtype: Union[TRTDataType, torch.dtype, np.dtype], to: Frameworks
) -> Union[np.dtype, torch.dtype, TRTDataType]:
    """
    Convert TensorRT, Numpy, or Torch data types to any other of those data types.
    Args:
        dtype (TRTDataType, torch.dtype, np.dtype): A TensorRT, Numpy, or Torch data type.
        to (Frameworks): The framework to convert the data type to.
    Returns:
        The equivalent data type in the requested framework.
    """
    assert to in Frameworks, f"Expected valid Framework for translation, got {to}"
    trt_major_version = int(trt.__version__.split(".")[0])
    if dtype in (np.int8, torch.int8, trt.int8):
        return DataTypeEquivalence[trt.int8][to]
    elif trt_major_version >= 7 and dtype in (np.bool_, torch.bool, trt.bool):
        return DataTypeEquivalence[trt.bool][to]
    elif dtype in (np.int32, torch.int32, trt.int32):
        return DataTypeEquivalence[trt.int32][to]
    elif dtype in (np.int64, torch.int64, trt.int64):
        return DataTypeEquivalence[trt.int64][to]
    elif dtype in (np.float16, torch.float16, trt.float16):
        return DataTypeEquivalence[trt.float16][to]
    elif dtype in (np.float32, torch.float32, trt.float32):
        return DataTypeEquivalence[trt.float32][to]
    else:
        raise TypeError("%s is not a supported dtype" % dtype)


def deallocate_module(module: torch.fx.GraphModule, delete_module: bool = True) -> None:
    """
    This is a helper function to delete the instance of module. We first move it to CPU and then
    delete the object. This function ensures the GPU memory occupied by the module is released effectively after this call
    """
    module.to(CPU_DEVICE)
    if delete_module:
        del module
    torch.cuda.empty_cache()
    gc.collect()


def use_python_runtime_parser(use_python_runtime: Optional[bool] = None) -> bool:
    """Parses a user-provided input argument regarding Python runtime

    Automatically handles cases where the user has not specified a runtime (None)

    Returns True if the Python runtime should be used, False if the C++ runtime should be used
    """
    using_python_runtime = use_python_runtime
    reason = ""

    # Runtime was manually specified by the user
    if using_python_runtime is not None:
        reason = "as requested by user"
    # Runtime was not manually specified by the user, automatically detect runtime
    else:
        try:
            from torch_tensorrt.dynamo.runtime import TorchTensorRTModule  # noqa: F401

            using_python_runtime = False
            reason = "since C++ dependency was detected as present"
        except ImportError:
            using_python_runtime = True
            reason = "since import failed, C++ dependency not installed"

    logger.info(
        f"Using {'Python-only' if using_python_runtime else 'Default'} Torch-TRT Runtime ({reason})"
    )

    return using_python_runtime


def cosine_similarity(gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> float:
    gt_tensor = gt_tensor.flatten().to(torch.float32)
    pred_tensor = pred_tensor.flatten().to(torch.float32)
    if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
        if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
            return 1.0
    res_t = torch.nn.functional.cosine_similarity(
        gt_tensor, pred_tensor, dim=0, eps=1e-6
    )
    res: float = res_t.cpu().detach().item()

    return res


def input_is_dynamic(inputs: Sequence[Union[Input, torch.Tensor, FakeTensor]]) -> bool:
    """
    Return true if any of inputs have dynamic shapes. Supported types are torch_tensorrt.Input | torch.Tensor
    """
    for input in inputs:
        if isinstance(input, torch.Tensor):
            if contains_sym_int(input.shape):
                return True
        elif isinstance(input, Input):
            if input.shape_mode == Input._ShapeMode.DYNAMIC:
                return True
        else:
            raise AssertionError(
                f"Invalid input type ({type(input)}) found. Supported types are torch_tensorrt.Input | torch.Tensor"
            )

    return False


def get_torch_tensor(
    input: Input,
    device: torch.device,
    mode: str = "",
) -> Union[int, torch.Tensor]:
    if input.is_shape_tensor:
        # TODO: All the shape tensors we've encountered so far are plain integers.
        # Validate this assumption on more models.
        return input.shape["opt_shape"][0]

    if len(mode) > 0:
        return input.example_tensor(mode).to(device)
    else:
        return input.torch_tensor.to(device)


def get_torch_inputs(
    inputs: Sequence[Input] | Dict[str, Any],
    device: Union[Device, torch.device, str],
    mode: str = "",
) -> Sequence[Union[int, torch.Tensor]] | Dict[str, Union[int, torch.Tensor]]:
    """
    Return the torch_tensor from the Input object. If mode is set, this implies
    user is using dynamic shaped inputs and return the corresponding input based
    on the mode requested.
    """
    device = to_torch_device(device)

    if isinstance(inputs, dict):
        result_dict: Dict[str, Union[int, torch.Tensor]] = {}
        for k, v in inputs.items():
            if isinstance(v, (list, tuple, dict)):
                result_dict[k] = get_torch_inputs(v, device)
            elif isinstance(v, Input):
                result_dict[k] = get_torch_tensor(v, device, mode)
        return result_dict
    else:
        result_list: List[Union[int, torch.Tensor]] = []
        for input in inputs:
            if isinstance(input, Input):
                result_list.append(get_torch_tensor(input, device, mode))
            elif isinstance(input, torch.Tensor):
                result_list.append(input.to(device))
            else:
                raise AssertionError(f"Input type {type(input)} is not a valid type")
        return result_list


def get_model_device(module: torch.fx.GraphModule) -> torch.device:
    """
    Returns the device on which the module parameters exist.
    """
    device = None
    for parameter in list(module.parameters()):
        if isinstance(parameter, (torch.nn.parameter.Parameter, torch.Tensor)):
            return parameter.device

    for buffer in list(module.buffers()):
        if isinstance(buffer, (torch.Tensor)):
            return buffer.device

    if device is None:
        device = to_torch_device(default_device())

    return device


def set_log_level(parent_logger: Any, level: Any) -> None:
    """
    Sets the log level to the user provided level.
    This is used to set debug logging at a global level
    at entry points of tracing, dynamo and torch_compile compilation.
    And set log level for c++ torch trt logger if runtime is available.
    """
    if parent_logger:
        parent_logger.setLevel(level)

    if ENABLED_FEATURES.torch_tensorrt_runtime:
        if level == logging.DEBUG:
            log_level = trt.ILogger.Severity.VERBOSE
        elif level == logging.INFO:
            log_level = trt.ILogger.Severity.INFO
        elif level == logging.WARNING:
            log_level = trt.ILogger.Severity.WARNING
        elif level == logging.ERROR:
            log_level = trt.ILogger.Severity.ERROR
        elif level == logging.CRITICAL:
            log_level = trt.ILogger.Severity.INTERNAL_ERROR
        else:
            raise AssertionError(f"{level} is not valid log level")

        torch.ops.tensorrt.set_logging_level(int(log_level))


def prepare_inputs(
    inputs: Input | torch.Tensor | Sequence[Any] | Dict[Any, Any],
    disable_memory_format_check: bool = False,
) -> Any:
    """
    We take a nested group of torch.Tensors or scalars and convert them into torchtrt.Input's
    """
    # Any tensors created inside this call will be FakeTensors if it's inside a torch.compile session
    # So, we disable fake mode temporarily.
    with unset_fake_temporarily():
        if inputs is None:
            return None

        elif isinstance(inputs, Input):
            return inputs

        elif isinstance(inputs, (torch.Tensor, int, float, bool)):
            return Input.from_tensor(
                torch.tensor(inputs),
                disable_memory_format_check=disable_memory_format_check,
            )

        elif isinstance(inputs, (list, tuple)):
            torchtrt_input_list = []
            for input_obj in inputs:
                torchtrt_input = prepare_inputs(
                    input_obj, disable_memory_format_check=disable_memory_format_check
                )
                torchtrt_input_list.append(torchtrt_input)

            return (
                torchtrt_input_list
                if isinstance(inputs, list)
                else tuple(torchtrt_input_list)
            )

        elif isinstance(inputs, dict):
            torchtrt_inputs_dict: Dict[Any, Any] = dict()

            for key, input_obj in inputs.items():
                torchtrt_input = prepare_inputs(
                    input_obj, disable_memory_format_check=disable_memory_format_check
                )
                torchtrt_inputs_dict[key] = torchtrt_input

            return torchtrt_inputs_dict

        else:
            raise ValueError(
                f"Invalid input type {type(inputs)} encountered in the dynamo_compile input parsing. "
                + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
            )


def parse_complex_tensor_structs(
    inputs: Input | torch.Tensor | Sequence[Any] | Dict[Any, Any],
    attribute_to_extract: str,
    apply_fn: Callable[[Any], Any] = lambda x: x,
) -> Any:
    """Parses complex structures of Tensors and returns a mirrored structure
    Extracts key attributes of each singular element, while reconstructing the struct
    Optionally applies a function to each attribute before returning
    """
    if isinstance(inputs, (torch.Tensor, Input)):
        return apply_fn(getattr(inputs, attribute_to_extract, None))
    elif isinstance(inputs, (int, float, bool)):
        # inputs is a python scalar value
        inputs_torch = torch.tensor(inputs)
        return apply_fn(getattr(inputs_torch, attribute_to_extract, None))

    elif isinstance(inputs, (list, tuple)):
        torchtrt_input_list = []
        for input_obj in inputs:
            torchtrt_input = parse_complex_tensor_structs(
                input_obj, attribute_to_extract, apply_fn
            )
            torchtrt_input_list.append(torchtrt_input)

        return (
            torchtrt_input_list
            if isinstance(inputs, list)
            else tuple(torchtrt_input_list)
        )

    elif isinstance(inputs, dict):
        torchtrt_inputs_dict: Dict[Any, Any] = dict()

        for key, input_obj in inputs.items():
            torchtrt_input = parse_complex_tensor_structs(
                input_obj, attribute_to_extract, apply_fn
            )
            torchtrt_inputs_dict[key] = torchtrt_input

        return torchtrt_inputs_dict

    else:
        raise ValueError(
            f"Invalid input type {type(inputs)} encountered during Dynamo input parsing. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )


def contains_sym_int(tensor: torch.Tensor) -> bool:
    """
    Returns true if the given tensor has symbolic shape.
    """
    return any(isinstance(dim, torch.SymInt) for dim in tensor)


def extract_var_range_info(symbolic_integer: torch.SymInt) -> Dict[str, int]:
    """
    This function returns the min, max, opt values of a symbolic integer.
    """
    node = symbolic_integer.node
    expr = node.expr
    shape_env = node.shape_env
    # An expr can be a independent SymInt node (eg: s0 or s1) or a composition of them eg: (48*s0 or s0*s1).
    # In the case of expr which has symbolic computation, bound_sympy evaluates them.
    # https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy
    # expr.xreplace replaces the symbolic variables with their current values and computes the expression.
    var_range = shape_env.var_to_range.get(expr, None) or shape_env.bound_sympy(expr)
    var_val = (
        shape_env.var_to_val.get(expr, None)
        or shape_env.unbacked_var_to_val.get(expr, None)
        or expr.xreplace(shape_env.var_to_val)
    )
    assert var_range, var_val
    min_val, max_val = int(var_range.lower), int(var_range.upper)
    # Torchdynamo 0/1 specialization outlier
    min_val = 1 if min_val == 2 else min_val
    min_max_opt = {}
    min_max_opt["min"] = min_val
    min_max_opt["max"] = max_val
    if isinstance(var_val, (sympy.core.numbers.Integer, int)):
        min_max_opt["opt"] = int(var_val)

    return min_max_opt


def unwrap_tensor_shape(
    tensor: Union[torch.Tensor, FakeTensor, torch.SymInt], mode: Optional[str] = ""
) -> Sequence[Union[int, Tuple[int, int]]]:
    """
    This is a helper function used to print/return the shape of the tensor.
    For regular torch.tensor's, it returns the static shape.
    For symbolic tensors, eg:(1, s0, 4), this function returns [1, [min, max], 4]. The min
    and max correspond to the lower and upper values of s0 symbolic dimension.
    """
    tensor_shape: List[Union[int, Tuple[int, int]]] = []
    # for dimension in tensor.shape:
    if isinstance(tensor, int):
        tensor_shape.append(tensor)
    elif isinstance(tensor, torch.SymInt):
        min_max_opt = extract_var_range_info(tensor)
        if mode:
            tensor_shape.append(min_max_opt[mode])
        else:
            tensor_shape.append((min_max_opt["min"], min_max_opt["max"]))
    elif isinstance(tensor, torch.SymFloat):
        # SymFloats can be an input to graph sometimes. Although SymFloat is scalar value, we treat it as a 1D tensor throughout Torch-TRT codebase.
        tensor_shape.append(1)
    elif isinstance(tensor, (torch.Tensor, FakeTensor)):
        for dimension in tensor.shape:
            tensor_shape.extend(unwrap_tensor_shape(dimension, mode=mode))

    return tuple(tensor_shape)


def unwrap_tensor_dtype(tensor: Union[torch.Tensor, FakeTensor, torch.SymInt]) -> Any:
    """
    Returns the dtype of torch.tensor or FakeTensor. For symbolic integers, we return int64
    """
    if isinstance(tensor, (torch.Tensor, FakeTensor)):
        return tensor.dtype
    elif isinstance(tensor, (int, float, bool)):
        return torch.tensor(tensor).dtype
    elif isinstance(tensor, torch.SymInt):
        return torch.int64
    elif isinstance(tensor, torch.SymFloat):
        return torch.float32
    elif tensor is None:
        # Case where we explicitly pass one of the inputs to be None (eg: FLUX.1-dev)
        return None
    else:
        raise ValueError(f"Found invalid tensor type {type(tensor)}")


def get_graph_io_attrs(
    io_nodes: Sequence[torch.fx.Node], attr_type: str
) -> Sequence[Any]:
    """
    Returns a list of attributes (shapes or dtypes) of the I/O nodes
    """
    assert attr_type in ["shape", "dtype"]
    attr_fn = unwrap_tensor_shape if attr_type == "shape" else unwrap_tensor_dtype
    graph_io_attrs = []
    for node in io_nodes:
        if "val" in node.meta:
            metadata = node.meta["val"]
            if isinstance(metadata, (tuple, list)):
                for tensor in metadata:
                    graph_io_attrs.append(attr_fn(tensor))
            else:
                graph_io_attrs.append(attr_fn(metadata))

    return graph_io_attrs


def parse_graph_io(module: torch.fx.GraphModule, dryrun_tracker: Any) -> None:
    """
    Parse the graph I/O shape/dtype info for the whole graph and store in the dryrun tracker
    """
    # Parse inputs of the graph
    input_nodes = [node for node in module.graph.nodes if node.op == "placeholder"]
    input_shapes = get_graph_io_attrs(input_nodes, "shape")
    input_dtypes = get_graph_io_attrs(input_nodes, "dtype")
    dryrun_tracker.input_shapes = input_shapes
    dryrun_tracker.input_dtypes = input_dtypes

    # Parse outputs of the graph
    mark_output_nodes = [node for node in module.graph.nodes if node.op == "output"]
    output_nodes = []
    for node in mark_output_nodes:
        output_nodes.extend(node.all_input_nodes)
    output_shapes = get_graph_io_attrs(output_nodes, "shape")
    output_dtypes = get_graph_io_attrs(output_nodes, "dtype")
    dryrun_tracker.output_shapes = output_shapes
    dryrun_tracker.output_dtypes = output_dtypes


def to_torch_device(device: Optional[Union[Device, torch.device, str]]) -> torch.device:
    """Cast a device-type to torch.device

    Returns the corresponding torch.device
    """
    if isinstance(device, Device):
        return device.to(torch.device)

    elif isinstance(device, torch.device):
        return device

    elif device is None:
        return torch.device(torch.cuda.current_device())

    else:
        return torch.device(device)


def to_torch_tensorrt_device(
    device: Optional[Union[Device, torch.device, str]],
) -> Device:
    """Cast a device-type to torch_tensorrt.Device

    Returns the corresponding torch_tensorrt.Device
    """
    return Device._from(device)


def parse_dynamo_kwargs(
    kwargs: Any,
) -> Tuple[CompilationSettings, Optional[BaseEngineCache]]:
    """Parses the kwargs field of a Dynamo backend

    Args:
        kwargs: Keyword arguments dictionary provided to the backend
    Returns:
        CompilationSettings object with relevant kwargs
    """
    # Initialize an empty CompilationSettings object
    settings = CompilationSettings()

    # If the user specifies keyword args, overwrite those fields in settings
    # Validate all specified kwargs to ensure they are true fields of the dataclass
    #
    # Note: kwargs provided by torch.compile are wrapped in the "options" key
    if kwargs:
        if "options" in kwargs and len(kwargs) == 1:
            kwargs = kwargs["options"]

        if "truncate_long_and_double" in kwargs:
            if (
                "truncate_double" in kwargs
                and kwargs["truncate_double"] is not _defaults.TRUNCATE_DOUBLE
            ):
                raise ValueError(
                    'Provided configuration for "truncate_double" and deprecated API "truncate_long_and_double". '
                    'Please only use "truncate_double".'
                )
            else:
                kwargs["truncate_double"] = kwargs["truncate_long_and_double"]
                warnings.warn(
                    'Compiler option "truncate_long_and_double" is deprecated in favor of "truncate_double" as int64 is now natively supported. '
                    "This option will be removed in the next version.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                del kwargs[
                    "truncate_long_and_double"
                ]  # Remove deprecated key after handling

        valid_attrs = {attr.name for attr in fields(settings)}
        valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_attrs}
        settings = replace(settings, **valid_kwargs)

    # TODO: Remove once Dynamo precisions refactoring is complete
    if "enabled_precisions" in kwargs:
        enabled_precisions = {dtype._from(e) for e in kwargs["enabled_precisions"]}

        if len(enabled_precisions) == 0:
            logger.info(
                f"No precision specified, defaulting to {_defaults.ENABLED_PRECISION}"
            )
            enabled_precisions = _defaults.ENABLED_PRECISIONS

        settings.enabled_precisions = enabled_precisions

    # Parse input runtime specification
    settings.use_python_runtime = use_python_runtime_parser(settings.use_python_runtime)

    # Ensure device is a torch_tensorrt Device
    settings.device = to_torch_tensorrt_device(settings.device)

    # Check and update device settings
    if "device" not in kwargs:
        logger.info(
            f"Device not specified, using Torch default current device - cuda:{settings.device.gpu_id}. "
            "If this is incorrect, please specify an input device, via the device keyword."
        )

    # Ignore and warn about require_full_compilation flag
    if settings.require_full_compilation:
        logger.warning(
            "Detected require_full_compilation=True for a torch.compile run. "
            "This option has no effect in torch.compile."
        )
        settings.require_full_compilation = False

    # If cache_built_engines and reuse_cached_engines are True but custom_engine_cache is not provided,
    # then create a default disk engine cache

    engine_cache = None
    if kwargs.get("cache_built_engines") or kwargs.get("reuse_cached_engines"):
        if kwargs.get("custom_engine_cache") is not None:
            engine_cache = kwargs.get("custom_engine_cache")
        else:
            from torch_tensorrt.dynamo._engine_cache import DiskEngineCache

            engine_cache_dir = kwargs.get(
                "engine_cache_dir", _defaults.ENGINE_CACHE_DIR
            )
            engine_cache_size = kwargs.get(
                "engine_cache_size", _defaults.ENGINE_CACHE_SIZE
            )
            engine_cache = DiskEngineCache(engine_cache_dir, engine_cache_size)

    if kwargs.get("torch_executed_ops"):
        settings.torch_executed_ops = kwargs.get("torch_executed_ops")

    logger.info("Compilation Settings: %s\n", settings)

    return settings, engine_cache


def req_torch_version(min_torch_version: str = "2.dev") -> Callable[..., Any]:
    """
    Create a decorator which verifies the Torch version installed
    against a specified version range

    Args:
        min_torch_version (str): The minimum required Torch version
        for the decorated function to work properly

    Returns:
        A decorator which raises a descriptive error message if
        an unsupported Torch version is used
    """

    def nested_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        def function_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Parse minimum and current Torch versions
            min_version = version.parse(min_torch_version)
            current_version = version.parse(torch.__version__)

            if current_version < min_version:
                raise AssertionError(
                    f"Expected Torch version {min_torch_version} or greater, "
                    + f"when calling {f}. Detected version {torch.__version__}"
                )
            else:
                return f(*args, **kwargs)

        return function_wrapper

    return nested_decorator


def check_module_output(
    new_module: torch.fx.GraphModule,
    refitted_module: torch.fx.GraphModule,
    arg_inputs: Any,
    kwarg_inputs: Any = None,
) -> bool:
    old_outputs, new_outputs = refitted_module(*arg_inputs), new_module(
        *arg_inputs, **kwarg_inputs
    )
    if type(old_outputs) != type(new_outputs):
        logger.warning("The output types are different. Output check is skipped.")
        return True
    return check_output_equal(old_outputs, new_outputs)


def check_output_equal(
    output1: Any,
    output2: Any,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> bool:
    if type(output1) != type(output2):
        logger.warning(
            "The output types are different. Check_output_equal will always return false."
        )
        return False

    if isinstance(output1, torch.Tensor):
        if output1.shape != output2.shape:
            return False
        return torch.allclose(output1, output2, rtol, atol)  # type: ignore

    elif isinstance(output1, (tuple, list)):
        if len(output1) != len(output2):
            return False
        for a, b in zip(output1, output2):
            if not check_output_equal(a, b):
                return False
            return True

    elif isinstance(output1, dict):
        if output1.keys() != output2.keys():
            return False
        for a, b in zip(output1.values(), output2.values()):
            if not check_output_equal(a, b):
                return False
        return True

    logger.warning(
        "The output type is not supported to be checked. Check_output_equal will always return false."
    )
    return False


def get_flat_args_with_check(
    exported_program: torch.export.ExportedProgram,
    args: list[Any],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Flatten args, kwargs using pytree, then, check specs.

    Args:
        args: List[Any] original args passed to __call__
        kwargs: Dict[str, Any] original kwargs passed to __call

    Returns:
        A tuple of (flat_args, received_spec)
        flat_args is flattend args / kwargs
        received_spec is the pytree spec produced while flattening the
        tuple (args, kwargs)
    """
    import torch.utils._pytree as pytree
    from torch.export._tree_utils import reorder_kwargs

    in_spec = exported_program.call_spec.in_spec
    if in_spec is not None:
        kwargs = reorder_kwargs(kwargs, in_spec)
    flat_args_with_path, received_spec = pytree.tree_flatten_with_path((args, kwargs))
    flat_args = tuple(x[1] for x in flat_args_with_path)
    return flat_args, received_spec


def get_metadata(
    gm: torch.fx.GraphModule, target_op: torch._ops.OpOverload
) -> List[Any]:
    """
    Return the list which has the metadata of all the target_op nodes present in the graph.
    """
    return [node.meta for node in gm.graph.nodes if node.target == target_op]


def set_metadata(
    gm: torch.fx.GraphModule, target_op: torch._ops.OpOverload, metadata: List[Any]
) -> None:
    """
    Return the list which has the metadata of all the target_op nodes present in the graph.
    """
    target_nodes = [node for node in gm.graph.nodes if node.target == target_op]
    assert len(target_nodes) == len(metadata)
    for idx, node in enumerate(target_nodes):
        node.meta = metadata[idx]


def copy_metadata(match_and_replacements: List[Any]) -> None:
    """
    Copy the metadata from anchor node to the replacement node. This should be used
    if the anchor node is replaced with only a single replacement node i.e one-one replacement.
    """
    for match_and_replacement in match_and_replacements:
        anchor_node = match_and_replacement.nodes_map[match_and_replacement.anchor]
        assert (
            len(match_and_replacement.replacements) == 1
        ), "Found more than 1 replacements for the anchor node."
        replacement_node = match_and_replacement.replacements[0]
        replacement_node.meta = anchor_node.meta


def flatten_nodes(nodes: Any) -> List[torch.fx.node.Node]:
    ret = []
    if isinstance(nodes, torch.fx.node.Node):
        ret.append(nodes)
    elif isinstance(nodes, (tuple, list)):
        for node in nodes:
            ret.extend(flatten_nodes(node))
    else:
        raise ValueError(
            f"expect torch.fx.node.Node or a tuple/list of torch.fx.node.Node type, got unexpected types: {type(nodes)=}"
        )
    return ret


def get_output_metadata(
    gm: torch.fx.GraphModule,
) -> List[Any]:
    outputs = [node for node in gm.graph.nodes if node.op == "output"]
    assert len(outputs) > 0
    outputs = outputs[0].args
    nodes = flatten_nodes(outputs)
    assert len(nodes) > 0
    return [node.meta for node in nodes]


def get_output_dtypes(output: Any, truncate_double: bool = False) -> List[dtype]:
    output_dtypes = []
    if isinstance(output, torch.fx.node.Node):
        if "val" in output.meta:
            output_meta = output.meta["val"]
            if isinstance(output_meta, (FakeTensor, torch.Tensor)):
                if truncate_double and output_meta.dtype == torch.float64:
                    output_dtypes.append(dtype.float32)
                else:
                    output_dtypes.append(dtype._from(output_meta.dtype))
            elif isinstance(output_meta, torch.SymInt):
                output_dtypes.append(dtype.int64)
        elif "tensor_meta" in output.meta:
            output_meta = output.meta["tensor_meta"]
            output_dtypes.append(dtype._from(output_meta.dtype))
        else:
            raise ValueError(
                f"node.name={output.name}: metadata does not exist, expect metadata exists for each output node"
            )
    elif isinstance(output, (tuple, list)):
        for ele in output:
            output_dtypes.extend(get_output_dtypes(ele))
    else:
        raise ValueError(
            f"got unexpected type {type(output)}, expected type is a torch.fx.node.Node or a tuple/list of torch.fx.node.Node"
        )
    return output_dtypes


def get_cpu_memory_usage() -> Any:
    return psutil.Process().memory_info().rss / 1024 / 1024


def release_host_and_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    if (
        platform.system() == "Linux"
        and os.environ.get("TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM", "0") == "1"
    ):
        try:
            libc = ctypes.CDLL("libc.so.6")
            if libc.malloc_trim(0) != 1:
                logger.warning("Failed to release CPU memory.")
        except Exception:
            logger.warning("Failed to release CPU memory.")
