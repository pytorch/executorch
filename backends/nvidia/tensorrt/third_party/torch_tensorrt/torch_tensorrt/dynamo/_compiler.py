from __future__ import annotations

import collections.abc
import logging
import os
import platform
import warnings
from typing import Any, Collection, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.export import ExportedProgram
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt._features import needs_cross_compile
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import _defaults, partitioning
from torch_tensorrt.dynamo._DryRunTracker import (
    DryRunTracker,
    PerSubgraphData,
    dryrun_stats_display,
    parse_non_trt_nodes,
)
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache, DiskEngineCache
from torch_tensorrt.dynamo._exporter import replace_execute_engine_no_op_node
from torch_tensorrt.dynamo.conversion import (
    CompilationSettings,
    UnsupportedOperatorException,
    convert_module,
    interpret_module_to_result,
    repair_double_inputs,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.debug._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo.debug._supports_debugger import fn_supports_debugger
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.partitioning._resource_partitioner import (
    resource_partition,
)
from torch_tensorrt.dynamo.utils import (
    deallocate_module,
    get_cpu_memory_usage,
    get_flat_args_with_check,
    get_output_metadata,
    parse_graph_io,
    prepare_inputs,
    to_torch_device,
    to_torch_tensorrt_device,
)

logger = logging.getLogger(__name__)


@needs_cross_compile  # type: ignore[misc]
def cross_compile_for_windows(
    exported_program: ExportedProgram,
    inputs: Optional[Sequence[Sequence[Any]]] = None,
    *,
    arg_inputs: Optional[Sequence[Sequence[Any]]] = None,
    kwarg_inputs: Optional[dict[Any, Any]] = None,
    device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
    disable_tf32: bool = _defaults.DISABLE_TF32,
    assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
    sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
    enabled_precisions: Union[
        Set[Union[torch.dtype, dtype]], Tuple[Union[torch.dtype, dtype]]
    ] = _defaults.ENABLED_PRECISIONS,
    engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
    num_avg_timing_iters: int = _defaults.NUM_AVG_TIMING_ITERS,
    workspace_size: int = _defaults.WORKSPACE_SIZE,
    dla_sram_size: int = _defaults.DLA_SRAM_SIZE,
    dla_local_dram_size: int = _defaults.DLA_LOCAL_DRAM_SIZE,
    dla_global_dram_size: int = _defaults.DLA_GLOBAL_DRAM_SIZE,
    truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
    require_full_compilation: bool = _defaults.REQUIRE_FULL_COMPILATION,
    min_block_size: int = _defaults.MIN_BLOCK_SIZE,
    torch_executed_ops: Optional[Collection[Target]] = None,
    torch_executed_modules: Optional[List[str]] = None,
    pass_through_build_failures: bool = _defaults.PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = _defaults.MAX_AUX_STREAMS,
    version_compatible: bool = _defaults.VERSION_COMPATIBLE,
    optimization_level: Optional[int] = _defaults.OPTIMIZATION_LEVEL,
    use_python_runtime: bool = _defaults.USE_PYTHON_RUNTIME,
    use_fast_partitioner: bool = _defaults.USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    dryrun: bool = _defaults.DRYRUN,
    hardware_compatible: bool = _defaults.HARDWARE_COMPATIBLE,
    timing_cache_path: str = _defaults.TIMING_CACHE_PATH,
    lazy_engine_init: bool = _defaults.LAZY_ENGINE_INIT,
    cache_built_engines: bool = _defaults.CACHE_BUILT_ENGINES,
    reuse_cached_engines: bool = _defaults.REUSE_CACHED_ENGINES,
    engine_cache_dir: str = _defaults.ENGINE_CACHE_DIR,
    engine_cache_size: int = _defaults.ENGINE_CACHE_SIZE,
    custom_engine_cache: Optional[BaseEngineCache] = _defaults.CUSTOM_ENGINE_CACHE,
    use_explicit_typing: bool = _defaults.USE_EXPLICIT_TYPING,
    use_fp32_acc: bool = _defaults.USE_FP32_ACC,
    refit_identical_engine_weights: bool = _defaults.REFIT_IDENTICAL_ENGINE_WEIGHTS,
    strip_engine_weights: bool = _defaults.STRIP_ENGINE_WEIGHTS,
    immutable_weights: bool = _defaults.IMMUTABLE_WEIGHTS,
    enable_weight_streaming: bool = _defaults.ENABLE_WEIGHT_STREAMING,
    tiling_optimization_level: str = _defaults.TILING_OPTIMIZATION_LEVEL,
    l2_limit_for_tiling: int = _defaults.L2_LIMIT_FOR_TILING,
    offload_module_to_cpu: bool = _defaults.OFFLOAD_MODULE_TO_CPU,
    use_distributed_mode_trace: bool = _defaults.USE_DISTRIBUTED_MODE_TRACE,
    enable_resource_partitioning: bool = _defaults.ENABLE_RESOURCE_PARTITIONING,
    cpu_memory_budget: Optional[int] = _defaults.CPU_MEMORY_BUDGET,
    **kwargs: Any,
) -> torch.fx.GraphModule:
    """Compile an ExportedProgram module using TensorRT in Linux for Inference in Windows

    Takes an exported program and a set of settings to configure the compiler
    and it will convert methods to AOT graphs which call equivalent TensorRT engines

    Arguments:
        exported_program (torch.export.ExportedProgram): Source module, running torch.export on a ``torch.nn.Module``
        inputs (Tuple[Any, ...]): List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type.

                .. code-block:: py

                    inputs=[
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
        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        assume_dynamic_shape_support (bool): Setting this to true enables the converters work for both dynamic and static shapes. Default: False
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precisions (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        truncate_double (bool): Truncate weights provided in double (float64) to float32
        require_full_compilation (bool): Require modules to be compiled end to end or return an error as opposed to returning a hybrid graph where operations that cannot be run in TensorRT are run in PyTorch
        min_block_size (int): The minimum number of contiguous TensorRT convertible operations in order to run a set of operations in TensorRT
        torch_executed_ops (Collection[Target]): Set of aten operators that must be run in PyTorch. An error will be thrown if this set is not empty but ``require_full_compilation`` is True
        torch_executed_modules (List[str]): List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
        pass_through_build_failures (bool): Error out if there are issues during compilation (only applicable to torch.compile workflows)
        max_aux_stream (Optional[int]): Maximum streams in the engine
        version_compatible (bool): Build the TensorRT engines compatible with future versions of TensorRT (Restrict to lean runtime operators to provide version forward compatibility for the engines)
        optimization_level: (Optional[int]): Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a lower optimization level. The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level, which is currently 5. Setting it to be greater than the maximum level results in identical behavior to the maximum level.
        use_python_runtime: (bool): Return a graph using a pure Python runtime, reduces options for serialization
        use_fast_partitioner: (bool): Use the adjacency based partitioning scheme instead of the global partitioner. Adjacency partitioning is faster but may not be optimal. Use the global paritioner (``False``) if looking for best performance
        enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the graph easier to convert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
        dryrun (bool): Toggle for "Dryrun" mode, running everything except conversion to TRT and logging outputs
        hardware_compatible (bool): Build the TensorRT engines compatible with GPU architectures other than that of the GPU on which the engine was built (currently works for NVIDIA Ampere and newer)
        timing_cache_path (str): Path to the timing cache if it exists (or) where it will be saved after compilation
        lazy_engine_init (bool): Defer setting up engines until the compilation of all engines is complete. Can allow larger models with multiple graph breaks to compile but can lead to oversubscription of GPU memory at runtime.
        cache_built_engines (bool): Whether to save the compiled TRT engines to storage
        reuse_cached_engines (bool): Whether to load the compiled TRT engines from storage
        engine_cache_dir (Optional[str]): Directory to store the cached TRT engines
        engine_cache_size (Optional[int]): Maximum hard-disk space (bytes) to use for the engine cache, default is 1GB. If the cache exceeds this size, the oldest engines will be removed by default
        custom_engine_cache (Optional[BaseEngineCache]): Engine cache instance to use for saving and loading engines. Users can provide their own engine cache by inheriting from BaseEngineCache. If used, engine_cache_dir and engine_cache_size will be ignored.
        use_explicit_typing (bool): This flag enables strong typing in TensorRT compilation which respects the precisions set in the Pytorch model. This is useful when users have mixed precision graphs.
        use_fp32_acc (bool): This option inserts cast to FP32 nodes around matmul layers and TensorRT ensures the accumulation of matmul happens in FP32. Use this only when FP16 precision is configured in enabled_precisions.
        refit_identical_engine_weights (bool): Refit engines with identical weights. This is useful when the same model is compiled multiple times with different inputs and the weights are the same. This will save time by reusing the same engine for different inputs.
        strip_engine_weights (bool): Strip engine weights from the serialized engine. This is useful when the engine is to be deployed in an environment where the weights are not required.
        immutable_weights (bool): Build non-refittable engines. This is useful for some layers that are not refittable. If this argument is set to true, `strip_engine_weights` and `refit_identical_engine_weights` will be ignored.
        enable_weight_streaming (bool): Enable weight streaming.
        tiling_optimization_level (str): The optimization level of tiling strategies. A higher level allows TensorRT to spend more time searching for better tiling strategy. We currently support ["none", "fast", "moderate", "full"].
        l2_limit_for_tiling (int): The target L2 cache usage limit (in bytes) for tiling optimization (default is -1 which means no limit).
        use_distributed_mode_trace (bool):  Using aot_autograd to trace the graph. This is enabled when DTensors or distributed tensors are present in distributed model
        enable_resource_partitioning (bool): Enable resource-aware partitioning. This is useful when the model is large and the CPU memory is limited.
        cpu_memory_budget (Optional[int]): The maximum amount of CPU memory to use for the compilation. If the compilation requires more memory than this budget, the compilation will fail.
        **kwargs: Any,
    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT

    """
    if platform.system() != "Linux" or platform.architecture()[0] != "64bit":
        raise RuntimeError(
            f"Cross compile for windows is only supported on x86-64 Linux architecture, current platform: {platform.system()=}, {platform.architecture()[0]=}"
        )

    if kwargs.get("debug", False):
        warnings.warn(
            "`debug` is deprecated. Please use `with torch_tensorrt.dynamo.Debugger(...)` to wrap your compilation call to enable debugging functionality.",
            DeprecationWarning,
            stacklevel=2,
        )

    if "truncate_long_and_double" in kwargs.keys():
        if truncate_double is not _defaults.TRUNCATE_DOUBLE:
            raise ValueError(
                'Provided configuration for "truncate_double" and deprecated API "truncate_long_and_double", please only use "truncate_double"'
            )
        else:
            truncate_double = kwargs["truncate_long_and_double"]
            warnings.warn(
                'Compiler option "truncate_long_and_double" is deprecated in favor of "truncate_double" as int64 is now natively supported, this option will be removed in the next version',
                DeprecationWarning,
                stacklevel=2,
            )

    if "refit" in kwargs.keys():
        warnings.warn(
            "`refit` is deprecated. Please set `immutable_weights=False` to build a refittable engine whose weights can be refitted.",
            DeprecationWarning,
            stacklevel=2,
        )
        if immutable_weights:
            raise ValueError(
                "Use flag `immutable_weights` only. Flag `refit` is deprecated."
            )
        else:
            immutable_weights = not kwargs["refit"]

    if "make_refittable" in kwargs.keys():
        warnings.warn(
            "`make_refittable` is deprecated. Please set `immutable_weights=False` to build a refittable engine whose weights can be refitted",
            DeprecationWarning,
            stacklevel=2,
        )
        if immutable_weights:
            raise ValueError(
                "Use flag `immutable_weights` only. Flag `make_refittable` is deprecated."
            )
        else:
            immutable_weights = not kwargs["make_refittable"]

    if refit_identical_engine_weights:
        if immutable_weights:
            raise ValueError(
                "`immutable_weights` must be False when `refit_identical_engine_weights` is True."
            )

    engine_capability = EngineCapability._from(engine_capability)

    if torch_executed_modules is not None and torch_executed_modules:
        logger.warning(
            f"Detected torch_executed_modules was non-empty: {torch_executed_modules}"
            "\nThis feature is unimplemented in Torch-TRT Dynamo currently."
        )

    if use_explicit_typing:
        if len(enabled_precisions) != 1 or not any(
            x in enabled_precisions
            for x in {torch.float32, dtype.f32, torch.float4_e2m1fn_x2, dtype.f4}
        ):
            raise AssertionError(
                f"use_explicit_typing was set to True, however found that enabled_precisions was also specified (saw: {enabled_precisions}, expected: dtype.f32, dtype.f4). enabled_precisions should not be used when use_explicit_typing=True"
            )

    if use_fp32_acc:
        logger.debug(
            "FP32 accumulation for matmul layers is enabled. This option should only be enabled if the model already has FP16 weights and has no effect if it has FP32 weights. \
                     This flag inserts casts around matmul layers and ensures TensorRT executes the matmul layers in FP16 with FP32 accumulation."
        )

    if enable_weight_streaming and not use_explicit_typing:
        raise AssertionError(
            "When enable_weight_streaming is enabled, it requires use_explicit_typing to be set to True"
        )
    # Aliasing inputs to arg_inputs for better understanding
    if arg_inputs is None and kwarg_inputs is None and inputs is None:
        raise AssertionError(
            "'arg_inputs', 'kwarg_inputs' and 'inputs' should not all be None."
        )

    elif arg_inputs is not None and inputs is not None:
        raise AssertionError(
            "'arg_inputs' and 'inputs' should not be used at the same time."
        )

    arg_inputs = inputs or arg_inputs

    if kwarg_inputs is None:
        kwarg_inputs = {}

    if not isinstance(arg_inputs, collections.abc.Sequence):
        arg_inputs = [arg_inputs]  # type: ignore

    # Prepare torch_trt inputs
    trt_arg_inputs: Sequence[Input] = prepare_inputs(arg_inputs)
    trt_kwarg_inputs: Optional[dict[Any, Any]] = prepare_inputs(kwarg_inputs)
    device = to_torch_tensorrt_device(device)
    enabled_precisions = {dtype._from(p) for p in enabled_precisions}

    compilation_options = {
        "enabled_precisions": (
            enabled_precisions if enabled_precisions else _defaults.ENABLED_PRECISIONS
        ),
        "device": device,
        "assume_dynamic_shape_support": assume_dynamic_shape_support,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": (
            torch_executed_ops if torch_executed_ops is not None else set()
        ),
        "pass_through_build_failures": pass_through_build_failures,
        "max_aux_streams": max_aux_streams,
        "version_compatible": version_compatible,
        "optimization_level": optimization_level,
        "use_python_runtime": False,
        "truncate_double": truncate_double,
        "use_fast_partitioner": use_fast_partitioner,
        "num_avg_timing_iters": num_avg_timing_iters,
        "enable_experimental_decompositions": enable_experimental_decompositions,
        "require_full_compilation": require_full_compilation,
        "disable_tf32": disable_tf32,
        "sparse_weights": sparse_weights,
        "engine_capability": engine_capability,
        "dla_sram_size": dla_sram_size,
        "dla_local_dram_size": dla_local_dram_size,
        "dla_global_dram_size": dla_global_dram_size,
        "dryrun": dryrun,
        "hardware_compatible": hardware_compatible,
        "timing_cache_path": timing_cache_path,
        "lazy_engine_init": lazy_engine_init,
        "cache_built_engines": cache_built_engines,
        "reuse_cached_engines": reuse_cached_engines,
        "refit_identical_engine_weights": refit_identical_engine_weights,
        "strip_engine_weights": strip_engine_weights,
        "immutable_weights": immutable_weights,
        "enable_cross_compile_for_windows": True,
        "enable_weight_streaming": enable_weight_streaming,
        "tiling_optimization_level": tiling_optimization_level,
        "l2_limit_for_tiling": l2_limit_for_tiling,
        "use_distributed_mode_trace": use_distributed_mode_trace,
        "enable_resource_partitioning": enable_resource_partitioning,
        "cpu_memory_budget": cpu_memory_budget,
    }

    # disable the following settings is not supported for cross compilation for windows feature
    unsupported_settings = (
        "use_python_runtime",
        "lazy_engine_init",
        "cache_built_engines",
        "reuse_cached_engines",
    )
    # disable these settings if anything is turned on
    for key, value in compilation_options.items():
        if key in unsupported_settings and value:
            compilation_options[key] = False
            logger.warning(
                f"arg: {key} is not supported for cross compilation for windows feature, hence it is disabled."
            )

    settings = CompilationSettings(**compilation_options)
    logger.info("Compilation Settings: %s\n", settings)
    exported_program = pre_export_lowering(exported_program, settings)
    exported_program = exported_program.run_decompositions(
        get_decompositions(enable_experimental_decompositions)
    )

    gm = exported_program.module()
    logger.debug("Input graph: " + str(gm.graph))

    # Apply lowering on the graph module
    gm = post_lowering(gm, settings)
    logger.debug("Lowered Input graph: " + str(gm.graph))
    # Move the weights in the state_dict to CPU
    if offload_module_to_cpu:
        deallocate_module(exported_program.module(), delete_module=False)
        logger.info(
            "The PyTorch model was moved to the CPU to allocate all GPU memory to TensorRT. To retain the model on the GPU, set offload_module_to_cpu=False"
        )
    else:
        remaining_memory, total_memory = torch.cuda.mem_get_info()
        if remaining_memory < total_memory // 2:
            logger.warning(
                "Remaining GPU memory may not be enough to compile the TensorRT engine for this model resulting in an OOM error, Consider setting offload_module_to_cpu=True"
            )
    trt_gm = compile_module(
        gm,
        trt_arg_inputs,
        trt_kwarg_inputs,
        settings,
    )
    return trt_gm


def compile(
    exported_program: ExportedProgram,
    inputs: Optional[Sequence[Sequence[Any]]] = None,
    *,
    arg_inputs: Optional[Sequence[Sequence[Any]]] = None,
    kwarg_inputs: Optional[dict[Any, Any]] = None,
    device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
    disable_tf32: bool = _defaults.DISABLE_TF32,
    assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
    sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
    enabled_precisions: Union[
        Set[Union[torch.dtype, dtype]], Tuple[Union[torch.dtype, dtype]]
    ] = _defaults.ENABLED_PRECISIONS,
    engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
    num_avg_timing_iters: int = _defaults.NUM_AVG_TIMING_ITERS,
    workspace_size: int = _defaults.WORKSPACE_SIZE,
    dla_sram_size: int = _defaults.DLA_SRAM_SIZE,
    dla_local_dram_size: int = _defaults.DLA_LOCAL_DRAM_SIZE,
    dla_global_dram_size: int = _defaults.DLA_GLOBAL_DRAM_SIZE,
    truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
    require_full_compilation: bool = _defaults.REQUIRE_FULL_COMPILATION,
    min_block_size: int = _defaults.MIN_BLOCK_SIZE,
    torch_executed_ops: Optional[Collection[Target]] = None,
    torch_executed_modules: Optional[List[str]] = None,
    pass_through_build_failures: bool = _defaults.PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = _defaults.MAX_AUX_STREAMS,
    version_compatible: bool = _defaults.VERSION_COMPATIBLE,
    optimization_level: Optional[int] = _defaults.OPTIMIZATION_LEVEL,
    use_python_runtime: bool = _defaults.USE_PYTHON_RUNTIME,
    use_fast_partitioner: bool = _defaults.USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    dryrun: bool = _defaults.DRYRUN,
    hardware_compatible: bool = _defaults.HARDWARE_COMPATIBLE,
    timing_cache_path: str = _defaults.TIMING_CACHE_PATH,
    lazy_engine_init: bool = _defaults.LAZY_ENGINE_INIT,
    cache_built_engines: bool = _defaults.CACHE_BUILT_ENGINES,
    reuse_cached_engines: bool = _defaults.REUSE_CACHED_ENGINES,
    engine_cache_dir: str = _defaults.ENGINE_CACHE_DIR,
    engine_cache_size: int = _defaults.ENGINE_CACHE_SIZE,
    custom_engine_cache: Optional[BaseEngineCache] = _defaults.CUSTOM_ENGINE_CACHE,
    use_explicit_typing: bool = _defaults.USE_EXPLICIT_TYPING,
    use_fp32_acc: bool = _defaults.USE_FP32_ACC,
    refit_identical_engine_weights: bool = _defaults.REFIT_IDENTICAL_ENGINE_WEIGHTS,
    strip_engine_weights: bool = _defaults.STRIP_ENGINE_WEIGHTS,
    immutable_weights: bool = _defaults.IMMUTABLE_WEIGHTS,
    enable_weight_streaming: bool = _defaults.ENABLE_WEIGHT_STREAMING,
    tiling_optimization_level: str = _defaults.TILING_OPTIMIZATION_LEVEL,
    l2_limit_for_tiling: int = _defaults.L2_LIMIT_FOR_TILING,
    offload_module_to_cpu: bool = _defaults.OFFLOAD_MODULE_TO_CPU,
    use_distributed_mode_trace: bool = _defaults.USE_DISTRIBUTED_MODE_TRACE,
    enable_autocast: bool = _defaults.ENABLE_AUTOCAST,
    autocast_low_precision_type: Optional[
        Union[torch.dtype, dtype]
    ] = _defaults.AUTOCAST_LOW_PRECISION_TYPE,
    autocast_excluded_nodes: Collection[str] = _defaults.AUTOCAST_EXCLUDED_NODES,
    autocast_excluded_ops: Collection[Target] = _defaults.AUTOCAST_EXCLUDED_OPS,
    autocast_max_output_threshold: float = _defaults.AUTOCAST_MAX_OUTPUT_THRESHOLD,
    autocast_max_depth_of_reduction: Optional[
        int
    ] = _defaults.AUTOCAST_MAX_DEPTH_OF_REDUCTION,
    autocast_calibration_dataloader: Optional[
        torch.utils.data.DataLoader
    ] = _defaults.AUTOCAST_CALIBRATION_DATALOADER,
    cpu_memory_budget: Optional[int] = _defaults.CPU_MEMORY_BUDGET,
    enable_resource_partitioning: bool = _defaults.ENABLE_RESOURCE_PARTITIONING,
    **kwargs: Any,
) -> torch.fx.GraphModule:
    """Compile an ExportedProgram module for NVIDIA GPUs using TensorRT

    Takes a existing TorchScript module and a set of settings to configure the compiler
    and will convert methods to JIT Graphs which call equivalent TensorRT engines

    Converts specifically the forward method of a TorchScript Module

    Arguments:
        exported_program (torch.export.ExportedProgram): Source module, running torch.export on a ``torch.nn.Module``
        inputs (Optional[Sequence[Sequence[Any]]]): List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type.

                .. code-block:: py

                    inputs=[
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
        arg_inputs (Optional[Sequence[Sequence[Any]]]): Same as inputs. Alias for better understanding with kwarg_inputs.
        kwarg_inputs (Optional[dict[Any, Any]]): kwarg inputs to the module forward function.
        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        assume_dynamic_shape_support (bool): Setting this to true enables the converters work for both dynamic and static shapes. Default: False
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precisions (Union[Set[Union[torch.dtype, dtype]], Tuple[Union[torch.dtype, dtype]]]): The set of datatypes that TensorRT can use when selecting kernels
        engine_capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        truncate_double (bool): Truncate weights provided in double (float64) to float32
        require_full_compilation (bool): Require modules to be compiled end to end or return an error as opposed to returning a hybrid graph where operations that cannot be run in TensorRT are run in PyTorch
        min_block_size (int): The minimum number of contiguous TensorRT convertible operations in order to run a set of operations in TensorRT
        torch_executed_ops (Optional[Collection[Target]]): Set of aten operators that must be run in PyTorch. An error will be thrown if this set is not empty but ``require_full_compilation`` is True
        torch_executed_modules (Optional[List[str]]): List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
        pass_through_build_failures (bool): Error out if there are issues during compilation (only applicable to torch.compile workflows)
        max_aux_streams (Optional[int]): Maximum streams in the engine
        version_compatible (bool): Build the TensorRT engines compatible with future versions of TensorRT (Restrict to lean runtime operators to provide version forward compatibility for the engines)
        optimization_level: (Optional[int]): Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a lower optimization level. The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level, which is currently 5. Setting it to be greater than the maximum level results in identical behavior to the maximum level.
        use_python_runtime: (bool): Return a graph using a pure Python runtime, reduces options for serialization
        use_fast_partitioner: (bool): Use the adjacency based partitioning scheme instead of the global partitioner. Adjacency partitioning is faster but may not be optimal. Use the global paritioner (``False``) if looking for best performance
        enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the graph easier to convert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
        dryrun (bool): Toggle for "Dryrun" mode, running everything except conversion to TRT and logging outputs
        hardware_compatible (bool): Build the TensorRT engines compatible with GPU architectures other than that of the GPU on which the engine was built (currently works for NVIDIA Ampere and newer)
        timing_cache_path (str): Path to the timing cache if it exists (or) where it will be saved after compilation
        lazy_engine_init (bool): Defer setting up engines until the compilation of all engines is complete. Can allow larger models with multiple graph breaks to compile but can lead to oversubscription of GPU memory at runtime.
        cache_built_engines (bool): Whether to save the compiled TRT engines to storage
        reuse_cached_engines (bool): Whether to load the compiled TRT engines from storage
        engine_cache_dir (str): Directory to store the cached TRT engines
        engine_cache_size (int): Maximum hard-disk space (bytes) to use for the engine cache, default is 1GB. If the cache exceeds this size, the oldest engines will be removed by default
        custom_engine_cache (Optional[BaseEngineCache]): Engine cache instance to use for saving and loading engines. Users can provide their own engine cache by inheriting from BaseEngineCache. If used, engine_cache_dir and engine_cache_size will be ignored.
        use_explicit_typing (bool): This flag enables strong typing in TensorRT compilation which respects the precisions set in the Pytorch model. This is useful when users have mixed precision graphs.
        use_fp32_acc (bool): This option inserts cast to FP32 nodes around matmul layers and TensorRT ensures the accumulation of matmul happens in FP32. Use this only when FP16 precision is configured in enabled_precisions.
        refit_identical_engine_weights (bool): Refit engines with identical weights. This is useful when the same model is compiled multiple times with different inputs and the weights are the same. This will save time by reusing the same engine for different inputs.
        strip_engine_weights (bool): Strip engine weights from the serialized engine. This is useful when the engine is to be deployed in an environment where the weights are not required.
        immutable_weights (bool): Build non-refittable engines. This is useful for some layers that are not refittable. If this argument is set to true, `strip_engine_weights` and `refit_identical_engine_weights` will be ignored.
        enable_weight_streaming (bool): Enable weight streaming.
        tiling_optimization_level (str): The optimization level of tiling strategies. A higher level allows TensorRT to spend more time searching for better tiling strategy. We currently support ["none", "fast", "moderate", "full"].
        l2_limit_for_tiling (int): The target L2 cache usage limit (in bytes) for tiling optimization (default is -1 which means no limit).
        offload_module_to_cpu (bool): Offload the module to CPU. This is useful when we need to minimize GPU memory usage.
        use_distributed_mode_trace (bool):  Using aot_autograd to trace the graph. This is enabled when DTensors or distributed tensors are present in distributed model
        enable_autocast (bool): Whether to enable autocast. If enabled, use_explicit_typing will be set to True.
        autocast_low_precision_type (Optional[Union[torch.dtype, dtype]]): The precision to reduce to. We currently support torch.float16 and torch.bfloat16. Default is None, which means no low precision is used.
        autocast_excluded_nodes (Collection[str]): The set of regex patterns to match user-specified node names that should remain in FP32. Default is [].
        autocast_excluded_ops (Collection[Target]): The set of targets (ATen ops) that should remain in FP32. Default is [].
        autocast_max_output_threshold (float): Maximum absolute value for node outputs, nodes with outputs greater than this value will remain in FP32. Default is 512.
        autocast_max_depth_of_reduction (Optional[int]): Maximum depth of reduction allowed in low precision. Nodes with higher reduction depths will remain in FP32. This helps prevent excessive accuracy loss in operations particularly sensitive to reduced precision, as higher-depth reductions may amplify computation errors in low precision formats. If not provided, infinity will be used. Default is None.
        autocast_calibration_dataloader (Optional[torch.utils.data.DataLoader]): The dataloader to use for autocast calibration. Default is None.
        enable_resource_partitioning (bool): Enable resource-aware partitioning. This is useful when the model is large and the CPU memory is limited.
        cpu_memory_budget (Optional[int]): The maximum amount of CPU memory to use for the compilation. If the compilation requires more memory than this budget, the compilation will fail.
        **kwargs: Any,
    Returns:
        torch.fx.GraphModule: Compiled FX Module, when run it will execute via TensorRT
    """

    if kwargs.get("debug", False):
        warnings.warn(
            "`debug` is deprecated. Please use `with torch_tensorrt.dynamo.Debugger(...)` to wrap your compilation call to enable debugging functionality.",
            DeprecationWarning,
            stacklevel=2,
        )

    if not kwargs.get("use_explicit_typing", False):
        warnings.warn(
            "`use_explicit_typing` is deprecated. This setting will be removed and you should enable autocast instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if "truncate_long_and_double" in kwargs.keys():
        if truncate_double is not _defaults.TRUNCATE_DOUBLE:
            raise ValueError(
                'Provided configuration for "truncate_double" and deprecated API "truncate_long_and_double", please only use "truncate_double"'
            )
        else:
            truncate_double = kwargs["truncate_long_and_double"]
            warnings.warn(
                'Compiler option "truncate_long_and_double" is deprecated in favor of "truncate_double" as int64 is now natively supported, this option will be removed in the next version',
                DeprecationWarning,
                stacklevel=2,
            )

    if "refit" in kwargs.keys():
        warnings.warn(
            "`refit` is deprecated. Please set `immutable_weights=False` to build a refittable engine whose weights can be refitted",
            DeprecationWarning,
            stacklevel=2,
        )
        if immutable_weights:
            raise ValueError(
                "Use flag `immutable_weights` only. Flag `refit` is deprecated."
            )
        else:
            immutable_weights = not kwargs["refit"]

    if "make_refittable" in kwargs.keys():
        warnings.warn(
            "`make_refittable` is deprecated. Please set `immutable_weights=False` to build a refittable engine whose weights can be refitted",
            DeprecationWarning,
            stacklevel=2,
        )
        if immutable_weights:
            raise ValueError(
                "Use flag `immutable_weights` only. Flag `make_refittable` is deprecated."
            )
        else:
            immutable_weights = not kwargs["make_refittable"]

    if refit_identical_engine_weights:
        if immutable_weights:
            raise ValueError(
                "`immutable_weights` must be False when `refit_identical_engine_weights` is True."
            )

    if (
        "enable_cross_compile_for_windows" in kwargs.keys()
        and kwargs["enable_cross_compile_for_windows"]
    ):
        raise ValueError(
            "Please use torch_tensorrt.dynamo.cross_compile_for_windows() if you want to cross compile the module in Linux for inferencing in Windows."
        )

    engine_capability = EngineCapability._from(engine_capability)

    if torch_executed_modules is not None and torch_executed_modules:
        logger.warning(
            f"Detected torch_executed_modules was non-empty: {torch_executed_modules}"
            "\nThis feature is unimplemented in Torch-TRT Dynamo currently."
        )

    if enable_autocast:
        use_explicit_typing = True
        logger.debug("Autocast is enabled, setting use_explicit_typing to True.")

    if use_explicit_typing:
        if len(enabled_precisions) != 1 or not any(
            x in enabled_precisions
            for x in {torch.float32, dtype.f32, torch.float4_e2m1fn_x2, dtype.f4}
        ):
            raise AssertionError(
                f"use_explicit_typing was set to True, however found that enabled_precisions was also specified (saw: {enabled_precisions}, expected: dtype.f32, dtype.f4). enabled_precisions should not be used when use_explicit_typing=True"
            )

    if autocast_low_precision_type is not None:
        if not isinstance(autocast_low_precision_type, (torch.dtype, dtype)):
            raise ValueError(
                f"autocast_low_precision_type must be a torch.dtype or torch_tensorrt._enums.dtype, got {type(autocast_low_precision_type)}"
            )
        if autocast_low_precision_type not in {
            torch.float16,
            torch.bfloat16,
        } and autocast_low_precision_type not in {dtype.f16, dtype.bf16}:
            raise ValueError(
                f"autocast_low_precision_type must be one of torch.float16, torch.bfloat16, dtype.f16, dtype.bf16, got {autocast_low_precision_type}"
            )

    if use_fp32_acc:
        logger.debug(
            "FP32 accumulation for matmul layers is enabled. This option should only be enabled if the model already has FP16 weights and has no effect if it has FP32 weights. \
                     This flag inserts casts around matmul layers and ensures TensorRT executes the matmul layers in FP16 with FP32 accumulation."
        )

    if enable_weight_streaming and not use_explicit_typing:
        raise AssertionError(
            "When enable_weight_streaming is enabled, it requires use_explicit_typing to be set to True"
        )
    # Aliasing inputs to arg_inputs for better understanding
    if arg_inputs is None and kwarg_inputs is None and inputs is None:
        raise AssertionError(
            "'arg_inputs', 'kwarg_inputs' and 'inputs' should not all be None."
        )

    elif arg_inputs is not None and inputs is not None:
        raise AssertionError(
            "'arg_inputs' and 'inputs' should not be used at the same time."
        )

    arg_inputs = inputs or arg_inputs

    if kwarg_inputs is None:
        kwarg_inputs = {}

    if not isinstance(arg_inputs, collections.abc.Sequence):
        arg_inputs = [arg_inputs]  # type: ignore

    # Prepare torch_trt inputs
    trt_arg_inputs: Sequence[Input] = prepare_inputs(arg_inputs)
    trt_kwarg_inputs: Optional[dict[Any, Any]] = prepare_inputs(kwarg_inputs)
    device = to_torch_tensorrt_device(device)
    enabled_precisions = {dtype._from(p) for p in enabled_precisions}

    engine_cache = None
    if cache_built_engines or reuse_cached_engines:
        engine_cache = (
            custom_engine_cache
            if custom_engine_cache is not None
            else DiskEngineCache(engine_cache_dir, engine_cache_size)
        )

    compilation_options = {
        "enabled_precisions": (
            enabled_precisions if enabled_precisions else _defaults.ENABLED_PRECISIONS
        ),
        "device": device,
        "assume_dynamic_shape_support": assume_dynamic_shape_support,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": (
            torch_executed_ops if torch_executed_ops is not None else set()
        ),
        "pass_through_build_failures": pass_through_build_failures,
        "max_aux_streams": max_aux_streams,
        "version_compatible": version_compatible,
        "optimization_level": optimization_level,
        "use_python_runtime": use_python_runtime,
        "truncate_double": truncate_double,
        "use_fast_partitioner": use_fast_partitioner,
        "num_avg_timing_iters": num_avg_timing_iters,
        "enable_experimental_decompositions": enable_experimental_decompositions,
        "require_full_compilation": require_full_compilation,
        "disable_tf32": disable_tf32,
        "sparse_weights": sparse_weights,
        "engine_capability": engine_capability,
        "dla_sram_size": dla_sram_size,
        "dla_local_dram_size": dla_local_dram_size,
        "dla_global_dram_size": dla_global_dram_size,
        "dryrun": dryrun,
        "hardware_compatible": hardware_compatible,
        "timing_cache_path": timing_cache_path,
        "lazy_engine_init": lazy_engine_init,
        "cache_built_engines": cache_built_engines,
        "reuse_cached_engines": reuse_cached_engines,
        "use_explicit_typing": use_explicit_typing,
        "use_fp32_acc": use_fp32_acc,
        "refit_identical_engine_weights": refit_identical_engine_weights,
        "strip_engine_weights": strip_engine_weights,
        "immutable_weights": immutable_weights,
        "enable_cross_compile_for_windows": False,
        "enable_weight_streaming": enable_weight_streaming,
        "tiling_optimization_level": tiling_optimization_level,
        "l2_limit_for_tiling": l2_limit_for_tiling,
        "offload_module_to_cpu": offload_module_to_cpu,
        "use_distributed_mode_trace": use_distributed_mode_trace,
        "enable_autocast": enable_autocast,
        "autocast_low_precision_type": autocast_low_precision_type,
        "autocast_excluded_nodes": autocast_excluded_nodes,
        "autocast_excluded_ops": autocast_excluded_ops,
        "autocast_max_output_threshold": autocast_max_output_threshold,
        "autocast_max_depth_of_reduction": autocast_max_depth_of_reduction,
        "autocast_calibration_dataloader": autocast_calibration_dataloader,
        "enable_resource_partitioning": enable_resource_partitioning,
        "cpu_memory_budget": cpu_memory_budget,
    }
    logger.debug(f"CPU memory usage before lowering: {get_cpu_memory_usage()} MB")
    settings = CompilationSettings(**compilation_options)
    logger.info("Compilation Settings: %s\n", settings)
    exported_program = pre_export_lowering(exported_program, settings)
    exported_program = exported_program.run_decompositions(
        get_decompositions(enable_experimental_decompositions)
    )

    gm = exported_program.module()
    # Move the weights in the state_dict to CPU
    logger.debug("Input graph: " + str(gm.graph))

    # Apply lowering on the graph module
    gm = post_lowering(gm, settings)
    logger.debug(f"CPU memory usage after post_lowering: {get_cpu_memory_usage()} MB")
    logger.debug("Lowered Input graph: " + str(gm.graph))

    # Move the weights in the state_dict to CPU
    if offload_module_to_cpu:
        deallocate_module(gm, delete_module=False)
        deallocate_module(exported_program.module(), delete_module=False)
        logger.info(
            "The PyTorch model was moved to the CPU to allocate all GPU memory to TensorRT. To retain the model on the GPU, set offload_module_to_cpu=False"
        )
        logger.debug(f"CPU memory usage after CPU offload: {get_cpu_memory_usage()} MB")
    else:
        remaining_memory, total_memory = torch.cuda.mem_get_info()
        if remaining_memory < total_memory // 2:
            logger.warning(
                "Remaining GPU memory may not be enough to compile the TensorRT engine for this model resulting in an OOM error, Consider setting offload_module_to_cpu=True"
            )
    trt_gm = compile_module(
        gm, trt_arg_inputs, trt_kwarg_inputs, settings, engine_cache
    )
    return trt_gm


@fn_supports_debugger  # type: ignore[misc]
def compile_module(
    gm: torch.fx.GraphModule,
    sample_arg_inputs: Sequence[Input],
    sample_kwarg_inputs: Optional[dict[Any, Any]] = None,
    settings: CompilationSettings = CompilationSettings(),
    engine_cache: Optional[BaseEngineCache] = None,
    *,
    _debugger_config: Optional[DebuggerConfig] = None,
) -> torch.fx.GraphModule:
    """Compile a traced FX module

    Includes: Partitioning + Conversion Phases

    Args:
        module: FX GraphModule to convert
        arg_inputs: Inputs to the module
        kwarg_inputs: kwargs to the module
        settings: Compilation settings
        engine_cache: Engine cache instance to store/load compiled engines
    Returns:
        Compiled FX GraphModule
    """
    if any(v.requires_grad for v in gm.state_dict().values()):
        logger.warning(
            "The model may be in training mode, which may affect the performance of the compiled model!"
        )
    dryrun_tracker = DryRunTracker()
    if sample_kwarg_inputs is None:
        sample_kwarg_inputs = {}

    # Configure user compilation settings to converters.
    CONVERTERS.set_compilation_settings(settings)

    # Check the number of supported operations in the graph
    num_supported_ops, total_ops = partitioning.get_graph_converter_support(
        gm, settings.torch_executed_ops
    )

    dryrun_tracker.total_ops_in_graph = total_ops
    dryrun_tracker.supported_ops_in_graph = num_supported_ops
    dryrun_tracker.compilation_settings = settings

    if settings.dryrun and settings.min_block_size > 1:
        logger.info(
            "It is recommended to run `dryrun` mode with `min_block_size=1`, "
            "for the most thorough analysis"
        )

    # If the number of supported operations is 0 or less than the block size, skip the subgraph
    # TODO: Add condition to second expression below when require_full_compilation is added
    if num_supported_ops == 0 or (
        num_supported_ops < settings.min_block_size and not settings.dryrun
    ):
        logger.warning(
            f"{num_supported_ops} supported operations detected in subgraph containing {total_ops} computational nodes. "
            f"Skipping this subgraph, since min_block_size was detected to be {settings.min_block_size}"
        )
        return gm
    else:
        logger.debug(
            f"Detected support for {num_supported_ops} operators out of {total_ops} in subgraph."
        )

    def contains_metadata(gm: torch.fx.GraphModule) -> bool:
        for node in gm.graph.nodes:
            if node.op != "output" and (not node.meta) and "val" not in node.meta:
                logger.warning(
                    f"Node {node.name} of op type {node.op} does not have metadata. This could sometimes lead to undefined behavior."
                )
                return False
        return True

    # Check if the module has metadata (shape, dtype).
    if not contains_metadata(gm):
        # TODO: For future, explore when nodes don't have metadata and if fake_tensor_prop can resolve this.
        logger.warning(
            "Some nodes do not have metadata (shape and dtype information). This could lead to problems sometimes if the graph has PyTorch and TensorRT segments."
        )

    # Store the original input spec for later use
    original_in_spec = getattr(gm, "_in_spec", None)
    original_out_spec = getattr(gm, "_out_spec", None)

    # Function to preserve and restore module specs
    def preserve_module_specs(
        in_spec: Any, out_spec: Any, target_module: torch.fx.GraphModule
    ) -> None:
        """
        Applies input and output specs to the target module.

        Args:
            in_spec: The input spec to apply
            out_spec: The output spec to apply
            target_module: The module to apply specs to
        """
        # Apply specs to target module
        if in_spec is not None:
            target_module._in_spec = in_spec
        if out_spec is not None:
            target_module._out_spec = out_spec

    # Partition module into components that can be TRT-accelerated
    fast_partitioner_failed = False
    # If specified, try using the fast partitioner and fall back to the global one on failure
    if settings.use_fast_partitioner:
        try:
            logger.info("Partitioning the graph via the fast partitioner")
            partitioned_module, supported_ops = partitioning.fast_partition(
                gm,
                min_block_size=settings.min_block_size,
                torch_executed_ops=settings.torch_executed_ops,
                require_full_compilation=settings.require_full_compilation,
                skip_fusion=(num_supported_ops == total_ops),
            )

        except torch.fx.passes.splitter_base.FxNetSplitterInternalError:
            logger.error(
                "Partitioning failed on the subgraph with fast partition. See trace above. "
                "Retrying with global partition.",
                exc_info=True,
            )

            fast_partitioner_failed = True
            settings.use_fast_partitioner = False

    if not settings.use_fast_partitioner:
        logger.info("Partitioning the graph via the global partitioner")
        partitioned_module, supported_ops = partitioning.global_partition(
            gm,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
            require_full_compilation=settings.require_full_compilation,
        )

    if settings.enable_resource_partitioning:
        partitioned_module = resource_partition(
            partitioned_module,
            cpu_memory_budget=settings.cpu_memory_budget,
        )

    dryrun_tracker.unsupported_ops = supported_ops.unsupported_operators

    # The global partitioner leaves non-TRT nodes as-is
    if not settings.use_fast_partitioner:
        dryrun_tracker.to_run_in_torch.extend(parse_non_trt_nodes(partitioned_module))

    submodule_node_dict = {}
    for node in partitioned_module.graph.nodes:
        if "_run_on_acc" not in node.name:
            continue
        submodule_node_dict[node.name] = node

    preserve_module_specs(original_in_spec, original_out_spec, partitioned_module)
    # Store TRT replicas of Torch subgraphs
    trt_modules = {}
    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those

    # Here we delete the frozen parameters from the graph module. Note this does not affect the submodules. We are going to delete the frozen parameters from the submodules in the convert_module function.
    # This is done to release CPU memory.
    for attr in dir(gm):
        if attr.startswith("_frozen_param"):
            delattr(gm, attr)

    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)
        # filter on the GraphModule
        if not isinstance(submodule, torch.fx.graph_module.GraphModule):
            continue
        # Criteria for a module to be convertible to TRT
        if settings.use_fast_partitioner and "_run_on_acc" not in name:
            dryrun_tracker.to_run_in_torch.extend(parse_non_trt_nodes(submodule))
            logger.debug(
                "Submodule in PyTorch: %s\n %s",
                str(name),
                str(submodule.graph),
            )
            submodule.to(to_torch_device(settings.device))
            continue

        if name not in submodule_node_dict:
            raise ValueError(
                f"node_name: {name} does not exist in the submodule node dictionary"
            )

        # set the submodule metadata back to the parent trt_module_node
        metadata_list = get_output_metadata(submodule)
        assert len(metadata_list) > 0
        metadata_keys = ["val", "tensor_meta"]
        for key in metadata_keys:
            if key not in submodule_node_dict[name].meta:
                meta_val_list = [
                    metadata[key] for metadata in metadata_list if key in metadata
                ]
                submodule_node_dict[name].meta[key] = meta_val_list
                logger.debug(
                    f"Updated metadata for node: {name} with its corresponding submodule outputs"
                )
                break

        subgraph_data = PerSubgraphData()
        subgraph_data.subgraph_name = name
        subgraph_data.subgraph_op_count = len(
            [
                node
                for node in submodule.graph.nodes
                if node.op in ("call_function", "call_method", "call_module")
            ]
        )

        # Get the submodule inputs for min, opt, max shapes of the graph inputs
        submodule_inputs = partitioning.construct_submodule_inputs(submodule)

        assert submodule_inputs is not None

        logger.debug(
            "Converting submodule: %s\n Input shapes: %s\n %s",
            str(name),
            [input.shape for input in submodule_inputs],
            str(submodule.graph),
        )

        # Handle long/double inputs if requested by the user
        if settings.truncate_double:
            submodule_inputs = repair_double_inputs(
                partitioned_module,
                submodule,
                submodule_inputs,
                to_torch_device(settings.device),
                name,
            )

        # Parse the subgraph I/O and store it
        parse_graph_io(submodule, subgraph_data)
        dryrun_tracker.tensorrt_graph_count += 1
        dryrun_tracker.per_subgraph_data.append(subgraph_data)
        torch.cuda.empty_cache()
        # Create TRT engines from submodule
        if not settings.dryrun:
            trt_module = convert_module(
                submodule,
                submodule_inputs,
                settings=settings,
                name=name,
                engine_cache=engine_cache,
            )

            trt_modules[name] = trt_module

            if _debugger_config:

                if _debugger_config.save_engine_profile:
                    if settings.use_python_runtime:
                        if _debugger_config.profile_format != "cudagraph":
                            raise ValueError(
                                "Profiling with TREX can only be enabled when using the C++ runtime. Python runtime profiling only support cudagraph visualization."
                            )
                        else:
                            trt_module.enable_profiling()
                    else:
                        if _debugger_config.profile_format == "cudagraph":
                            raise ValueError(
                                "Profiling with Cudagraph can only be enabled when using the Python runtime. C++ runtime profiling only support TREX/Perfetto visualization."
                            )
                        else:
                            path = os.path.join(
                                _debugger_config.logging_dir,
                                "engine_visualization_profile",
                            )
                            os.makedirs(path, exist_ok=True)
                            trt_module.enable_profiling(
                                profiling_results_dir=path,
                                profile_format=_debugger_config.profile_format,
                            )

                if _debugger_config.save_layer_info:
                    with open(
                        os.path.join(
                            _debugger_config.logging_dir, "engine_layer_info.json"
                        ),
                        "w",
                    ) as f:
                        f.write(trt_module.get_layer_info())

    # Only set the requires_unique_output flag for the last TRT Module when user has access to the output tensor

    # Parse the graph I/O and store it in dryrun tracker
    parse_graph_io(gm, dryrun_tracker)

    # Replace all FX Modules with TRT Modules
    for name, trt_module in trt_modules.items():
        setattr(partitioned_module, name, trt_module)
        if settings.lazy_engine_init and not settings.enable_cross_compile_for_windows:
            trt_module = getattr(partitioned_module, name)
            trt_module.setup_engine()

    output_node = list(partitioned_module.graph.nodes)[-1]
    for arg in output_node.args:
        for output in arg:
            target = output.target
            if "_run_on_acc" not in str(target):
                continue
            getattr(partitioned_module, target).set_output_tensors_as_unowned(True)

    # Reset settings object to user specification after fallback to global partitioning mode
    if fast_partitioner_failed:
        settings.use_fast_partitioner = True

    dryrun_stats_display(dryrun_tracker, settings.dryrun)

    return partitioned_module


def convert_exported_program_to_serialized_trt_engine(
    exported_program: ExportedProgram,
    inputs: Optional[Sequence[Sequence[Any]]] = None,
    *,
    arg_inputs: Optional[Sequence[Sequence[Any]]] = None,
    kwarg_inputs: Optional[dict[Any, Any]] = None,
    device: Optional[Union[Device, torch.device, str]] = _defaults.DEVICE,
    disable_tf32: bool = _defaults.DISABLE_TF32,
    assume_dynamic_shape_support: bool = _defaults.ASSUME_DYNAMIC_SHAPE_SUPPORT,
    sparse_weights: bool = _defaults.SPARSE_WEIGHTS,
    enabled_precisions: Union[
        Set[Union[torch.dtype, dtype]], Tuple[Union[torch.dtype, dtype]]
    ] = _defaults.ENABLED_PRECISIONS,
    engine_capability: EngineCapability = _defaults.ENGINE_CAPABILITY,
    num_avg_timing_iters: int = _defaults.NUM_AVG_TIMING_ITERS,
    workspace_size: int = _defaults.WORKSPACE_SIZE,
    dla_sram_size: int = _defaults.DLA_SRAM_SIZE,
    dla_local_dram_size: int = _defaults.DLA_LOCAL_DRAM_SIZE,
    dla_global_dram_size: int = _defaults.DLA_GLOBAL_DRAM_SIZE,
    truncate_double: bool = _defaults.TRUNCATE_DOUBLE,
    require_full_compilation: bool = _defaults.REQUIRE_FULL_COMPILATION,
    min_block_size: int = _defaults.MIN_BLOCK_SIZE,
    torch_executed_ops: Optional[Collection[Target]] = None,
    torch_executed_modules: Optional[List[str]] = None,
    pass_through_build_failures: bool = _defaults.PASS_THROUGH_BUILD_FAILURES,
    max_aux_streams: Optional[int] = _defaults.MAX_AUX_STREAMS,
    version_compatible: bool = _defaults.VERSION_COMPATIBLE,
    optimization_level: Optional[int] = _defaults.OPTIMIZATION_LEVEL,
    use_python_runtime: bool = _defaults.USE_PYTHON_RUNTIME,
    use_fast_partitioner: bool = _defaults.USE_FAST_PARTITIONER,
    enable_experimental_decompositions: bool = _defaults.ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    dryrun: bool = _defaults.DRYRUN,
    hardware_compatible: bool = _defaults.HARDWARE_COMPATIBLE,
    timing_cache_path: str = _defaults.TIMING_CACHE_PATH,
    lazy_engine_init: bool = _defaults.LAZY_ENGINE_INIT,
    cache_built_engines: bool = _defaults.CACHE_BUILT_ENGINES,
    reuse_cached_engines: bool = _defaults.REUSE_CACHED_ENGINES,
    engine_cache_dir: str = _defaults.ENGINE_CACHE_DIR,
    engine_cache_size: int = _defaults.ENGINE_CACHE_SIZE,
    custom_engine_cache: Optional[BaseEngineCache] = _defaults.CUSTOM_ENGINE_CACHE,
    use_explicit_typing: bool = _defaults.USE_EXPLICIT_TYPING,
    use_fp32_acc: bool = _defaults.USE_FP32_ACC,
    refit_identical_engine_weights: bool = _defaults.REFIT_IDENTICAL_ENGINE_WEIGHTS,
    strip_engine_weights: bool = _defaults.STRIP_ENGINE_WEIGHTS,
    immutable_weights: bool = _defaults.IMMUTABLE_WEIGHTS,
    enable_weight_streaming: bool = _defaults.ENABLE_WEIGHT_STREAMING,
    tiling_optimization_level: str = _defaults.TILING_OPTIMIZATION_LEVEL,
    l2_limit_for_tiling: int = _defaults.L2_LIMIT_FOR_TILING,
    offload_module_to_cpu: bool = _defaults.OFFLOAD_MODULE_TO_CPU,
    use_distributed_mode_trace: bool = _defaults.USE_DISTRIBUTED_MODE_TRACE,
    **kwargs: Any,
) -> bytes:
    """Convert an ExportedProgram to a serialized TensorRT engine

    Converts an ExportedProgram to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        exported_program (torch.export.ExportedProgram): Source module, running torch.export on a ``torch.nn.Module``
        inputs (Optional[Sequence[Sequence[Any]]]): List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type.

                .. code-block:: py

                    inputs=[
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
        arg_inputs (Optional[Sequence[Sequence[Any]]]): Same as inputs. Alias for better understanding with kwarg_inputs.
        kwarg_inputs (Optional[dict[Any, Any]]): kwarg inputs to the module forward function.
        device (Union(torch_tensorrt.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=torch_tensorrt.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        assume_dynamic_shape_support (bool): Setting this to true enables the converters work for both dynamic and static shapes. Default: False
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precisions (Union[Set[Union[torch.dtype, dtype]], Tuple[Union[torch.dtype, dtype]]]): The set of datatypes that TensorRT can use when selecting kernels
        engine_capability (torch_tensorrt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        truncate_double (bool): Truncate weights provided in double (float64) to float32
        require_full_compilation (bool): Require modules to be compiled end to end or return an error as opposed to returning a hybrid graph where operations that cannot be run in TensorRT are run in PyTorch
        min_block_size (int): The minimum number of contiguous TensorRT convertible operations in order to run a set of operations in TensorRT
        torch_executed_ops (Optional[Collection[Target]]): Set of aten operators that must be run in PyTorch. An error will be thrown if this set is not empty but ``require_full_compilation`` is True
        torch_executed_modules (Optional[List[str]]): List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
        pass_through_build_failures (bool): Error out if there are issues during compilation (only applicable to torch.compile workflows)
        max_aux_streams (Optional[int]): Maximum streams in the engine
        version_compatible (bool): Build the TensorRT engines compatible with future versions of TensorRT (Restrict to lean runtime operators to provide version forward compatibility for the engines)
        optimization_level: (Optional[int]): Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a lower optimization level. The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level, which is currently 5. Setting it to be greater than the maximum level results in identical behavior to the maximum level.
        use_python_runtime: (bool): Return a graph using a pure Python runtime, reduces options for serialization
        use_fast_partitioner: (bool): Use the adjacency based partitioning scheme instead of the global partitioner. Adjacency partitioning is faster but may not be optimal. Use the global paritioner (``False``) if looking for best performance
        enable_experimental_decompositions (bool): Use the full set of operator decompositions. These decompositions may not be tested but serve to make the graph easier to convert to TensorRT, potentially increasing the amount of graphs run in TensorRT.
        dryrun (bool): Toggle for "Dryrun" mode, running everything except conversion to TRT and logging outputs
        hardware_compatible (bool): Build the TensorRT engines compatible with GPU architectures other than that of the GPU on which the engine was built (currently works for NVIDIA Ampere and newer)
        timing_cache_path (str): Path to the timing cache if it exists (or) where it will be saved after compilation
        lazy_engine_init (bool): Defer setting up engines until the compilation of all engines is complete. Can allow larger models with multiple graph breaks to compile but can lead to oversubscription of GPU memory at runtime.
        cache_built_engines (bool): Whether to save the compiled TRT engines to storage
        reuse_cached_engines (bool): Whether to load the compiled TRT engines from storage
        engine_cache_dir (str): Directory to store the cached TRT engines
        engine_cache_size (int): Maximum hard-disk space (bytes) to use for the engine cache, default is 1GB. If the cache exceeds this size, the oldest engines will be removed by default
        custom_engine_cache (Optional[BaseEngineCache]): Engine cache instance to use for saving and loading engines. Users can provide their own engine cache by inheriting from BaseEngineCache. If used, engine_cache_dir and engine_cache_size will be ignored.
        use_explicit_typing (bool): This flag enables strong typing in TensorRT compilation which respects the precisions set in the Pytorch model. This is useful when users have mixed precision graphs.
        use_fp32_acc (bool): This option inserts cast to FP32 nodes around matmul layers and TensorRT ensures the accumulation of matmul happens in FP32. Use this only when FP16 precision is configured in enabled_precisions.
        refit_identical_engine_weights (bool): Refit engines with identical weights. This is useful when the same model is compiled multiple times with different inputs and the weights are the same. This will save time by reusing the same engine for different inputs.
        strip_engine_weights (bool): Strip engine weights from the serialized engine. This is useful when the engine is to be deployed in an environment where the weights are not required.
        immutable_weights (bool): Build non-refittable engines. This is useful for some layers that are not refittable. If this argument is set to true, `strip_engine_weights` and `refit_identical_engine_weights` will be ignored.
        enable_weight_streaming (bool): Enable weight streaming.
        tiling_optimization_level (str): The optimization level of tiling strategies. A higher level allows TensorRT to spend more time searching for better tiling strategy. We currently support ["none", "fast", "moderate", "full"].
        l2_limit_for_tiling (int): The target L2 cache usage limit (in bytes) for tiling optimization (default is -1 which means no limit).
        offload_module_to_cpu (bool): Offload the module to CPU. This is useful when we need to minimize GPU memory usage.
        use_distributed_mode_trace (bool):  Using aot_autograd to trace the graph. This is enabled when DTensors or distributed tensors are present in distributed model.
        **kwargs: Any,
    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """

    if kwargs.get("debug", False):
        warnings.warn(
            "`debug` is deprecated. Please use `with torch_tensorrt.dynamo.Debugger(...)` to wrap your compilation call to enable debugging functionality.",
            DeprecationWarning,
            stacklevel=2,
        )

    if "truncate_long_and_double" in kwargs.keys():
        if truncate_double is not _defaults.TRUNCATE_DOUBLE:
            raise ValueError(
                'Provided configuration for "truncate_double" and deprecated API "truncate_long_and_double", please only use "truncate_double"'
            )
        else:
            truncate_double = kwargs["truncate_long_and_double"]
            warnings.warn(
                'Compiler option "truncate_long_and_double" is deprecated in favor of "truncate_double" as int64 is now natively supported, this option will be removed in the next version',
                DeprecationWarning,
                stacklevel=2,
            )

    if "refit" in kwargs.keys():
        warnings.warn(
            "`refit` is deprecated. Please set `immutable_weights=False` to build a refittable engine whose weights can be refitted",
            DeprecationWarning,
            stacklevel=2,
        )
        if immutable_weights:
            raise ValueError(
                "Use flag `immutable_weights` only. Flag `refit` is deprecated."
            )
        else:
            immutable_weights = not kwargs["refit"]

    if "make_refittable" in kwargs.keys():
        warnings.warn(
            "`make_refittable` is deprecated. Please set `immutable_weights=False` to build a refittable engine whose weights can be refitted",
            DeprecationWarning,
            stacklevel=2,
        )
        if immutable_weights:
            raise ValueError(
                "Use flag `immutable_weights` only. Flag `make_refittable` is deprecated."
            )
        else:
            immutable_weights = not kwargs["make_refittable"]

    if refit_identical_engine_weights:
        if immutable_weights:
            raise ValueError(
                "`immutable_weights` must be False when `refit_identical_engine_weights` is True."
            )

    if (
        "enable_cross_compile_for_windows" in kwargs.keys()
        and kwargs["enable_cross_compile_for_windows"]
    ):
        raise ValueError(
            "Please use torch_tensorrt.dynamo.cross_compile_for_windows() if you want to cross compile the module in Linux for inferencing in Windows."
        )

    engine_capability = EngineCapability._from(engine_capability)

    if torch_executed_modules is not None and torch_executed_modules:
        logger.warning(
            f"Detected torch_executed_modules was non-empty: {torch_executed_modules}"
            "\nThis feature is unimplemented in Torch-TRT Dynamo currently."
        )

    if use_explicit_typing:
        if len(enabled_precisions) != 1 or not any(
            x in enabled_precisions
            for x in {torch.float32, dtype.f32, torch.float4_e2m1fn_x2, dtype.f4}
        ):
            raise AssertionError(
                f"use_explicit_typing was set to True, however found that enabled_precisions was also specified (saw: {enabled_precisions}, expected: dtype.f32, dtype.f4). enabled_precisions should not be used when use_explicit_typing=True"
            )

    if use_fp32_acc:
        logger.debug(
            "FP32 accumulation for matmul layers is enabled. This option should only be enabled if the model already has FP16 weights and has no effect if it has FP32 weights. \
                     This flag inserts casts around matmul layers and ensures TensorRT executes the matmul layers in FP16 with FP32 accumulation."
        )

    if enable_weight_streaming and not use_explicit_typing:
        raise AssertionError(
            "When enable_weight_streaming is enabled, it requires use_explicit_typing to be set to True"
        )
    # Aliasing inputs to arg_inputs for better understanding
    if arg_inputs is None and kwarg_inputs is None and inputs is None:
        raise AssertionError(
            "'arg_inputs', 'kwarg_inputs' and 'inputs' should not all be None."
        )

    elif arg_inputs is not None and inputs is not None:
        raise AssertionError(
            "'arg_inputs' and 'inputs' should not be used at the same time."
        )

    arg_inputs = inputs or arg_inputs

    if kwarg_inputs is None:
        kwarg_inputs = {}

    if not isinstance(arg_inputs, collections.abc.Sequence):
        arg_inputs = [arg_inputs]  # type: ignore

    # Prepare torch_trt inputs
    trt_arg_inputs: Sequence[Input] = prepare_inputs(arg_inputs)
    trt_kwarg_inputs: Optional[dict[str, Any]] = prepare_inputs(kwarg_inputs)
    device = to_torch_tensorrt_device(device)
    enabled_precisions = {dtype._from(p) for p in enabled_precisions}

    engine_cache = None
    if cache_built_engines or reuse_cached_engines:
        engine_cache = (
            custom_engine_cache
            if custom_engine_cache is not None
            else DiskEngineCache(engine_cache_dir, engine_cache_size)
        )

    compilation_options = {
        "enabled_precisions": (
            enabled_precisions if enabled_precisions else _defaults.ENABLED_PRECISIONS
        ),
        "device": device,
        "assume_dynamic_shape_support": assume_dynamic_shape_support,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        "torch_executed_ops": (
            torch_executed_ops if torch_executed_ops is not None else set()
        ),
        "pass_through_build_failures": pass_through_build_failures,
        "max_aux_streams": max_aux_streams,
        "version_compatible": version_compatible,
        "optimization_level": optimization_level,
        "use_python_runtime": use_python_runtime,
        "truncate_double": truncate_double,
        "use_fast_partitioner": use_fast_partitioner,
        "num_avg_timing_iters": num_avg_timing_iters,
        "enable_experimental_decompositions": enable_experimental_decompositions,
        "require_full_compilation": require_full_compilation,
        "disable_tf32": disable_tf32,
        "sparse_weights": sparse_weights,
        "engine_capability": engine_capability,
        "dla_sram_size": dla_sram_size,
        "dla_local_dram_size": dla_local_dram_size,
        "dla_global_dram_size": dla_global_dram_size,
        "dryrun": dryrun,
        "hardware_compatible": hardware_compatible,
        "timing_cache_path": timing_cache_path,
        "lazy_engine_init": lazy_engine_init,
        "cache_built_engines": cache_built_engines,
        "reuse_cached_engines": reuse_cached_engines,
        "use_explicit_typing": use_explicit_typing,
        "use_fp32_acc": use_fp32_acc,
        "refit_identical_engine_weights": refit_identical_engine_weights,
        "strip_engine_weights": strip_engine_weights,
        "immutable_weights": immutable_weights,
        "enable_cross_compile_for_windows": False,
        "enable_weight_streaming": enable_weight_streaming,
        "tiling_optimization_level": tiling_optimization_level,
        "l2_limit_for_tiling": l2_limit_for_tiling,
        "offload_module_to_cpu": offload_module_to_cpu,
        "use_distributed_mode_trace": use_distributed_mode_trace,
    }

    settings = CompilationSettings(**compilation_options)
    logger.info("Compilation Settings: %s\n", settings)
    exported_program = pre_export_lowering(exported_program, settings)
    exported_program = exported_program.run_decompositions(
        get_decompositions(enable_experimental_decompositions)
    )

    gm = exported_program.module()
    # Move the weights in the state_dict to CPU
    logger.debug("Input graph: " + str(gm.graph))

    # Apply lowering on the graph module
    gm = post_lowering(gm, settings)
    logger.debug("Lowered Input graph: " + str(gm.graph))

    # Move the weights in the state_dict to CPU
    if offload_module_to_cpu:
        deallocate_module(exported_program.module(), delete_module=False)
        logger.info(
            "The PyTorch model was moved to the CPU to allocate all GPU memory to TensorRT. To retain the model on the GPU, set offload_module_to_cpu=False"
        )
    else:
        remaining_memory, total_memory = torch.cuda.mem_get_info()
        if remaining_memory < total_memory // 2:
            logger.warning(
                "Remaining GPU memory may not be enough to compile the TensorRT engine for this model resulting in an OOM error, Consider setting offload_module_to_cpu=True"
            )

    flattened_input_list = get_flat_args_with_check(
        exported_program, list(trt_arg_inputs), trt_kwarg_inputs
    )[0]

    try:
        interpreter_result = interpret_module_to_result(
            gm,
            inputs=flattened_input_list,
            settings=settings,
            engine_cache=engine_cache,
        )
    except UnsupportedOperatorException:
        logger.error(
            f"Conversion of module {gm} not currently fully supported or convertible!",
            exc_info=True,
        )
    except Exception as e:
        logger.error(
            f"While interpreting the module got an error: {e}",
            exc_info=True,
        )

    serialized_engine: bytes = interpreter_result.serialized_engine
    return serialized_engine


@needs_cross_compile  # type: ignore[misc]
def save_cross_compiled_exported_program(
    gm: torch.fx.GraphModule,
    file_path: str,
) -> None:
    """
    Save cross compiled exported program to disk.

    Arguments:
        module (torch.fx.GraphModule): Cross compiled Torch-TensorRT module
        file_path (str): the file path where the exported program will be saved to disk
    """
    if not file_path:
        raise ValueError("File path cannot be empty. Please provide a valid file path")

    from torch_tensorrt.dynamo._exporter import export

    exp_program = export(gm, cross_compile_module=True)
    torch.export.save(exp_program, file_path)
    logger.debug(f"successfully saved the module for windows at {file_path}")


def load_cross_compiled_exported_program(file_path: str = "") -> Any:
    """
    Load an ExportedProgram file in Windows which was previously cross compiled in Linux

    Arguments:
        file_path (str): Path to file on the disk

    Raises:
        ValueError: If the api is not called in windows or there is no file or the file is a valid ExportedProgram file
    """
    if not file_path:
        raise ValueError("File path cannot be empty. Please provide a valid file path")

    if platform.system() != "Windows" or platform.machine() != "AMD64":
        raise ValueError(
            "cross runtime compiled model for windows can only be loaded in Windows system"
        )

    try:
        logger.debug(f"Loading the provided file {file_path} using torch.export.load()")
        # TODO: think about how to handle the torch.jit.load route?
        exp_program = torch.export.load(file_path)
    except Exception as e:
        logger.info(
            f"Loading the provided file {file_path} via torch.export.load() failed with the following error: {e}",
            exc_info=True,
        )
        raise ValueError(
            f"cross_load the file {file_path} doesn't correspond to a valid ExportedProgram. Please verify the file path."
        )

    return replace_execute_engine_no_op_node(exp_program)
