from dataclasses import dataclass, field
from typing import Any, Collection, Optional, Set, Tuple, Union

import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import EngineCapability, dtype
from torch_tensorrt.dynamo._defaults import (
    ASSUME_DYNAMIC_SHAPE_SUPPORT,
    AUTOCAST_CALIBRATION_DATALOADER,
    AUTOCAST_EXCLUDED_NODES,
    AUTOCAST_EXCLUDED_OPS,
    AUTOCAST_LOW_PRECISION_TYPE,
    AUTOCAST_MAX_DEPTH_OF_REDUCTION,
    AUTOCAST_MAX_OUTPUT_THRESHOLD,
    CACHE_BUILT_ENGINES,
    CPU_MEMORY_BUDGET,
    DISABLE_TF32,
    DLA_GLOBAL_DRAM_SIZE,
    DLA_LOCAL_DRAM_SIZE,
    DLA_SRAM_SIZE,
    DRYRUN,
    ENABLE_AUTOCAST,
    ENABLE_CROSS_COMPILE_FOR_WINDOWS,
    ENABLE_EXPERIMENTAL_DECOMPOSITIONS,
    ENABLE_RESOURCE_PARTITIONING,
    ENABLE_WEIGHT_STREAMING,
    ENABLED_PRECISIONS,
    ENGINE_CAPABILITY,
    HARDWARE_COMPATIBLE,
    IMMUTABLE_WEIGHTS,
    L2_LIMIT_FOR_TILING,
    LAZY_ENGINE_INIT,
    MAX_AUX_STREAMS,
    MIN_BLOCK_SIZE,
    NUM_AVG_TIMING_ITERS,
    OFFLOAD_MODULE_TO_CPU,
    OPTIMIZATION_LEVEL,
    PASS_THROUGH_BUILD_FAILURES,
    REFIT_IDENTICAL_ENGINE_WEIGHTS,
    REQUIRE_FULL_COMPILATION,
    REUSE_CACHED_ENGINES,
    SPARSE_WEIGHTS,
    STRIP_ENGINE_WEIGHTS,
    TILING_OPTIMIZATION_LEVEL,
    TIMING_CACHE_PATH,
    TRUNCATE_DOUBLE,
    USE_DISTRIBUTED_MODE_TRACE,
    USE_EXPLICIT_TYPING,
    USE_FAST_PARTITIONER,
    USE_FP32_ACC,
    USE_PYTHON_RUNTIME,
    VERSION_COMPATIBLE,
    WORKSPACE_SIZE,
    default_device,
)


@dataclass
class CompilationSettings:
    """Compilation settings for Torch-TensorRT Dynamo Paths

    Args:
        enabled_precisions (Set[dtype]): Available kernel dtype precisions
        debug (bool): Whether to print out verbose debugging information
        workspace_size (int): Workspace TRT is allowed to use for the module (0 is default)
        min_block_size (int): Minimum number of operators per TRT-Engine Block
        torch_executed_ops (Collection[Target]): Collection of operations to run in Torch, regardless of converter coverage
        pass_through_build_failures (bool): Whether to fail on TRT engine build errors (True) or not (False)
        max_aux_streams (Optional[int]): Maximum number of allowed auxiliary TRT streams for each engine
        version_compatible (bool): Provide version forward-compatibility for engine plan files
        optimization_level (Optional[int]): Builder optimization 0-5, higher levels imply longer build time,
            searching for more optimization options. TRT defaults to 3
        use_python_runtime (Optional[bool]): Whether to strictly use Python runtime or C++ runtime. To auto-select a runtime
            based on C++ dependency presence (preferentially choosing C++ runtime if available), leave the
            argument as None
        truncate_double (bool): Whether to truncate float64 TRT engine inputs or weights to float32
        use_fast_partitioner (bool): Whether to use the fast or global graph partitioning system
        enable_experimental_decompositions (bool): Whether to enable all core aten decompositions
            or only a selected subset of them
        device (Device): GPU to compile the model on
        require_full_compilation (bool): Whether to require the graph is fully compiled in TensorRT.
            Only applicable for `ir="dynamo"`; has no effect for `torch.compile` path
        assume_dynamic_shape_support (bool): Setting this to true enables the converters work for both dynamic and static shapes. Default: False
        disable_tf32 (bool): Whether to disable TF32 computation for TRT layers
        sparse_weights (bool): Whether to allow the builder to use sparse weights
        engine_capability (trt.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        dla_sram_size (int): Fast software managed RAM used by DLA to communicate within a layer.
        dla_local_dram_size (int): Host RAM used by DLA to share intermediate tensor data across operations
        dla_global_dram_size (int): Host RAM used by DLA to store weights and metadata for execution
        dryrun (Union[bool, str]): Toggle "Dryrun" mode, which runs everything through partitioning, short of conversion to
            TRT Engines. Prints detailed logs of the graph structure and nature of partitioning. Optionally saves the
            output to a file if a string path is specified
        hardware_compatible (bool): Build the TensorRT engines compatible with GPU architectures other than that of the GPU on which the engine was built (currently works for NVIDIA Ampere and newer)
        timing_cache_path (str): Path to the timing cache if it exists (or) where it will be saved after compilation
        cache_built_engines (bool): Whether to save the compiled TRT engines to storage
        reuse_cached_engines (bool): Whether to load the compiled TRT engines from storage
        use_strong_typing (bool): This flag enables strong typing in TensorRT compilation which respects the precisions set in the Pytorch model. This is useful when users have mixed precision graphs.
        use_fp32_acc (bool): This option inserts cast to FP32 nodes around matmul layers and TensorRT ensures the accumulation of matmul happens in FP32. Use this only when FP16 precision is configured in enabled_precisions.
        refit_identical_engine_weights (bool): Whether to refit the engine with identical weights
        strip_engine_weights (bool): Whether to strip the engine weights
        immutable_weights (bool): Build non-refittable engines. This is useful for some layers that are not refittable. If this argument is set to true, `strip_engine_weights` and `refit_identical_engine_weights` will be ignored
        enable_weight_streaming (bool): Enable weight streaming.
        enable_cross_compile_for_windows (bool): By default this is False means TensorRT engines can only be executed on the same platform where they were built.
            True will enable cross-platform compatibility which allows the engine to be built on Linux and run on Windows
        tiling_optimization_level (str): The optimization level of tiling strategies. A higher level allows TensorRT to spend more time searching for better tiling strategy. We currently support ["none", "fast", "moderate", "full"].
        l2_limit_for_tiling (int): The target L2 cache usage limit (in bytes) for tiling optimization (default is -1 which means no limit).
        use_distributed_mode_trace (bool):  Using aot_autograd to trace the graph. This is enabled when DTensors or distributed tensors are present in distributed model
        enable_autocast (bool): Whether to enable autocast. If enabled, use_explicit_typing will be set to True.
        autocast_low_precision_type (Optional[Union[torch.dtype, dtype]]): The precision to reduce to. We currently support torch.float16 and torch.bfloat16. Default is None, which means no low precision is used.
        autocast_excluded_nodes (Collection[str]): The set of regex patterns to match user-specified node names that should remain in FP32. Default is [].
        autocast_excluded_ops (Collection[Target]): The set of targets (ATen ops) that should remain in FP32. Default is [].
        autocast_max_output_threshold (float): Maximum absolute value for node outputs, nodes with outputs greater than this value will remain in FP32. Default is 512.
        autocast_max_depth_of_reduction (Optional[int]): Maximum depth of reduction allowed in low precision. Nodes with higher reduction depths will remain in FP32. This helps prevent excessive accuracy loss in operations particularly sensitive to reduced precision, as higher-depth reductions may amplify computation errors in low precision formats. If not provided, infinity will be used. Default is None.
        autocast_calibration_dataloader (Optional[torch.utils.data.DataLoader]): The dataloader to use for autocast calibration. Default is None.
    """

    enabled_precisions: Set[dtype] = field(default_factory=lambda: ENABLED_PRECISIONS)
    workspace_size: int = WORKSPACE_SIZE
    min_block_size: int = MIN_BLOCK_SIZE
    torch_executed_ops: Collection[Target] = field(default_factory=set)
    pass_through_build_failures: bool = PASS_THROUGH_BUILD_FAILURES
    max_aux_streams: Optional[int] = MAX_AUX_STREAMS
    version_compatible: bool = VERSION_COMPATIBLE
    optimization_level: Optional[int] = OPTIMIZATION_LEVEL
    use_python_runtime: Optional[bool] = USE_PYTHON_RUNTIME
    truncate_double: bool = TRUNCATE_DOUBLE
    use_fast_partitioner: bool = USE_FAST_PARTITIONER
    enable_experimental_decompositions: bool = ENABLE_EXPERIMENTAL_DECOMPOSITIONS
    device: Device = field(default_factory=default_device)
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION
    disable_tf32: bool = DISABLE_TF32
    assume_dynamic_shape_support: bool = ASSUME_DYNAMIC_SHAPE_SUPPORT
    sparse_weights: bool = SPARSE_WEIGHTS
    engine_capability: EngineCapability = field(
        default_factory=lambda: ENGINE_CAPABILITY
    )
    num_avg_timing_iters: int = NUM_AVG_TIMING_ITERS
    dla_sram_size: int = DLA_SRAM_SIZE
    dla_local_dram_size: int = DLA_LOCAL_DRAM_SIZE
    dla_global_dram_size: int = DLA_GLOBAL_DRAM_SIZE
    dryrun: Union[bool, str] = DRYRUN
    hardware_compatible: bool = HARDWARE_COMPATIBLE
    timing_cache_path: str = TIMING_CACHE_PATH
    lazy_engine_init: bool = LAZY_ENGINE_INIT
    cache_built_engines: bool = CACHE_BUILT_ENGINES
    reuse_cached_engines: bool = REUSE_CACHED_ENGINES
    use_explicit_typing: bool = USE_EXPLICIT_TYPING
    use_fp32_acc: bool = USE_FP32_ACC
    refit_identical_engine_weights: bool = REFIT_IDENTICAL_ENGINE_WEIGHTS
    strip_engine_weights: bool = STRIP_ENGINE_WEIGHTS
    immutable_weights: bool = IMMUTABLE_WEIGHTS
    enable_weight_streaming: bool = ENABLE_WEIGHT_STREAMING
    enable_cross_compile_for_windows: bool = ENABLE_CROSS_COMPILE_FOR_WINDOWS
    tiling_optimization_level: str = TILING_OPTIMIZATION_LEVEL
    l2_limit_for_tiling: int = L2_LIMIT_FOR_TILING
    use_distributed_mode_trace: bool = USE_DISTRIBUTED_MODE_TRACE
    offload_module_to_cpu: bool = OFFLOAD_MODULE_TO_CPU
    enable_autocast: bool = ENABLE_AUTOCAST
    autocast_low_precision_type: Optional[dtype] = AUTOCAST_LOW_PRECISION_TYPE
    autocast_excluded_nodes: Collection[str] = field(
        default_factory=lambda: AUTOCAST_EXCLUDED_NODES
    )
    autocast_excluded_ops: Collection[Target] = field(
        default_factory=lambda: AUTOCAST_EXCLUDED_OPS
    )
    autocast_max_output_threshold: float = AUTOCAST_MAX_OUTPUT_THRESHOLD
    autocast_max_depth_of_reduction: Optional[int] = AUTOCAST_MAX_DEPTH_OF_REDUCTION
    autocast_calibration_dataloader: Optional[torch.utils.data.DataLoader] = (
        AUTOCAST_CALIBRATION_DATALOADER
    )
    enable_resource_partitioning: bool = ENABLE_RESOURCE_PARTITIONING
    cpu_memory_budget: int = CPU_MEMORY_BUDGET

    def __getstate__(self) -> dict[str, Any]:
        from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
            ConverterRegistry,
        )

        state = self.__dict__.copy()
        state["torch_executed_ops"] = {
            op if isinstance(op, str) else ConverterRegistry.qualified_name_or_str(op)
            for op in state["torch_executed_ops"]
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


# If any of the following setting is changed, the engine should be rebuilt.
_SETTINGS_TO_BE_ENGINE_INVARIANT = {
    "enabled_precisions",
    "max_aux_streams",
    "version_compatible",
    "optimization_level",
    "disable_tf32",
    "sparse_weights",
    "engine_capability",
    "hardware_compatible",
    "refit_identical_engine_weights",
    "immutable_weights",
    "enable_weight_streaming",
    "tiling_optimization_level",
    "l2_limit_for_tiling",
    "enable_autocast",
    "autocast_low_precision_type",
    "autocast_excluded_nodes",
    "autocast_excluded_ops",
    "autocast_max_output_threshold",
    "autocast_max_depth_of_reduction",
    "autocast_calibration_dataloader",
}


if not hasattr(trt.SerializationFlag, "INCLUDE_REFIT"):  # for TensorRT < 10.14
    _SETTINGS_TO_BE_ENGINE_INVARIANT.add("strip_engine_weights")


def settings_are_compatible(
    set_a: CompilationSettings, set_b: CompilationSettings
) -> Tuple[bool, Set[str]]:
    incompatible_settings: Set[str] = set()

    for f in _SETTINGS_TO_BE_ENGINE_INVARIANT:
        if getattr(set_a, f) != getattr(set_b, f):
            incompatible_settings.add(f)

    if len(incompatible_settings) == 0:
        return True, set()
    else:
        return False, incompatible_settings
