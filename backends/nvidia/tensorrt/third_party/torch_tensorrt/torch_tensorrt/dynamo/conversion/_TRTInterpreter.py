import gc
import logging
import os
import warnings
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import tensorrt as trt
import torch
import torch.fx
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.fx.node import _get_qualified_name
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils._python_dispatch import _disable_current_modes
from torch_tensorrt import ENABLED_FEATURES
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import needs_refit
from torch_tensorrt._Input import Input
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    CallingConvention,
)
from torch_tensorrt.dynamo.conversion._TRTBuilderMonitor import TRTBulderMonitor
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_node_io,
    get_node_name,
    get_trt_tensor,
    to_torch,
)
from torch_tensorrt.dynamo.debug._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo.debug._supports_debugger import cls_supports_debugger
from torch_tensorrt.dynamo.observer import Observer
from torch_tensorrt.dynamo.utils import (
    DYNAMIC_DIM,
    deallocate_module,
    get_cpu_memory_usage,
    to_torch_device,
)
from torch_tensorrt.logging import TRT_LOGGER

_LOGGER: logging.Logger = logging.getLogger(__name__)

TRT_INTERPRETER_CALL_PRE_OBSERVER: Observer[Callable[[torch.fx.GraphModule], None]] = (
    Observer("TRT_INTERPRETER_CALL_PRE_OBSERVER")
)


class UnsupportedOperatorException(RuntimeError):
    pass


class TRTInterpreterResult(NamedTuple):
    engine: trt.ICudaEngine
    input_names: Sequence[str]
    output_names: Sequence[str]
    weight_name_map: Optional[dict[Any, Any]]
    requires_output_allocator: bool


@cls_supports_debugger
class TRTInterpreter(torch.fx.Interpreter):  # type: ignore[misc]
    def __init__(
        self,
        module: torch.fx.GraphModule,
        input_specs: Sequence[Input],
        output_dtypes: Optional[Sequence[dtype]] = None,
        compilation_settings: CompilationSettings = CompilationSettings(),
        engine_cache: Optional[BaseEngineCache] = None,
        *,
        _debugger_config: Optional[DebuggerConfig] = None,
    ):
        super().__init__(module)

        self.logger = TRT_LOGGER
        self.builder = trt.Builder(self.logger)
        self._debugger_config = _debugger_config
        flag = 0
        # rtx build, strongly typed is enabled by default, can not set it by builder config
        if ENABLED_FEATURES.tensorrt_rtx:
            if not compilation_settings.use_explicit_typing:
                warnings.warn(
                    "Strongly typed is enabled by default in torch-tensorrt-rtx build,  setting use_explicit_typing to True"
                )
                compilation_settings.use_explicit_typing = True
        else:
            if compilation_settings.use_explicit_typing:
                STRONGLY_TYPED = 1 << (int)(
                    trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
                )
                flag |= STRONGLY_TYPED

        self.ctx = ConversionContext(
            self.builder.create_network(flag), compilation_settings
        )

        self.compilation_settings = compilation_settings
        if not CONVERTERS.compilation_settings:
            # Configure user compilation settings to converters.
            CONVERTERS.set_compilation_settings(compilation_settings)
        self.validate_compile_settings()
        assert TRTInterpreter._all_precisions_supported(
            compilation_settings.enabled_precisions
        ), f"Attempted to enable kernel precisions that are not supported (got: {compilation_settings.enabled_precisions}, support: {_defaults.SUPPORTED_KERNEL_PRECISIONS})"
        missing_ops = self.validate_conversion()
        if missing_ops:
            warnings.warn(
                "Interpretation will fail due to missing operations \n"
                + "\n".join(f"{i}" for i in missing_ops)
            )

        self.optimization_profiles: Optional[List[trt.IOptimizationProfile]] = (
            [self.builder.create_optimization_profile()]
            if any(
                input_spec.shape_mode == Input._ShapeMode.DYNAMIC
                for input_spec in input_specs
            )
            else None
        )

        self.input_specs = input_specs
        self.input_specs_iter = 0
        self._cur_node_name: Optional[str] = None
        self._cur_node: Optional[torch.fx.Node] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._itensor_to_tensor_meta: Dict[trt.tensorrt.ITensor, TensorMetadata] = (
            dict()
        )

        # Data types for TRT Module output Tensors
        self.output_dtypes = (
            [dtype._from(o) for o in output_dtypes] if output_dtypes else None
        )

        # Mapping of constants to shapes and dtypes
        self.const_mapping: Dict[str, Tuple[Sequence[int], str]] = {}
        self.weight_name_map: Optional[Dict[str, Any]] = None

        # Engine cache for storing and reusing TRT engines
        self.engine_cache = engine_cache

    def validate_conversion(self) -> Set[str]:
        missing_converters: Set[str] = set()

        for node in self.module.graph.nodes:
            if node.op == "call_function" and CONVERTERS.get(node) is None:
                missing_converters.add(f"{node.op} {_get_qualified_name(node.target)}")
            elif node.op == "call_method" and CONVERTERS.get(node) is None:
                missing_converters.add(f"{node.op} torch.Tensor.{node.target}")
            elif node.op == "call_module":
                submod = self.fetch_attr(node.target)
                submod_type = getattr(submod, "_base_class_origin", type(submod))
                if CONVERTERS.get(node) is None:
                    missing_converters.add(f"{node.op} {torch.typename(submod_type)}")

        return missing_converters

    @staticmethod
    def _args_str(args: List[Any]) -> str:
        def clean_repr(x: Any, depth: int = 0) -> Any:
            if isinstance(x, trt.ITensor):
                return f"{x.name} <tensorrt.ITensor [shape={x.shape}, dtype={x.dtype}]>"
            elif isinstance(x, torch.Tensor):
                return f"<torch.Tensor [shape={x.shape}, dtype={x.dtype}]>"
            elif isinstance(x, np.ndarray):
                return (
                    f"<torch.Tensor as np.ndarray [shape={x.shape}, dtype={x.dtype}]>"
                )
            elif isinstance(x, Sequence) and not isinstance(x, str):
                if depth < 3:
                    return type(x)([clean_repr(i, depth=depth + 1) for i in x])  # type: ignore[call-arg]
                else:
                    return "(...)"
            else:
                return f"{x} <{type(x).__name__}>"

        str_args = [clean_repr(a) for a in args]
        return repr(tuple(str_args))

    @staticmethod
    def _all_precisions_supported(enabled_precisions: Set[dtype]) -> bool:
        return enabled_precisions.issubset(_defaults.SUPPORTED_KERNEL_PRECISIONS)

    def validate_compile_settings(self) -> None:
        if ENABLED_FEATURES.tensorrt_rtx:
            if dtype.bfloat16 in self.compilation_settings.enabled_precisions:
                raise RuntimeError("TensorRT-RTX does not support bfloat16!")
            return

        if (
            dtype.i8 in self.compilation_settings.enabled_precisions
            and not self.builder.platform_has_fast_int8
        ):
            raise RuntimeError("Current platform doesn't support fast native int8!")

        if (
            dtype.f16 in self.compilation_settings.enabled_precisions
            and not self.builder.platform_has_fast_fp16
        ):
            warnings.warn("Current platform doesn't support fast native fp16!")

    def _populate_trt_builder_config(
        self,
        strict_type_constraints: bool = False,
        algorithm_selector: Optional[trt.IAlgorithmSelector] = None,
        tactic_sources: Optional[int] = None,
    ) -> trt.IBuilderConfig:
        builder_config = self.builder.create_builder_config()

        if self._debugger_config and self._debugger_config.engine_builder_monitor:
            builder_config.progress_monitor = TRTBulderMonitor()

        if self.compilation_settings.workspace_size != 0:
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, self.compilation_settings.workspace_size
            )

        if is_tensorrt_version_supported("8.2"):
            builder_config.profiling_verbosity = (
                trt.ProfilingVerbosity.DETAILED
                if self._debugger_config and self._debugger_config.save_engine_profile
                else trt.ProfilingVerbosity.LAYER_NAMES_ONLY
            )

        if is_tensorrt_version_supported("8.6"):
            if self.compilation_settings.max_aux_streams is not None:
                _LOGGER.info(
                    f"Setting max aux streams to {self.compilation_settings.max_aux_streams}"
                )
                builder_config.max_aux_streams = (
                    self.compilation_settings.max_aux_streams
                )
            if self.compilation_settings.version_compatible:
                _LOGGER.info("Using version compatible")
                builder_config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
                builder_config.set_flag(trt.BuilderFlag.EXCLUDE_LEAN_RUNTIME)
            if self.compilation_settings.hardware_compatible:
                _LOGGER.info("Using hardware compatible")
                builder_config.hardware_compatibility_level = (
                    trt.HardwareCompatibilityLevel.AMPERE_PLUS
                )
            if self.compilation_settings.optimization_level is not None:
                _LOGGER.info(
                    f"Using optimization level {self.compilation_settings.optimization_level}"
                )
                builder_config.builder_optimization_level = (
                    self.compilation_settings.optimization_level
                )

        builder_config.engine_capability = (
            self.compilation_settings.engine_capability.to(trt.EngineCapability)
        )
        builder_config.avg_timing_iterations = (
            self.compilation_settings.num_avg_timing_iters
        )

        if self.compilation_settings.device.device_type == trt.DeviceType.DLA:
            device_info = torch.cuda.get_device_properties(
                self.compilation_settings.device.gpu_id
            )
            assert (device_info.major == 8 and device_info.minor == 7) or (
                device_info.major == 7 and device_info.minor == 2
            ), "DLA is not available on non AGX systems"
            builder_config.DLA_core = self.compilation_settings.device.dla_core
            _LOGGER.info(f"Using DLA core {self.compilation_settings.device.dla_core}")
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_MANAGED_SRAM,
                self.compilation_settings.dla_sram_size,
            )
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_LOCAL_DRAM,
                self.compilation_settings.dla_local_dram_size,
            )
            builder_config.set_memory_pool_limit(
                trt.MemoryPoolType.DLA_GLOBAL_DRAM,
                self.compilation_settings.dla_global_dram_size,
            )

        if not self.compilation_settings.use_explicit_typing:
            if dtype.float16 in self.compilation_settings.enabled_precisions:
                builder_config.set_flag(trt.BuilderFlag.FP16)

            if dtype.int8 in self.compilation_settings.enabled_precisions:
                builder_config.set_flag(trt.BuilderFlag.INT8)

            if dtype.fp8 in self.compilation_settings.enabled_precisions:
                builder_config.set_flag(trt.BuilderFlag.FP8)

            if dtype.bfloat16 in self.compilation_settings.enabled_precisions:
                builder_config.set_flag(trt.BuilderFlag.BF16)

        if self.compilation_settings.sparse_weights:
            builder_config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        if self.compilation_settings.disable_tf32:
            builder_config.clear_flag(trt.BuilderFlag.TF32)

        if self.compilation_settings.immutable_weights:
            # non-refittable engine
            if self.compilation_settings.strip_engine_weights:
                _LOGGER.warning("strip_engine_weights will be ignored.")
            if self.compilation_settings.refit_identical_engine_weights:
                _LOGGER.warning("refit_identical_engine_weights will be ignored.")
        else:
            # refittable engine
            if self.compilation_settings.refit_identical_engine_weights:
                builder_config.set_flag(trt.BuilderFlag.REFIT_IDENTICAL)
            else:
                builder_config.set_flag(trt.BuilderFlag.REFIT)

            if self.compilation_settings.strip_engine_weights:
                builder_config.set_flag(trt.BuilderFlag.STRIP_PLAN)

        if strict_type_constraints:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if self.optimization_profiles is not None:
            if len(self.optimization_profiles) > 0:
                for optimization_profile in self.optimization_profiles:
                    builder_config.add_optimization_profile(optimization_profile)

        if algorithm_selector:
            builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
            builder_config.algorithm_selector = algorithm_selector

        if tactic_sources is not None:
            builder_config.set_tactic_sources(tactic_sources=tactic_sources)

        if self.compilation_settings.enable_cross_compile_for_windows:
            builder_config.runtime_platform = trt.RuntimePlatform.WINDOWS_AMD64
            _LOGGER.info(
                "Setting runtime_platform as trt.RuntimePlatform.WINDOWS_AMD64"
            )

        if self.compilation_settings.enable_weight_streaming:
            builder_config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)

        if is_tensorrt_version_supported("10.8"):
            TilingOptimizationLevel = {
                "none": trt.TilingOptimizationLevel.NONE,
                "fast": trt.TilingOptimizationLevel.FAST,
                "moderate": trt.TilingOptimizationLevel.MODERATE,
                "full": trt.TilingOptimizationLevel.FULL,
            }
            assert (
                self.compilation_settings.tiling_optimization_level
                in TilingOptimizationLevel
            ), f"Invalid tiling optimization level: {self.compilation_settings.tiling_optimization_level}. We currently support {TilingOptimizationLevel.keys()}."
            builder_config.tiling_optimization_level = TilingOptimizationLevel[
                self.compilation_settings.tiling_optimization_level
            ]

            if self.compilation_settings.l2_limit_for_tiling != -1:
                builder_config.l2_limit_for_tiling = (
                    self.compilation_settings.l2_limit_for_tiling
                )

        return builder_config

    def _create_timing_cache(
        self,
        builder_config: trt.IBuilderConfig,
        timing_cache_path: str = "",
    ) -> None:
        """
        Create a timing cache to enable faster build time for TRT engines.
        By default the timing_cache_path="/tmp/timing_cache.bin"
        """
        buffer = b""
        if os.path.isfile(timing_cache_path):
            # Load from existing cache
            with open(timing_cache_path, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
        cache = builder_config.create_timing_cache(buffer)
        builder_config.set_timing_cache(cache, False)

    def _save_timing_cache(
        self,
        builder_config: trt.IBuilderConfig,
        timing_cache_path: str,
    ) -> None:
        """
        This is called after a TensorRT engine is built. Save the timing cache
        """
        timing_cache = builder_config.get_timing_cache()
        os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
        with open(timing_cache_path, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    def _construct_trt_network_def(self) -> None:
        """
        Run the interpreter on each node to get TRT INetwork
        """
        TRT_INTERPRETER_CALL_PRE_OBSERVER.observe(self.module)

        self.input_specs_iter = 0
        run_module_start_time = datetime.now()
        super().run()
        _LOGGER.info(
            f"TRT INetwork construction elapsed time: {datetime.now() - run_module_start_time}"
        )

    @staticmethod
    def find_weight(
        weight_name: str,
        weight_refit_map: dict[str, Any],
        state_dict: dict[str, Any],
        device: torch.device,
    ) -> str:
        """
        We need to build map from engine weight name to state_dict weight name.
        The purpose of this function is to find the corresponding weight name in module state_dict.

        weight_name: the target weight name we want to search for
        np_map: the map from weight name to np values in INetworkDefinition
        state_dict: state of the graph module
        """
        with unset_fake_temporarily():
            network_weight = weight_refit_map[weight_name].to(device)
            for sd_w_name, sd_weight in state_dict.items():
                if TRTInterpreter.check_weight_equal(sd_weight, network_weight, device):
                    del state_dict[sd_w_name]
                    return sd_w_name
            return ""

    @staticmethod
    def check_weight_equal(
        sd_weight: torch.tensor,
        network_weight: Union[torch.Tensor, np.ndarray],
        device: torch.device,
    ) -> Any:
        with unset_fake_temporarily():
            if network_weight.device != device:
                network_weight = network_weight.to(device)
            try:
                return sd_weight.shape == network_weight.shape and torch.all(
                    torch.abs(sd_weight - network_weight) < 0.01
                )
            except Exception:
                return torch.all(sd_weight == network_weight)

    @needs_refit  # type: ignore[misc]
    def _save_weight_mapping(self) -> None:
        """
        Construct the weight name mapping from engine weight name to state_dict weight name.
        Cache the weight name for future refitting usecases.
        Two-stage weight name tracing:
        1. Name transformation from engine weight name to state_dict weight name
        2. Value mapping that, for each weight in INetworkDefinition search for identical weight in state_dict
        """

        MODULE_MAP = {
            "SCALE": (
                trt.IScaleLayer,
                [
                    (
                        "scale",
                        "SCALE",
                        ("weight", "bias", "running_mean", "running_var"),
                    ),
                    (
                        "shift",
                        "SHIFT",
                        ("weight", "bias", "running_mean", "running_var"),
                    ),
                ],
            ),
            "CONVOLUTION": (
                trt.IConvolutionLayer,
                [("kernel", "KERNEL", "weight"), ("bias", "BIAS", "bias")],
            ),
            "DECONVOLUTION": (
                trt.IDeconvolutionLayer,
                [("kernel", "KERNEL", "weight"), ("bias", "BIAS", "bias")],
            ),
            "CONSTANT": (
                trt.IConstantLayer,
                [("weights", "CONSTANT", ("weight", "bias"))],
            ),
        }
        """
        The structure of this map is:
        {
            layer_type: (
                Corresponding ILayer type to cast,
                [
                    (
                        ILayer weight attribute,
                        Weight name postfix in TRT Engine,
                        Weight name postfix in state_dict
                    ),
                    ...
                ]
            )
        }
        """
        _LOGGER.info("Building weight name mapping...")
        # Stage 1: Name mapping
        torch_device = to_torch_device(self.compilation_settings.device)
        sd = {k: v.to(torch_device) for k, v in self.module.state_dict().items()}
        weight_name_map: dict[str, Any] = {}
        weight_refit_map = self.ctx.weight_refit_map
        constant_mapping = {k: v for k, v in weight_refit_map.items() if v.size == 1}
        net = self.ctx.net
        for i in range(net.num_layers):
            layer = net[i]
            layer_type: str = layer.type.name
            if layer_type in MODULE_MAP:
                # Name mapping
                for weight_type, weight_name, torch_attr in MODULE_MAP[layer_type][1]:
                    engine_weight_name = f"{layer.name} {weight_name}"
                    # Infer the corresponding weight name(s) in state_dict
                    sd_weight_name_list = (
                        layer.name.split("-")[-1]
                        .replace("[", "")
                        .replace("]", "")
                        .split("/")
                    )
                    sd_weight_name: Any = ".".join(
                        [i for i in sd_weight_name_list[:-1] if i]
                    )
                    suffix = sd_weight_name_list[-1]
                    # Retrieve each weight name(s) in state_dict
                    if layer_type == "CONSTANT":
                        if (
                            "embedding" in suffix
                            or "weight" in suffix
                            or "mm_other" in suffix
                        ):
                            sd_weight_name = f"{sd_weight_name}.weight"
                        elif "running_mean" in suffix:
                            sd_weight_name = f"{sd_weight_name}.running_mean"
                        elif "running_var" in suffix:
                            sd_weight_name = f"{sd_weight_name}.running_var"
                        elif "bias" in suffix:
                            sd_weight_name = f"{sd_weight_name}.bias"
                        else:
                            sd_weight_name = f"{sd_weight_name}.unknown"
                    elif layer_type == "SCALE":
                        # Batch norm needs all weights to calculate scale and shift
                        sd_weight_name = [f"{sd_weight_name}.{n}" for n in torch_attr]
                    else:
                        sd_weight_name = f"{sd_weight_name}.{torch_attr}"

                    if engine_weight_name in weight_refit_map:
                        weight_name_map[engine_weight_name] = sd_weight_name

        # Stage 2: Value mapping
        for engine_weight_name, sd_weight_name in weight_name_map.items():
            if "SCALE" in engine_weight_name:
                # There is no direct connection in batch_norm layer. So skip it
                pass
            elif sd_weight_name not in sd or not TRTInterpreter.check_weight_equal(
                sd[sd_weight_name], weight_refit_map[engine_weight_name], torch_device
            ):
                weight_name_map[engine_weight_name] = TRTInterpreter.find_weight(
                    engine_weight_name, weight_refit_map, sd, torch_device
                )
                if (
                    weight_name_map[engine_weight_name] != ""
                    and engine_weight_name in constant_mapping
                ):
                    # If the weight is found in state_dict, remove it from constant_mapping
                    del constant_mapping[engine_weight_name]

            weight_name_map[engine_weight_name] = [
                weight_name_map[engine_weight_name],
                weight_refit_map[engine_weight_name].dtype,
            ]

        weight_name_map["constant_mapping"] = constant_mapping
        self.weight_name_map = weight_name_map

        del weight_refit_map, sd
        gc.collect()
        torch.cuda.empty_cache()

    def run(
        self,
        strict_type_constraints: bool = False,
        algorithm_selector: Optional[trt.IAlgorithmSelector] = None,
        tactic_sources: Optional[int] = None,
    ) -> TRTInterpreterResult:
        """
        Build TensorRT engine with some configs.
        Args:
            strict_type_constraints: Usually we should set it to False unless we want to control the precision of certain layer for numeric reasons.
            algorithm_selector: set up algorithm selection for certain layer
            tactic_sources: set up tactic sources for certain layer
        Return:
            TRTInterpreterResult
        """
        self._construct_trt_network_def()
        _LOGGER.debug(
            f"CPU memory usage after network construction: {get_cpu_memory_usage()} MB"
        )

        if not self.compilation_settings.immutable_weights:
            self._save_weight_mapping()

        if self.compilation_settings.offload_module_to_cpu:
            deallocate_module(self.module)

        build_engine_start_time = datetime.now()
        _LOGGER.info("Not found cached TRT engines. Start building engine.")

        builder_config = self._populate_trt_builder_config(
            strict_type_constraints, algorithm_selector, tactic_sources
        )

        self._create_timing_cache(
            builder_config, self.compilation_settings.timing_cache_path
        )

        if (
            ENABLED_FEATURES.tensorrt_rtx
            or self.compilation_settings.version_compatible
        ):
            # TODO: When TRT-RTX matures, change it to build_engine_with_config
            serialized_engine = self.builder.build_serialized_network(
                self.ctx.net, builder_config
            )
            runtime = trt.Runtime(TRT_LOGGER)
            cuda_engine = runtime.deserialize_cuda_engine(serialized_engine)
        else:

            cuda_engine = self.builder.build_engine_with_config(
                self.ctx.net, builder_config
            )
        assert cuda_engine

        _LOGGER.debug(
            f"CPU memory usage after engine building: {get_cpu_memory_usage()} MB"
        )

        _LOGGER.info(
            f"Build TRT engine elapsed time: {datetime.now() - build_engine_start_time}"
        )
        self.ctx.clear_cpu_weights_reference_holder()

        self._save_timing_cache(
            builder_config, self.compilation_settings.timing_cache_path
        )

        return TRTInterpreterResult(
            cuda_engine,
            self._input_names,
            self._output_names,
            self.weight_name_map,
            self.ctx.requires_output_allocator,
        )

    def run_node(self, n: torch.fx.Node) -> torch.fx.Node:
        self._cur_node_name = get_node_name(n)
        self._cur_node = n

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                f"Converting node {self._cur_node_name} (kind: {n.target}, args: {TRTInterpreter._args_str(n.args)})"
            )

        trt_node: torch.fx.Node = super().run_node(n)

        if n.op == "get_attr":
            self.const_mapping[str(n)] = (tuple(trt_node.shape), str(trt_node.dtype))

        _LOGGER.info(
            f"Converted node {self._cur_node_name} [{n.target}] ({get_node_io(n, self.const_mapping)})"
        )

        if isinstance(trt_node, trt.ITensor):
            self._itensor_to_tensor_meta[trt_node] = n.meta.get("tensor_meta")

        return trt_node

    def placeholder(self, target: str, args: Any, kwargs: Any) -> trt.ITensor:
        self._input_names.append(target)
        current_input = self.input_specs[self.input_specs_iter]
        self.input_specs_iter += 1
        # Set optimization profile for dynamic input shape
        shape = None
        if current_input.shape_mode == Input._ShapeMode.DYNAMIC:
            assert isinstance(current_input.shape, dict)
            shape = []
            min_shape = current_input.shape["min_shape"]
            opt_shape = current_input.shape["opt_shape"]
            max_shape = current_input.shape["max_shape"]
            # TODO: Does not support disjoint optimization profiles?
            assert self.optimization_profiles is not None
            assert len(min_shape) == len(opt_shape) == len(max_shape)
            if current_input.is_shape_tensor:
                # For shape_tensors, min/opt/max_shapes correspond to actual values
                # of the shapes provided during runtime
                self.optimization_profiles[0].set_shape_input(
                    target, min_shape, opt_shape, max_shape
                )
                shape.append(len(opt_shape))
            else:
                self.optimization_profiles[0].set_shape(
                    target, min_shape, opt_shape, max_shape
                )

                for i in range(len(min_shape)):
                    if min_shape[i] == opt_shape[i] == max_shape[i]:
                        shape.append(min_shape[i])
                    else:
                        # -1 to represent the dynamic dimension
                        shape.append(DYNAMIC_DIM)
        elif (
            not current_input.is_shape_tensor
            and current_input.shape_mode == Input._ShapeMode.STATIC
        ):
            assert isinstance(current_input.shape, tuple)
            shape = list(current_input.shape)
        else:
            raise RuntimeError(
                f"Unable to access shape spec for input: {target} (got: {current_input})"
            )

        trt_input_dtype = current_input.dtype.to(trt.DataType, use_default=True)
        _LOGGER.debug(
            f"Adding input to in-progress INetwork: {target} [shape={shape}, dtype={trt_input_dtype}]"
        )

        return self.ctx.net.add_input(
            name=target,
            shape=tuple(shape),
            dtype=trt_input_dtype,
        )

    def call_module(
        self, target: str, args: Any, kwargs: Any
    ) -> Any:  # Probably should be Tuple[trt.ITensor]? Case for Any?
        assert isinstance(target, str)
        submod = self.fetch_attr(target)
        submod_type = getattr(submod, "_base_class_origin", type(submod))
        converter_packet = CONVERTERS.get(self._cur_node)

        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of module of type {submod_type} not currently supported!"
            )

        converter, calling_convention, _ = converter_packet

        assert self._cur_node_name is not None

        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, submod, args, kwargs, self._cur_node_name)
        else:
            return converter(self.ctx, submod, args, kwargs, self._cur_node_name)

    def call_function(self, target: str, args: Any, kwargs: Any) -> Any:
        # TODO: Why is this stateful? We should be able to take in the inputs
        converter_packet = CONVERTERS.get(self._cur_node)
        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of function {torch.typename(target)} not currently supported!"
            )

        converter, calling_convention, converter_info = converter_packet
        if converter_info.get("requires_output_allocator", False):
            self.ctx.requires_output_allocator = True
            _LOGGER.debug(f"{target} requires output allocator")

        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        else:
            return converter(self.ctx, target, args, kwargs, self._cur_node_name)

    def get_attr(self, target: str, args: Any, kwargs: Any) -> torch.Tensor:
        with _disable_current_modes(), unset_fake_temporarily():
            frozen_attr = self.fetch_attr(target)
            if isinstance(frozen_attr, torch.nn.Parameter):
                constant_tensor = frozen_attr.data
            else:
                constant_tensor = frozen_attr

            return to_torch(constant_tensor)

    def call_method(self, target: str, args: Any, kwargs: Any) -> Any:
        assert isinstance(target, str)
        converter_packet = CONVERTERS.get(self._cur_node)

        if converter_packet is None:
            raise UnsupportedOperatorException(
                f"Conversion of method {target} not currently supported!"
            )
        converter, calling_convention, _ = converter_packet

        if calling_convention is CallingConvention.LEGACY:
            return converter(self.ctx.net, target, args, kwargs, self._cur_node_name)
        else:
            return converter(self.ctx, target, args, kwargs, self._cur_node_name)

    def output(self, target: str, args: Any, kwargs: Any) -> List[Any]:
        assert len(args) == 1
        if isinstance(args[0], tuple):
            outputs = args[0]
        elif isinstance(args[0], list):
            outputs = tuple(args[0])
        else:
            outputs = (args[0],)

        for output_idx in range(len(outputs)):
            output = outputs[output_idx]

            if not isinstance(output, trt.ITensor):
                new_output = get_trt_tensor(self.ctx, output, target)
                outputs = (
                    outputs[:output_idx] + (new_output,) + outputs[output_idx + 1 :]
                )

        if not all(isinstance(output, trt.ITensor) for output in outputs):
            raise RuntimeError("TensorRT requires all outputs to be Tensor!")

        if self.output_dtypes is not None and len(self.output_dtypes) != len(outputs):
            raise RuntimeError(
                f"Specified output dtypes ({len(self.output_dtypes)}) differ from number of outputs ({len(outputs)})"
            )

        marked_outputs_ids = []
        for i, output in enumerate(outputs):
            # In some cases, the same output tensor may be marked multiple times, such as _to_copy,
            # so we skip marking if the output is already marked
            if id(output) in marked_outputs_ids:
                continue
            marked_outputs_ids.append(id(output))

            name = f"output{i}"

            output_dtype = dtype.unknown
            if any(
                op_name in output.name.split("_")
                for op_name in (
                    "eq",
                    "gt",
                    "lt",
                    "or",
                    "xor",
                    "and",
                    "not",
                    "ne",
                    "isinf",
                    "isnan",
                    "any",
                )
            ):
                output_dtype = dtype.b
            elif self.output_dtypes is not None:
                if self.output_dtypes[i] == dtype.i64:
                    output = self.ctx.net.add_cast(
                        output, dtype.i64.to(trt.DataType)
                    ).get_output(0)
                    output_dtype = dtype.i64
                else:
                    output_dtype = self.output_dtypes[i]

            self.ctx.net.mark_output(output)
            if output_dtype is not dtype.unknown:
                output.dtype = output_dtype.to(trt.DataType, use_default=True)
            output.name = name

            self._output_names.append(name)
            _LOGGER.debug(
                f"Marking output {name} [shape={output.shape}, dtype={output.dtype}]"
            )

        return list(outputs)
