from __future__ import annotations

import io
import logging
from typing import Any, List, NamedTuple, Optional, Sequence

import tensorrt as trt
import torch
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings, settings_are_compatible
from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
    TRTInterpreter,
    TRTInterpreterResult,
)
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule
from torch_tensorrt.dynamo.utils import (
    get_cpu_memory_usage,
    get_output_dtypes,
    release_host_and_device_memory,
)
from torch_tensorrt.logging import TRT_LOGGER

logger = logging.getLogger(__name__)


class SerializedInterpreterResult(NamedTuple):
    serialized_engine: bytes
    input_names: Sequence[str]
    output_names: Sequence[str]
    weight_name_map: Optional[dict[Any, Any]]
    requires_output_allocator: bool


def infer_module_output_dtypes(
    module: torch.fx.GraphModule,
    truncate_double: bool = False,
) -> List[dtype]:
    """
    This function get the output dtypes from node.meta['val'] which was set during dynamo compile_module step
    and truncates them accordingly.
    """
    outputs = [node for node in module.graph.nodes if node.op == "output"]
    outputs = outputs[0].args
    return get_output_dtypes(outputs, truncate_double)  # type: ignore[no-any-return]


def insert_engine_to_cache(
    hash_val: Optional[str],
    interpreter_result: TRTInterpreterResult,
    engine_cache: BaseEngineCache,
    settings: CompilationSettings,
    inputs: Sequence[Input],
) -> bool:
    if hash_val is None:
        logger.warning("Hash value is not provided, so the engine will not be cached")
        return False

    if not ENABLED_FEATURES.refit:
        logger.warning(
            "Refit feature is not available, so the engine cache will not be used"
        )
        return False

    # Cache the weight-stripped engine regardless of the `strip_engine_weights` setting
    if engine_cache.check(hash_val) is not None:
        logger.info(
            f"Detected that the engine with hash: {hash_val} exists in cache. It will be refreshed"
        )

    serialization_config = interpreter_result.engine.create_serialization_config()
    # for TensorRT >= 10.14, we save weight-stripped engine in cache
    if hasattr(trt.SerializationFlag, "INCLUDE_REFIT"):
        serialization_config.set_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)
    serialized_engine_in_cache = interpreter_result.engine.serialize_with_config(
        serialization_config
    )

    # Insert weight-stripped engine to cache
    engine_cache.insert(
        hash_val,
        (
            serialized_engine_in_cache,
            interpreter_result.input_names,
            interpreter_result.output_names,
            inputs,
            settings,
            interpreter_result.weight_name_map,
            interpreter_result.requires_output_allocator,
        ),
    )
    logger.info(f"Engine with hash: {hash_val} was successfully inserted into cache")
    return True


def pull_cached_engine(
    hash_val: Optional[str],
    module: torch.fx.GraphModule,
    engine_cache: BaseEngineCache,
    settings: CompilationSettings,
    inputs: Sequence[Input],
) -> Optional[SerializedInterpreterResult]:
    if hash_val is None:
        logger.warning(
            "Hash value is not provided, so the engine cache will not be used"
        )
        return None

    if not ENABLED_FEATURES.refit:
        logger.warning(
            "Refit feature is not available, so the engine cache will not be used"
        )
        return None

    # query the cached TRT engine
    cached_data = engine_cache.check(hash_val)
    if cached_data is not None:  # hit the cache
        (
            serialized_engine,  # weight-stripped engine
            input_names,
            output_names,
            cached_engine_inputs,
            cached_engine_compilation_settings,
            weight_name_map,
            requires_output_allocator,
        ) = cached_data

        setting_compatiblity, incompattible_settings = settings_are_compatible(
            settings, cached_engine_compilation_settings
        )
        assert (
            setting_compatiblity
        ), f"Attempted to refit a cached engine with incompatible settings: {incompattible_settings}, (old_settings: {cached_engine_compilation_settings}, new_settings: {settings})"

        for i, e in enumerate(
            [Input.equivalent_spec(c, i) for c, i in zip(cached_engine_inputs, inputs)]
        ):
            assert (
                e
            ), f"Attempted to refit a cached engine built for a different input size (input: {i}, cached size: {cached_engine_inputs[i]}, new size: {inputs[i]}"

        logger.info(
            f"Found the cached engine with hash {hash_val} that corresponds to this graph. It is directly loaded."
        )

        # refit the cached engine with the new graph module
        if not settings.strip_engine_weights:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(
                serialized_engine
            )  # weight-stripped engine

            from torch_tensorrt.dynamo._refit import (
                _refit_single_trt_engine_with_gm,
            )

            # weight-stripped engine --in place--> weight-included engine
            _refit_single_trt_engine_with_gm(
                new_gm=module,
                old_engine=engine,
                input_list=inputs,
                settings=settings,
                weight_name_map=weight_name_map,
            )

            serialization_config = engine.create_serialization_config()
            serialization_config.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)
            # for TensorRT >= 10.14, we set INCLUDE_REFIT flag to make the engine refittable
            if hasattr(trt.SerializationFlag, "INCLUDE_REFIT"):
                serialization_config.set_flag(trt.SerializationFlag.INCLUDE_REFIT)
            serialized_engine = engine.serialize_with_config(serialization_config)

            del engine
            with io.BytesIO() as engine_bytes:
                engine_bytes.write(serialized_engine)
                serialized_engine = engine_bytes.getvalue()

        return SerializedInterpreterResult(
            serialized_engine=serialized_engine,
            input_names=input_names,
            output_names=output_names,
            weight_name_map=weight_name_map,
            requires_output_allocator=requires_output_allocator,
        )
    return None


def interpret_module_to_result(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
    engine_cache: Optional[BaseEngineCache] = None,
) -> SerializedInterpreterResult:
    """Interpret an FX module to a TRTInterpreterResult
    Args:
        module: FX GraphModule to interpret
        inputs: It requires a sequence of FLATTENED Inputs representing inputs to the module. It should include both arg_inputs and kwarg_inputs, if applicable.
        settings: Compilation settings
        engine_cache: Engine cache instance
    Returns:
        SerializedInterpreterResult
    """

    # engine_cache could be None if:
    # 1) engine_cache is not passed in when calling this function like convert_exported_program_to_serialized_trt_engine etc., or
    # 2) both cache_built_engines and reuse_cached_engines are False

    is_engine_caching_supported = (
        engine_cache is not None
        and ENABLED_FEATURES.refit
        and not settings.immutable_weights
    )
    # calculate the hash only once. It will be used in pulling and inserting the engine.
    hash_val = (
        engine_cache.get_hash(module, inputs, settings)  # type: ignore
        if is_engine_caching_supported
        and (settings.cache_built_engines or settings.reuse_cached_engines)
        else None
    )

    if settings.reuse_cached_engines:
        if engine_cache is None:
            logger.warning(
                "Engine cache is not provided, so the engine will not be reused from cache"
            )
        elif not ENABLED_FEATURES.refit:
            logger.warning(
                "Refit feature is not available, so the engine will not be reused from cache"
            )
        elif settings.immutable_weights:
            logger.warning(
                "The engine weights are immutable, so the engine will not be reused from cache"
            )
        else:
            serialized_interpreter_result = pull_cached_engine(
                hash_val, module, engine_cache, settings, inputs
            )
            if serialized_interpreter_result is not None:  # hit the cache
                return serialized_interpreter_result

    output_dtypes = infer_module_output_dtypes(
        module, truncate_double=settings.truncate_double
    )

    interpreter = TRTInterpreter(
        module,
        inputs,
        output_dtypes=output_dtypes,
        compilation_settings=settings,
        engine_cache=engine_cache,
    )

    interpreter_result = interpreter.run()
    # Delete the frozen parameters from the module to release CPU memory
    del interpreter
    for attr in dir(module):
        if attr.startswith("_frozen_param"):
            delattr(module, attr)
    release_host_and_device_memory()
    logger.debug(
        f"CPU memory usage after clearing frozen parameters and building memory in conversion: {get_cpu_memory_usage()} MB"
    )

    if settings.cache_built_engines:
        if engine_cache is None:
            logger.warning(
                "Engine cache is not provided, so the engine will not be cached"
            )
        elif not ENABLED_FEATURES.refit:
            logger.warning(
                "Refit feature is not available, so the engine will not be cached"
            )
        elif settings.immutable_weights:
            logger.warning(
                "The engine weights are immutable, so the engine will not be cached"
            )
        else:
            _ = insert_engine_to_cache(
                hash_val, interpreter_result, engine_cache, settings, inputs
            )

    serialized_engine = interpreter_result.engine.serialize()
    with io.BytesIO() as engine_bytes:
        engine_bytes.write(serialized_engine)
        serialized_engine = engine_bytes.getvalue()
        logger.debug(
            f"CPU memory usage after serializing engine: {get_cpu_memory_usage()} MB"
        )

    serialized_interpreter_result = SerializedInterpreterResult(
        serialized_engine=serialized_engine,
        input_names=interpreter_result.input_names,
        output_names=interpreter_result.output_names,
        weight_name_map=interpreter_result.weight_name_map,
        requires_output_allocator=interpreter_result.requires_output_allocator,
    )

    return serialized_interpreter_result


def convert_module(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
    name: str = "",
    engine_cache: Optional[BaseEngineCache] = None,
) -> PythonTorchTensorRTModule | TorchTensorRTModule:
    """Convert an FX module to a TRT module
    Args:
        module: FX GraphModule to convert
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
        name: TRT engine name
        engine_cache: Engine cache instance
    Returns:
        PythonTorchTensorRTModule or TorchTensorRTModule
    """
    serialized_interpreter_result = interpret_module_to_result(
        module, inputs, settings, engine_cache=engine_cache
    )

    rt_cls = PythonTorchTensorRTModule

    if ENABLED_FEATURES.torch_tensorrt_runtime and not settings.use_python_runtime:
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        rt_cls = TorchTensorRTModule

    elif (
        not ENABLED_FEATURES.torch_tensorrt_runtime and not settings.use_python_runtime
    ):
        logger.info(
            "Since Torch-TensorRT runtime is not available, using Python Runtime, some features may not be available"
        )

    return rt_cls(
        serialized_engine=serialized_interpreter_result.serialized_engine,
        input_binding_names=list(serialized_interpreter_result.input_names),
        output_binding_names=list(serialized_interpreter_result.output_names),
        name=name,
        settings=settings,
        weight_name_map=serialized_interpreter_result.weight_name_map,
        requires_output_allocator=serialized_interpreter_result.requires_output_allocator,
    )
