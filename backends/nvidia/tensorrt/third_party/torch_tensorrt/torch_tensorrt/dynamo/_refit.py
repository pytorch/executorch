from __future__ import annotations

import collections.abc
import copy
import gc
import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import tensorrt as trt
import torch
from torch.export import ExportedProgram
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import needs_refit
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo._exporter import inline_torch_modules
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._conversion import infer_module_output_dtypes
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter
from torch_tensorrt.dynamo.conversion.impl.normalization.ops import (
    batch_norm_constant_folding,
)
from torch_tensorrt.dynamo.conversion.truncate_double import repair_double_inputs
from torch_tensorrt.dynamo.lowering import (
    clean_up_graph_after_modifications,
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)
from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
    PythonTorchTensorRTModule,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    ENGINE_IDX,
    SERIALIZED_METADATA_IDX,
    TorchTensorRTModule,
)
from torch_tensorrt.dynamo.utils import (
    CPU_DEVICE,
    check_module_output,
    get_model_device,
    get_torch_inputs,
    to_torch_device,
    to_torch_tensorrt_device,
)
from torch_tensorrt.logging import TRT_LOGGER

logger = logging.getLogger(__name__)


@needs_refit  # type: ignore[misc]
def construct_refit_mapping(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
) -> dict[str, np.ndarray]:
    """Find out the weight mapping between weight in exported program and TensorRT engine
    Args:
        module: FX GraphModule to interpret
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
    Returns:
        Mapping from weight name in TensorRT to actual weight value in np.ndarray
    """

    output_dtypes = infer_module_output_dtypes(
        module,
        truncate_double=settings.truncate_double,
    )

    # Use Interpreter
    interpreter = TRTInterpreter(
        module,
        inputs,
        output_dtypes=output_dtypes,
        compilation_settings=settings,
    )
    interpreter._construct_trt_network_def()
    weight_refit_map: dict[str, torch.Tensor] = interpreter.ctx.weight_refit_map

    return weight_refit_map


@needs_refit  # type: ignore[misc]
def construct_refit_mapping_from_weight_name_map(
    weight_name_map: dict[Any, Any],
    state_dict: dict[Any, Any],
    settings: CompilationSettings,
) -> dict[Any, Any]:
    engine_weight_map = {}
    for engine_weight_name, (sd_weight_name, np_weight_type) in weight_name_map.items():
        # Add more constant folding converters here
        trt_dtype = dtype._from(np_weight_type).to(trt.DataType)
        torch_dtype = dtype._from(np_weight_type).to(torch.dtype)
        if engine_weight_name.split(" ")[-1] in ["SCALE", "SHIFT"]:
            # Batch Norm Layer
            params = {}
            for w in sd_weight_name:
                params[w.split(".")[-1]] = state_dict[w].cuda()
            # Batch norm constant folding

            scale, shift = batch_norm_constant_folding(**params, eps=1e-5)
            # Set scale to scale or shift to shift
            engine_weight_map[engine_weight_name] = eval(
                engine_weight_name.split(" ")[-1].lower()
            )

        elif sd_weight_name not in state_dict:
            # If weights is not in sd, we can leave it unchanged
            continue
        else:

            engine_weight_map[engine_weight_name] = state_dict[sd_weight_name].to(
                to_torch_device(settings.device)
            )

        engine_weight_map[engine_weight_name] = (
            engine_weight_map[engine_weight_name]
            .clone()
            .reshape(-1)
            .contiguous()
            .to(torch_dtype),
            trt_dtype,
        )

    return engine_weight_map


@needs_refit  # type: ignore[misc]
def _refit_single_trt_engine_with_gm(
    new_gm: torch.fx.GraphModule,
    old_engine: trt.ICudaEngine,
    input_list: Sequence[Any],
    settings: CompilationSettings = CompilationSettings(),
    weight_name_map: Optional[dict[str, List[str]]] = None,
) -> None:
    """
    Refit a TensorRT Engine in place
    """

    with unset_fake_temporarily():
        refitted = set()
        torch_device = get_model_device(new_gm)
        refitter = trt.Refitter(old_engine, TRT_LOGGER)
        weight_list = refitter.get_all_weights()

        if weight_name_map:
            # Get the refitting mapping
            trt_wt_location = (
                trt.TensorLocation.DEVICE
                if torch_device.type == "cuda"
                else trt.TensorLocation.HOST
            )

            constant_mapping: dict[str, Any] = weight_name_map.pop(
                "constant_mapping", {}
            )  # type: ignore
            mapping = construct_refit_mapping_from_weight_name_map(
                weight_name_map, new_gm.state_dict(), settings
            )
            constant_mapping_with_type = {}

            for constant_name, val in constant_mapping.items():
                np_weight_type = val.dtype
                val_tensor = torch.from_numpy(val).cuda()
                trt_dtype = dtype._from(np_weight_type).to(trt.DataType)
                torch_dtype = dtype._from(np_weight_type).to(torch.dtype)
                constant_mapping_with_type[constant_name] = (
                    val_tensor.clone().reshape(-1).contiguous().to(torch_dtype),
                    trt_dtype,
                )

            mapping.update(constant_mapping_with_type)

            for layer_name in weight_list:
                if layer_name not in mapping:
                    logger.warning(f"{layer_name} is not found in weight mapping.")
                    continue
                # Use Numpy to create weights
                weight, weight_dtype = mapping[layer_name]
                trt_wt_tensor = trt.Weights(
                    weight_dtype, weight.data_ptr(), torch.numel(weight)
                )
                refitter.set_named_weights(layer_name, trt_wt_tensor, trt_wt_location)
            assert (
                len(refitter.get_missing_weights()) == 0
            ), "Fast refitting failed due to incomplete mapping"

        else:
            mapping = construct_refit_mapping(new_gm, input_list, settings)
            trt_wt_location = trt.TensorLocation.HOST
            for layer_name in weight_list:
                if layer_name not in mapping:
                    raise AssertionError(f"{layer_name} is not found in weight mapping")
                # Use Tensor to create weights
                weight = mapping[layer_name]
                trt_dtype = dtype._from(weight.dtype).to(trt.DataType)
                trt_wt_tensor = trt.Weights(
                    trt_dtype, weight.data_ptr(), torch.numel(weight)
                )
                refitter.set_named_weights(layer_name, trt_wt_tensor, trt_wt_location)
                refitted.add(layer_name)

            if len(refitted) != len(weight_list):
                logger.warning("Not all weights have been refitted!!!")

        if not refitter.refit_cuda_engine():
            logger.error("Error: failed to refit new weights.")
            raise AssertionError("Refitting failed.")


@needs_refit  # type: ignore[misc]
def refit_module_weights(
    compiled_module: torch.fx.GraphModule | ExportedProgram,
    new_weight_module: ExportedProgram,
    arg_inputs: Optional[Tuple[Any, ...]] = None,
    kwarg_inputs: Optional[dict[str, Any]] = None,
    verify_output: bool = False,
    use_weight_map_cache: bool = True,
    in_place: bool = False,
) -> torch.fx.GraphModule:
    """
    Refit a compiled graph module with ExportedProgram. This performs weight updates in compiled_module without recompiling the engine.

    Args:
        compiled_module: compiled TensorRT module that needs to be refitted.
                        This compiled_module should be compmiled by torch_tensorrt.dynamo.compile
                        or load it from disk using trt.load.
        new_weight_module: exported program with the updated weights. This one should have the same model architecture as the compiled module.
        arg_inputs: sample arg inputs. Optional, needed if output check
        kwarg_inputs: sample kwarg inputs. Optional, needed if output check
        verify_output: whether to verify output of refitted module
    Returns:
        A new compiled TensorRT module that has the updated weights.
    """
    inline_module = False
    if isinstance(compiled_module, ExportedProgram):
        compiled_module = compiled_module.module()

    if len(list(compiled_module.named_children())) == 0:
        inline_module = True

    if not in_place:
        compiled_module = copy.deepcopy(compiled_module)
    elif inline_module:
        raise AssertionError(
            "Exported program does not support modifying in place. Please set in_place to false and use the returned graph module."
        )

    # Get the settings and check the setting to be uniform
    settings: Optional[CompilationSettings] = None
    if inline_module:
        # Obtain the settings
        compiled_submodules = [
            (name.replace("_engine", ""), engine)
            for name, engine in compiled_module.__dict__.items()
            if "engine" in name
        ]
        # [('_run_on_acc_0', inline_module)]
        encoded_metadata = compiled_submodules[0][1].__getstate__()[0][
            SERIALIZED_METADATA_IDX
        ]
        assert (
            encoded_metadata != ""
        ), "The engine provided is either not refittable or was built with a version of Torch-TensorRT that is too old, please recompile using the latest version"
        settings = TorchTensorRTModule.decode_metadata(encoded_metadata)["settings"]
        # Handle torch modules
        compiled_submodules_map = dict(compiled_submodules)
        for name, submodule in compiled_module.named_children():
            compiled_submodules_map[name] = submodule

    else:
        # Handle torch modules
        compiled_submodules_map = {}
        guard_fn_modules = []
        for name, submodule in compiled_module.named_children():
            if (
                not isinstance(
                    submodule,
                    (
                        PythonTorchTensorRTModule,
                        TorchTensorRTModule,
                        torch.nn.modules.module.Module,
                    ),
                )
                or "_run_on_gpu" in name
            ):
                continue

            # When we re-export the graph module, torch.export._unlift.GuardsFn modules are being added as submodules.
            if isinstance(submodule, torch.export._unlift.GuardsFn):
                guard_fn_modules.append(name)
                continue
            # Obtain the settings

            compiled_submodules = [
                (name.replace("_engine", ""), engine)
                for name, engine in submodule.__dict__.items()
                if "engine" in name
            ]

            settings = None
            try:
                # If the gm is not inlined or transformed by retracing, the settings is stored in the submodule
                settings = submodule.settings
            except AttributeError:

                encoded_metadata = [
                    engine for name, engine in compiled_submodules if name == "engine"
                ][0].__getstate__()[0][SERIALIZED_METADATA_IDX]
                assert (
                    encoded_metadata != ""
                ), "The engine provided is either not refittable or was built with a version of Torch-TensorRT that is too old, please recompile using the latest version"
                settings = TorchTensorRTModule.decode_metadata(encoded_metadata)[
                    "settings"
                ]

            compiled_submodules_map[name] = submodule

        # Delete the guard fn modules to avoid the guard fn modules being refitted
        # First, remove nodes in the graph that reference the guard function modules
        for node in list(compiled_module.graph.nodes):
            if node.op == "call_module" and node.target in guard_fn_modules:
                compiled_module.graph.erase_node(node)

        # Now delete the submodules themselves
        for guard_fn_module_name in guard_fn_modules:
            # delattr(compiled_module, guard_fn_module_name)
            compiled_module.delete_submodule(guard_fn_module_name)

        # Clean up the graph
        clean_up_graph_after_modifications(compiled_module)

    assert settings is not None

    assert (
        not settings.immutable_weights
    ), "Refitting is not enabled. Please recompile the engine with immutable_weights=False."

    device = to_torch_tensorrt_device(settings.device)
    if arg_inputs:
        if not isinstance(arg_inputs, collections.abc.Sequence):
            # Prepare torch_trt inputs
            arg_inputs = [arg_inputs]
        torch_inputs = get_torch_inputs(arg_inputs, device)

    torch_kwarg_inputs: Any = {}
    if kwarg_inputs:
        torch_kwarg_inputs = get_torch_inputs(kwarg_inputs, device)
    runtime = trt.Runtime(TRT_LOGGER)
    if not isinstance(new_weight_module, ExportedProgram):
        raise AssertionError(
            f"Input graph should be an ExportedProgram but got type {type(new_weight_module)}"
        )
    new_weight_module = pre_export_lowering(new_weight_module, settings)
    new_weight_module = new_weight_module.run_decompositions(
        get_decompositions(settings.enable_experimental_decompositions)
    )
    new_gm = new_weight_module.module()

    logger.debug("Input graph: " + str(new_gm.graph))
    # Apply lowering on the graph module

    new_gm = post_lowering(new_gm, settings)

    logger.debug("Lowered Input graph: " + str(new_gm.graph))

    # Set torch-executed ops
    CONVERTERS.set_compilation_settings(settings)

    # Check the number of supported operations in the graph
    num_supported_ops, total_ops = partitioning.get_graph_converter_support(
        new_gm, settings.torch_executed_ops
    )

    if num_supported_ops == 0 or (
        num_supported_ops < settings.min_block_size and not settings.dryrun
    ):
        logger.warning(
            f"{num_supported_ops} supported operations detected in subgraph containing {total_ops} computational nodes. "
            f"Skipping this subgraph, since min_block_size was detected to be {settings.min_block_size}"
        )
        return new_gm
    else:
        logger.debug(
            f"Detected support for {num_supported_ops} operators out of {total_ops} in subgraph."
        )

    # If specified, try using the fast partitioner and fall back to the global one on failure
    if settings.use_fast_partitioner:
        try:
            logger.info("Partitioning the graph via the fast partitioner")
            new_partitioned_module, supported_ops = partitioning.fast_partition(
                new_gm,
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

            settings.use_fast_partitioner = False

    if not settings.use_fast_partitioner:
        logger.info("Partitioning the graph via the global partitioner")
        new_partitioned_module, supported_ops = partitioning.global_partition(
            new_gm,
            min_block_size=settings.min_block_size,
            torch_executed_ops=settings.torch_executed_ops,
            require_full_compilation=settings.require_full_compilation,
        )

    # Done Partition
    if inline_module:
        # Preprocess the partitioned module to be in the same format as the inline module
        inline_torch_modules(new_partitioned_module)
        new_partitioned_module.delete_all_unused_submodules()
        # Check the number of partitions and name
        assert {sm[0] for sm in new_partitioned_module.named_children()} == set(
            compiled_submodules_map.keys()
        ), "New weights module is not compatible with previously compiled Torch-TensorRT module"
    else:
        assert {sm[0] for sm in new_partitioned_module.named_children()} == {
            sm[0] for sm in compiled_module.named_children()
        }, "New weights module is not compatible with previously compiled Torch-TensorRT module"
    # 2. TODO: Check the hash of source fx.Graph and new fx.Graph

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, new_submodule in new_partitioned_module.named_children():
        # Refit each submodule
        # Extract engine from the submodule
        try:
            if inline_module:
                weight_name_map = None
                compiled_submodule = compiled_submodules_map[name]
                # If this is a torch module, load the old state_dict
                if "_run_on_acc" not in name:
                    compiled_submodule.load_state_dict(new_submodule.state_dict())
                    continue
                else:
                    engine_info = compiled_submodule.__getstate__()[0]
                    engine = get_engine_from_encoded_engine(
                        engine_info[ENGINE_IDX], runtime
                    )
                    if use_weight_map_cache:
                        encoded_metadata = compiled_submodule.__getstate__()[0][
                            SERIALIZED_METADATA_IDX
                        ]
                        weight_name_map = TorchTensorRTModule.decode_metadata(
                            encoded_metadata
                        )["weight_name_map"]
                        if not weight_name_map:
                            use_weight_map_cache = False
                            logger.warning(
                                "This engine does not have a weight map cache. Rebuilding the weight map"
                            )
            else:
                compiled_submodule = getattr(compiled_module, name)
                if "_run_on_acc" not in name:
                    compiled_submodule.load_state_dict(new_submodule.state_dict())
                    continue

                weight_name_map = None
                if use_weight_map_cache:
                    try:
                        weight_name_map = compiled_submodule.weight_name_map
                    except AttributeError:
                        if isinstance(compiled_submodule, torch.nn.Module):
                            # Torch retrace module
                            assert (
                                not settings.use_python_runtime
                            ), "Refitting a torch retraced module is only supported with use_python_runtime=False"
                            encoded_metadata = [
                                engine
                                for name, engine in compiled_submodules
                                if name == "engine"
                            ][0].__getstate__()[0][SERIALIZED_METADATA_IDX]
                            weight_name_map = TorchTensorRTModule.decode_metadata(
                                encoded_metadata
                            )["weight_name_map"]

                        if not isinstance(
                            compiled_submodule, torch.fx.graph_module.GraphModule
                        ):
                            logger.warning(
                                "The module was compiled with an old version of Torch-TensorRT. Rebuilding the weight map."
                            )
                    if not weight_name_map:
                        use_weight_map_cache = False
                        logger.warning(
                            "This engine does not have a weight map cache. Rebuilding the weight map"
                        )

                # Rexporting the TRT compiled graph module and loading it back doesn't preserve the instance type and registers
                # the compiled submodule as torch.nn.Module. So we use settings.use_python_runtime to determine the instance type.
                if settings.use_python_runtime:
                    engine = compiled_submodule.engine
                else:
                    engine_info = compiled_submodule.engine.__getstate__()[0]
                    engine = get_engine_from_encoded_engine(
                        engine_info[ENGINE_IDX], runtime
                    )
        except AttributeError:
            raise AssertionError(
                "The type of graph module is not supported for refitting or two compiled modules do not match."
            )

        # Get the submodule inputs for min, opt, max shapes of the graph inputs
        submodule_inputs = partitioning.construct_submodule_inputs(new_submodule)
        logger.debug(
            "Refitting Submodule name: %s\n",
            str(name),
        )
        assert submodule_inputs is not None
        # Handle long/double inputs if requested by the user
        if settings.truncate_double:
            submodule_inputs = repair_double_inputs(
                new_partitioned_module,
                new_submodule,
                submodule_inputs,
                to_torch_device(settings.device),
                name,
            )
        try:
            _refit_single_trt_engine_with_gm(
                new_gm=new_submodule,
                old_engine=engine,
                input_list=submodule_inputs,
                settings=settings,
                weight_name_map=weight_name_map,
            )

        except AssertionError as e:
            # If fast_refit is used and failed, we fall back to regular refit
            logger.warning(e)
            if use_weight_map_cache and weight_name_map:
                _refit_single_trt_engine_with_gm(
                    new_gm=new_submodule,
                    old_engine=engine,
                    input_list=submodule_inputs,
                    settings=settings,
                    weight_name_map=None,
                )

        # clear EXCLUDE_WEIGHTS flag and set INCLUDE_REFIT flag to make the engine refittable
        serialization_config = engine.create_serialization_config()
        serialization_config.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)
        # INCLUDE_REFIT flag is only available as of TensorRT 10.14
        if hasattr(trt.SerializationFlag, "INCLUDE_REFIT"):
            serialization_config.set_flag(trt.SerializationFlag.INCLUDE_REFIT)
        serialized_engine = engine.serialize_with_config(serialization_config)

        if isinstance(compiled_submodule, PythonTorchTensorRTModule):
            compiled_submodule.serialized_engine = bytes(serialized_engine)
        elif isinstance(compiled_submodule, TorchTensorRTModule):
            compiled_submodule.engine = None  # Clear the engine for TorchTensorRTModule, otherwise it won't be updated
            compiled_submodule.serialized_engine = bytes(serialized_engine)
            compiled_submodule.setup_engine()
        elif inline_module:
            new_engine_info = list(engine_info)
            new_engine_info[ENGINE_IDX] = bytes(serialized_engine)
            refitted_engine = torch.classes.tensorrt.Engine(tuple(new_engine_info))
            setattr(compiled_module, f"{name}_engine", refitted_engine)
        elif isinstance(compiled_submodule, torch.nn.Module):
            # Torch retrace module
            new_engine_info = list(engine_info)
            new_engine_info[ENGINE_IDX] = bytes(serialized_engine)
            refitted_engine = torch.classes.tensorrt.Engine(tuple(new_engine_info))
            compiled_submodule.engine = refitted_engine
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    if verify_output and arg_inputs is not None:
        new_gm.to(to_torch_device(settings.device))
        if check_module_output(
            new_module=new_gm,
            refitted_module=compiled_module,
            arg_inputs=torch_inputs,
            kwarg_inputs=torch_kwarg_inputs,
        ):
            logger.info("Refitting Succeed!")
            new_gm.to(CPU_DEVICE)
        else:
            if weight_name_map:
                logger.warning(
                    "Refitting with weight_name_map yielded incorrect result! The outputs do not match."
                )
                return refit_module_weights(
                    compiled_module,
                    new_weight_module,
                    arg_inputs,
                    kwarg_inputs,
                    verify_output,
                    use_weight_map_cache=False,
                    in_place=in_place,
                )
            logger.error("Refitting Failed! The outputs do not match.")
            new_gm.to(CPU_DEVICE)
    else:
        logger.info("Refitting Completed! Output verification skipped.")

    return compiled_module


# Util functions -----------
import base64


def get_engine_from_encoded_engine(
    encoded_engine: str, runtime: trt.Runtime
) -> trt.ICudaEngine:
    serialized_engine = base64.b64decode(encoded_engine)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine
