#  Copyright © 2023 Apple Inc. All rights reserved.

# CoreML backend for delegating a EdgeProgram to CoreML.

import json
import logging

import shutil
import uuid
from dataclasses import asdict, dataclass
from enum import Enum

from pathlib import Path

from typing import Any, Dict, final, List, Optional, Tuple

import coremltools as ct
import coremltools.optimize as cto
import executorchcoreml

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class COMPILE_SPEC_KEYS(Enum):
    COMPUTE_UNITS = "compute_units"
    MODEL_TYPE = "model_type"
    MIN_DEPLOYMENT_TARGET = "min_deployment_target"
    MODEL_COMPUTE_PRECISION = "model_compute_precision"
    OP_LINEAR_QUANTIZER_CONFIG = "op_linear_quantizer_config"


class MODEL_PATHS(Enum):
    MODEL = "model.mlpackage"
    COMPILED_MODEL = "model.mlmodelc"
    METADATA = "metadata.json"
    DEBUG_INFO = "debug_info.json"


@dataclass
class ModelMetadata:
    # The model input names.
    inputNames: List[str]
    # The model output names.
    outputNames: List[str]
    # The model identifier.
    identifier: str


@dataclass
class ModelDebugInfo:
    # Version info.
    versionInfo: Dict[str, str]
    # Mapping from debug symbol to operation path.
    debugSymbolToOperationPath: Dict[str, List[Dict[str, str]]]
    # Mapping from debug symbol to handle.
    debugSymbolToHandles: Dict[str, List[int]]


@final
class CoreMLBackend(BackendDetails):
    class MODEL_TYPE(Enum):
        MODEL = "model"
        COMPILED_MODEL = "compiled_model"

    @staticmethod
    def generate_model_type_compile_spec(model_type: MODEL_TYPE) -> CompileSpec:
        """
        Returns the compile spec representing the given model type.

        If the model type is ``MODEL_TYPE.Model`` then the ``CoreMLBackend`` returns
        the in-memory representation of the ``mlpackage`` contents.

        If the model type is ``MODEL_TYPE.COMPILED_MODEL`` then the ``CoreMLBackend`` compiles the model
        and returns the in-memory representation of ``mlmodelc`` (compiled model) contents.
        """
        return CompileSpec(
            COMPILE_SPEC_KEYS.MODEL_TYPE.value, model_type.value.encode("utf-8")
        )

    @staticmethod
    def model_type_from_compile_specs(compile_specs: List[CompileSpec]) -> MODEL_TYPE:
        """
        Returns the model type by parsing the list of compile specs.
        """
        for compile_spec in compile_specs:
            if compile_spec.key == COMPILE_SPEC_KEYS.MODEL_TYPE.value:
                return CoreMLBackend.MODEL_TYPE(compile_spec.value.decode("utf-8"))

        return CoreMLBackend.MODEL_TYPE.MODEL

    @staticmethod
    def generate_compute_precision_compile_spec(
        compute_precision: ct.precision,
    ) -> CompileSpec:
        """
        Returns the compile spec representing the model compute precision, for additional details
        please refer to the documentation for ``coremltools.precision``.
        """
        return CompileSpec(
            COMPILE_SPEC_KEYS.MODEL_COMPUTE_PRECISION.value,
            compute_precision.value.encode("utf-8"),
        )

    @staticmethod
    def model_compute_precision_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> ct.precision:
        """
        Returns the model's compute precision by parsing the list of compile specs.
        """
        for compile_spec in compile_specs:
            if compile_spec.key == COMPILE_SPEC_KEYS.MODEL_COMPUTE_PRECISION.value:
                return ct.precision(compile_spec.value.decode("utf-8"))

        return ct.precision.FLOAT16

    @staticmethod
    def generate_minimum_deployment_target_compile_spec(
        min_deployment_target: ct.target,
    ) -> CompileSpec:
        """
        Returns the compile spec representing the minimum deployment target on which the model can run,
        for additional details please refer to the documentation for ``coremltools.target``.
        """
        return CompileSpec(
            COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_TARGET.value,
            str(min_deployment_target.value).encode("utf-8"),
        )

    @staticmethod
    def min_deployment_target_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> ct.target:
        """
        Returns the minimum deployment target by parsing the list of compile specs.
        """
        for compile_spec in compile_specs:
            if compile_spec.key == COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_TARGET.value:
                compile_spec_value: int = int(compile_spec.value.decode("utf-8"))
                return ct.target(compile_spec_value)

        return ct.target.iOS15

    @staticmethod
    def compute_unit_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> ct.ComputeUnit:
        """
        Returns the minimum deployment target by parsing the list of compile specs.
        """
        for compile_spec in compile_specs:
            if compile_spec.key == COMPILE_SPEC_KEYS.COMPUTE_UNITS.value:
                return ct.ComputeUnit[compile_spec.value.decode("utf-8").upper()]

        return ct.ComputeUnit.ALL

    @staticmethod
    def generate_compute_unit_compile_spec(
        compute_unit: ct.ComputeUnit,
    ) -> CompileSpec:
        """
        Returns the compile spec representing the compute units on which the model can run, for additional details
        please refer to the documentation for ``coremltools.ComputeUnit`.
        """
        return CompileSpec(
            COMPILE_SPEC_KEYS.COMPUTE_UNITS.value,
            compute_unit.name.lower().encode("utf-8"),
        )

    @staticmethod
    def generate_op_linear_quantizer_config_compile_spec(
        op_linear_quantizer_config: Dict,
    ) -> CompileSpec:
        """
        Returns the compile spec representing the model post conversion quantization,
        which is a dict that will construct cto.coreml.OpLinearQuantizerConfig
        """
        str_representation = json.dumps(op_linear_quantizer_config)
        byte_representation = str_representation.encode("utf-8")
        return CompileSpec(
            COMPILE_SPEC_KEYS.OP_LINEAR_QUANTIZER_CONFIG.value,
            byte_representation,
        )

    @staticmethod
    def op_linear_quantizer_config_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> cto.coreml.OpLinearQuantizerConfig:
        """
        Returns the model's post conversion quantization by parsing the list of compile specs.
        """
        for compile_spec in compile_specs:
            if compile_spec.key == COMPILE_SPEC_KEYS.OP_LINEAR_QUANTIZER_CONFIG.value:
                config_dict_str = compile_spec.value.decode("utf-8")
                config_dict = json.loads(config_dict_str)
                config = cto.coreml.OpLinearQuantizerConfig._from_dict(config_dict)
                return config

        return None

    @staticmethod
    def generate_compile_specs(
        compute_unit: ct.ComputeUnit = ct.ComputeUnit.ALL,
        minimum_deployment_target: ct.target = ct.target.iOS15,
        compute_precision: ct.precision = ct.precision.FLOAT16,
        model_type: MODEL_TYPE = MODEL_TYPE.MODEL,
        op_linear_quantizer_config: Optional[Dict] = None,
    ) -> List[CompileSpec]:
        """
        Returns the list of compile specs that's used by CoreMLBackend to lower the module.
        """
        compile_specs: List[CompileSpec] = []
        compile_specs.append(
            CoreMLBackend.generate_compute_unit_compile_spec(compute_unit)
        )
        compile_specs.append(
            CoreMLBackend.generate_minimum_deployment_target_compile_spec(
                minimum_deployment_target
            )
        )
        compile_specs.append(
            CoreMLBackend.generate_compute_precision_compile_spec(compute_precision)
        )
        compile_specs.append(CoreMLBackend.generate_model_type_compile_spec(model_type))
        if op_linear_quantizer_config is not None:
            compile_specs.append(
                CoreMLBackend.generate_op_linear_quantizer_config_compile_spec(
                    op_linear_quantizer_config
                )
            )

        return compile_specs

    @staticmethod
    def model_metadata_from_spec(
        model_spec: ct.proto.Model_pb2, identifier: str  # pyre-ignore
    ) -> ModelMetadata:
        input_names: List[str] = [input.name for input in model_spec.description.input]
        output_names = [output.name for output in model_spec.description.output]

        return ModelMetadata(
            inputNames=input_names, outputNames=output_names, identifier=identifier
        )

    @staticmethod
    def get_debug_symbol(operation_path: List[Dict[str, str]]) -> Optional[str]:
        if len(operation_path) == 0:
            return None

        operator_name: Optional[str] = operation_path[-1].get("Operator", None)
        output_name: Optional[str] = operation_path[-1].get("Output", None)
        if output_name is None or operator_name is None:
            return None

        return output_name + ":" + operator_name

    @staticmethod
    def get_model_debug_info(model_package_dir: Path) -> Optional[ModelDebugInfo]:
        delegate_info_file = model_package_dir / "executorch_debug_handle_mapping.json"

        if not delegate_info_file.is_file():
            return None

        delegate_info: Optional[Dict[str, Any]] = None

        try:
            with open(delegate_info_file) as f:
                delegate_info = json.load(f)
        except ValueError:
            return None

        if delegate_info is None:
            return None

        debug_handle_to_operation_path_mapping: Optional[Dict[str, Any]] = (
            delegate_info.get("mapping", None)
        )

        if debug_handle_to_operation_path_mapping is None:
            return None

        debug_symbol_to_operation_path: Dict[str, List[Dict[str, str]]] = {}
        debug_symbol_to_handles: Dict[str, List[int]] = {}
        for (
            debug_handle,
            operation_paths,
        ) in debug_handle_to_operation_path_mapping.items():
            debug_handle_value: Optional[int] = None
            try:
                debug_handle_value = int(debug_handle)
            except ValueError:
                debug_handle_value = None

            if debug_handle_value is None:
                continue

            for operation_path in operation_paths:
                debug_symbol: Optional[str] = CoreMLBackend.get_debug_symbol(
                    operation_path=operation_path
                )

                if debug_symbol is None:
                    continue

                debug_handle_values: List[int] = debug_symbol_to_handles.get(
                    debug_symbol, []
                )
                debug_handle_values.append(debug_handle_value)
                debug_symbol_to_handles[debug_symbol] = debug_handle_values

                debug_symbol_to_operation_path[debug_symbol] = operation_path

        version_info: Dict[str, str] = delegate_info.get("version", {})

        return ModelDebugInfo(
            versionInfo=version_info,
            debugSymbolToOperationPath=debug_symbol_to_operation_path,
            debugSymbolToHandles=debug_symbol_to_handles,
        )

    @staticmethod
    def save_model_metadata(model_metadata: ModelMetadata, model_dir_path: Path):
        # Store model metadata.
        model_metadata_path = Path(model_dir_path) / MODEL_PATHS.METADATA.value
        model_metadata_json = json.dumps(asdict(model_metadata))
        with open(model_metadata_path, "w") as outfile:
            outfile.write(model_metadata_json)

    @staticmethod
    def save_model_debug_info(model_debug_info: ModelDebugInfo, model_dir_path: Path):
        # Store model debug info.
        model_debug_info_path = Path(model_dir_path) / MODEL_PATHS.DEBUG_INFO.value
        model_debug_info_json = json.dumps(asdict(model_debug_info))
        with open(model_debug_info_path, "w") as outfile:
            outfile.write(model_debug_info_json)

    @staticmethod
    def preprocess_model(
        mlmodel: ct.models.MLModel, model_type: MODEL_TYPE
    ) -> PreprocessResult:
        identifier = "executorch_" + str(uuid.uuid4())
        dir_path: Path = Path("tmp") / identifier
        model_dir_path: Path = dir_path / "lowered_module"
        model_spec: ct.proto.Model_pb2 = mlmodel.get_spec()
        model_metadata: ModelMetadata = CoreMLBackend.model_metadata_from_spec(
            model_spec=model_spec,
            identifier=identifier,
        )

        # Save model.
        model_path = model_dir_path / MODEL_PATHS.MODEL.value
        mlmodel.save(str(model_path))
        # Extract delegate mapping file.
        model_debug_info: Optional[ModelDebugInfo] = CoreMLBackend.get_model_debug_info(
            model_path
        )

        match model_type:
            case CoreMLBackend.MODEL_TYPE.COMPILED_MODEL:
                shutil.rmtree(str(model_path.resolve()))
                model_path = model_dir_path / MODEL_PATHS.COMPILED_MODEL.value
                compiled_model_path = mlmodel.get_compiled_model_path()
                shutil.move(
                    compiled_model_path,
                    str(model_path.resolve()),
                )

            case _:
                pass

        CoreMLBackend.save_model_metadata(
            model_metadata=model_metadata, model_dir_path=model_dir_path
        )
        if model_debug_info is not None:
            CoreMLBackend.save_model_debug_info(
                model_debug_info=model_debug_info, model_dir_path=model_dir_path
            )

        processed_bytes: bytes = (
            executorchcoreml.flatten_directory_contents(str(model_dir_path.resolve()))
            or b""
        )

        debug_handle_map: Optional[Dict[str, Tuple[int]]] = None
        if model_debug_info is not None:
            debug_handle_map = {
                key: tuple(value)
                for key, value in model_debug_info.debugSymbolToHandles.items()
            }

        shutil.rmtree(str(dir_path.resolve()))
        return PreprocessResult(
            processed_bytes=processed_bytes,
            debug_handle_map=debug_handle_map,
        )

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        model_type: CoreMLBackend.MODEL_TYPE = (
            CoreMLBackend.model_type_from_compile_specs(
                compile_specs,
            )
        )
        model_compute_precision: ct.precision = (
            CoreMLBackend.model_compute_precision_from_compile_specs(compile_specs)
        )
        minimum_deployment_target: ct.target = (
            CoreMLBackend.min_deployment_target_from_compile_specs(compile_specs)
        )
        compute_units: ct.ComputeUnit = CoreMLBackend.compute_unit_from_compile_specs(
            compile_specs
        )
        op_linear_quantizer_config = (
            CoreMLBackend.op_linear_quantizer_config_from_compile_specs(compile_specs)
        )

        mlmodel = ct.convert(
            model=edge_program,
            source="pytorch",
            convert_to="mlprogram",
            pass_pipeline=ct.PassPipeline.DEFAULT,
            skip_model_load=True,
            compute_precision=model_compute_precision,
            minimum_deployment_target=minimum_deployment_target,
            compute_units=compute_units,
        )

        if op_linear_quantizer_config is not None:
            logger.warning(
                "Core ML Backend op_linear_quantizer_config API is experimental"
            )
            config = cto.coreml.OptimizationConfig(
                global_config=op_linear_quantizer_config,
                # skip embedding
                op_type_configs={"gather": None},
            )
            mlmodel = cto.coreml.linear_quantize_weights(mlmodel, config=config)

        return CoreMLBackend.preprocess_model(mlmodel, model_type=model_type)
