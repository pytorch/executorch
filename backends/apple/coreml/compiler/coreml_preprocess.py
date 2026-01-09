#  Copyright © 2023 Apple Inc. All rights reserved.

# CoreML backend for delegating a EdgeProgram to CoreML.

import json
import logging

import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from enum import Enum

from pathlib import Path

from typing import Any, Dict, final, List, Optional, Tuple

import coremltools as ct
import coremltools.optimize as cto
from executorch.backends.apple.coreml import executorchcoreml
from executorch.backends.apple.coreml.compiler.enumerated_shape_utils import (
    _get_ct_inputs,
    _SymbolicShapeToEnumeratedShapeMap,
)
from executorch.backends.apple.coreml.logging import get_coreml_log_level
from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec

from executorch.backends.apple.coreml.compiler.torch_ops import *  # noqa: F401, F403

logger = logging.getLogger(__name__)
logger.setLevel(get_coreml_log_level(default_level=logging.WARNING))


class COMPILE_SPEC_KEYS(Enum):
    COMPUTE_UNITS = "compute_units"
    MODEL_TYPE = "model_type"
    MIN_DEPLOYMENT_TARGET = "min_deployment_target"
    MODEL_COMPUTE_PRECISION = "model_compute_precision"
    OP_LINEAR_QUANTIZER_CONFIG = "op_linear_quantizer_config"
    ENUMERATED_SHAPES = "enumerated_shapes"
    PASS_PIPELINE = "pass_pipeline"


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
        min_deployment_target: Optional[ct.target],
    ) -> CompileSpec:
        """
        Returns the compile spec representing the minimum deployment target on which the model can run,
        for additional details please refer to the documentation for ``coremltools.target``.
        """
        value = str("").encode("utf-8")
        if min_deployment_target is not None:
            value = str(min_deployment_target.value).encode("utf-8")
        return CompileSpec(
            COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_TARGET.value,
            value,
        )

    @staticmethod
    def min_deployment_target_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> Optional[ct.target]:
        """
        Returns the minimum deployment target by parsing the list of compile specs.
        """
        for compile_spec in compile_specs:
            if compile_spec.key == COMPILE_SPEC_KEYS.MIN_DEPLOYMENT_TARGET.value:
                value = compile_spec.value.decode("utf-8")
                if value == "":
                    return None
                compile_spec_value: int = int(value)
                return ct.target(compile_spec_value)

        return None

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
    def generate_pass_pipeline_compile_spec(pass_names: List[str]) -> CompileSpec:
        """
        Creates a compile spec representing the pass pipeline to be used by the CoreML backend
        :param pass_names: the list of pass names
        """
        str_representation = json.dumps(pass_names)
        byte_representation = str_representation.encode("utf-8")
        return CompileSpec(COMPILE_SPEC_KEYS.PASS_PIPELINE.value, byte_representation)

    @staticmethod
    def pass_pipeline_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> ct.PassPipeline:
        """
        Creates a PassPipeline from the list of compile specs, or returns the default if none are provided.
        """
        for compile_spec in compile_specs:
            if compile_spec.key == COMPILE_SPEC_KEYS.PASS_PIPELINE.value:
                pass_names_str = compile_spec.value.decode("utf-8")
                pass_names = json.loads(pass_names_str)
                return ct.PassPipeline(
                    pass_names, pipeline_name="executorch_user_pipeline"
                )

        return ct.PassPipeline.DEFAULT

    @staticmethod
    def generate_enumerated_shapes_compile_spec(
        ep: ExportedProgram,
        enumerated_shapes: Dict[str, List[List[int]]],
    ) -> CompileSpec:
        """
        Returns the compile spec representing the model enumerated shapes
        enumerated_shapes is a dictionary for each input to its enumerated shapes, e.g.,

        enumerated_shapes = {
         {"x": [[1, 1, 24], [8, 9, 24]]
         {"y": [[1, 6], [30, 6]],
        ]

        means the model can handle x can be shape [1, 1, 24] or [8, 9, 24] and y can be shape [1, 6] or [30, 6].

        Only multiple inputs can have enumerated shapes if using iOS18 or later.
        In this case, each input must have the same number of enumerated shapes, and these shapes are tied together
        by their order in the list. For example, the model above can handle x with shape [1, 1, 24] and y with shape [1, 6],
        or x with shape [8, 9, 24] and y with shape [30, 6], but not x with shape [1, 1, 24] and y with shape [30, 6].

        Passing incorrect shapes at runtime will result in an error.
        """
        emap = _SymbolicShapeToEnumeratedShapeMap.from_exported_program(
            ep,
            enumerated_shapes,
        )
        str_representation = emap.to_json()
        byte_representation = str_representation.encode("utf-8")
        return CompileSpec(
            COMPILE_SPEC_KEYS.ENUMERATED_SHAPES.value,
            byte_representation,
        )

    @staticmethod
    def enumerated_shapes_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> cto.coreml.OpLinearQuantizerConfig:
        """
        Returns the model's post conversion quantization by parsing the list of compile specs.
        """
        for compile_spec in compile_specs:
            if compile_spec.key == COMPILE_SPEC_KEYS.ENUMERATED_SHAPES.value:
                emap_json = compile_spec.value.decode("utf-8")
                emap = _SymbolicShapeToEnumeratedShapeMap.from_json(emap_json)
                return emap
        return None

    @staticmethod
    def generate_compile_specs(
        compute_unit: ct.ComputeUnit = ct.ComputeUnit.ALL,
        minimum_deployment_target: Optional[ct.target] = None,
        compute_precision: ct.precision = ct.precision.FLOAT16,
        model_type: MODEL_TYPE = MODEL_TYPE.MODEL,
        op_linear_quantizer_config: Optional[Dict] = None,
        pass_names: Optional[List[str]] = None,
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
        if pass_names is not None:
            compile_specs.append(
                CoreMLBackend.generate_pass_pipeline_compile_spec(pass_names)
            )

        return compile_specs

    @staticmethod
    def model_metadata_from_spec(
        model_spec: ct.proto.Model_pb2, identifier: str  # pyre-ignore
    ) -> ModelMetadata:
        input_names: List[str] = [input.name for input in model_spec.description.input]
        output_names = [output.name for output in model_spec.description.output]

        if len(output_names) == 0:
            raise ValueError("Cannot lower a model with no outputs in CoreML.")
        if len(input_names) == 0:
            assert (
                model_spec.specificationVersion >= 9
            ), "Deploying a model with no inputs in CoreML requires you set minimum_deployment_target to iOS18 or later in the CoreMLPartitioner."

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
        dir_path: Path = Path(tempfile.gettempdir()) / identifier
        model_dir_path: Path = dir_path / "lowered_module"
        model_spec: ct.proto.Model_pb2 = mlmodel.get_spec()
        logger.warning(
            f"The model with identifier {identifier} was exported with CoreML specification version {model_spec.specificationVersion}, and it will not run on all version of iOS/macOS."
            " See https://apple.github.io/coremltools/mlmodel/Format/Model.html#model for information on what OS versions are compatible with this specifcation version."
            " If you want to control the deployment target, please set the minimum_deployment_target compile spec in the CoreMLPartitioner."
        )

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
        logger.info(f"Edge program: {edge_program}")
        model_type: CoreMLBackend.MODEL_TYPE = (
            CoreMLBackend.model_type_from_compile_specs(
                compile_specs,
            )
        )
        model_compute_precision: ct.precision = (
            CoreMLBackend.model_compute_precision_from_compile_specs(compile_specs)
        )
        minimum_deployment_target: Optional[ct.target] = (
            CoreMLBackend.min_deployment_target_from_compile_specs(compile_specs)
        )
        compute_units: ct.ComputeUnit = CoreMLBackend.compute_unit_from_compile_specs(
            compile_specs
        )
        op_linear_quantizer_config = (
            CoreMLBackend.op_linear_quantizer_config_from_compile_specs(compile_specs)
        )
        enumerated_shapes = CoreMLBackend.enumerated_shapes_from_compile_specs(
            compile_specs
        )
        pass_pipeline: ct.PassPipeline = CoreMLBackend.pass_pipeline_from_compile_specs(
            compile_specs
        )

        # If using enumerated shapes, we need to pass the inputs to CoreML's convert() function
        # explicitly
        ct_inputs = None
        if enumerated_shapes is not None:
            ct_inputs = _get_ct_inputs(edge_program, enumerated_shapes)

            # Check there are not multiple enumerated inputs if iOS is below 18
            if (minimum_deployment_target is None) or (
                minimum_deployment_target < ct.target.iOS18
            ):
                n_enumerated_inputs = 0
                for ct_in in ct_inputs:
                    if isinstance(ct_in.shape, ct.EnumeratedShapes):
                        n_enumerated_inputs += 1
                if n_enumerated_inputs > 1:
                    raise ValueError(
                        f"You're program has {n_enumerated_inputs}, but the minimum_deployment_target is set to {minimum_deployment_target}.  Multiple enumerated inputs requires iOS18 or later."
                    )

        # Load the model if MODEL_TYPE is 'COMPILED_MODEL'. This step is necessary because
        # get_compiled_model_path() requires a loaded model.
        skip_model_load = model_type != CoreMLBackend.MODEL_TYPE.COMPILED_MODEL
        mlmodel = ct.convert(
            model=edge_program,
            source="pytorch",
            convert_to="mlprogram",
            pass_pipeline=pass_pipeline,
            skip_model_load=skip_model_load,
            compute_precision=model_compute_precision,
            minimum_deployment_target=minimum_deployment_target,
            compute_units=compute_units,
            inputs=ct_inputs,
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
