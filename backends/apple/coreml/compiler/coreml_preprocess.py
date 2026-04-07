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

from typing import Any, Dict, final, List, Optional, Tuple, Union

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
    MULTIMETHOD_WEIGHT_SHARING_STRATEGY = "multimethod_weight_sharing_strategy"


class MULTIMETHOD_WEIGHT_SHARING_STRATEGY(Enum):
    """Strategy for sharing weights across methods in multi-method models.

    When exporting a model with multiple methods (e.g., prefill and decode),
    these strategies control how CoreML models are organized and how weights
    are shared. Different strategies have different tradeoffs — experiment
    with them to find the best fit for your use case.

    DISABLED:
        Each method is compiled into its own independent CoreML model.
        No weight sharing occurs; weights are duplicated across methods.
        Simplest strategy with no constraints on model structure.

    POSITIONAL:
        Partitions are aligned by index across methods. Partition 0 from
        all methods are combined into one multifunction CoreML model,
        partition 1 into another, and so on. This enables weight sharing
        for parameters that appear at the same partition index. Requires
        all methods to have the same number of partitions.

    ONE_BLOB:
        All partitions from all methods are packed into a single
        multifunction CoreML model. This maximizes weight sharing
        opportunities (any parameter can be shared across any method)
        and does not require partition counts to match. However, it may
        result in longer compile times and higher peak memory since the
        entire model — including any method-specific (non-shared) weights
        — lives in a single blob.
    """

    DISABLED = "disabled"
    POSITIONAL = "positional"
    ONE_BLOB = "one_blob"


class MODEL_PATHS(Enum):
    MODEL = "model.mlpackage"
    COMPILED_MODEL = "model.mlmodelc"
    METADATA = "metadata.json"
    DEBUG_INFO = "debug_info.json"


@dataclass
class MethodMetadata:
    # The method input names.
    inputNames: List[str]
    # The method output names.
    outputNames: List[str]


@dataclass
class ModelMetadata:
    # The model input names (for single-method models).
    inputNames: List[str]
    # The model output names (for single-method models).
    outputNames: List[str]
    # The model identifier.
    identifier: str


@dataclass
class MultifunctionModelMetadata:
    # The model identifier.
    identifier: str
    # Per-method metadata (method name -> MethodMetadata).
    methods: Dict[str, MethodMetadata]


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
    def generate_multimethod_weight_sharing_strategy_compile_spec(
        strategy: "MULTIMETHOD_WEIGHT_SHARING_STRATEGY",
    ) -> CompileSpec:
        """
        Returns the compile spec representing the multimethod weight sharing strategy.

        Args:
            strategy: The weight sharing strategy to use when combining methods.
                POSITIONAL: Partitions must align positionally across methods; enables
                    weight sharing via NamedDataStore. Raises error if partitions don't align.
                ONE_BLOB: All partitions from all methods are combined into a single
                    multifunction model. No partition count alignment required.
                DISABLED: Methods are processed independently with no weight sharing.
        """
        return CompileSpec(
            COMPILE_SPEC_KEYS.MULTIMETHOD_WEIGHT_SHARING_STRATEGY.value,
            strategy.value.encode("utf-8"),
        )

    @staticmethod
    def multimethod_weight_sharing_strategy_from_compile_specs(
        compile_specs: List[CompileSpec],
    ) -> "MULTIMETHOD_WEIGHT_SHARING_STRATEGY":
        """
        Returns the multimethod weight sharing strategy by parsing the list of compile specs.
        Defaults to DISABLED if not specified.
        """
        for compile_spec in compile_specs:
            if (
                compile_spec.key
                == COMPILE_SPEC_KEYS.MULTIMETHOD_WEIGHT_SHARING_STRATEGY.value
            ):
                return MULTIMETHOD_WEIGHT_SHARING_STRATEGY(
                    compile_spec.value.decode("utf-8")
                )

        return MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED

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
    def save_model_metadata(
        model_metadata: Union[ModelMetadata, MultifunctionModelMetadata],
        model_dir_path: Path,
    ):
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
    def _convert_to_mlmodel(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
        skip_model_load: bool = True,
    ) -> ct.models.MLModel:
        """
        Convert an ExportedProgram to a CoreML MLModel.

        Args:
            edge_program: The edge program to convert
            compile_specs: Compile specs for this conversion
            skip_model_load: Whether to skip loading the model (for efficiency)

        Returns:
            The converted MLModel
        """
        model_compute_precision = (
            CoreMLBackend.model_compute_precision_from_compile_specs(compile_specs)
        )
        minimum_deployment_target = (
            CoreMLBackend.min_deployment_target_from_compile_specs(compile_specs)
        )
        compute_units = CoreMLBackend.compute_unit_from_compile_specs(compile_specs)
        pass_pipeline = CoreMLBackend.pass_pipeline_from_compile_specs(compile_specs)
        enumerated_shapes = CoreMLBackend.enumerated_shapes_from_compile_specs(
            compile_specs
        )

        # If using enumerated shapes, pass inputs explicitly to CoreML's convert()
        ct_inputs = None
        if enumerated_shapes is not None:
            ct_inputs = _get_ct_inputs(edge_program, enumerated_shapes)

            # Check there are not multiple enumerated inputs if iOS is below 18
            if (minimum_deployment_target is None) or (
                minimum_deployment_target < ct.target.iOS18
            ):
                n_enumerated_inputs = sum(
                    1
                    for ct_in in ct_inputs
                    if isinstance(ct_in.shape, ct.EnumeratedShapes)
                )
                if n_enumerated_inputs > 1:
                    raise ValueError(
                        f"Your program has {n_enumerated_inputs} enumerated inputs, "
                        f"but minimum_deployment_target is {minimum_deployment_target}. "
                        "Multiple enumerated inputs requires iOS18 or later."
                    )

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

        # Apply quantization if specified
        op_linear_quantizer_config = (
            CoreMLBackend.op_linear_quantizer_config_from_compile_specs(compile_specs)
        )
        if op_linear_quantizer_config is not None:
            logger.warning(
                "Core ML Backend op_linear_quantizer_config API is experimental"
            )
            config = cto.coreml.OptimizationConfig(
                global_config=op_linear_quantizer_config,
                op_type_configs={"gather": None},
            )
            mlmodel = cto.coreml.linear_quantize_weights(mlmodel, config=config)

        return mlmodel

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
            CoreMLBackend.model_type_from_compile_specs(compile_specs)
        )

        # Load the model if MODEL_TYPE is 'COMPILED_MODEL'. This step is necessary because
        # get_compiled_model_path() requires a loaded model.
        skip_model_load = model_type != CoreMLBackend.MODEL_TYPE.COMPILED_MODEL

        mlmodel = CoreMLBackend._convert_to_mlmodel(
            edge_program, compile_specs, skip_model_load=skip_model_load
        )

        return CoreMLBackend.preprocess_model(mlmodel, model_type=model_type)

    @classmethod
    def preprocess_multimethod(  # noqa: C901
        cls,
        edge_programs: Dict[str, List[ExportedProgram]],
        compile_specs: Dict[str, List[List[CompileSpec]]],
    ) -> Dict[str, List[PreprocessResult]]:
        """
        Preprocess multiple methods, optionally combining them into CoreML multifunction models.

        The behavior is controlled by the MULTIMETHOD_WEIGHT_SHARING_STRATEGY compile spec:

        POSITIONAL (default):
            Converts each method's ExportedPrograms to mlpackages, then combines
            corresponding partitions across methods using CoreML's multifunction API
            (ct.utils.save_multifunction). This enables weight sharing on disk between
            methods (e.g., decode and prefill for LLMs).

            For each partition index, we create one multifunction model that combines
            that partition from all methods. This requires all methods to have the same
            number of partitions. Raises ValueError if partition counts don't match.

            To avoid duplication, we store the combined model ONCE in NamedDataStore
            with a unique key. Each method's processed_bytes contains a JSON reference
            to the model in NamedDataStore.

        DISABLED:
            Each method is processed independently with no weight sharing. Falls back
            to the default BackendDetails.preprocess_multimethod() implementation.

        Args:
            edge_programs: Dictionary mapping method name to list of partitioned ExportedPrograms
            compile_specs: Dictionary mapping method name to list of CompileSpecs for each partition.
                The MULTIMETHOD_WEIGHT_SHARING_STRATEGY is read from the first method's first
                partition compile specs.

        Returns:
            Dictionary mapping method name to list of PreprocessResults. When using POSITIONAL
            strategy, each method's processed_bytes contains a JSON reference to the shared
            model in NamedDataStore.
        """
        from executorch.exir._serialize._named_data_store import NamedDataStore

        method_names = list(edge_programs.keys())

        if len(method_names) <= 1:
            # Fall back to default implementation for single method
            return super().preprocess_multimethod(edge_programs, compile_specs)

        # Get compile specs from the first method's first partition
        first_method = method_names[0]
        first_compile_specs = compile_specs[first_method][0]

        # Check the weight sharing strategy
        weight_sharing_strategy = (
            cls.multimethod_weight_sharing_strategy_from_compile_specs(
                first_compile_specs
            )
        )

        if weight_sharing_strategy == MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED:
            # Process each method independently with no weight sharing
            logger.info(
                "Multimethod weight sharing is DISABLED. Processing methods independently."
            )
            return super().preprocess_multimethod(edge_programs, compile_specs)

        assert weight_sharing_strategy in (
            MULTIMETHOD_WEIGHT_SHARING_STRATEGY.POSITIONAL,
            MULTIMETHOD_WEIGHT_SHARING_STRATEGY.ONE_BLOB,
        )

        model_type: CoreMLBackend.MODEL_TYPE = cls.model_type_from_compile_specs(
            first_compile_specs
        )

        # Create a temporary directory for all the mlpackages
        temp_dir = Path(tempfile.mkdtemp())

        # Structure: method_mlpackage_paths[method_name][partition_idx] = path
        method_mlpackage_paths: Dict[str, List[Path]] = {
            method_name: [] for method_name in method_names
        }

        # Create a NamedDataStore to hold the shared multifunction models
        named_data_store = NamedDataStore()

        try:
            # Convert each method's partitions to mlpackages
            for method_name in method_names:
                for partition_idx, edge_program in enumerate(
                    edge_programs[method_name]
                ):
                    method_compile_specs = compile_specs[method_name][partition_idx]

                    logger.info(
                        f"Converting method '{method_name}' partition {partition_idx} to mlpackage..."
                    )

                    # Convert to CoreML using shared helper
                    mlmodel = cls._convert_to_mlmodel(
                        edge_program, method_compile_specs, skip_model_load=True
                    )

                    # Save the mlpackage
                    mlpackage_path = (
                        temp_dir / f"{method_name}_partition_{partition_idx}.mlpackage"
                    )
                    mlmodel.save(str(mlpackage_path))
                    method_mlpackage_paths[method_name].append(mlpackage_path)

                    logger.info(
                        f"Saved method '{method_name}' partition {partition_idx} to {mlpackage_path}"
                    )

            if weight_sharing_strategy == MULTIMETHOD_WEIGHT_SHARING_STRATEGY.ONE_BLOB:
                return cls._preprocess_one_blob(
                    method_names=method_names,
                    edge_programs=edge_programs,
                    method_mlpackage_paths=method_mlpackage_paths,
                    model_type=model_type,
                    named_data_store=named_data_store,
                    temp_dir=temp_dir,
                )

            assert (
                weight_sharing_strategy
                == MULTIMETHOD_WEIGHT_SHARING_STRATEGY.POSITIONAL
            )
            return cls._preprocess_positional(
                method_names=method_names,
                edge_programs=edge_programs,
                method_mlpackage_paths=method_mlpackage_paths,
                model_type=model_type,
                named_data_store=named_data_store,
                temp_dir=temp_dir,
            )

        finally:
            # Clean up temporary directory
            shutil.rmtree(str(temp_dir))

    @classmethod
    def _preprocess_positional(
        cls,
        method_names: List[str],
        edge_programs: Dict[str, List[ExportedProgram]],
        method_mlpackage_paths: Dict[str, List[Path]],
        model_type: "CoreMLBackend.MODEL_TYPE",
        named_data_store: Any,
        temp_dir: Path,
    ) -> Dict[str, List[PreprocessResult]]:
        """
        POSITIONAL strategy: for each partition index, combine that partition
        from all methods into a single multifunction model.
        """
        first_method = method_names[0]
        num_partitions = len(edge_programs[first_method])

        for method_name, programs in edge_programs.items():
            if len(programs) != num_partitions:
                raise ValueError(
                    f"Method '{method_name}' has {len(programs)} partitions, but "
                    f"'{first_method}' has {num_partitions}. POSITIONAL weight sharing "
                    "strategy requires all methods to have the same number of partitions. "
                    "Use MULTIMETHOD_WEIGHT_SHARING_STRATEGY.ONE_BLOB (which supports "
                    "different partition counts per method) or "
                    "MULTIMETHOD_WEIGHT_SHARING_STRATEGY.DISABLED if methods should "
                    "be processed independently."
                )

        combined_processed_bytes: List[bytes] = []
        debug_handle_maps: List[Optional[Dict[str, Tuple[int]]]] = []
        model_keys: List[str] = []

        for partition_idx in range(num_partitions):
            logger.info(
                f"Combining partition {partition_idx} from all methods into multifunction model..."
            )

            desc = ct.utils.MultiFunctionDescriptor()
            for method_name in method_names:
                mlpackage_path = method_mlpackage_paths[method_name][partition_idx]
                desc.add_function(
                    str(mlpackage_path),
                    src_function_name="main",
                    target_function_name=method_name,
                )

            desc.default_function_name = first_method

            combined_path = temp_dir / f"combined_partition_{partition_idx}.mlpackage"
            ct.utils.save_multifunction(desc, str(combined_path))

            logger.info(
                f"Saved combined multifunction model for partition {partition_idx} to {combined_path}"
            )

            model_dir_path = temp_dir / f"lowered_module_partition_{partition_idx}"
            model_dir_path.mkdir(exist_ok=True)

            if model_type == CoreMLBackend.MODEL_TYPE.COMPILED_MODEL:
                output_model_path = model_dir_path / MODEL_PATHS.COMPILED_MODEL.value
                combined_model_loaded = ct.models.MLModel(str(combined_path))
                compiled_path = combined_model_loaded.get_compiled_model_path()
                shutil.move(compiled_path, str(output_model_path))
            else:
                output_model_path = model_dir_path / MODEL_PATHS.MODEL.value
                shutil.copytree(str(combined_path), str(output_model_path))

            identifier = "executorch_" + str(uuid.uuid4())

            methods_metadata: Dict[str, MethodMetadata] = {}
            for method_name in method_names:
                method_mlpackage_path = method_mlpackage_paths[method_name][
                    partition_idx
                ]
                method_model = ct.models.MLModel(
                    str(method_mlpackage_path), skip_model_load=True
                )
                method_spec = method_model.get_spec()
                input_names = [inp.name for inp in method_spec.description.input]
                output_names = [out.name for out in method_spec.description.output]
                methods_metadata[method_name] = MethodMetadata(
                    inputNames=input_names,
                    outputNames=output_names,
                )
                logger.info(
                    f"Extracted metadata for method '{method_name}' partition {partition_idx}: "
                    f"{len(input_names)} inputs, {len(output_names)} outputs"
                )

            multifunction_metadata = MultifunctionModelMetadata(
                identifier=identifier,
                methods={k: asdict(v) for k, v in methods_metadata.items()},
            )

            cls.save_model_metadata(multifunction_metadata, model_dir_path)

            processed_bytes = (
                executorchcoreml.flatten_directory_contents(
                    str(model_dir_path.resolve())
                )
                or b""
            )
            combined_processed_bytes.append(processed_bytes)

            model_key = f"coreml_{identifier}"
            model_keys.append(model_key)
            named_data_store.add_named_data(model_key, processed_bytes)

            logger.info(
                f"Created combined processed bytes for partition {partition_idx} ({len(processed_bytes)} bytes)"
            )
            logger.info(f"Stored in NamedDataStore with key '{model_key}'")

            debug_handle_maps.append(None)

        named_data_store_output = named_data_store.get_named_data_store_output()

        preprocess_results: Dict[str, List[PreprocessResult]] = {
            method_name: [] for method_name in method_names
        }

        for partition_idx in range(num_partitions):
            debug_handle_map = debug_handle_maps[partition_idx]

            for method_name in method_names:
                reference = {
                    "version": 1,
                    "key": model_keys[partition_idx],
                    "functionName": method_name,
                }
                MAGIC_NUMBER = b"CMJR"
                reference_bytes = MAGIC_NUMBER + json.dumps(reference).encode("utf-8")

                preprocess_results[method_name].append(
                    PreprocessResult(
                        processed_bytes=reference_bytes,
                        debug_handle_map=debug_handle_map,
                        data_store_output=named_data_store_output,
                    )
                )

        return preprocess_results

    @classmethod
    def _preprocess_one_blob(
        cls,
        method_names: List[str],
        edge_programs: Dict[str, List[ExportedProgram]],
        method_mlpackage_paths: Dict[str, List[Path]],
        model_type: "CoreMLBackend.MODEL_TYPE",
        named_data_store: Any,
        temp_dir: Path,
    ) -> Dict[str, List[PreprocessResult]]:
        """
        ONE_BLOB strategy: combine ALL partitions from ALL methods into a single
        multifunction model. Function names use "{method_name}__{partition_idx}"
        encoding. No partition count alignment is required.
        """
        first_method = method_names[0]

        logger.info(
            "ONE_BLOB: Combining all partitions from all methods into a single multifunction model..."
        )

        # Build a single MultiFunctionDescriptor with all method x partition combinations
        desc = ct.utils.MultiFunctionDescriptor()
        for method_name in method_names:
            for partition_idx, mlpackage_path in enumerate(
                method_mlpackage_paths[method_name]
            ):
                function_name = f"{method_name}__{partition_idx}"
                desc.add_function(
                    str(mlpackage_path),
                    src_function_name="main",
                    target_function_name=function_name,
                )
                logger.info(
                    f"ONE_BLOB: Added function '{function_name}' from {mlpackage_path}"
                )

        desc.default_function_name = f"{first_method}__0"

        combined_path = temp_dir / "combined_all.mlpackage"
        ct.utils.save_multifunction(desc, str(combined_path))

        logger.info(f"ONE_BLOB: Saved combined multifunction model to {combined_path}")

        # Create output directory for the single combined model
        model_dir_path = temp_dir / "lowered_module_one_blob"
        model_dir_path.mkdir(exist_ok=True)

        if model_type == CoreMLBackend.MODEL_TYPE.COMPILED_MODEL:
            output_model_path = model_dir_path / MODEL_PATHS.COMPILED_MODEL.value
            combined_model_loaded = ct.models.MLModel(str(combined_path))
            compiled_path = combined_model_loaded.get_compiled_model_path()
            shutil.move(compiled_path, str(output_model_path))
        else:
            output_model_path = model_dir_path / MODEL_PATHS.MODEL.value
            shutil.copytree(str(combined_path), str(output_model_path))

        identifier = "executorch_" + str(uuid.uuid4())

        # Extract metadata for every method x partition function
        methods_metadata: Dict[str, MethodMetadata] = {}
        for method_name in method_names:
            for partition_idx, mlpackage_path in enumerate(
                method_mlpackage_paths[method_name]
            ):
                function_name = f"{method_name}__{partition_idx}"
                method_model = ct.models.MLModel(
                    str(mlpackage_path), skip_model_load=True
                )
                method_spec = method_model.get_spec()
                input_names = [inp.name for inp in method_spec.description.input]
                output_names = [out.name for out in method_spec.description.output]
                methods_metadata[function_name] = MethodMetadata(
                    inputNames=input_names,
                    outputNames=output_names,
                )
                logger.info(
                    f"ONE_BLOB: Extracted metadata for '{function_name}': "
                    f"{len(input_names)} inputs, {len(output_names)} outputs"
                )

        multifunction_metadata = MultifunctionModelMetadata(
            identifier=identifier,
            methods={k: asdict(v) for k, v in methods_metadata.items()},
        )

        cls.save_model_metadata(multifunction_metadata, model_dir_path)

        # Flatten the single model directory to bytes
        processed_bytes = (
            executorchcoreml.flatten_directory_contents(str(model_dir_path.resolve()))
            or b""
        )

        # Store in NamedDataStore with a single key
        model_key = f"coreml_{identifier}"
        named_data_store.add_named_data(model_key, processed_bytes)

        logger.info(
            f"ONE_BLOB: Stored {len(processed_bytes)} bytes in NamedDataStore with key '{model_key}'"
        )

        named_data_store_output = named_data_store.get_named_data_store_output()

        # Build PreprocessResults — all point to the same NamedDataStore key,
        # differing only in functionName.
        preprocess_results: Dict[str, List[PreprocessResult]] = {
            method_name: [] for method_name in method_names
        }

        MAGIC_NUMBER = b"CMJR"
        for method_name in method_names:
            for partition_idx in range(len(edge_programs[method_name])):
                function_name = f"{method_name}__{partition_idx}"
                reference = {
                    "version": 1,
                    "key": model_key,
                    "functionName": function_name,
                }
                reference_bytes = MAGIC_NUMBER + json.dumps(reference).encode("utf-8")

                preprocess_results[method_name].append(
                    PreprocessResult(
                        processed_bytes=reference_bytes,
                        debug_handle_map=None,
                        data_store_output=named_data_store_output,
                    )
                )

        return preprocess_results
