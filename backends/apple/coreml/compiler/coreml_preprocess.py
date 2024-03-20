#  Copyright © 2023 Apple Inc. All rights reserved.

# CoreML backend for delegating a EdgeProgram to CoreML.

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from enum import Enum

from pathlib import Path

from typing import Dict, final, List

import coremltools as ct
import executorchcoreml

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


class COMPILE_SPEC_KEYS(Enum):
    COMPUTE_UNITS = "compute_units"
    MODEL_TYPE = "model_type"
    MIN_DEPLOYMENT_TARGET = "min_deployment_target"
    MODEL_COMPUTE_PRECISION = "model_compute_precision"


@dataclass
class ModelMetadata:
    # The model input names.
    inputNames: List[str]
    # The model output names.
    outputNames: List[str]
    # The model identifier.
    identifier: str


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
    def generate_compile_specs(
        compute_unit: ct.ComputeUnit = ct.ComputeUnit.ALL,
        minimum_deployment_target: ct.target = ct.target.iOS15,
        compute_precision: ct.precision = ct.precision.FLOAT16,
        model_type: MODEL_TYPE = MODEL_TYPE.MODEL,
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

        return compile_specs

    @staticmethod
    def model_metadata_from_spec(model_spec: ct.proto.Model_pb2) -> Dict[str, str]:
        input_names: List[str] = [input.name for input in model_spec.description.input]
        output_names = [output.name for output in model_spec.description.output]
        identifier = uuid.uuid4()

        return ModelMetadata(
            inputNames=input_names, outputNames=output_names, identifier=str(identifier)
        )

    @staticmethod
    def to_bytes(mlmodel: ct.models.MLModel, model_type: MODEL_TYPE) -> bytes:
        dir_path: Path = Path("tmp")
        model_dir_path: Path = dir_path / "lowered_module"
        model_spec: ct.proto.Model_pb2 = mlmodel.get_spec()
        model_metadata: ModelMetadata = CoreMLBackend.model_metadata_from_spec(
            model_spec
        )
        match model_type:
            case CoreMLBackend.MODEL_TYPE.MODEL:
                # Store model.
                model_path = model_dir_path / "model.mlpackage"
                mlmodel.save(model_path)

            case CoreMLBackend.MODEL_TYPE.COMPILED_MODEL:
                # Store compiled model
                model_path = model_dir_path / "model.mlmodelc"
                compiled_model_path = mlmodel.get_compiled_model_path()

                shutil.copytree(
                    compiled_model_path,
                    str(model_path.resolve()),
                    dirs_exist_ok=True,
                )

        # Store model metadata.
        model_metadata_path = Path(model_dir_path) / "metadata.json"
        model_metadata_json = json.dumps(asdict(model_metadata))
        with open(model_metadata_path, "w") as outfile:
            outfile.write(model_metadata_json)

        # flatten directory contents and convert it to bytes
        flattened_bytes = executorchcoreml.flatten_directory_contents(
            str(model_dir_path.resolve())
        )

        shutil.rmtree(str(model_dir_path.resolve()))
        return flattened_bytes

    @classmethod
    def preprocess(
        cls,
        edge_program: ExportedProgram,
        module_compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        model_type: CoreMLBackend.MODEL_TYPE = (
            CoreMLBackend.model_type_from_compile_specs(
                module_compile_specs,
            )
        )

        model_compute_precision: ct.precision = (
            CoreMLBackend.model_compute_precision_from_compile_specs(
                module_compile_specs
            )
        )

        minimum_deployment_target: ct.target = (
            CoreMLBackend.min_deployment_target_from_compile_specs(module_compile_specs)
        )

        skip_model_load: bool = False
        match model_type:
            case CoreMLBackend.MODEL_TYPE.MODEL:
                skip_model_load = True

            case CoreMLBackend.MODEL_TYPE.COMPILED_MODEL:
                skip_model_load = False

        mlmodel = ct.convert(
            model=edge_program,
            source="pytorch",
            convert_to="mlprogram",
            pass_pipeline=ct.PassPipeline.DEFAULT,
            skip_model_load=skip_model_load,
            compute_precision=model_compute_precision,
            minimum_deployment_target=minimum_deployment_target,
        )

        processed_bytes = CoreMLBackend.to_bytes(mlmodel, model_type=model_type)
        return PreprocessResult(
            processed_bytes=processed_bytes,
        )
