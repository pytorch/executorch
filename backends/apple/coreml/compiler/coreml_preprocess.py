#  Copyright © 2023 Apple Inc. All rights reserved.

# CoreML backend for delegating a EdgeProgram to CoreML.

import json
import shutil
import uuid

from pathlib import Path

from typing import final, List

import coremltools as ct
import executorchcoreml

from executorch.exir.backend.backend_details import (
    BackendDetails,
    ExportedProgram,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec


@final
class CoreMLBackend(BackendDetails):
    @staticmethod
    def to_bytes(mlmodel):
        dir_path = Path("tmp")
        model_dir_path = dir_path / "lowered_module"
        Path(model_dir_path).mkdir(parents=True, exist_ok=True)
        model_path = model_dir_path / "model.mlpackage"
        mlmodel.save(model_path)

        # save model metdata
        spec = mlmodel.get_spec()
        input_names = [input.name for input in spec.description.input]
        output_names = [output.name for output in spec.description.output]
        identifier = uuid.uuid4()

        model_metadata = {
            "inputNames": input_names,
            "outputNames": output_names,
            "identifier": str(identifier),
        }

        # store metadata
        model_metadata_path = Path(model_dir_path) / "metadata.json"
        json_object = json.dumps(model_metadata)
        with open(model_metadata_path, "w") as outfile:
            outfile.write(json_object)

        # flatten directory contents and convert it to bytes
        flattened_bytes = executorchcoreml.flatten_directory_contents(
            str(model_dir_path.resolve())
        )
        shutil.rmtree(str(model_dir_path.resolve()))
        return flattened_bytes

    @classmethod
    # pyre-ignore
    def preprocess(
        cls,
        edge_program: ExportedProgram,
        module_compile_spec: List[CompileSpec],
    ) -> PreprocessResult:
        mlmodel = ct.convert(
            model=edge_program,
            source="pytorch",
            convert_to="mlprogram",
            pass_pipeline=ct.PassPipeline.DEFAULT,
            skip_model_load=True,
        )
        flattened_bytes = CoreMLBackend.to_bytes(mlmodel)
        return PreprocessResult(
            processed_bytes=flattened_bytes,
        )
