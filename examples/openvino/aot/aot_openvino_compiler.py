# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import executorch
import timm
import torch
import torchvision.models as torchvision_models
from transformers import AutoModel
from executorch.exir.backend.backend_details import CompileSpec
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from torch.export import export, ExportedProgram
from torch.export.exported_program import ExportedProgram
import argparse

# Function to load a model based on the selected suite
def load_model(suite: str, model_name: str):
    if suite == "timm":
        return timm.create_model(model_name, pretrained=True)
    elif suite == "torchvision":
        if not hasattr(torchvision_models, model_name):
            raise ValueError(f"Model {model_name} not found in torchvision.")
        return getattr(torchvision_models, model_name)(pretrained=True)
    elif suite == "huggingface":
        return AutoModel.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model suite: {suite}")

def main(suite: str, model_name: str, input_shape, device: str):
    # Ensure input_shape is a tuple
    if isinstance(input_shape, list):
        input_shape = tuple(input_shape)
    elif not isinstance(input_shape, tuple):
        raise ValueError("Input shape must be a list or tuple.")

    # Load the selected model
    model = load_model(suite, model_name)
    model = model.eval()

    # Provide input
    example_args = (torch.randn(*input_shape), )

    # Export to aten dialect using torch.export
    aten_dialect: ExportedProgram = export(model, example_args)

    # Convert to edge dialect and lower the module to the backend with a custom partitioner
    compile_spec = [CompileSpec("device", device.encode())]
    lowered_module: EdgeProgramManager = to_edge_transform_and_lower(aten_dialect, partitioner=[OpenvinoPartitioner(compile_spec),])

    # Apply backend-specific passes
    exec_prog = lowered_module.to_executorch(config=executorch.exir.ExecutorchBackendConfig())

    # Serialize and save it to a file
    with open(f"{model_name}.pte", "wb") as file:
        exec_prog.write_to_file(file)
    print(f"Model exported and saved as {model_name}.pte on {device}.")

if __name__ == "__main__":
    # Argument parser for dynamic inputs
    parser = argparse.ArgumentParser(description="Export models with executorch.")
    parser.add_argument("--suite", type=str, required=True, choices=["timm", "torchvision", "huggingface"],
                        help="Select the model suite (timm, torchvision, huggingface).")
    parser.add_argument("--model", type=str, required=True, help="Model name to be loaded.")
    parser.add_argument("--input_shape", type=eval, required=True,
                        help="Input shape for the model as a list or tuple (e.g., [1, 3, 224, 224] or (1, 3, 224, 224)).")
    parser.add_argument("--device", type=str, default="CPU",
                        help="Target device for compiling the model (e.g., CPU, GPU). Default is CPU.")

    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.suite, args.model, args.input_shape, args.device)
