# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import nncf.experimental
import nncf.experimental.torch
import executorch
import nncf
import timm
import torch
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
from transformers import AutoModel
from executorch.exir.backend.backend_details import CompileSpec
from executorch.backends.openvino.preprocess import OpenvinoBackend
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir import EdgeProgramManager, to_edge
from torch.export import export, ExportedProgram
from torch.export.exported_program import ExportedProgram
import argparse
from executorch.backends.openvino import OpenVINOQuantizer
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)


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


def load_calibration_dataset(dataset_path: str):
    val_dir = f"{dataset_path}/val"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose(
            [
                transforms.Resize(64), # for tiny imagenet
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    calibration_dataset = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    return calibration_dataset


def quantize_model(model: torch.fx.GraphModule, calibration_dataset: torch.utils.data.DataLoader, subset_size=300):
    quantizer = OpenVINOQuantizer()

    print("PTQ: Annotate the model...")
    annotated_model = prepare_pt2e(model, quantizer)
    
    print("PTQ: Calibrate the model...")
    for idx, data in enumerate(calibration_dataset):
        if idx >= subset_size:
            break
        annotated_model(data[0])

    print("PTQ: Convert the quantized model...")
    quantized_model = convert_pt2e(annotated_model)
    return quantized_model


def main(suite: str, model_name: str, input_shape, quantize: bool, dataset_path: str, device: str):
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

    # Export the model to the aten dialect
    aten_dialect: ExportedProgram = export(model, example_args)

    if quantize:
        # Quantize model
        if not dataset_path:
            raise ValueError("Quantization requires a calibration dataset.")
        calibration_dataset = load_calibration_dataset(dataset_path)

        captured_model = aten_dialect.module()
        quantized_model = quantize_model(captured_model, calibration_dataset)
        aten_dialect: ExportedProgram = export(quantized_model, example_args)

    # Convert to edge dialect
    edge_program: EdgeProgramManager = to_edge(aten_dialect)
    to_be_lowered_module = edge_program.exported_program()

    # Lower the module to the backend with a custom partitioner
    compile_spec = [CompileSpec("device", device.encode())]
    lowered_module = edge_program.to_backend(OpenvinoPartitioner(compile_spec))

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
    parser.add_argument("--quantize", action="store_true", help="Enable model quantization.")
    parser.add_argument("--dataset", type=str, help="Path to the calibration dataset.")
    parser.add_argument("--device", type=str, default="CPU",
                        help="Target device for compiling the model (e.g., CPU, GPU). Default is CPU.")

    args = parser.parse_args()

    # Run the main function with parsed arguments
    with nncf.torch.disable_patching():
        main(args.suite, args.model, args.input_shape, args.quantize, args.dataset, args.device)
