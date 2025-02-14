# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import argparse
import os
import shutil
import subprocess
from itertools import islice
from pathlib import Path

import executorch
import numpy as np
import timm
import torch
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from executorch.backends.openvino import OpenVINOQuantizer
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir import EdgeProgramManager
from executorch.exir import to_edge
from executorch.exir.backend.backend_details import CompileSpec
from sklearn.metrics import accuracy_score
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.export import export
from torch.export.exported_program import ExportedProgram
from transformers import AutoModel

import nncf
from nncf.experimental.torch.fx.quantization.quantize_pt2e import quantize_pt2e


# Function to load a model based on the selected suite
def load_model(suite: str, model_name: str):
    if suite == "timm":
        return timm.create_model(model_name, pretrained=True)
    elif suite == "torchvision":
        if not hasattr(torchvision_models, model_name):
            msg = f"Model {model_name} not found in torchvision."
            raise ValueError(msg)
        return getattr(torchvision_models, model_name)(pretrained=True)
    elif suite == "huggingface":
        return AutoModel.from_pretrained(model_name)
    else:
        msg = f"Unsupported model suite: {suite}"
        raise ValueError(msg)


def load_calibration_dataset(dataset_path: str, batch_size: int, suite: str, model: torch.nn.Module, model_name: str):
    val_dir = f"{dataset_path}/val"

    if suite == "torchvision":
        transform = torchvision_models.get_model_weights(model_name).DEFAULT.transforms()
    elif suite == "timm":
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    else:
        msg = f"Validation is not supported yet for the suite {suite}"
        raise ValueError(msg)

    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    calibration_dataset = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    return calibration_dataset


def dump_inputs(calibration_dataset, dest_path):
    input_files, targets = [], []
    for idx, data in enumerate(calibration_dataset):
        feature, target = data
        targets.extend(target)
        file_name = f"{dest_path}/input_{idx}_0.raw"
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature)
        feature.detach().numpy().tofile(file_name)
        input_files.append(file_name)

    return input_files, targets


def quantize_model(
    captured_model: torch.fx.GraphModule, calibration_dataset: torch.utils.data.DataLoader, use_nncf: bool
) -> torch.fx.GraphModule:
    quantizer = OpenVINOQuantizer()

    print("PTQ: Quantize the model")
    default_subset_size = 300
    batch_size = calibration_dataset.batch_size
    subset_size = (default_subset_size // batch_size) + int(default_subset_size % batch_size > 0)

    def transform(x):
        return x[0]

    if use_nncf:

        quantized_model = quantize_pt2e(
            captured_model,
            quantizer,
            subset_size=subset_size,
            calibration_dataset=nncf.Dataset(calibration_dataset, transform_func=transform),
            fold_quantize=False,
        )
    else:
        annotated_model = prepare_pt2e(captured_model, quantizer)

        print("PTQ: Calibrate the model...")
        for data in islice(calibration_dataset, subset_size):
            annotated_model(transform(data))

        print("PTQ: Convert the quantized model...")
        quantized_model = convert_pt2e(annotated_model, fold_quantize=False)

    return quantized_model


def validate_model(model_file_name: str, calibration_dataset: torch.utils.data.DataLoader) -> float:
    # 1: Dump inputs
    dest_path = Path("tmp_inputs")
    out_path = Path("tmp_outputs")
    for d in [dest_path, out_path]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    input_files, targets = dump_inputs(calibration_dataset, dest_path)
    inp_list_file = dest_path / "in_list.txt"
    with open(inp_list_file, "w") as f:
        f.write("\n".join(input_files) + "\n")

    # 2: Run the executor
    print("Run openvino_executor_runner...")

    subprocess.run(
        [
            "../../../cmake-openvino-out/examples/openvino/openvino_executor_runner",
            f"--model_path={model_file_name}",
            f"--input_list_path={inp_list_file}",
            f"--output_folder_path={out_path}",
        ]
    )

    # 3: load the outputs and compare with the targets
    predictions = []
    for i in range(len(input_files)):
        tensor = np.fromfile(out_path / f"output_{i}_0.raw", dtype=np.float32)
        predictions.extend(torch.tensor(tensor).reshape(-1, 1000).argmax(-1))

    return accuracy_score(predictions, targets)


def main(
    suite: str,
    model_name: str,
    input_shape,
    quantize: bool,
    validate: bool,
    dataset_path: str,
    device: str,
    batch_size: int,
    quantization_flow: str,
):
    # Load the selected model
    model = load_model(suite, model_name)
    model = model.eval()

    if dataset_path:
        calibration_dataset = load_calibration_dataset(dataset_path, batch_size, suite, model, model_name)
        input_shape = tuple(next(iter(calibration_dataset))[0].shape)
        print(f"Input shape retrieved from the model config: {input_shape}")
    # Ensure input_shape is a tuple
    elif isinstance(input_shape, (list, tuple)):
        input_shape = tuple(input_shape)
    else:
        msg = "Input shape must be a list or tuple."
        raise ValueError(msg)
    # Provide input
    example_args = (torch.randn(*input_shape),)

    # Export the model to the aten dialect
    aten_dialect: ExportedProgram = export(model, example_args)

    if quantize:
        if suite == "huggingface":
            msg = f"Quantization of {suite} models did not support yet."
            raise ValueError(msg)

        # Quantize model
        if not dataset_path:
            msg = "Quantization requires a calibration dataset."
            raise ValueError(msg)
        quantized_model = quantize_model(
            aten_dialect.module(), calibration_dataset, use_nncf=quantization_flow == "nncf"
        )

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
    model_file_name = f"{model_name}_{'int8' if quantize else 'fp32'}.pte"
    with open(model_file_name, "wb") as file:
        exec_prog.write_to_file(file)
    print(f"Model exported and saved as {model_file_name} on {device}.")

    if validate:
        if suite == "huggingface":
            msg = f"Validation of {suite} models did not support yet."
            raise ValueError(msg)

        if not dataset_path:
            msg = "Validation requires a calibration dataset."
            raise ValueError(msg)

        print("Start validation of the model:")
        acc_top1 = validate_model(model_file_name, calibration_dataset)
        print(f"acc@1: {acc_top1}")


if __name__ == "__main__":
    # Argument parser for dynamic inputs
    parser = argparse.ArgumentParser(description="Export models with executorch.")
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=["timm", "torchvision", "huggingface"],
        help="Select the model suite (timm, torchvision, huggingface).",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name to be loaded.")
    parser.add_argument(
        "--input_shape",
        type=eval,
        help="Input shape for the model as a list or tuple (e.g., [1, 3, 224, 224] or (1, 3, 224, 224)).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the validation. Default batch_size == 1."
        " The dataset length must be evenly divisible by the batch size.",
    )
    parser.add_argument("--quantize", action="store_true", help="Enable model quantization.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable model validation. --dataset argument is required for the validation.",
    )
    parser.add_argument("--dataset", type=str, help="Path to the validation dataset.")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Target device for compiling the model (e.g., CPU, GPU). Default is CPU.",
    )
    parser.add_argument(
        "--quantization_flow",
        type=str,
        choices=["pt2e", "nncf"],
        default="nncf",
        help="Select the quantization flow (nncf or pt2e):"
        " pt2e is the default torch.ao quantization flow, while"
        " nncf is a custom method with additional algorithms to improve model performance.",
    )

    args = parser.parse_args()

    # Run the main function with parsed arguments
    # Disable nncf patching as export of the patched model is not supported.
    with nncf.torch.disable_patching():
        main(
            args.suite,
            args.model,
            args.input_shape,
            args.quantize,
            args.validate,
            args.dataset,
            args.device,
            args.batch_size,
            args.quantization_flow,
        )
