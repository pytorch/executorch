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
#from nncf.experimental.torch.fx.quantization.quantizer.openvino_quantizer import OpenVINOQuantizer
from nncf.experimental.torch.fx.quantization.quantize_pt2e import quantize_pt2e
from torch.ao.quantization.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)
from sklearn.metrics import accuracy_score
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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


def load_calibration_dataset(dataset_path: str, suite: str, model: torch.nn.Module):
    val_dir = f"{dataset_path}/val"

    if suite == "torchvision":
        transform = torchvision_models.get_model_weights(model.name).transforms()
    else:
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=transform
    )

    calibration_dataset = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    return calibration_dataset


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
        if suite == "huggingface":
            raise ValueError("Quantization of {suite} models did not support yet.")

        # Quantize model
        if not dataset_path:
            raise ValueError("Quantization requires a calibration dataset.")
        calibration_dataset = load_calibration_dataset(dataset_path, suite, model)

        captured_model = aten_dialect.module()
        #visualize_fx_model(captured_model, f"{model_name}_fp32.svg")
        quantizer = OpenVINOQuantizer()

        print("PTQ: Quantize the model")
        def transform(x):
            return x[0]

        quantized_model = quantize_pt2e(captured_model, quantizer, calibration_dataset=nncf.Dataset(calibration_dataset, transform_func=transform), fold_quantize=False)

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
    model_name = f"{model_name}_{'int8' if quantize else 'fp32'}.pte"
    with open(model_name, "wb") as file:
        exec_prog.write_to_file(file)
    print(f"Model exported and saved as {model_name} on {device}.")

    if quantize:
        print("Start validation of the quantized model:")

        # 1: Dump inputs
        import os
        import shutil

        dest_path = "tmp_inputs"
        out_path = "tmp_outputs"
        targets, input_files = [], []
        for d in [dest_path, out_path]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        input_list = ""
        for idx, data in enumerate(calibration_dataset):
            feature, target = data
            targets.append(target)
            file_name = f"{dest_path}/input_{idx}_0.raw"
            input_list += file_name + " "
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature)
            feature.detach().numpy().tofile(file_name)
            input_files.append(file_name)

        inp_list_file = os.path.join(dest_path, "in_list.txt")
        with open(inp_list_file, "w") as f:
            input_list = input_list.strip() + "\n"
            f.write(input_list)

        # 2: Run the executor
        print("Run openvino_executor_runner...")
        import subprocess
        breakpoint()
        subprocess.run(["../../../cmake-openvino-out/examples/openvino/openvino_executor_runner",
                    f"--model_path={model_name}",
                    f"--input_list_path={inp_list_file}",
                    f"--output_folder_path={out_path}",
                    #f"--num_iter={len(input_files)}"
        ])

        # 3: load the outputs and compare with the targets
        import numpy as np
        predictions = []
        for i in range(len(input_files)):
            predictions.append(
                np.fromfile(
                    os.path.join(out_path, f"output_{i}.raw"), dtype=np.float32
                )
            )

        k_val = [1, 5]
        acc_top1 = accuracy_score(predictions, targets)
        print(f"acc@1: {acc_top1}")


from torch.fx.passes.graph_drawer import FxGraphDrawer
def visualize_fx_model(model: torch.fx.GraphModule, output_svg_path: str):
    g = FxGraphDrawer(model, output_svg_path)
    g.get_dot_graph().write_svg(output_svg_path)

def generate_inputs(dest_path: str, file_name: str, inputs=None, input_list=None):
    input_list_file = None
    input_files = []

    # Prepare input list
    if input_list is not None:
        input_list_file = f"{dest_path}/{file_name}"
        with open(input_list_file, "w") as f:
            f.write(input_list)
            f.flush()

    # Prepare input data
    if inputs is not None:
        for idx, data in enumerate(inputs):
            for i, d in enumerate(data):
                file_name = f"{dest_path}/input_{idx}_{i}.raw"
                if not isinstance(d, torch.Tensor):
                    d = torch.tensor(d)
                d.detach().numpy().tofile(file_name)
                input_files.append(file_name)

    return input_list_file, input_files

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
