# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import time

import timm

import torch
import torchvision.models as torchvision_models

from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export import export, ExportedProgram
from transformers import AutoModel


# Function to load a model based on the selected suite
def load_model(suite: str, model_name: str):
    """
    Loads a pre-trained model from the specified model suite.

    :param suite: The suite from which to load the model. Supported values are:
        - "timm": Uses `timm.create_model` to load the model.
        - "torchvision": Loads a model from `torchvision.models`. Raises an error if the model does not exist.
        - "huggingface": Loads a transformer model using `AutoModel.from_pretrained`.
    :param model_name: The name of the model to load.
    :return: The loaded model instance.
    :raises ValueError: If the specified model suite is unsupported or the model is not found.
    """
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


def main(
    suite: str,
    model_name: str,
    model_path: str,
    input_shape,
    device: str,
    num_iterations: int,
    warmup_iterations: int,
    input_path: str,
    output_path: str,
):
    """
    Main function to load, quantize, and infer a model.

    :param suite: The model suite to use (e.g., "timm", "torchvision", "huggingface").
    :param model_name: The name of the model to load.
    :param input_shape: The input shape for the model.
    :param device: The device to run the model on (e.g., "cpu", "gpu").
    :param num_iterations: Number of iterations to execute inference.
    """
    # Custom check to ensure suite is provided with model name
    if model_name and not suite:
        print("Error: --suite argument should be provided with --model")
        sys.exit(1)

    if input_path:
        print("Loading input tensor from ", input_path)
        sample_inputs = (torch.load(input_path, weights_only=False),)
    else:
        print("Generating random input tensor with shape of  ", input_shape)
        sample_inputs = (torch.randn(input_shape),)

    if model_name:
        print("Downloading model")
        print("suite: ", suite)
        print("model: ", model_name)
        model = load_model(suite, model_name)
        model = model.eval()

        exported_program: ExportedProgram = export(model, sample_inputs)
        compile_spec = [CompileSpec("device", device.encode())]
        edge: EdgeProgramManager = to_edge_transform_and_lower(
            exported_program,
            partitioner=[
                OpenvinoPartitioner(compile_spec),
            ],
        )

        exec_prog = edge.to_executorch()
        executorch_module = _load_for_executorch_from_buffer(exec_prog.buffer)
    else:
        print("Loading model from ", model_path)
        with open(model_path, "rb") as f:
            model_buffer = f.read()  # Read model file into buffer
        executorch_module = _load_for_executorch_from_buffer(model_buffer)

    if warmup_iterations > 0:
        print("Warmup begins for ", warmup_iterations, " iterations")
        for _i in range(warmup_iterations):
            out = executorch_module.run_method("forward", sample_inputs)

    print("Execution begins for ", num_iterations, " iterations")
    time_total = 0
    for _i in range(num_iterations):
        time_start = time.time()
        out = executorch_module.run_method("forward", sample_inputs)
        time_end = time.time()
        time_total += time_end - time_start

    print("Average inference time: ", (time_total / float(num_iterations)), " secs")

    if output_path:
        torch.save(out, output_path)


if __name__ == "__main__":
    # Argument parser for dynamic inputs
    parser = argparse.ArgumentParser(description="Export models with executorch.")
    parser.add_argument(
        "--suite",
        type=str,
        required=False,
        choices=["timm", "torchvision", "huggingface"],
        help="Select the model suite (timm, torchvision, huggingface).",
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, help="Model name to be loaded.")
    model_group.add_argument(
        "--model_path", type=str, help="Model path to .pte file to be loaded."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_shape",
        type=eval,
        help="Input shape for the model as a list or tuple (e.g., [1, 3, 224, 224] or (1, 3, 224, 224)).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Target device for compiling the model (e.g., CPU, GPU). Default is CPU.",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=1,
        help="Number of iterations to execute inference",
    )
    parser.add_argument(
        "--warmup_iter",
        type=int,
        default=0,
        help="Number of iterations to execute for warmup",
    )
    input_group.add_argument(
        "--input_tensor_path",
        type=str,
        help="Optional raw tensor input file to load the input from",
    )
    parser.add_argument(
        "--output_tensor_path",
        type=str,
        help="Optional output file path to save raw output tensor",
    )

    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(
        args.suite,
        args.model,
        args.model_path,
        args.input_shape,
        args.device,
        args.num_iter,
        args.warmup_iter,
        args.input_tensor_path,
        args.output_tensor_path,
    )
