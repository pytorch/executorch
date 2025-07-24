# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="import-untyped,import-not-found"

import argparse
import time
from typing import cast, List, Optional

import executorch

import nncf.torch
import timm
import torch
import torchvision.models as torchvision_models
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.backends.openvino.quantizer import quantize_model
from executorch.exir import (
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.backend_details import CompileSpec
from executorch.runtime import Runtime
from sklearn.metrics import accuracy_score
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.export import export
from torch.export.exported_program import ExportedProgram
from torchvision import datasets
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


def load_calibration_dataset(
    dataset_path: str,
    batch_size: int,
    suite: str,
    model: torch.nn.Module,
    model_name: str,
):
    """
    Loads a calibration dataset for model quantization.

    :param dataset_path: Path to the dataset directory.
    :param batch_size: Number of samples per batch.
    :param suite: The model suite used for preprocessing transformations. Supported values are:
        - "torchvision": Uses predefined transformations for torchvision models.
        - "timm": Uses dataset transformations based on the model's pretrained configuration.
    :param model: The model instance, required for timm transformation resolution.
    :param model_name: The model name, required for torchvision transformations.
    :return: A DataLoader instance for the calibration dataset.
    :raises ValueError: If the suite is unsupported for validation.
    """
    val_dir = f"{dataset_path}/val"

    if suite == "torchvision":
        transform = torchvision_models.get_model_weights(
            model_name
        ).DEFAULT.transforms()
    elif suite == "timm":
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
    else:
        msg = f"Validation is not supported yet for the suite {suite}"
        raise ValueError(msg)

    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    calibration_dataset = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return calibration_dataset


def infer_model(
    exec_prog: ExecutorchProgramManager,
    inputs,
    num_iter: int,
    warmup_iter: int,
    output_path: str,
) -> float:
    """
    Executes inference and reports the average timing.

    :param exec_prog: ExecutorchProgramManager of the lowered model
    :param inputs: The inputs for the model.
    :param num_iter: The number of iterations to execute inference for timing.
    :param warmup_iter: The number of iterations to execute inference for warmup before timing.
    :param output_path: Path to the output tensor file to save the output of inference..
    :return: The average inference timing.
    """
    # Load model from buffer
    runtime = Runtime.get()
    program = runtime.load_program(exec_prog.buffer)
    method = program.load_method("forward")
    if method is None:
        raise ValueError("Load method failed")

    # Execute warmup
    out = None
    for _i in range(warmup_iter):
        out = method.execute(inputs)

    # Execute inference and measure timing
    time_total = 0.0
    for _i in range(num_iter):
        time_start = time.time()
        out = method.execute(inputs)
        time_end = time.time()
        time_total += time_end - time_start

    # Save output tensor as raw tensor file
    if output_path:
        assert out is not None
        torch.save(out, output_path)

    # Return average inference timing
    return time_total / float(num_iter)


def validate_model(
    exec_prog: ExecutorchProgramManager,
    calibration_dataset: torch.utils.data.DataLoader,
) -> float:
    """
    Validates the model using the calibration dataset.

    :param exec_prog: ExecutorchProgramManager of the lowered model
    :param calibration_dataset: A DataLoader containing calibration data.
    :return: The accuracy score of the model.
    """
    # Load model from buffer
    runtime = Runtime.get()
    program = runtime.load_program(exec_prog.buffer)
    method = program.load_method("forward")
    if method is None:
        raise ValueError("Load method failed")

    # Iterate over the dataset and run the executor
    predictions: List[int] = []
    targets = []
    for _idx, data in enumerate(calibration_dataset):
        feature, target = data
        targets.extend(target)
        out = list(method.execute((feature,)))
        predictions.extend(torch.stack(out).reshape(-1, 1000).argmax(-1))

    # Check accuracy
    return accuracy_score(predictions, targets)


def main(  # noqa: C901
    suite: str,
    model_name: str,
    input_shape,
    save_model: bool,
    model_file_name: str,
    quantize: bool,
    validate: bool,
    dataset_path: str,
    device: str,
    batch_size: int,
    infer: bool,
    num_iter: int,
    warmup_iter: int,
    input_path: str,
    output_path: str,
):
    """
    Main function to load, quantize, and validate a model.

    :param suite: The model suite to use (e.g., "timm", "torchvision", "huggingface").
    :param model_name: The name of the model to load.
    :param input_shape: The input shape for the model.
    :param save_model: Whether to save the compiled model as a .pte file.
    :param model_file_name: Custom file name to save the exported model.
    :param quantize: Whether to quantize the model.
    :param validate: Whether to validate the model.
    :param dataset_path: Path to the dataset for calibration/validation.
    :param device: The device to run the model on (e.g., "cpu", "gpu").
    :param batch_size: Batch size for dataset loading.
    :param infer: Whether to execute inference and report timing.
    :param num_iter: The number of iterations to execute inference for timing.
    :param warmup_iter: The number of iterations to execute inference for warmup before timing.
    :param input_path: Path to the input tensor file to read the input for inference.
    :param output_path: Path to the output tensor file to save the output of inference..

    """

    # Load the selected model
    model = load_model(suite, model_name)
    model = model.eval()

    calibration_dataset: Optional[torch.utils.data.DataLoader] = None

    if dataset_path:
        calibration_dataset = load_calibration_dataset(
            dataset_path, batch_size, suite, model, model_name
        )
        if calibration_dataset is not None:
            input_shape = tuple(next(iter(calibration_dataset))[0].shape)
            print(f"Input shape retrieved from the model config: {input_shape}")
        else:
            msg = "Quantization requires a valid calibration dataset"
            raise ValueError(msg)
    # Ensure input_shape is a tuple
    elif isinstance(input_shape, (list, tuple)):
        input_shape = tuple(input_shape)
    else:
        msg = "Input shape must be a list or tuple."
        raise ValueError(msg)
    # Provide input
    if input_path:
        example_args = (torch.load(input_path, weights_only=False),)
    elif suite == "huggingface":
        if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
            vocab_size = model.config.vocab_size
        else:
            vocab_size = 30522
        example_args = (torch.randint(0, vocab_size, input_shape, dtype=torch.int64),)
    else:
        example_args = (torch.randn(*input_shape),)

    # Export the model to the aten dialect
    aten_dialect: ExportedProgram = export(model, example_args)

    if quantize and calibration_dataset:
        if suite == "huggingface":
            msg = f"Quantization of {suite} models did not support yet."
            raise ValueError(msg)

        # Quantize model
        if not dataset_path:
            msg = "Quantization requires a calibration dataset."
            raise ValueError(msg)

        subset_size = 300
        batch_size = calibration_dataset.batch_size or 1
        subset_size = (subset_size // batch_size) + int(subset_size % batch_size > 0)

        def transform_fn(x):
            return x[0]

        quantized_model = quantize_model(
            cast(torch.fx.GraphModule, aten_dialect.module()),  # type: ignore[redundant-cast]
            calibration_dataset,
            subset_size=subset_size,
            transform_fn=transform_fn,
        )

        aten_dialect = export(quantized_model, example_args)

    # Convert to edge dialect and lower the module to the backend with a custom partitioner
    compile_spec = [CompileSpec("device", device.encode())]
    lowered_module: EdgeProgramManager = to_edge_transform_and_lower(
        aten_dialect,
        partitioner=[
            OpenvinoPartitioner(compile_spec),
        ],
    )

    # Apply backend-specific passes
    exec_prog = lowered_module.to_executorch(
        config=executorch.exir.ExecutorchBackendConfig()
    )

    # Serialize and save it to a file
    if save_model:
        if not model_file_name:
            model_file_name = f"{model_name}_{'int8' if quantize else 'fp32'}.pte"
        with open(model_file_name, "wb") as file:
            exec_prog.write_to_file(file)
        print(f"Model exported and saved as {model_file_name} on {device}.")

    if validate and calibration_dataset:
        if suite == "huggingface":
            msg = f"Validation of {suite} models did not support yet."
            raise ValueError(msg)

        if not dataset_path:
            msg = "Validation requires a calibration dataset."
            raise ValueError(msg)

        print("Start validation of the model:")
        acc_top1 = validate_model(exec_prog, calibration_dataset)
        print(f"acc@1: {acc_top1}")

    if infer:
        print("Start inference of the model:")
        avg_time = infer_model(
            exec_prog, example_args, num_iter, warmup_iter, output_path
        )
        print(f"Average inference time: {avg_time}")


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
    parser.add_argument(
        "--model", type=str, required=True, help="Model name to be loaded."
    )
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
    parser.add_argument(
        "--export", action="store_true", help="Export the compiled model as .pte file."
    )
    parser.add_argument(
        "--model_file_name",
        type=str,
        help="Custom file name to save the exported model.",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Enable model quantization."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable model validation. --dataset argument is required for the validation.",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run inference and report timing.",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=1,
        help="The number of iterations to execute inference for timing.",
    )
    parser.add_argument(
        "--warmup_iter",
        type=int,
        default=0,
        help="The number of iterations to execute inference for warmup before timing.",
    )
    parser.add_argument(
        "--input_tensor_path",
        type=str,
        help="Path to the input tensor file to read the input for inference.",
    )
    parser.add_argument(
        "--output_tensor_path",
        type=str,
        help="Path to the output tensor file to save the output of inference.",
    )
    parser.add_argument("--dataset", type=str, help="Path to the validation dataset.")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="Target device for compiling the model (e.g., CPU, GPU). Default is CPU.",
    )

    args = parser.parse_args()

    # Run the main function with parsed arguments
    # Disable nncf patching as export of the patched model is not supported.
    with nncf.torch.disable_patching():
        main(
            args.suite,
            args.model,
            args.input_shape,
            args.export,
            args.model_file_name,
            args.quantize,
            args.validate,
            args.dataset,
            args.device,
            args.batch_size,
            args.infer,
            args.num_iter,
            args.warmup_iter,
            args.input_tensor_path,
            args.output_tensor_path,
        )
