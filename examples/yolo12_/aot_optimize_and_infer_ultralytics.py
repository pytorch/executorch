# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="import-untyped,import-not-found"

import argparse
import time
from typing import Any, cast, Iterator, List, Optional, Tuple

import cv2

import executorch

import nncf.torch
import numpy as np
import timm
import torch
import torchvision.models as torchvision_models
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.backends.openvino.quantizer import quantize_model

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
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
from ultralytics import YOLO


class CV2VideoIter:
    def __init__(self, cap) -> None:
        self._cap = cap

    def __iter__(self):
        return self

    def __next__(self):
        success, frame = self._cap.read()
        if not success:
            raise StopIteration()
        return frame

    def __len__(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))


class CV2VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, cap) -> None:
        super().__init__()
        self._iter = CV2VideoIter(cap)

    def __iter__(self) -> Iterator:
        return self._iter

    def __len__(self):
        return len(self._iter)




def lower_to_openvino(
    aten_dialect: torch.ExportedProgram,
    example_args: Tuple[Any, ...],
    transform_fn: callable,
    device: str,
    calibration_dataset: CV2VideoDataset,
    subset_size: int,
    quantize: bool,
) -> ExecutorchProgramManager:
    if quantize:
        quantized_model = quantize_model(
            cast(torch.fx.GraphModule, aten_dialect.module()),
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
        compile_config=EdgeCompileConfig(
            _skip_dim_order=True,
        ),
    )

    # Apply backend-specific passes
    return lowered_module.to_executorch(
        config=executorch.exir.ExecutorchBackendConfig()
    )


def lower_to_xnnpack(
    aten_dialect: torch.ExportedProgram,
    example_args: Tuple[Any, ...],
    transform_fn: callable,
    device: str,
    calibration_dataset: CV2VideoDataset,
    subset_size: int,
    quantize: bool,
) -> ExecutorchProgramManager:
    edge = to_edge_transform_and_lower(
        aten_dialect,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False if args.quantize else True,
            _skip_dim_order=True,  # TODO(T182187531): enable dim order in xnnpack
        ),
    )

    return edge.to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )


def main(
    model_name: str,
    quantize: bool,
    video_path: str,
    subset_size: int,
    backend: str,
    device: str,
):
    """
    Main function to load, quantize, and validate a model.

    :param model_name: The name of the model to load.
    :param quantize: Whether to quantize the model.
    :param video_path: Path to the video to use for the calibration
    :param device: The device to run the model on (e.g., "cpu", "gpu").
    """

    # Load the selected model
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Setup pre-processing
    np_dummy_tensor = np.ones((height, width, 3))
    model.predict(np_dummy_tensor, imgsz=((height, width)), device="cpu")

    pt_model = model.model.to(torch.device("cpu"))

    def transform_fn(frame):
        input_tensor = model.predictor.preprocess([frame])
        return input_tensor

    example_args = (transform_fn(np_dummy_tensor),)
    with torch.no_grad():
        aten_dialect = torch.export.export(pt_model, args=example_args)

    if backend == "openvino":
        lower_fn = lower_to_openvino
    elif backend == "xnnpack":
        lower_fn = lower_to_xnnpack

    exec_prog = lower_fn(
        aten_dialect=aten_dialect,
        example_args=example_args,
        transform_fn=transform_fn,
        device=device,
        calibration_dataset=CV2VideoDataset(cap),
        subset_size=subset_size,
        quantize=quantize,
    )

    if not model_file_name:
        model_file_name = f"{model_name}_{'int8' if quantize else 'fp32'}_{backend}.pte"
    with open(model_file_name, "wb") as file:
        exec_prog.write_to_file(file)
    print(f"Model exported and saved as {model_file_name} on {device}.")


if __name__ == "__main__":
    # Argument parser for dynamic inputs
    parser = argparse.ArgumentParser(description="Export models with executorch.")
    parser.add_argument(
        "--input_dims",
        type=eval,
        help="Dimesion in format [hight, weight] or (hight, weight). Input shape for the model as a list or tuple (e.g., [1, 3, 224, 224] or (1, 3, 224, 224)).",
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

