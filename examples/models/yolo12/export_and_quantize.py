# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code="import-untyped,import-not-found"


import argparse
from itertools import islice
from typing import Any, Iterator, Tuple

import cv2
import executorch
import nncf.torch
import numpy as np
import torch
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.backends.openvino.quantizer import OpenVINOQuantizer
from executorch.backends.openvino.quantizer.quantizer import QuantizationMode
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchBackendConfig,
    ExecutorchProgramManager,
    to_edge_transform_and_lower,
)
from executorch.exir.backend.backend_details import CompileSpec
from nncf.experimental.torch.fx import quantize_pt2e
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.graph_drawer import FxGraphDrawer
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


def visualize_fx_model(model: torch.fx.GraphModule, output_svg_path: str):
    g = FxGraphDrawer(model, output_svg_path)
    g.get_dot_graph().write_svg(output_svg_path)


def lower_to_openvino(
    aten_dialect: ExportedProgram,
    example_args: Tuple[Any, ...],
    transform_fn: callable,
    device: str,
    calibration_dataset: CV2VideoDataset,
    subset_size: int,
    quantize: bool,
) -> ExecutorchProgramManager:
    if quantize:
        target_input_dims = tuple(example_args[0].shape[2:])

        def ext_transform_fn(sample):
            sample = transform_fn(sample)
            return pad_to_target(sample, target_input_dims)

        quantizer = OpenVINOQuantizer(mode=QuantizationMode.INT8_TRANSFORMER)
        quantizer.set_ignored_scope(
            types=["mul", "sub", "sigmoid", "__getitem__"],
            subgraphs=[nncf.Subgraph(inputs=["cat_18"], outputs=["output"])]
        )
        quantized_model = quantize_pt2e(
            aten_dialect.module(),
            quantizer,
            nncf.Dataset(calibration_dataset, ext_transform_fn),
            subset_size=subset_size,
            smooth_quant=True,
            fold_quantize=False
        )

        visualize_fx_model(quantized_model, "tmp_quantized_model.svg")
        aten_dialect = torch.export.export(quantized_model, example_args)
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
    aten_dialect: ExportedProgram,
    example_args: Tuple[Any, ...],
    transform_fn: callable,
    device: str,
    calibration_dataset: CV2VideoDataset,
    subset_size: int,
    quantize: bool,
) -> ExecutorchProgramManager:
    if quantize:
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(
            is_per_channel=False,
            is_dynamic=False,
        )
        quantizer.set_global(operator_config)
        m = prepare_pt2e(aten_dialect.module(), quantizer)
        # calibration
        target_input_dims = tuple(example_args[0].shape[2:])
        print("Start quantization...")
        for sample in islice(calibration_dataset, subset_size):
            sample = transform_fn(sample)
            sample = pad_to_target(sample, target_input_dims)
            m(sample)
        m = convert_pt2e(m)
        print("Quantized succsessfully!")
        aten_dialect = torch.export.export(m, example_args)

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


def pad_to_target(
    image: torch.Tensor,
    target_size: Tuple[int, int],
):
    if image.shape[2:] == target_size:
        return image
    img_h, img_w = image.shape[2:]
    target_h, target_w = target_size

    diff_h = target_h - img_h
    pad_h_from = diff_h // 2
    pad_h_to = -(pad_h_from + diff_h % 2) or None
    diff_w = target_w - img_w
    pad_w_from = diff_w // 2
    pad_w_to = -(pad_w_from + diff_w % 2) or None

    result = torch.zeros(
        (
            1,
            3,
        )
        + target_size,
        device=image.device,
        dtype=image.dtype,
    )
    result[:, :, pad_h_from:pad_h_to, pad_w_from:pad_w_to] = image
    return result


def main(
    model_name: str,
    input_dims: Tuple[int, int],
    quantize: bool,
    video_path: str,
    subset_size: int,
    backend: str,
    device: str,
):
    """
    Main function to load, quantize, and export an Yolo model model.

    :param model_name: The name of the YOLO model to load.
    :param quantize: Whether to quantize the model.
    :param video_path: Path to the video to use for the calibration
    :param backend: The Executorch inference backend (e.g., "openvino", "xnnpack").
    :param device: The device to run the model on (e.g., "cpu", "gpu").
    """

    # Load the selected model
    model = YOLO(model_name)

    if quantize:
        if video_path is None:
            raise RuntimeError(
                "Could not quantize model without the video for the calibration."
                " --video_path parameter is needed."
            )
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"Calibration video dims: h: {height} w: {width}")
        calibration_dataset = CV2VideoDataset(cap)
    else:
        calibration_dataset = None

    # Setup pre-processing
    np_dummy_tensor = np.ones((input_dims[0], input_dims[1], 3))
    model.predict(np_dummy_tensor, imgsz=((input_dims[0], input_dims[1])), device="cpu")

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
        calibration_dataset=calibration_dataset,
        subset_size=subset_size,
        quantize=quantize,
    )

    model_file_name = f"{model_name}_{'int8' if quantize else 'fp32'}_{backend}.pte"
    with open(model_file_name, "wb") as file:
        exec_prog.write_to_file(file)
    print(f"Model exported and saved as {model_file_name} on {device}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export FP32 and INT8 Ultralytics Yolo models with executorch."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="yolo12s",
        help="Ultralytics yolo model name.",
    )
    parser.add_argument(
        "--input_dims",
        type=eval,
        default=[640, 640],
        help="Input model dimensions in format [hight, weight] or (hight, weight). Default models dimensions are [640, 640]",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to the input video file to use for the quantization callibration.",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Enable model quantization."
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=300,
        help="Subset size for the quantized model calibration. The default value is 300.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openvino",
        choices=["openvino", "xnnpack"],
        help="Select the Executorch inference backend (openvino, xnnpack). openvino by default.",
    )
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
            model_name=args.model_name,
            input_dims=args.input_dims,
            quantize=args.quantize,
            video_path=args.video_path,
            subset_size=args.subset_size,
            backend=args.backend,
            device=args.device,
        )
