# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a PTQ HTP FCB ResNet50 and optionally verify it on devices."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from executorch.backends.qualcomm.export_utils import (
    make_quantizer,
    QnnConfig,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.models.resnet import ResNet50Model
from executorch.examples.qualcomm.util_scripts.fcb_resnet50 import get_device_soc_model
from executorch.examples.qualcomm.utils import get_imagenet_dataset, topk_accuracy
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def quantize_model(
    model: torch.nn.Module,
    soc_models: list[QcomChipset],
    calibration_inputs,
) -> torch.nn.Module:
    if not calibration_inputs:
        raise ValueError("calibration_inputs must not be empty")
    exported_model = torch.export.export(
        model, calibration_inputs[0], strict=True
    ).module()
    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_8a8w,
        per_channel_conv=True,
        soc_model=soc_models,
    )
    prepared_model = prepare_pt2e(exported_model, quantizer)
    for calibration_input in calibration_inputs:
        prepared_model(*calibration_input)
    return convert_pt2e(prepared_model)


def export_fcb(model: torch.nn.Module, soc_models: list[QcomChipset], sample_input):
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_models,
        backend_options=[
            generate_htp_compiler_spec(use_fp16=False) for _ in soc_models
        ],
    )
    return (
        to_edge_transform_and_lower_to_qnn(
            module={"forward": model},
            inputs={"forward": sample_input},
            compiler_specs={"forward": compiler_specs},
        )
        .to_executorch()
        .buffer
    )


def run_on_device(
    pte_path: Path,
    output_dir: Path,
    host: str | None,
    device: str,
    build_folder: str,
    requested_socs: list[str],
    inputs,
    targets,
):
    soc_model = get_device_soc_model(host, device)
    if soc_model not in requested_socs:
        raise RuntimeError(
            f"device {device} has {soc_model}, not one of the prepared SoCs {requested_socs}"
        )
    adb = SimpleADB(
        qnn_config=QnnConfig(
            soc_model=soc_model,
            build_folder=build_folder,
            device=device,
            host=host,
        ),
        pte_path=str(pte_path),
        workspace=f"/data/local/tmp/qnn_fcb_resnet50_quantized/{device}",
    )
    adb.push(inputs=inputs, init_env=True)
    adb.execute(custom_runner_cmd=f"rm -rf {adb.output_folder}")
    adb.execute(method_index=0)
    device_outputs = output_dir / device
    shutil.rmtree(device_outputs, ignore_errors=True)
    adb.pull(str(device_outputs), device_output_path=adb.output_folder)

    predictions = []
    for index in range(len(inputs)):
        raw_output = next(device_outputs.rglob(f"output_{index}_0.raw"))
        predictions.append(np.fromfile(raw_output, dtype=np.float32))

    top1 = topk_accuracy(predictions, targets, 1).item()
    top5 = topk_accuracy(predictions, targets, 5).item()
    print(f"device {device} ({soc_model}): top_1={top1}% top_5={top5}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soc_models", nargs="+", required=True, choices=QcomChipset.__members__
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("/tmp/qnn_fcb_resnet50_quantized")
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="path to the ImageNet validation folder used for PTQ calibration",
    )
    parser.add_argument("--calibration_samples", type=int, default=100)
    parser.add_argument("--host")
    parser.add_argument("--devices", nargs="+", default=[])
    parser.add_argument("--build_folder", default="build-android")
    args = parser.parse_args()
    if len(args.soc_models) < 2:
        parser.error("FCB requires at least two SoCs")

    model = ResNet50Model().get_eager_model().eval()
    calibration_inputs, targets = get_imagenet_dataset(
        dataset_path=args.dataset,
        data_size=args.calibration_samples,
        image_shape=(256, 256),
        crop_size=224,
        shuffle=False,
    )
    if not calibration_inputs:
        parser.error("no calibration images found in --dataset")
    sample_input = calibration_inputs[0]
    soc_models = [QcomChipset[name] for name in args.soc_models]
    quantized_model = quantize_model(model, soc_models, calibration_inputs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pte_path = args.output_dir / "resnet50_fcb_quantized.pte"
    pte_path.write_bytes(export_fcb(quantized_model, soc_models, sample_input))

    for device_with_host in args.devices:
        if ":" in device_with_host:
            host, device = device_with_host.split(":", 1)
        else:
            host, device = args.host, device_with_host
        run_on_device(
            pte_path,
            args.output_dir,
            host,
            device,
            args.build_folder,
            args.soc_models,
            calibration_inputs,
            targets,
        )


if __name__ == "__main__":
    main()
