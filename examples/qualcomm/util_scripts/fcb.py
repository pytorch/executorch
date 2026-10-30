# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export and optionally run Flexible Context Binary examples."""

import argparse
import shutil
import subprocess
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
from executorch.examples.qualcomm.utils import get_imagenet_dataset, topk_accuracy
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


def get_device_soc_model(host: str | None, device: str) -> str:
    command = ["adb"]
    if host:
        command.extend(["-H", host])
    command.extend(["-s", device, "shell", "getprop", "ro.soc.model"])
    soc_model = subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()
    if soc_model not in QcomChipset.__members__:
        raise RuntimeError(f"device {device} reported unsupported SoC {soc_model!r}")
    return soc_model


def get_devices(args):
    devices = []
    for device in args.devices:
        if ":" in device:
            devices.append(tuple(device.split(":", 1)))
        else:
            devices.append((args.host, device))
    return devices


def make_compiler_specs(
    soc_models,
    use_fp16,
    fcb_reference_weight_sharing=True,
    use_weight_sharing=False,
):
    return generate_qnn_executorch_compiler_spec(
        soc_model=soc_models,
        backend_options=[
            generate_htp_compiler_spec(
                use_fp16=use_fp16,
                use_weight_sharing=use_weight_sharing,
            )
            for _ in soc_models
        ],
        fcb_reference_weight_sharing=fcb_reference_weight_sharing,
    )


def export_fcb(module, inputs, compiler_specs):
    """Export a model to a Flexible Context Binary."""
    return (
        to_edge_transform_and_lower_to_qnn(
            module=module,
            inputs=inputs,
            compiler_specs=compiler_specs,
        )
        .to_executorch()
        .buffer
    )


def make_adb(args, host, device, pte_path, workspace):
    soc_model = get_device_soc_model(host, device)
    if soc_model not in args.soc_models:
        raise RuntimeError(
            f"device {device} has {soc_model}, not one of the prepared SoCs "
            f"{args.soc_models}"
        )
    return soc_model, SimpleADB(
        qnn_config=QnnConfig(
            soc_model=soc_model,
            build_folder=args.build_folder,
            device=device,
            host=host,
        ),
        pte_path=str(pte_path),
        workspace=f"/data/local/tmp/{workspace}/{device}",
    )


def quantize_model(model, soc_models, calibration_inputs):
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


def run_resnet50(args, quantized):
    model = ResNet50Model().get_eager_model().eval()
    soc_models = [QcomChipset[name] for name in args.soc_models]
    eval_inputs, targets = get_imagenet_dataset(
        dataset_path=args.dataset,
        data_size=args.eval_samples,
        image_shape=(256, 256),
        crop_size=224,
        shuffle=False,
    )
    if not eval_inputs:
        raise ValueError("no images found in --dataset")

    if quantized:
        calibration_inputs, _ = get_imagenet_dataset(
            dataset_path=args.dataset,
            data_size=args.calibration_samples,
            image_shape=(256, 256),
            crop_size=224,
            shuffle=False,
        )
        if not calibration_inputs:
            raise ValueError("no calibration images found in --dataset")
        model = quantize_model(model, soc_models, calibration_inputs)
        pte_stem = "resnet50_fcb_quantized"
    else:
        pte_stem = "resnet50_fcb"

    pte_results = []
    execution_pte = None
    for reference_weight_sharing in (True, False):
        pte_path = args.output_dir / (
            f"{pte_stem}_reference_weight_sharing={reference_weight_sharing}.pte"
        )
        pte_bytes = export_fcb(
            model,
            eval_inputs[0],
            make_compiler_specs(
                soc_models,
                use_fp16=not quantized,
                fcb_reference_weight_sharing=reference_weight_sharing,
            ),
        )
        pte_path.write_bytes(pte_bytes)
        pte_results.append((reference_weight_sharing, len(pte_bytes)))
        if reference_weight_sharing:
            execution_pte = pte_path

    print(f"Target SoCs: {args.soc_models}")
    print(f"{'Ref Weight Share':<18} | PTE Size (Bytes)")
    for reference_weight_sharing, size in pte_results:
        print(f"{str(reference_weight_sharing):<18} | {size:>16,}")

    for host, device in get_devices(args):
        soc_model, adb = make_adb(
            args, host, device, execution_pte, f"qnn_fcb_{args.model}"
        )
        adb.push(inputs=eval_inputs, init_env=True)
        adb.execute(custom_runner_cmd=f"rm -rf {adb.output_folder}")
        adb.execute(method_index=0)
        device_outputs = args.output_dir / device
        shutil.rmtree(device_outputs, ignore_errors=True)
        device_outputs.parent.mkdir(parents=True, exist_ok=True)
        adb.pull(str(device_outputs), device_output_path=adb.output_folder)
        predictions = [
            np.fromfile(
                next(device_outputs.rglob(f"output_{index}_0.raw")),
                dtype=np.float32,
            )
            for index in range(len(eval_inputs))
        ]
        top1 = topk_accuracy(predictions, targets, 1).item()
        top5 = topk_accuracy(predictions, targets, 5).item()
        print(f"device {device} ({soc_model}): top_1={top1}% top_5={top5}%")


class TwoConvs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(1, 3, 10)
        self.second = torch.nn.Conv2d(3, 2, 10)

    def forward(self, x):
        return self.second(self.first(x))


def run_weight_sharing(args):
    model = TwoConvs().eval()
    modules = {"two_convs": model, "second": model.second}
    inputs = {
        "two_convs": (torch.randn(1, 1, 80, 80),),
        "second": (torch.randn(1, 3, 60, 60),),
    }
    method_names = sorted(modules)
    modules = {name: modules[name] for name in method_names}
    inputs = {name: inputs[name] for name in method_names}
    soc_models = [QcomChipset[name] for name in args.soc_models]
    results = []
    execution_pte = None

    for use_weight_sharing in (True, False):
        for reference_weight_sharing in (True, False):
            compiler_specs = make_compiler_specs(
                soc_models,
                use_fp16=False,
                fcb_reference_weight_sharing=reference_weight_sharing,
                use_weight_sharing=use_weight_sharing,
            )
            pte_bytes = export_fcb(
                modules,
                inputs,
                {name: compiler_specs for name in modules},
            )
            pte_path = args.output_dir / (
                f"weight_sharing={use_weight_sharing}_"
                f"reference_weight_sharing={reference_weight_sharing}.pte"
            )
            pte_path.write_bytes(pte_bytes)
            results.append(
                (use_weight_sharing, reference_weight_sharing, len(pte_bytes))
            )
            if use_weight_sharing and reference_weight_sharing:
                execution_pte = pte_path

    print(f"Target SoCs: {args.soc_models}")
    print(f"{'Weight Share':<14} | {'Ref Weight Share':<18} | PTE Size (Bytes)")
    for use_weight_sharing, reference_weight_sharing, size in results:
        print(
            f"{str(use_weight_sharing):<14} | "
            f"{str(reference_weight_sharing):<18} | {size:>16,}"
        )

    for host, device in get_devices(args):
        soc_model, adb = make_adb(
            args, host, device, execution_pte, "qnn_fcb_weight_sharing"
        )
        for method_index, name in enumerate(modules):
            module = modules[name]
            expected = module(*inputs[name]).detach()
            adb.push(inputs=[inputs[name]], init_env=method_index == 0)
            adb.execute(custom_runner_cmd=f"rm -rf {adb.output_folder}")
            adb.execute(method_index=method_index)
            device_outputs = args.output_dir / device / name
            shutil.rmtree(device_outputs, ignore_errors=True)
            device_outputs.parent.mkdir(parents=True, exist_ok=True)
            adb.pull(str(device_outputs), device_output_path=adb.output_folder)
            raw_output = next(device_outputs.rglob("*.raw"))
            actual = torch.from_file(
                str(raw_output), dtype=expected.dtype, size=expected.numel()
            ).reshape(expected.shape)
            torch.testing.assert_close(actual, expected, rtol=1, atol=1e-1)
            print(f"device {device} ({soc_model}) method {method_index} ({name}): PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("resnet50", "weight_sharing"),
        help="FCB example to export",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Export ResNet50 in FP16 instead of quantizing it",
    )
    parser.add_argument(
        "--soc_models",
        nargs="+",
        required=True,
        choices=QcomChipset.__members__,
        help="Target Qualcomm SoCs to include in the FCB",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("/tmp/qnn_fcb"))
    parser.add_argument(
        "--dataset",
        help=(
            "Path to the ImageNet validation folder used for ResNet50 calibration "
            "and accuracy evaluation. For example, --dataset imagenet-mini/val "
            "for https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000"
        ),
    )
    parser.add_argument(
        "--calibration_samples",
        type=int,
        default=None,
        help="Maximum ImageNet samples for ResNet50 PTQ calibration. Defaults to all samples.",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=20,
        help="ImageNet samples pushed to each device and used for evaluation (default: 20).",
    )
    parser.add_argument(
        "--host",
        help="ADB host for remote devices. Omit it for locally connected devices.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=[],
        help="Devices in '[host:]serial' format",
    )
    parser.add_argument("--build_folder", default="build-android")
    args = parser.parse_args()

    if len(set(args.soc_models)) < 2:
        parser.error("FCB requires at least two distinct SoCs")
    if args.model == "resnet50" and not args.dataset:
        parser.error("--dataset is required to calibrate and evaluate resnet50")
    if args.eval_samples <= 0:
        parser.error("--eval_samples must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if args.model == "resnet50":
        run_resnet50(args, quantized=not args.use_fp16)
    else:
        run_weight_sharing(args)


if __name__ == "__main__":
    main()
