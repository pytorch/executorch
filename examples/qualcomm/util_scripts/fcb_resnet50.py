# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export an offline HTP FCB ResNet50 and optionally verify it on devices."""

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from executorch.backends.qualcomm.export_utils import QnnConfig, SimpleADB
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)
from executorch.examples.models.resnet import ResNet50Model


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


def export_fcb(model: torch.nn.Module, soc_models: list[QcomChipset], sample_input):
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_models,
        backend_options=[generate_htp_compiler_spec(use_fp16=True) for _ in soc_models],
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
    sample_input,
    expected: torch.Tensor,
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
        workspace=f"/data/local/tmp/qnn_fcb_resnet50/{device}",
    )
    adb.push(inputs=[sample_input], init_env=True)
    adb.execute(custom_runner_cmd=f"rm -rf {adb.output_folder}")
    adb.execute(method_index=0)
    device_outputs = output_dir / device
    shutil.rmtree(device_outputs, ignore_errors=True)
    adb.pull(str(device_outputs), device_output_path=adb.output_folder)
    raw_output = next(device_outputs.rglob("*.raw"))
    actual = torch.from_file(
        str(raw_output), dtype=expected.dtype, size=expected.numel()
    ).reshape(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-1)
    print(f"device {device} ({soc_model}): PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soc_models", nargs="+", required=True, choices=QcomChipset.__members__
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("/tmp/qnn_fcb_resnet50")
    )
    parser.add_argument("--host")
    parser.add_argument("--devices", nargs="+", default=[])
    parser.add_argument("--build_folder", default="build-android")
    args = parser.parse_args()
    if len(args.soc_models) < 2:
        parser.error("FCB requires at least two SoCs")

    torch.manual_seed(0)
    model = ResNet50Model().get_eager_model().eval()
    sample_input = (torch.randn(1, 3, 224, 224),)
    expected = model(*sample_input).detach()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pte_path = args.output_dir / "resnet50_fcb.pte"
    pte_path.write_bytes(
        export_fcb(model, [QcomChipset[name] for name in args.soc_models], sample_input)
    )

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
            sample_input,
            expected,
        )


if __name__ == "__main__":
    main()
