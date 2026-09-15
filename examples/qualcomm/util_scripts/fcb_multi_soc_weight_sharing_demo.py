# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure FCB multi-SoC reference-weight sharing during QNN AOT.

Examples of Multi-Device and Multi-Host execution:

1. Standard single device / single host setup:
   python -m examples.qualcomm.util_scripts.fcb_multi_soc_weight_sharing_demo \
       --soc_models SM8650 SM8750 SM8850 SM8550 \
       --host adb-host1 --devices adb-device1

2. Advanced multi-device / multi-host setup:
   We can target multiple devices across different adb servers by passing 
   the devices in the format '[host:]device_id'. This enables testing models 
   across various HTP architectures concurrently.
   
   python -m examples.qualcomm.util_scripts.fcb_multi_soc_weight_sharing_demo \
       --soc_models SM8650 SM8750 SM8850 SM8550 \
       --devices adb-host1:adb-device1 adb-host2:adb-device2
"""

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List

import torch
from executorch.backends.qualcomm.export_utils import QnnConfig, SimpleADB

from executorch.backends.qualcomm.serialization.qc_schema import (
    _soc_info_table,
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    to_edge_transform_and_lower_to_qnn,
)


class TwoConvs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(1, 3, 10)
        self.second = torch.nn.Conv2d(3, 2, 10)

    def forward(self, x):
        return self.second(self.first(x))


model = TwoConvs().eval()
modules = {"two_convs": model, "second": model.second}

inputs = {
    "two_convs": (torch.randn(1, 1, 80, 80),),
    "second": (torch.randn(1, 3, 60, 60),),
}


def export_fcb(
    soc_models: List[QcomChipset],
    fcb_reference_weight_sharing: bool,
    use_weight_sharing: bool,
):
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=soc_models,
        backend_options=[
            generate_htp_compiler_spec(
                use_fp16=False, use_weight_sharing=use_weight_sharing
            )
            for _ in soc_models
        ],
        fcb_reference_weight_sharing=fcb_reference_weight_sharing,
    )
    program = to_edge_transform_and_lower_to_qnn(
        module=modules,
        inputs=inputs,
        compiler_specs={name: compiler_specs for name in modules},
    ).to_executorch()
    return program.buffer


def get_device_soc_model(host, device):
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


def get_htp_arch(soc_model: str) -> str:
    try:
        chipset = QcomChipset[soc_model]
        if chipset in _soc_info_table:
            return _soc_info_table[chipset].htp_info.htp_arch.name
    except Exception:
        pass
    return "UNKNOWN"


def run_on_device(pte_path, output_dir, host, device, build_folder, requested_socs):
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
        workspace=f"/data/local/tmp/qnn_fcb_weight_sharing/{device}",
    )
    for method_index, (name, module) in enumerate(sorted(modules.items())):
        expected = module(*inputs[name]).detach()
        adb.push(inputs=[inputs[name]], init_env=method_index == 0)
        adb.execute(custom_runner_cmd=f"rm -rf {adb.output_folder}")
        adb.execute(method_index=method_index)
        device_outputs = output_dir / device / name
        shutil.rmtree(device_outputs, ignore_errors=True)
        device_outputs.parent.mkdir(parents=True, exist_ok=True)
        adb.pull(str(device_outputs), device_output_path=adb.output_folder)
        raw_output = next(device_outputs.rglob("*.raw"))
        actual = torch.from_file(
            str(raw_output), dtype=expected.dtype, size=expected.numel()
        ).reshape(expected.shape)
        torch.testing.assert_close(actual, expected, rtol=1, atol=1e-1)
        shutil.move(raw_output, output_dir / f"{device}_{name}.raw")
        print(f"device {device} ({soc_model}) method {method_index} ({name}): PASS")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--soc_models",
        nargs="+",
        required=True,
        choices=QcomChipset.__members__,
        help="Target Qualcomm SoCs to compile context binaries for",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/tmp/qnn_fcb_weight_sharing"),
        help="Directory to save compile and runtime artifacts",
    )
    parser.add_argument(
        "--host",
        help="Fallback ADB server host name or IP address if no host prefix is specified in --devices",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=[],
        help="List of target devices to test. Can be specified in the format '[host:]device_id' to run across multiple ADB servers concurrently.",
    )
    parser.add_argument(
        "--build_folder",
        default="build-android",
        help="The android compilation folder containing target executables and runner binaries",
    )
    args = parser.parse_args()

    if len(args.soc_models) < 2:
        parser.error("FCB requires at least two SoCs")

    devices_with_hosts = []
    for d in args.devices:
        if ":" in d:
            parts = d.split(":", 1)
            devices_with_hosts.append((parts[0], parts[1]))
        else:
            devices_with_hosts.append((args.host, d))

    print("=" * 80)
    print("  DETECTED DEVICES REPORT  ".center(80, "="))
    print("=" * 80)
    print(f"{'Host':<20} | {'Device ID':<16} | {'SoC Model':<12} | {'HTP Arch':<10}")
    print("-" * 80)
    for host, device in devices_with_hosts:
        try:
            soc_model = get_device_soc_model(host, device)
            htp_arch = get_htp_arch(soc_model)
            print(
                f"{str(host or 'localhost'):<20} | {device:<16} | {soc_model:<12} | {htp_arch:<10}"
            )
        except Exception as e:
            print(f"{str(host or 'localhost'):<20} | {device:<16} | ERROR: {e}")
    print("=" * 80 + "\n")

    soc_models = [QcomChipset[model] for model in args.soc_models]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print(f"socs: { args.soc_models}")
    results = []
    for ws in [True, False]:
        for rws in [True, False]:
            pte_bytes = export_fcb(soc_models, rws, ws)
            pte_path = args.output_dir / f"ws={ws}_rws={rws}.pte"
            pte_path.write_bytes(pte_bytes)
            results.append(
                {
                    "ws": ws,
                    "rws": rws,
                    "pte_size": len(pte_bytes),
                }
            )
    print("\n" + "=" * 80)
    print("  FCB MULTI-SOC WEIGHT SHARING SUMMARY REPORT  ".center(80, "="))
    print("=" * 80)
    print(f"Target SoCs: {args.soc_models}")
    print("-" * 80)
    print(f"{'Weight Share':<14} | {'Ref Weight Share':<18} | {'PTE Size (Bytes)':<16}")
    print("-" * 80)
    for res in results:
        ws_str = str(res["ws"])
        rws_str = str(res["rws"])
        size_str = f"{res['pte_size']:,}"
        print(f"{ws_str:<14} | {rws_str:<18} | {size_str:>16}")
    print("-" * 80)
    for host, device in devices_with_hosts:
        run_on_device(
            pte_path,
            args.output_dir,
            host,
            device,
            args.build_folder,
            args.soc_models,
        )


if __name__ == "__main__":
    main()
