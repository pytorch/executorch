# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tutorial: LPAI depth-first cut plan via lpai_custom_prepare_config_file.

Depth-first cut splits one op's compute (and its working set) into several
smaller tiles so a graph that doesn't otherwise fit into LPAI's on-chip memory
can still be compiled. It is a memory-fitting mechanism, not a speedup lever:
expect comparable or slightly worse latency on a graph that already compiles
fine without it.

This script builds a tiny single-conv model, writes the two JSON files the
feature needs (a cut_plan file naming the node(s) to cut, and a
custom_prepare_config file that points at it), then lowers and runs the model
on-device once with lpai_custom_prepare_config_file set (cut), and compares
the device output against the CPU reference.

Requirements:
  * QNN SDK >= 2.51 (host-side QNN API >= 2.40). On an older SDK it logs
    a warning and drops the option, so this script just runs uncut. 
  * Please refer to the QNN documentation, as cut support is limited to certain operators.
  * A cut_plan entry naming more than one node must enumerate every node on
    every path from the first to the last listed node.

See generate_lpai_compiler_spec() in backends/qualcomm/utils/utils.py for the
full custom_prepare_config_file / cut_plan schema.
"""

import json
import os
from multiprocessing.connection import Client

import numpy as np
import torch

from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    Device,
    QnnConfig,
    setup_common_args_and_variables,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.examples.qualcomm.utils import make_output_dir


class ConvReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def write_custom_prepare_config_file(artifact_dir, nodes, h_branches, w_branches):
    cut_plan_path = f"{artifact_dir}/cut_plan_nodes.json"
    with open(cut_plan_path, "w") as f:
        json.dump(
            [{"nodes": nodes, "h_branches": h_branches, "w_branches": w_branches}], f
        )

    custom_prepare_config_path = f"{artifact_dir}/custom_prepare_config.json"
    with open(custom_prepare_config_path, "w") as f:
        json.dump({"cut_plan": cut_plan_path}, f)
    return custom_prepare_config_path


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)
    assert qnn_config.backend == QnnExecuTorchBackendType.kLpaiBackend, (
        "The depth-first cut plan is only applied by the LPAI backend; pass "
        "--backend lpai."
    )

    os.makedirs(args.artifact, exist_ok=True)

    qnn_config.lpai_custom_prepare_config_file = write_custom_prepare_config_file(
        args.artifact, args.nodes, args.h_branches, args.w_branches
    )

    model = ConvReluModel().eval()
    inputs = [(torch.randn(1, 3, 8, 8),)]
    reference_output = model(*inputs[0])

    pte_filename = "lpai_dfcut"
    build_executorch_binary(
        model=model,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=inputs,
        quant_dtype=QuantDtype.use_8a8w,
    )

    device = Device(
        qnn_config=qnn_config,
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
    )
    device.push(inputs=inputs)
    device.execute()

    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    device.pull(host_output_path=args.artifact)

    device_output = torch.from_numpy(
        np.fromfile(f"{output_data_folder}/output_0_0.raw", dtype=np.float32)
    ).reshape(reference_output.shape)

    is_close = torch.allclose(device_output, reference_output, atol=1e-1, rtol=1)
    max_abs_diff = (device_output - reference_output).abs().max().item()

    result = {"is_close": is_close, "max_abs_diff": max_abs_diff}
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps(result))
    else:
        print(f"output within tolerance: {is_close} (max abs diff: {max_abs_diff})")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./lpai_depth_first_cut",
        default="./lpai_depth_first_cut",
        type=str,
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        default=["aten_convolution_default"],
        help="Node name(s) to cut. For more than one node, every node on "
        "every path from the first to the last listed node must be included.",
    )
    parser.add_argument(
        "--h_branches", type=int, default=2, help="Number of height-axis tiles."
    )
    parser.add_argument(
        "--w_branches", type=int, default=1, help="Number of width-axis tiles."
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
