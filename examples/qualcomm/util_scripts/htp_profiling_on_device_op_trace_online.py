# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTP profiling demo — on-device OpTrace with online_prepare.

This example exports a `.pte` with `QnnConfig.online_prepare=True`, generates
on-device HTP OpTrace results with `generate_htp_profile_result()`, and opens
the QHAS report with QAIRT Visualizer.

Internal detail: online_prepare stores a graph description in the `.pte`; QNN
tools create the profiled context and schematic during profiling.

For the offline_prepare OpTrace route, see
htp_profiling_on_device_op_trace_offline.py. For host-only Hextimate, see
htp_profiling_on_host_hextimate.py.
"""

import json
import os
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm.debugger.utils import generate_htp_profile_result
from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.tests.models import SimpleModel
from executorch.backends.qualcomm.utils.utils import get_soc_to_chipset_map


def main(args) -> None:
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)
    assert qnn_config.online_prepare, (
        "This demo uses on-device OpTrace with online_prepare; pass "
        "--online_prepare (or set online_prepare=True in your config file). "
        "For offline prepare, use htp_profiling_on_device_op_trace_offline.py."
    )

    model = SimpleModel()
    example_inputs = [(torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))]

    pte_filename = "qnn_simple_model"
    os.makedirs(args.artifact, exist_ok=True)

    # lower to QNN
    build_executorch_binary(
        model=model,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=example_inputs,
        quant_dtype=QuantDtype.use_8a8w,
    )

    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
    )
    artifacts = generate_htp_profile_result(
        artifact_dir=args.artifact,
        soc_id=get_soc_to_chipset_map()[args.soc_model],
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        inputs=example_inputs,
        adb=adb,
    )

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({
                "artifacts": [
                    {
                        "binary_path": a.binary_path,
                        "mode": a.mode,
                        "prepare_mode": a.prepare_mode,
                        "chrometrace_json": a.chrometrace_json,
                        "qhas_json": a.qhas_json,
                        "qhas_html": a.qhas_html,
                    }
                    for a in artifacts
                ],
            }))
        return

    try:
        import qairt_visualizer
    except ImportError:
        for a in artifacts:
            print(f"QHAS HTML: {a.qhas_html}")
        return

    # Visualize each binary. Online-prepare emits .dlc, which supports the
    # graph view; offline .bin binaries do not.
    for a in artifacts:
        if a.binary_path.endswith(".dlc"):
            qairt_visualizer.view(a.binary_path, reports=a.visualizer_reports())
        else:
            qairt_visualizer.view(reports=a.visualizer_reports())


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        type=str,
        default="",
        help="The folder to store the exported program",
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
