# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from multiprocessing.connection import Client

import qairt_visualizer
import torch
from executorch.backends.qualcomm.debugger.utils import generate_optrace
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
    model = SimpleModel()
    example_inputs = [(torch.ones(1, 32, 28, 28), torch.ones(1, 32, 28, 28))]

    pte_filename = "qnn_simple_model"
    os.makedirs(args.artifact, exist_ok=True)

    assert (
        qnn_config.profile_level == 3
    ), "Please turn profile_level to 3 for the purpose of this tutorial."

    # lower to QNN
    build_executorch_binary(
        model=model,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=example_inputs,
        quant_dtype=QuantDtype.use_8a8w,
    )

    # generate optrace and QHAS
    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
    )
    binaries_trace = generate_optrace(
        args.artifact,
        get_soc_to_chipset_map()[args.soc_model],
        adb,
        f"{args.artifact}/{pte_filename}.pte",
        example_inputs,
    )

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"binaries_trace": binaries_trace}))
    else:
        # Visualize the model and reports
        for binary, (optrace, qhas) in binaries_trace.items():
            file_extension = os.path.splitext(binary)[-1]
            if file_extension == ".bin":
                qairt_visualizer.view(reports=[optrace, qhas])
            elif file_extension == ".dlc":
                # We only show graph for dlc binary
                qairt_visualizer.view(binary, reports=[optrace, qhas])


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
