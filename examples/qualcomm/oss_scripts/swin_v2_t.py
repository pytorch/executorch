# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os

from multiprocessing.connection import Client

import numpy as np

import torch
import torchvision
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    FoldQDQ,
    get_capture_program_passes,
    get_passes_dependency_for_capture_program,
    QCOM_PASS_ACTIVATE_KEY,
    QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_imagenet_dataset,
    make_output_dir,
    make_quantizer,
    setup_common_args_and_variables,
    SimpleADB,
    topk_accuracy,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RewritePartition(ExportPass):
    """
    Rewrite 6D window partition pattern to 5D one.
    """

    def __init__(self):
        super(RewritePartition, self).__init__()

    def call(self, graph_module: torch.fx.GraphModule):
        graph = graph_module.graph
        # math equivalent implementation
        for node in graph.nodes:
            if (
                node.op == "call_function"
                and node.target == exir_ops.edge.aten.permute_copy.default
                and node.args[1] == [0, 1, 3, 2, 4, 5]
            ):
                # adjust original view node to take 5D tensor
                view_node = node.args[0]
                b, n_window_h, window_h, n_window_w, window_w, c = view_node.args[1]
                shape = [b, n_window_h, window_h, n_window_w, window_w * c]
                view_node.args = (view_node.args[0], shape)
                view_node.meta["val"] = view_node.meta["val"].reshape(shape)
                # change current permute node accordingly
                axis_order = [0, 1, 3, 2, 4]
                node.args = (view_node, axis_order)
                node.meta["val"] = view_node.meta["val"].permute(axis_order)

        graph_module.recompile()
        return PassResult(graph_module, True)


def main(args):
    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    data_num = 100
    if args.ci:
        inputs = [(torch.rand(1, 3, 224, 224),)]
        logging.warning(
            "This option is for CI to verify the export flow. It uses random input and will result in poor accuracy."
        )
    else:
        inputs, targets = get_imagenet_dataset(
            dataset_path=f"{args.dataset}",
            data_size=data_num,
            image_shape=(256, 256),
            crop_size=224,
        )

    pte_filename = "swin_v2_t_qnn_q8"
    instance = torchvision.models.swin_v2_t(weights="IMAGENET1K_V1").eval()
    passes_job = get_capture_program_passes()
    passes_job[RewritePartition] = {
        QCOM_PASS_ACTIVATE_KEY: True,
        QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY: {},
    }
    passes_dep = get_passes_dependency_for_capture_program()
    passes_dep[RewritePartition] = [FoldQDQ]
    build_executorch_binary(
        instance,
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        custom_quantizer=make_quantizer(
            quant_dtype=QuantDtype.use_8a8w,
            per_channel_linear=True,
        ),
        shared_buffer=args.shared_buffer,
        passes_job=passes_job,
        passes_dependency=passes_dep,
    )

    if args.compile_only:
        return

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
        target=args.target,
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

    # top-k analysis
    predictions = []
    for i in range(data_num):
        predictions.append(
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
        )

    k_val = [1, 5]
    topk = [topk_accuracy(predictions, targets, k).item() for k in k_val]
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({f"top_{k}": topk[i] for i, k in enumerate(k_val)}))
    else:
        for i, k in enumerate(k_val):
            print(f"top_{k}->{topk[i]}%")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-d",
        "--dataset",
        help=(
            "path to the validation folder of ImageNet dataset. "
            "e.g. --dataset imagenet-mini/val "
            "for https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)"
        ),
        type=str,
        required=False,
    )
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./swin_v2_t",
        default="./swin_v2_t",
        type=str,
    )

    args = parser.parse_args()
    args.validate(args)
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
