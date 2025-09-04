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
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_imagenet_dataset,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
    topk_accuracy,
)

from transformers import AutoModelForImageClassification
from transformers.models.swin import modeling_swin


# Copy from transformers/models/swin/modeling_swin.py in transformers 4.47.1
# (QCOM) Transform 6D dim to 5D dim
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape
    # ====================Qualcomm Changed=================================
    input_feature = input_feature.view(
        batch_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size * num_channels,  # Merge the last two dimensions
    )
    windows = input_feature.permute(0, 1, 3, 2, 4).contiguous()
    windows = windows.view(-1, window_size, window_size, num_channels)
    # =====================================================================
    return windows


# Copy from transformers/models/swin/modeling_swin.py in transformers 4.47.1
# (QCOM) Transform 6D dim to 5D dim tests on huggingface version (4.47.1)
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    # ====================Qualcomm Changed=================================
    windows = windows.view(
        -1,
        height // window_size,
        width // window_size,
        window_size,
        window_size * num_channels,  # Merge the last two dimensions
    )
    windows = windows.permute(0, 1, 3, 2, 4).contiguous()
    windows = windows.view(-1, height, width, num_channels)
    # =====================================================================
    return windows


# (QCOM) Replace the original window_partition and window_reverse functions
# in the modeling_swin module with the new ones, due to QNN SDK does not support 6D tensor.
modeling_swin.window_partition = window_partition
modeling_swin.window_reverse = window_reverse


def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

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

    module = (
        AutoModelForImageClassification.from_pretrained(
            "microsoft/swin-tiny-patch4-window7-224"
        )
        .eval()
        .to("cpu")
    )

    pte_filename = "swin_qnn_q8"
    build_executorch_binary(
        module.eval(),
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
        shared_buffer=args.shared_buffer,
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
        "Default ./swin_transformer",
        default="./swin_transformer",
        type=str,
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
