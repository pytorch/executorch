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

    pte_filename = "vit_b_16_qnn_q8"
    instance = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").eval()
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
        "Default ./vit_b_16",
        default="./vit_b_16",
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
