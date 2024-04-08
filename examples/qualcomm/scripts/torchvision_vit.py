# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from multiprocessing.connection import Client

import numpy as np

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.models.torchvision_vit.model import TorchVisionViTModel
from executorch.examples.qualcomm.scripts.utils import (
    build_executorch_binary,
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
    topk_accuracy,
)


def get_dataset(dataset_path, data_size):
    from torchvision import datasets, transforms

    def get_data_loader():
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        imagenet_data = datasets.ImageFolder(dataset_path, transform=preprocess)
        return torch.utils.data.DataLoader(
            imagenet_data,
            shuffle=True,
        )

    # prepare input data
    inputs, targets, input_list = [], [], ""
    data_loader = get_data_loader()
    for index, data in enumerate(data_loader):
        if index >= data_size:
            break
        feature, target = data
        inputs.append(feature)
        targets.append(target)
        input_list += f"input_{index}_0.raw\n"

    return inputs, targets, input_list


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
        required=True,
    )
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. " "Default ./vit",
        default="./vit",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    data_num = 100
    inputs, targets, input_list = get_dataset(
        dataset_path=f"{args.dataset}",
        data_size=data_num,
    )
    pte_filename = "vit_qnn"
    instance = TorchVisionViTModel()
    build_executorch_binary(
        instance.get_eager_model().eval(),
        instance.get_example_inputs(),
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        quant_dtype=QuantDtype.use_8a8w,
        shared_buffer=args.shared_buffer,
    )
    # setup required paths accordingly
    # qnn_sdk       : QNN SDK path setup in environment variable
    # artifact_path : path where artifacts were built
    # pte_path      : path where executorch binary was stored
    # device_id     : serial number of android device
    # workspace     : folder for storing artifacts on android device
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        artifact_path=f"{args.build_folder}",
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
    )
    adb.push(inputs=inputs, input_list=input_list)
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
