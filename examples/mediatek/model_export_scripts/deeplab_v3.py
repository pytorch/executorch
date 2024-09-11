# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random

import numpy as np

import torch
from executorch.backends.mediatek import Precision
from executorch.examples.mediatek.aot_utils.oss_utils.utils import (
    build_executorch_binary,
)
from executorch.examples.models.deeplab_v3 import DeepLabV3ResNet101Model


class NhwcWrappedModel(torch.nn.Module):
    def __init__(self):
        super(NhwcWrappedModel, self).__init__()
        self.deeplabv3 = DeepLabV3ResNet101Model().get_eager_model()

    def forward(self, input1):
        nchw_input1 = input1.permute(0, 3, 1, 2)
        nchw_output = self.deeplabv3(nchw_input1)
        return nchw_output.permute(0, 2, 3, 1)


def get_dataset(data_size, dataset_dir, download):
    from torchvision import datasets, transforms

    input_size = (224, 224)
    preprocess = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = list(
        datasets.VOCSegmentation(
            root=os.path.join(dataset_dir, "voc_image"),
            year="2009",
            image_set="val",
            transform=preprocess,
            download=download,
        )
    )

    # prepare input data
    random.shuffle(dataset)
    inputs, targets, input_list = [], [], ""
    for index, data in enumerate(dataset):
        if index >= data_size:
            break
        image, target = data
        inputs.append((image.unsqueeze(0).permute(0, 2, 3, 1),))
        targets.append(np.array(target.resize(input_size)))
        input_list += f"input_{index}_0.bin\n"

    return inputs, targets, input_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./deeplab_v3",
        default="./deeplab_v3",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--download",
        help="If specified, download VOCSegmentation dataset by torchvision API",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    data_num = 100
    inputs, targets, input_list = get_dataset(
        data_size=data_num, dataset_dir=args.artifact, download=args.download
    )

    # save data to inference on device
    input_list_file = f"{args.artifact}/input_list.txt"
    with open(input_list_file, "w") as f:
        f.write(input_list)
        f.flush()
    for idx, data in enumerate(inputs):
        for i, d in enumerate(data):
            file_name = f"{args.artifact}/input_{idx}_{i}.bin"
            d.detach().numpy().tofile(file_name)
            if idx == 0:
                print("inp shape: ", d.detach().numpy().shape)
                print("inp type: ", d.detach().numpy().dtype)
    for idx, data in enumerate(targets):
        file_name = f"{args.artifact}/golden_{idx}_0.bin"
        data.tofile(file_name)
        if idx == 0:
            print("golden shape: ", data.shape)
            print("golden type: ", data.dtype)

    # build pte
    pte_filename = "deeplabV3Resnet101_mtk"
    instance = NhwcWrappedModel()
    build_executorch_binary(
        instance.eval(),
        (torch.randn(1, 224, 224, 3),),
        f"{args.artifact}/{pte_filename}",
        inputs,
        quant_dtype=Precision.A8W8,
    )
