# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
from executorch.backends.mediatek import Precision
from executorch.examples.mediatek.aot_utils.oss_utils.utils import (
    build_executorch_binary,
)
from executorch.examples.models.inception_v3 import InceptionV3Model


class NhwcWrappedModel(torch.nn.Module):
    def __init__(self):
        super(NhwcWrappedModel, self).__init__()
        self.inception = InceptionV3Model().get_eager_model()

    def forward(self, input1):
        nchw_input1 = input1.permute(0, 3, 1, 2)
        output = self.inception(nchw_input1)
        return output


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
        feature = feature.permute(0, 2, 3, 1)  # NHWC
        inputs.append((feature,))
        targets.append(target)
        input_list += f"input_{index}_0.bin\n"

    return inputs, targets, input_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        help="path for storing generated artifacts by this example. "
        "Default ./inceptionV3",
        default="./inceptionV3",
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

    # save data to inference on device
    input_list_file = f"{args.artifact}/input_list.txt"
    with open(input_list_file, "w") as f:
        f.write(input_list)
        f.flush()
    for idx, data in enumerate(inputs):
        for i, d in enumerate(data):
            file_name = f"{args.artifact}/input_{idx}_{i}.bin"
            d.detach().numpy().tofile(file_name)
    for idx, data in enumerate(targets):
        file_name = f"{args.artifact}/golden_{idx}_0.bin"
        data.detach().numpy().tofile(file_name)

    pte_filename = "inceptionV3_mtk"
    instance = NhwcWrappedModel()
    build_executorch_binary(
        instance.eval(),
        (torch.randn(1, 224, 224, 3),),
        f"{args.artifact}/{pte_filename}",
        inputs,
        quant_dtype=Precision.A8W8,
    )
