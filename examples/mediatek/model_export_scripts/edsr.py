# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np

import torch
from executorch.backends.mediatek import Precision
from executorch.examples.mediatek.aot_utils.oss_utils.utils import (
    build_executorch_binary,
)
from executorch.examples.models.edsr import EdsrModel

from PIL import Image
from torch.utils.data import Dataset
from torchsr.datasets import B100
from torchvision.transforms.functional import to_tensor


class NhwcWrappedModel(torch.nn.Module):
    def __init__(self):
        super(NhwcWrappedModel, self).__init__()
        self.edsr = EdsrModel().get_eager_model()

    def forward(self, input1):
        nchw_input1 = input1.permute(0, 3, 1, 2)
        nchw_output = self.edsr(nchw_input1)
        return nchw_output.permute(0, 2, 3, 1)


class SrDataset(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str):
        self.input_size = np.asanyarray([224, 224])
        self.hr = []
        self.lr = []

        for file in sorted(os.listdir(hr_dir)):
            self.hr.append(self._resize_img(os.path.join(hr_dir, file), 2))

        for file in sorted(os.listdir(lr_dir)):
            self.lr.append(self._resize_img(os.path.join(lr_dir, file), 1))

        if len(self.hr) != len(self.lr):
            raise AssertionError(
                "The number of high resolution pics is not equal to low "
                "resolution pics"
            )

    def __getitem__(self, idx: int):
        return self.hr[idx], self.lr[idx]

    def __len__(self):
        return len(self.lr)

    def _resize_img(self, file: str, scale: int):
        with Image.open(file) as img:
            return (
                to_tensor(img.resize(tuple(self.input_size * scale)))
                .unsqueeze(0)
                .permute(0, 2, 3, 1)
            )

    def get_input_list(self):
        input_list = ""
        for i in range(len(self.lr)):
            input_list += f"input_{i}_0.bin\n"
        return input_list


def get_b100(
    dataset_dir: str,
):
    hr_dir = f"{dataset_dir}/sr_bm_dataset/SRBenchmarks/benchmark/B100/HR"
    lr_dir = f"{dataset_dir}/sr_bm_dataset/SRBenchmarks/benchmark/B100/LR_bicubic/X2"

    if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
        B100(root=f"{dataset_dir}/sr_bm_dataset", scale=2, download=True)

    return SrDataset(hr_dir, lr_dir)


def get_dataset(hr_dir: str, lr_dir: str, default_dataset: str, dataset_dir: str):
    if not (lr_dir and hr_dir) and not default_dataset:
        raise RuntimeError(
            "Nither custom dataset is provided nor using default dataset."
        )

    if (lr_dir and hr_dir) and default_dataset:
        raise RuntimeError("Either use custom dataset, or use default dataset.")

    if default_dataset:
        return get_b100(dataset_dir)

    return SrDataset(hr_dir, lr_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./edsr",
        default="./edsr",
        type=str,
    )

    parser.add_argument(
        "-r",
        "--hr_ref_dir",
        help="Path to the high resolution images",
        default="",
        type=str,
    )

    parser.add_argument(
        "-l",
        "--lr_dir",
        help="Path to the low resolution image inputs",
        default="",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--default_dataset",
        help="If specified, download and use B100 dataset by torchSR API",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    dataset = get_dataset(
        args.hr_ref_dir, args.lr_dir, args.default_dataset, args.artifact
    )

    inputs, targets, input_list = dataset.lr, dataset.hr, dataset.get_input_list()

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

    # build pte
    pte_filename = "edsr_mtk"
    instance = NhwcWrappedModel()
    build_executorch_binary(
        instance.eval(),
        (inputs[0],),
        f"{args.artifact}/{pte_filename}",
        [(input,) for input in inputs],
        quant_dtype=Precision.A8W8,
    )
