# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import sys
from multiprocessing.connection import Client

import numpy as np
import piq
import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.models.edsr import EdsrModel
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)

from PIL import Image
from torch.utils.data import Dataset
from torchsr.datasets import B100
from torchvision.transforms.functional import to_pil_image, to_tensor


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
            return to_tensor(img.resize(tuple(self.input_size * scale))).unsqueeze(0)

    def get_input_list(self):
        input_list = ""
        for i in range(len(self.lr)):
            input_list += f"input_{i}_0.raw\n"
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


def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

    dataset = get_dataset(
        args.hr_ref_dir, args.lr_dir, args.default_dataset, args.artifact
    )

    inputs, targets, input_list = dataset.lr, dataset.hr, dataset.get_input_list()
    pte_filename = "edsr_qnn_q8"
    instance = EdsrModel()

    build_executorch_binary(
        instance.get_eager_model().eval(),
        (inputs[0],),
        args.model,
        f"{args.artifact}/{pte_filename}",
        [(input,) for input in inputs],
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
        shared_buffer=args.shared_buffer,
    )

    if args.compile_only:
        sys.exit(0)

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
    adb.push(inputs=inputs, input_list=input_list)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    output_pic_folder = f"{args.artifact}/output_pics"
    make_output_dir(output_data_folder)
    make_output_dir(output_pic_folder)

    output_raws = []

    def post_process():
        cnt = 0
        output_shape = tuple(targets[0].size())
        for f in sorted(
            os.listdir(output_data_folder), key=lambda f: int(f.split("_")[1])
        ):
            filename = os.path.join(output_data_folder, f)
            if re.match(r"^output_[0-9]+_[1-9].raw$", f):
                os.remove(filename)
            else:
                output = np.fromfile(filename, dtype=np.float32)
                output = torch.tensor(output).reshape(output_shape).clamp(0, 1)
                output_raws.append(output)
                to_pil_image(output.squeeze(0)).save(
                    os.path.join(output_pic_folder, str(cnt) + ".png")
                )
                cnt += 1

    adb.pull(output_path=args.artifact, callback=post_process)

    psnr_list = []
    ssim_list = []
    for i, hr in enumerate(targets):
        psnr_list.append(piq.psnr(hr, output_raws[i]))
        ssim_list.append(piq.ssim(hr, output_raws[i]))

    avg_PSNR = sum(psnr_list).item() / len(psnr_list)
    avg_SSIM = sum(ssim_list).item() / len(ssim_list)
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"PSNR": avg_PSNR, "SSIM": avg_SSIM}))
    else:
        print(f"Average of PNSR is: {avg_PSNR}")
        print(f"Average of SSIM is: {avg_SSIM}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

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
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
