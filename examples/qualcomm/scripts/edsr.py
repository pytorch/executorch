# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import re
from multiprocessing.connection import Client

import numpy as np
import piq
import torch
from executorch.examples.models.edsr import EdsrModel
from executorch.examples.qualcomm.scripts.utils import (
    build_executorch_binary,
    make_output_dir,
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


def annotate_forward(gm: torch.fx.GraphModule) -> None:
    """
    This function is specific for EDSR. It constructs a nn module, which is
    inherited from nn.conv2d.
    The source_fn of the rewritten nn module turns out to be a string "forward"
    """
    import itertools

    from executorch.backends.qualcomm.qnn_quantizer import (
        get_ptq_per_channel_weight_config,
    )
    from executorch.backends.qualcomm.qnn_quantizer_utils import (
        _is_annotated,
        QUANT_ANNOTATION_KEY,
    )
    from torch.ao.quantization.quantize_pt2e import QuantizationAnnotation
    from torch.fx import Node
    from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

    conv_partitions = get_source_partitions(gm.graph, ["forward"])
    conv_partitions = list(itertools.chain(*conv_partitions.values()))
    quantization_config = get_ptq_per_channel_weight_config()
    for conv_partition in conv_partitions:
        if len(conv_partition.output_nodes) > 1:
            raise ValueError("conv partition has more than one output node")
        conv_node = conv_partition.output_nodes[0]
        if (
            conv_node.op != "call_function"
            or conv_node.target != torch.ops.aten.conv2d.default
        ):
            raise ValueError(f"{conv_node} is not an aten conv2d operator")
        # skip annotation if it is already annotated
        if _is_annotated([conv_node]):
            continue

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_spec = quantization_config.input_activation
        input_qspec_map[input_act] = input_spec

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = quantization_config.weight

        if len(conv_node.args) > 2:
            bias = conv_node.args[2]
            if isinstance(bias, Node):
                if callable(quantization_config.bias):
                    input_qspec_map[bias] = quantization_config.bias(conv_node)
                else:
                    input_qspec_map[bias] = quantization_config.bias

        conv_node.meta[QUANT_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )


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
        "-b",
        "--build_folder",
        help="path to cmake binary directory for android, e.g., /path/to/build_android",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--device",
        help="serial number for android device communicated via ADB.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-H",
        "--host",
        help="hostname where android device is connected.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="SoC model of current device. e.g. 'SM8550' for Snapdragon 8 Gen 2.",
        type=str,
        required=True,
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
    parser.add_argument(
        "--ip",
        help="IPC address for delivering execution result",
        default="",
        type=str,
    )
    parser.add_argument(
        "--port",
        help="IPC port for delivering execution result",
        default=-1,
        type=int,
    )

    # QNN_SDK_ROOT might also be an argument, but it is used in various places.
    # So maybe it's fine to just use the environment.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError("Environment variable QNN_SDK_ROOT must be set")
    print(f"QNN_SDK_ROOT={os.getenv('QNN_SDK_ROOT')}")

    if "LD_LIBRARY_PATH" not in os.environ:
        print(
            "[Warning] LD_LIBRARY_PATH is not set. If errors like libQnnHtp.so "
            "not found happen, please follow setup.md to set environment."
        )
    else:
        print(f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    dataset = get_dataset(
        args.hr_ref_dir, args.lr_dir, args.default_dataset, args.artifact
    )

    inputs, targets, input_list = dataset.lr, dataset.hr, dataset.get_input_list()
    pte_filename = "edsr_qnn"
    instance = EdsrModel()

    build_executorch_binary(
        instance.get_eager_model().eval(),
        (inputs[0],),
        args.model,
        f"{args.artifact}/{pte_filename}",
        [(input,) for input in inputs],
        custom_annotations=(annotate_forward,),
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

    avg_PNSR = sum(psnr_list).item() / len(psnr_list)
    avg_SSIM = sum(ssim_list).item() / len(ssim_list)
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"PNSR": avg_PNSR, "SSIM": avg_SSIM}))
    else:
        print(f"Average of PNSR is: {avg_PNSR}")
        print(f"Average of SSIM is: {avg_SSIM}")
