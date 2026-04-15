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
from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    make_quantizer,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.examples.qualcomm.utils import make_output_dir, topk_accuracy
from PIL import Image
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification


def get_imagenet_dataset(dataset_path, data_size, shuffle=True):

    def get_data_loader():
        imagenet_data = datasets.ImageFolder(dataset_path)
        return torch.utils.data.DataLoader(
            imagenet_data,
            shuffle=shuffle,
        )

    # prepare input data
    inputs, targets = [], []
    data_loader = get_data_loader()
    image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-xx-small")
    for index, data in enumerate(data_loader.dataset.imgs):
        if index >= data_size:
            break
        data_path, target = data
        image = Image.open(data_path).convert("RGB")
        feature = image_processor(images=image, return_tensors="pt")
        inputs.append((feature["pixel_values"],))
        targets.append(torch.tensor(target))

    return inputs, targets


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

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
        )

    module = (
        AutoModelForImageClassification.from_pretrained("apple/mobilevit-xx-small")
        .eval()
        .to("cpu")
    )

    pte_filename = "mobilevit_v1_qnn"
    quantizer = {
        QnnExecuTorchBackendType.kGpuBackend: None,
        QnnExecuTorchBackendType.kHtpBackend: make_quantizer(
            quant_dtype=QuantDtype.use_16a8w,
            eps=2**-12,
            backend=qnn_config.backend,
            soc_model=qnn_config.soc_model,
        ),
    }[qnn_config.backend]
    build_executorch_binary(
        model=module.eval(),
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=inputs,
        custom_quantizer=quantizer,
    )

    adb = SimpleADB(
        qnn_config=qnn_config,
        pte_path=f"{args.artifact}/{pte_filename}.pte",
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(host_output_path=args.artifact)

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
        "Default ./mobilevit_v1",
        default="./mobilevit_v1",
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
