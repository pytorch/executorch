# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import random
import re
from multiprocessing.connection import Client

import numpy as np
import torch

from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.examples.models.deeplab_v3 import DeepLabV3ResNet101Model
from executorch.examples.qualcomm.utils import make_output_dir, segmentation_metrics


def get_dataset(data_size, dataset_dir, download):
    import numpy as np
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
            year="2012",
            image_set="val",
            transform=preprocess,
            download=download,
        )
    )

    # prepare input data
    random.shuffle(dataset)
    inputs, targets = [], []
    for index, data in enumerate(dataset):
        if index >= data_size:
            break
        image, target = data
        inputs.append((image.unsqueeze(0),))
        targets.append(np.array(target.resize(input_size)))

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
        inputs, targets = get_dataset(
            data_size=data_num, dataset_dir=args.artifact, download=args.download
        )

    pte_filename = "dlv3_qnn"
    instance = DeepLabV3ResNet101Model()
    quant_dtype = {
        QnnExecuTorchBackendType.kLpaiBackend: QuantDtype.use_8a8w,
        QnnExecuTorchBackendType.kGpuBackend: None,
        QnnExecuTorchBackendType.kHtpBackend: QuantDtype.use_8a8w,
    }[qnn_config.backend]

    build_executorch_binary(
        model=instance.get_eager_model().eval(),
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=inputs,
        quant_dtype=quant_dtype,
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

    # remove the auxiliary output and data processing
    classes = [
        "Backround",
        "Aeroplane",
        "Bicycle",
        "Bird",
        "Boat",
        "Bottle",
        "Bus",
        "Car",
        "Cat",
        "Chair",
        "Cow",
        "DiningTable",
        "Dog",
        "Horse",
        "MotorBike",
        "Person",
        "PottedPlant",
        "Sheep",
        "Sofa",
        "Train",
        "TvMonitor",
    ]

    def post_process():
        for f in os.listdir(output_data_folder):
            filename = os.path.join(output_data_folder, f)
            if re.match(r"^output_[0-9]+_[1-9].raw$", f):
                os.remove(filename)
            else:
                output = np.fromfile(filename, dtype=np.float32)
                output_shape = [len(classes), 224, 224]
                output = output.reshape(output_shape)
                output.argmax(0).astype(np.uint8).tofile(filename)

    adb.pull(host_output_path=args.artifact, callback=post_process)

    # segmentation metrics
    predictions = []
    for i in range(data_num):
        predictions.append(
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.uint8
            )
        )

    pa, mpa, miou, cls_iou = segmentation_metrics(predictions, targets, classes)
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps({"PA": float(pa), "MPA": float(mpa), "MIoU": float(miou)})
            )
    else:
        print(f"PA   : {pa}%")
        print(f"MPA  : {mpa}%")
        print(f"MIoU : {miou}%")
        print(f"CIoU : \n{json.dumps(cls_iou, indent=2)}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

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

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
