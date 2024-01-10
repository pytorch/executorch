# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
import re
import sys
from multiprocessing.connection import Client

import numpy as np

from executorch.examples.models.deeplab_v3 import DeepLabV3ResNet101Model
from executorch.examples.qualcomm.scripts.utils import (
    build_executorch_binary,
    make_output_dir,
    segmentation_metrics,
    SimpleADB,
)


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
    inputs, targets, input_list = [], [], ""
    for index, data in enumerate(dataset):
        if index >= data_size:
            break
        image, target = data
        inputs.append((image.unsqueeze(0),))
        targets.append(np.array(target.resize(input_size)))
        input_list += f"input_{index}_0.raw\n"

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
        "-d",
        "--download",
        help="If specified, download VOCSegmentation dataset by torchvision API",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--compile_only",
        help="If specified, only compile the model.",
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

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

    data_num = 100
    inputs, targets, input_list = get_dataset(
        data_size=data_num, dataset_dir=args.artifact, download=args.download
    )
    pte_filename = "dlv3_qnn"
    instance = DeepLabV3ResNet101Model()

    build_executorch_binary(
        instance.get_eager_model().eval(),
        instance.get_example_inputs(),
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
    )

    if args.compile_only:
        sys.exit(0)

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

    adb.pull(output_path=args.artifact, callback=post_process)

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
