# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from multiprocessing.connection import Client
from pprint import PrettyPrinter

import numpy as np
import torch

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)


def create_data_lists(voc07_path, data_size):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param voc07_path: path to the 'VOC2007' folder
    """
    from utils import parse_annotation

    voc07_path = os.path.abspath(voc07_path)

    # Test data
    test_images = []
    test_objects = []
    n_objects = 0

    # Find IDs of images in the test data
    with open(os.path.join(voc07_path, "ImageSets/Main/test.txt")) as f:
        ids = f.read().splitlines()

    for index, id in enumerate(ids):
        if index >= data_size:
            break
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, "Annotations", id + ".xml"))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, "JPEGImages", id + ".jpg"))

    assert len(test_objects) == len(test_images)

    # TEST_images.json stores the file name of the images, and TEST_objects.json stores info such as boxes, labels, and difficulties
    with open(os.path.join(voc07_path, "TEST_images.json"), "w") as j:
        json.dump(test_images, j)
    with open(os.path.join(voc07_path, "TEST_objects.json"), "w") as j:
        json.dump(test_objects, j)

    print(
        "\nThere are %d test images containing a total of %d objects. Files have been saved to %s."
        % (len(test_images), n_objects, os.path.abspath(voc07_path))
    )


def get_dataset(data_size, dataset_dir, download):
    from datasets import PascalVOCDataset
    from torchvision import datasets

    if download:
        datasets.VOCSegmentation(
            root=os.path.join(dataset_dir, "voc_image"),
            year="2007",
            image_set="test",
            download=True,
        )
    voc07_path = os.path.join(dataset_dir, "voc_image", "VOCdevkit", "VOC2007")
    create_data_lists(voc07_path, data_size)

    # voc07_path is where the data and ground truth json file will be stored
    test_dataset = PascalVOCDataset(voc07_path, split="test", keep_difficult=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, collate_fn=test_dataset.collate_fn
    )

    inputs, input_list = [], ""
    true_boxes = []
    true_labels = []
    true_difficulties = []
    for index, (images, boxes, labels, difficulties) in enumerate(test_loader):
        if index >= data_size:
            break
        inputs.append((images,))
        input_list += f"input_{index}_0.raw\n"
        true_boxes.extend(boxes)
        true_labels.extend(labels)
        true_difficulties.extend(difficulties)

    return inputs, input_list, true_boxes, true_labels, true_difficulties


def SSD300VGG16(pretrained_weight_model):
    from model import SSD300

    model = SSD300(n_classes=21)
    # TODO: If possible, it's better to set weights_only to True
    # https://pytorch.org/docs/stable/generated/torch.load.html
    checkpoint = torch.load(
        pretrained_weight_model, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"].state_dict())

    return model.eval()


def main(args):
    sys.path.insert(0, args.oss_repo)

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

    data_num = 100
    inputs, input_list, true_boxes, true_labels, true_difficulties = get_dataset(
        data_size=data_num, dataset_dir=args.artifact, download=args.download
    )

    pte_filename = "ssd300_vgg16_qnn"
    model = SSD300VGG16(args.pretrained_weight)

    sample_input = (torch.randn((1, 3, 300, 300)),)
    build_executorch_binary(
        model,
        sample_input,
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_8a8w,
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
    )
    adb.push(inputs=inputs, input_list=input_list)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    det_boxes = []
    det_labels = []
    det_scores = []

    def post_process():
        from utils import calculate_mAP

        np.set_printoptions(threshold=np.inf)

        # output_xxx_0.raw is output of boxes, and output_xxx_1.raw is output of classes
        for file_index in range(data_num):
            boxes_filename = os.path.join(
                output_data_folder, f"output_{file_index}_0.raw"
            )
            category_filename = os.path.join(
                output_data_folder, f"output_{file_index}_1.raw"
            )

            predicted_locs = np.fromfile(boxes_filename, dtype=np.float32).reshape(
                [1, 8732, 4]
            )
            predicted_locs = torch.tensor(predicted_locs)

            predicted_scores = np.fromfile(category_filename, dtype=np.float32).reshape(
                [1, 8732, 21]
            )
            predicted_scores = torch.tensor(predicted_scores)

            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=200,
            )

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)

        pp = PrettyPrinter()
        # Calculate mAP
        APs, mAP = calculate_mAP(
            det_boxes,
            det_labels,
            det_scores,
            true_boxes,
            true_labels,
            true_difficulties,
        )
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"mAP": float(mAP)}))
        else:
            print("\nMean Average Precision (mAP): %.3f" % mAP)
            pp.pprint(APs)

    adb.pull(output_path=args.artifact, callback=post_process)


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./ssd300_vgg16",
        default="./ssd300_vgg16",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--download",
        help="If specified, download VOCSegmentation dataset by torchvision API",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--oss_repo",
        help=(
            "Repository that contains model backbone and score calculation."
            "e.g., --M ./a-PyTorch-Tutorial-to-Object-Detection"
            "Please clone the repository from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection"
        ),
        type=str,
        required=True,
    )

    parser.add_argument(
        "-p",
        "--pretrained_weight",
        help=(
            "Location of model pretrained weight."
            "e.g., -p ./checkpoint_ssd300.pth.tar"
            "Pretrained model can be found in the link https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection, under the Training Section"
        ),
        type=str,
        required=True,
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
