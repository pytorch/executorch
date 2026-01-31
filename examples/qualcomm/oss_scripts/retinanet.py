# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import sys
from multiprocessing.connection import Client

import numpy as np

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_backend_type,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)


def get_instance():
    import torchvision
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights

    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    )

    # the post-process part in vanilla forward method failed to be exported
    # here we only gather the network structure for torch.export.export to work
    def forward_without_metrics(self, image):
        features = self.backbone(image)
        return self.head(list(features.values()))

    model.forward = lambda img: forward_without_metrics(model, img)
    return model.eval()


def get_dataset(data_size, dataset_dir):
    from torchvision import datasets, transforms

    class COCODataset(datasets.CocoDetection):
        def __init__(self, dataset_root):
            self.images_path = os.path.join(dataset_root, "val2017")
            self.annots_path = os.path.join(
                dataset_root, "annotations/instances_val2017.json"
            )
            self.img_shape = (640, 640)
            self.preprocess = transforms.Compose(
                [
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Resize(self.img_shape),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            with open(self.annots_path, "r") as f:
                data = json.load(f)
                categories = data["categories"]
                self.label_names = {
                    category["id"]: category["name"] for category in categories
                }

            super().__init__(root=self.images_path, annFile=self.annots_path)

        def __getitem__(self, index):
            img, target = super().__getitem__(index)

            bboxes, labels = [], []
            for obj in target:
                bboxes.append(self.resize_bbox(obj["bbox"], img.size))
                labels.append(obj["category_id"])

            # return empty list if no label exists
            return (
                self.preprocess(img),
                torch.stack(bboxes) if len(bboxes) > 0 else [],
                torch.Tensor(labels).to(torch.int) if len(labels) > 0 else [],
            )

        def resize_bbox(self, bbox, orig_shape):
            # bypass if no label exists
            if len(bbox) == 0:
                return

            y_scale = float(self.img_shape[0]) / orig_shape[0]
            x_scale = float(self.img_shape[1]) / orig_shape[1]
            # bbox: [(upper-left) x, y, w, h]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            # rescale bbox according to image shape
            bbox[0] = y_scale * bbox[0]
            bbox[2] = y_scale * bbox[2]
            bbox[1] = x_scale * bbox[1]
            bbox[3] = x_scale * bbox[3]
            return torch.Tensor(bbox)

    dataset = COCODataset(dataset_root=dataset_dir)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True)
    inputs = []
    bboxes, targets = [], []
    for index, (img, boxes, labels) in enumerate(test_loader):
        if index >= data_size:
            break
        inputs.append((img,))
        bboxes.append(boxes)
        targets.append(labels)

    return inputs, bboxes, targets, dataset.label_names


def calculate_precision(
    true_boxes, true_labels, det_boxes, det_labels, tp, fp, top_k, iou_thres
):
    import torchvision

    def collect_data(boxes, labels, top_k=-1):
        # extract data up to top_k length
        top_k = labels.size(0) if top_k == -1 else top_k
        len_labels = min(labels.size(0), top_k)
        boxes, labels = boxes[:len_labels, :], labels[:len_labels]
        # how many labels do we have in current data
        cls = set(labels[:len_labels].tolist())
        map = {index: [] for index in cls}
        # stack data in same class
        for j in range(len_labels):
            index = labels[j].item()
            if index in cls:
                map[index].append(boxes[j, :])
        return {k: torch.stack(v) for k, v in map.items()}

    preds = collect_data(det_boxes, det_labels, top_k=top_k)
    targets = collect_data(true_boxes.squeeze(0), true_labels.squeeze(0))
    # evaluate data with labels presenting in ground truth data
    for index in targets.keys():
        # there is no precision gain for predictions not present in ground truth data
        if index in preds:
            # targets shape: (M, 4), preds shape: (N, 4)
            # shape after box_iou: (M, N), iou shape: (M)
            # true-positive: how many predictions meet the iou threshold. i.e. k of M
            # false-positive: M - true-positive = M - k
            iou, _ = torchvision.ops.box_iou(targets[index], preds[index]).max(0)
            tps = torch.where(iou >= iou_thres, 1, 0).sum().item()
            tp[index - 1] += tps
            fp[index - 1] += iou.nelement() - tps


def eval_metric(instance, heads, images, bboxes, targets, classes):
    tp, fp = classes * [0], classes * [0]
    head_label = ["cls_logits", "bbox_regression"]

    # feature size should be changed if input size got altered
    feature_size = [80, 40, 20, 10, 5]
    feature_maps = [torch.zeros(1, 256, h, h) for h in feature_size]
    for head, image, true_boxes, true_labels in zip(heads, images, bboxes, targets):
        anchors = instance.anchor_generator(
            image_list=image,
            feature_maps=feature_maps,
        )
        num_anchors_per_level = [hw**2 * 9 for hw in feature_size]
        # split outputs per level
        split_head_outputs = {
            head_label[i]: list(h.split(num_anchors_per_level, dim=1))
            for i, h in enumerate(head)
        }
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
        # compute the detections (based on official post-process method)
        detection = instance.postprocess_detections(
            head_outputs=split_head_outputs,
            anchors=split_anchors,
            image_shapes=[image.image_sizes],
        )
        # no contribution to precision
        if len(true_labels) == 0:
            continue
        # here we select top 10 confidence and iou >= 0.5 as the criteria
        calculate_precision(
            true_boxes=true_boxes,
            true_labels=true_labels,
            det_boxes=detection[0]["boxes"],
            det_labels=detection[0]["labels"],
            tp=tp,
            fp=fp,
            top_k=10,
            iou_thres=0.5,
        )

    # remove labels which does not appear in current dataset
    AP = torch.Tensor(
        [
            tp[i] * 1.0 / (tp[i] + fp[i]) if tp[i] + fp[i] > 0 else -1
            for i in range(len(tp))
        ]
    )
    missed_labels = torch.where(AP == -1, 1, 0).sum()
    mAP = AP.where(AP != -1, 0).sum() / (AP.nelement() - missed_labels)
    return AP, mAP.item()


def main(args):
    from pprint import PrettyPrinter

    from torchvision.models.detection.image_list import ImageList

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist
    os.makedirs(args.artifact, exist_ok=True)

    model = get_instance()

    # retrieve dataset
    data_num = 100
    # 91 classes appear in COCO dataset
    n_classes, n_coord_of_bbox = 91, 4
    inputs, bboxes, targets, label_names = get_dataset(
        data_size=data_num, dataset_dir=args.dataset
    )
    pte_filename = "retinanet_qnn"
    backend = get_backend_type(args.backend)
    quant_dtype = {
        QnnExecuTorchBackendType.kGpuBackend: None,
        QnnExecuTorchBackendType.kHtpBackend: QuantDtype.use_8a8w,
    }[backend]
    build_executorch_binary(
        model,
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=quant_dtype,
        backend=backend,
        shared_buffer=args.shared_buffer,
        online_prepare=args.online_prepare,
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
        target=args.target,
        backend=backend,
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    adb.pull(host_output_path=args.artifact)

    predictions, classes = [], [n_classes, n_coord_of_bbox]
    for i in range(data_num):
        result = []
        for j, dim in enumerate(classes):
            data_np = np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_{j}.raw"),
                dtype=np.float32,
            )
            result.append(torch.from_numpy(data_np).reshape(1, -1, dim))
        predictions.append(result)

    # evaluate metrics
    AP, mAP = eval_metric(
        instance=model,
        heads=predictions,
        images=[ImageList(img[0], tuple(img[0].shape[-2:])) for img in inputs],
        bboxes=bboxes,
        targets=targets,
        classes=n_classes,
    )

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"mAP": mAP}))
    else:
        print("\nMean Average Precision (mAP): %.3f" % mAP)
        print("\nAverage Precision of Classes (AP):")
        PrettyPrinter().pprint(
            {label_names[i + 1]: AP[i].item() for i in range(n_classes) if AP[i] != -1}
        )


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./retinanet",
        default="./retinanet",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help=(
            "path to the validation folder of COCO2017 dataset. "
            "e.g. --dataset PATH/TO/COCO (which contains 'val_2017' & 'annotations'), "
            "dataset could be downloaded via http://images.cocodataset.org/zips/val2017.zip & "
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ),
        type=str,
        required=True,
    )

    args = parser.parse_args()
    args.validate(args)
    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
