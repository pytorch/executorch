# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

import numpy as np
import piq
import torch


def check_data(target_f, predict_f):
    target_files = os.listdir(target_f)
    predict_files = os.listdir(predict_f)
    if len(target_files) != len(predict_files):
        raise RuntimeError(
            "Data number in target folder and prediction folder must be same"
        )

    predict_set = set(predict_files)
    for f in target_files:
        # target file naming rule is golden_sampleId_outId.bin
        # predict file naming rule is output_sampleId_outId.bin
        pred_name = f.replace("golden", "output")
        try:
            predict_set.remove(pred_name)
        except KeyError:
            raise RuntimeError(f"Cannot find {pred_name} in {predict_f}")

    if predict_set:
        target_name = next(predict_set).replace("output", "golden")
        raise RuntimeError(f"Cannot find {target_name} in {target_f}")


def eval_topk(target_f, predict_f):
    def solve(prob, target, k):
        _, indices = torch.topk(prob, k=k, sorted=True)
        golden = torch.reshape(target, [-1, 1])
        correct = golden == indices
        if torch.any(correct):
            return 1
        else:
            return 0

    target_files = os.listdir(target_f)

    cnt10 = 0
    cnt50 = 0
    for target_name in target_files:
        pred_name = target_name.replace("golden", "output")

        pred_npy = np.fromfile(os.path.join(predict_f, pred_name), dtype=np.float32)
        target_npy = np.fromfile(os.path.join(target_f, target_name), dtype=np.int64)[0]
        cnt10 += solve(torch.from_numpy(pred_npy), torch.from_numpy(target_npy), 10)
        cnt50 += solve(torch.from_numpy(pred_npy), torch.from_numpy(target_npy), 50)

    print("Top10 acc:", cnt10 * 100.0 / len(target_files))
    print("Top50 acc:", cnt50 * 100.0 / len(target_files))


def eval_piq(target_f, predict_f):
    target_files = os.listdir(target_f)

    psnr_list = []
    ssim_list = []
    for target_name in target_files:
        pred_name = target_name.replace("golden", "output")
        hr = np.fromfile(os.path.join(target_f, target_name), dtype=np.float32)
        hr = hr.reshape((1, 448, 448, 3))
        hr = np.moveaxis(hr, 3, 1)
        hr = torch.from_numpy(hr)

        sr = np.fromfile(os.path.join(predict_f, pred_name), dtype=np.float32)
        sr = sr.reshape((1, 448, 448, 3))
        sr = np.moveaxis(sr, 3, 1)
        sr = torch.from_numpy(sr).clamp(0, 1)

        psnr_list.append(piq.psnr(hr, sr))
        ssim_list.append(piq.ssim(hr, sr))

    avg_psnr = sum(psnr_list).item() / len(psnr_list)
    avg_ssim = sum(ssim_list).item() / len(ssim_list)

    print(f"Avg of PSNR is: {avg_psnr}")
    print(f"Avg of SSIM is: {avg_ssim}")


def eval_segmentation(target_f, predict_f):
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

    target_files = os.listdir(target_f)

    def make_confusion(goldens, predictions, num_classes):
        def histogram(golden, predict):
            mask = golden < num_classes
            hist = np.bincount(
                num_classes * golden[mask].astype(int) + predict[mask],
                minlength=num_classes**2,
            ).reshape(num_classes, num_classes)
            return hist

        confusion = np.zeros((num_classes, num_classes))
        for g, p in zip(goldens, predictions):
            confusion += histogram(g.flatten(), p.flatten())

        return confusion

    pred_list = []
    target_list = []
    for target_name in target_files:
        pred_name = target_name.replace("golden", "output")
        target_npy = np.fromfile(os.path.join(target_f, target_name), dtype=np.uint8)
        target_npy = target_npy.reshape((224, 224))
        target_list.append(target_npy)

        pred_npy = np.fromfile(os.path.join(predict_f, pred_name), dtype=np.float32)
        pred_npy = pred_npy.reshape((224, 224, len(classes)))
        pred_npy = pred_npy.argmax(2).astype(np.uint8)
        pred_list.append(pred_npy)

    eps = 1e-6
    confusion = make_confusion(target_list, pred_list, len(classes))

    pa = np.diag(confusion).sum() / (confusion.sum() + eps)
    mpa = np.mean(np.diag(confusion) / (confusion.sum(axis=1) + eps))
    iou = np.diag(confusion) / (
        confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion) + eps
    )
    miou = np.mean(iou)
    cls_iou = dict(zip(classes, iou))

    print(f"PA   : {pa}")
    print(f"MPA  : {mpa}")
    print(f"MIoU : {miou}")
    print(f"CIoU : \n{json.dumps(cls_iou, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_f",
        help="folder of target data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--out_f",
        help="folder of model prediction data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--eval_type",
        help="Choose eval type from: topk, piq, segmentation",
        type=str,
        choices=["topk", "piq", "segmentation"],
        required=True,
    )

    args = parser.parse_args()

    check_data(args.target_f, args.out_f)

    if args.eval_type == "topk":
        eval_topk(args.target_f, args.out_f)
    elif args.eval_type == "piq":
        eval_piq(args.target_f, args.out_f)
    elif args.eval_type == "segmentation":
        eval_segmentation(args.target_f, args.out_f)
