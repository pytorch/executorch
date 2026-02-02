# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from contextlib import contextmanager

from multiprocessing.connection import Client

import numpy as np

import torch
import torch.nn.functional as F
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.models.torchvision_vit.model import TorchVisionViTModel
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_imagenet_dataset,
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
    topk_accuracy,
)


# Copied from torch/nn/functional.py
# QNN does not have 5D permute optimization. Fuse to a single 4D optimization
# Changed unsqueeze(0).transpose(0, -2).squeeze(-2) to permute(2, 0, 1, 3)
def _in_projection_packed_custom(q, k, v, w, b=None) -> list[torch.Tensor]:
    from torch.nn.functional import linear

    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            proj = linear(q, w, b)
            # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
            proj = proj.unflatten(-1, (3, E)).permute(2, 0, 1, 3).contiguous()
            # pyrefly: ignore  # bad-return
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = linear(q, w_q, b_q)
            kv_proj = linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is deliberate for better memory coalescing and keeping same order as chunk()
            kv_proj = kv_proj.unflatten(-1, (2, E)).permute(2, 0, 1, 3).contiguous()
            # pyrefly: ignore  # bad-return
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        # pyrefly: ignore  # bad-return
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# Context manager to patch temporarily, so it won't affect other users using F._in_projection_packed
@contextmanager
def PermuteInProjectionPacked():
    # Save the original function so it can be restored later
    _original_in_projection_packed = F._in_projection_packed
    F._in_projection_packed = _in_projection_packed_custom
    try:
        yield
    finally:
        F._in_projection_packed = _original_in_projection_packed


def main(args):
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
            image_shape=(256, 256),
            crop_size=224,
        )

    pte_filename = "vit_qnn_q8"
    instance = TorchVisionViTModel().get_eager_model().eval()

    with PermuteInProjectionPacked():
        build_executorch_binary(
            instance,
            inputs[0],
            args.model,
            f"{args.artifact}/{pte_filename}",
            inputs,
            quant_dtype=QuantDtype.use_8a8w,
            shared_buffer=args.shared_buffer,
        )

    if args.compile_only:
        return

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
        "Default ./torchvision_vit",
        default="./torchvision_vit",
        type=str,
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
