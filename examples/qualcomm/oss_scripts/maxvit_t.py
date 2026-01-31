# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import json
import logging
import os

from multiprocessing.connection import Client

import numpy as np

import torch
import torch.nn.functional as F
import torchvision

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_backend_type,
    get_imagenet_dataset,
    make_output_dir,
    make_quantizer,
    setup_common_args_and_variables,
    SimpleADB,
    topk_accuracy,
)
from torchvision.models.maxvit import (
    PartitionAttentionLayer,
    RelativePositionalMultiHeadAttention,
)


class WindowPartition(torch.nn.Module):
    """
    Partition the input tensor into non-overlapping windows.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, p: int) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
            p (int): Number of partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, H/P, W/P, P*P, C].
        """
        B, C, H, W = x.shape
        P = p
        # chunk up H and W dimensions
        x = x.reshape(B * C, H // P, P, W // P, P)
        x = x.permute(0, 1, 3, 2, 4)
        # colapse P * P dimension
        x = x.reshape(B, C, (H // P) * (W // P), P * P)
        return x.permute(0, 2, 3, 1)


class WindowDepartition(torch.nn.Module):
    """
    Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W].
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: torch.Tensor, p: int, h_partitions: int, w_partitions: int
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, (H/P * W/P), P*P, C].
            p (int): Number of partitions.
            h_partitions (int): Number of vertical partitions.
            w_partitions (int): Number of horizontal partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions
        x = x.permute(0, 3, 1, 2)
        # split P * P dimension into 2 P tile dimensionsa
        x = x.reshape(B * C, HP, WP, P, P)
        # permute into B * C, HP, P, WP, P
        x = x.permute(0, 1, 3, 2, 4)
        # reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)
        return x


def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x (Tensor): Input tensor with expected layout of [B, G, P, D].
    Returns:
        Tensor: Output tensor with expected layout of [B, G, P, D].
    """
    B, G, P, D = x.shape
    H, DH = self.n_heads, self.head_dim

    qkv = self.to_qkv(x)
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    q = q.reshape(B * G, P, H, DH).permute(0, 2, 1, 3)
    k = k.reshape(B * G, P, H, DH).permute(0, 2, 1, 3)
    v = v.reshape(B * G, P, H, DH).permute(0, 2, 1, 3)

    k = k * self.scale_factor
    dot_prod = torch.einsum("B H I D, B H J D -> B H I J", q, k)
    pos_bias = self.get_relative_positional_bias()

    dot_prod = F.softmax(dot_prod + pos_bias, dim=-1)

    out = torch.einsum("B H I J, B H J D -> B H I D", dot_prod, v)
    out = out.permute(0, 2, 1, 3).reshape(B, G, P, D)

    out = self.merge(out)
    return out


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

    pte_filename = "maxvit_t_qnn"
    instance = torchvision.models.maxvit_t(weights="IMAGENET1K_V1").eval()
    for block in instance.blocks:
        for layer in block.layers:
            for sub_layer in layer.layers:
                if isinstance(sub_layer, PartitionAttentionLayer):
                    sub_layer.partition_op = WindowPartition()
                    sub_layer.departition_op = WindowDepartition()
                    for attn_sub_layer in sub_layer.attn_layer:
                        if isinstance(
                            attn_sub_layer, RelativePositionalMultiHeadAttention
                        ):
                            attn_sub_layer.forward = functools.partial(
                                forward, attn_sub_layer
                            )

    backend = get_backend_type(args.backend)
    quantizer = {
        QnnExecuTorchBackendType.kGpuBackend: None,
        QnnExecuTorchBackendType.kHtpBackend: make_quantizer(
            quant_dtype=QuantDtype.use_8a8w,
            per_channel_linear=True,
        ),
    }[backend]
    build_executorch_binary(
        instance,
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        custom_quantizer=quantizer,
        backend=backend,
        shared_buffer=args.shared_buffer,
        online_prepare=args.online_prepare,
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
        backend=backend,
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
        "Default ./maxvit_t",
        default="./maxvit_t",
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
