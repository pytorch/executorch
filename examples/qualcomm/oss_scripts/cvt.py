# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import types
from multiprocessing.connection import Client

import numpy as np

import torch
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_imagenet_dataset,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
    topk_accuracy,
)
from transformers import AutoModelForImageClassification
from transformers.models.cvt.modeling_cvt import CvtSelfAttention


# Copy from transformers/models/cvt/modeling_cvt.py in transformers 4.47.1
# torch.einsum("bhlk,bhtk->bhlt", [query, key]) will result in prepare failed due to 5D tensor with decompose_einsum.
# TODO: once HTP fixed, this workaround can be removed
def attention_forward_without_einsum(self, hidden_state, height, width):
    if self.with_cls_token:
        cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
    batch_size, hidden_size, num_channels = hidden_state.shape
    # rearrange "b (h w) c -> b c h w"
    hidden_state = hidden_state.permute(0, 2, 1).view(
        batch_size, num_channels, height, width
    )

    key = self.convolution_projection_key(hidden_state)
    query = self.convolution_projection_query(hidden_state)
    value = self.convolution_projection_value(hidden_state)

    if self.with_cls_token:
        query = torch.cat((cls_token, query), dim=1)
        key = torch.cat((cls_token, key), dim=1)
        value = torch.cat((cls_token, value), dim=1)

    head_dim = self.embed_dim // self.num_heads

    query = self.rearrange_for_multi_head_attention(self.projection_query(query))
    key = self.rearrange_for_multi_head_attention(self.projection_key(key))
    value = self.rearrange_for_multi_head_attention(self.projection_value(value))
    # ====================Qualcomm Changed=================================
    attention_score = query @ key.transpose(-1, -2)
    attention_score = attention_score * self.scale
    # attention_score = torch.einsum("bhlk,bhtk->bhlt", [query, key]) * self.scale
    # =====================================================================
    attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
    attention_probs = self.dropout(attention_probs)
    # ====================Qualcomm Changed=================================
    context = attention_probs @ value
    # context = torch.einsum("bhlt,bhtv->bhlv", [attention_probs, value])
    # =====================================================================
    # rearrange"b h t d -> b t (h d)"
    _, _, hidden_size, _ = context.shape
    context = (
        context.permute(0, 2, 1, 3)
        .contiguous()
        .view(batch_size, hidden_size, self.num_heads * head_dim)
    )
    return context


def _replace_attention(
    module: torch.nn.Module,
):
    for _, child in module.named_children():
        if isinstance(child, CvtSelfAttention):
            child.forward = types.MethodType(  # pyre-ignore
                attention_forward_without_einsum, child
            )
        else:
            _replace_attention(child)
    return module


def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    if not args.compile_only and args.device is None:
        raise RuntimeError(
            "device serial is required if not compile only. "
            "Please specify a device serial by -s/--device argument."
        )

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

    module = (
        AutoModelForImageClassification.from_pretrained("microsoft/cvt-13")
        .eval()
        .to("cpu")
    )
    # Fix prepare failed due to einsum
    module = _replace_attention(module)
    pte_filename = "cvt_qnn_q8"
    build_executorch_binary(
        module.eval(),
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
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
    )
    adb.push(inputs=inputs)
    adb.execute()

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    adb.pull(output_path=args.artifact)

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
        help="path for storing generated artifacts by this example. " "Default ./cvt",
        default="./cvt",
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
