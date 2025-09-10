# Copyright (c) MediaTek Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from aot_utils.oss_utils.utils import build_executorch_binary, get_masked_language_model_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer


def main(args):
    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)
    data_size = 100

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    inputs, targets = get_masked_language_model_dataset(
        args.dataset, tokenizer, data_size
    )

    # build pte
    module = AutoModelForMaskedLM.from_pretrained(
        "distilbert/distilbert-base-uncased"
    ).eval()
    pte_filename = "distilbert_mtk"

    build_executorch_binary(
        module,
        inputs[0],
        f"{args.artifact}/{pte_filename}",
        inputs,
        skip_op_name={"aten_embedding_default", "aten_where_self"},
    )

    # save data to inference on device
    input_list_file = f"{args.artifact}/input_list.txt"
    with open(input_list_file, "w") as f:
        for i in range(len(inputs)):
            f.write(f"input_{i}_0.bin input_{i}_1.bin\n")
    for idx, data in enumerate(inputs):
        for i, d in enumerate(data):
            file_name = f"{args.artifact}/input_{idx}_{i}.bin"
            d.detach().numpy().tofile(file_name)
    for idx, data in enumerate(targets):
        file_name = f"{args.artifact}/golden_{idx}_0.bin"
        data.detach().numpy().tofile(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./distilbert",
        default="./distilbert",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help=(
            "path to the validation text. "
            "e.g. --dataset wikisent2.txt "
            "for https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences"
        ),
        default="wikisent2.txt",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    main(args)
