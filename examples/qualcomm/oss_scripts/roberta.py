# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import logging
import os
from multiprocessing.connection import Client

import evaluate
import numpy as np
import torch

from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_masked_language_model_dataset,
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)
from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_instance(args):
    module = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").eval()
    return module


def main(args):
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    os.makedirs(args.artifact, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    data_size = 100
    if args.ci:
        random_ids = torch.randint(low=0, high=100, size=(1, 100), dtype=torch.int32)
        attention_mask = torch.ones((1, 100), dtype=torch.float32)
        inputs = [
            (
                random_ids,
                attention_mask,
            )
        ]
        logging.warning(
            "This option is for CI to verify the export flow. It uses random input and will result in poor accuracy."
        )
    else:
        inputs, targets = get_masked_language_model_dataset(
            args.dataset, tokenizer, data_size
        )

    # Get the Roberta model.
    model = get_instance(args)
    pte_filename = "roberta_qnn_q16"

    # lower to QNN
    passes_job = get_capture_program_passes()
    build_executorch_binary(
        model,
        inputs[0],
        args.model,
        f"{args.artifact}/{pte_filename}",
        dataset=inputs,
        skip_node_id_set=skip_node_id_set,
        skip_node_op_set=skip_node_op_set,
        quant_dtype=QuantDtype.use_16a8w,
        passes_job=passes_job,
        shared_buffer=args.shared_buffer,
    )

    if args.compile_only:
        return

    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{pte_filename}"
    pte_path = f"{args.artifact}/{pte_filename}.pte"

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=pte_path,
        workspace=workspace,
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
        target=args.target,
    )
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    # demo
    mask_token = tokenizer.mask_token
    text = f"Hello I'm a {mask_token} model."
    sample_input = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=inputs[0][0].shape[1],
    )
    sample_input["input_ids"] = sample_input["input_ids"].to(torch.int32)
    sample_input["attention_mask"] = sample_input["attention_mask"].to(torch.float32)
    sample_input = tuple(sample_input.values())
    golden = model(*sample_input)[0]
    adb.push(inputs=[sample_input])
    adb.execute()
    adb.pull(output_path=args.artifact)

    print(f"input: {tokenizer.batch_decode(sample_input[0])}")
    print(f"golden output: {tokenizer.batch_decode(golden.argmax(axis=2))}")
    predictions = np.fromfile(
        os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
    ).reshape([1, inputs[0][0].shape[1], -1])
    print(f"QNN output: {tokenizer.batch_decode(predictions.argmax(axis=2))}")

    # accuracy analysis
    adb.push(inputs=inputs)
    adb.execute()
    adb.pull(output_path=args.artifact)
    goldens, predictions = [], []
    for i in range(len(inputs)):
        indice = [i for i, x in enumerate(targets[i]) if x != -100]
        goldens.extend(targets[i][indice].tolist())
        prediction = (
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
            .reshape([1, inputs[0][0].shape[1], -1])
            .argmax(axis=-1)
        )
        predictions.extend(prediction[0, indice].tolist())
    metric = evaluate.load("accuracy")
    results = metric.compute(predictions=predictions, references=goldens)
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"accuracy": results["accuracy"]}))
    else:
        print(f"accuracy: {results['accuracy']}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./Roberta_qnn",
        default="./roberta",
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
        type=str,
        required=False,
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
