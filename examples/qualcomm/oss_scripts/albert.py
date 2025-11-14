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
from transformers import AlbertConfig, AutoModelForMaskedLM, AutoTokenizer


def main(args):
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    os.makedirs(args.artifact, exist_ok=True)
    data_size = 100

    model_name = "albert/albert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, hidden_act="gelu")

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

    config = AlbertConfig.from_pretrained(model_name)
    config.hidden_act = "gelu"
    module = AutoModelForMaskedLM.from_pretrained(model_name, config=config).eval()
    pte_filename = "albert_qnn_q16"

    # Skip lowering/compilation if using pre-generated PTE
    if not args.pre_gen_pte:
        # lower to QNN
        passes_job = get_capture_program_passes()
        build_executorch_binary(
            module,
            inputs[0],
            args.model,
            f"{args.artifact}/{pte_filename}",
            dataset=inputs,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            quant_dtype=QuantDtype.use_16a16w,
            passes_job=passes_job,
            shared_buffer=args.shared_buffer,
        )

    if args.compile_only:
        return

    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{pte_filename}"
    pte_path = (
        f"{args.pre_gen_pte}/{pte_filename}.pte"
        if args.pre_gen_pte
        else f"{args.artifact}/{pte_filename}.pte"
    )

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

    # accuracy analysis
    adb.push(inputs=inputs)
    adb.execute()
    adb.pull(output_path=args.artifact)
    # since the original nn.Module could not perform well on this task either
    # we only measure the relative accuracy here
    goldens, predictions, nominal_predictions = [], [], []
    for i in range(len(inputs)):
        indices = [i for i, x in enumerate(targets[i]) if x != -100]
        goldens.extend(targets[i][indices].tolist())
        nominal_prediction = module(*inputs[i])
        nominal_predictions.extend(
            nominal_prediction.logits.argmax(axis=-1)[0, indices].tolist()
        )
        prediction = (
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
            .reshape([1, inputs[0][0].shape[1], -1])
            .argmax(axis=-1)
        )
        predictions.extend(prediction[0, indices].tolist())

    metric = evaluate.load("accuracy")
    nominal_results = metric.compute(
        predictions=nominal_predictions, references=goldens
    )
    device_results = metric.compute(predictions=predictions, references=goldens)
    result = device_results["accuracy"] / nominal_results["accuracy"]
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"accuracy": result}))
    else:
        print(f"accuracy: {device_results}")
        print(f"accuracy with nn.Module as golden: {result}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./albert",
        default="./albert",
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
