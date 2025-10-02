# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from multiprocessing.connection import Client

import evaluate
import numpy as np
import torch
import transformers
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_capture_program_passes,
)

from executorch.backends.qualcomm.quantizer.custom_annotation import annotate_eurobert
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_masked_language_model_dataset,
    make_output_dir,
    make_quantizer,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

TRANSFORMERS_VERSION = "4.48.0"


def main(args):
    assert (
        transformers.__version__ >= TRANSFORMERS_VERSION
    ), f"Please ensure transformers version >= {TRANSFORMERS_VERSION}, current version is {transformers.__version__}"

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    os.makedirs(args.artifact, exist_ok=True)

    module_id = "EuroBERT/EuroBERT-210m"
    tokenizer = AutoTokenizer.from_pretrained(module_id)
    model = AutoModelForMaskedLM.from_pretrained(
        module_id, trust_remote_code=True
    ).eval()
    config = AutoConfig.from_pretrained(module_id, trust_remote_code=True)

    def replace_rms_norm_with_native_rms_norm(module: torch.nn.Module):
        for name, child in module.named_children():
            if child._get_name() == "EuroBertRMSNorm":
                rms_norm = torch.nn.RMSNorm(
                    [config.hidden_size], eps=child.variance_epsilon
                )
                rms_norm.weight = child.weight
                setattr(
                    module,
                    name,
                    rms_norm,
                )
            else:
                replace_rms_norm_with_native_rms_norm(child)
        return module

    replace_rms_norm_with_native_rms_norm(model)

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

    pte_filename = "eurobert_qnn_q16"

    # lower to QNN
    passes_job = get_capture_program_passes()
    quantizer = make_quantizer(
        quant_dtype=QuantDtype.use_16a16w,
    )
    quantizer.add_custom_quant_annotations((annotate_eurobert,))
    with torch.no_grad():
        build_executorch_binary(
            model,
            inputs[0],
            args.model,
            f"{args.artifact}/{pte_filename}",
            dataset=inputs,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            custom_quantizer=quantizer,
            passes_job=passes_job,
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
    )
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    # accuracy analysis
    adb.push(inputs=inputs)
    adb.execute()
    adb.pull(output_path=args.artifact)
    goldens, predictions = [], []
    for i in range(len(inputs)):
        indices = [i for i, x in enumerate(targets[i]) if x != -100]
        goldens.extend(targets[i][indices].tolist())

        prediction = (
            np.fromfile(
                os.path.join(output_data_folder, f"output_{i}_0.raw"), dtype=np.float32
            )
            .reshape([1, inputs[0][0].shape[1], -1])
            .argmax(axis=-1)
        )
        predictions.extend(prediction[0, indices].tolist())
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
        help="path for storing generated artifacts and output by this example. Default ./eurobert",
        default="./eurobert",
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
