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

from executorch.backends.qualcomm.quantizer.custom_annotation import annotate_eurobert
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)

from executorch.examples.qualcomm.utils import (
    build_executorch_binary,
    get_backend_type,
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
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")

    assert (
        transformers.__version__ >= TRANSFORMERS_VERSION
    ), f"Please ensure transformers version >= {TRANSFORMERS_VERSION}, current version is {transformers.__version__}"

    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    os.makedirs(args.artifact, exist_ok=True)

    model_name = "EuroBERT/EuroBERT-210m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        rope_scaling={
            "rope_type": "linear",
            "factor": 1.0,
            "rope_theta": 250000,
        },
    )
    module = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, config=config
    ).eval()

    data_size = 100
    if args.ci:
        random_ids = torch.randint(low=0, high=100, size=(1, 100), dtype=torch.int32)
        attention_mask = torch.zeros((1, 100), dtype=torch.float32)
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

    pte_filename = "eurobert_qnn"

    # Skip lowering/compilation if using pre-generated PTE
    if not args.pre_gen_pte:
        # lower to QNN
        def get_custom_quantizer():
            quantizer = make_quantizer(
                quant_dtype=QuantDtype.use_16a16w,
                eps=2**-20,
            )
            quantizer.add_custom_quant_annotations((annotate_eurobert,))
            return quantizer

        backend = get_backend_type(args.backend)
        quantizer = {
            QnnExecuTorchBackendType.kGpuBackend: None,
            QnnExecuTorchBackendType.kHtpBackend: get_custom_quantizer(),
        }[backend]
        with torch.no_grad():
            build_executorch_binary(
                module,
                inputs[0],
                args.model,
                f"{args.artifact}/{pte_filename}",
                dataset=inputs,
                skip_node_id_set=skip_node_id_set,
                skip_node_op_set=skip_node_op_set,
                custom_quantizer=quantizer,
                backend=backend,
                shared_buffer=args.shared_buffer,
                online_prepare=args.online_prepare,
            )

    if args.compile_only:
        return

    pte_path = (
        f"{args.pre_gen_pte}/{pte_filename}.pte"
        if args.pre_gen_pte
        else f"{args.artifact}/{pte_filename}.pte"
    )
    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=f"{args.build_folder}",
        pte_path=pte_path,
        workspace=f"/data/local/tmp/executorch/{pte_filename}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
        target=args.target,
        backend=backend,
    )
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    # accuracy analysis
    adb.push(inputs=inputs)
    adb.execute()
    adb.pull(host_output_path=args.artifact)
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
