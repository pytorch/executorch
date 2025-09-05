# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import json
import logging
import os
import subprocess
from multiprocessing.connection import Client

import torch

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype

from executorch.examples.qualcomm.oss_scripts.llm_utils.qnn_decoder_model_manager import (
    get_qnn_llm_edge_manager,
    HUGGING_FACE_REPO_IDS,
)

from executorch.examples.qualcomm.utils import (
    make_output_dir,
    parse_skip_delegation_node,
    setup_common_args_and_variables,
    SimpleADB,
)

from transformers import AutoTokenizer

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

PTE_FILENAME = "qwen_qnn_q16"


def compile(args):  # noqa: C901
    skip_node_id_set, skip_node_op_set = parse_skip_delegation_node(args)

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    manager = get_qnn_llm_edge_manager(
        args.decoder_model, args.max_seq_len, args.enable_spinquant_r3
    )

    fixed_point_type = {}
    if args.ptq:
        if args.ptq == "8a8w":
            fixed_point_type["io_type"] = torch.uint8
            fixed_point_type["kv_type"] = torch.uint8
        elif args.ptq in (
            "16a8w",
            "16a4w",
            "16a4w_block",
            "16a16w",
        ):
            fixed_point_type["io_type"] = torch.uint16
            fixed_point_type["kv_type"] = torch.uint16
        else:
            raise ValueError(
                f"No support for quant type {args.ptq}. Support 8a8w, 16a8w, 16a4w and 16a4w_block."
            )
        quant_dtype = getattr(QuantDtype, f"use_{args.ptq}")
        model_id = HUGGING_FACE_REPO_IDS[args.decoder_model]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer_json_path = tokenizer.save_pretrained(args.artifact)[-1]

        manager.pt2e_quantize(
            quant_dtype,
            fixed_point_type,
            args.calibration_tasks,
            args.calibration_limit,
            args.prompt,
            tokenizer_json_path,
        )

    manager.to_edge_transform_and_lower_to_qnn(
        args.model, skip_node_id_set, skip_node_op_set
    )
    if args.ptq:
        logits_quant_attrs = manager.get_logits_quant_attrs()
        json.dump(
            {
                "scale": logits_quant_attrs["scale"],
                "zero_point": logits_quant_attrs["zero_point"],
            },
            open(f"{args.artifact}/{PTE_FILENAME}_quant_attrs.txt", "w"),
        )

    manager.to_executorch(args.artifact, PTE_FILENAME)


def inference(args):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{PTE_FILENAME}"
    pte_path = f"{args.artifact}/{PTE_FILENAME}.pte"
    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/result.txt", "r") as f:
            outputs.append(f.read())

    model_id = HUGGING_FACE_REPO_IDS[args.decoder_model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_json_path = tokenizer.save_pretrained(args.artifact)[-1]
    seq_len = args.max_seq_len
    if args.enable_x86_64:
        # x86 emulator is intended for CI and not performance. Check only the first few tokens.
        seq_len = min(seq_len, 16)

        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        target = "x86_64-linux-clang"
        runner_cmd = " ".join(
            [
                f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{args.build_folder}/lib &&",
                f"{args.build_folder}/examples/models/llama/llama_main",
                f'--prompt "{args.prompt}"',
                f"--tokenizer_path {tokenizer_json_path}",
                f"--model_path {pte_path}",
                f"--seq_len {seq_len}",
                "--temperature 0",
                f" > {output_data_folder}/result.txt",
            ]
        )
        subprocess.run(
            runner_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
        )
        post_process()
    else:
        runner_cmd = " ".join(
            [
                f"cd {workspace} &&",
                "./llama_main",
                f'--prompt "{args.prompt}"',
                "--tokenizer_path tokenizer.json",
                f"--model_path {PTE_FILENAME}.pte",
                f"--seq_len {seq_len}",
                "--temperature 0",
                " > outputs/result.txt",
            ]
        )
        adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path=f"{args.build_folder}",
            pte_path=pte_path,
            workspace=workspace,
            device_id=args.device,
            host_id=args.host,
            soc_model=args.model,
            runner="examples/models/llama/llama_main",
        )
        # No pregen inputs, input_list is not required
        adb.push(inputs=[], input_list="", files=[tokenizer_json_path])
        adb.execute(custom_runner_cmd=runner_cmd)

        adb.pull(output_path=args.artifact, callback=post_process)

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "result": outputs,
                    }
                )
            )
    else:
        for idx, output in enumerate(outputs):
            logging.info(f"Results[{idx}]:\n{output}")


def main(args):
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")

    if args.compile_only:
        compile(args)
    elif args.pre_gen_pte:
        inference(args)
    else:
        compile(args)
        inference(args)


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example.",
        default="qwen2_5",
        type=str,
    )

    parser.add_argument(
        "-P",
        "--ptq",
        choices=["8a8w", "16a8w", "16a4w", "16a4w_block"],
        help="If specified, will do PTQ quantization.",
        type=str,
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="Run the pre-generated Qwen in the given directory.",
        type=str,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for Qwen.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--decoder_model",
        choices=["qwen2.5_0.5B", "qwen2.5_0.5B_instruct", "qwen2.5_1.5B_instruct"],
        help="The Qwen model to export. Current available options are: [qwen2.5_0.5B, qwen2.5_0.5B_instruct, qwen2.5_1.5B_instruct]",
        required=True,
    )

    parser.add_argument(
        "--max_seq_len",
        help="This refers to maximum number of tokens that the model can process & consider at once to generate predictions/responses.",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--calibration_tasks",
        nargs="+",
        type=str,
        default=None,
        help="Tasks for GPTQ calibration from lm_eval",
    )
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=None,
        help="number of samples used for calibration from lm_eval",
    )
    parser.add_argument(
        "--enable_spinquant_r3",
        action="store_true",
        help="Specify to enable spin quant R3",
    )

    try:
        args = parser.parse_args()
        args.validate(args)
        if args.artifact is None:
            args.artifact = args.decoder_model
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
