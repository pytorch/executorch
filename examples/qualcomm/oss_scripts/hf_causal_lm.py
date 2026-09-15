# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import getpass
import json
import logging
import os
import subprocess
from multiprocessing.connection import Client

import torch
from executorch.backends.qualcomm.export_utils import (
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)

from executorch.examples.qualcomm.oss_scripts.llm_utils.qnn_decoder_model_manager import (
    get_qnn_llm_edge_manager,
    HUGGING_FACE_REPO_IDS,
)
from executorch.examples.qualcomm.utils import make_output_dir

from transformers import AutoTokenizer

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

PTE_FILENAME = "hf_causal_lm_qnn"

# Map the HF decoder_model keys (HUGGING_FACE_REPO_IDS) to the version strings
# the shared qnn_llama_runner understands (see runner.cpp Runner()).
DECODER_MODEL_VERSION = {
    "llama3_2-1b": "llama3",
    "qwen2_5-0_5b": "qwen2_5",
    "qwen2_5-1_5b_instruct": "qwen2_5",
    "qwen2_5-0_5b_instruct": "qwen2_5",
    "qwen3-0_6b": "qwen3",
    "smollm2_135m": "smollm2_135m",
    "granite-3_3-2b": "granite",
}


def compile(args: argparse.Namespace, qnn_config: QnnConfig):  # noqa: C901

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    manager = get_qnn_llm_edge_manager(args.decoder_model, args.max_seq_len)

    fixed_point_type = {}
    if not args.use_fp16:
        kv_bits = manager.quant_recipe.get_kv_io_bit_width()
        if kv_bits == 8:
            fixed_point_type["kv_type"] = torch.uint8
        elif kv_bits == 16:
            fixed_point_type["kv_type"] = torch.uint16
        else:
            raise RuntimeError(f"unknown kv io bit width {kv_bits}")

        logits_bits = manager.quant_recipe.get_logits_output_bit_width()
        if logits_bits == 16:
            fixed_point_type["io_type"] = torch.uint16
        else:
            raise ValueError("Only support uint16 logits output for quantized hf llm.")

        model_id = HUGGING_FACE_REPO_IDS[args.decoder_model]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer_json_path = tokenizer.save_pretrained(args.artifact)[-1]

        manager.pt2e_quantize(
            fixed_point_type,
            args.calibration_tasks,
            args.calibration_limit,
            args.prompt,
            tokenizer_json_path,
            qnn_config.backend,
            qnn_config.soc_model,
        )

    manager.to_edge_transform_and_lower_to_qnn(
        qnn_config.soc_model,
        qnn_config.skip_delegate_node_ids,
        qnn_config.skip_delegate_node_ops,
    )
    if not args.use_fp16:
        logits_quant_attrs = manager.get_logits_quant_attrs()
        json.dump(
            {
                "scale": logits_quant_attrs["scale"],
                "zero_point": logits_quant_attrs["zero_point"],
            },
            open(f"{args.artifact}/{PTE_FILENAME}_quant_attrs.txt", "w"),
        )

    manager.to_executorch(args.artifact, PTE_FILENAME)


def inference(args: argparse, qnn_config: QnnConfig):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{PTE_FILENAME}"
    pte_path = f"{args.artifact}/{PTE_FILENAME}.pte"
    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/result.txt", "r") as f:
            text = f.read()
        # In tokenized-prompt mode the runner echoes the prompt-file path instead
        # of the prompt text; drop it and prepend the real prompt for readability.
        prefix = os.path.basename(tokenized_prompt_path)
        if text.startswith(prefix):
            text = text[len(prefix) :]
        outputs.append(args.prompt + text)

    model_id = HUGGING_FACE_REPO_IDS[args.decoder_model]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_json_path = tokenizer.save_pretrained(args.artifact)[-1]
    seq_len = args.max_seq_len
    runner_bin = "examples/qualcomm/oss_scripts/llama/qnn_llama_runner"
    decoder_model_version = DECODER_MODEL_VERSION[args.decoder_model]

    # The base (non-instruct) HF models were not trained on the runner's chat
    # template. Tokenize the raw prompt here (matching the Python calibration
    # path) and feed it via --tokenized_prompt so the runner skips
    # get_formatted_prompt. File format: raw little-endian uint64 tokens.
    import numpy as np

    prompt_token_ids = tokenizer(args.prompt)["input_ids"]
    tokenized_prompt_path = f"{args.artifact}/tokenized_prompt.raw"
    np.asarray(prompt_token_ids, dtype=np.uint64).tofile(tokenized_prompt_path)
    if args.enable_x86_64:
        # x86 emulator is intended for CI and not performance. Check only the first few tokens.
        seq_len = min(seq_len, 16)

        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        target = "x86_64-linux-clang"
        runner_cmd = " ".join(
            [
                f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{args.build_folder}/lib &&",
                f"{args.build_folder}/{runner_bin}",
                f"--tokenized_prompt {tokenized_prompt_path}",
                f"--decoder_model_version {decoder_model_version}",
                "--eval_mode 0",
                f"--tokenizer_path {tokenizer_json_path}",
                f"--model_path {pte_path}",
                f"--seq_len {seq_len}",
                "--temperature 0",
                f"--output_path {output_data_folder}/result.txt",
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
                "./qnn_llama_runner",
                "--tokenized_prompt tokenized_prompt.raw",
                f"--decoder_model_version {decoder_model_version}",
                "--eval_mode 0",
                "--tokenizer_path tokenizer.json",
                f"--model_path {PTE_FILENAME}.pte",
                f"--seq_len {seq_len}",
                "--temperature 0",
                "--output_path outputs/result.txt",
            ]
        )
        adb = SimpleADB(
            qnn_config=qnn_config,
            pte_path=pte_path,
            workspace=workspace,
            runner=runner_bin,
        )
        # No pregen inputs, input_list is not required
        adb.push(inputs=[], files=[tokenizer_json_path, tokenized_prompt_path])
        adb.execute(custom_runner_cmd=runner_cmd)

        adb.pull(host_output_path=args.artifact, callback=post_process)

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
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

    if args.compile_only:
        compile(args, qnn_config)
    elif args.pre_gen_pte:
        inference(args, qnn_config)
    else:
        compile(args, qnn_config)
        inference(args, qnn_config)


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example.",
        default="hf_causal_lm",
        type=str,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for LLM.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-F",
        "--use_fp16",
        help="If specified, will run in fp16 precision and discard ptq setting",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--decoder_model",
        choices=list(HUGGING_FACE_REPO_IDS.keys()),
        help=f"The Hugging Face decoder model to export. Available options are: {list(HUGGING_FACE_REPO_IDS.keys())}",
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

    try:
        args = parser.parse_args()

        if args.artifact is None:
            args.artifact = args.decoder_model
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
