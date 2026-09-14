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

from executorch.backends.qualcomm.export_utils import QnnConfig, SimpleADB

from executorch.examples.qualcomm.utils import make_output_dir

from transformers import AutoTokenizer

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


def inference(args: argparse, qnn_config: QnnConfig, pte_name: str):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{pte_name}"
    pte_path = f"{args.artifact}/{pte_name}.pte"
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

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_id)
    tokenizer_json_path = tokenizer.save_pretrained(args.artifact)[-1]
    seq_len = args.max_seq_len
    runner_bin = "examples/qualcomm/oss_scripts/llama/qnn_llama_runner"

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
                "--eval_mode 0",
                "--tokenizer_path tokenizer.json",
                f"--model_path {pte_name}.pte",
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
