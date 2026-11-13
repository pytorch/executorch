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
    Device,
    get_backend_type,
    make_quantizer,
    QnnConfig,
    setup_common_args_and_variables,
)
from executorch.backends.qualcomm.hf_transformers.api import get_quant_recipe
from executorch.examples.qualcomm.utils import make_output_dir
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.exporters import ExecutorchExporter, ExecutorchQnnConfig
from transformers.exporters.quantizers import PT2EQuantizer

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

PTE_FILENAME = "hf_causal_lm_qnn"


def build_model(model_id: str, max_seq_len: int):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="float32",
        attn_implementation="eager",
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_cache_len=max_seq_len,
        ),
    ).eval()
    return model


def inference(args: argparse, qnn_config: QnnConfig, pte_name: str):
    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/{pte_name}"
    pte_path = f"{args.artifact}/{pte_name}.pte"
    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/result.txt", "r") as f:
            outputs.append(f.read())

    tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_id)
    tokenizer_json_path = tokenizer.save_pretrained(args.artifact)[-1]
    seq_len = args.max_seq_len
    runner_bin = "examples/models/llama/llama_main"

    if args.enable_x86_64:
        # x86 emulator is intended for CI and not performance. Check only the first few tokens.
        seq_len = min(seq_len, 16)

        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        target = "x86_64-linux-clang"
        runner_cmd = " ".join(
            [
                f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{args.build_folder}/lib &&",
                f"{args.build_folder}/{runner_bin}",
                f'--prompt "{args.prompt}"',
                f"--tokenizer_path {tokenizer_json_path}",
                f"--model_path {pte_path}",
                f"--seq_len {seq_len}",
                "--temperature 0",
                f"> {output_data_folder}/result.txt",
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
                f"--model_path {pte_name}.pte",
                f"--seq_len {seq_len}",
                "--temperature 0",
                "> outputs/result.txt",
            ]
        )
        device = Device(
            qnn_config=qnn_config,
            pte_path=pte_path,
            workspace=workspace,
            runner=runner_bin,
        )
        # No pregen inputs, input_list is not required
        device.push(inputs=[], files=[tokenizer_json_path])
        device.execute(custom_runner_cmd=runner_cmd)

        device.pull(host_output_path=args.artifact, callback=post_process)

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
    os.makedirs(args.artifact, exist_ok=True)
    logging.info(f"Loading {args.decoder_model_id}...")
    model = build_model(args.decoder_model_id, args.max_seq_len)
    soc_list = [args.soc_model]  # , "SC8380XP", "QCS9100"
    if not args.pre_gen_pte:
        tokenizer = AutoTokenizer.from_pretrained(args.decoder_model_id)
        token_list = tokenizer.encode(args.prompt, bos=True, eos=False)
        calibration_dataset = [
            {
                "input_ids": torch.tensor([[token]]),
                "cache_position": torch.tensor([idx]),
            }
            for idx, token in enumerate(token_list)
        ]

        quantizer = None
        if not args.use_fp16:
            quantizer = make_quantizer(
                backend=get_backend_type(args.backend), soc_model=soc_list
            )
            quant_recipe = get_quant_recipe(args.decoder_model_id)
            quantizer.set_recipe(quant_recipe.recipe)
            quantizer.set_convert_linear_to_conv2d(True)
            quantizer = PT2EQuantizer(
                quantizer, calibration_dataset=calibration_dataset
            )
        et_program_manager = ExecutorchExporter().export(
            model=model,
            sample_inputs=calibration_dataset[0],
            config=ExecutorchQnnConfig(
                soc_models=soc_list,
                quantizer=quantizer,
                backend="qnn",
                alloc_graph_input=False,
            ),
        )

        pte_path = f"{args.artifact}/{PTE_FILENAME}.pte"
        with open(pte_path, "wb") as f:
            et_program_manager.write_to_file(f)
        logging.info(f"wrote {pte_path} ({os.path.getsize(pte_path)} bytes)")

    if not args.compile_only:
        inference(
            args,
            QnnConfig.load_config(args.config_file if args.config_file else args),
            PTE_FILENAME,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = setup_common_args_and_variables()
    parser.add_argument("-a", "--artifact", default="hf_exporter")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument(
        "--use_fp16", action="store_true", help="Skip PT2E quantization."
    )
    parser.add_argument(
        "--decoder_model_id",
        help="The Hugging Face ID to export, e.g., 'NousResearch/Llama-3.2-1B'",
        default="NousResearch/Llama-3.2-1B",
    )

    try:
        args = parser.parse_args()
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
