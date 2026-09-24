# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

from executorch.backends.qualcomm.export_utils import (
    QnnConfig,
    setup_common_args_and_variables,
)
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
from executorch.examples.qualcomm.oss_scripts.hf_causal_lm import inference
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.exporters import ExecutorchExporter, ExecutorchQnnLlmConfig


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

PTE_FILENAME = "hf_causal_lm_qnn"


def build_model(model_id: str, max_seq_len: int):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="float32",
        attn_implementation="eager",
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            max_length=max_seq_len,
            cache_config={"batch_size": 1, "max_cache_len": max_seq_len},
        ),
    ).eval()


def main(args):
    os.makedirs(args.artifact, exist_ok=True)
    logging.info(f"Loading {args.decoder_model_id}...")
    model = build_model(args.decoder_model_id, args.max_seq_len)

    if not args.pre_gen_pte:
        et_program_manager = ExecutorchExporter().export(
            model,
            None,
            ExecutorchQnnLlmConfig(
                model_id=args.decoder_model_id,
                max_seq_len=args.max_seq_len,
                soc_model=args.soc_model,
                use_fp16=args.use_fp16,
                calibration_dataset=[args.prompt],
                quantizer=None if args.use_fp16 else QnnQuantizer(),
                backend="qnn",
                alloc_graph_input=False,
                alloc_graph_output=False,
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

    args = parser.parse_args()
    main(args)
