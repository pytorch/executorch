# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.export._trace
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge
from torch.nn.attention import SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.executorch import convert_and_export_with_cache
from transformers.modeling_utils import PreTrainedModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hfm",
        "--hf_model_repo",
        required=True,
        default=None,
        help="a valid huggingface model repo name",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        required=False,
        default=None,
        help="output name of the exported model",
    )

    args = parser.parse_args()

    # Configs to HF model
    device = "cpu"
    dtype = torch.float32
    batch_size = 1
    max_length = 123
    cache_implementation = "static"
    attn_implementation = "sdpa"

    # Load and configure a HF model
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_repo,
        attn_implementation=attn_implementation,
        device_map=device,
        torch_dtype=dtype,
        generation_config=GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=max_length,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_length,
            },
        ),
    )
    print(f"{model.config}")
    print(f"{model.generation_config}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_repo)
    input_ids = tokenizer([""], return_tensors="pt").to(device)["input_ids"]
    cache_position = torch.tensor([0], dtype=torch.long)

    def _get_constant_methods(model: PreTrainedModel):
        return {
            "get_dtype": 5 if model.config.torch_dtype == torch.float16 else 6,
            "get_bos_id": model.config.bos_token_id,
            "get_eos_id": model.config.eos_token_id,
            "get_head_dim": model.config.hidden_size / model.config.num_attention_heads,
            "get_max_batch_size": model.generation_config.cache_config.batch_size,
            "get_max_seq_len": model.generation_config.cache_config.max_cache_len,
            "get_n_bos": 1,
            "get_n_eos": 1,
            "get_n_kv_heads": model.config.num_key_value_heads,
            "get_n_layers": model.config.num_hidden_layers,
            "get_vocab_size": model.config.vocab_size,
            "use_kv_cache": model.generation_config.use_cache,
        }

    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():

        exported_prog = convert_and_export_with_cache(model, input_ids, cache_position)
        prog = (
            to_edge(
                exported_prog,
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                ),
                constant_methods=_get_constant_methods(model),
            )
            .to_backend(XnnpackPartitioner())
            .to_executorch(ExecutorchBackendConfig(extract_delegate_segments=True))
        )
        out_name = args.output_name if args.output_name else model.config.model_type
        filename = os.path.join("./", f"{out_name}.pte")
        with open(filename, "wb") as f:
            prog.write_to_file(f)
            print(f"Saved exported program to {filename}")


if __name__ == "__main__":
    main()
