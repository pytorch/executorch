import argparse
import logging
import os
import sys
import traceback
import copy
import torch
from examples.portable.utils import export_to_edge
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from torch.nn.attention import SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hfm",
        "--hf_model_repo",
        required=False,
        default=None,
        help="a valid huggingface model repo name",
    )
    parser.add_argument("--compile", required=False, action="store_true", help="run HF model in eager with torch.compile")
    parser.add_argument("--export", required=False, action="store_true", help="export HF model to ExecuTorch")

    args = parser.parse_args()

    # Configs to HF model
    device = "cpu"
    dtype = torch.float32
    max_batch_size = 1
    max_seq_len = 32
    cache_implementation = "static"
    attn_implementation = "sdpa"
    prompt = "" # Use empty prompt as a hack to avoid parallel prefill in order to verify the correctness in eager

    # Load a HF model in eager
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_repo)
    config = AutoConfig.from_pretrained(
        args.hf_model_repo,
        torch_dtype=dtype,
        use_cache=True,
        max_length=max_seq_len,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_repo,
        config=config,
        attn_implementation=attn_implementation,
        device_map=device,
    )
    # Make sure generation config is consistent with the model config
    # TODO: In HF cache impl is a generation time config. To make the HF models work
    # properly with ExecuTorch, this needs to be a config at model construction time
    # and should not change at generation runtime.
    model.generation_config.cache_implementation = cache_implementation
    model.generation_config.max_length = max_seq_len # HF is setting this independently from config.max_length, and use this one to construct static kv cache
    print(f"DEBUG model config = {model.config}")
    print(f"DEBUG model generation_config = {model.generation_config}")

    if args.compile:
        # torch.compile
        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        input_tokens = tokenizer(prompt, return_tensors="pt")
        outputs = compiled_model.generate(**input_tokens)
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"DEBUG output_texts: {output_texts}")

    if args.export:
        # torch.export
        input_tokens = tokenizer(prompt, return_tensors="pt")
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            prog = export_to_edge(
                model,
                (
                    torch.tensor([[1]], dtype=torch.long), # tokens, with kv cache our input token length is always just 1 token.
                    torch.tensor([0], dtype=torch.long), # input_pos, what token of output are we on.)
                ),
                edge_compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                ),
                edge_constant_methods={
                    "get_bos_id": config.bos_token_id,
                    "get_dtype": 5 if config.torch_dtype == torch.float16 else 6,
                    "get_eos_id": config.eos_token_id,
                    "get_head_dim": config.hidden_size / config.num_attention_heads,
                    "get_max_batch_size": max_batch_size,
                    "get_max_seq_len": max_seq_len,
                    "get_n_bos": 1,
                    "get_n_eos": 1,
                    "get_n_kv_heads": config.num_key_value_heads,
                    "get_n_layers": config.num_hidden_layers,
                    "use_kv_cache": config.use_cache,
                    "get_vocab_size": config.vocab_size,
                },
                verbose=False,
            ).to_executorch(
                ExecutorchBackendConfig(
                    extract_constant_segment=True,
                    extract_delegate_segments=True
                )
            )
            filename = os.path.join("./", f"{config.model_type}.pte")
            with open(filename, "wb") as f:
                prog.write_to_file(f)
                print(f"Saved exported program to {filename}")


if __name__ == "__main__":
    main()
