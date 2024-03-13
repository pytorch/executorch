import argparse
import os
import sys
import traceback

from examples.portable.utils import export_to_edge
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackDynamicallyQuantizedPartitioner,
)
from executorch.exir.capture import EdgeCompileConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


HF_LLM_MODLE_REPOS = [
    # architecture: llama
    ("mistralai/Mixtral-8x7B-v0.1",),
    ("mistralai/Mistral-7B-v0.1",),
    ("bofenghuang/vigogne-2-7b-chat",),
    ("hfl/chinese-llama-2-7b",),
    ("hfl/chinese-alpaca-2-7b",),
    ("TheBloke/koala-7B-HF",),
    # architecture: falcon
    ("tiiuae/falcon-7b",),
    # architecture: gpt2
    ("openai-community/gpt2", "GOOD"),
    # architecture: gptj
    ("EleutherAI/gpt-j-6b", "GOOD"),
    # architecture: gptneox
    ("EleutherAI/gpt-neo-1.3B", "GOOD"),
    ("EleutherAI/gpt-neox-20b",),
    # architecture: plamo
    ("pfnet/plamo-13b", "GOOD"),
    # architecture: orion
    ("OrionStarAI/Orion-14B-Base",),
    ("OrionStarAI/Orion-14B-Chat",),
    # architecture: mpt
    ("mosaicml/mpt-7b",),
    # architecture: baichuan
    ("baichuan-inc/Baichuan2-7B-Base",),
    ("baichuan-inc/Baichuan-7B", "GOOD"),
    # architecture: rwkv
    ("RWKV/rwkv-5-world-1b5",),
    # architecture: starcoder
    ("bigcode/starcoder", "GOOD"),
    ("bigcode/starcoder2-3b",),  # Not avialable in 4.38.2
    # architecture: persimmon
    ("adept/persimmon-8b-chat", "GOOD"),
    # architecture: refact
    ("smallcloudai/Refact-1_6B-fim",),
    # architecture: bloom
    ("bigscience/bloom-7b1", "GOOD"),
    # architecture: stablelm
    ("stabilityai/stablelm-3b-4e1t", "GOOD"),
    ("stabilityai/stablelm-2-1_6b", "GOOD"),
    # architecture: qwen
    ("Qwen/Qwen-7B-Chat",),
    ("Qwen/Qwen1.5-7B-Chat",),
    # architecture: phi2
    ("microsoft/phi-2", "GOOD"),
    ("microsoft/phi-1_5", "GOOD"),
    ("microsoft/phi-1", "GOOD"),
    # architecture: codeshell
    ("WisdomShell/CodeShell-7B",),
    # architecture: internlm2
    ("internlm/internlm2-7b", "GOOD"),
    # architecture: minicpm
    ("openbmb/MiniCPM-2B-sft-bf16",),
    # architecture: gemma
    ("google/gemma-2b",),
    ("google/gemma-7b",),
    # unknown
    ("microsoft/biogpt", "GOOD"),
    ("deepseek-ai/deepseek-llm-7b-chat",),
    ("01-ai/Yi-6B",),
    ("BAAI/Aquila-7B", "GOOD"),
    ("BAAI/Aquila2-7B", "GOOD"),
    ("THUDM/chatglm3-6b",),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-hfm",
        "--hf_model_repo",
        required=False,
        default=None,
        help="a valid huggingface model repo name",
    )
    parser.add_argument("-o", "--output-dir", default=".", help="output directory")
    parser.add_argument(
        "-etx",
        "--to_et_w_xnn",
        required=False,
        action="store_true",
        help="export ExecuTorch with XNNPACK",
    )
    parser.add_argument(
        "-sg",
        "--skip_good",
        required=False,
        action="store_true",
        help="skip good models",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist.")
        return sys.exit(1)

    hf_model_repos = (
        HF_LLM_MODLE_REPOS if not args.hf_model_repo else [(args.hf_model_repo,)]
    )

    device = "cpu"
    prompt = "What is your favourite condiment?"
    result_summary = {}

    for repo_config in hf_model_repos:
        hf_model_repo, status = repo_config[:2] + ("",) * (2 - len(repo_config))
        if args.skip_good and status == "GOOD":
            print(f"Skip model '{hf_model_repo}'.")
            result_summary[hf_model_repo] = "SKIPPED"
            continue

        model_name = hf_model_repo.split("/")[-1].lower()
        try:
            print("\n\n#######################################################")
            print(f"Load model '{hf_model_repo}' from Hugginface.")
            print("#######################################################")
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_repo, trust_remote_code=True
            ).to(device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model_repo, trust_remote_code=True
            )
            model_inputs = (
                tokenizer([prompt], return_tensors="pt").to(device)["input_ids"],
            )

            edge_m = export_to_edge(
                model,
                model_inputs,
                edge_compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                ),
            )
            print(f"Exported '{hf_model_repo}' successfully.")

            if args.to_et_w_xnn:
                partitioners = {}
                partitioners[
                    XnnpackDynamicallyQuantizedPartitioner.__name__
                ] = XnnpackDynamicallyQuantizedPartitioner()
                prog = edge_m.to_backend(partitioners).to_executorch()
                print(f"Lowered graph:\n{prog.exported_program().graph}")
                filename = os.path.join(args.output_dir, f"xnnpack_{model_name}.pte")
                with open(filename, "wb") as f:
                    f.write(prog.buffer)
                    print(f"Saved exported program to {filename}")

            result_summary[hf_model_repo] = "GOOD"

        except Exception:
            filename = os.path.join(args.output_dir, f"err_{model_name}.txt")
            with open(filename, "w") as f:
                traceback.print_exc(file=f)
            print(
                f"Failed to export '{hf_model_repo}'.\nRecorded stack trace to {filename}"
            )
            result_summary[hf_model_repo] = "ERROR"

    print(f"Export summary: \n{result_summary}.")


if __name__ == "__main__":
    main()
