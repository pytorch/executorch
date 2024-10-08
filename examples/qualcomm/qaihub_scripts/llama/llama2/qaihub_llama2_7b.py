# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from multiprocessing.connection import Client

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManagerAdaptor
import torch
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (  # noqa: F401
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import (
    from_context_binary,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    generate_qnn_executorch_option,
)
from executorch.examples.qualcomm.qaihub_scripts.utils.utils import (
    gen_pte_from_ctx_bin,
    get_encoding,
)
from executorch.examples.qualcomm.utils import (
    setup_common_args_and_variables,
    SimpleADB,
)


def main(args):
    os.makedirs(args.artifact, exist_ok=True)

    target_names = (
        [
            f"llama_v2_7b_chat_quantized_PromptProcessor_{i}_Quantized.bin"
            for i in range(1, 5)
        ]
        if args.use_prompt_processor
        else [
            f"llama_v2_7b_chat_quantized_TokenGenerator_{i}_Quantized.bin"
            for i in range(1, 5)
        ]
    )

    # common part for compile & inference
    backend_options = generate_htp_compiler_spec(
        use_fp16=False,
        use_multi_contexts=True,
    )
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.model),
        backend_options=backend_options,
        is_from_context_binary=True,
    )

    if args.use_prompt_processor:
        pte_name = "qaihub_llama2_7b_prompt"
        last_shard_num_inputs = 4
        last_shard_num_outputs = 513
    else:
        pte_name = "qaihub_llama2_7b_token"
        last_shard_num_inputs = 516
        last_shard_num_outputs = 513

    if args.pre_gen_pte is None:
        # create custom operators as context loader
        bundle_programs = [
            from_context_binary(f"{args.context_binaries}/{target}", f"ctx_loader_{i}")
            for i, target in enumerate(target_names)
        ]
        pte_names = [f"{pte_name}_{i}" for i in range(len(target_names))]
        pte_files = gen_pte_from_ctx_bin(
            args.artifact, pte_names, compiler_specs, bundle_programs
        )
    else:
        pte_files = [f"{args.pre_gen_pte}/{pte_name}_{i}.pte" for i in range(4)]

    if args.compile_only:
        return

    def get_logit_encoding(path_to_last_shard: str):
        with open(f"{args.context_binaries}/{path_to_last_shard}", "rb") as f:
            ctx_bin = f.read()
            qnn_mgr = PyQnnManagerAdaptor.QnnManager(
                generate_qnn_executorch_option(compiler_specs), ctx_bin
            )
            assert qnn_mgr.Init().value == 0, "failed to load context binary"
            qnn_mgr.AllocateTensor()
            logits = qnn_mgr.GetGraphOutputs()[-1]
            encoding = logits.GetEncodings()
            qnn_mgr.Destroy()
            return encoding.data["scale"].item(), encoding.data["offset"].item()

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        build_path=args.build_folder,
        pte_path=pte_files,
        workspace=f"/data/local/tmp/executorch/{pte_name}",
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        runner="examples/qualcomm/qaihub_scripts/llama/qaihub_llama2_7b_runner",
    )
    output_file = "result.txt"
    pos_embs_file = ["freq_cos", "freq_sin"]
    encoding = get_encoding(
        path_to_shard=f"{args.context_binaries}/{target_names[-1]}",
        compiler_specs=compiler_specs,
        get_input=False,
        get_output=True,
        num_input=last_shard_num_inputs,
        num_output=last_shard_num_outputs,
    )[0]
    scale = encoding["scale"][-1]
    offset = encoding["offset"][-1]
    outputs = []
    runner_args = [
        *[
            f"--sharded_{i+1}_path {os.path.basename(pte_file)}"
            for i, pte_file in enumerate(pte_files)
        ],
        *[f"--{fname}_path {fname}.raw" for fname in pos_embs_file],
        f"--output_path {adb.output_folder}/{output_file}",
        f"--tokenizer_path {os.path.basename(args.tokenizer_bin)}",
        f"--prompt '{args.prompt}'",
        f"--temperature {args.temperature}",
        f"--seq_len {args.seq_len}",
        f"--eval_mode {0 if args.use_prompt_processor else 1}",
        f"--logits_scale {scale}",
        f"--logits_offset {-offset}",
    ]
    runner_cmds = " ".join(
        [
            f"cd {adb.workspace} &&",
            f"./qaihub_llama2_7b_runner {' '.join(runner_args)}",
        ]
    )

    def compute_pos_embedding():
        head_dim, max_seq_len, theta = 128, 1024, 10000.0
        base = torch.arange(0, head_dim, 2)
        freqs = 1.0 / (theta ** (base[: (head_dim // 2)].float() / head_dim))
        t = torch.arange(max_seq_len * 2)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        freqs_cis = freqs_cis[0:max_seq_len]
        freqs_real = torch.view_as_real(freqs_cis)
        return freqs_real[:, :, 0], freqs_real[:, :, 1]

    def post_process():
        with open(f"{args.artifact}/outputs/{output_file}", "r") as f:
            outputs.append(f.read())

    custom_files = [args.tokenizer_bin]
    for var_name, freq in zip(pos_embs_file, compute_pos_embedding()):
        custom_files.append(f"{adb.working_dir}/{var_name}.raw")
        scale, offset = (freq.max() - freq.min()) / 65535, 32768
        freq = (freq / scale + offset).clip(min=0, max=65535).detach()
        freq.to(dtype=torch.uint16).numpy().tofile(custom_files[-1])

    if not args.skip_push:
        adb.push(files=custom_files)
    adb.execute(custom_runner_cmd=runner_cmds)
    adb.pull(args.artifact, callback=post_process)
    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "result": outputs[0],
                    }
                )
            )
    else:
        print(outputs[0])


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./llama2_qai_hub",
        default="./llama2_qai_hub",
        type=str,
    )

    parser.add_argument(
        "--context_binaries",
        help="path to context binaries generated from qai_hub",
        required=True,
    )

    parser.add_argument(
        "--use_prompt_processor",
        help="tokens will be evaluated all at once",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--tokenizer_bin",
        help="llama2 tokenizer binary",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--seq_len",
        help="ouput sequence length for llama2",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--temperature",
        help="sampling temperature for llama2",
        default=0.0,
        type=float,
    )

    parser.add_argument(
        "--prompt",
        help="user prompts for llama2",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="folder path to pre-compiled ptes",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)
