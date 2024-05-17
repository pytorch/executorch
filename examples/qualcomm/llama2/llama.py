# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import codecs
import json
import os
import sys

from functools import partial

import torch

from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.utils import convert_linear_to_conv2d
from executorch.examples.qualcomm.llama2.model.static_llama import LlamaModel, ModelArgs
from executorch.examples.qualcomm.scripts.utils import (
    build_executorch_binary,
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
)

from sentencepiece import SentencePieceProcessor
from torch.ao.quantization.observer import MinMaxObserver


def create_device_inputs(example_inputs):
    # TODO: support batch inputs if necessary
    input_list = ""
    inputs, flat_inputs = [], []
    for input in example_inputs:
        if isinstance(input, list):
            for inp in input:
                flat_inputs.append(inp)
        else:
            flat_inputs.append(input)

    for i, data in enumerate(flat_inputs):
        input_list += f"input_0_{i}.raw "
        inputs.append(data)

    input_list += "\n"
    return tuple(inputs), input_list


def calibrate(example_inputs, module: torch.fx.GraphModule):
    sp_model = SentencePieceProcessor(model_file="tokenizer.model")
    _, _, atten_mask, k_caches, v_caches = example_inputs

    # TODO: change criteria & support batch inputs if necessary
    pos = torch.tensor(0, dtype=torch.int32)
    token_list = [sp_model.bos_id()]
    user_prompts = ["Once"]
    for prompt in user_prompts:
        token_list += sp_model.encode(prompt)

    def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        probs_sort, probs_indices = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0
        probs_sort /= probs_sort.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(probs_sort, num_samples=1)
        return probs_indices.gather(dim=-1, index=next_token)

    with torch.no_grad():
        while token_list[-1] != sp_model.eos_id() and pos < 128:
            logits, new_k_caches, new_v_caches = module(
                torch.full((1, 1), token_list[pos]),
                torch.full((1, 1), pos),
                atten_mask,
                *k_caches,
                *v_caches,
            )
            k_caches = [
                torch.cat([k_cache[:, 1:, :], new_k_caches[i]], dim=1)
                for i, k_cache in enumerate(k_caches)
            ]
            v_caches = [
                torch.cat([v_cache[:, 1:, :], new_v_caches[i]], dim=1)
                for i, v_cache in enumerate(v_caches)
            ]

            pos += 1
            atten_mask[0][-pos - 1] = 0
            if pos >= len(token_list):
                probs = torch.softmax(logits[:, -1] / 0.8, dim=-1)
                token_list.append(sample_top_p(probs, 0.9).item())

    print(f"calibration data:\n{sp_model.decode(token_list)}")


if __name__ == "__main__":
    parser = setup_common_args_and_variables()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./llama2_qnn",
        default="./llama2_qnn",
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
        "-P",
        "--ptq",
        help="If specified, will do PTQ quantization. default is 16bits activation and 4bits weight. Support 8a8w and 16a4w.",
        default="16a4w",
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama2 checkpoint.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--params",
        help="Pass llama2 params json file.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_bin",
        help="Pass llama2 tokenizer binary.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama2 tokenizer model.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for llama2.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--seq_len",
        help="Ouput sequence length for llama2.",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--temperature",
        help="Sampling temperature for llama2.",
        default=0.8,
        type=float,
    )

    parser.add_argument(
        "--pre_gen_pte",
        help="Pre-generated llama2.",
        type=str,
    )

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    with open(args.params) as f:
        config = ModelArgs(**json.load(f))
        # TODO: support batch inputs if necessary
        config.max_batch_size = 1

    state_dict = torch.load(args.checkpoint)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    with torch.device("meta"):
        instance = LlamaModel(config)
    instance.load_state_dict(state_dict, strict=False, assign=True)

    inputs, input_list = create_device_inputs(instance.get_export_inputs())
    pte_filename = "llama2_qnn"

    if args.ptq == "8a8w":
        quant_dtype = QuantDtype.use_8a8w
    elif args.ptq == "16a4w":
        quant_dtype = QuantDtype.use_16a4w
    else:
        raise AssertionError(
            f"No support for quant type {args.ptq}. Support 8a8w and 16a4w."
        )

    if args.use_fp16:
        quant_dtype = None
    else:
        assert args.tokenizer_model is not None, "Need tokenizer model for calibration"

    # prepare sha if the function is provided
    for l in instance.layers:
        if getattr(l.attention, "prepare_sha", None):
            l.attention.prepare_sha()

    if args.pre_gen_pte is None:
        build_executorch_binary(
            # try this if you want: convert_linear_to_conv2d(instance.eval()),
            instance.eval(),
            inputs,
            args.model,
            f"{args.artifact}/{pte_filename}",
            partial(calibrate, instance.get_example_inputs()),
            custom_annotations=(),
            quant_dtype=quant_dtype,
            per_channel_linear=True,
            shared_buffer=args.shared_buffer,
            metadata=instance.get_metadata(),
            direct_io=True,
            act_observer=MinMaxObserver,
        )

    if args.compile_only:
        sys.exit(0)

    # build custom commands for qnn_llama_runner
    pte_path = (
        f"{args.artifact}/{pte_filename}.pte"
        if args.pre_gen_pte is None
        else args.pre_gen_pte
    )
    workspace = f"/data/local/tmp/executorch/{pte_filename}"
    runner_args = " ".join(
        [
            f"--model_path {pte_filename}.pte",
            "--output_folder_path outputs",
            "--input_list_path input_list.txt",
            f"--tokenizer_path {os.path.basename(args.tokenizer_bin)}",
            f"--prompt {args.prompt}",
            f"--seq_len {args.seq_len}",
            f"--temperature {args.temperature}",
        ]
    )
    runner_cmd = " ".join(
        [
            f"cd {workspace} &&",
            "export ADSP_LIBRARY_PATH=. &&",
            "export LD_LIBRARY_PATH=. &&",
            f"./qnn_llama_runner {runner_args}",
        ]
    )

    adb = SimpleADB(
        qnn_sdk=os.getenv("QNN_SDK_ROOT"),
        artifact_path=f"{args.build_folder}",
        pte_path=pte_path,
        workspace=workspace,
        device_id=args.device,
        host_id=args.host,
        soc_model=args.model,
        shared_buffer=args.shared_buffer,
        runner="examples/qualcomm/qnn_llama_runner",
    )
    adb.push(inputs=[inputs], input_list=input_list, files=[args.tokenizer_bin])
    adb.execute(custom_runner_cmd=runner_cmd)

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        for f in sorted(
            os.listdir(output_data_folder), key=lambda f: int(f.split("_")[1])
        ):
            with codecs.open(
                os.path.join(output_data_folder, f),
                "r",
                encoding="utf-8",
                errors="replace",
            ) as fdata:
                outputs.append(fdata.read())

    adb.pull(output_path=args.artifact, callback=post_process)

    for idx, output in enumerate(outputs):
        print(f"Results[{idx}]:\n{output}")
