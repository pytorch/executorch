# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import getpass
import json
import logging
import os
import subprocess
import sys
from multiprocessing.connection import Client
from typing import Dict

import torch
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
    get_soc_to_chipset_map,
)
from executorch.examples.qualcomm.oss_scripts.llama import (
    LLMModelConfig,
    SUPPORTED_LLM_MODELS,
)
from executorch.examples.qualcomm.oss_scripts.llama.dataset import DatasetBuilder
from executorch.examples.qualcomm.oss_scripts.llama.decoder_constants import (
    AUDIO_ENCODER,
    DECODER_GRAPH_NAMES,
    EVAL_MODE,
    TEXT_DECODER,
    TEXT_EMBEDDING,
    TEXT_EMBEDDING_GRAPH_NAMES,
    TEXT_ENCODER,
    VISION_ENCODER,
    VISION_ENCODER_INPUT_FILENAME,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_utils import (
    QnnRunnerEvalWrapper,
)
from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import (
    MultiModalManager,
    next_power_of_two,
)
from executorch.examples.qualcomm.utils import (
    make_output_dir,
    setup_common_args_and_variables,
    SimpleADB,
)

try:
    from lm_eval.evaluator import simple_evaluate
except ImportError:
    raise ImportError(
        "Please install the llm eval dependency via examples/models/llama/install_requirements.sh"
    )

sys.setrecursionlimit(4096)
FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)
# Avoid the error message "Could not initialize NNPACK! Reason: Unsupported hardware."
torch.backends.nnpack.set_flags(False)


def compile(
    args,
    decoder_model_config: LLMModelConfig,
    pte_filenames: Dict[str, str],
    tokenizer,
    calibration_data,
):
    os.makedirs(args.artifact, exist_ok=True)
    multi_modal_mgr = MultiModalManager(control_args=args, config=decoder_model_config)

    # perform ptq
    multi_modal_mgr.quantize(
        calibration_data=calibration_data,
        tokenizer=tokenizer,
    )

    # Prepare dataset
    compile_specs = {
        AUDIO_ENCODER: None,
        TEXT_ENCODER: None,
        VISION_ENCODER: None,
        TEXT_EMBEDDING: None,
        TEXT_DECODER: None,
    }
    is_modality = False
    # compile spec for multimodlity encoder
    for modality in compile_specs:
        if not hasattr(decoder_model_config, modality):
            continue

        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
        )
        encoder_compile_specs = generate_qnn_executorch_compiler_spec(
            soc_model=get_soc_to_chipset_map()[args.model],
            backend_options=backend_options,
        )
        compile_specs[modality] = encoder_compile_specs
        is_modality = True

    # text embedding compilation spec: default we use quantization version, since embedding is huge
    if is_modality:
        backend_options = generate_htp_compiler_spec(
            use_fp16=False,
            # x86 emulator does not support weight sharing
            use_weight_sharing=not args.enable_x86_64,
        )
        compile_specs[TEXT_EMBEDDING] = [
            generate_qnn_executorch_compiler_spec(
                soc_model=get_soc_to_chipset_map()[args.model],
                backend_options=backend_options,
                shared_buffer=not args.enable_x86_64,  # x86 emulator does not support shared buffer
            )
        ] * len(TEXT_EMBEDDING_GRAPH_NAMES)

    # compile spec for text decoder
    backend_options = generate_htp_compiler_spec(
        use_fp16=False,
        use_multi_contexts=decoder_model_config.num_sharding > 1,
        # x86 emulator does not support weight sharing
        use_weight_sharing=not args.enable_x86_64,
    )
    compile_specs[TEXT_DECODER] = [
        generate_qnn_executorch_compiler_spec(
            soc_model=get_soc_to_chipset_map()[args.model],
            backend_options=backend_options,
            shared_buffer=not args.enable_x86_64,
            use_mha2sha=True,
        )
    ] * len(DECODER_GRAPH_NAMES)

    # perform compilation
    multi_modal_mgr.compile(compile_specs=compile_specs, pte_filenames=pte_filenames)


def inference(
    args,
    decoder_model_config: LLMModelConfig,
    pte_filenames: Dict[str, str],
    runtime_tokenizer_path,
    tokenizer,
):

    assert args.model_mode in EVAL_MODE, f"Unknown model_mode: {args.model_mode}."

    is_modality = hasattr(decoder_model_config, VISION_ENCODER) or hasattr(
        decoder_model_config, AUDIO_ENCODER
    )

    pte_path = (
        f"{args.pre_gen_pte}/{pte_filenames[TEXT_DECODER]}.pte"
        if args.pre_gen_pte
        else f"{args.artifact}/{pte_filenames[TEXT_DECODER]}.pte"
    )

    # For decoder-only models, enable accuracy evaluation using perplexity
    # TODO: Add support for multimodal accuracy evaluation (e.g., VLM)
    if not is_modality and args.run_lm_eval:
        # Generate the eval wrapper
        eval_wrapper = QnnRunnerEvalWrapper(
            args=args,
            pte_path=pte_path,
            tokenizer=tokenizer,
            runtime_tokenizer_path=runtime_tokenizer_path,
        )

        # Evaluate the model
        with torch.no_grad():
            eval_results = simple_evaluate(
                model=eval_wrapper,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                limit=args.limit,
            )

        if args.ip and args.port != -1:
            assert len(args.tasks) == 1, "CI currently supports 1 lm_eval task only."
            match args.tasks[0]:
                case "wikitext":
                    wiki_ppl = eval_results["results"][args.tasks[0]][
                        "word_perplexity,none"
                    ]
                    pte_size = os.path.getsize(pte_path)
                    with Client((args.ip, args.port)) as conn:
                        conn.send(
                            json.dumps(
                                {
                                    "wiki_ppl": wiki_ppl,
                                    "pte_size": pte_size,
                                    "inference_speed": eval_wrapper.inference_speed,
                                }
                            )
                        )
                case "hellaswag":
                    acc_norm = eval_results["results"][args.tasks[0]]["acc_norm,none"]
                    pte_size = os.path.getsize(pte_path)
                    with Client((args.ip, args.port)) as conn:
                        conn.send(
                            json.dumps(
                                {
                                    "acc_norm": acc_norm,
                                    "pte_size": pte_size,
                                    "inference_speed": eval_wrapper.inference_speed,
                                }
                            )
                        )
                case _:
                    raise RuntimeError(
                        "CI currently supports [wikitext, hellaswag] only."
                    )

        else:
            for task, res in eval_results["results"].items():
                logging.info(f"{task}: {res}")
        return

    workspace = f"/data/local/tmp/{getpass.getuser()}/executorch/single_llama"

    # collect output data
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)
    outputs = []

    def post_process():
        with open(f"{args.artifact}/outputs/outputs.txt", "r") as f:
            outputs.append(f.read())

    seq_len = args.max_seq_len
    multi_prompts = " ".join([f'--prompt "{prompt}"' for prompt in args.prompt])
    lookahead_args = " ".join(
        [
            f"--window {args.window}",
            f"--gcap {args.gcap}",
            f"--ngram {args.ngram}",
        ]
    )
    runner_args = " ".join(
        [
            multi_prompts,
            f"--eval_mode {EVAL_MODE[args.model_mode]}",
            f"--temperature {args.temperature}",
            f"--system_prompt '{args.system_prompt}'",
            lookahead_args if args.model_mode == "lookahead" else "",
        ]
    )

    runner_cmd = ""
    performance_output_path = "outputs/inference_speed.txt"
    if args.enable_x86_64:
        # x86 emulator is intended for CI and not performance. Check only the first few tokens.
        seq_len = min(seq_len, 16)

        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        target = "x86_64-linux-clang"
        if not is_modality:
            runner_cmd = " ".join(
                [
                    f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{args.build_folder}/lib &&",
                    f"./{args.build_folder}/examples/qualcomm/oss_scripts/llama/qnn_llama_runner",
                    f"--decoder_model_version {decoder_model_config.decoder_model_version}",
                    f"--tokenizer_path {runtime_tokenizer_path}",
                    f"--model_path {pte_path}",
                    f"--seq_len {seq_len}",
                    f"--output_path {args.artifact}/outputs/outputs.txt",
                    f"--performance_output_path {args.artifact}/{performance_output_path}",
                    runner_args,
                ]
            )
        else:
            # x86 emulator is intended for CI and not performance. Check only the first few tokens.
            # For multimodal models, use 128 tokens (vs 16 for text-only) due to longer sequence length required for modality embeddings.
            seq_len = min(seq_len, 128)
            encoder_pte_path = (
                f"{args.pre_gen_pte}/{pte_filenames[VISION_ENCODER]}.pte"
                if args.pre_gen_pte
                else f"{args.artifact}/{pte_filenames[VISION_ENCODER]}.pte"
            )
            text_embedding_pte_path = (
                f"{args.pre_gen_pte}/{pte_filenames[TEXT_EMBEDDING]}.pte"
                if args.pre_gen_pte
                else f"{args.artifact}/{pte_filenames[TEXT_EMBEDDING]}.pte"
            )
            runner_cmd = " ".join(
                [
                    f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{args.build_folder}/lib &&",
                    f"./{args.build_folder}/examples/qualcomm/oss_scripts/llama/qnn_multimodal_runner",
                    f"--decoder_model_version {decoder_model_config.decoder_model_version}",
                    f"--tokenizer_path {runtime_tokenizer_path}",
                    f"--decoder_path {pte_path}",
                    f"--encoder_path {encoder_pte_path}",
                    f"--embedding_path {text_embedding_pte_path}",
                    f"--image_path {args.artifact}/{VISION_ENCODER_INPUT_FILENAME}.raw",
                    f"--seq_len {seq_len}",
                    f"--output_path {args.artifact}/outputs/outputs.txt",
                    f"--performance_output_path {args.artifact}/{performance_output_path}",
                    runner_args,
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
        if not is_modality:
            runner_cmd = " ".join(
                [
                    f"cd {workspace} &&",
                    f"./qnn_llama_runner",
                    f"--decoder_model_version {decoder_model_config.decoder_model_version}",
                    f"--tokenizer_path {os.path.basename(runtime_tokenizer_path)}",
                    f"--model_path {pte_filenames[TEXT_DECODER]}.pte",
                    f"--seq_len {seq_len}",
                    "--output_path outputs/outputs.txt",
                    f"--performance_output_path {performance_output_path}",
                    "--shared_buffer",
                    runner_args,
                ]
            )
        else:
            encoder_pte_path = (
                f"{args.pre_gen_pte}/{pte_filenames[VISION_ENCODER]}.pte"
                if args.pre_gen_pte
                else f"{args.artifact}/{pte_filenames[VISION_ENCODER]}.pte"
            )
            text_embedding_pte_path = (
                f"{args.pre_gen_pte}/{pte_filenames[TEXT_EMBEDDING]}.pte"
                if args.pre_gen_pte
                else f"{args.artifact}/{pte_filenames[TEXT_EMBEDDING]}.pte"
            )
            runner_cmd = " ".join(
                [
                    f"cd {workspace} &&",
                    f"./qnn_multimodal_runner",
                    f"--decoder_model_version {decoder_model_config.decoder_model_version}",
                    f"--tokenizer_path {os.path.basename(runtime_tokenizer_path)}",
                    f"--decoder_path {pte_filenames[TEXT_DECODER]}.pte",
                    f"--encoder_path {pte_filenames[VISION_ENCODER]}.pte",
                    f"--embedding_path {pte_filenames[TEXT_EMBEDDING]}.pte",
                    f"--image_path {VISION_ENCODER_INPUT_FILENAME}.raw",
                    f"--seq_len {seq_len}",
                    "--output_path outputs/outputs.txt",
                    f"--performance_output_path {performance_output_path}",
                    "--shared_buffer",
                    runner_args,
                ]
            )

        adb = SimpleADB(
            qnn_sdk=os.getenv("QNN_SDK_ROOT"),
            build_path=f"{args.build_folder}",
            pte_path=(
                pte_path
                if not is_modality
                else [pte_path, encoder_pte_path, text_embedding_pte_path]
            ),
            workspace=workspace,
            device_id=args.device,
            host_id=args.host,
            soc_model=args.model,
            shared_buffer=True,
            target=args.target,
            runner=(
                f"examples/qualcomm/oss_scripts/llama/qnn_llama_runner"
                if not is_modality
                else f"examples/qualcomm/oss_scripts/llama/qnn_multimodal_runner"
            ),
        )

        # No pregen inputs, input_list is not required
        if not args.skip_push:
            # Always use image from artifact folder since that's where it's saved during preprocessing
            # regardless of whether pre_gen_pte is used (pre_gen_pte only applies to .pte model files)
            image_path = f"{args.artifact}/{VISION_ENCODER_INPUT_FILENAME}.raw"
            adb.push(
                inputs=[],
                files=[runtime_tokenizer_path] + ([image_path] if is_modality else []),
            )
        adb.execute(custom_runner_cmd=runner_cmd)
        adb.pull(output_path=args.artifact, callback=post_process)

    if args.ip and args.port != -1:
        inference_speed = 0
        with open(
            f"{os.path.abspath(args.artifact)}/{performance_output_path}", "r"
        ) as f:
            inference_speed = float(f.read())

        # Prepare validation results for CI system
        validation_results = {
            "result": outputs,
            "inference_speed": inference_speed,
            "pte_size": os.path.getsize(pte_path),
        }

        # Add multimodal-specific metrics if applicable
        if is_modality:
            validation_results.update(
                {
                    "encoder_pte_size": os.path.getsize(encoder_pte_path),
                    "text_embedding_pte_size": os.path.getsize(text_embedding_pte_path),
                }
            )

        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps(validation_results))
    else:
        for idx, output in enumerate(outputs):
            logging.info(f"Results[{idx}]:\n{output}")


def _build_tasks_parser(parser):
    parser.add_argument(
        "--run_lm_eval",
        help="If enabled, this will use the tasks provided under args.tasks to calibrate the model",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=None,
        help="list of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="number of samples to evalulate. If not set, evaluate all samples",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )

    return parser


def _build_parser():
    parser = setup_common_args_and_variables()
    parser = _build_tasks_parser(parser)
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts and output by this example. Default ./llama_qnn",
        default="./llama_qnn",
        type=str,
    )

    parser.add_argument(
        "--decoder_model",
        choices=list(SUPPORTED_LLM_MODELS.keys()),
        help=f"The llm model to export. Current available options are: { SUPPORTED_LLM_MODELS.keys()}",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        help="Pass llama checkpoint.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--params",
        help="Pass llama params json file.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_bin",
        help="For Llama2. Pass Llama2 tokenizer binary.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--tokenizer_model",
        help="Pass llama tokenizer model.",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--prompt",
        help="User prompts for Llama. When multiple prompts are entered, a multi-turn conversation will be initiated. Note that this feature is currently for testing purposes only.",
        required=True,
        type=str,
        nargs="+",
    )

    parser.add_argument(
        "--system_prompt",
        help="For Llama3/Granite. Tells the model what kind of assistant it should be. For example, You are a helpful AI assistant for travel tips and recommendations. Default is None",
        default="",
        type=str,
    )

    parser.add_argument(
        "--temperature",
        help="Sampling temperature for llama.",
        default=0.8,
        type=float,
    )

    parser.add_argument(
        "-d",
        "--dtype-override",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Override the dtype of the model (default is the checkpoint dtype). Options: fp32",
    )

    parser.add_argument(
        "--model_mode",
        help="Export and inference kv mode, hybrid mode, or lookahead decoding mode",
        default="hybrid",
        choices=["kv", "hybrid", "lookahead"],
        type=str,
    )

    parser.add_argument(
        "--max_seq_len",
        help="This refers to maximum number of tokens that the model can process & consider at once to generate predictions/responses.",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--prefill_ar_len",
        help="The auto-regression (AR) length determines the number of tokens to consume and the number of logits to produce. Use this option to process the prompt and generate the key-value (kv) cache, which serves as a prompt processor for hybrid and lookahead mode.",
        default=32,
        type=int,
    )

    parser.add_argument(
        "-E",
        "--embedding-quantize",
        default=None,
        type=str,
        help="Fallback to cpu embedding operator and type of embedding quantization, '<bitwidth>,<groupsize>', e.g., '4,32'.",
    )

    parser.add_argument(
        "--ngram",
        help="Represents the size of the n-grams used in the lookahead process.",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--window",
        help="Determines how many future tokens the algorithm attempts to predict in each step.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--gcap",
        help="Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--image_path",
        help="Path to the image file for multimodal language models (MLLM). If not specified, the default image from encoder/encoder_config.py will be used. The image should be preprocessed and saved in raw binary format.",
        default=None,
        type=str,
    )

    parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def export_llama(args) -> None:
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")
    if args.run_lm_eval and args.model_mode != "kv":
        raise RuntimeError("Eval device perplexity is only supported for KV mode")
    if args.run_lm_eval and args.tasks is None:
        raise RuntimeError("Please provide --tasks to eval perplexity")
    assert (
        args.decoder_model in SUPPORTED_LLM_MODELS
    ), f"Unknown decoder_model: {args.decoder_model}."
    decoder_model_config = SUPPORTED_LLM_MODELS[args.decoder_model]
    logging.info(f"*** {args.decoder_model} ***\n%s", str(decoder_model_config))

    # Specify pte filenames
    if args.model_mode == "kv":
        pte_filename = "kv_llama_qnn"
    elif args.model_mode == "hybrid":
        assert (
            args.max_seq_len >= args.prefill_ar_len
        ), "Please ensure max_seq_len is >= prefill_ar_len"
        pte_filename = "hybrid_llama_qnn"
    elif args.model_mode == "lookahead":
        assert (
            args.max_seq_len >= args.prefill_ar_len
        ), "Please ensure max_seq_len is >= prefill_ar_len"
        assert args.max_seq_len > next_power_of_two(
            (args.window + args.gcap) * (args.ngram - 1)
        ), "Please ensure max_seq_len is > next_power_of_two((args.window + args.gcap) * (args.ngram - 1))"
        pte_filename = "lookahead_llama_qnn"
    else:
        raise RuntimeError(f"Unknown model_mode: {args.model_mode}.")
    if args.decoder_model == "stories260k":
        pte_filename = f"{args.decoder_model}_" + pte_filename
    pte_filenames = {
        TEXT_DECODER: pte_filename,
        AUDIO_ENCODER: f"{AUDIO_ENCODER}_qnn",
        TEXT_ENCODER: f"{TEXT_ENCODER}_qnn",
        VISION_ENCODER: f"{VISION_ENCODER}_qnn",
        TEXT_EMBEDDING: f"{TEXT_EMBEDDING}_qnn",
    }
    # Prepare tokenizer
    tokenizer_wrapper = TokenizerWrapper(
        args,
        decoder_model_config,
    )
    runtime_tokenizer_path, tokenizer, chat_template = (
        tokenizer_wrapper.get_runtime_tokenizer(
            args.tokenizer_model, args.tokenizer_bin
        )
    )

    # Prepare dataset
    dataset_builder = DatasetBuilder(args, decoder_model_config, tokenizer_wrapper)
    calibration_data = dataset_builder.prepare_calibration_dataset(
        args.prompt, chat_template
    )

    # TODO: Implement multi-turn conversation support for multimodal models (vision/audio).
    assert (
        not (
            hasattr(decoder_model_config, VISION_ENCODER)
            or hasattr(decoder_model_config, AUDIO_ENCODER)
        )
    ) or (len(args.prompt) <= 1), (
        "Multimodal models currently do not support multi-turn. "
        "Please set `--prompt` to 1 or switch to a unimodal (text-only) decoder."
    )

    if args.pre_gen_pte:
        inference(
            args, decoder_model_config, pte_filenames, runtime_tokenizer_path, tokenizer
        )
        print(f"Finish the running pre_gen_pte from {args.pre_gen_pte}")
        return

    if args.compile_only:
        compile(
            args,
            decoder_model_config,
            pte_filenames,
            tokenizer,
            calibration_data,
        )

        if args.ip and args.port != -1:
            pte_path = f"{args.artifact}/{pte_filename}.pte"
            pte_size = os.path.getsize(pte_path)
            with Client((args.ip, args.port)) as conn:
                conn.send(
                    json.dumps(
                        {
                            "pte_size": pte_size,
                        }
                    )
                )
        print(f"Finish compile_only and save to {args.artifact}")
        return

    compile(
        args,
        decoder_model_config,
        pte_filenames,
        tokenizer,
        calibration_data,
    )
    inference(
        args, decoder_model_config, pte_filenames, runtime_tokenizer_path, tokenizer
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        export_llama(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise Exception(e)


# flake8: noqa: C901
if __name__ == "__main__":
    main()
