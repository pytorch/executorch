# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: reenable pyre after fixing the issues
# pyre-ignore-all-errors

import json
import logging
import os
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
    DECODE_QDQ_FILENAME,
    DECODER_GRAPH_NAMES,
    EVAL_MODE,
    PROMPT_EVAL,
    SQNR_EVAL,
    TASKS_EVAL,
    TEXT_DECODER,
    TEXT_EMBEDDING,
    TEXT_EMBEDDING_GRAPH_NAMES,
    TEXT_ENCODER,
    VISION_ENCODER,
)
from executorch.examples.qualcomm.oss_scripts.llama.decoder_runtime_evaluator import (
    DefaultEval,
    SqnrEval,
    TaskEval,
)

from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import TokenizerWrapper
from executorch.examples.qualcomm.oss_scripts.llama.wrappers import (
    MultiModalManager,
    next_power_of_two,
)
from executorch.examples.qualcomm.utils import setup_common_args_and_variables
from torchao.quantization.utils import compute_error


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
    # compile spec for multimodality encoder
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
    chat_template,
):

    assert args.model_mode in EVAL_MODE, f"Unknown model_mode: {args.model_mode}."

    is_modality = hasattr(decoder_model_config, VISION_ENCODER) or hasattr(
        decoder_model_config, AUDIO_ENCODER
    )
    decoder_pte_path = (
        f"{args.pre_gen_pte}/{pte_filenames[TEXT_DECODER]}.pte"
        if args.pre_gen_pte
        else f"{args.artifact}/{pte_filenames[TEXT_DECODER]}.pte"
    )
    pte_paths = {TEXT_DECODER: decoder_pte_path}
    eval_results = {
        "pte_size": os.path.getsize(decoder_pte_path),
    }

    if is_modality:
        vision_encoder_pte_path = (
            f"{args.pre_gen_pte}/{pte_filenames[VISION_ENCODER]}.pte"
            if args.pre_gen_pte
            else f"{args.artifact}/{pte_filenames[VISION_ENCODER]}.pte"
        )
        text_embedding_pte_path = (
            f"{args.pre_gen_pte}/{pte_filenames[TEXT_EMBEDDING]}.pte"
            if args.pre_gen_pte
            else f"{args.artifact}/{pte_filenames[TEXT_EMBEDDING]}.pte"
        )
        eval_results.update(
            {
                "encoder_pte_size": os.path.getsize(vision_encoder_pte_path),
                "text_embedding_pte_size": os.path.getsize(text_embedding_pte_path),
            }
        )
        pte_paths.update(
            {
                VISION_ENCODER: vision_encoder_pte_path,
                TEXT_EMBEDDING: text_embedding_pte_path,
            }
        )

    if PROMPT_EVAL in args.eval_methods:
        prompt_evaluator = DefaultEval(
            args=args,
            pte_paths=pte_paths,
            runtime_tokenizer_path=runtime_tokenizer_path,
            is_modality=is_modality,
        )
        output_prompt = prompt_evaluator.run(prompt=args.prompt)
        eval_results.update(
            {
                "inference_speed": prompt_evaluator.inference_speed,
                "result": output_prompt,
            }
        )
        for idx, output in enumerate(output_prompt):
            logging.info(f"Device Inference Results[{idx}]:\n{output}")

    if SQNR_EVAL in args.eval_methods:
        assert not is_modality, "Modality Model does not support SQNR_EVAL."
        tokenizer_wrapper = TokenizerWrapper(
            args,
            decoder_model_config,
        )
        prompt = (
            tokenizer_wrapper.apply_prompt_template(
                chat_template, args.prompt[0], args.system_prompt
            )
            if chat_template is not None
            else args.prompt[0]
        )
        multi_modal_mgr = MultiModalManager(
            control_args=args, config=decoder_model_config
        )
        source_model = multi_modal_mgr.text_decoder.decode.decoder
        sqnr_evaluator = SqnrEval(
            source_model=source_model,
            get_example_inputs=source_model.get_example_inputs,
            args=args,
            pte_paths=pte_paths,
            tokenizer=tokenizer,
            runtime_tokenizer_path=runtime_tokenizer_path,
            is_modality=is_modality,
        )
        sqnr, golden_logits, _ = sqnr_evaluator.run(prompt=prompt)
        logging.info(f"SQNR Eval Score between FP32 nn.Module and QNN: {sqnr}")
        eval_results.update(
            {
                "sqnr": sqnr,
                "inference_speed": sqnr_evaluator.inference_speed,
            }
        )

        qdq_ep_path = (
            f"{args.pre_gen_pte}/{DECODE_QDQ_FILENAME}"
            if args.pre_gen_pte
            else f"{args.artifact}/{DECODE_QDQ_FILENAME}"
        )
        if os.path.exists(qdq_ep_path):
            qdq_ep = torch.export.load(qdq_ep_path)
            qdq_sqnr_evaluator = SqnrEval(
                source_model=qdq_ep.module(),
                get_example_inputs=source_model.get_example_inputs,
                args=args,
                pte_paths=pte_paths,
                tokenizer=tokenizer,
                runtime_tokenizer_path=runtime_tokenizer_path,
                is_modality=is_modality,
            )
            qdq_sqnr, cpu_qdq_logits, _ = qdq_sqnr_evaluator.run(prompt=prompt)
            eval_results["qdq_sqnr"] = qdq_sqnr
            logging.info(f"SQNR Eval Score between CPU QDQ and QNN: {qdq_sqnr}")
            logging.info(
                f"SQNR Eval Score between FP32 nn.Module and CPU QDQ: {compute_error(golden_logits, cpu_qdq_logits).item()}"
            )
        else:
            logging.info(
                f"Couldn't find saved qdq_ep under {qdq_ep_path}, skip eval sqnr for CPU QDQ."
            )

    if TASKS_EVAL in args.eval_methods:
        assert not is_modality, "Modality Model does not support TASKS_EVAL."
        # Generate the eval wrapper
        ppl_evaluator = TaskEval(
            args=args,
            pte_paths=pte_paths,
            tokenizer=tokenizer,
            runtime_tokenizer_path=runtime_tokenizer_path,
            is_modality=is_modality,
        )
        ppl_eval_result = ppl_evaluator.run()
        eval_results["inference_speed"] = ppl_evaluator.inference_speed

        for task, res in ppl_eval_result["results"].items():
            match task:
                case "wikitext":
                    wiki_ppl = ppl_eval_result["results"][task]["word_perplexity,none"]
                    eval_results["wiki_ppl"] = wiki_ppl
                case "hellaswag":
                    acc_norm = ppl_eval_result["results"][task]["acc_norm,none"]
                    eval_results["acc_norm"] = acc_norm
                case _:
                    if args.ip and args.port != -1:
                        raise RuntimeError(
                            "CI currently supports [wikitext, hellaswag] only."
                        )
            logging.info(f"{task}: {res}")

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps(eval_results))


def _build_parser():
    parser = setup_common_args_and_variables()
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

    parser.add_argument(
        "--eval_methods",
        choices=[PROMPT_EVAL, TASKS_EVAL, SQNR_EVAL],
        nargs="+",
        default=[PROMPT_EVAL],
        help="Choose eval methods(default: prompt_eval). Users can provide more than 1 eval methods. For example: --eval_methods tasks_eval sqnr_eval."
        "Following eval methods are supported:"
        "1) prompt_eval: Model will generate the output response based on the provided prompt through the flag --prompt."
        "2) tasks_eval: This will eval the tasks provided through the flag --tasks."
        "3) sqnr_eval: This will eval the sqnr between between QNN's output logit V.S. Static Llama nn.Module's output logit. Eval is based on the provided prompt through the --prompt flag. Please note that sqnr will only eval the prompt's logit but not the new generated token's logit.",
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

    parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def export_llama(args) -> None:
    if args.compile_only and args.pre_gen_pte:
        raise RuntimeError("Cannot set both compile_only and pre_gen_pte as true")
    if (TASKS_EVAL or SQNR_EVAL) in args.eval_methods and args.model_mode not in {
        "kv",
        "hybrid",
    }:
        raise RuntimeError(
            "Eval device perplexity is only supported for KV mode. Hybrid mode will only use KV mode when evaluating tasks/sqnr."
        )
    if TASKS_EVAL in args.eval_methods and args.tasks is None:
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
            args,
            decoder_model_config,
            pte_filenames,
            runtime_tokenizer_path,
            tokenizer,
            chat_template,
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
        args,
        decoder_model_config,
        pte_filenames,
        runtime_tokenizer_path,
        tokenizer,
        chat_template,
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
