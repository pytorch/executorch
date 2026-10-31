# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import subprocess
from dataclasses import dataclass, replace
from multiprocessing.connection import Listener
from typing import List

import pytest

from executorch.backends.qualcomm.tests.rework.conftest import (
    add_default_cmds,
    get_ipc,
    require_paths,
)
from executorch.backends.qualcomm.tests.rework.src.e2e_metrics import (
    assert_metric,
    get_metric_constraint,
    LLM_MODELS,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require(request, artifact_names: list[str] = None):
    """Skip if any artifact is None; fail loudly if the path doesn't exist."""
    # executorch_root + artifact_dir are always required for e2e tests."""
    qnn_config = request.getfixturevalue("qnn_config")
    options = {
        n: request.config.getoption(n)
        for n in ["executorch_root", "artifact_dir"]
        + (artifact_names if artifact_names else [])
    }
    # extend this if the option needs no path validation
    keys_to_remove = ["model_name", "static_llm_eval_method", "use_fp16"]
    if request.config.getoption("model_name") not in _LLAMA_MODELS:
        keys_to_remove.append("llama_artifacts")
    require_paths({k: v for k, v in options.items() if k not in keys_to_remove})
    return *options.values(), qnn_config


def _run(cmds, qnn_config):
    ip, port = get_ipc(qnn_config)
    p = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
    with Listener((ip, port)) as listener:
        conn = listener.accept()
        p.communicate()
        return json.loads(conn.recv())


def _check(msg):
    if "Error" in msg:
        pytest.fail(msg["Error"])
    return msg


# ---------------------------------------------------------------------------
# LLM models
# ---------------------------------------------------------------------------


_LLAMA_MODELS = {"llama3_2-1b_instruct", "llama3_2-3b_instruct"}


# ---------------------------------------------------------------------------
# Multimodality model specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLLMSpecs:
    max_seq_len: int


@dataclass(frozen=True)
class ALMSpecs(MLLMSpecs):
    audio_path: str
    golden_audio_feature: str


@dataclass(frozen=True)
class VLMSpecs(MLLMSpecs):
    image_path: str
    golden_image_feature: str


ALM_SPECS = {
    "granite_speech_3_3-2b": ALMSpecs(
        max_seq_len=1024,
        audio_path="https://huggingface.co/ibm-granite/granite-speech-3.3-2b/resolve/main/10226_10111_000000.wav?download=true",
        golden_audio_feature="after his nap,",
    ),
}

VLM_SPECS = {
    "smolvlm_500m_instruct": VLMSpecs(
        max_seq_len=1024,
        image_path="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
        golden_image_feature="city",
    ),
    "internvl3_1b": VLMSpecs(
        max_seq_len=1024,
        image_path="http://images.cocodataset.org/val2017/000000039769.jpg",
        golden_image_feature="cats",
    ),
}


# ---------------------------------------------------------------------------
# TestExampleLLMScript
# ---------------------------------------------------------------------------


class StaticLLM:
    @staticmethod
    def test(request):  # noqa: C901
        (
            root,
            artifact,
            model_name,
            static_llm_eval_method,
            llama_artifacts,
            qnn_config,
        ) = _require(
            request, ["model_name", "static_llm_eval_method", "llama_artifacts"]
        )
        assert (
            model_name in LLM_MODELS
        ), f"Unable to find {model_name} under LLM_MODELS."

        is_llama_model = model_name in _LLAMA_MODELS
        if is_llama_model:
            if not llama_artifacts:
                pytest.skip("missing required arg: --llama_artifacts")
            require_paths(
                {
                    "checkpoint": f"{llama_artifacts}/consolidated.00.pth",
                    "params": f"{llama_artifacts}/params.json",
                    "tokenizer": f"{llama_artifacts}/tokenizer.model",
                }
            )

        prompt = (
            "I would like to learn python, could you teach me with a simple example?"
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--prompt",
            prompt,
            "--temperature",
            "0",
            "--decoder_model",
            model_name,
            "--model_mode",
            "kv",
            "--max_seq_len",
            "1024",
            "--max_context_len",
            "1024",
        ]

        match static_llm_eval_method:
            case "wikitext_ppl":
                cmds.extend(
                    [
                        "--eval_methods",
                        "tasks_eval",
                        "--eval_tasks",
                        "wikitext",
                        "--eval_limit",
                        "1",
                        "--calib_tasks",
                        "wikitext",
                        "--calib_limit",
                        "1",
                    ]
                )
            case "hellaswag_acc_norm":
                cmds.extend(
                    [
                        "--eval_methods",
                        "tasks_eval",
                        "--eval_tasks",
                        "hellaswag",
                        "--eval_limit",
                        "10",
                        "--calib_tasks",
                        "hellaswag",
                        "--calib_limit",
                        "10",
                    ]
                )
            case "sqnr":
                cmds.extend(
                    [
                        "--eval_tasks",
                        "wikitext",
                        "--eval_limit",
                        "1",
                        "--eval_methods",
                        "sqnr_eval",
                        "--calib_tasks",
                        "wikitext",
                        "--calib_limit",
                        "1",
                    ]
                )
            case _:
                logging.warning(
                    "No llm eval method chosen. Only generate model output."
                )
                cmds.extend(["--calib_tasks", "wikitext", "--calib_limit", "1"])

        if is_llama_model:
            cmds.extend(
                [
                    "--checkpoint",
                    f"{llama_artifacts}/consolidated.00.pth",
                    "--params",
                    f"{llama_artifacts}/params.json",
                    "--tokenizer_model",
                    f"{llama_artifacts}/tokenizer.model",
                ]
            )

        if model_name == "gemma4-e2b":
            cmds.extend(["--embedding-quantize", "4,32"])

        add_default_cmds(cmds, qnn_config)

        msg = _check(_run(cmds, qnn_config))
        logging.info(f"Model Name: {model_name}\nTarget Device: {qnn_config.soc_model}")
        logging.info(f"Eval Result: {msg}")

        workload = f"llm:{model_name}"
        assert_metric(workload, "pte_size", msg["pte_size"], qnn_config)

        if not qnn_config.compile_only:
            if static_llm_eval_method:
                metric = {
                    "wikitext_ppl": "wiki_ppl",
                    "hellaswag_acc_norm": "acc_norm",
                    "sqnr": "sqnr",
                }[static_llm_eval_method]
                assert (
                    get_metric_constraint(workload, metric, qnn_config) is not None
                ), f"{model_name} currently does not support {static_llm_eval_method}."
                assert_metric(workload, metric, msg[metric], qnn_config)

            if (
                not qnn_config.enable_x86_64
                and get_metric_constraint(workload, "inference_speed", qnn_config)
                is not None
            ):
                assert_metric(
                    workload, "inference_speed", msg["inference_speed"], qnn_config
                )


class StaticLLMEvalLimit:
    @staticmethod
    def test(request):
        root, artifact, model_name, llama_artifacts, qnn_config = _require(
            request, ["model_name", "llama_artifacts"]
        )
        assert (
            model_name in LLM_MODELS
        ), f"Unable to find {model_name} under LLM_MODELS."

        is_llama_model = model_name in _LLAMA_MODELS
        if is_llama_model:
            if not llama_artifacts:
                pytest.skip("missing required arg: --llama_artifacts")
            require_paths(
                {
                    "checkpoint": f"{llama_artifacts}/consolidated.00.pth",
                    "params": f"{llama_artifacts}/params.json",
                    "tokenizer": f"{llama_artifacts}/tokenizer.model",
                }
            )

        command_config = replace(qnn_config, compile_only=False, pre_gen_pte=None)

        def run_llama(extra_cmds: List[str]):
            cmds = [
                "python",
                f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
                "--artifact",
                artifact,
                "--build_folder",
                qnn_config.build_folder,
                "--prompt",
                "I would like to learn python, could you teach me with a simple example?",
                "--temperature",
                "0",
                "--decoder_model",
                model_name,
                "--model_mode",
                "kv",
                "--max_seq_len",
                "1024",
                "--max_context_len",
                "1024",
            ]
            cmds.extend(extra_cmds)
            add_default_cmds(cmds, command_config)

            msg = _check(_run(cmds, qnn_config))
            return msg

        extra_cmds = [
            "--compile_only",
            "--calib_tasks",
            "wikitext",
            "--calib_limit",
            "1",
        ]
        extra_cmds_for_llama = [
            "--checkpoint",
            f"{llama_artifacts}/consolidated.00.pth",
            "--params",
            f"{llama_artifacts}/params.json",
            "--tokenizer_model",
            f"{llama_artifacts}/tokenizer.model",
        ]
        if is_llama_model:
            extra_cmds.extend(extra_cmds_for_llama)

        if model_name == "gemma4-e2b":
            extra_cmds.extend(["--embedding-quantize", "4,32"])

        run_llama(extra_cmds)

        def eval_ppl(eval_limit: int) -> float:
            extra = [
                "--pre_gen_pte",
                artifact,
                "--eval_methods",
                "tasks_eval",
                "--eval_tasks",
                "wikitext",
                "--eval_limit",
                str(eval_limit),
            ]
            if is_llama_model:
                extra.extend(extra_cmds_for_llama)
            if qnn_config.device:
                extra.extend(["--device", qnn_config.device])
            return run_llama(extra)["wiki_ppl"]

        ppl_1 = eval_ppl(1)
        ppl_3 = eval_ppl(3)
        logging.info(f"wiki_ppl: eval_limit=1: {ppl_1}, eval_limit=3: {ppl_3}")
        assert ppl_1 != ppl_3


class Codegen2:
    @staticmethod
    def test(request):
        root, artifact, qnn_config = _require(request)
        prompt = "def hello_world():"
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--prompt",
            prompt,
            "--temperature",
            "0",
            "--decoder_model",
            "codegen2_1b",
            "--model_mode",
            "kv",
            "--max_seq_len",
            "128",
            "--max_context_len",
            "128",
            "--calib_tasks",
            "wikitext",
            "--calib_limit",
            "1",
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        golden_start_with = "def hello_world():"
        if not qnn_config.compile_only:
            assert msg["result"][0].startswith(
                golden_start_with
            ), f"Expected Output: {golden_start_with}. Actual Output: {msg['result'][0]}"
        if not qnn_config.enable_x86_64:
            assert_metric("codegen2", "pte_size", msg["pte_size"], qnn_config)
        if not qnn_config.compile_only and not qnn_config.enable_x86_64:
            assert_metric(
                "codegen2", "inference_speed", msg["inference_speed"], qnn_config
            )


class LlamaStories260K:
    @staticmethod
    def test(request):
        root, artifact, llama_artifacts, qnn_config = _require(
            request, ["llama_artifacts"]
        )
        require_paths(
            {
                "checkpoint": f"{llama_artifacts}/stories260K.pt",
                "params": f"{llama_artifacts}/params.json",
                "tokenizer_model": f"{llama_artifacts}/tokenizer.model",
                "tokenizer_bin": f"{llama_artifacts}/tokenizer.bin",
            }
        )
        prompt = "Once"
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--checkpoint",
            f"{llama_artifacts}/stories260K.pt",
            "--params",
            f"{llama_artifacts}/params.json",
            "--tokenizer_model",
            f"{llama_artifacts}/tokenizer.model",
            "--tokenizer_bin",
            f"{llama_artifacts}/tokenizer.bin",
            "--prompt",
            prompt,
            "--temperature",
            "0",
            "--decoder_model",
            "stories260k",
            "--model_mode",
            "hybrid",
            "--prefill_ar_len",
            "32",
            "--max_seq_len",
            "128",
            "--max_context_len",
            "128",
            "--calib_tasks",
            "wikitext",
            "--calib_limit",
            "1",
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        golden_start_with = "Once upon a time,"
        if not qnn_config.compile_only:
            assert msg["result"][0].startswith(
                golden_start_with
            ), f"Expected Output: {golden_start_with}. Actual Output: {msg['result'][0]}"
        if not qnn_config.enable_x86_64:
            assert_metric("llama_stories_260k", "pte_size", msg["pte_size"], qnn_config)
        if not qnn_config.compile_only and not qnn_config.enable_x86_64:
            assert_metric(
                "llama_stories_260k",
                "inference_speed",
                msg["inference_speed"],
                qnn_config,
            )


class LlamaStories110M:
    @staticmethod
    def test(request):
        root, artifact, llama_artifacts, use_fp16, qnn_config = _require(
            request, ["llama_artifacts", "use_fp16"]
        )
        require_paths(
            {
                "checkpoint": f"{llama_artifacts}/stories110M.pt",
                "params": f"{llama_artifacts}/params.json",
                "tokenizer_model": f"{llama_artifacts}/tokenizer.model",
                "tokenizer_bin": f"{llama_artifacts}/tokenizer.bin",
            }
        )
        prompt = "Once"
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--checkpoint",
            f"{llama_artifacts}/stories110M.pt",
            "--params",
            f"{llama_artifacts}/params.json",
            "--tokenizer_model",
            f"{llama_artifacts}/tokenizer.model",
            "--tokenizer_bin",
            f"{llama_artifacts}/tokenizer.bin",
            "--prompt",
            prompt,
            "--temperature",
            "0",
            "--decoder_model",
            "stories110m",
            "--model_mode",
            "hybrid",
            "--prefill_ar_len",
            "32",
            "--max_seq_len",
            "128",
            "--max_context_len",
            "128",
            "--calib_tasks",
            "wikitext",
            "--calib_limit",
            "1",
        ]
        if use_fp16:
            cmds.append("--use_fp16")
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        golden_start_with = "Once upon a time,"
        if not qnn_config.compile_only:
            assert msg["result"][0].startswith(
                golden_start_with
            ), f"Expected Output: {golden_start_with}. Actual Output: {msg['result'][0]}"
        workload = "llama_stories_110m:fp16" if use_fp16 else "llama_stories_110m:int8"
        if not qnn_config.enable_x86_64:
            assert_metric(workload, "pte_size", msg["pte_size"], qnn_config)
        if (
            not qnn_config.compile_only
            and not qnn_config.enable_x86_64
            and get_metric_constraint(workload, "inference_speed", qnn_config)
            is not None
        ):
            assert_metric(
                workload, "inference_speed", msg["inference_speed"], qnn_config
            )


class AttentionSink:
    @staticmethod
    def test(request):
        root, artifact, qnn_config = _require(request)
        model_name = "smollm2_135m"
        prompt = (
            "I would like to learn python, could you teach me with a simple example?"
        )
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--prompt",
            prompt,
            "--temperature",
            "0",
            "--decoder_model",
            model_name,
            "--model_mode",
            "kv",
            "--max_seq_len",
            "2048",
            "--max_context_len",
            "1024",
            "--eval_methods",
            "tasks_eval",
            "--eval_tasks",
            "wikitext",
            "--eval_limit",
            "1",
            "--calib_tasks",
            "wikitext",
            "--calib_limit",
            "1",
            "--use_attention_sink",
            "4,32",
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        if not qnn_config.compile_only:
            assert_metric(
                "attention_sink",
                "attention_sink_evictor_pte_size",
                msg["attention_sink_evictor_pte_size"],
                qnn_config,
            )
            assert_metric(f"llm:{model_name}", "wiki_ppl", msg["wiki_ppl"], qnn_config)


class HFCausalLM:
    @staticmethod
    def test(request):
        root, artifact, qnn_config = _require(request)
        prompt = "My favourite condiment is "
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/hf_causal_lm.py",
            "--prompt",
            prompt,
            "--decoder_model",
            "qwen2_5-0_5b",
            "--ptq",
            "16a8w",
            "--enable_spinquant_r3",
            "--max_seq_len",
            "128",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
        ]
        add_default_cmds(cmds, qnn_config)
        msg = _check(_run(cmds, qnn_config))
        golden_start_with = "My favourite condiment is iced tea."
        if not qnn_config.compile_only:
            assert msg["result"][0].startswith(
                golden_start_with
            ), f"Expected Output: '{golden_start_with}' Actual Output: '{msg['result'][0]}'"


class StaticLLMQAT:
    @staticmethod
    def test(request):
        root, artifact, qnn_config = _require(request)
        if qnn_config.compile_only:
            pytest.skip("tasks_eval requires on-device inference")

        def run_eval(
            calib_limit: int, train_limit: int, extra_args: List[str] = None
        ) -> float:
            prompt = "I would like to learn python, could you teach me with a simple example?"
            cmds = [
                "python",
                f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
                "--artifact",
                artifact,
                "--build_folder",
                qnn_config.build_folder,
                "--prompt",
                prompt,
                "--temperature",
                "0",
                "--decoder_model",
                "smollm2_135m",
                "--model_mode",
                "kv",
                "--max_seq_len",
                "1024",
                "--max_context_len",
                "1024",
                "--eval_methods",
                "tasks_eval",
                "--eval_tasks",
                "wikitext",
                "--eval_limit",
                "1",
                "--qat",
                "--calib_tasks",
                "wikitext",
                "--calib_limit",
                str(calib_limit),
                "--train_tasks",
                "wikitext",
                "--train_limit",
                str(train_limit),
            ]
            if extra_args:
                cmds.extend(extra_args)
            add_default_cmds(cmds, qnn_config)
            msg = _check(_run(cmds, qnn_config))
            return msg["wiki_ppl"]

        ptq_ppl = run_eval(
            calib_limit=1, train_limit=1, extra_args=["--freeze_all_params"]
        )
        qat_ppl = run_eval(calib_limit=1, train_limit=1)
        logging.info(f"QAT PPL={qat_ppl:.2f}")
        logging.info(f"PTQ PPL={ptq_ppl:.2f}")
        assert (
            qat_ppl < ptq_ppl
        ), f"Expected QAT PPL ({qat_ppl:.2f}) < PTQ PPL({ptq_ppl:.2f})"


# ---------------------------------------------------------------------------
# TestExampleMultimodalityScript
# ---------------------------------------------------------------------------


class StaticASR:
    @staticmethod
    def test(request):
        root, artifact, model_name, qnn_config = _require(request, ["model_name"])

        if qnn_config.enable_x86_64:
            pytest.skip(
                "Skipping the check for the static ASR model on x86 due to long execution time."
            )

        alm_specs: ALMSpecs = ALM_SPECS[model_name]
        prompt = "can you transcribe the speech into a written format?"
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--prompt",
            prompt,
            "--audio_path",
            alm_specs.audio_path,
            "--temperature",
            "0",
            "--decoder_model",
            model_name,
            "--model_mode",
            "kv",
            "--max_seq_len",
            str(alm_specs.max_seq_len),
            "--calib_samples",
            "./examples/qualcomm/oss_scripts/llama/assets/samples/audio.json",
        ]
        add_default_cmds(cmds, qnn_config)

        msg = _check(_run(cmds, qnn_config))

        if not qnn_config.compile_only:
            model_out = msg["result"][0]
            assert alm_specs.golden_audio_feature in model_out.lower(), (
                f"Expected Output contains feature: '{alm_specs.golden_audio_feature}' "
                f"Actual Output: '{model_out}'"
            )

        workload = f"asr:{model_name}"
        assert_metric(
            workload,
            "audio_encoder_pte_size",
            msg["audio_encoder_pte_size"],
            qnn_config,
        )
        assert_metric(
            workload,
            "tok_embedding_pte_size",
            msg["tok_embedding_pte_size"],
            qnn_config,
        )
        assert_metric(workload, "pte_size", msg["pte_size"], qnn_config)
        if (
            not qnn_config.compile_only
            and get_metric_constraint(workload, "inference_speed", qnn_config)
            is not None
        ):
            assert_metric(
                workload, "inference_speed", msg["inference_speed"], qnn_config
            )


class StaticVLM:
    @staticmethod
    def test(request):
        root, artifact, model_name, qnn_config = _require(request, ["model_name"])

        vlm_specs: VLMSpecs = VLM_SPECS[model_name]
        prompt = "Can you describe this image?"
        cmds = [
            "python",
            f"{root}/examples/qualcomm/oss_scripts/llama/llama.py",
            "--artifact",
            artifact,
            "--build_folder",
            qnn_config.build_folder,
            "--prompt",
            prompt,
            "--image_path",
            vlm_specs.image_path,
            "--temperature",
            "0",
            "--decoder_model",
            model_name,
            "--model_mode",
            "kv",
            "--max_seq_len",
            str(vlm_specs.max_seq_len),
            "--calib_samples",
            "./examples/qualcomm/oss_scripts/llama/assets/samples/vision.json",
        ]
        add_default_cmds(cmds, qnn_config)

        msg = _check(_run(cmds, qnn_config))

        if not qnn_config.compile_only:
            model_out = msg["result"][0]
            assert vlm_specs.golden_image_feature in model_out, (
                f"Expected Output contains feature: '{vlm_specs.golden_image_feature}' "
                f"Actual Output: '{model_out}'"
            )

        workload = f"vlm:{model_name}"
        if not qnn_config.enable_x86_64:
            assert_metric(
                workload,
                "vision_encoder_pte_size",
                msg["vision_encoder_pte_size"],
                qnn_config,
            )
            assert_metric(
                workload,
                "tok_embedding_pte_size",
                msg["tok_embedding_pte_size"],
                qnn_config,
            )
            assert_metric(workload, "pte_size", msg["pte_size"], qnn_config)
        if (
            not qnn_config.compile_only
            and not qnn_config.enable_x86_64
            and get_metric_constraint(workload, "inference_speed", qnn_config)
            is not None
        ):
            assert_metric(
                workload, "inference_speed", msg["inference_speed"], qnn_config
            )
