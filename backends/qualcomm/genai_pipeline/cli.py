# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GenAI Pipeline CLI entry point.

Provides a command-line interface for running the GenAI Pipeline
with LLM models. Replaces the need to understand llama.py's internal
structure — users specify model, SoC, and prompt, and the pipeline
handles the rest.

Usage:
    python -m backends.qualcomm.genai_pipeline.cli \\
        --model llama3_2-1b_instruct \\
        --soc SM8750 \\
        --prompt "Hello, world!" \\
        --artifact-dir ./output

    # Compile only (no device inference):
    python -m backends.qualcomm.genai_pipeline.cli \\
        --model llama3_2-1b_instruct \\
        --soc SM8750 \\
        --compile-only

    # List supported models:
    python -m backends.qualcomm.genai_pipeline.cli --list-models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from executorch.backends.qualcomm.genai_pipeline.configs.inference_output_config import (
    InferenceOutputConfig,
)
from executorch.backends.qualcomm.genai_pipeline.control_args import (
    ControlArgs,
    DEFAULT_ARTIFACT,
    DEFAULT_BACKEND,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CALIB_HF_LIMIT,
    DEFAULT_CALIB_LIMIT,
    DEFAULT_DTYPE_OVERRIDE,
    DEFAULT_EVAL_LIMIT,
    DEFAULT_EVAL_METHOD,
    DEFAULT_GCAP,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MODEL_MODE,
    DEFAULT_NGRAM,
    DEFAULT_PREFILL_AR_LEN,
    DEFAULT_TARGET,
    DEFAULT_TEMPERATURE,
    DEFAULT_WINDOW,
    PROMPT_EVAL,
    SQNR_EVAL,
    TASKS_EVAL,
)
from executorch.backends.qualcomm.genai_pipeline.stages import (
    CompilationStage,
    InferenceStage,
    ModelPreparationStage,
    QuantizationStage,
)

logger = logging.getLogger(__name__)

PROGRAM_NAME = "genai_pipeline"
PROGRAM_DESCRIPTION = (
    "GenAI Pipeline CLI — standardized LLM deployment on Qualcomm platforms"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=PROGRAM_DESCRIPTION,
    )

    # --- Model selection ---
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., 'llama3_2-1b_instruct', 'qwen2_5-0_5b'). "
        "Use --list-models to see available options.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all supported models and exit.",
    )

    # --- Model identification and input paths ---
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load the weights from. Required for the models whose "
        "registry row carries no local weights.",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Params JSON describing the model's shapes. Required for the models "
        "whose registry row carries no params file (the llama family).",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help="Tokenizer model to load instead of the model's own.",
    )
    parser.add_argument(
        "--tokenizer-bin",
        type=str,
        default=None,
        help="Tokenizer binary, for Llama2-era models that ship one.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=DEFAULT_ARTIFACT,
        help=f"Directory for compiled artifacts (default: {DEFAULT_ARTIFACT}).",
    )

    # --- Quantization ---
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use FP16 precision (skip quantization).",
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Enable Quantization-Aware Training (QAT). If not set, defaults to PTQ.",
    )
    parser.add_argument(
        "--embedding-quantize",
        type=str,
        default=None,
        help="Fall back to the CPU embedding operator and quantize it, as "
        "'<bitwidth>,<groupsize>' -- e.g. '4,32'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for decoder quantization. Larger values raise "
        f"throughput at the cost of host memory, and affect only the "
        f"calibration graph (default: {DEFAULT_BATCH_SIZE}).",
    )

    # --- Calibration data selection ---
    parser.add_argument(
        "--calib-tasks",
        type=str,
        nargs="+",
        default=None,
        help="lm-eval tasks to draw calibration samples from, e.g. "
        "--calib-tasks wikitext.",
    )
    parser.add_argument(
        "--calib-samples",
        type=str,
        nargs="+",
        default=None,
        help="One or more JSON files of calibration samples, each a flat list of "
        "objects with a 'messages' list (and 'files' for multimodal models). "
        "Multiple files are merged.",
    )
    parser.add_argument(
        "--calib-limit",
        type=int,
        default=DEFAULT_CALIB_LIMIT,
        help=f"How many samples to calibrate on (default: {DEFAULT_CALIB_LIMIT}).",
    )
    parser.add_argument(
        "--calib-num-fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of few-shot examples in each calibration sample.",
    )
    parser.add_argument(
        "--calib-hf-dataset",
        type=str,
        default=None,
        help="HuggingFace chat dataset for additional calibration data "
        "(e.g. 'HuggingFaceTB/smol-smoltalk').",
    )
    parser.add_argument(
        "--calib-hf-limit",
        type=int,
        default=DEFAULT_CALIB_HF_LIMIT,
        help="Number of samples to load from --calib-hf-dataset "
        f"(default: {DEFAULT_CALIB_HF_LIMIT}).",
    )

    # --- Evaluation ---
    parser.add_argument(
        "--eval-methods",
        choices=[PROMPT_EVAL, TASKS_EVAL, SQNR_EVAL],
        type=str,
        nargs="+",
        default=[DEFAULT_EVAL_METHOD],
        help=f"Choose eval methods(default: {DEFAULT_EVAL_METHOD}). Users can provide more than 1 eval methods. For example: --eval-methods tasks_eval sqnr_eval. "
        "Following eval methods are supported: "
        "1) prompt_eval: Model will generate the output response based on the provided prompt through the flag --prompt. "
        "2) tasks_eval: This will eval the tasks provided through the flag --eval-tasks. "
        "3) sqnr_eval: This will eval the sqnr between QNN's output logit V.S. Static Llama nn.Module's output logit. Eval is based on the provided prompt through the --prompt flag. Please note that sqnr will only eval the prompt's logit but not the new generated token's logit.",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        nargs="+",
        default=None,
        help="lm-eval tasks to evaluate on, e.g. --eval-tasks wikitext.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=DEFAULT_EVAL_LIMIT,
        help=f"How many samples to evaluate on (default: {DEFAULT_EVAL_LIMIT}).",
    )
    parser.add_argument(
        "--eval-num-fewshot",
        type=int,
        default=None,
        metavar="N",
        help="Number of few-shot examples in each evaluation sample.",
    )

    # --- Backend and SoC selection ---
    parser.add_argument(
        "--soc",
        type=str,
        default=None,
        help="Target SoC model.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=["htp", "gpu"],
        help=f"QNN backend type (default: {DEFAULT_BACKEND}).",
    )

    # --- Runtime prompts, multimodal inputs and runtime tokenizer path ---
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=["Hello, how are you?"],
        help="User prompt(s) for text generation.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="System prompt for models that support one.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature for text generation (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        nargs="+",
        default=[],
        help="Audio file(s) for multimodal (ALM) models.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        nargs="+",
        default=[],
        help="Image file(s) for multimodal (VLM) models.",
    )
    parser.add_argument(
        "--runtime-tokenizer-path",
        type=str,
        default=None,
        help="Path to the runtime tokenizer.",
    )

    # Graph shapes and modes
    parser.add_argument(
        "--model-mode",
        type=str,
        default=DEFAULT_MODEL_MODE,
        choices=["kv", "hybrid", "lookahead"],
        help=f"Decoder model mode (default: {DEFAULT_MODEL_MODE}).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_SEQ_LEN}).",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help=(
            "Maximum context length. Overrides --max-seq-len until GenAI Pipeline "
            f"supports attention sink (default: {DEFAULT_MAX_SEQ_LEN})."
        ),
    )
    parser.add_argument(
        "--prefill-ar-len",
        type=int,
        default=DEFAULT_PREFILL_AR_LEN,
        help=f"Prefill auto regressive length (default: {DEFAULT_PREFILL_AR_LEN}).",
    )
    parser.add_argument(
        "--dtype-override",
        type=str,
        default=DEFAULT_DTYPE_OVERRIDE,
        choices=["fp32", "fp16"],
        help=f"Override the dtype the model is loaded at "
        f"(default: {DEFAULT_DTYPE_OVERRIDE}).",
    )

    # --- Lookahead decoding shape parameters (--model-mode lookahead) ---
    parser.add_argument(
        "--ngram",
        type=int,
        default=DEFAULT_NGRAM,
        help=f"(lookahead) N-gram size (default: {DEFAULT_NGRAM}).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help=f"(lookahead) Window size (default: {DEFAULT_WINDOW}).",
    )
    parser.add_argument(
        "--gcap",
        type=int,
        default=DEFAULT_GCAP,
        help=f"(lookahead) Guess capacity (default: {DEFAULT_GCAP}).",
    )

    # --- Flow control ---
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the model (skip inference).",
    )
    parser.add_argument(
        "--pre-gen-pte",
        type=str,
        default=None,
        help="Directory of .pte artifacts from a previous genai_pipeline "
        "compilation; legacy llama.py artifacts are not recognized.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    # --- Device ---
    parser.add_argument(
        "--build-folder",
        type=str,
        default=None,
        help="path to cmake binary directory for target platform, e.g., /path/to/build-android",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Target platform for deployment, one of export_utils.SUPPORTED_TARGETS",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="hostname where android device is connected.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="serial number for android device communicated via ADB.",
    )
    parser.add_argument(
        "--enable-x86-64",
        action="store_true",
        help="Run with the x86 QNN runner.",
    )
    parser.add_argument(
        "--online-prepare",
        action="store_true",
        help="If specified, compose QNN graph on device.",
    )

    # --- Features ---
    # Long context feature: Attention Sink
    parser.add_argument(
        "--use-attention-sink",
        default=None,
        type=str,
        help="Use the attention sink feature to have fluent multi-round conversations. Specify the settings as '<sink_size>,<batch_eviction_size>', for example, '4,32'."
        "This setting is for compilation. Once you compile with a chosen <sink_size> and <batch_eviction_size>, they cannot be changed at runtime. If you need to update them, you can recompile the attention sink module along with llama.py.",
    )

    return parser


def list_models() -> None:
    """Log all supported models and exit."""
    from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
        get_supported_models,
    )

    models = get_supported_models()
    logger.info("Supported models (%d):", len(models))
    for model in models:
        logger.info("  - %s", model)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments for consistency and constraints.

    Args:
        args: Parsed CLI arguments.

    Raises:
        ValueError: If arguments violate constraints.
    """
    # Imported here rather than as --target choices: export_utils loads the QNN
    # Python bindings, which building the parser should not need.
    from executorch.backends.qualcomm.export_utils import SUPPORTED_TARGETS

    if not args.soc:
        raise ValueError("--soc is required to run the GenAI Pipeline.")

    if args.target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"--target {args.target} is not supported; choose from {SUPPORTED_TARGETS}"
        )

    # TODO: The legacy path already supports the QAT feature,
    # but it is not yet introduced in GenAI Pipeline.
    if args.qat:
        raise ValueError("QAT is not yet supported.")

    # TODO: The legacy path supports embedding quantization, but GenAI Pipeline
    # does not yet support it.
    if args.embedding_quantize is not None:
        raise ValueError(
            "`embedding_quantize` is currently unsupported in GenAI Pipeline."
        )

    # TODO: The legacy path already supports the attention sink feature,
    # but it is not yet introduced in GenAI Pipeline.
    if args.use_attention_sink is not None:
        raise ValueError(
            "`use_attention_sink` is currently unsupported in GenAI Pipeline."
        )

    # Mirrors the legacy flow's restriction.
    if args.model_mode == "hybrid" and args.online_prepare:
        raise ValueError("Currently hybrid mode is not compatible with online_prepare.")

    # As in the legacy flow: with both, every stage is skipped and nothing runs.
    if args.compile_only and args.pre_gen_pte:
        raise ValueError("Cannot set both compile_only and pre_gen_pte as true")

    _validate_eval_args(args)
    _validate_multimodal_args(args)

    # Pre-generated PTEs skip quantization, as in the legacy flow.
    if not (args.use_fp16 or args.pre_gen_pte) and not (
        args.calib_tasks or args.calib_samples or args.calib_hf_dataset
    ):
        raise ValueError(
            "Quantization requires calibration data. Provide --calib-tasks, "
            "--calib-samples, or --calib-hf-dataset."
        )


def _validate_eval_args(args: argparse.Namespace) -> None:
    """Validate the evaluation methods and the arguments each one needs.

    Repeated methods are dropped in place, keeping their first occurrence.

    Args:
        args: Parsed CLI arguments.

    Raises:
        ValueError: If a requested method lacks an argument it needs, or
            tasks_eval or sqnr_eval is requested for a mode other than kv or
            hybrid.
    """
    # Each method runs once, as in the legacy flow: a repeat would overwrite its
    # result and share its output directory.
    eval_methods = list(dict.fromkeys(args.eval_methods))
    if eval_methods != args.eval_methods:
        logger.warning("Ignoring repeated --eval-methods: %s", args.eval_methods)
        args.eval_methods = eval_methods

    # The runner evaluates logits in KV mode only, as the legacy flow intends:
    # a lookahead PTE's kv_forward does not decode one token at a time.
    for method in (TASKS_EVAL, SQNR_EVAL):
        if method not in args.eval_methods:
            continue
        if args.model_mode not in {"kv", "hybrid"}:
            raise ValueError(
                f"{method} is only supported for KV mode. Hybrid mode will only "
                "use KV mode when evaluating tasks/sqnr."
            )
        if args.model_mode != "kv":
            logger.warning(
                "%s runs its forwards in kv mode; other methods keep %s",
                method,
                args.model_mode,
            )

    if TASKS_EVAL in args.eval_methods:
        if not args.eval_tasks:
            raise ValueError(
                "--eval-tasks is required when --eval-methods includes tasks_eval"
            )

    if SQNR_EVAL in args.eval_methods:
        # As in the legacy flow, which scores the first prompt.
        if len(args.prompt) > 1:
            logger.warning(
                "sqnr_eval scores only the first of %d --prompt values",
                len(args.prompt),
            )
        if args.batch_size != 1:
            raise ValueError("sqnr_eval supports only --batch-size 1")


def _validate_multimodal_args(args: argparse.Namespace) -> None:
    """Validate the media inputs against whether and how the model is multimodal.

    A multimodal model's batch size is forced to 1 in place, as multi-batch is
    not supported for it yet.

    Args:
        args: Parsed CLI arguments.

    Raises:
        ValueError: If media is given to a text model, or does not fit the
            multimodal model's modality.
    """
    from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
        is_multimodal,
        validate_modality_inputs,
    )

    if not is_multimodal(args.model):
        if args.audio_path or args.image_path:
            raise ValueError("--audio-path and --image-path require a multimodal model")
        return

    if args.batch_size != 1:
        logger.warning(
            "Multi-batch is not supported for multimodal LLMs yet; "
            "forcing batch_size from %d to 1.",
            args.batch_size,
        )
        args.batch_size = 1

    # Media inputs matter only when inference runs.
    if not args.compile_only:
        validate_modality_inputs(args.model, args.image_path, args.audio_path)


def _skips_model_preparation(args: argparse.Namespace) -> bool:
    """Whether model preparation is skipped for pre-generated PTEs.

    tasks_eval needs the tokenizer and sqnr_eval the FP32 model, both from model
    preparation, so they keep it even with pre-generated PTEs.

    Args:
        args: Parsed CLI arguments.

    Returns:
        ``True`` when ``--pre-gen-pte`` is given and no evaluation method needs
        model preparation.
    """
    return bool(args.pre_gen_pte) and not {TASKS_EVAL, SQNR_EVAL} & set(
        args.eval_methods
    )


def _load_runtime_tokenizer_path(args: argparse.Namespace, model_config: Any) -> str:
    """Resolve the runtime tokenizer as model preparation's ``TokenizerWrapper`` does.

    Used when model preparation is skipped, so the runner still gets the same
    tokenizer file -- the ``.bin`` or tiktoken file the user supplied, or the
    HF ``tokenizer.json`` saved into the artifact directory -- rather than a
    guessed path.

    Args:
        args: Parsed CLI arguments.
        model_config: The model's registry config.

    Returns:
        The runtime tokenizer path.
    """
    from executorch.examples.qualcomm.oss_scripts.llama.tokenizer import (
        TokenizerWrapper,
    )

    # A copy carrying TokenizerWrapper's legacy field names, so ``args`` stays
    # untouched.
    control_args = replace(
        ControlArgs.from_namespace(args),
        decoder_model=args.model,
        artifact=args.artifact_dir,
    )
    return TokenizerWrapper(control_args, model_config).runtime_tokenizer_path


def _prepare_configs(args: argparse.Namespace) -> dict:
    """Prepare model configs, transforms, and dataset options.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Dict with keys: model_config, model_arch, weight_transforms,
        module_transforms, dataset_options, model_options, quantize_options.
    """
    from executorch.backends.qualcomm.genai_pipeline.datasets.dataset_options import (
        DatasetOptions,
    )
    from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
        get_modality,
        get_model_arch,
        get_model_config,
        get_model_layout,
        get_model_num_sharding,
        get_quant_dtype,
        get_quant_recipe,
        get_source_transform,
        get_state_dict_loader,
        is_multimodal,
    )

    # TODO: Separate sequence and context lengths once attention sink is
    # supported in GenAI Pipeline. Until then, use one resolved graph shape.
    if args.max_context_len is None:
        args.max_context_len = args.max_seq_len
    else:
        args.max_seq_len = args.max_context_len

    model_config = get_model_config(args.model)
    logger.info("Model config loaded")

    model_arch = get_model_arch(args.model, args)
    logger.info("Model arch loaded")

    weight_transforms, module_transforms = get_source_transform(
        args.model,
        control_args=args,
    )
    logger.info("Loaded source transforms for model '%s'", args.model)

    num_sharding = get_model_num_sharding(args.model)
    state_dict_loaders = get_state_dict_loader(
        args.model,
        control_args=args,
    )
    model_options = {
        "model_arch": model_arch,
        "weight_transforms": weight_transforms,
        "module_transforms": module_transforms,
        "embedding_quantize": args.embedding_quantize,
        "num_shardings": num_sharding,
        "model_layout": get_model_layout(args.model),
        "state_dict_loader": state_dict_loaders,
    }

    quantize_options = {
        "quant_dtype": get_quant_dtype(args.model),
        "quant_recipe": get_quant_recipe(args.model),
    }
    # Multi-image conversations keep the encoder in FP16, as the legacy flow
    # does: quantization makes the images harder for the model to tell apart.
    if is_multimodal(args.model) and len(args.image_path) > 1:
        quantize_options["skip_quantize"] = [get_modality(args.model)]

    dataset_options = replace(
        DatasetOptions.from_namespace(args),
        llm_config=model_config,
    )

    return {
        "dataset_options": dataset_options,
        "model_options": model_options,
        "quantize_options": quantize_options,
        # An explicit path wins; without model preparation, resolve it here.
        "runtime_tokenizer_path": (
            args.runtime_tokenizer_path
            or (
                _load_runtime_tokenizer_path(args, model_config)
                if _skips_model_preparation(args)
                else None
            )
        ),
    }


def get_model_preparation_stage(
    args: argparse.Namespace,
) -> Optional["ModelPreparationStage"]:
    """Construct the model preparation stage.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The model preparation stage, or ``None`` when pre-generated PTEs make it
        unnecessary; see ``_skips_model_preparation``.
    """
    if _skips_model_preparation(args):
        logger.info("Skipping model preparation: using pre-generated PTE artifacts")
        return None

    from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
        get_model_loader_adapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.executorch_model_preparation_strategy import (
        ExecuTorchModelPreparationStrategy,
    )

    model_loader_adapter = get_model_loader_adapter(args.model, args)
    model_preparation_stage = ModelPreparationStage(
        ExecuTorchModelPreparationStrategy(
            model_loader_adapter=model_loader_adapter,
            # Reports TokenizerWrapper's runtime tokenizer for the runner.
            export_tokenizer=True,
        )
    )
    logger.info("Created model preparation stage for model '%s'", args.model)
    return model_preparation_stage


def get_quantization_stage(
    args: argparse.Namespace,
    dataset_options: Dict[str, Any],
    quantize_options: Dict[str, Any],
) -> Optional["QuantizationStage"]:
    """Construct the quantization stage.

    Args:
        args: Parsed CLI arguments.
        dataset_options: Calibration dataset options.
        quantize_options: Per-component quantization dtype and recipes.

    Returns:
        The quantization stage, or ``None`` with pre-generated PTEs or FP16.
    """
    if args.pre_gen_pte:
        logger.info("Skipping quantization: using pre-generated PTE artifacts")
        return None
    if args.use_fp16:
        logger.info("Skipping quantization: FP16 inference requested")
        return None

    from executorch.backends.qualcomm.genai_pipeline.datasets import (
        get_calibration_dataset_adapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
        get_quantizer_adapter,
        is_multimodal,
    )
    from executorch.backends.qualcomm.genai_pipeline.strategies.quantization.executorch_quantization_strategy import (
        ExecuTorchQuantizationStrategy,
    )

    model_is_multimodal = is_multimodal(args.model)

    calibration_data_adapter = get_calibration_dataset_adapter(
        dataset_options, is_multimodal=model_is_multimodal
    )
    quantizer_adapter = get_quantizer_adapter(args.model)
    recipe_names = {
        component: (
            recipe.__name__ if isinstance(recipe, type) else type(recipe).__name__
        )
        for component, recipe in quantize_options["quant_recipe"].items()
        if recipe is not None
    }
    logger.info(
        "Quantizer adapter created with recipes: %s",
        recipe_names or "None",
    )
    quantization_stage = QuantizationStage(
        ExecuTorchQuantizationStrategy(
            quantizer_adapter=quantizer_adapter,
            calibration_data_adapter=calibration_data_adapter,
        )
    )
    return quantization_stage


def get_compilation_stage(
    args: argparse.Namespace,
) -> Optional[CompilationStage]:
    """Construct the compilation stage with the default compiler adapter.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The compilation stage, or ``None`` with pre-generated PTEs.
    """
    if args.pre_gen_pte:
        logger.info("Skipping compilation: using pre-generated PTE artifacts")
        return None

    from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.default_compiler_adapter import (
        DefaultCompilerAdapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.strategies.compilation.executorch_compilation_strategy import (
        ExecuTorchCompilationStrategy,
    )

    compilation_stage = CompilationStage(
        ExecuTorchCompilationStrategy(compiler_adapter=DefaultCompilerAdapter())
    )
    logger.info("Created compilation stage for model '%s'", args.model)
    return compilation_stage


def get_inference_stage(args: argparse.Namespace) -> Optional[InferenceStage]:
    """Construct the inference stage from the model and evaluation-method lookups.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The inference stage, or ``None`` when only compiling.
    """
    if args.compile_only:
        logger.info("Skipping inference: compile-only requested")
        return None

    from executorch.backends.qualcomm.export_utils import QnnConfig
    from executorch.backends.qualcomm.genai_pipeline.evaluation.evaluator_lookup import (
        get_evaluators,
    )
    from executorch.backends.qualcomm.genai_pipeline.inference.platforms import (
        get_runner_platform,
    )
    from executorch.backends.qualcomm.genai_pipeline.model_lookup import is_multimodal
    from executorch.backends.qualcomm.genai_pipeline.strategies.inference.executorch_inference_strategy import (
        ExecuTorchInferenceStrategy,
    )
    from executorch.backends.qualcomm.genai_pipeline.strategies.inference.llm_runner_adapter import (
        LLMRunnerAdapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.strategies.inference.mllm_runner_adapter import (
        MLLMRunnerAdapter,
    )

    qnn_config = QnnConfig.load_config(
        argparse.Namespace(**vars(args), soc_model=args.soc)
    )
    platform = get_runner_platform(qnn_config, args.enable_x86_64)
    adapter = (
        MLLMRunnerAdapter(platform)
        if is_multimodal(args.model)
        else LLMRunnerAdapter(platform)
    )
    # Checked before compilation rather than when the runner is set up after it.
    runner = Path(qnn_config.build_folder) / adapter.RUNNER
    if not runner.is_file():
        raise FileNotFoundError(
            f"Runner not found: {runner}; build it into --build-folder"
        )

    evaluators = get_evaluators(args.eval_methods, multimodal=is_multimodal(args.model))

    inference_stage = InferenceStage(
        ExecuTorchInferenceStrategy(
            device_runner_adapter=adapter, evaluators=evaluators
        )
    )
    logger.info("Created inference stage for model '%s'", args.model)
    return inference_stage


def run_pipeline(args: argparse.Namespace) -> InferenceOutputConfig:
    """Run the GenAI Pipeline with the given arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The inference results keyed by evaluation method; empty when inference
        is skipped.
    """
    from executorch.backends.qualcomm.genai_pipeline.engine_proxy import EngineProxy
    from executorch.backends.qualcomm.genai_pipeline.genai_pipeline import GenAIPipeline
    from executorch.backends.qualcomm.genai_pipeline.pipeline_context import (
        PipelineContext,
    )
    from executorch.backends.qualcomm.genai_pipeline.pipeline_types import (
        EngineType,
        STAGE_COMPILATION,
        STAGE_INFERENCE,
        STAGE_MODEL_PREPARATION,
        STAGE_QUANTIZATION,
    )
    from executorch.backends.qualcomm.serialization.qc_schema import (
        QnnExecuTorchBackendType,
    )

    logger.info("Args initialized: model=%s, soc=%s", args.model, args.soc)

    # Validate CLI constraints and normalize unsupported model configurations.
    _validate_args(args)

    # Prepare extra options required by each pipeline stage.
    configs = _prepare_configs(args)

    # Build pipeline context
    extra_options = {
        "backend": args.backend,
        # Compilation reads it from the context to compile for the x86 emulator.
        "enable_x86_64": args.enable_x86_64,
        "model_mode": args.model_mode,
        # Compilation reads it from the context to compose the graph on device.
        "online_prepare": args.online_prepare,
        "max_context_len": args.max_context_len,
        "max_seq_len": args.max_seq_len,
        "use_fp16": args.use_fp16,
        "compile_only": args.compile_only,
        "pre_gen_pte": args.pre_gen_pte,
        # The inference strategy prefers it over model preparation's tokenizer.
        "runtime_tokenizer_path": configs["runtime_tokenizer_path"],
        "verbose": args.verbose,
        "model_options": configs["model_options"],
        "quantize_options": configs["quantize_options"],
        "dataset_options": configs["dataset_options"],
        "eval_methods": args.eval_methods,
        "eval_tasks": args.eval_tasks,
        "eval_limit": args.eval_limit,
        "eval_num_fewshot": args.eval_num_fewshot,
        # Runner settings, read through ControlArgs; device and build settings
        # reach the runner through QnnConfig.load_config(args).
        "temperature": args.temperature,
        "system_prompt": args.system_prompt,
        "ngram": args.ngram,
        "window": args.window,
        "gcap": args.gcap,
        "audio_path": args.audio_path,
        "image_path": args.image_path,
    }
    context = (
        PipelineContext.builder()
        .with_model(args.model)
        .with_soc(args.soc)
        .with_prompt(args.prompt)
        .with_artifact_dir(args.artifact_dir)
        .with_extra_options(extra_options)
        .build()
    )
    logger.info(
        "Pipeline context: model=%s, soc=%s", context.model_name, context.soc_model
    )
    logger.info("Model '%s' configured for %s", args.model, args.soc)
    logger.info("Artifact dir: %s", args.artifact_dir)
    logger.info("Mode: %s", args.model_mode)

    # Get pipeline stages
    model_preparation_stage = get_model_preparation_stage(args)
    quantization_stage = get_quantization_stage(
        args,
        dataset_options=configs["dataset_options"],
        quantize_options=configs["quantize_options"],
    )
    compilation_stage = get_compilation_stage(args)
    inference_stage = get_inference_stage(args)

    # Build GenAI Pipeline
    pipeline = GenAIPipeline(
        model_preparation_stage=model_preparation_stage,
        quantization_stage=quantization_stage,
        compilation_stage=compilation_stage,
        inference_stage=inference_stage,
        engine_proxy=EngineProxy(
            stage_engines={
                STAGE_MODEL_PREPARATION: EngineType.EXECUTORCH,
                STAGE_QUANTIZATION: EngineType.EXECUTORCH,
                STAGE_COMPILATION: EngineType.EXECUTORCH,
                STAGE_INFERENCE: EngineType.EXECUTORCH,
            },
            backend_type=QnnExecuTorchBackendType[
                f"k{args.backend.capitalize()}Backend"
            ],
        ),
    )

    result = pipeline.invoke(context)
    return result


def show_inference_results(result: InferenceOutputConfig) -> None:
    """Log results returned by each configured evaluation method.

    Args:
        result: Pipeline inference output with results keyed by evaluation method.
    """
    for method, inference_result in result.results.items():
        logger.info("\n========== %s =========", method)
        if inference_result.generations:
            for output in inference_result.generations:
                logger.info("Output: %s", output.rstrip())
        if inference_result.eval_results:
            logger.info(
                "Evaluation results:\n%s",
                json.dumps(inference_result.eval_results, indent=2, default=str),
            )
        if inference_result.performance_metrics:
            logger.info(
                "Performance metrics: %s",
                inference_result.performance_metrics,
            )


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    Args:
        argv: Command-line arguments. If None, reads from sys.argv.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s %(asctime)s %(name)s] %(message)s",
    )

    # Handle --list-models
    if args.list_models:
        list_models()
        return

    # Validate --model is required
    if not args.model:
        parser.error("--model is required (use --list-models to see options)")
    args.model = args.model.lower()

    # Run pipeline
    try:
        result = run_pipeline(args)
        show_inference_results(result)
        logger.info("Pipeline completed")
    except KeyError as e:
        logger.error("Error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
