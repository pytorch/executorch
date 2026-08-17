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
import logging
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional

from executorch.backends.qualcomm.genai_pipeline.control_args import (
    DEFAULT_ARTIFACT_DIR,
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
    DEFAULT_SOC_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRAIN_HF_LIMIT,
    DEFAULT_TRAIN_LIMIT,
    DEFAULT_TRAIN_VAL_RATIO,
    DEFAULT_WINDOW,
)
from executorch.backends.qualcomm.genai_pipeline.genai_pipeline import QuantizationStage
from executorch.backends.qualcomm.genai_pipeline.stages.model_preparation_stage import (
    ModelPreparationStage,
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
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Directory for compiled artifacts (default: {DEFAULT_ARTIFACT_DIR}).",
    )

    # --- Quantization ---
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use FP16 precision (skip quantization).",
    )
    parser.add_argument(
        "--embedding-quantize",
        type=str,
        default=None,
        help="Fall back to the CPU embedding operator and quantize it, as "
        "'<bitwidth>,<groupsize>' -- e.g. '4,32'.",
    )
    parser.add_argument(
        "--quant-recipe-suggestion",
        action="store_true",
        help="Emit a per-layer mixed-precision recipe suggestion during PTQ.",
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

    # --- Quantization-aware training ---
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Enable Quantization-Aware Training (QAT). If not set, defaults to PTQ.",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="(QAT) YAML file overriding training configuration defaults.",
    )
    parser.add_argument(
        "--lr-config",
        type=str,
        default=None,
        help="(QAT) YAML file configuring optimizer parameter groups.",
    )
    parser.add_argument(
        "--train-tasks",
        type=str,
        nargs="+",
        default=None,
        help="(QAT) lm-eval tasks for training data. Use --calib-tasks to "
        "specify calibration data separately.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=DEFAULT_TRAIN_LIMIT,
        help=f"(QAT) Number of samples for train tasks (default: {DEFAULT_TRAIN_LIMIT}).",
    )
    parser.add_argument(
        "--train-hf-dataset",
        type=str,
        default=None,
        help="(QAT) HuggingFace instruct dataset for training "
        "(e.g. 'HuggingFaceTB/smol-smoltalk').",
    )
    parser.add_argument(
        "--train-hf-limit",
        type=int,
        default=DEFAULT_TRAIN_HF_LIMIT,
        help="(QAT) Number of samples to load from --train-hf-dataset "
        f"(default: {DEFAULT_TRAIN_HF_LIMIT}).",
    )
    parser.add_argument(
        "--train-val-ratio",
        type=float,
        default=DEFAULT_TRAIN_VAL_RATIO,
        help="(QAT) Fraction of non-calib samples used for training; the "
        "remainder becomes validation. 1.0 disables validation "
        f"(default: {DEFAULT_TRAIN_VAL_RATIO}).",
    )
    parser.add_argument(
        "--freeze-all-params",
        action="store_true",
        help="(QAT) Freeze model weights so only quantization parameters are updated.",
    )

    # --- Evaluation ---
    parser.add_argument(
        "--eval-tasks",
        type=str,
        nargs="+",
        default=None,
        help="lm-eval tasks to evaluate on, e.g. --eval-tasks wikitext.",
    )
    parser.add_argument(
        "--eval-methods",
        type=str,
        nargs="+",
        default=[DEFAULT_EVAL_METHOD],
        help=f"Evaluation methods to run (default: {DEFAULT_EVAL_METHOD}).",
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
        default=DEFAULT_SOC_MODEL,
        help=f"Target SoC model (default: {DEFAULT_SOC_MODEL}).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        choices=["htp", "gpu"],
        help=f"QNN backend type (default: {DEFAULT_BACKEND}).",
    )

    # --- Runtime prompts and multimodal inputs ---
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
        default=DEFAULT_MAX_SEQ_LEN,  # TODO: Add DEFAULT_MAX_CONTEXT_LEN once attention sink is introduced in GenAI Pipeline.
        help=f"Maximum context length (default: {DEFAULT_MAX_SEQ_LEN}).",
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
        help="Directory containing pre-generated .pte artifacts.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
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

    from executorch.backends.qualcomm.genai_pipeline.model_lookup import is_multimodal

    if is_multimodal(args.model) and args.batch_size != 1:
        logger.warning(
            "Multi-batch is not supported for multimodal LLMs yet; "
            "forcing batch_size from %d to 1.",
            args.batch_size,
        )
        args.batch_size = 1


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
        get_model_arch,
        get_model_config,
        get_model_num_sharding,
        get_quant_dtype,
        get_quant_recipe,
        get_source_transform,
        get_state_dict_loader,
    )

    # TODO: The legacy path already supports attention sink, but GenAI Pipeline
    # does not yet. Until attention sink is supported, keep max_context_len
    # equal to max_seq_len.
    logger.info(
        "Setting max_context_len=%s to match max_seq_len because "
        "attention sink is not yet supported in GenAI Pipeline.",
        args.max_seq_len,
    )
    args.max_context_len = args.max_seq_len

    model_config = get_model_config(args.model)
    logger.info("Model config loaded")

    model_arch = get_model_arch(args.model, args, model_config=model_config)
    logger.info("Model arch loaded")

    weight_transforms, module_transforms = get_source_transform(
        args.model,
        control_args=args,
        model_config=model_config,
    )
    logger.info("Loaded source transforms for model '%s'", args.model)

    num_sharding = get_model_num_sharding(args.model)
    state_dict_loaders = get_state_dict_loader(
        args.model,
        control_args=args,
        model_config=model_config,
    )
    model_options = {
        "model_arch": model_arch,
        "weight_transforms": weight_transforms,
        "module_transforms": module_transforms,
        "embedding_quantize": args.embedding_quantize,
        "num_shardings": num_sharding,
        "state_dict_loader": state_dict_loaders,
    }

    quantize_options = {
        "quant_dtype": get_quant_dtype(args.model),
        "quant_recipe": get_quant_recipe(args.model),
    }

    dataset_options = replace(
        DatasetOptions.from_namespace(args),
        llm_config=model_config,
    )

    return {
        "dataset_options": dataset_options,
        "model_options": model_options,
        "quantize_options": quantize_options,
    }


def get_model_preparation_stage(args: argparse.Namespace) -> "ModelPreparationStage":
    from executorch.backends.qualcomm.genai_pipeline.model_lookup import (
        get_model_loader_adapter,
    )
    from executorch.backends.qualcomm.genai_pipeline.strategies.model_preparation.executorch_model_preparation_strategy import (
        ExecuTorchModelPreparationStrategy,
    )

    model_loader_adapter = get_model_loader_adapter(args.model, args)
    model_preparation_stage = ModelPreparationStage(
        ExecuTorchModelPreparationStrategy(model_loader_adapter=model_loader_adapter)
    )
    return model_preparation_stage


def get_quantization_stage(
    args: argparse.Namespace,
    dataset_options: Dict[str, Any],
    quantize_options: Dict[str, Any],
) -> "QuantizationStage":
    from executorch.backends.qualcomm.genai_pipeline.datasets import (
        get_calibration_dataset_adapter,
        get_eval_dataset_adapter,
        get_training_dataset_adapter,
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
    training_data_adapter = get_training_dataset_adapter(
        dataset_options, is_multimodal=model_is_multimodal
    )
    evaluation_data_adapter = get_eval_dataset_adapter(
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
            training_data_adapter=training_data_adapter,
            evaluation_data_adapter=evaluation_data_adapter,
        )
    )
    return quantization_stage


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the GenAI Pipeline with the given arguments.

    Args:
        args: Parsed CLI arguments.
    """
    from executorch.backends.qualcomm.genai_pipeline.engine_proxy import EngineProxy
    from executorch.backends.qualcomm.genai_pipeline.genai_pipeline import GenAIPipeline
    from executorch.backends.qualcomm.genai_pipeline.pipeline_context import (
        PipelineContext,
    )
    from executorch.backends.qualcomm.genai_pipeline.pipeline_types import (
        EngineType,
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
        "model_mode": args.model_mode,
        "max_seq_len": args.max_seq_len,
        "use_fp16": args.use_fp16,
        "compile_only": args.compile_only,
        "verbose": args.verbose,
        "model_options": configs["model_options"],
        "quantize_options": configs["quantize_options"],
        "dataset_options": configs["dataset_options"],
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

    # Build GenAI Pipeline
    pipeline = GenAIPipeline(
        model_preparation_stage=get_model_preparation_stage(args),
        quantization_stage=(
            # FP16 skips quantization.
            get_quantization_stage(
                args,
                dataset_options=configs["dataset_options"],
                quantize_options=configs["quantize_options"],
            )
            if not args.use_fp16
            else None
        ),
        compilation_stage=None,
        inference_stage=None,
        engine_proxy=EngineProxy(
            stage_engines={
                STAGE_MODEL_PREPARATION: EngineType.EXECUTORCH,
                STAGE_QUANTIZATION: EngineType.EXECUTORCH,
            },
            backend_type=QnnExecuTorchBackendType[
                f"k{args.backend.capitalize()}Backend"
            ],
        ),
    )

    pipeline.invoke(context)

    # Currently, only model preparation and quantization are supported.
    # TODO: Add support for compilation and inference.
    if args.compile_only:
        logger.info("Compile-only mode (no device inference)")


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

    # Run pipeline
    try:
        run_pipeline(args)
    except KeyError as e:
        logger.error("Error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
