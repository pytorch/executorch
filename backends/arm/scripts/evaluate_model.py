# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
import logging
import os
import sys

from pathlib import Path

from typing import Any

# examples/models/model_factory.py is problematic because it performs relative
# importing when os.getcwd() is executorch/. For that to work we need to ensure
# that executorch/ is also in the path prior to importing model_factory.py
# (transitively).
EXECUTORCH_ROOT_DIR = str(Path(__file__).parents[3])
if EXECUTORCH_ROOT_DIR not in sys.path:
    sys.path.append(EXECUTORCH_ROOT_DIR)


import torch

from executorch.backends.arm.scripts.aot_arm_compiler import (
    CALIBRATION_MAX_SAMPLES,
    dump_delegation_info,
    get_model_and_inputs_from_name,
    load_calibration_samples,
    quantize_model,
    QuantMode,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.util._factory import create_partitioner
from executorch.backends.arm.util.arm_model_evaluator import (
    Evaluator,
    FileCompressionEvaluator,
    ImageNetEvaluator,
    NumericalModelEvaluator,
)
from executorch.examples.models import MODEL_NAME_TO_MODEL

from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.utils.data import DataLoader


_EVALUATORS = [
    "numerical",
    "imagenet",
]

_QUANT_MODES = [
    "int8",
    "a16w8",
]

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _get_args():
    parser = argparse.ArgumentParser(
        "Evaluate a model quantized and/or delegated for the Arm backend."
        " Evaluations include numerical comparison to the original model"
        "and/or top-1/top-5 accuracy if applicable."
    )
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help="Model file .py/.pth/.pt or a model from examples/models."
        f" Available models from examples/models: {', '.join(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "-t",
        "--target",
        action="store",
        required=True,
        help=(
            "For Arm backend delegated models, pick the target."
            " Examples of valid targets: TOSA-1.0+INT, TOSA-1.0+FP+bf16"
        ),
    )
    parser.add_argument(
        "-q",
        "--quant_mode",
        required=False,
        default=None,
        choices=_QUANT_MODES,
        help="Quantize the model using the requested mode.",
    )
    parser.add_argument(
        "--calibration_data",
        required=False,
        default=None,
        help=(
            "Optional calibration data file or directory. If a directory is "
            "provided, up to 1000 samples are used for calibration. "
            "Supported files: Common image formats (e.g., .png or .jpg) if "
            "using imagenet evaluator, otherwise .pt/.pth files. If not provided,"
            "quantized models are calibrated on their example inputs."
        ),
    )
    parser.add_argument(
        "--no_delegate",
        action="store_false",
        dest="delegate",
        default=True,
        help=(
            "Disable delegation for cases where a quantized but non-delegated "
            "model is to be tested."
        ),
    )
    parser.add_argument(
        "-e",
        "--evaluators",
        required=True,
        help=(
            "Comma-separated list of evaluators to use. " f"Valid values: {_EVALUATORS}"
        ),
    )
    parser.add_argument(
        "--evaluation_dataset",
        required=False,
        default=None,
        help="Provide path to evaluation dataset directory. (only applicable for ImageNet evaluation).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size to use for ImageNet evaluation. (only applicable for ImageNet evaluation).",
    )
    parser.add_argument(
        "-s",
        "--so_library",
        required=False,
        default=None,
        help="Path to .so library to load custom ops from before evaluation.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set the logging level to debug."
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPE_MAP.keys()),
        default=None,
        help="Cast the model to evaluate and its inputs to the given dtype.",
    )
    parser.add_argument(
        "-i",
        "--intermediates",
        action="store",
        required=True,
        help="Store intermediate output (like TOSA artifacts) at the specified directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        default=None,
        help="Path to JSON file where evaluation metrics will be stored.",
    )
    args = parser.parse_args()

    LOGGING_FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    logging_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=logging_level, format=LOGGING_FORMAT, force=True)

    if args.quant_mode is None and not args.delegate:
        raise ValueError(
            "The model to test must be either quantized or delegated (--quant_mode or --delegate)."
        )

    if args.calibration_data is not None and args.quant_mode is None:
        raise ValueError("--calibration_data requires --quant_mode to be enabled.")

    if args.quant_mode is not None and args.dtype is not None:
        raise ValueError("Cannot specify --dtype when --quant_mode is enabled.")

    evaluators: list[Evaluator] = [
        entry.strip() for entry in args.evaluators.split(",") if entry.strip()
    ]
    unknown = [entry for entry in evaluators if entry not in _EVALUATORS]
    if not evaluators:
        raise ValueError("At least one evaluator must be specified in --evaluators.")
    if unknown:
        raise ValueError(
            "Unknown evaluators in --evaluators: " f"{', '.join(sorted(set(unknown)))}"
        )
    args.evaluators = evaluators

    if "imagenet" in args.evaluators and args.evaluation_dataset is None:
        raise ValueError("Evaluation dataset must be provided for ImageNet evaluation.")

    # Default output path to intermediates folder with name based on target and extensions
    if args.output is None:
        args.output = os.path.join(args.intermediates, f"{args.target}_metrics.json")

    try:
        TosaSpecification.create_from_string(args.target)
    except ValueError as e:
        raise ValueError(f"Invalid target format for --target: {e}")

    return args


def _get_compile_spec(args) -> TosaCompileSpec:
    tosa_spec = TosaSpecification.create_from_string(args.target)
    compile_spec = TosaCompileSpec(tosa_spec)

    if args.intermediates is not None:
        compile_spec.dump_intermediate_artifacts_to(args.intermediates)

    return compile_spec


def _build_imagenet_calibration_samples(
    calibration_dir: str, max_samples: int
) -> list[tuple[torch.Tensor, ...]]:
    dataset = ImageNetEvaluator.load_imagenet_folder(calibration_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    samples: list[tuple[torch.Tensor, ...]] = []
    for image, _ in loader:
        samples.append((image,))
        if len(samples) >= max_samples:
            break
    return samples


def _evaluate(
    args, model_name, ref_model, eval_model, example_inputs
) -> dict[str, Any]:
    evaluators: list[Evaluator] = []

    # Add evaluator for compression ratio of TOSA file
    intermediates_path = Path(args.intermediates)
    tosa_paths = list(intermediates_path.glob("*.tosa"))
    if tosa_paths:
        evaluators.append(FileCompressionEvaluator(model_name, str(tosa_paths[0])))
    else:
        logging.warning(
            f"No TOSA file found in {args.intermediates} for compression evaluation"
        )

    # Add user-specified evaluators
    for evaluator_name in args.evaluators:
        evaluator: Evaluator
        match evaluator_name:
            case "numerical":
                evaluator = NumericalModelEvaluator(
                    model_name,
                    ref_model,
                    eval_model,
                    example_inputs,
                    eval_dtype=_DTYPE_MAP.get(args.dtype, None),
                )
            case "imagenet":
                evaluator = ImageNetEvaluator(
                    model_name,
                    eval_model,
                    batch_size=args.batch_size,
                    validation_dataset_path=args.evaluation_dataset,
                    eval_dtype=_DTYPE_MAP.get(args.dtype, None),
                )
            case _:
                raise AssertionError(f"Unknown evaluator {evaluator_name}")
        evaluators.append(evaluator)

    # Run evaluators
    metrics: dict[str, Any] = {}
    for evaluator in evaluators:
        result = evaluator.evaluate()
        metrics |= result

    return metrics


def main() -> None:
    try:
        args = _get_args()
    except ValueError as e:
        logging.error(f"Argument error: {e}")
        sys.exit(1)

    # if we have custom ops, register them before processing the model
    if args.so_library is not None:
        logging.info(f"Loading custom ops from {args.so_library}")
        torch.ops.load_library(args.so_library)

    # Get the model and its example inputs
    original_model, example_inputs = get_model_and_inputs_from_name(
        args.model_name, None
    )

    # Use original model as reference to compare against
    ref_model = original_model.eval()
    eval_model = ref_model
    eval_inputs = example_inputs

    # Cast model and inputs to eval_dtype if specified
    if args.dtype is not None:
        eval_dtype = _DTYPE_MAP[args.dtype]
        eval_model = copy.deepcopy(original_model).to(eval_dtype).eval()
        eval_inputs = tuple(
            inp.to(eval_dtype) if isinstance(inp, torch.Tensor) else inp
            for inp in example_inputs
        )

    # Export the model
    exported_program = torch.export.export(eval_model, eval_inputs)

    model_name = os.path.basename(os.path.splitext(args.model_name)[0])
    if args.intermediates:
        os.makedirs(args.intermediates, exist_ok=True)

        # We only support Python3.10 and above, so use a later pickle protocol
        torch.export.save(
            exported_program,
            f"{args.intermediates}/{model_name}_exported_program.pt2",
            pickle_protocol=5,
        )

    compile_spec = _get_compile_spec(args)

    # Quantize the model if requested
    if args.quant_mode is not None:
        calibration_samples = None
        if (
            "imagenet" in args.evaluators
            and args.calibration_data is not None
            and Path(args.calibration_data).is_dir()
        ):
            calibration_samples = _build_imagenet_calibration_samples(
                args.calibration_data, CALIBRATION_MAX_SAMPLES
            )
        else:
            calibration_samples = load_calibration_samples(
                args.calibration_data, example_inputs
            )

        match args.quant_mode:
            case "a16w8":
                quant_mode = QuantMode.A16W8
            case "int8":
                quant_mode = QuantMode.INT8
            case _:
                raise AssertionError(f"Unknown quantization mode: {args.quant_mode}")

        eval_model, exported_program = quantize_model(
            exported_program.module(),
            eval_inputs,
            compile_spec,
            model_name,
            True,
            quant_mode,
            calibration_samples,
        )

    # Delegate the model to Arm backend if requested
    if args.delegate:
        partitioner = create_partitioner(compile_spec)
        edge = to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )
        exported_program = edge.exported_program()
        eval_model = exported_program.module()

        dump_delegation_info(edge, args.intermediates)

    # Evaluate the model
    metrics = _evaluate(args, model_name, ref_model, eval_model, example_inputs)

    # Dump result as JSON
    output = {"name": model_name, "target": args.target, "metrics": metrics}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
