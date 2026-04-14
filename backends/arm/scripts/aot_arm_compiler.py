# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import copy
import logging
import os
import sys

from enum import Enum

from pathlib import Path

from typing import Any, List, Optional, Tuple

import torch
from executorch.backends.arm.common.arm_compile_spec import ArmCompileSpec
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    get_symmetric_a16w8_quantization_config,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.compile_spec import TosaCompileSpec
from executorch.backends.arm.util._factory import create_partitioner, create_quantizer

from executorch.backends.arm.vgf import VgfCompileSpec
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager

from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
    ReplaceQuantNodesPass,
)
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.devtools import BundledProgram, generate_etrecord
from executorch.devtools.backend_debug import get_delegation_info
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

from executorch.extension.export_util.utils import save_pte_program
from tabulate import tabulate  # type: ignore[import-untyped]
from torch.export import ExportedProgram
from torch.fx import GraphModule

# Quantize model if required using the standard export quantizaion flow.
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

# Maximum number of samples to use for calibration when quantizing.
CALIBRATION_MAX_SAMPLES = 1000


class QuantMode(str, Enum):
    INT8 = "INT8"
    A16W8 = "A16W8"


def _save_bundled_program(executorch_program, method_test_suites, output_path: str):
    """Serialize a bundled program to disk."""
    bundled_program = BundledProgram(executorch_program, method_test_suites)
    bundled_program_buffer = serialize_from_bundled_program_to_flatbuffer(
        bundled_program
    )

    with open(output_path, "wb") as file:
        file.write(bundled_program_buffer)


def _load_example_inputs(model_input: str | None) -> Any:  # nosec B614
    """Load example inputs from a `.pt` file when a path is provided."""
    if model_input is None:
        return None

    logging.info(f"Load model input from {model_input}")

    if model_input.endswith(".pt"):
        return torch.load(
            model_input, weights_only=False
        )  # nosec B614 trusted artifacts

    raise RuntimeError(
        f"Model input data '{model_input}' is not a valid name. Use --model_input "
        "<FILE>.pt e.g. saved with torch.save()"
    )


def _load_internal_model(
    model_name: str, example_inputs: Any
) -> Optional[Tuple[torch.nn.Module, Any]]:
    """Load a bundled example model from the internal `MODELS` mapping."""
    logging.info(
        "Loading internal models is deprecated. Use --model_name <FILE>.py/.pt "
        "or a model from examples/models."
    )

    if model_name not in MODELS:
        return None

    logging.info(f"Internal model {model_name}")

    model = MODELS[model_name]()
    inputs = (
        example_inputs
        if example_inputs is not None
        else MODELS[model_name].example_input  # type: ignore[attr-defined]
    )

    return model, inputs


def _load_registered_model(
    model_name: str, example_inputs: Any
) -> Optional[Tuple[torch.nn.Module, Any]]:
    """Load a registered example model from `examples.models`."""
    if model_name not in MODEL_NAME_TO_MODEL:
        return None

    logging.warning(
        "Using a model from examples/models. Not all of these are currently supported."
    )
    logging.info(
        f"Load {model_name} -> {MODEL_NAME_TO_MODEL[model_name]} from examples/models"
    )

    model, tmp_example_inputs, _, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[model_name]
    )
    inputs = example_inputs if example_inputs is not None else tmp_example_inputs

    return model, inputs


def _load_python_module_model(
    model_name: str, example_inputs: Any
) -> Optional[Tuple[torch.nn.Module, Any]]:
    """Load a model and inputs from a Python source file.

    The file must define `ModelUnderTest` and `ModelInputs` attributes.

    """
    if not model_name.endswith(".py"):
        return None

    logging.info(
        f"Load model file {model_name} "
        "Variable ModelUnderTest=<Model> ModelInputs=<ModelInput>"
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location("tmp_model", model_name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model file {model_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["tmp_model"] = module
    model = module.ModelUnderTest
    inputs = example_inputs if example_inputs is not None else module.ModelInputs

    return model, inputs


def _load_serialized_model(
    model_name: str, example_inputs: Any
) -> Optional[Tuple[torch.nn.Module, Any]]:  # nosec B614
    """Load a serialized Torch model saved via `torch.save`."""
    if not model_name.endswith((".pth", ".pt")):
        return None

    logging.info(f"Load model file {model_name}")

    model = torch.load(model_name, weights_only=False)  # nosec B614 trusted inputs
    if example_inputs is None:
        raise RuntimeError(
            f"Model '{model_name}' requires input data specify --model_input <FILE>.pt"
        )

    return model, example_inputs


def _apply_replace_quant_nodes(edge, target: str, direct_drive: bool):
    """Apply the replace_quant_nodes pass to the edge graph module."""

    if target != "vgf" and not direct_drive:
        edge = edge.transform([ReplaceQuantNodesPass()])
    return edge


def get_model_and_inputs_from_name(
    model_name: str, model_input: str | None
) -> Tuple[torch.nn.Module, Any]:
    """Resolve a model name into a model instance and example inputs.

    Args:
        model_name: Identifier for the model. It can be a key in
            `MODEL_NAME_TO_MODEL`, a Python module path, or a serialized
            model file path.
        model_input: Optional path to a `.pt` file containing example inputs.

    Returns:
        Tuple of `(model, example_inputs)` ready for compilation.

    Raises:
        RuntimeError: If the model cannot be resolved or required inputs are
            missing.

    """
    example_inputs = _load_example_inputs(model_input)

    loaders = (
        _load_internal_model,
        _load_registered_model,
        _load_python_module_model,
        _load_serialized_model,
    )

    for loader in loaders:
        result = loader(model_name, example_inputs)
        if result is not None:
            model, example_inputs = result
            logging.debug(f"Loaded model: {model}")
            logging.debug(f"Loaded input: {example_inputs}")
            return model, example_inputs

    raise RuntimeError(
        f"Model '{model_name}' is not a valid name. Use --help for a list of available models."
    )


def as_input_tuple(sample: object) -> Tuple[torch.Tensor, ...]:
    if isinstance(sample, tuple):
        return sample
    if isinstance(sample, list):
        return tuple(sample)
    if isinstance(sample, torch.Tensor):
        return (sample,)
    if isinstance(sample, dict):
        if "pixel_values" in sample:
            return (sample["pixel_values"],)
        raise ValueError("Calibration sample dict must contain 'pixel_values' key.")
    raise ValueError(
        "Calibration sample must be a Tensor, tuple, list, or dict with "
        "'pixel_values'."
    )


def load_calibration_sample(
    path: str, example_inputs: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    suffix = Path(path).suffix.lower()
    if suffix in {".pt", ".pth"}:
        sample = torch.load(path, weights_only=False)  # nosec B614 trusted inputs
        return as_input_tuple(sample)
    raise ValueError(f"Unsupported calibration file type: {path}")


def load_calibration_samples(
    calibration_data: str | None,
    example_inputs: Tuple[torch.Tensor, ...],
) -> Optional[List[Tuple[torch.Tensor, ...]]]:
    if calibration_data is None:
        return None

    path = Path(calibration_data)
    if path.is_file():
        return [load_calibration_sample(str(path), example_inputs)]

    if not path.is_dir():
        raise ValueError(
            f"Calibration data path '{calibration_data}' is not a file or directory."
        )

    supported_suffixes = {".pt", ".pth"}
    candidates = sorted(
        str(p)
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_suffixes
    )
    if not candidates:
        raise ValueError(
            f"No supported calibration files found in directory '{calibration_data}'."
        )

    samples: List[Tuple[torch.Tensor, ...]] = []
    for candidate in candidates[:CALIBRATION_MAX_SAMPLES]:
        samples.append(load_calibration_sample(candidate, example_inputs))

    return samples


def _validate_calibration_sample(
    calibration_sample: Tuple[torch.Tensor, ...],
    example_inputs: Tuple[torch.Tensor, ...],
) -> None:
    expected_len = len(example_inputs)

    if len(calibration_sample) != expected_len:
        raise ValueError(
            "Calibration sample has %d inputs, expected %d."
            % (len(calibration_sample), expected_len)
        )
    for input_idx, (expected, actual) in enumerate(
        zip(example_inputs, calibration_sample)
    ):
        if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
            if expected.shape != actual.shape:
                raise ValueError(
                    "Calibration sample input %d shape %s does not match "
                    "expected shape %s."
                    % (input_idx, list(actual.shape), list(expected.shape))
                )
        elif type(expected) is not type(actual):
            raise ValueError(
                "Calibration sample input %d type %s does not match "
                "expected type %s."
                % (input_idx, type(actual).__name__, type(expected).__name__)
            )


def quantize(
    model: GraphModule,
    model_name: str,
    compile_specs: ArmCompileSpec,
    example_inputs: Tuple[torch.Tensor],
    quant_mode: QuantMode = QuantMode.INT8,
    calibration_samples: Optional[List[Tuple[torch.Tensor, ...]]] = None,
) -> GraphModule:
    """This is the official recommended flow for quantization in pytorch 2.0
    export.
    """
    logging.info("Quantizing Model...")
    logging.debug(f"Original model: {model}")

    quantizer = create_quantizer(compile_specs)

    match quant_mode:
        case QuantMode.INT8:
            operator_config = get_symmetric_quantization_config(is_per_channel=True)
        case QuantMode.A16W8:
            if compile_specs.tosa_spec.support_extension("int16"):
                operator_config = get_symmetric_a16w8_quantization_config(
                    is_per_channel=True
                )
            else:
                raise ValueError(
                    f"Context TOSA spec {compile_specs.tosa_spec} doesn't support int16"
                )
        case _:
            raise ValueError(f"Unsupported quantization mode: {quant_mode}")

    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)

    if calibration_samples is None:
        calibration_samples = [example_inputs]

    for sample in calibration_samples:
        _validate_calibration_sample(sample, example_inputs)
        m(*sample)

    m = convert_pt2e(m)
    logging.debug(f"Quantized model: {m}")
    return m


class QuantAddTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        return a + a

    example_input = (torch.rand([13, 3], dtype=torch.float32),)  # a - normal values
    can_delegate = True  # when quantized


class QuantAddTest2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        p = a + a
        q = b + b
        r = p + q
        return p, q, r

    example_input = (
        torch.randn([13, 7, 3], dtype=torch.float32),
        torch.randn([13, 7, 3], dtype=torch.float32),
    )
    can_delegate = True  # when quantized


class QuantOpTest(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, w, x, y, z):
        o1 = w - x
        o2 = o1 + y
        o3 = o2 * z
        return o1, o2, o3

    example_input = (
        torch.randn([3, 1, 2], dtype=torch.float32),  # w - normal values
        torch.randn([3, 5, 2], dtype=torch.float32),  # x - normal values
        torch.randn([3, 5, 1], dtype=torch.float32)
        * -0.000001,  # y - small -ve values, needs to be calibration for tests
        torch.randn([3, 5, 2], dtype=torch.float32) * 1000,  # z - large values
    )
    can_delegate = True  # when quantized


class QuantLinearTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a simple linear layer
        self.linear = torch.nn.Linear(61, 37)

    def forward(self, x):
        return self.linear(x)

    example_input = (torch.randn([8, 61], dtype=torch.float32),)
    can_delegate = True


MODELS = {
    "qadd": QuantAddTest,
    "qadd2": QuantAddTest2,
    "qops": QuantOpTest,
    # TODO: Remove this from here, once we have dedicated MCU test pipeline ready. This is an interim solution.
    # See https://github.com/pytorch/executorch/discussions/13944
    "qlinear": QuantLinearTest,
}

TARGETS = [
    "ethos-u55-32",
    "ethos-u55-64",
    "ethos-u55-128",
    "ethos-u55-256",
    "ethos-u85-128",
    "ethos-u85-256",
    "ethos-u85-512",
    "ethos-u85-1024",
    "ethos-u85-2048",
    "vgf",
    "TOSA-1.0+INT",
    "TOSA-1.0+FP",
    "TOSA-1.0+INT+int16",
    "cortex-m55+int8",
]


def _get_compile_spec(args) -> ArmCompileSpec:
    compile_spec: ArmCompileSpec

    if args.target.startswith("TOSA"):
        tosa_spec = TosaSpecification.create_from_string(args.target)
        compile_spec = TosaCompileSpec(tosa_spec)
    elif "ethos-u" in args.target:
        extra_flags = ["--verbose-operators", "--verbose-cycle-estimate"]
        if args.enable_debug_mode is not None:
            extra_flags.append("--enable-debug-db")
        if args.direct_drive:
            extra_flags.append("--separate-io-regions")
            extra_flags.append("--cop-format=COP2")
        compile_spec = EthosUCompileSpec(
            args.target,
            system_config=args.system_config,
            memory_mode=args.memory_mode,
            extra_flags=extra_flags,
            config_ini=args.config,
        )
    elif "vgf" in args.target:
        if args.quantize:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
        else:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP")
        compile_spec = VgfCompileSpec(tosa_spec)
    else:
        raise RuntimeError(f"Unkown target {args.target}")

    if args.intermediates is not None:
        compile_spec.dump_intermediate_artifacts_to(args.intermediates)

    if args.enable_debug_mode is not None:
        mode = ArmCompileSpec.DebugMode[args.enable_debug_mode.upper()]
        compile_spec.dump_debug_info(mode)

    return compile_spec


def dump_delegation_info(edge, intermediate_files_folder: Optional[str] = None):
    graph_module = edge.exported_program().graph_module
    delegation_info = get_delegation_info(graph_module)
    df = delegation_info.get_operator_delegation_dataframe()
    table = tabulate(df, headers="keys", tablefmt="fancy_grid")
    delegation_info_string = f"Delegation info:\n{delegation_info.get_summary()}\nDelegation table:\n{table}\n"
    logging.info(delegation_info_string)
    if intermediate_files_folder is not None:
        delegation_file_path = os.path.join(
            intermediate_files_folder, "delegation_info.txt"
        )
        with open(delegation_file_path, "w") as file:
            file.write(delegation_info_string)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help=f"Model file .py/.pth/.pt or a model from examples/models. Valid names: {set(MODEL_NAME_TO_MODEL.keys())}",
    )
    parser.add_argument(
        "--model_input",
        required=False,
        default=None,
        help="Provide model input .pt file, or python variable name",
    )
    parser.add_argument(
        "-d",
        "--delegate",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing ArmBackend delegated model",
    )
    parser.add_argument(
        "--bundleio",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing BundleIO bpte file with input/output test/ref data.",
    )
    parser.add_argument(
        "--etrecord",
        action="store_true",
        required=False,
        default=False,
        help="Flag for producing a etrecord file.",
    )
    parser.add_argument(
        "-t",
        "--target",
        action="store",
        required=False,
        default="ethos-u55-128",
        choices=TARGETS,
        help=f"Target backend. For delegated models: Ethos-U/VGF/TOSA variants. For non-delegated: cortex-m55+int8 (CMSIS-NN portable kernels). Valid targets: {TARGETS}",
    )
    # TODO: Remove --evaluate and --evaluate_config completely after a suitable time.
    # They are deprecated and no longer functional in this script.
    parser.add_argument(
        "-e",
        "--evaluate",
        required=False,
        nargs="?",
        const="generic",
        choices=["generic", "mv2", "deit_tiny", "resnet18"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-c",
        "--evaluate_config",
        required=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        required=False,
        default=False,
        help="Produce a quantized model",
    )
    parser.add_argument(
        "--calibration_data",
        required=False,
        default=None,
        help=(
            "Optional calibration data file or directory. If a directory is "
            "provided, up to 1000 samples are used for calibration. "
            "Supported files: .pt/.pth. If not provided,"
            "quantized models are calibrated on their example inputs."
        ),
    )
    parser.add_argument(
        "-s",
        "--so_library",
        required=False,
        default=None,
        help="Provide path to custom .so library.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Set the logging level to debug."
    )
    parser.add_argument(
        "-i",
        "--intermediates",
        action="store",
        required=False,
        help="Store intermediate output (like TOSA artifacts) somewhere.",
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=False,
        help="Filename (if .pte or .bpte is used) or a folder for outputs, if not specified the default is to place files in cwd.",
    )
    parser.add_argument(
        "--system_config",
        required=False,
        default=None,
        help="System configuration to select from the Vela configuration file (see vela.ini). This option must match the selected target, default is for an optimal system 'Ethos_U55_High_End_Embedded'/'Ethos_U85_SYS_DRAM_High'",
    )
    parser.add_argument(
        "--memory_mode",
        required=False,
        default=None,
        help="Memory mode to select from the Vela configuration file (see vela.ini). Default is 'Shared_Sram' for Ethos-U55 targets and 'Sram_Only' for Ethos-U85 targets",
    )
    parser.add_argument(
        "--config",
        required=False,
        default="Arm/vela.ini",
        help="Specify custom vela configuration file (vela.ini) for Ethos-U targets.",
    )
    parser.add_argument(
        "--non_strict_export",
        dest="strict_export",
        required=False,
        action="store_false",
        help="Disable strict checking while exporting models.",
    )
    parser.add_argument(
        "--enable_qdq_fusion_pass",
        action="store_true",
        help="[DEPRECATED] This flag is no longer used and will be removed in a future release.",
    )
    parser.add_argument(
        "--enable_debug_mode",
        required=False,
        choices=["json", "tosa"],
        help="Flag to enable ATen-to-TOSA debug mode and dumping of Vela's debug database.",
    )
    parser.add_argument(
        "--direct_drive",
        action="store_true",
        required=False,
        default=False,
        help="Flag for enabling direct drive.",
    )
    args = parser.parse_args()

    LOGGING_FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
    logging_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=logging_level, format=LOGGING_FORMAT, force=True)

    if args.calibration_data is not None and not args.quantize:
        raise RuntimeError("--calibration_data requires --quantize to be enabled.")

    # if we have custom ops, register them before processing the model
    if args.so_library is not None:
        logging.info(f"Loading custom ops from {args.so_library}")
        torch.ops.load_library(args.so_library)

    if (
        args.model_name in MODELS.keys()
        and args.delegate is True
        and MODELS[args.model_name].can_delegate is False
    ):
        raise RuntimeError(f"Model {args.model_name} cannot be delegated.")

    if args.evaluate is not None or args.evaluate_config is not None:
        logging.error(
            "Model evaluation is no longer supported in this script."
            " Use evaluate_model.py instead. Ignore and continue."
        )

    return args


def _save_bpte_program(
    exec_prog,
    original_model: torch.nn.Module,
    output_name: str,
    example_inputs: Tuple[torch.Tensor, ...],
    args,
):
    # Construct MethodTestSuite for Each Method

    # Generate Test Suites
    method_names = [
        method.name for method in exec_prog.executorch_program.execution_plan
    ]

    program_inputs = {m_name: [example_inputs] for m_name in method_names}

    method_test_suites: List[MethodTestSuite] = []
    for m_name in method_names:
        method_inputs = program_inputs[m_name]

        # To create a bundled program, we first create every test cases from input. We leverage eager model
        # to generate expected output for each test input, and use MethodTestCase to hold the information of
        # each test case. We gather all MethodTestCase for same method into one MethodTestSuite, and generate
        # bundled program by all MethodTestSuites.
        method_test_cases: List[MethodTestCase] = []

        if args.intermediates:
            # Save model.pth
            intermediates_path = Path(args.intermediates)
            model_path = os.path.join(intermediates_path, "model.pth")
            try:
                torch.save(original_model, model_path)
            except:
                logging.warning(f"Could not torch.save(model, {model_path})")

        method_index = 0
        for method_input in method_inputs:
            output_ref = original_model(*method_input)

            logging.debug(f"input_{method_index}: {method_input}")
            logging.debug(f"output_ref_{method_index}: {output_ref}")

            if args.intermediates:
                # Save model input and referece output
                input_path = os.path.join(
                    intermediates_path, f"input_{method_index}.pt"  # type: ignore[possibly-undefined]
                )
                try:
                    torch.save(method_input, input_path)
                except:
                    logging.warning(
                        f"Could not torch.save(input_{method_index}, {input_path})"
                    )
                refoutput_path = os.path.join(
                    intermediates_path, f"output_ref_{method_index}.pt"
                )
                try:
                    torch.save(output_ref, refoutput_path)
                except:
                    logging.warning(
                        f"Could not torch.save(output_ref_{method_index}, {refoutput_path})"
                    )

            method_test_cases.append(
                MethodTestCase(
                    inputs=method_input,
                    expected_outputs=output_ref,
                )
            )

            method_index = method_index + 1

        method_test_suites.append(
            MethodTestSuite(
                method_name=m_name,
                test_cases=method_test_cases,
            )
        )

    # Generate BundledProgram
    output_dir = os.path.dirname(output_name)
    os.makedirs(output_dir, exist_ok=True)
    _save_bundled_program(exec_prog, method_test_suites, output_name)


def quantize_model(
    model: GraphModule,
    example_inputs: Tuple[torch.Tensor],
    compile_spec,
    model_name: str,
    strict_export: bool,
    quant_mode: QuantMode,
    calibration_samples: Optional[List[Tuple[torch.Tensor, ...]]],
) -> Tuple[GraphModule, ExportedProgram]:
    model_quant = quantize(
        model,
        model_name,
        compile_spec,
        example_inputs,
        quant_mode,
        calibration_samples,
    )
    # Wrap quantized model back into an exported_program
    exported_program = torch.export.export(
        model_quant, example_inputs, strict=strict_export
    )

    return model_quant, exported_program


def _to_edge_TOSA_delegate(
    target: str,
    exported_program: ExportedProgram,
    compile_spec,
    model: GraphModule,
    quant_mode: Optional[QuantMode],
    example_inputs: Tuple[torch.Tensor],
    model_name: str,
    strict_export: bool,
    calibration_samples: Optional[List[Tuple[torch.Tensor, ...]]],
    direct_drive: bool,
):
    model_quant = None
    if quant_mode is not None:
        model_quant, exported_program = quantize_model(
            model,
            example_inputs,
            compile_spec,
            model_name,
            strict_export,
            quant_mode,
            calibration_samples,
        )

    partitioner = create_partitioner(compile_spec)

    edge = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    # Replace quantized_decomposed::{quantize,dequantize}_per_tensor nodes
    # with cortex_m:: equivalents for int8 QDQ ops remaining outside the
    # delegated subgraph.
    edge = _apply_replace_quant_nodes(edge, target, direct_drive)

    return model_quant, edge


def _to_edge_cortex_m(
    exported_program: ExportedProgram,
    args,
    model: GraphModule,
    example_inputs: Tuple[torch.Tensor],
    calibration_samples: Optional[List[Tuple[torch.Tensor, ...]]],
):
    """Cortex-M/CMSIS-NN compilation path with no delegation."""
    logging.info("Using Cortex-M/CMSIS-NN compilation path (no delegation)")

    def _to_channels_last(x):
        if isinstance(x, torch.Tensor):
            if x.dim() == 4 and not x.is_contiguous(memory_format=torch.channels_last):
                logging.warning(
                    "Converting input tensor with shape %s to channels_last",
                    list(x.shape),
                )
                return x.to(memory_format=torch.channels_last)
            return x
        elif isinstance(x, tuple):
            return tuple(_to_channels_last(t) for t in x)
        return x

    if not args.quantize:
        logging.warning(
            "Quantization is DISABLED. Cortex-M typically requires quantization."
        )
        model_quant = None
    else:
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        example_inputs = tuple(_to_channels_last(x) for x in example_inputs)

        quantizer = CortexMQuantizer()
        prepared = prepare_pt2e(model, quantizer)

        if calibration_samples is None:
            calibration_samples = [example_inputs]

        for sample in calibration_samples:
            prepared(*tuple(_to_channels_last(x) for x in sample))

        model_quant = convert_pt2e(prepared)

        exported_program = torch.export.export(
            model_quant, example_inputs, strict=args.strict_export
        )

    edge = to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(
            preserve_ops=[
                torch.ops.aten.linear.default,
                torch.ops.aten.hardsigmoid.default,
                torch.ops.aten.hardsigmoid_.default,
                torch.ops.aten.hardswish.default,
                torch.ops.aten.hardswish_.default,
            ],
            _check_ir_validity=False,
        ),
    )

    pass_manager = CortexMPassManager(edge.exported_program())
    edge._edge_programs["forward"] = pass_manager.transform()

    return model_quant, edge


def _to_edge_no_delegate(
    args,
    exported_program: ExportedProgram,
    compile_spec,
    model: GraphModule,
    quant_mode: Optional[QuantMode],
    example_inputs: Tuple[torch.Tensor],
    model_name: str,
    strict_export: bool,
    calibration_samples: Optional[List[Tuple[torch.Tensor, ...]]],
):
    model_quant = None
    if quant_mode is not None:
        # As we can target multiple output encodings, one must
        # be specified.
        model, exported_program = quantize_model(
            model,
            example_inputs,
            compile_spec,
            model_name,
            strict_export,
            quant_mode,
            calibration_samples,
        )
        model_quant = model

    edge = to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    # Replace quantized_decomposed::{quantize,dequantize}_per_tensor nodes
    # with cortex_m:: equivalents for int8 QDQ ops remaining outside the
    # delegated subgraph.
    edge = _apply_replace_quant_nodes(edge, args.target, args.direct_drive)

    return model_quant, edge


def main() -> None:  # noqa: C901
    args = _get_args()

    # Pick model from one of the supported lists
    original_model, example_inputs = get_model_and_inputs_from_name(
        args.model_name, args.model_input
    )
    calibration_samples = load_calibration_samples(
        args.calibration_data, example_inputs
    )
    model = original_model.eval()

    # export under the assumption we quantize, the exported form also works
    # in to_edge if we don't quantize
    exported_program = torch.export.export(
        model, example_inputs, strict=args.strict_export
    )

    model = exported_program.module()

    if args.enable_qdq_fusion_pass:
        logging.warning(
            "--enable_qdq_fusion_pass is deprecated and has no effect. "
            "Quantized node replacement is now handled within the "
            "respective compilation paths."
        )

    model_name = os.path.basename(os.path.splitext(args.model_name)[0])
    if args.intermediates:
        os.makedirs(args.intermediates, exist_ok=True)

        # We only support Python3.10 and above, so use a later pickle protocol
        torch.export.save(
            exported_program,
            f"{args.intermediates}/{model_name}_exported_program.pt2",
            pickle_protocol=5,
        )

    # Quantize if required
    model_quant = None
    if args.quantize:
        quant_mode = QuantMode.A16W8 if "int16" in args.target else QuantMode.INT8
    else:
        quant_mode = None

    if args.target == "cortex-m55+int8":
        # Cortex-M path: CMSIS-NN portable kernels, no delegation
        if args.delegate:
            logging.warning(
                "--delegate is ignored for target 'cortex-m55+int8' "
                "(this target does not use delegated ops)."
            )
            args.delegate = False
        model_quant, edge = _to_edge_cortex_m(
            exported_program,
            args,
            model,
            example_inputs,
            calibration_samples,
        )
    elif args.delegate:
        # As we can target multiple output encodings, one must
        # be specified.
        model_quant, edge = _to_edge_TOSA_delegate(
            args.target,
            exported_program,
            _get_compile_spec(args),
            model,
            quant_mode,
            example_inputs,
            args.model_name,
            args.strict_export,
            calibration_samples,
            args.direct_drive,
        )
    else:
        model_quant, edge = _to_edge_no_delegate(
            args,
            exported_program,
            _get_compile_spec(args),
            model,
            quant_mode,
            example_inputs,
            args.model_name,
            args.strict_export,
            calibration_samples,
        )

    dump_delegation_info(edge, args.intermediates)

    edge_program_manager_copy = copy.deepcopy(edge)

    try:
        exec_prog = edge.to_executorch(
            config=ExecutorchBackendConfig(extract_delegate_segments=False)
        )
    except RuntimeError as e:
        if "Missing out variants" in str(e.args[0]):
            raise RuntimeError(
                e.args[0]
                + ".\nThis likely due to an external so library not being loaded. Supply a path to it with the -s flag."
            ).with_traceback(e.__traceback__) from None
        else:
            raise e

    output_name = f"{model_name}" + (
        f"_arm_delegate_{args.target}"
        if args.delegate is True
        else f"_arm_{args.target}"
    )

    if args.bundleio:
        output_file_name = f"{output_name}.bpte"
    else:
        output_file_name = f"{output_name}.pte"

    if args.output is not None:
        if args.output.endswith(".pte") or args.output.endswith(".bpte"):
            # --output is a pte or bundle pte filename use it as output name
            if args.bundleio and not args.output.endswith(".bpte"):
                raise RuntimeError(
                    f"--bundleio expects a .bpte file ending to --output and not .pte {args.output}"
                )
            if not args.bundleio and not args.output.endswith(".pte"):
                raise RuntimeError(
                    f"When not using --bundleio a .bpte file should not be use as --output {args.output}"
                )
            output_file_name = args.output
            output_dir = os.path.dirname(output_file_name)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        else:
            # --output is a folder
            os.makedirs(args.output, exist_ok=True)
            output_file_name = os.path.join(args.output, output_file_name)

    if args.bundleio:
        # Realize the quantization impact on numerics when generating reference output
        reference_model = original_model if not model_quant else model_quant
        _save_bpte_program(
            exec_prog, reference_model, output_file_name, example_inputs, args
        )
        print(f"Bundle PTE file saved as {output_file_name}")
    else:
        save_pte_program(exec_prog, output_file_name)
        print(f"PTE file saved as {output_file_name}")

    if args.bundleio or args.etrecord:
        etrecord_file_name = os.path.splitext(output_file_name)[0] + "_etrecord.bin"
        try:
            generate_etrecord(etrecord_file_name, edge_program_manager_copy, exec_prog)
            print(f"ETRecord saved as {etrecord_file_name}")
        except Exception as e:
            # Treat ETRecord failures as non-fatal only when generated as a side-effect
            # of --bundleio. When --etrecord is explicitly requested, fail loudly.
            if args.bundleio and not args.etrecord:
                logging.warning(f"ETRecord generation failed (non-fatal): {e}")
            else:
                raise


if __name__ == "__main__":
    main()
