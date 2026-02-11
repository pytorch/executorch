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

from pathlib import Path

# Add Executorch root to path so this script can be run from anywhere
_EXECUTORCH_DIR = Path(__file__).resolve().parents[2]
_EXECUTORCH_DIR_STR = str(_EXECUTORCH_DIR)
if _EXECUTORCH_DIR_STR not in sys.path:
    sys.path.insert(0, _EXECUTORCH_DIR_STR)

from typing import Any, Dict, List, Optional, Tuple

import torch
from examples.devtools.scripts.export_bundled_program import save_bundled_program
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

# To use Cortex-M backend
from executorch.backends.cortex_m.passes.convert_to_cortex_m_pass import (
    ConvertToCortexMPass,
)

from executorch.backends.cortex_m.passes.quantized_op_fusion_pass import (
    QuantizedOpFusionPass,
)

from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (
    ReplaceQuantNodesPass,
)

from executorch.devtools import generate_etrecord
from executorch.devtools.backend_debug import get_delegation_info
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite

from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)

from executorch.extension.export_util.utils import save_pte_program
from tabulate import tabulate
from torch.export import ExportedProgram
from torch.fx import GraphModule
from torch.utils.data import DataLoader

# Quantize model if required using the standard export quantizaion flow.
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=FORMAT)

_arm_model_evaluator = None


def _load_arm_model_evaluator() -> Any:
    """Lazily import arm_model_evaluator to avoid heavy deps when not evaluating."""
    global _arm_model_evaluator
    if _arm_model_evaluator is not None:
        return _arm_model_evaluator

    try:
        from executorch.backends.arm.util import arm_model_evaluator as arm_eval
    except Exception as exc:
        raise RuntimeError(
            "Unable to run evaluation because arm_model_evaluator could not be imported. "
            "You probably need to install torchvision or rerun without --evaluate. "
            f"Original import error: {exc}"
        )

    _arm_model_evaluator = arm_eval
    return arm_eval


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
        else MODELS[model_name].example_input
    )

    return model, inputs


def _load_registered_model(
    model_name: str, example_inputs: Any
) -> Optional[Tuple[torch.nn.Module, Any]]:
    """Load a registered example model from `examples.models`."""
    if model_name not in MODEL_NAME_TO_MODEL:
        return None

    logging.warning(
        "Using a model from examples/models not all of these are currently supported"
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


def quantize(
    model: GraphModule,
    model_name: str,
    compile_specs: EthosUCompileSpec | VgfCompileSpec | TosaCompileSpec,
    example_inputs: Tuple[torch.Tensor],
    evaluator_name: str | None,
    evaluator_config: Dict[str, Any] | None,
    is_int16x8: bool = False,
) -> GraphModule:
    """This is the official recommended flow for quantization in pytorch 2.0
    export.

    """
    logging.info("Quantizing Model...")
    logging.debug(f"Original model: {model}")

    quantizer = create_quantizer(compile_specs)

    if is_int16x8:
        if compile_specs.tosa_spec.support_extension("int16"):
            operator_config = get_symmetric_a16w8_quantization_config(
                is_per_channel=True
            )
        else:
            raise ValueError(
                f"Context TOSA spec {compile_specs.tosa_spec} doesn't support int16"
            )
    else:
        operator_config = get_symmetric_quantization_config(is_per_channel=True)

    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)

    dataset = get_calibration_data(
        model_name, example_inputs, evaluator_name, evaluator_config
    )

    # The dataset could be a tuple of tensors or a DataLoader
    # These two cases need to be accounted for
    if isinstance(dataset, DataLoader):
        for sample, _ in dataset:
            m(sample)
    else:
        m(*dataset)

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

CALIBRATION_DATA = {
    "qadd": (torch.randn(32, 2, 1),),
    "qadd2": (
        torch.randn(32, 2, 1),
        torch.randn(32, 2, 1),
    ),
    "qops": (
        torch.randn(32, 2, 1),
        torch.randn(32, 2, 1),
        torch.randn(32, 2, 1) * -0.000001,
        torch.randn(32, 2, 1) * 1000,
    ),
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
]


def get_calibration_data(
    model_name: str,
    example_inputs: Tuple[torch.Tensor],
    evaluator_name: str | None,
    evaluator_config: str | None,
):
    # Firstly, if the model is being evaluated, take the evaluators calibration function if it has one
    if evaluator_name is not None:
        arm_eval = _load_arm_model_evaluator()
        evaluator_data = arm_eval.evaluator_calibration_data(
            evaluator_name, evaluator_config
        )
        if evaluator_data is not None:
            return evaluator_data

    # If the model is in the CALIBRATION_DATA dictionary, get the data from there
    # This is used for the simple model examples provided
    if model_name in CALIBRATION_DATA:
        return CALIBRATION_DATA[model_name]

    # As a last resort, fallback to the scripts previous behavior and return the example inputs
    return example_inputs


def get_compile_spec(
    target: str,
    intermediates: Optional[str] = None,
    system_config: Optional[str] = None,
    memory_mode: Optional[str] = None,
    quantize: bool = False,
    config: Optional[str] = None,
    debug_mode: Optional[str] = None,
    direct_drive: bool = False,
) -> TosaCompileSpec | EthosUCompileSpec | VgfCompileSpec:
    compile_spec = None
    if target.startswith("TOSA"):
        try:
            tosa_spec = TosaSpecification.create_from_string(target)
        except Exception:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
        compile_spec = TosaCompileSpec(tosa_spec)
    elif "ethos-u" in target:
        extra_flags = ["--verbose-operators", "--verbose-cycle-estimate"]
        if debug_mode is not None:
            extra_flags.append("--enable-debug-db")
        if direct_drive:
            extra_flags.append("--separate-io-regions")
            extra_flags.append("--cop-format=COP2")
        compile_spec = EthosUCompileSpec(
            target,
            system_config=system_config,
            memory_mode=memory_mode,
            extra_flags=extra_flags,
            config_ini=config,
        )
    elif "vgf" in target:
        if quantize:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
        else:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP")
        compile_spec = VgfCompileSpec(tosa_spec)
    else:
        raise RuntimeError(f"Unkown target {target}")

    if intermediates is not None:
        compile_spec.dump_intermediate_artifacts_to(intermediates)

    if debug_mode is not None:
        mode = ArmCompileSpec.DebugMode[debug_mode.upper()]
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


def get_args():
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
        help=f"For ArmBackend delegated models, pick the target, and therefore the instruction set generated. valid targets are {TARGETS}",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        required=False,
        nargs="?",
        const="generic",
        choices=["generic", "mv2", "deit_tiny", "resnet18"],
        help="Flag for running evaluation of the model.",
    )
    parser.add_argument(
        "-c",
        "--evaluate_config",
        required=False,
        default=None,
        help="Provide path to evaluator config, if it is required.",
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
        help="Store intermediate output (like TOSA artefacts) somewhere.",
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
        help="Enable the Quantized qdq fusion Op passes",
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

    if args.evaluate and (
        args.quantize is None or args.intermediates is None or (not args.delegate)
    ):
        raise RuntimeError(
            "--evaluate requires --quantize, --intermediates and --delegate to be enabled."
        )

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, force=True)

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

    return args


def save_bpte_program(exec_prog, original_model: torch.nn.Module, output_name: str):
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
                    intermediates_path, f"input_{method_index}.pt"
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
    save_bundled_program(exec_prog, method_test_suites, output_name)


def quantize_model(
    args,
    model: GraphModule,
    example_inputs: Tuple[torch.Tensor],
    compile_spec,
) -> Tuple[GraphModule, ExportedProgram]:

    is_int16x8 = True if args.target == "TOSA-1.0+INT+int16" else False
    model_quant = quantize(
        model,
        args.model_name,
        compile_spec,
        example_inputs,
        args.evaluate,
        args.evaluate_config,
        is_int16x8,
    )
    # Wrap quantized model back into an exported_program
    exported_program = torch.export.export(
        model_quant, example_inputs, strict=args.strict_export
    )

    return model_quant, exported_program


def to_edge_TOSA_delegate(
    exported_program: ExportedProgram,
    args,
    model: GraphModule,
    example_inputs: Tuple[torch.Tensor],
):
    # As we can target multiple output encodings, one must
    # be specified.
    compile_spec = get_compile_spec(
        args.target,
        args.intermediates,
        args.system_config,
        args.memory_mode,
        args.quantize,
        args.config,
        args.enable_debug_mode,
        args.direct_drive,
    )

    model_quant = None
    if args.quantize:
        model_quant, exported_program = quantize_model(
            args, model, example_inputs, compile_spec
        )

    partitioner = create_partitioner(compile_spec)

    edge = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    return model_quant, edge


def to_edge_no_delegate(
    exported_program: ExportedProgram,
    args,
    model: GraphModule,
    example_inputs: Tuple[torch.Tensor],
):
    model_quant = None
    if args.quantize:
        # As we can target multiple output encodings, one must
        # be specified.
        compile_spec = get_compile_spec(
            args.target,
            args.intermediates,
            args.system_config,
            args.memory_mode,
            args.quantize,
            args.config,
            args.enable_debug_mode,
            args.direct_drive,
        )
        model, exported_program = quantize_model(
            args, model, example_inputs, compile_spec
        )
        model_quant = model

    edge = to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    return model_quant, edge


def transform_for_cortex_m_backend(edge_program_manager, args):
    # Let's make sure we are using optimized Cortex M backend
    # NB: If we can't find and replace ops those are expected to be replaced,
    # bad things will happen at runtime, like "missing operator" errors!

    # Instantiate the mandatory ReplaceQuantNodesPass
    passes = [ReplaceQuantNodesPass]
    if args.enable_qdq_fusion_pass:
        passes += [ConvertToCortexMPass, QuantizedOpFusionPass]
    current_edge = edge_program_manager
    for pass_cls in passes:
        transform_pass = (
            pass_cls(current_edge.exported_program())
            if pass_cls.__name__ == "QuantizedLinearFusionPass"
            else pass_cls()
        )
        current_edge = current_edge.transform([transform_pass])
    return current_edge


if __name__ == "__main__":  # noqa: C901
    args = get_args()

    # Pick model from one of the supported lists
    original_model, example_inputs = get_model_and_inputs_from_name(
        args.model_name, args.model_input
    )
    model = original_model.eval()

    # export under the assumption we quantize, the exported form also works
    # in to_edge if we don't quantize
    exported_program = torch.export.export(
        model, example_inputs, strict=args.strict_export
    )

    model = exported_program.module()
    model_fp32 = model

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
    if args.delegate:
        model_quant, edge = to_edge_TOSA_delegate(
            exported_program, args, model, example_inputs
        )
    else:
        model_quant, edge = to_edge_no_delegate(
            exported_program, args, model, example_inputs
        )

    # Cortex-m ops are never included in vgf or direct-drive
    if args.target != "vgf" and not args.direct_drive:
        # Transform so we can use ops from the Cortex M backend
        edge = transform_for_cortex_m_backend(edge, args)

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
        else:
            # --output is a folder
            output_file_name = os.path.join(args.output, output_file_name)

    if args.bundleio or args.etrecord:
        etrecord_file_name = os.path.splitext(output_file_name)[0] + "_etrecord.bin"
        # Generate ETRecord
        generate_etrecord(etrecord_file_name, edge_program_manager_copy, exec_prog)
        print(f"ETRecord saved as {etrecord_file_name}")

    if args.bundleio:
        # Realize the quantization impact on numerics when generating reference output
        reference_model = original_model if not model_quant else model_quant
        save_bpte_program(exec_prog, reference_model, output_file_name)
        print(f"Bundle PTE file saved as {output_file_name}")
    else:
        save_pte_program(exec_prog, output_file_name)
        print(f"PTE file saved as {output_file_name}")

    if args.evaluate:
        arm_eval = _load_arm_model_evaluator()
        arm_eval.evaluate_model(
            args.model_name,
            args.intermediates,
            args.target,
            model_fp32,
            model_quant,
            example_inputs,
            args.evaluate,
            args.evaluate_config,
        )
