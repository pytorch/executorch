# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import copy
import json
import logging
import os

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from examples.devtools.scripts.export_bundled_program import save_bundled_program
from executorch.backends.arm.arm_backend import (
    ArmCompileSpecBuilder,
    is_ethosu,
    is_tosa,
    is_vgf,
)
from executorch.backends.arm.ethosu import EthosUPartitioner
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    get_symmetric_quantization_config,
    TOSAQuantizer,
    VgfQuantizer,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.backends.arm.tosa.partitioner import TOSAPartitioner
from executorch.backends.arm.tosa.specification import get_tosa_spec

from executorch.backends.arm.util.arm_model_evaluator import (
    GenericModelEvaluator,
    MobileNetV2Evaluator,
)

from executorch.backends.arm.vgf_partitioner import VgfPartitioner

# To use Cortex-M backend
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
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.extension.export_util.utils import save_pte_program
from tabulate import tabulate
from torch.utils.data import DataLoader

# Quantize model if required using the standard export quantizaion flow.
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=FORMAT)


def get_model_and_inputs_from_name(
    model_name: str, model_input: str | None
) -> Tuple[torch.nn.Module, Any]:
    """Given the name of an example pytorch model, return it and example inputs.

    Raises RuntimeError if there is no example model corresponding to the given name.
    """
    example_inputs = None
    if model_input is not None:
        logging.info(f"Load model input from {model_input}")
        if model_input.endswith(".pt"):
            example_inputs = torch.load(model_input, weights_only=False)
        else:
            raise RuntimeError(
                f"Model input data '{model_input}' is not a valid name. Use --model_input <FILE>.pt e.g. saved with torch.save()"
            )

    # Case 1: Model is defined in this file
    if model_name in models.keys():
        logging.info(f"Internal model {model_name}")
        model = models[model_name]()
        if example_inputs is None:
            example_inputs = models[model_name].example_input
    # Case 2: Model is defined in examples/models/
    elif model_name in MODEL_NAME_TO_MODEL.keys():
        logging.warning(
            "Using a model from examples/models not all of these are currently supported"
        )
        logging.info(
            f"Load {model_name} -> {MODEL_NAME_TO_MODEL[model_name]} from examples/models"
        )

        model, tmp_example_inputs, _, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL[model_name]
        )
        if example_inputs is None:
            example_inputs = tmp_example_inputs
    # Case 3: Model is in an external python file loaded as a module.
    #         ModelUnderTest should be a torch.nn.module instance
    #         ModelInputs should be a tuple of inputs to the forward function
    elif model_name.endswith(".py"):
        logging.info(
            f"Load model file {model_name}   Variable ModelUnderTest=<Model> ModelInputs=<ModelInput>"
        )
        import importlib.util

        # load model's module and add it
        spec = importlib.util.spec_from_file_location("tmp_model", model_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.ModelUnderTest
        if example_inputs is None:
            example_inputs = module.ModelInputs
    # Case 4: Model is in an saved model file torch.save(model)
    elif model_name.endswith(".pth") or model_name.endswith(".pt"):
        logging.info(f"Load model file {model_name}")
        model = torch.load(model_name, weights_only=False)
        if example_inputs is None:
            raise RuntimeError(
                f"Model '{model_name}' requires input data specify --model_input <FILE>.pt"
            )
    else:
        raise RuntimeError(
            f"Model '{model_name}' is not a valid name. Use --help for a list of available models."
        )
    logging.debug(f"Loaded model: {model}")
    logging.debug(f"Loaded input: {example_inputs}")
    return model, example_inputs


def quantize(
    model: torch.nn.Module,
    model_name: str,
    compile_specs: list[CompileSpec],
    example_inputs: Tuple[torch.Tensor],
    evaluator_name: str | None,
    evaluator_config: Dict[str, Any] | None,
) -> torch.nn.Module:
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    logging.info("Quantizing Model...")
    logging.debug(f"Original model: {model}")
    quantizer = None
    if is_ethosu(compile_specs):
        quantizer = EthosUQuantizer(compile_specs)
    elif is_tosa(compile_specs):
        quantizer = TOSAQuantizer(get_tosa_spec(compile_specs))
    elif is_vgf(compile_specs):
        quantizer = VgfQuantizer(compile_specs)
    else:
        raise RuntimeError("Unsupported compilespecs for quantization!")

    operator_config = get_symmetric_quantization_config()
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


# Simple example models
class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x

    example_input = (torch.ones(5, dtype=torch.int32),)
    can_delegate = True


class AddModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

    example_input = (
        torch.ones(5, dtype=torch.int32),
        torch.ones(5, dtype=torch.int32),
    )
    can_delegate = True


class AddModule3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return (x + y, x + x)

    example_input = (
        torch.ones(5, dtype=torch.int32),
        torch.ones(5, dtype=torch.int32),
    )
    can_delegate = True


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


class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        z = self.softmax(x)
        return z

    example_input = (torch.ones(2, 2),)
    can_delegate = True


class MultipleOutputsModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (x * y, x.sum(dim=-1, keepdim=True))

    example_input = (torch.randn(10, 4, 5), torch.randn(10, 4, 5))
    can_delegate = True


models = {
    "add": AddModule,
    "add2": AddModule2,
    "add3": AddModule3,
    "qadd": QuantAddTest,
    "qadd2": QuantAddTest2,
    "qops": QuantOpTest,
    "softmax": SoftmaxModule,
    "MultipleOutputsModule": MultipleOutputsModule,
}

calibration_data = {
    "add": (torch.randn(1, 5),),
    "add2": (
        torch.randn(1, 5),
        torch.randn(1, 5),
    ),
    "add3": (
        torch.randn(32, 5),
        torch.randn(32, 5),
    ),
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
    "softmax": (torch.randn(32, 2, 2),),
}

evaluators = {
    "generic": GenericModelEvaluator,
    "mv2": MobileNetV2Evaluator,
}

targets = [
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
]


def get_calibration_data(
    model_name: str,
    example_inputs: Tuple[torch.Tensor],
    evaluator_name: str | None,
    evaluator_config: str | None,
):
    # Firstly, if the model is being evaluated, take the evaluators calibration function if it has one
    if evaluator_name is not None:
        evaluator = evaluators[evaluator_name]

        if hasattr(evaluator, "get_calibrator"):
            assert evaluator_config is not None

            config_path = Path(evaluator_config)
            with config_path.open() as f:
                config = json.load(f)

            if evaluator_name == "mv2":
                return evaluator.get_calibrator(
                    training_dataset_path=config["training_dataset_path"]
                )
            else:
                raise RuntimeError(f"Unknown evaluator: {evaluator_name}")

    # If the model is in the calibration_data dictionary, get the data from there
    # This is used for the simple model examples provided
    if model_name in calibration_data:
        return calibration_data[model_name]

    # As a last resort, fallback to the scripts previous behavior and return the example inputs
    return example_inputs


def get_compile_spec(
    target: str,
    intermediates: Optional[str] = None,
    system_config: Optional[str] = None,
    memory_mode: Optional[str] = None,
    quantize: bool = False,
    config: Optional[str] = None,
) -> list[CompileSpec]:
    spec_builder = None
    if target.startswith("TOSA"):
        try:
            tosa_spec = TosaSpecification.create_from_string(target)
        except:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
        spec_builder = ArmCompileSpecBuilder().tosa_compile_spec(tosa_spec)
    elif "ethos-u" in target:
        spec_builder = ArmCompileSpecBuilder().ethosu_compile_spec(
            target,
            system_config=system_config,
            memory_mode=memory_mode,
            extra_flags="--verbose-operators --verbose-cycle-estimate",
            config_ini=config,
        )
    elif "vgf" in target:
        if quantize:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT")
        else:
            tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+FP")
        spec_builder = ArmCompileSpecBuilder().vgf_compile_spec(tosa_spec)

    if intermediates is not None:
        spec_builder.dump_intermediate_artifacts_to(intermediates)

    return spec_builder.build()


def evaluate_model(
    model_name: str,
    intermediates: str,
    model_fp32: torch.nn.Module,
    model_int8: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor],
    evaluator_name: str,
    evaluator_config: str | None,
) -> None:
    evaluator = evaluators[evaluator_name]

    # Get the path of the TOSA flatbuffer that is dumped
    intermediates_path = Path(intermediates)
    tosa_paths = list(intermediates_path.glob("*.tosa"))

    if evaluator.REQUIRES_CONFIG:
        assert evaluator_config is not None

        config_path = Path(evaluator_config)
        with config_path.open() as f:
            config = json.load(f)

        if evaluator_name == "mv2":
            init_evaluator = evaluator(
                model_name,
                model_fp32,
                model_int8,
                example_inputs,
                str(tosa_paths[0]),
                config["batch_size"],
                config["validation_dataset_path"],
            )
        else:
            raise RuntimeError(f"Unknown evaluator {evaluator_name}")
    else:
        init_evaluator = evaluator(
            model_name, model_fp32, model_int8, example_inputs, str(tosa_paths[0])
        )

    quant_metrics = init_evaluator.evaluate()
    output_json_path = intermediates_path / "quant_metrics.json"

    with output_json_path.open("w") as json_file:
        json.dump(quant_metrics, json_file)


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
        help=f"Model file .py/.pth/.pt, builtin model or a model from examples/models. Valid names: {set(list(models.keys())+list(MODEL_NAME_TO_MODEL.keys()))}",
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
        choices=targets,
        help=f"For ArmBackend delegated models, pick the target, and therefore the instruction set generated. valid targets are {targets}",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        required=False,
        nargs="?",
        const="generic",
        choices=["generic", "mv2"],
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
        help="Specify custom vela configuration file (vela.ini)",
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
        help="Enable the QuantizedOpFusionPass fusion step",
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
        args.model_name in models.keys()
        and args.delegate is True
        and models[args.model_name].can_delegate is False
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


def quantize_model(args, model: torch.nn.Module, example_inputs, compile_spec):
    model_int8 = quantize(
        model,
        args.model_name,
        compile_spec,
        example_inputs,
        args.evaluate,
        args.evaluate_config,
    )
    # Wrap quantized model back into an exported_program
    exported_program = torch.export.export(
        model_int8, example_inputs, strict=args.strict_export
    )

    return model_int8, exported_program


def to_edge_TOSA_delegate(
    exported_program, args, model: torch.nn.Module, example_inputs
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
    )

    model_int8 = None
    if args.quantize:
        model_int8, exported_program = quantize_model(
            args, model, example_inputs, compile_spec
        )
        model = model_int8

    if is_ethosu(compile_spec):
        partitioner = EthosUPartitioner(compile_spec)
    elif is_tosa(compile_spec):
        partitioner = TOSAPartitioner(compile_spec)
    elif is_vgf(compile_spec):
        partitioner = VgfPartitioner(compile_spec)
    else:
        raise RuntimeError(f"Unhandled compile spec: {compile_spec}")

    edge = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    return model_int8, edge


def to_edge_no_delegate(exported_program, args, model: torch.nn.Module, example_inputs):
    model_int8 = None
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
        )
        model, exported_program = quantize_model(
            args, model, example_inputs, compile_spec
        )
        model_int8 = model

    edge = to_edge_transform_and_lower(
        exported_program,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    return model_int8, edge


def transform_for_cortex_m_backend(edge, args):
    # Let's make sure we are using optimized Cortex M backend
    # NB: If we can't find and replace ops those are expected to be replaced,
    # bad things will happen at runtime, like "missing operator" errors!

    # Instantiate the mandatory ReplaceQuantNodesPass
    passes = [ReplaceQuantNodesPass()]

    # Conditionally add the QuantizedOpFusionPass
    if args.enable_qdq_fusion_pass:
        passes.append(QuantizedOpFusionPass())

    # Apply the passes
    edge = edge.transform(passes)

    return edge


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

    if args.intermediates:
        os.makedirs(args.intermediates, exist_ok=True)

    # Quantize if required
    model_int8 = None
    if args.delegate:
        model_int8, edge = to_edge_TOSA_delegate(
            exported_program, args, model, example_inputs
        )
    else:
        model_int8, edge = to_edge_no_delegate(
            exported_program, args, model, example_inputs
        )

    if args.target != "vgf":
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

    model_name = os.path.basename(os.path.splitext(args.model_name)[0])
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
        reference_model = original_model if not model_int8 else model_int8
        save_bpte_program(exec_prog, reference_model, output_file_name)
        print(f"Bundle PTE file saved as {output_file_name}")
    else:
        save_pte_program(exec_prog, output_file_name)
        print(f"PTE file saved as {output_file_name}")

    if args.evaluate:
        evaluate_model(
            args.model_name,
            args.intermediates,
            model_fp32,
            model_int8,
            example_inputs,
            args.evaluate,
            args.evaluate_config,
        )
