# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import json
import logging
import os

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.backends.arm.arm_partitioner import ArmPartitioner
from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)

from executorch.backends.arm.util.arm_model_evaluator import (
    GenericModelEvaluator,
    MobileNetV2Evaluator,
)
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.extension.export_util.utils import save_pte_program
from tabulate import tabulate

# Quantize model if required using the standard export quantizaion flow.
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.utils.data import DataLoader

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=FORMAT)


def get_model_and_inputs_from_name(model_name: str) -> Tuple[torch.nn.Module, Any]:
    """Given the name of an example pytorch model, return it and example inputs.

    Raises RuntimeError if there is no example model corresponding to the given name.
    """
    # Case 1: Model is defined in this file
    if model_name in models.keys():
        model = models[model_name]()
        example_inputs = models[model_name].example_input
    # Case 2: Model is defined in examples/models/
    elif model_name in MODEL_NAME_TO_MODEL.keys():
        logging.warning(
            "Using a model from examples/models not all of these are currently supported"
        )
        model, example_inputs, _, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL[model_name]
        )
    # Case 3: Model is in an external python file loaded as a module.
    #         ModelUnderTest should be a torch.nn.module instance
    #         ModelInputs should be a tuple of inputs to the forward function
    elif model_name.endswith(".py"):
        import importlib.util

        # load model's module and add it
        spec = importlib.util.spec_from_file_location("tmp_model", model_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.ModelUnderTest
        example_inputs = module.ModelInputs

    else:
        raise RuntimeError(
            f"Model '{model_name}' is not a valid name. Use --help for a list of available models."
        )

    return model, example_inputs


def quantize(
    model: torch.nn.Module,
    model_name: str,
    example_inputs: Tuple[torch.Tensor],
    evaluator_name: str | None,
    evaluator_config: Dict[str, Any] | None,
) -> torch.nn.Module:
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    logging.info("Quantizing Model...")
    logging.debug(f"Original model: {model}")
    quantizer = ArmQuantizer()

    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
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


class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        z = self.softmax(x)
        return z

    example_input = (torch.ones(2, 2),)
    can_delegate = False


class MultipleOutputsModule(torch.nn.Module):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (x * y, x.sum(dim=-1, keepdim=True))

    example_input = (torch.randn(10, 4, 5), torch.randn(10, 4, 5))
    can_delegate = True


models = {
    "add": AddModule,
    "add2": AddModule2,
    "add3": AddModule3,
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
    "TOSA",
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
    reorder_inputs: Optional[str] = None,
    system_config: Optional[str] = None,
    memory_mode: Optional[str] = None,
) -> ArmCompileSpecBuilder:
    spec_builder = None
    if target == "TOSA":
        spec_builder = (
            ArmCompileSpecBuilder()
            .tosa_compile_spec("TOSA-0.80+BI")
            .set_permute_memory_format(True)
        )
    elif "ethos-u55" in target:
        spec_builder = (
            ArmCompileSpecBuilder()
            .ethosu_compile_spec(
                target,
                system_config=system_config,
                memory_mode=memory_mode,
                extra_flags="--debug-force-regor --output-format=raw --verbose-operators --verbose-cycle-estimate",
            )
            .set_permute_memory_format(True)
            .set_quantize_io(True)
            .set_input_order(reorder_inputs)
        )
    elif "ethos-u85" in target:
        spec_builder = (
            ArmCompileSpecBuilder()
            .ethosu_compile_spec(
                target,
                system_config=system_config,
                memory_mode=memory_mode,
                extra_flags="--output-format=raw --verbose-operators --verbose-cycle-estimate",
            )
            .set_permute_memory_format(True)
            .set_quantize_io(True)
            .set_input_order(reorder_inputs)
        )

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
        help=f"Provide model name. Valid ones: {set(list(models.keys())+list(MODEL_NAME_TO_MODEL.keys()))}",
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
        help="Provide path to so library. E.g., cmake-out/examples/portable/custom_ops/libcustom_ops_aot_lib.so",
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
        help="Location for outputs, if not the default of cwd.",
    )
    parser.add_argument(
        "-r",
        "--reorder_inputs",
        type=str,
        required=False,
        default=None,
        help="Provide the order of the inputs. This can be required when inputs > 1.",
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
    args = parser.parse_args()

    if args.evaluate and (
        args.quantize is None or args.intermediates is None or (not args.delegate)
    ):
        raise RuntimeError(
            "--evaluate requires --quantize, --intermediates and --delegate to be enabled."
        )

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT, force=True)

    if args.quantize and not args.so_library:
        logging.warning(
            "Quantization enabled without supplying path to libcustom_ops_aot_lib using -s flag."
            + "This is required for running quantized models with unquantized input."
        )

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

    if args.system_config is None:
        if "u55" in args.target:
            args.system_config = "Ethos_U55_High_End_Embedded"
        elif "u85" in args.target:
            args.system_confg = "Ethos_U85_SYS_DRAM_Mid"
        else:
            raise RuntimeError(f"Invalid target name {args.target}")

    if args.memory_mode is None:
        if "u55" in args.target:
            args.memory_mode = "Shared_Sram"
        elif "u85" in args.target:
            args.memory_mode = "Sram_Only"
        else:
            raise RuntimeError(f"Invalid target name {args.target}")

    return args


if __name__ == "__main__":
    args = get_args()

    # Pick model from one of the supported lists
    model, example_inputs = get_model_and_inputs_from_name(args.model_name)
    model = model.eval()

    # export_for_training under the assumption we quantize, the exported form also works
    # in to_edge if we don't quantize
    exported_program = torch.export.export_for_training(model, example_inputs)
    model = exported_program.module()
    model_fp32 = model

    # Quantize if required
    model_int8 = None
    if args.quantize:
        model = quantize(
            model, args.model_name, example_inputs, args.evaluate, args.evaluate_config
        )
        model_int8 = model
        # Wrap quantized model back into an exported_program
        exported_program = torch.export.export_for_training(model, example_inputs)

    if args.intermediates:
        os.makedirs(args.intermediates, exist_ok=True)

    if args.delegate:
        # As we can target multiple output encodings from ArmBackend, one must
        # be specified.
        compile_spec = get_compile_spec(
            args.target,
            args.intermediates,
            args.reorder_inputs,
            args.system_config,
            args.memory_mode,
        )
        edge = to_edge_transform_and_lower(
            exported_program,
            partitioner=[ArmPartitioner(compile_spec)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )
    else:
        edge = to_edge_transform_and_lower(
            exported_program,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )

    dump_delegation_info(edge, args.intermediates)

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

    if args.output is not None:
        output_name = os.path.join(args.output, output_name)

    save_pte_program(exec_prog, output_name)

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
