# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import logging
import os

import torch

from executorch.backends.arm.arm_backend import ArmCompileSpecBuilder
from executorch.backends.arm.arm_partitioner import ArmPartitioner
from executorch.backends.arm.quantizer.arm_quantizer import (
    ArmQuantizer,
    get_symmetric_quantization_config,
)
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.extension.export_util.utils import export_to_edge, save_pte_program

# Quantize model if required using the standard export quantizaion flow.
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from ..models import MODEL_NAME_TO_MODEL
from ..models.model_factory import EagerModelFactory

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=FORMAT)


def get_model_and_inputs_from_name(model_name: str):
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
        model, example_inputs, _ = EagerModelFactory.create_model(
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


def quantize(model, example_inputs):
    """This is the official recommended flow for quantization in pytorch 2.0 export"""
    logging.info("Quantizing Model...")
    logging.debug(f"Original model: {model}")
    quantizer = ArmQuantizer()
    # if we set is_per_channel to True, we also need to add out_variant of quantize_per_channel/dequantize_per_channel
    operator_config = get_symmetric_quantization_config(is_per_channel=False)
    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    logging.debug(f"Quantized model: {m}")
    # make sure we can export to flat buffer
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


models = {
    "add": AddModule,
    "add2": AddModule2,
    "add3": AddModule3,
    "softmax": SoftmaxModule,
}

targets = [
    "ethos-u85-128",
    "ethos-u55-128",
    "TOSA",
]

if __name__ == "__main__":
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
        help=f"For ArmBackend delegated models, pick the target, and therefore the instruction set generated. valid targets are {targets}",
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

    args = parser.parse_args()

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

    # Pick model from one of the supported lists
    model, example_inputs = get_model_and_inputs_from_name(args.model_name)
    model = model.eval()

    # pre-autograd export. eventually this will become torch.export
    model = torch._export.capture_pre_autograd_graph(model, example_inputs)

    # Quantize if required
    if args.quantize:
        model = quantize(model, example_inputs)

    edge = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )

    # As we can target multiple output encodings from ArmBackend, one must
    # be specified.
    compile_spec = None
    if args.delegate is True:
        if args.target == "TOSA":
            compile_spec = (
                ArmCompileSpecBuilder()
                .tosa_compile_spec()
                .set_permute_memory_format(True)
            )
        elif args.target == "ethos-u55-128":
            compile_spec = (
                ArmCompileSpecBuilder()
                .ethosu_compile_spec(
                    "ethos-u55-128",
                    system_config="Ethos_U55_High_End_Embedded",
                    memory_mode="Shared_Sram",
                    extra_flags="--debug-force-regor --output-format=raw",
                )
                .set_permute_memory_format(
                    args.model_name in MODEL_NAME_TO_MODEL.keys()
                )
                .set_quantize_io(True)
            )
        elif args.target == "ethos-u85-128":
            compile_spec = (
                ArmCompileSpecBuilder()
                .ethosu_compile_spec(
                    "ethos-u85-128",
                    system_config="Ethos_U85_SYS_DRAM_Mid",
                    memory_mode="Shared_Sram",
                    extra_flags="--output-format=raw",
                )
                .set_permute_memory_format(True)
            )
        else:
            raise RuntimeError(f"Expected a target in {targets}, found {args.target}")
        if args.intermediates is not None:
            compile_spec.dump_intermediate_artifacts_to(args.intermediates)
        compile_spec = compile_spec.build()

    logging.debug(f"Exported graph:\n{edge.exported_program().graph}")
    if args.delegate is True:
        edge = edge.to_backend(ArmPartitioner(compile_spec))
        logging.debug(f"Lowered graph:\n{edge.exported_program().graph}")

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
        f"_arm_delegate_{args.target}" if args.delegate is True else ""
    )

    if args.output is not None:
        output_name = os.path.join(args.output, output_name)

    save_pte_program(exec_prog, output_name)
