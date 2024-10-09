# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import argparse
import logging

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

    # 1. pick model from one of the supported lists
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
    logging.debug(f"Exported graph:\n{edge.exported_program().graph}")
    if args.delegate is True:
        edge = edge.to_backend(
            ArmPartitioner(
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
                .build()
            )
        )
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

    model_name = f"{args.model_name}" + (
        "_arm_delegate" if args.delegate is True else ""
    )
    save_pte_program(exec_prog, model_name)
