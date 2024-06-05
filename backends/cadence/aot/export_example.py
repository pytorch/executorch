# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from executorch.backends.cadence.aot.ops_registrations import *  # noqa

import os
from typing import Any, Tuple

from executorch.backends.cadence.aot.compiler import export_to_cadence, export_to_edge
from executorch.backends.cadence.aot.quantizer.fusion_pass import QuantFusion
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceQuantizer
from executorch.exir import ExecutorchProgramManager
from torch import nn
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from .utils import print_ops_info


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def _save_pte_program(
    prog: ExecutorchProgramManager, model_name: str, output_dir: str = ""
) -> None:
    if model_name.endswith(".pte"):
        filename = model_name
    else:
        filename = os.path.join(output_dir, f"{model_name}.pte")

    try:
        with open(filename, "wb") as file:
            prog.write_to_file(file)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")


def export_model(
    model: nn.Module,
    example_inputs: Tuple[Any, ...],
    file_name: str = "CadenceDemoModel",
):
    # Quantizer
    quantizer = CadenceQuantizer()

    # Export
    model_exp = capture_pre_autograd_graph(model, example_inputs)

    # Prepare
    prepared_model = prepare_pt2e(model_exp, quantizer)
    prepared_model(*example_inputs)

    # Convert
    converted_model = convert_pt2e(prepared_model)

    # pyre-fixme[16]: Pyre doesn't get that CadenceQuantizer has a patterns attribute
    patterns = [q.pattern for q in quantizer.quantizers]
    QuantFusion(patterns)(converted_model)

    # Get edge program
    edge_prog_manager = export_to_edge(converted_model, example_inputs)

    # Get edge program after Cadence specific passes
    cadence_prog_manager = export_to_cadence(converted_model, example_inputs)

    exec_prog = cadence_prog_manager.to_executorch()

    logging.info("Final exported graph:")
    exec_prog.exported_program().graph_module.graph.print_tabular()

    # Print some information to terminal
    print_ops_info(
        edge_prog_manager.exported_program().graph_module,
        cadence_prog_manager.exported_program().graph_module,
    )

    # Save the program as (default name is CadenceDemoModel.pte)
    _save_pte_program(exec_prog, file_name)
