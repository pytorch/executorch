# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from .meta_registrations import *  # noqa

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from ...portable.utils import save_pte_program

from .compiler import export_to_edge
from .quantizer import (
    CadenceBaseQuantizer,
    QuantFusion,
    ReplacePT2DequantWithCadenceDequant,
    ReplacePT2QuantWithCadenceQuant,
)
from .utils import print_ops_info


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def export_model(model, example_inputs):
    # Quantizer
    quantizer = CadenceBaseQuantizer()

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

    # Get edge program (note: the name will change to export_to_cadence in future PRs)
    edge_prog_manager, expo_prog = export_to_edge(
        converted_model, example_inputs, pt2_quant=True
    )

    # Run a couple required passes for quant/dequant ops
    cadence_prog_manager = edge_prog_manager.transform(
        [ReplacePT2QuantWithCadenceQuant(), ReplacePT2DequantWithCadenceDequant()],
        check_ir_validity=False,
    )

    exec_prog = cadence_prog_manager.to_executorch()

    logging.info(
        f"Final exported graph module:\n{exec_prog.exported_program().graph_module}"
    )

    # Print some information to terminal
    print_ops_info(
        expo_prog.graph_module,
        edge_prog_manager.exported_program().graph_module,
        cadence_prog_manager.exported_program().graph_module,
    )

    # Save the program as CadenceDemoModel.pte
    save_pte_program(exec_prog, "CadenceDemoModel")
