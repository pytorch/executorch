# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging

from .meta_registrations import *  # noqa

from executorch.exir import ExecutorchBackendConfig
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from ...portable.utils import save_pte_program

from .compiler import export_to_edge
from .quantizer import (
    QuantFusion,
    ReplacePT2DequantWithXtensaDequant,
    ReplacePT2QuantWithXtensaQuant,
    XtensaBaseQuantizer,
)


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


def export_xtensa_model(model, example_inputs):
    # Quantizer
    quantizer = XtensaBaseQuantizer()

    # Export
    model_exp = capture_pre_autograd_graph(model, example_inputs)

    # Prepare
    prepared_model = prepare_pt2e(model_exp, quantizer)
    prepared_model(*example_inputs)

    # Convert
    converted_model = convert_pt2e(prepared_model)

    # pyre-fixme[16]: Pyre doesn't get that XtensaQuantizer has a patterns attribute
    patterns = [q.pattern for q in quantizer.quantizers]
    QuantFusion(patterns)(converted_model)

    # Get edge program (note: the name will change to export_to_xtensa in future PRs)
    edge_prog_manager = export_to_edge(converted_model, example_inputs, pt2_quant=True)

    # Run a couple required passes for quant/dequant ops
    xtensa_prog_manager = edge_prog_manager.transform(
        [ReplacePT2QuantWithXtensaQuant(), ReplacePT2DequantWithXtensaDequant()]
    )

    exec_prog = xtensa_prog_manager.to_executorch(config=ExecutorchBackendConfig())

    logging.info(f"Final exported graph module:\n{exec_prog.exported_program().graph_module}")

    # Save the program as XtensaDemoModel.pte
    save_pte_program(exec_prog, "XtensaDemoModel")
