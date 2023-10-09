# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example script for exporting simple models to flatbuffer

import logging
from typing import Tuple

from .meta_registrations import *  # noqa

import torch
import torch._export as export

from executorch import exir
from executorch.exir import ExirExportedProgram
from executorch.exir.tracer import Value
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from .quantizer import (
    QuantFusion,
    ReplacePT2DequantWithXtensaDequant,
    ReplacePT2QuantWithXtensaQuant,
    XtensaQuantizer,
)


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)


_EDGE_COMPILE_CONFIG = exir.EdgeCompileConfig(
    _check_ir_validity=False,
)


def _to_core_aten(
    model: torch.fx.GraphModule,
    example_inputs: Tuple[Value, ...],
) -> ExirExportedProgram:
    # post autograd export. eventually this will become .to_core_aten
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            f"Expected passed in model to be an instance of fx.GraphModule, got {type(model)}"
        )
    core_aten_exir_ep = exir.capture(
        model, example_inputs, exir.CaptureConfig(enable_aot=True)
    )
    return core_aten_exir_ep


def _core_aten_to_edge(
    core_aten_exir_ep: ExirExportedProgram,
    edge_compile_config=_EDGE_COMPILE_CONFIG,
) -> ExirExportedProgram:
    edge = core_aten_exir_ep.to_edge(edge_compile_config)
    return edge


def export_to_edge(
    model: torch.fx.GraphModule,
    example_inputs: Tuple[Value, ...],
    edge_compile_config=_EDGE_COMPILE_CONFIG,
) -> ExirExportedProgram:
    core_aten_exir_ep = _to_core_aten(model, example_inputs)
    return _core_aten_to_edge(core_aten_exir_ep, edge_compile_config)


def save_pte_program(buffer, model_name):
    filename = f"{model_name}.pte"
    try:
        with open(filename, "wb") as file:
            file.write(buffer)
            logging.info(f"Saved exported program to {filename}")
    except Exception as e:
        logging.error(f"Error while saving to {filename}: {e}")


if __name__ == "__main__":
    in_features = 32
    out_features = 16
    bias = True
    shape = [64, in_features]

    class Model(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool):
            super().__init__()
            self.output_linear = torch.nn.Linear(in_features, out_features, bias=bias)
            self.layer_norm = torch.nn.LayerNorm(out_features)

        def forward(self, x: torch.Tensor):
            output_linear_out = self.output_linear(x)
            layer_norm_out = self.layer_norm(output_linear_out)
            return output_linear_out, layer_norm_out

    model = Model(in_features, out_features, bias)
    model.eval()

    example_inputs = (torch.ones(shape),)

    # Quantizer
    quantizer = XtensaQuantizer()

    # Export
    model_exp = capture_pre_autograd_graph(model, example_inputs)

    # Prepare
    prepared_model = prepare_pt2e(model_exp, quantizer)
    prepared_model(*example_inputs)

    # Convert
    converted_model = convert_pt2e(prepared_model, fold_quantize=True)

    # pyre-fixme[16]: Pyre doesn't get that XtensaQuantizer has a patterns attribute
    patterns = [q.pattern for q in quantizer.quantizers]
    QuantFusion(patterns)(converted_model)

    # pre-autograd export. eventually this will become torch.export
    converted_model_exp = export.capture_pre_autograd_graph(
        converted_model, example_inputs
    )

    converted_model_exp = torch.ao.quantization.move_exported_model_to_eval(
        converted_model_exp
    )

    core_aten_exir_ep = _to_core_aten(converted_model_exp, example_inputs)

    edge_m = _core_aten_to_edge(core_aten_exir_ep, _EDGE_COMPILE_CONFIG)

    # Replace quant/dequant ops with custom xtensa versions
    edge_m.transform(ReplacePT2QuantWithXtensaQuant())
    edge_m.transform(ReplacePT2DequantWithXtensaDequant())

    # Get executorch program
    exec_prog = edge_m.to_executorch(None)

    # Save the program as XtensaDemoModel.pte
    save_pte_program(exec_prog.buffer, "XtensaDemoModel")
