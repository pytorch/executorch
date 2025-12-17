# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import Any, Optional, Union

import torch
from torch._inductor.decomposition import remove_decompositions
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e
from torchao.quantization.pt2e.quantizer import Quantizer

logger: logging.Logger = logging.getLogger(__name__)
QuantArgs = tuple[float, int, int, int, torch.dtype]


@torch.no_grad()
def trace(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    is_qat: bool = False,
    strict: bool = False,
    ops_to_keep: Optional[list[torch._ops.OpOverload]] = None,
) -> torch.export.ExportedProgram:
    if is_qat:
        model.train()
    else:
        model.eval()

    decomp_table = torch.export.default_decompositions()
    # pyre-fixme[6]: For 1st argument expected `Dict[typing.Callable[..., typing.Any
    remove_decompositions(decomp_table, ops_to_keep)
    program = torch.export.export(model, inputs, strict=strict).run_decompositions(
        decomp_table
    )

    return program


def prepare(
    traced_program: torch.export.ExportedProgram,
    quantizer: Quantizer,
    is_qat: bool = False,
) -> torch.fx.GraphModule:
    traced_model = traced_program.module()
    assert isinstance(traced_model, torch.fx.GraphModule)

    if is_qat:
        prepared_model = prepare_qat_pt2e(traced_model, quantizer)
    else:
        prepared_model = prepare_pt2e(traced_model, quantizer)

    return prepared_model


def extract_input_quant_params_from_graph(
    module: GraphModule,
    input_names: list[str],
) -> dict[int, QuantArgs]:
    """
    Extract quantization parameters from the FX graph for model inputs.
    """
    quant_args: dict[int, QuantArgs] = {}
    found_names: set[str] = set()

    if not input_names:
        return quant_args

    for idx, name in enumerate(input_names):
        for node in module.graph.nodes:
            if node.op != "call_function":
                continue

            if (
                node.args
                and isinstance(node.args[0], torch.fx.Node)
                and node.args[0].name == name
                and not node.name.startswith("_assert_tensor_metadata")
                and "quantize_per_tensor" in str(node.target)
            ):
                args = node.args[1:]
                if len(args) >= 5:
                    quant_args[idx] = (
                        float(args[0]),  # scale
                        int(args[1]),  # zero_point
                        int(args[2]),  # qmin
                        int(args[3]),  # qmax
                        args[4],  # dtype
                    )
                    found_names.add(name)
                break

    missing_names = set(input_names) - found_names
    if missing_names:
        raise ValueError(
            f"Could not find quantization parameters for input(s): {sorted(missing_names)}. "
            f"Make sure these input names exist in the graph and quantization parameters."
        )

    return quant_args


class QuantizedInputWrapper(torch.nn.Module):
    """
    Wrapper that allows a quantized model to accept quantized inputs.

    If no input_names or quant_args are provided, the wrapper passes inputs
    through unchanged (no dequantization).

    Args:
        module: The quantized GraphModule to wrap.
        input_names: Optional list of input placeholder names in the graph.
            If provided, extracts quant params from graph.
        quant_args: Optional dict mapping input index to (scale, zero_point, qmin, qmax, dtype).
            If provided, uses these directly instead of extracting from graph.

    Example:
        # Extract from graph
        wrapper = QuantizedInputWrapper(quantized_module, input_names=["x"])

        # Explicit quant args
        wrapper = QuantizedInputWrapper(
            quantized_module,
            quant_args={0: (1/255, 0, 0, 255, torch.uint8)},
        )
    """

    def __init__(
        self,
        module: GraphModule,
        input_args: Optional[Union[list[str], dict[int, QuantArgs]]] = None,
    ) -> None:
        super().__init__()
        self.module: GraphModule = module
        self.quant_args: dict[int, QuantArgs] = {}

        if input_args is not None:
            logger.warning(
                "Warning: Using pre-quantized inputs. This should only be done when calibration has been confirmed."
                "Incorrect quantization parameters can lead to significant accuracy degradation."
            )
        if isinstance(input_args, list):
            self.quant_args = extract_input_quant_params_from_graph(module, input_args)
        elif isinstance(input_args, dict):
            self.quant_args = input_args

    def forward(self, *args: torch.Tensor) -> Any:
        """Run inference, dequantizing configured inputs."""
        dequantized_args = []
        for index, node in enumerate(args):
            if index in self.quant_args:
                scale, zp, qmin, qmax, dtype = self.quant_args[index]
                node = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                    node, scale, zp, qmin, qmax, dtype
                )
            dequantized_args.append(node)

        return self.module(*dequantized_args)
