# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch

from executorch.exir import EdgeProgramManager
from executorch.exir.dialects._ops import ops as exir_ops

from executorch.exir.pass_base import ExportPass
from executorch.exir.tensor import scalar_type_enum
from torch.fx.passes.infra.pass_base import PassResult

logger = logging.getLogger(__name__)


def quantize_input(
    exported_program, input_index, qparams: Optional[Dict[str, Any]] = None
):
    """
    Modify the program to expect quantized input at given index. The input is expected
    to be quantizing this input as the first step. Must be called before
    permute_input_layout. Returns the scale, zero point, qmin, qmax, and dtype of the
    expected quantization.
    """
    graph = exported_program.graph_module.graph
    name = exported_program.graph_signature.user_inputs[input_index]
    placeholders = [n for n in graph.nodes if n.op == "placeholder" and n.name == name]
    assert placeholders
    target_placeholder = placeholders[0]

    if len(target_placeholder.users) != 1:
        raise ValueError(f"Input {input_index} has more than one users")
    quantize = next(iter(target_placeholder.users))
    if (
        quantize.target
        != exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    ):
        raise ValueError(f"Input {input_index} is not used by a quantize op")

    # If user specified qparams are different from args of quantize op, we do requantization instead of eliminating quantize op
    need_requant = False
    if qparams is not None:
        assert all(
            qparam in qparams for qparam in ["scale", "zp", "dtype"]
        ), "dtype/scale/zp must be specified in qparam for input requantization"
        if qparams["dtype"] != quantize.args[5]:
            if any(
                dtype
                not in [torch.int8, torch.uint8, torch.bool, torch.int16, torch.uint16]
                for dtype in [qparams["dtype"], quantize.args[5]]
            ):
                raise ValueError(
                    f"Only limited data types are supported for requantization, but got {qparams['dtype']} -> {quantize.args[5]}"
                )

            need_requant = True
        elif (
            not np.isclose(qparams["scale"], quantize.args[1])
            or qparams["zp"] != quantize.args[2]
        ):
            need_requant = True

    if need_requant:
        assert qparams is not None
        dtype = qparams["dtype"]
        qmin = torch.iinfo(dtype).min
        qmax = torch.iinfo(dtype).max
        scale = qparams["scale"]
        zero_point = qparams["zp"]
        quant_args = (scale, zero_point, qmin, qmax, dtype)
        logger.info(
            f"Modifying program to requantize quantized input at index {input_index}"
        )
        logger.info(f"Quantization parameters: {quant_args}")

        with exported_program.graph_module.graph.inserting_before(quantize):
            input_dequant = exported_program.graph_module.graph.call_function(
                exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
                args=(
                    target_placeholder,
                    *quant_args,
                ),
            )
            input_dequant.meta["input_qparams"] = [
                {
                    "scale": scale,
                    "zero_point": zero_point,
                    "qmin": qmin,
                    "qmax": qmax,
                    "dtype": dtype,
                }
            ]
            input_dequant.meta["val"] = quantize.meta["val"].to(torch.float32)
            target_placeholder.meta["val"] = target_placeholder.meta["val"].to(dtype)
            quantize.replace_input_with(target_placeholder, input_dequant)
    else:
        quant_args = quantize.args[1:]
        logger.info(f"Modifying program to take quantized input at index {input_index}")
        logger.info(f"Quantization parameters: {quant_args}")

        target_placeholder.meta["val"] = (
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
                target_placeholder.meta["val"], *quant_args
            )
        )
        quantize.replace_all_uses_with(quantize.args[0])

    exported_program.graph_module.graph.eliminate_dead_code()
    return quant_args


def quantize_output(exported_program, output_index):
    """
    Modify the program to produce quantized output at given index. The model is expected
    to be dequantizing this output as the last step. Must be called before
    permute_output_layout. Returns the scale, zero point, qmin, qmax, and dtype of the
    output quantization.
    """
    graph = exported_program.graph_module.graph
    outputs = [n for n in graph.nodes if n.op == "output"]
    if len(outputs) != 1:
        raise NotImplementedError("Only 1 output node is supported")

    output_node = outputs[0]
    output_list = list(output_node.args[0])
    if output_index >= len(output_list):
        raise ValueError(
            f"{len(output_list)} outputs available, "
            + f"output index out of bounds: {output_index}"
        )

    target_output = output_list[output_index]
    if (
        target_output.target
        != exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
    ):
        raise ValueError("Output {output_index} is not a dequantize op")

    dequant = target_output
    output_list[output_index] = dequant.args[0]
    output_node.args = (output_list,)
    dequant_args = dequant.args[1:]
    graph.eliminate_dead_code()

    logger.info(
        f"Modifying program to produce quantized output at index {output_index}"
    )
    logger.info(f"Dequantization parameters: {dequant_args}")
    return dequant_args


def get_config_method_name(
    prefix: Optional[str] = "forward",
    arg_type: str = "input",
    index: int = 0,
    key: str = "scale",
):
    if prefix is None:
        prefix = ""
    else:
        prefix = prefix + "_"
    assert arg_type in ["input", "output"], "arg_type must be either input or output"
    assert index >= 0, "index must be non-negative"
    assert key in [
        "scale",
        "zp",
        "quant_min",
        "quant_max",
        "dtype",
    ], "key must be one of scale, zp, quant_min, quant_max, dtype"
    return f"{prefix}{arg_type}{index}_{key}"


class QuantizeInputs(ExportPass):
    def __init__(
        self,
        edge_program_manager: EdgeProgramManager,
        quantized_inputs_idx: Union[Dict[int, Dict[str, Any]], List[int]],
        method_name: Optional[str] = None,
    ):
        super().__init__()
        self.edge_program_manager = edge_program_manager

        self.quantized_inputs_idx_dict = {}
        if isinstance(quantized_inputs_idx, dict):
            self.quantized_inputs_idx_dict = quantized_inputs_idx
        else:
            for idx in quantized_inputs_idx:
                self.quantized_inputs_idx_dict[idx] = None
        self.param_prefix_name = method_name

    def call(self, graph_module: torch.fx.GraphModule):
        for i, qparams in self.quantized_inputs_idx_dict.items():
            quant_args = quantize_input(
                self.edge_program_manager.exported_program(), i, qparams
            )

            if not self.edge_program_manager._config_methods:
                self.edge_program_manager._config_methods = {}

            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "input", i, "scale")
            ] = quant_args[0]
            self.edge_program_manager._config_methods[  # pyre-ignore
                get_config_method_name(self.param_prefix_name, "input", i, "zp")
            ] = quant_args[1]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "input", i, "quant_min")
            ] = quant_args[2]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "input", i, "quant_max")
            ] = quant_args[3]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "input", i, "dtype")
            ] = scalar_type_enum(quant_args[4])
        return PassResult(graph_module, True)


class QuantizeOutputs(ExportPass):
    def __init__(
        self,
        edge_program_manager: EdgeProgramManager,
        quantized_outputs_idx_list: List[int],
        method_name: Optional[str] = None,
    ):
        super().__init__()
        self.edge_program_manager = edge_program_manager
        self.quantized_outputs_idx_list = quantized_outputs_idx_list
        self.param_prefix_name = method_name

    def call(self, graph_module: torch.fx.GraphModule):
        for i in self.quantized_outputs_idx_list:
            dequant_args = quantize_output(
                self.edge_program_manager.exported_program(), i
            )  # noqa F841

            if not self.edge_program_manager._config_methods:
                self.edge_program_manager._config_methods = {}

            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "output", i, "scale")
            ] = dequant_args[0]
            self.edge_program_manager._config_methods[  # pyre-ignore
                get_config_method_name(self.param_prefix_name, "output", i, "zp")
            ] = dequant_args[1]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "output", i, "quant_min")
            ] = dequant_args[2]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "output", i, "quant_max")
            ] = dequant_args[3]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "output", i, "dtype")
            ] = scalar_type_enum(dequant_args[4])

        return PassResult(graph_module, True)
