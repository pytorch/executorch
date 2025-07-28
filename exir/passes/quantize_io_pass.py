# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree

#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

import torch
import torch.fx as fx

from executorch.exir import EdgeProgramManager, ExportedProgram
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
    if quantize.target not in [
        exir_ops.edge.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
    ]:
        raise ValueError(
            f"Input {input_index} is not used by a quantize op. It's used by {quantize.target}"
        )

    if (
        quantize.target
        == exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
    ):
        replacement_op_dequant = (
            exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default
        )
        replacement_op_quant = (
            exir_ops.edge.quantized_decomposed.quantize_per_tensor.default
        )
    elif quantize.target == torch.ops.quantized_decomposed.quantize_per_tensor.default:
        replacement_op_dequant = (
            torch.ops.quantized_decomposed.dequantize_per_tensor.default
        )
        replacement_op_quant = (
            torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
    else:
        raise ValueError(f"Invalid quantize op: {quantize.target}")

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
                replacement_op_dequant,
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

        target_placeholder.meta["val"] = replacement_op_quant(
            target_placeholder.meta["val"], *quant_args
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

    output_node = graph.output_node()
    output_list = list(output_node.args[0])
    if output_index >= len(output_list):
        raise ValueError(
            f"{len(output_list)} outputs available, "
            + f"output index out of bounds: {output_index}"
        )

    target_output = output_list[output_index]
    if target_output.target not in [
        exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    ]:
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
        exported_program: Optional[ExportedProgram] = None,
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
        self.exported_program = exported_program
        self.quant_args = {}

    def edge_manager_update_quant_config_method(self, idx, quant_args):
        if self.edge_program_manager is not None:
            if not self.edge_program_manager._config_methods:
                self.edge_program_manager._config_methods = {}

            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "input", idx, "scale")
            ] = quant_args[0]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "input", idx, "zp")
            ] = quant_args[1]
            self.edge_program_manager._config_methods[
                get_config_method_name(
                    self.param_prefix_name, "input", idx, "quant_min"
                )
            ] = quant_args[2]
            self.edge_program_manager._config_methods[
                get_config_method_name(
                    self.param_prefix_name, "input", idx, "quant_max"
                )
            ] = quant_args[3]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "input", idx, "dtype")
            ] = scalar_type_enum(quant_args[4])

    def edge_manager_update_quant_config_methods_all(self):
        if self.edge_program_manager is not None:
            for idx, val in self.quant_args.items():
                self.edge_manager_update_quant_config_method(idx, val)

    def call(self, graph_module: torch.fx.GraphModule):
        for i, qparams in self.quantized_inputs_idx_dict.items():
            exported_program = (
                self.edge_program_manager.exported_program()
                if self.edge_program_manager is not None
                else self.exported_program
            )
            self.quant_args[i] = quantize_input(exported_program, i, qparams)
            self.edge_manager_update_quant_config_method(i, self.quant_args[i])

        return PassResult(graph_module, True)


class QuantizeOutputs(ExportPass):
    def __init__(
        self,
        edge_program_manager: EdgeProgramManager,
        quantized_outputs_idx_list: List[int],
        method_name: Optional[str] = None,
        exported_program: Optional[ExportedProgram] = None,
    ):
        super().__init__()
        self.edge_program_manager = edge_program_manager
        self.quantized_outputs_idx_list = quantized_outputs_idx_list
        self.param_prefix_name = method_name
        self.exported_program = exported_program
        self.dequant_args = {}

    def edge_manager_update_quant_config_method(self, idx, dequant_args):
        if self.edge_program_manager is not None:
            if not self.edge_program_manager._config_methods:
                self.edge_program_manager._config_methods = {}

            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "output", idx, "scale")
            ] = dequant_args[0]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "output", idx, "zp")
            ] = dequant_args[1]
            self.edge_program_manager._config_methods[
                get_config_method_name(
                    self.param_prefix_name, "output", idx, "quant_min"
                )
            ] = dequant_args[2]
            self.edge_program_manager._config_methods[
                get_config_method_name(
                    self.param_prefix_name, "output", idx, "quant_max"
                )
            ] = dequant_args[3]
            self.edge_program_manager._config_methods[
                get_config_method_name(self.param_prefix_name, "output", idx, "dtype")
            ] = scalar_type_enum(dequant_args[4])

    def edge_manager_update_quant_config_methods_all(self):
        if self.edge_program_manager is not None:
            for idx, val in self.dequant_args.items():
                self.edge_manager_update_quant_config_method(idx, val)

    def call(self, graph_module: torch.fx.GraphModule):
        for i in self.quantized_outputs_idx_list:
            exported_program = (
                self.edge_program_manager.exported_program()
                if self.edge_program_manager is not None
                else self.exported_program
            )
            self.dequant_args[i] = quantize_output(exported_program, i)  # noqa F841
            self.edge_manager_update_quant_config_method(i, self.dequant_args[i])

        return PassResult(graph_module, True)


def extract_io_quant_params(
    edge_prog: EdgeProgramManager,
    *,
    input_idxs: Sequence[int] = (0,),
    output_idxs: Sequence[int] = (0,),
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns quantization parameters such as scale/zero_point:
      {
        "inputs": {
          <placeholder_name>: {"scale": float, "zero_point": int}
        },
        "outputs": {
          <node_name>: {"scale": float, "zero_point": int}
        }
      }

    Note that this function will strip out the IO quantize/dequantize ops as
    it records their parameters, so if you need to preserve the original graph
    you need to make a copy with copy.deepcopy before.

    Note that `to_edge_transform_and_lower` should be called before.
    """
    # Use IO passes
    passes = []
    for idx in input_idxs:
        passes.append(QuantizeInputs(edge_prog, [idx]))
    for idx in output_idxs:
        passes.append(QuantizeOutputs(edge_prog, [idx]))

    # Apply them
    edge_prog = edge_prog.transform(passes)

    cfg = getattr(edge_prog, "_config_methods", {}) or {}

    # We need GraphModule to find node names
    gm = edge_prog.exported_program().graph_module

    input_names = _gather_io_names(gm, side="input")
    output_names = _gather_io_names(gm, side="output")

    # Build the result dict
    result = {"inputs": {}, "outputs": {}}
    for key, val in cfg.items():
        if key.startswith("input"):
            prefix, section, names = "input", "inputs", input_names
        elif key.startswith("output"):
            prefix, section, names = "output", "outputs", output_names
        else:
            continue

        idx_str, param = key[len(prefix) :].split("_", 1)
        idx = int(idx_str)
        name = names[idx]
        # We need to map 'zp' to 'zero_point'
        out_param = "zero_point" if param in ("zp", "zero_point") else param
        result[section].setdefault(name, {})[out_param] = val

    return result


def _gather_io_names(gm: fx.GraphModule, side: str):
    """
    For 'input', returns placeholder names in graph order.
    For 'output', returns names of output nodes.
    """
    if side == "input":
        return [n.name for n in gm.graph.nodes if n.op == "placeholder"]

    if side == "output":

        def _flatten(args):
            out = []

            def rec(x):
                if isinstance(x, (tuple, list)):
                    for y in x:
                        rec(y)
                elif isinstance(x, fx.Node):
                    out.append(x)

            rec(args)
            return out

        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        return [n.name for n in _flatten(output_node.args)]

    raise ValueError(f"Unknown side: {side}")
