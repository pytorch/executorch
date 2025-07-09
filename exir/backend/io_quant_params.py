# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Sequence

import torch.fx as fx
from executorch.exir import EdgeProgramManager
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs


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
