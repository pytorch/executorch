# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import operator
from typing import Optional

import torch
from torch import fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger: logging.Logger = logging.getLogger(__name__)


def _build_placeholder_key_map(
    exported_program: "torch.export.ExportedProgram",
) -> dict[str, str]:
    """Build a map from placeholder-style names (underscored) to state_dict keys (dotted).

    ExportedProgram placeholder nodes use names like `p_input_bn_0_weight` or
    `b_input_bn_0_running_mean`. The corresponding state_dict keys use dotted
    module paths like `input_bn.0.weight`. This builds a reverse lookup by
    flattening each dotted key with the `p_`/`b_` prefixes that `torch.export`
    generates.
    """
    result: dict[str, str] = {}

    def _add_key(dotted_key: str, prefix: str) -> None:
        flat = prefix + dotted_key.replace(".", "_")
        result[flat] = dotted_key

    for key in exported_program.state_dict:
        _add_key(key, "p_")
        _add_key(key, "b_")
        _add_key(key, "arg_")
        result[key] = key

    for name, _ in exported_program.named_buffers():
        _add_key(name, "b_")
        _add_key(name, "p_")
        _add_key(name, "arg_")
        result[name] = name

    constants = getattr(exported_program, "constants", None) or {}
    for key in constants:
        _add_key(key, "b_")
        _add_key(key, "p_")

    return result


def _resolve_placeholder_tensor(
    node_name: str,
    exported_program: "torch.export.ExportedProgram",
    key_map: dict[str, str],
) -> Optional[torch.Tensor]:
    dotted = key_map.get(node_name)
    if dotted is None:
        return None

    sd = exported_program.state_dict
    if dotted in sd:
        val = sd[dotted]
        return val.data if isinstance(val, torch.nn.Parameter) else val

    constants = getattr(exported_program, "constants", None) or {}
    if dotted in constants:
        return constants[dotted]

    for name, buf in exported_program.named_buffers():
        if name == dotted:
            return buf

    return None


def _get_tensor(
    graph_module: fx.GraphModule,
    node: fx.Node,
    exported_program: Optional["torch.export.ExportedProgram"] = None,
    key_map: Optional[dict[str, str]] = None,
) -> Optional[torch.Tensor]:
    if node.op == "get_attr":
        parts = node.target.split(".")
        obj = graph_module
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        if isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
            return obj.data if isinstance(obj, torch.nn.Parameter) else obj
        return None
    if (
        node.op == "placeholder"
        and exported_program is not None
        and key_map is not None
    ):
        return _resolve_placeholder_tensor(node.name, exported_program, key_map)
    return None


def _set_tensor(
    graph_module: fx.GraphModule,
    node: fx.Node,
    value: torch.Tensor,
    exported_program: Optional["torch.export.ExportedProgram"] = None,
    key_map: Optional[dict[str, str]] = None,
) -> bool:
    if node.op == "get_attr":
        parts = node.target.split(".")
        obj = graph_module
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return False
        setattr(obj, parts[-1], value)
        return True
    if (
        node.op == "placeholder"
        and exported_program is not None
        and key_map is not None
    ):
        dotted = key_map.get(node.name)
        if dotted is None:
            return False
        sd = exported_program.state_dict
        if dotted in sd:
            if isinstance(sd[dotted], torch.nn.Parameter):
                sd[dotted] = torch.nn.Parameter(value, requires_grad=False)
            else:
                sd[dotted] = value
            return True
    return False


class FoldQATConvBNPass(PassBase):
    """
    Fold QAT Conv-BN simulated fusion patterns by absorbing the BN bias
    correction into the conv bias tensor and removing the simulation chain.

    Math: bn_out = dq(q(conv_out)) + C
    where C = (orig_bias - running_mean) * bn_weight / sqrt(running_var + eps) + bn_bias

    Instead of creating new constant nodes, C is folded into the conv's
    existing quantized bias parameter.
    """

    def __init__(
        self,
        exported_program: Optional["torch.export.ExportedProgram"] = None,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        graph = graph_module.graph
        changed = False
        nodes_to_erase = []
        key_map = (
            _build_placeholder_key_map(self.exported_program)
            if self.exported_program is not None
            else None
        )

        for bn_node in list(graph.nodes):
            if bn_node.target not in (
                torch.ops.aten.batch_norm.default,
                torch.ops.aten._native_batch_norm_legit_no_training.default,
            ):
                continue

            if bn_node.target == torch.ops.aten.batch_norm.default:
                if bn_node.args[5] is not False:
                    continue
                bn_weight_node = bn_node.args[1]
                bn_bias_node = bn_node.args[2]
                bn_mean_node = bn_node.args[3]
                bn_var_node = bn_node.args[4]
                eps = bn_node.args[7]
            else:
                bn_weight_node = bn_node.args[1]
                bn_bias_node = bn_node.args[2]
                bn_mean_node = bn_node.args[3]
                bn_var_node = bn_node.args[4]
                eps = bn_node.args[6]

            bn_input = bn_node.args[0]
            has_orig_bias = False
            add_before_bn = None
            reshape_orig_bias = None

            if bn_input.target == torch.ops.aten.add.Tensor:
                add_before_bn = bn_input
                div_output = add_before_bn.args[0]
                reshape_orig_bias = add_before_bn.args[1]
                if div_output.target == torch.ops.aten.div.Tensor:
                    has_orig_bias = True
                else:
                    continue
            elif bn_input.target == torch.ops.aten.div.Tensor:
                div_output = bn_input
            else:
                continue

            dq_intermediate = div_output.args[0]
            reshape_scale = div_output.args[1]

            if (
                dq_intermediate.target
                != torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                continue

            if reshape_scale.target not in (
                torch.ops.aten.reshape.default,
                torch.ops.aten.view.default,
            ):
                continue
            scale_node = reshape_scale.args[0]
            if scale_node.target != torch.ops.aten.div.Tensor:
                continue
            sqrt_node = scale_node.args[1]
            if sqrt_node.target != torch.ops.aten.sqrt.default:
                continue
            add_var_eps = sqrt_node.args[0]
            if add_var_eps.target != torch.ops.aten.add.Tensor:
                continue

            # Trace back to find the conv node through the q/dq chain
            q_intermediate = dq_intermediate.args[0]
            if (
                q_intermediate.target
                != torch.ops.quantized_decomposed.quantize_per_tensor.default
            ):
                continue
            conv_node = q_intermediate.args[0]
            if conv_node.target not in (
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.convolution.default,
            ):
                continue

            # Find the conv bias dq node
            if len(conv_node.args) < 3:
                continue
            conv_bias_dq = conv_node.args[2]
            if not isinstance(conv_bias_dq, fx.Node):
                continue
            if (
                conv_bias_dq.target
                != torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                continue
            conv_bias_param_node = conv_bias_dq.args[0]
            bias_scale = conv_bias_dq.args[1]
            bias_zp = conv_bias_dq.args[2]
            bias_qmin = conv_bias_dq.args[3]
            bias_qmax = conv_bias_dq.args[4]

            # Get all tensor values
            orig_bias = None
            if has_orig_bias:
                if reshape_orig_bias.target not in (
                    torch.ops.aten.reshape.default,
                    torch.ops.aten.view.default,
                ):
                    continue
                orig_bias_node = reshape_orig_bias.args[0]
                orig_bias = _get_tensor(
                    graph_module, orig_bias_node, self.exported_program, key_map
                )

            bn_weight = _get_tensor(
                graph_module, bn_weight_node, self.exported_program, key_map
            )
            bn_bias = _get_tensor(
                graph_module, bn_bias_node, self.exported_program, key_map
            )
            running_mean = _get_tensor(
                graph_module, bn_mean_node, self.exported_program, key_map
            )
            running_var = _get_tensor(
                graph_module, bn_var_node, self.exported_program, key_map
            )
            conv_bias_tensor = _get_tensor(
                graph_module, conv_bias_param_node, self.exported_program, key_map
            )

            required = [bn_weight, bn_bias, running_mean, running_var, conv_bias_tensor]
            if has_orig_bias:
                required.append(orig_bias)
            if any(t is None for t in required):
                continue

            # Compute fused bias correction C
            if has_orig_bias:
                C = (orig_bias - running_mean) * bn_weight / torch.sqrt(
                    running_var + eps
                ) + bn_bias
            else:
                C = -running_mean * bn_weight / torch.sqrt(running_var + eps) + bn_bias

            # Fold C into the conv bias: dequant existing bias, add C, requant
            bias_float = (conv_bias_tensor.float() - bias_zp) * bias_scale
            new_bias_float = bias_float + C
            new_bias_int = torch.clamp(
                torch.round(new_bias_float / bias_scale) + bias_zp,
                bias_qmin,
                bias_qmax,
            ).to(conv_bias_tensor.dtype)

            _set_tensor(
                graph_module,
                conv_bias_param_node,
                new_bias_int,
                self.exported_program,
                key_map,
            )

            # Rewire: dq_intermediate -> bn_output users
            if (
                bn_node.target
                == torch.ops.aten._native_batch_norm_legit_no_training.default
            ):
                for user in list(bn_node.users.keys()):
                    if user.target == operator.getitem and user.args[1] == 0:
                        user.replace_all_uses_with(dq_intermediate)
                        nodes_to_erase.append(user)
                    elif user.target == operator.getitem:
                        nodes_to_erase.append(user)
            else:
                bn_node.replace_all_uses_with(dq_intermediate)

            # Remove simulation chain and batch_norm
            remove_candidates = [
                bn_node,
                div_output,
                reshape_scale,
                scale_node,
                sqrt_node,
                add_var_eps,
            ]
            if add_before_bn is not None:
                remove_candidates.append(add_before_bn)
            if reshape_orig_bias is not None:
                remove_candidates.append(reshape_orig_bias)

            nodes_to_erase.extend(remove_candidates)
            changed = True

        for node in reversed(nodes_to_erase):
            if len(node.users) == 0:
                graph.erase_node(node)

        if changed:
            graph.eliminate_dead_code()
            graph_module.recompile()

        return PassResult(graph_module, changed)
