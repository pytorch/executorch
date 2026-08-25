# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
from typing import Any, Optional

import torch
from executorch.backends.cadence.aot.pass_utils import get_arg
from executorch.backends.transforms.quantize_fused_convbn_bias_pass import (
    _get_bias_tensor_ep,
    _quantize_fused_conv_bias,
    _set_param_ep,
)
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    get_param,
    is_buffer,
    is_lifted_tensor_constant,
    is_param,
)
from torch._guards import detect_fake_mode
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.infra.pass_base import PassBase, PassResult


_QAT_CONV_TARGETS: tuple[Any, ...] = (
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
)

_BN_TARGETS: tuple[Any, ...] = (
    torch.ops.aten.batch_norm.default,
    torch.ops.aten._native_batch_norm_legit_no_training.default,
)

_RESHAPE_TARGETS: tuple[Any, ...] = (
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
)

_DQ_PER_TENSOR: Any = torch.ops.quantized_decomposed.dequantize_per_tensor.default
_Q_PER_TENSOR: Any = torch.ops.quantized_decomposed.quantize_per_tensor.default


def _write_param_or_buffer(
    exported_program: ExportedProgram,
    node: torch.fx.Node,
    value: torch.Tensor,
) -> bool:
    """Overwrite the placeholder backing `node` with `value`. Returns True on
    success.

    Handles the three writable placeholder kinds: parameters, lifted tensor
    constants, and buffers (including ones introduced by constant folding,
    e.g. `b__frozen_paramN`). `_set_param_ep` only covers the first two and
    raises on buffers — but the QAT chain commonly resolves to a frozen
    buffer-backed conv bias, which still needs the BN correction folded into
    its `state_dict` slot.
    """
    sig = exported_program.graph_signature
    if (
        node.name in sig.inputs_to_parameters
        or node.name in sig.inputs_to_lifted_tensor_constants
    ):
        _set_param_ep(exported_program, node, value)
        return True
    if node.name in sig.inputs_to_buffers:
        name = sig.inputs_to_buffers[node.name]
        exported_program.state_dict[name] = value
        # Mirror `_set_param_ep`: keep node.meta in sync so downstream passes
        # see the new value as a constant FakeTensor.
        fake_mode = detect_fake_mode(
            tuple(
                n.meta["val"]
                for n in exported_program.graph.nodes
                if n.op == "placeholder"
            )
        )
        if fake_mode is not None:
            node.meta["val"] = fake_mode.from_tensor(value, static_shapes=True)
            node.meta["val"].constant = value
        return True
    return False


def _arg0_node(node: torch.fx.Node) -> torch.fx.Node:
    """Return the first positional arg as a torch.fx.Node.

    Workaround for aten ops whose first schema param is `self`:
    `Node.normalized_arguments(normalize_to_only_use_kwargs=True)` keeps `self`
    positional instead of promoting it to a kwarg, so `get_arg(node, "self", ...)`
    raises KeyError. Direct positional indexing is the only reliable read.
    """
    val = node.args[0]
    assert isinstance(
        val, torch.fx.Node
    ), f"expected fx.Node at args[0] of {node}, got {type(val).__name__}"
    return val


def _resolve_param_tensor(
    exported_program: ExportedProgram,
    node: Optional[torch.fx.Node],
) -> Optional[torch.Tensor]:
    """Read a tensor backing a placeholder node (param, buffer, or lifted constant).
    Returns None if the node is None or not resolvable as one of those kinds."""
    if node is None:
        return None
    if is_param(exported_program, node):
        return get_param(exported_program, node)
    if is_buffer(exported_program, node):
        return get_buffer(exported_program, node)
    if is_lifted_tensor_constant(exported_program, node):
        return get_lifted_tensor_constant(exported_program, node)
    return None


class FuseQATConvBN(PassBase):
    """
    Folds the QAT Conv-BN simulation chain (inserted by `prepare_qat_pt2e`) into
    the conv's quantized bias. Cleans up `batch_norm` nodes and the surrounding
    sqrt/div/add ops that TorchAO's `_fold_conv_bn_qat` matcher fails to fold
    when Cadence's quantizer annotates conv biases with INT32 quantization.

    The chain looks like:
        conv -> q -> dq -> div(scale) -> add(orig_bias) -> batch_norm
    where scale = bn_weight / sqrt(running_var + eps).

    Two-step `call()`:
      1. Bias prep - for each conv, create a zero-filled quantized bias if
         missing, or quantize a float bias as per-tensor int32. Required so
         step 2 has a quantized bias slot to write the BN correction into.
      2. Fold - for each matched chain, compute the BN correction
            C = (orig_bias - running_mean) * bn_weight / sqrt(running_var + eps) + bn_bias
         and absorb it into the conv's quantized bias in place. Erase the chain
         + batch_norm node.

    Always runs after `convert_pt2e`, so params are placeholders inside an
    `ExportedProgram` (no `get_attr` nodes). `exported_program` is required.
    """

    def __init__(
        self,
        exported_program: ExportedProgram,
        default_zero_bias: bool = True,
    ) -> None:
        super().__init__()
        self.exported_program = exported_program
        self.default_zero_bias = default_zero_bias

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Step 1: prep biases so step 2 has quantized bias slots to write into.
        prep_modified = self._prep_conv_biases(graph_module)

        # Step 2: fold the BN correction into the (now-quantized) bias and
        # delete the simulation chain + batch_norm.
        fold_modified = self._fold_qat_chains(graph_module)

        if prep_modified or fold_modified:
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, prep_modified or fold_modified)

    def _prep_conv_biases(self, graph_module: torch.fx.GraphModule) -> bool:
        """Delegate bias prep to the shared helper. Creates zero biases for
        biasless convs and quantizes any float biases."""
        ep = self.exported_program
        return _quantize_fused_conv_bias(
            graph_module,
            conv_targets=_QAT_CONV_TARGETS + (torch.ops.aten.conv_transpose2d.input,),
            unsqueeze_targets=(
                torch.ops.aten.unsqueeze_copy.default,
                torch.ops.aten.unsqueeze.default,
            ),
            dq_per_tensor=_DQ_PER_TENSOR,
            dq_per_channel=torch.ops.quantized_decomposed.dequantize_per_channel.default,
            get_bias_tensor=lambda n: _get_bias_tensor_ep(ep, n),
            set_param=lambda n, t, insert_before=None: _set_param_ep(ep, n, t),
            get_weight_scale_tensor=lambda n: get_buffer(ep, n),
            default_zero_bias=self.default_zero_bias,
        )

    def _fold_qat_chains(self, graph_module: torch.fx.GraphModule) -> bool:
        """Walk batch_norm nodes, match the QAT simulation chain, and fold
        the BN correction into the conv bias."""
        ep = self.exported_program
        graph = graph_module.graph
        nodes_to_erase: list[torch.fx.Node] = []
        changed = False

        for bn_node in list(graph.nodes):
            if bn_node.target not in _BN_TARGETS:
                continue
            match = self._match_qat_chain(bn_node)
            if match is None:
                continue

            bn_weight = _resolve_param_tensor(ep, match["bn_weight"])
            bn_bias = _resolve_param_tensor(ep, match["bn_bias"])
            bn_mean = _resolve_param_tensor(ep, match["bn_mean"])
            bn_var = _resolve_param_tensor(ep, match["bn_var"])
            conv_bias = _resolve_param_tensor(ep, match["conv_bias_param"])
            orig_bias = _resolve_param_tensor(ep, match["orig_bias_node"])
            # orig_bias is allowed to be None (no add_orig_bias branch). Other
            # tensors must resolve; if any fail, skip this match.
            if (
                bn_weight is None
                or bn_bias is None
                or bn_mean is None
                or bn_var is None
                or conv_bias is None
            ):
                continue
            if match["orig_bias_node"] is not None and orig_bias is None:
                continue

            tensors: dict[str, torch.Tensor] = {
                "bn_weight": bn_weight,
                "bn_bias": bn_bias,
                "bn_mean": bn_mean,
                "bn_var": bn_var,
                "conv_bias": conv_bias,
            }
            new_bias = self._compute_folded_bias(match, tensors, orig_bias)
            if not _write_param_or_buffer(ep, match["conv_bias_param"], new_bias):
                continue

            self._rewire_and_collect_erase(bn_node, match, nodes_to_erase)
            changed = True

        for node in reversed(nodes_to_erase):
            if len(node.users) == 0:
                graph.erase_node(node)
        return changed

    @staticmethod
    def _match_qat_chain(  # noqa: C901
        bn_node: torch.fx.Node,
    ) -> Optional[dict[str, Any]]:
        """Walk back from a batch_norm node and return the matched chain
        components, or None if the pattern doesn't match."""
        if bn_node.target == torch.ops.aten.batch_norm.default:
            if get_arg(bn_node, "training", bool) is not False:
                return None
        eps = get_arg(bn_node, "eps", float)

        # BN input is either `add(div, reshape(orig_bias))` (conv had a bias)
        # or `div` directly (no original bias).
        bn_input = get_arg(bn_node, "input", torch.fx.Node)
        if bn_input.target == torch.ops.aten.add.Tensor:
            add_orig_bias: Optional[torch.fx.Node] = bn_input
            div_output = _arg0_node(bn_input)
            reshape_orig_bias: Optional[torch.fx.Node] = get_arg(
                bn_input, "other", torch.fx.Node
            )
            if div_output.target != torch.ops.aten.div.Tensor:
                return None
        elif bn_input.target == torch.ops.aten.div.Tensor:
            add_orig_bias = None
            reshape_orig_bias = None
            div_output = bn_input
        else:
            return None

        # Scale chain: div_output = div(dq_intermediate, reshape(div(_, sqrt(add(running_var, eps)))))
        dq_intermediate = _arg0_node(div_output)
        reshape_scale = get_arg(div_output, "other", torch.fx.Node)
        if dq_intermediate.target != _DQ_PER_TENSOR:
            return None
        if reshape_scale.target not in _RESHAPE_TARGETS:
            return None
        scale_node = _arg0_node(reshape_scale)
        if scale_node.target != torch.ops.aten.div.Tensor:
            return None
        sqrt_node = get_arg(scale_node, "other", torch.fx.Node)
        if sqrt_node.target != torch.ops.aten.sqrt.default:
            return None
        add_var_eps = _arg0_node(sqrt_node)
        if add_var_eps.target != torch.ops.aten.add.Tensor:
            return None

        # Conv chain: dq_intermediate <- q_intermediate <- conv(.., bias=conv_bias_dq)
        q_intermediate = get_arg(dq_intermediate, "input", torch.fx.Node)
        if q_intermediate.target != _Q_PER_TENSOR:
            return None
        conv_node = get_arg(q_intermediate, "input", torch.fx.Node)
        if conv_node.target not in _QAT_CONV_TARGETS:
            return None
        conv_bias_dq = get_arg(conv_node, "bias", torch.fx.Node)
        if conv_bias_dq.target != _DQ_PER_TENSOR:
            return None

        # When the chain has the add(orig_bias) branch, the orig_bias is reshaped
        # before the add. Resolve through the reshape; absent reshape = no match.
        orig_bias_node: Optional[torch.fx.Node] = None
        if reshape_orig_bias is not None:
            if reshape_orig_bias.target not in _RESHAPE_TARGETS:
                return None
            orig_bias_node = _arg0_node(reshape_orig_bias)

        return {
            "eps": eps,
            "bn_weight": get_arg(bn_node, "weight", torch.fx.Node),
            "bn_bias": get_arg(bn_node, "bias", torch.fx.Node),
            "bn_mean": get_arg(bn_node, "running_mean", torch.fx.Node),
            "bn_var": get_arg(bn_node, "running_var", torch.fx.Node),
            "div_output": div_output,
            "reshape_scale": reshape_scale,
            "scale_node": scale_node,
            "sqrt_node": sqrt_node,
            "add_var_eps": add_var_eps,
            "dq_intermediate": dq_intermediate,
            "add_orig_bias": add_orig_bias,
            "reshape_orig_bias": reshape_orig_bias,
            "orig_bias_node": orig_bias_node,
            "conv_bias_param": get_arg(conv_bias_dq, "input", torch.fx.Node),
            "bias_scale": get_arg(conv_bias_dq, "scale", float),
            "bias_zp": get_arg(conv_bias_dq, "zero_point", int),
            "bias_qmin": get_arg(conv_bias_dq, "quant_min", int),
            "bias_qmax": get_arg(conv_bias_dq, "quant_max", int),
        }

    @staticmethod
    def _compute_folded_bias(
        match: dict[str, Any],
        tensors: dict[str, torch.Tensor],
        orig_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute new int bias = round((bias_float + C) / scale) + zp, clamped."""
        scale = match["bias_scale"]
        zp = match["bias_zp"]
        qmin = match["bias_qmin"]
        qmax = match["bias_qmax"]

        running_std = torch.sqrt(tensors["bn_var"] + match["eps"])
        if orig_bias is not None:
            correction = (orig_bias - tensors["bn_mean"]) * tensors[
                "bn_weight"
            ] / running_std + tensors["bn_bias"]
        else:
            correction = (
                -tensors["bn_mean"] * tensors["bn_weight"] / running_std
                + tensors["bn_bias"]
            )
        bias_float = (tensors["conv_bias"].float() - zp) * scale
        new_bias_float = bias_float + correction
        return torch.clamp(torch.round(new_bias_float / scale) + zp, qmin, qmax).to(
            tensors["conv_bias"].dtype
        )

    @staticmethod
    def _rewire_and_collect_erase(
        bn_node: torch.fx.Node,
        match: dict[str, Any],
        nodes_to_erase: list[torch.fx.Node],
    ) -> None:
        """Replace BN output with the dequantized conv output and queue the
        intermediate ops for deletion."""
        if (
            bn_node.target
            == torch.ops.aten._native_batch_norm_legit_no_training.default
        ):
            for user in list(bn_node.users):
                assert user.target == operator.getitem
                if user.args[1] == 0:
                    user.replace_all_uses_with(match["dq_intermediate"])
                nodes_to_erase.append(user)
        else:
            bn_node.replace_all_uses_with(match["dq_intermediate"])

        nodes_to_erase.extend(
            n
            for n in [
                bn_node,
                match["div_output"],
                match["reshape_scale"],
                match["scale_node"],
                match["sqrt_node"],
                match["add_var_eps"],
                match["add_orig_bias"],
                match["reshape_orig_bias"],
            ]
            if n is not None
        )
