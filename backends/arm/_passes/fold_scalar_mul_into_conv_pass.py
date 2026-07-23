# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Set, Type

import torch
from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.arm_pass_utils import (
    get_constant_placeholder_kind,
    get_param_tensor,
    is_carried_quant_state,
)
from executorch.backends.transforms.utils import (
    create_constant_placeholder,
    delete_constant_placeholder,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx import GraphModule, Node
from torch.fx.node import Argument

ConvScaleMatch = tuple[Node, torch.Tensor, Optional[Node]]


class FoldScalarMulIntoConvPass(ArmPass):
    """Fold constant output-channel scaling after convolution.

    Rewrites patterns equivalent to ``conv(x, weight, bias, ...) * scale``
    to a convolution with scaled weight and bias constants. ``scale`` may be
    a Python scalar, a scalar tensor constant, or a tensor constant
    broadcastable over the convolution output with only the channel dimension
    non-unit.

    The rewrite is exact in floating point, but it moves the scaling *inside*
    the convolution and therefore changes the convolution output's quantization
    (its activation range, and hence its requantization). That is harmless for a
    plain feed-forward output, but if the (scaled) convolution output feeds a
    *carried/stateful quantization boundary* -- a value whose quantization is
    shared across invocations (recurrent state) -- the shifted requantization
    drifts the carried state and the error compounds over the sequence. To stay
    safe, the fold is skipped when the convolution output transitively reaches
    such a boundary, identified as either:

    * a value written into a mutable buffer / input (``BUFFER_MUTATION`` /
      ``USER_INPUT_MUTATION`` in the graph signature), which is structurally
      visible; or
    * a value the model marked with ``mark_carried_quant_state`` -- needed for
      *functional* carried state (state passed in and returned as plain I/O),
      which has no structural signal before quantization runs.

    """

    _passes_required_after: Set[Type[ExportPass]] = set()

    _CONV_OPS = {
        torch.ops.aten.conv2d.default,
        torch.ops.aten.convolution.default,
        exir_ops.edge.aten.convolution.default,
    }
    _MUL_TENSOR_OPS = {
        torch.ops.aten.mul.Tensor,
        exir_ops.edge.aten.mul.Tensor,
    }
    _MUL_SCALAR_OPS = {
        torch.ops.aten.mul.Scalar,
        exir_ops.edge.aten.mul.Scalar,
    }

    def __init__(
        self,
        exported_program: Optional[torch.export.ExportedProgram] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.exported_program = exported_program

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        modified = False
        constant_placeholders_to_delete: set[Node] = set()
        # Carried/stateful quant boundaries: values written to a mutable buffer /
        # input (structurally visible), plus any value the model explicitly marked
        # as carried quantized state (functional recurrent state, which has no
        # structural signal before quantization).
        state_boundary_names = self._state_mutation_value_names()
        state_boundary_names |= {
            node.name for node in graph.nodes if is_carried_quant_state(node)
        }

        for node in list(graph.nodes):
            if not self.allowed_to_transform(node.meta):
                continue
            match = self._match_mul_after_conv(node)
            if match is None:
                continue

            # Do not fold when the (scaled) conv output feeds a carried/stateful
            # quant boundary: moving the scale into the weights shifts the conv
            # output requantization, which drifts persisted quantized state.
            if self._feeds_state_boundary(node, state_boundary_names):
                continue

            conv_node, scale, scale_node = match
            folded_placeholders = self._fold_scale_into_conv(
                graph, conv_node, node, scale
            )
            if folded_placeholders is not None:
                constant_placeholders_to_delete.update(folded_placeholders)
                if scale_node is not None:
                    constant_placeholders_to_delete.add(scale_node)
                modified = True

        if modified:
            graph.eliminate_dead_code()
            if self.exported_program is not None:
                for constant_placeholder in constant_placeholders_to_delete:
                    if len(constant_placeholder.users) == 0:
                        delete_constant_placeholder(
                            self.exported_program, constant_placeholder
                        )
            graph.lint()
            graph_module.recompile()

        return PassResult(graph_module, modified)

    def _state_mutation_value_names(self) -> Set[str]:
        """Names of graph output values that write a carried/stateful boundary.

        These are the values bound to ``BUFFER_MUTATION`` / ``USER_INPUT_MUTATION``
        outputs in the exported program's graph signature -- i.e. mutable buffers
        or inputs whose value persists across invocations (recurrent state). When
        the graph signature is unavailable (``exported_program is None``) no state
        boundary can be identified and the set is empty.

        """
        if self.exported_program is None:
            return set()
        names: Set[str] = set()
        for spec in self.exported_program.graph_signature.output_specs:
            if spec.kind in (
                OutputKind.BUFFER_MUTATION,
                OutputKind.USER_INPUT_MUTATION,
            ):
                name = getattr(spec.arg, "name", None)
                if name is not None:
                    names.add(name)
        return names

    def _feeds_state_boundary(self, node: Node, state_names: Set[str]) -> bool:
        """True if ``node``'s output transitively reaches a state-mutation
        value.
        """
        if not state_names:
            return False
        seen: Set[Node] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current.name in state_names:
                return True
            stack.extend(current.users)
        return False

    def _match_mul_after_conv(self, node: Node) -> Optional[ConvScaleMatch]:
        if node.op != "call_function":
            return None

        candidates: tuple[tuple[Argument, Argument], ...]
        if node.target in self._MUL_TENSOR_OPS:
            lhs, rhs = node.args[:2]
            candidates = ((lhs, rhs), (rhs, lhs))
        elif node.target in self._MUL_SCALAR_OPS:
            candidates = ((node.args[0], node.args[1]),)
        else:
            return None

        for conv_candidate, scale_candidate in candidates:
            if not isinstance(conv_candidate, Node):
                continue
            if not self._is_foldable_conv(conv_candidate, node):
                continue
            scale = self._get_scale_tensor(scale_candidate)
            if scale is None:
                continue
            channel_scale = self._get_channel_scale(conv_candidate, scale)
            if channel_scale is None:
                continue
            scale_node = scale_candidate if isinstance(scale_candidate, Node) else None
            return conv_candidate, channel_scale, scale_node

        return None

    def _is_foldable_conv(self, conv_node: Node, mul_node: Node) -> bool:
        is_call_function = conv_node.op == "call_function"
        is_convolution = conv_node.target in self._CONV_OPS
        if not (is_call_function and is_convolution):
            return False
        if len(conv_node.users) != 1 or mul_node not in conv_node.users:
            return False
        if conv_node.target == torch.ops.aten.conv2d.default:
            return len(conv_node.args) >= 3
        if len(conv_node.args) < 9:
            return False
        transposed = bool(conv_node.args[6])
        return not transposed

    def _get_scale_tensor(self, scale) -> Optional[torch.Tensor]:
        if isinstance(scale, (int, float)):
            return torch.tensor(float(scale), dtype=torch.float32)
        if not isinstance(scale, Node):
            return None
        try:
            scale_tensor = self._get_constant_tensor(scale)
        except RuntimeError:
            return None
        if scale_tensor is None:
            return None
        if not torch.is_floating_point(scale_tensor):
            return None
        return scale_tensor.detach()

    def _get_channel_scale(
        self, conv_node: Node, scale: torch.Tensor
    ) -> Optional[torch.Tensor]:
        output_val = conv_node.meta.get("val")
        if output_val is None or not hasattr(output_val, "shape"):
            return None
        output_shape = tuple(output_val.shape)
        if len(output_shape) < 2:
            return None
        out_channels = int(output_shape[1])

        if scale.numel() == 1:
            return scale.reshape(1).expand(out_channels).clone()

        if scale.numel() != out_channels:
            return None

        scale_shape = tuple(scale.shape)
        if len(scale_shape) > len(output_shape):
            return None

        rank_padding = (1,) * (len(output_shape) - len(scale_shape))
        padded_shape = rank_padding + scale_shape
        for dim, size in enumerate(padded_shape):
            if dim == 1:
                if int(size) not in (1, out_channels):
                    return None
            elif int(size) != 1:
                return None

        return scale.reshape(out_channels)

    def _fold_scale_into_conv(  # noqa: C901
        self,
        graph: torch.fx.Graph,
        conv_node: Node,
        mul_node: Node,
        channel_scale: torch.Tensor,
    ) -> Optional[set[Node]]:
        weight_node = conv_node.args[1]
        bias_node = conv_node.args[2]
        if not isinstance(weight_node, Node):
            return None
        if bias_node is not None and not isinstance(bias_node, Node):
            return None

        try:
            weight = self._get_constant_tensor(weight_node)
            bias = (
                self._get_constant_tensor(bias_node)
                if isinstance(bias_node, Node)
                else None
            )
        except RuntimeError:
            return None

        if weight is None:
            return None
        if not torch.is_floating_point(weight):
            return None
        if bias is not None and not torch.is_floating_point(bias):
            return None

        scale = channel_scale.to(device=weight.device, dtype=weight.dtype)
        scaled_weight = weight.detach().clone() * scale.reshape(
            (-1,) + (1,) * (weight.dim() - 1)
        )
        scaled_bias = None
        if bias is not None:
            scaled_bias = bias.detach().clone() * scale.to(
                device=bias.device, dtype=bias.dtype
            )

        if self.exported_program is None:
            if not self._can_mutate_get_attr(weight_node, conv_node):
                return None
            if isinstance(bias_node, Node) and not self._can_mutate_get_attr(
                bias_node, conv_node
            ):
                return None
            if not self._set_get_attr_tensor(weight_node, scaled_weight):
                return None
            if isinstance(bias_node, Node) and scaled_bias is not None:
                if not self._set_get_attr_tensor(bias_node, scaled_bias):
                    return None
            scaled_weight_node = weight_node
            scaled_bias_node = bias_node
            folded_placeholders = set()
        else:
            weight_kind = self._constant_kind(weight_node, InputKind.PARAMETER)
            bias_kind = (
                self._constant_kind(bias_node, InputKind.PARAMETER)
                if isinstance(bias_node, Node)
                else InputKind.PARAMETER
            )

            with graph.inserting_before(weight_node):
                scaled_weight_node = create_constant_placeholder(
                    exp_program=self.exported_program,
                    graph=graph,
                    kind=weight_kind,
                    name=f"{weight_node.name}_folded_mul",
                    data=scaled_weight,
                )
                scaled_bias_node = None
                if scaled_bias is not None:
                    assert isinstance(bias_node, Node)
                    with graph.inserting_before(bias_node):
                        scaled_bias_node = create_constant_placeholder(
                            exp_program=self.exported_program,
                            graph=graph,
                            kind=bias_kind,
                            name=f"{bias_node.name}_folded_mul",
                            data=scaled_bias,
                        )

            folded_placeholders = {weight_node}
            if isinstance(bias_node, Node):
                folded_placeholders.add(bias_node)

        conv_node.args = (
            conv_node.args[0],
            scaled_weight_node,
            scaled_bias_node,
            *conv_node.args[3:],
        )
        mul_node.replace_all_uses_with(conv_node)
        graph.erase_node(mul_node)
        return folded_placeholders

    def _get_constant_tensor(self, node: Node) -> Optional[torch.Tensor]:
        if self.exported_program is not None:
            return get_param_tensor(self.exported_program, node)
        if node.op != "get_attr":
            return None
        owning_module = node.graph.owning_module
        if owning_module is None:
            return None
        target = str(node.target)
        tensor: torch.Tensor | None
        try:
            tensor = owning_module.get_parameter(target)
        except AttributeError:
            try:
                tensor = owning_module.get_buffer(target)
            except AttributeError:
                tensor = getattr(owning_module, target, None)
        return tensor if isinstance(tensor, torch.Tensor) else None

    def _can_mutate_get_attr(self, node: Node, conv_node: Node) -> bool:
        return node.op == "get_attr" and set(node.users) == {conv_node}

    def _set_get_attr_tensor(self, node: Node, tensor: torch.Tensor) -> bool:
        owning_module = node.graph.owning_module
        if owning_module is None:
            return False
        target = str(node.target)
        attr_tensor: torch.Tensor | None
        try:
            attr_tensor = owning_module.get_parameter(target)
        except AttributeError:
            try:
                attr_tensor = owning_module.get_buffer(target)
            except AttributeError:
                attr_tensor = getattr(owning_module, target, None)
        if not isinstance(attr_tensor, torch.Tensor):
            return False
        with torch.no_grad():
            attr_tensor.copy_(
                tensor.to(device=attr_tensor.device, dtype=attr_tensor.dtype)
            )
        return True

    def _constant_kind(self, node: Node, default: InputKind) -> InputKind:
        if self.exported_program is None:
            return default
        try:
            return get_constant_placeholder_kind(self.exported_program, node)
        except RuntimeError:
            return default
