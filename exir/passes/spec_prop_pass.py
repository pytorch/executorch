# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional

import torch
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature
from torch.fx.node import Node
from torch.utils import _pytree as pytree


# pyre-ignore
def make_spec(x):
    if isinstance(x, ProxyValue):
        return make_spec(x.node.meta["val"])
    elif isinstance(x, torch.Tensor):
        return TensorSpec.from_tensor(x)
    elif isinstance(x, (int, bool, float)):
        return x
    else:
        return None


def _is_mutable_buffer(
    node: Node, graph_signature: Optional[ExportGraphSignature] = None
) -> bool:
    """
    Check if the node is mutable buffer according to the provided graph signature.
    """
    # graph signature is None for memory planning passes not called from EdgeProgramManager, these paths are deprecated so mutable buffers are not supported on them.
    if graph_signature is None:
        return False
    if node.op == "placeholder":
        if isinstance(node.target, str):
            if node.target in graph_signature.inputs_to_buffers:
                fqn = graph_signature.inputs_to_buffers[node.target]
                # if the buffer is mutated then record that
                if fqn in graph_signature.buffers_to_mutate.values():
                    return True
    return False


class SpecPropPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()

    def on_attr(self, attr: ProxyValue) -> None:
        attr.node.meta["spec"] = pytree.tree_map_only(
            torch.Tensor,
            make_spec,
            attr.data,
        )

    def update_placeholder_tensor_specs(
        self,
        exported_program: torch.export.ExportedProgram,
        graph_module: torch.fx.GraphModule,
    ) -> None:
        """
        Update the tensor specs for all placeholder nodes such that
        placeholders that are parameters are marked as constant.
        """
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            if "spec" not in node.meta:
                raise RuntimeError(f"Placeholder node {node} missing meta['spec']")
            spec = node.meta["spec"]
            if isinstance(node.target, str) and (
                node.target in exported_program.graph_signature.inputs_to_parameters
                or (
                    node.target in exported_program.graph_signature.inputs_to_buffers
                    and not _is_mutable_buffer(node, exported_program.graph_signature)
                )
                or node.target
                in exported_program.graph_signature.inputs_to_lifted_tensor_constants
            ):
                spec.const = True

    # pyre-ignore
    def placeholder(self, name: str, arg, meta):
        meta["spec"] = make_spec(arg)
        return super().placeholder(name, arg, meta)

    # pyre-ignore
    def call_operator(self, op, args, kwargs, meta):
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        meta["spec"] = pytree.tree_map(make_spec, op(*args_data, **kwargs_data))
        return super().call_operator(op, args, kwargs, meta)

    # pyre-ignore
    def call_getitem(self, value, key: int, meta):
        meta["spec"] = value.node.meta["spec"][key]
        return super().call_getitem(value, key, meta)

    # pyre-ignore
    def call_cond(self, pred, true_fn, false_fn, inputs, meta):
        # true_fn/false_fn return tensors of the same shape, so we can pick
        # either one here.
        *_, true_out_node = true_fn.graph.nodes
        meta["spec"] = pytree.tree_map(make_spec, true_out_node.meta["val"])
        return super().call_cond(pred, true_fn, false_fn, inputs, meta)

    def call_while(
        self,
        cond_fn: torch.fx.GraphModule,
        body_fn: torch.fx.GraphModule,
        carried_inputs: List[ProxyValue],
        additional_inputs: List[ProxyValue],
        meta: NodeMetadata,
    ):
        meta["spec"] = pytree.tree_map(make_spec, carried_inputs)
        return super().call_while(
            cond_fn, body_fn, carried_inputs, additional_inputs, meta
        )

    def call_map(
        self,
        f: torch.fx.GraphModule,
        mapped_args: List[ProxyValue],
        operands: List[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        mapped_dim_size = [arg.data for arg in mapped_args][0].size(0)
        *_, body_out_node = f.graph.nodes
        body_out_node_fake_tensor = body_out_node.meta["val"]
        map_fake_tensor = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.new_empty(mapped_dim_size, *x.shape),
            body_out_node_fake_tensor,
        )
        meta["spec"] = pytree.tree_map(make_spec, map_fake_tensor)
        return super().call_map(f, mapped_args, operands, meta)

    def call_scan(
        self,
        combine_fn: torch.fx.GraphModule,
        init: List[ProxyValue],
        xs: List[ProxyValue],
        additional_inputs: List[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        """
        Propagate specs for scan higher-order operation.

        Scan returns (final_carry, stacked_outputs) where:
        - final_carry: Same shape as init (NOT stacked, just the final carry state)
        - stacked_outputs: Outputs stacked along dim 0 with scan_length

        The combine_fn signature is:
            combine_fn(*init, *xs_slice, *additional_inputs) -> (*next_carry, *y_slice)

        So the combine_fn outputs are split into:
        - First len(init) outputs: carry values (same shape as init)
        - Remaining outputs: y values (to be stacked)

        Memory Layout Note:
        The specs created here are for the FINAL outputs of the scan operation:
        - carry specs: Working carry buffers that persist across iterations.
          These are SEPARATE from combine_fn's output buffers. The emitter
          must copy from combine_fn's temporary carry output to these buffers
          after each iteration (in-place op.out(x, out=x) is unsafe).
        - y specs: Pre-allocated stacked buffers filled via et_copy_index.

        The combine_fn's internal temporary buffers are allocated separately
        via memory planning with alloc_graph_input=True, alloc_graph_output=True.
        """
        # Get scan length from first xs tensor
        scan_length = [arg.data for arg in xs][0].size(0)

        # Get the output node from combine_fn
        *_, body_out_node = combine_fn.graph.nodes
        body_out_fake = body_out_node.meta["val"]

        # The combine_fn outputs are: (*next_carry, *y_slice)
        # Split them based on the number of init values
        num_carry = len(init)

        # Flatten the outputs to handle them uniformly
        flat_body_out, out_spec = pytree.tree_flatten(body_out_fake)

        # Split into carry outputs and y outputs
        carry_out = flat_body_out[:num_carry]
        y_out = flat_body_out[num_carry:]

        # Create specs:
        # - Carry: same shape as combine_fn output (NOT stacked)
        #   These are working buffers that get updated each iteration
        # - Y: stacked along dim 0 with scan_length
        carry_fake = carry_out  # Carry keeps same shape

        y_fake = [
            x.new_empty(scan_length, *x.shape) if isinstance(x, torch.Tensor) else x
            for x in y_out
        ]

        # Combine carry and stacked y outputs
        combined_fake = carry_fake + y_fake

        meta["spec"] = pytree.tree_map(make_spec, combined_fake)
        return super().call_scan(combine_fn, init, xs, additional_inputs, meta)

    # pyre-ignore
    def call_delegate(self, lowered_module, args, kwargs, meta):
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        # If spec is missing, re-genenrate it with args data
        if "spec" not in meta:
            meta["spec"] = pytree.tree_map(
                make_spec,
                executorch_call_delegate(lowered_module, *args_data),
            )
        return super().call_delegate(lowered_module, args, kwargs, meta)

    # pyre-ignore
    def output(self, results, meta):
        # pyre-ignore
        def get_spec(x):
            if isinstance(x, ProxyValue):
                return x.node.meta["spec"]
            else:
                return make_spec(x)

        meta["spec"] = pytree.tree_map(get_spec, results)
        return super().output(results, meta)
