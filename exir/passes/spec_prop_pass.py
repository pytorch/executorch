# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
from typing import Optional

import torch
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.pass_base import ExportPass, ProxyValue
from executorch.exir.schema import TensorShapeDynamism
from executorch.exir.tensor import TensorSpec
from torch.export.exported_program import ExportGraphSignature
from torch.fx.node import Node
from torch.fx.passes.infra.pass_base import PassResult
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

    def __call__(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Re-trace metadata to ensure it's up to date.
        res = ExportPass()(graph_module)
        assert res is not None
        gm = res.graph_module

        def get_spec(x):
            if hasattr(x, "meta"):
                return x.meta.get("spec", None)
            else:
                return None

        for module in gm.modules():
            if isinstance(module, torch.fx.GraphModule):
                for node in module.graph.nodes:
                    meta_val = node.meta.get("val", None)

                    if node.op == "output":
                        node.meta["spec"] = pytree.tree_map(get_spec, node.args[0])
                    elif node.op == "call_function" and node.target == operator.getitem:
                        value_spec = pytree.tree_map(get_spec, node.args[0])
                        node.meta["spec"] = value_spec[node.args[1]]
                    elif (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        # Note: We currently rely on delegate node specs not being regenerated,
                        # as the spec is set somewhat manually when adding the call delegate node.
                        # If we regenerate, it can change and break lowering (it becomes a tuple?).
                        # Ideally, we should figure out how to make the spec regeneration not break
                        # things.
                        #
                        # We do need to regenerate non-call-delegate node specs, as this pass is called
                        # multiple times in some lowering paths (backends can and do call it).
                        if "spec" not in node.meta:
                            node.meta["spec"] = pytree.tree_map(make_spec, meta_val)
                    else:
                        node.meta["spec"] = pytree.tree_map(make_spec, meta_val)
        return res

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        return self(graph_module)

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

        # For dynamic shapes, initialize with size 0 in the mapped dimension.
        # The et_copy_index op will resize as it writes to each index.
        # Check if the mapped dimension is symbolic (dynamic).
        is_dynamic = isinstance(mapped_dim_size, torch.SymInt)
        init_size = 0 if is_dynamic else mapped_dim_size

        map_fake_tensor = pytree.tree_map_only(
            torch.Tensor,
            lambda x: x.new_empty(init_size, *x.shape),
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
        # Get the scan length - this may be symbolic for dynamic shapes
        xs_tensor = [arg.data for arg in xs][0]
        scan_length = xs_tensor.size(0)

        *_, body_out_node = combine_fn.graph.nodes
        body_out_fake = body_out_node.meta["val"]

        num_carry = len(init)
        flat_body_out, out_spec = pytree.tree_flatten(body_out_fake)

        carry_out = flat_body_out[:num_carry]
        y_out = flat_body_out[num_carry:]

        # Check if the scan dimension is symbolic (dynamic)
        is_dynamic = isinstance(scan_length, torch.SymInt)

        # For the y outputs, we need to use the upper bound size to allocate memory,
        # but also mark the tensor spec as DYNAMIC_BOUND so it can be resized at runtime.
        if is_dynamic:
            # Get the upper bound by evaluating the symbolic int
            # Using hint gives us the concrete upper bound value
            upper_bound_size = scan_length.node.shape_env.size_hint(
                scan_length.node.expr
            )
        else:
            upper_bound_size = scan_length

        carry_fake = carry_out
        y_fake = [
            (
                x.new_empty(upper_bound_size, *x.shape)
                if isinstance(x, torch.Tensor)
                else x
            )
            for x in y_out
        ]

        combined_fake = carry_fake + y_fake

        # Create specs from the fake tensors
        specs = pytree.tree_map(make_spec, combined_fake)

        # For dynamic shapes, mark the y_output specs as DYNAMIC_BOUND
        # so that et_copy_index can resize them at runtime
        if is_dynamic and isinstance(specs, list):
            for i in range(num_carry, len(specs)):
                if isinstance(specs[i], TensorSpec):
                    specs[i].shape_dynamism = TensorShapeDynamism.DYNAMIC_BOUND

        meta["spec"] = specs
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
