# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import operator
from typing import Dict, Optional

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


def _get_concrete_to_ub(
    graph_module: torch.fx.GraphModule,
) -> Dict[int, int]:
    """Build concrete dim value -> upper bound mapping from placeholder info.

    Uses ``_placeholder_dynamic_dims`` from ``graph_module.meta`` (set during
    edge transform from the original ExportedProgram range_constraints) to
    build a mapping from trace-time concrete dimension values to their
    original upper bounds.
    """
    ph_dynamic: Dict[str, list] = graph_module.meta.get(
        "_placeholder_dynamic_dims", {}
    )
    concrete_to_ub: Dict[int, int] = {}
    for node in graph_module.graph.nodes:
        if node.op != "placeholder" or node.name not in ph_dynamic:
            continue
        val = node.meta.get("val", None)
        if val is None or not isinstance(val, torch.Tensor):
            continue
        ub_shape = ph_dynamic[node.name]
        for concrete, ub in zip(val.shape, ub_shape):
            c = int(concrete)
            if isinstance(ub, int) and ub != c:
                concrete_to_ub[c] = ub
    return concrete_to_ub


def _restore_dynamic_shape_info(
    graph_module: torch.fx.GraphModule,
    concrete_to_ub: Dict[int, int],
) -> None:
    """Restore dynamic shape info on TensorSpec objects that lost SymInt dims.

    Uses the concrete_to_ub mapping to find TensorSpecs whose shapes contain
    a dynamic dimension and marks them as DYNAMIC_BOUND with the correct
    upper-bound shape stored in ``_upper_bound_shape``.
    """
    if not concrete_to_ub:
        return
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            spec = node.meta.get("spec", None)
            if spec is None:
                continue
            flat_specs = pytree.tree_flatten(spec)[0]
            for s in flat_specs:
                if not isinstance(s, TensorSpec):
                    continue
                if getattr(s, "_upper_bound_shape", None) is not None:
                    continue
                ub_shape = []
                has_dynamic = False
                for d in s.shape:
                    d_int = int(d)
                    if d_int in concrete_to_ub:
                        ub_shape.append(concrete_to_ub[d_int])
                        has_dynamic = True
                    else:
                        ub_shape.append(d_int)
                if has_dynamic:
                    s.shape_dynamism = TensorShapeDynamism.DYNAMIC_BOUND
                    s._upper_bound_shape = ub_shape


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
        # Save dynamic dim mapping before re-tracing, since ExportPass can
        # lose SymInt shapes for intermediate tensors (e.g. with while_loop).
        concrete_to_ub = _get_concrete_to_ub(graph_module)

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

        # Restore dynamic shape info lost during ExportPass re-tracing.
        _restore_dynamic_shape_info(gm, concrete_to_ub)

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
