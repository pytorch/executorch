# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import List, Optional

import torch
from executorch.exir.error import internal_assert, InternalError
from executorch.exir.operator.convert import (
    is_inplace_variant,
    output_to_aliased_input_map,
    unwrap_op_overload,
)
from executorch.exir.tensor import TensorSpec


def is_inplace_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    if not isinstance(target, torch._ops.OpOverload) and not isinstance(
        getattr(target, "_op", None), torch._ops.OpOverload
    ):
        return False
    op = unwrap_op_overload(target)
    return is_inplace_variant(op._schema.name, op._schema.overload_name)


def alias_inplace_result_specs(node: torch.fx.Node) -> None:  # noqa: C901
    """Alias an in-place op's result TensorSpec(s) onto the corresponding
    input's spec.

    In-place ops (schema kind == inplace) mutate one or more of their
    inputs and return tensors that alias them, declared via the
    ``Tensor(a!)`` schema annotation. To make the memory planner treat
    result and aliased input as one storage, we copy the input's spec
    object onto the output's ``node.meta["spec"]`` slot during spec
    propagation, before consumers capture it.

    Output→input correspondence is computed via
    ``output_to_aliased_input_map``, which matches each return's
    write-alias set against the inputs that share it.

    Gating:

    - Only runs for in-place nodes (caller checks ``is_inplace_node``).
    - Multi-output in-place ops are supported when each return's alias
      set matches exactly one input's alias set.
    - Falls through silently when alias info is absent or unparseable,
      preserving the original spec. ``logging.debug`` records each
      early-return reason so silent regressions are observable.
    """
    target = node.target
    op = unwrap_op_overload(target)

    schema = op._schema
    out_to_in = output_to_aliased_input_map(schema)
    if not out_to_in:
        logging.debug(
            f"alias_inplace_result_specs: schema for {op} declares no "
            f"write-aliased outputs matching an input; skipping."
        )
        return

    # Normalize the current spec container into a list for uniform
    # handling. Caller guarantees this node was identified as an
    # in-place op with a meta spec (see `is_inplace_node`), so an
    # unrecognized container shape is a real bug — assert loudly so it
    # surfaces in tests rather than silently disabling aliasing.
    current = node.meta.get("spec")
    if isinstance(current, TensorSpec):
        out_specs_list: List[Optional[TensorSpec]] = [current]
        return_container_kind = "scalar"
    elif isinstance(current, (list, tuple)):
        out_specs_list = list(current)
        return_container_kind = type(current).__name__
    else:
        raise InternalError(
            f"alias_inplace_result_specs: in-place node {node.name} "
            f"({op}) has unrecognized spec container of type "
            f"{type(current).__name__!r}; expected TensorSpec, list, "
            f"or tuple."
        )

    # Compute new spec for each return; None means "keep original".
    # Mutated inputs are usually positional (the `Tensor(a!)` `self`
    # arg), but custom ops may pass them via kwargs — fall back to
    # `node.kwargs[arg_name]` in that case.
    replacements: List[Optional[TensorSpec]] = [None] * len(out_specs_list)

    for out_idx, in_idx in out_to_in.items():
        if out_idx >= len(out_specs_list):
            logging.debug(
                f"alias_inplace_result_specs: schema for {op} declares "
                f"return {out_idx} but spec container has only "
                f"{len(out_specs_list)} entries; skipping this return."
            )
            continue
        in_node = _resolve_mutated_input(node, schema, in_idx)
        if in_node is None:
            continue
        # NOTE: alias unconditionally — including when the input is a
        # placeholder (named buffer / mutable input). Skipping
        # placeholders would leave a dangling out-arg on in-place op
        # instructions whose self is a buffer; the runtime kernel
        # writes to that out-arg's storage rather than mutating the
        # buffer, producing incorrect results. The companion change in
        # `_emit_spec` (emit/_emitter.py) deduplicates by spec identity
        # so this aliasing doesn't produce two Values for the same FQN.
        in_spec = in_node.meta.get("spec")
        if not isinstance(in_spec, TensorSpec):
            continue
        replacements[out_idx] = in_spec

    if not any(r is not None for r in replacements):
        return

    # Assemble the new spec container, preserving the original shape.
    new_list = [
        replacements[i] if replacements[i] is not None else out_specs_list[i]
        for i in range(len(out_specs_list))
    ]
    if return_container_kind == "scalar":
        if isinstance(new_list[0], TensorSpec):
            node.meta["spec"] = new_list[0]
    elif return_container_kind == "list":
        node.meta["spec"] = list(new_list)
    elif return_container_kind == "tuple":
        node.meta["spec"] = tuple(new_list)


def _resolve_mutated_input(
    node: torch.fx.Node, schema: torch.FunctionSchema, in_idx: int
) -> Optional[torch.fx.Node]:
    """Return the node passed as argument ``in_idx`` of an in-place op,
    preferring positional args and falling back to kwargs by name (custom
    ops may pass ``Tensor(a!)`` args via kwargs)."""
    in_node: object
    if in_idx < len(node.args):
        in_node = node.args[in_idx]
    else:
        arg_name = (
            schema.arguments[in_idx].name if in_idx < len(schema.arguments) else None
        )
        if arg_name is None or arg_name not in node.kwargs:
            logging.debug(
                f"_resolve_mutated_input: schema for {node.target} "
                f"expects mutated input at position {in_idx} "
                f"(name={arg_name!r}) but it is supplied neither "
                "positionally nor via kwargs; skipping."
            )
            return None
        in_node = node.kwargs[arg_name]
    return in_node if isinstance(in_node, torch.fx.Node) else None


def verify_inplace_result_aliases(graph_module: torch.fx.GraphModule) -> None:
    """Assert that every write-aliased in-place result shares its mutated
    input's TensorSpec.

    ``SpecPropPass`` establishes this aliasing before consumers capture the
    result spec. An in-place op added after spec propagation would otherwise
    get its own buffer that the kernel never writes.
    """
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if not is_inplace_node(node):
                continue
            schema = unwrap_op_overload(node.target)._schema
            result_specs = node.meta.get("spec")
            if isinstance(result_specs, TensorSpec):
                result_specs = [result_specs]
            if not isinstance(result_specs, (list, tuple)):
                continue
            for out_idx, in_idx in output_to_aliased_input_map(schema).items():
                in_node = _resolve_mutated_input(node, schema, in_idx)
                if in_node is None or out_idx >= len(result_specs):
                    continue
                in_spec = in_node.meta.get("spec")
                if not isinstance(in_spec, TensorSpec):
                    continue
                internal_assert(
                    result_specs[out_idx] is in_spec,
                    f"In-place node {node.name} ({node.target}) result "
                    f"{out_idx} does not share the TensorSpec of its mutated "
                    f"input {in_node.name}. Run SpecPropPass after introducing "
                    "in-place ops and before memory planning.",
                )
