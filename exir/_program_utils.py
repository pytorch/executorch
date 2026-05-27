# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from torch.export.exported_program import (
    ConstantArgument,
    ExportGraphSignature,
    InputSpec,
    OutputSpec,
)


def _get_updated_range_constraints(gm):
    def get_shape_env(gm):
        vals = [
            node.meta["val"]
            for node in gm.graph.nodes
            if node.meta.get("val", None) is not None
        ]
        from torch._guards import detect_fake_mode  # type: ignore[21]

        fake_mode = detect_fake_mode(vals)
        if fake_mode is not None:
            return fake_mode.shape_env
        for v in vals:
            if isinstance(v, torch.SymInt):
                return v.node.shape_env

    shape_env = get_shape_env(gm)
    if shape_env is None:
        return {}
    range_constraints = {
        shape_env.replacements.get(k, k): v for k, v in shape_env.var_to_range.items()
    }
    # Only when we have an unbacked symint, and it's used as constructor inputs,
    # runtime_var_to_range will make a difference compated to var_to_range.
    # e.g. [2, oo) -> [0, oo)
    for k, v in shape_env.var_to_range.items():
        if k not in shape_env.replacements:
            range_constraints[k] = v
    return range_constraints


def _get_updated_graph_signature(
    old_signature: ExportGraphSignature,
    new_gm: torch.fx.GraphModule,
) -> ExportGraphSignature:
    """
    Update the graph signature's user_input/user_outputs.
    """
    new_input_specs = []
    i = 0
    for node in new_gm.graph.nodes:
        if node.op != "placeholder":
            continue

        assert i < len(
            old_signature.input_specs
        ), "Number of inputs changed after transformation"
        old_input_spec = old_signature.input_specs[i]
        arg = (
            old_input_spec.arg
            if isinstance(old_input_spec.arg, ConstantArgument)
            # pyre-fixme[20]: Argument `class_fqn` expected.
            else type(old_input_spec.arg)(node.name)
        )
        new_input_specs.append(
            InputSpec(
                old_input_spec.kind,
                arg,
                old_input_spec.target,
                persistent=old_input_spec.persistent,
            )
        )
        i += 1

    output_node = new_gm.graph.output_node()
    assert output_node.op == "output"

    new_output_specs = []
    for i, node in enumerate(output_node.args[0]):
        assert i < len(
            old_signature.output_specs
        ), "Number of outputs changed after transformation"
        old_output_spec = old_signature.output_specs[i]
        arg = (
            old_output_spec.arg
            if isinstance(old_output_spec.arg, ConstantArgument)
            # pyre-fixme[20]: Argument `class_fqn` expected.
            else type(old_output_spec.arg)(node.name)
        )
        new_output_specs.append(
            OutputSpec(old_output_spec.kind, arg, old_output_spec.target)
        )

    new_signature = ExportGraphSignature(
        input_specs=new_input_specs, output_specs=new_output_specs
    )
    return new_signature
