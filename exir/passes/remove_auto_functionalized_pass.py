# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import List

import torch
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    get_mutable_arg_names,
)
from torch.export import ExportedProgram


def unsafe_remove_auto_functionalized_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    This pass removes an instances of the higher order op 'auto_functionalized' and modifies the graph to have the original mutator op. This pass does zero safety checks to make sure that this inplace mutation is safe.
    """
    auto_functionalize_nodes: List[torch.fx.Node] = []
    for node in ep.graph.nodes:
        if node.op == "call_function" and node.target is auto_functionalized:
            auto_functionalize_nodes.append(node)

    # Update every use of the HOP
    for node in reversed(auto_functionalize_nodes):
        func = node.args[0]
        original_kwargs = node.kwargs
        assert isinstance(func, torch._ops.OpOverload)

        with ep.graph.inserting_before(node):
            # This makes the call_function refer to every arg as a kwarg, this is weird but probably fine?
            new_node = ep.graph.call_function(func, kwargs=node.kwargs)
        for k, v in node.meta.items():
            new_node.meta[k] = v

        # Replace auto_functionalize(func, args) with just func(args)
        node.replace_all_uses_with(new_node)

        mutable_args_names = get_mutable_arg_names(new_node.target)
        output_specs = ep.graph_signature.output_specs

        # update the users of the auto_func node (the getitem nodes)
        for user in list(new_node.users.keys()):
            assert user.target == operator.getitem
            # getitem corresponding to a mutated input, just replace all uses with the original input
            if user.args[1] >= len(func._schema.returns):
                assert user.args[1] <= len(mutable_args_names)

                # If the result of getitem was used in an output node, update the output spec with the correct name
                adusted_index = user.args[1] - len(func._schema.returns)
                original_arg = original_kwargs[mutable_args_names[adusted_index]]
                for spec in output_specs:
                    if spec.arg.name == user.name:
                        spec.arg.name = original_arg.name  # pyre-ignore
                        break

                # This is a little fragile/implementation dependent, but the order of the mutable args is the same as the order
                # of the getitem calls following the HOP.
                user.replace_all_uses_with(
                    original_kwargs[mutable_args_names[adusted_index]]
                )

        if len(func._schema.returns) == 1:
            # If the function has 1 return then it will just directly return the
            # result -- we don't need a getitem. So we can replace all the
            # getitem(auto_functionalized, 0) with just the note itself.
            for user in list(new_node.users.keys()):
                if user.args[1] == 0:
                    user.replace_all_uses_with(new_node)

                    # Same case as above, update the output spec if getitem result used in an output node
                    for spec in output_specs:
                        if spec.arg.name == user.name:
                            spec.arg.name = new_node.name
                            break

        new_node.meta["val"] = node.meta["val"][0]
        ep.graph.erase_node(node)

    ep.graph.eliminate_dead_code()
    return ep
