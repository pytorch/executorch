# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Public API for the ExecuTorch numerical debugger.

Backend-agnostic intermediate-value tap:

- Runtime side : USER_OUTPUT taps (this module — works through delegates without
                 any backend-side changes)

Typical usage:

    from executorch.devtools.intermediate_output_tap import (
        compare_aot_runtime_dataframe,
        tap_intermediate_outputs, strip_taps_, STATS,
    )

    ep = export(model, example_inputs)
    ep_tapped, specs = tap_intermediate_outputs(ep, reducer=STATS)
    aot_flat, _ = pytree.tree_flatten(ep_tapped.module()(*example_inputs))
    edge = to_edge_transform_and_lower(ep_tapped, partitioner=[XnnpackPartitioner()])
    strip_taps_(edge)
    et_program = edge.to_executorch()

    rt_flat = runtime.forward(*example_inputs)
    df = compare_aot_runtime_dataframe(specs, aot_flat, rt_flat)
"""

# Importing this module registers torch.ops.executorch_devtools.tap.Tensor.
from executorch.devtools.intermediate_output_tap import custom_ops_lib  # noqa: F401
from executorch.devtools.intermediate_output_tap._convenience import (
    compare_aot_runtime_dataframe,
    tap_compare,
)
from executorch.devtools.intermediate_output_tap._reducers import (
    FULL_TENSOR,
    get_reducer,
    StatReducer,
    STATS,
)
from executorch.devtools.intermediate_output_tap._selectors import (
    NodeSelector,
    select_all,
    select_all_call_function,
    select_any,
    select_by_module_class,
    select_by_module_path,
    select_by_op_type,
    select_not,
)
from executorch.devtools.intermediate_output_tap._spec import TapSpec
from executorch.devtools.intermediate_output_tap._strip_pass import strip_taps_
from executorch.devtools.intermediate_output_tap._tap_pass import (
    find_tap_nodes,
    is_tap_node,
    tap_intermediate_outputs_,
    TapRule,
)


__all__ = [
    # Core API
    "tap_intermediate_outputs_",
    "strip_taps_",
    "TapSpec",
    "TapRule",
    # Convenience
    "tap_compare",
    "compare_aot_runtime_dataframe",
    # Reducers
    "StatReducer",
    "FULL_TENSOR",
    "STATS",
    "get_reducer",
    # Selectors
    "NodeSelector",
    "select_all_call_function",
    "select_by_op_type",
    "select_by_module_path",
    "select_by_module_class",
    "select_any",
    "select_all",
    "select_not",
    # Helpers
    "find_tap_nodes",
    "is_tap_node",
]
