# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Public API for the ExecuTorch numerical debugger.

Backend-agnostic intermediate-value tap that complements the existing
Inspector framework:

- AOT side : `IntermediateOutputCapturer` (existing)
- Runtime side : ETDump intermediate output events (existing, opaque inside delegates)
- Runtime side : USER_OUTPUT taps (this module — works through delegates without
                 any backend-side changes)

Typical usage:

    from executorch.devtools.intermediate_output_tap import (
        tap_intermediate_outputs, strip_taps_, DEFAULT_STATS,
    )

    ep = export(model, example_inputs)
    ep_tapped, specs = tap_intermediate_outputs(ep, reducer=DEFAULT_STATS)
    edge = to_edge_transform_and_lower(ep_tapped, partitioner=[XnnpackPartitioner()])
    strip_taps_(edge)
    et_program = edge.to_executorch()

    flat_outputs = runtime.forward(*example_inputs)
    df = inspector.calculate_numeric_gap_from_taps(flat_outputs, specs)
"""

from executorch.devtools.intermediate_output_tap import (
    custom_ops_lib,  # noqa: F401  ensures torch.ops.executorch_devtools.tap is registered
)
from executorch.devtools.intermediate_output_tap._convenience import (
    format_tap_dataframe,
    specs_to_dataframe,
    tap_all_and_run,
)
from executorch.devtools.intermediate_output_tap._reducers import (
    ABS_MAX_ONLY,
    DEFAULT_STATS,
    FULL_TENSOR,
    get_reducer,
    MIN_MAX_MEAN,
    StatReducer,
)
from executorch.devtools.intermediate_output_tap._selectors import (
    NodeSelector,
    select_all,
    select_all_call_function,
    select_any,
    select_by_meta_tag,
    select_by_module_path,
    select_by_op_type,
    select_not,
)
from executorch.devtools.intermediate_output_tap._spec import TapSpec
from executorch.devtools.intermediate_output_tap._strip_pass import strip_taps_
from executorch.devtools.intermediate_output_tap._tap_pass import (
    find_tap_nodes,
    is_tap_node,
    tap_intermediate_outputs,
)


__all__ = [
    # Core API
    "tap_intermediate_outputs",
    "strip_taps_",
    "TapSpec",
    # Convenience
    "tap_all_and_run",
    "specs_to_dataframe",
    "format_tap_dataframe",
    # Reducers
    "StatReducer",
    "FULL_TENSOR",
    "ABS_MAX_ONLY",
    "MIN_MAX_MEAN",
    "DEFAULT_STATS",
    "get_reducer",
    # Selectors
    "NodeSelector",
    "select_all_call_function",
    "select_by_op_type",
    "select_by_module_path",
    "select_by_meta_tag",
    "select_any",
    "select_all",
    "select_not",
    # Helpers
    "find_tap_nodes",
    "is_tap_node",
]
