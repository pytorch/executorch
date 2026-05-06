# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
One-line convenience wrapper for the most common smoke-test workflow:

    df = tap_all_and_run(model, example_inputs, partitioner=[XnnpackPartitioner()])

Exports `model`, taps every call_function, lowers with the user's partitioner,
runs through the ExecuTorch runtime, and returns a pandas DataFrame of one row
per tap (one column per stat field). No Inspector setup, no ETRecord. For
AOT-vs-runtime numerical comparison, use Inspector.calculate_numeric_gap_from_taps,
then `format_tap_dataframe(df, tap_specs)` to get a friendly view.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
from executorch.devtools.intermediate_output_tap._reducers import StatReducer
from executorch.devtools.intermediate_output_tap._selectors import (
    NodeSelector,
    select_all_call_function,
)
from executorch.devtools.intermediate_output_tap._spec import TapSpec
from executorch.devtools.intermediate_output_tap._strip_pass import strip_taps_
from executorch.devtools.intermediate_output_tap._tap_pass import (
    tap_intermediate_outputs,
)


def tap_all_and_run(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    partitioner: list | None = None,
    reducer: str | StatReducer = "DEFAULT_STATS",
    selector: NodeSelector | None = None,
    skip_if_no_debug_handle: bool = True,
) -> pd.DataFrame:
    """
    Export -> tap -> lower -> strip -> to_executorch -> run -> DataFrame.

    Returns a DataFrame indexed by tap with columns:
        node_name, op_target, debug_handle, output_index, reducer_name, plus
        one column per reducer field (or `value` for FULL_TENSOR).
    """
    from executorch.exir import to_edge_transform_and_lower

    selector = selector or select_all_call_function()
    ep = torch.export.export(model, example_inputs, strict=True)
    ep_tapped, specs = tap_intermediate_outputs(
        ep,
        selector=selector,
        reducer=reducer,
        skip_if_no_debug_handle=skip_if_no_debug_handle,
    )
    edge = to_edge_transform_and_lower(
        ep_tapped, partitioner=partitioner or []
    )
    strip_taps_(edge)
    et_program = edge.to_executorch()

    flat_outputs = _run_pte(et_program, example_inputs)
    return specs_to_dataframe(specs, flat_outputs)


def _run_pte(et_program, example_inputs: tuple[Any, ...]) -> Sequence[Any]:
    from executorch.runtime import Runtime, Verification

    with tempfile.TemporaryDirectory() as temp_dir:
        pte_path = os.path.join(temp_dir, "model.pte")
        et_program.save(pte_path)
        rt = Runtime.get()
        program = rt.load_program(pte_path, verification=Verification.Minimal)
        method = program.load_method("forward")
        return method.execute(example_inputs)


def specs_to_dataframe(
    specs: Sequence[TapSpec],
    flat_outputs: Sequence[Any],
) -> pd.DataFrame:
    """Build a per-tap DataFrame from the tap_specs + flat output tuple."""
    rows = []
    for spec in specs:
        runtime_value = flat_outputs[spec.output_index]
        row: dict[str, Any] = {
            "node_name": spec.node_name,
            "op_target": spec.op_target,
            "debug_handle": spec.debug_handle,
            "output_index": spec.output_index,
            "reducer_name": spec.reducer_name,
        }
        if spec.fields:
            tensor_vals = (
                runtime_value.detach().cpu().tolist()
                if isinstance(runtime_value, torch.Tensor)
                else list(runtime_value)
            )
            for i, field in enumerate(spec.fields):
                row[field] = tensor_vals[i] if i < len(tensor_vals) else None
        else:
            row["value"] = runtime_value
        rows.append(row)
    return pd.DataFrame(rows)


def format_tap_dataframe(
    df: pd.DataFrame,
    tap_specs: Sequence[TapSpec],
) -> pd.DataFrame:
    """
    Reshape the raw DataFrame returned by
    `Inspector.calculate_numeric_gap_from_taps` into a friendlier per-tap,
    per-field view.

    The raw DataFrame uses the existing Inspector comparator format, which
    packs the reducer's stat tensor into a list of 0-d tensors and labels
    rows by the post-strip reducer node name (e.g. `aten_stack_default`).
    This helper:
      - matches each raw row to a TapSpec (by reducer_node_name)
      - renames `aot_ops`/`runtime_ops` columns to a single `node_name` (the
        original source node name, e.g. `linear`, `linear_1`)
      - expands the reducer stat tensor into one column per field
        (e.g. `aot_min`, `rt_min`, `aot_max`, `rt_max`, ...)
      - flattens the gap to a single float
      - drops the verbose `aot_intermediate_output` / `runtime_intermediate_output`
        list columns

    Returns a DataFrame with columns:
        node_name, op_target, reducer_name,
        gap,
        aot_<field1>, rt_<field1>, aot_<field2>, rt_<field2>, ...
    """
    # Map reducer_node_name -> spec for quick lookup.
    name_to_spec: dict[str, TapSpec] = {
        s.reducer_node_name: s
        for s in tap_specs
        if s.reducer_node_name is not None
    }

    rows = []
    for _, row in df.iterrows():
        aot_ops = row.get("aot_ops", [])
        spec = None
        for op in aot_ops or []:
            if op in name_to_spec:
                spec = name_to_spec[op]
                break
        if spec is None:
            # Couldn't match — keep a thin row with whatever we have.
            rows.append(
                {
                    "node_name": ",".join(aot_ops or []),
                    "op_target": "?",
                    "reducer_name": "?",
                    "gap": _flatten_gap(row.get("gap")),
                }
            )
            continue

        new_row: dict[str, Any] = {
            "node_name": spec.node_name,
            "op_target": spec.op_target,
            "reducer_name": spec.reducer_name,
            "gap": _flatten_gap(row.get("gap")),
        }
        aot_vals = _to_float_list(row.get("aot_intermediate_output"))
        rt_vals = _to_float_list(row.get("runtime_intermediate_output"))
        for i, field in enumerate(spec.fields):
            new_row[f"aot_{field}"] = aot_vals[i] if i < len(aot_vals) else None
            new_row[f"rt_{field}"] = rt_vals[i] if i < len(rt_vals) else None
        if not spec.fields:  # FULL_TENSOR
            new_row["aot_value"] = row.get("aot_intermediate_output")
            new_row["rt_value"] = row.get("runtime_intermediate_output")
        rows.append(new_row)
    return pd.DataFrame(rows)


def _flatten_gap(g: Any) -> float | None:
    if g is None:
        return None
    if isinstance(g, list):
        if not g:
            return None
        g = g[0]
    if isinstance(g, torch.Tensor):
        return float(g)
    try:
        return float(g)
    except (TypeError, ValueError):
        return None


def _to_float_list(v: Any) -> list[float]:
    if v is None:
        return []
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, list):
        out: list[float] = []
        for x in v:
            if isinstance(x, torch.Tensor):
                out.append(float(x))
            else:
                try:
                    out.append(float(x))
                except (TypeError, ValueError):
                    out.append(float("nan"))
        return out
    return []
