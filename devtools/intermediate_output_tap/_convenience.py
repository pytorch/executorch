# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Convenience helpers built on top of `tap_intermediate_outputs_` / `strip_taps_`:

* `tap_compare`: one-shot helper that exports a model, taps it, lowers with
  the user's partitioner, runs through the ExecuTorch runtime, and returns
  the AOT-vs-runtime comparison DataFrame plus the tap specs. The simplest
  way to use the intermediate-output tap.
* `specs_to_dataframe`: build a per-tap DataFrame from a tap_specs list and
  the runtime's flat output tuple.
* `compare_aot_runtime_dataframe`: side-by-side AOT-vs-runtime DataFrame from
  the flat outputs of the *tapped* ExportedProgram (eager) and the post-strip
  runtime program.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from typing import Any

import pandas as pd
import torch
import torch.utils._pytree as pytree
from executorch.devtools.intermediate_output_tap._spec import TapSpec
from executorch.devtools.intermediate_output_tap._strip_pass import strip_taps_
from executorch.devtools.intermediate_output_tap._tap_pass import (
    tap_intermediate_outputs_,
    TapRule,
)


def tap_compare(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    partitioner: list | None = None,
    *,
    rules: Sequence[TapRule] | TapRule | None = None,
    error_on_empty: bool = True,
) -> tuple[pd.DataFrame, list[TapSpec]]:
    """
    One-shot AOT-vs-runtime numerical-debugging helper.

    Runs the full pipeline: export -> tap -> capture AOT reference values
    -> lower with `partitioner` -> strip -> to_executorch -> runtime
    -> AOT-vs-runtime DataFrame.

    Args:
        model: Eager nn.Module to debug.
        example_inputs: Positional args to the model's forward.
        partitioner: Optional list of partitioners passed to
            `to_edge_transform_and_lower`. Defaults to `[]` (no delegation).
        rules: Same semantics as `tap_intermediate_outputs_` — a sequence of
            `(selector, reducer)` pairs (or a single tuple as sugar).
            Defaults to `[(select_all_call_function(), STATS)]`.
        error_on_empty: Same semantics as `tap_intermediate_outputs_`.

    Returns:
        A `(df, specs)` tuple where:
          - `df`: side-by-side AOT-vs-runtime DataFrame from
            `compare_aot_runtime_dataframe`.
          - `specs`: list of `TapSpec`s in tap-creation order.
    """
    from executorch.exir import to_edge_transform_and_lower

    ep = torch.export.export(model, example_inputs, strict=True)
    ep_t, specs = tap_intermediate_outputs_(
        ep,
        rules=rules,
        error_on_empty=error_on_empty,
    )

    # AOT-side reference values: tap.Tensor's eager impl applies the reducer,
    # so the flat outputs of the tapped EP already contain reduced values at
    # the same positions the runtime will use.
    aot_out = ep_t.module()(*example_inputs)
    aot_flat, _ = pytree.tree_flatten(aot_out)

    edge = to_edge_transform_and_lower(ep_t, partitioner=partitioner or [])
    strip_taps_(edge)
    et_program = edge.to_executorch()

    flat_inputs, _ = pytree.tree_flatten(example_inputs)
    rt_flat = list(_run_pte(et_program, flat_inputs))

    df = compare_aot_runtime_dataframe(specs, aot_flat, rt_flat)
    return df, specs


def _run_pte(et_program, example_inputs: tuple[Any, ...]) -> Sequence[Any]:
    from executorch.runtime import Runtime, Verification

    with tempfile.TemporaryDirectory() as temp_dir:
        pte_path = os.path.join(temp_dir, "model.pte")
        et_program.save(pte_path)
        rt = Runtime.get()
        program = rt.load_program(pte_path, verification=Verification.Minimal)
        method = program.load_method("forward")
        return method.execute(example_inputs)


def _flat_floats(v: Any) -> list[float]:
    """Flatten a tap value (tensor / list / scalar) to a flat list of floats."""
    if isinstance(v, torch.Tensor):
        return [
            float(x) for x in v.detach().to(torch.float32).cpu().reshape(-1).tolist()
        ]
    if isinstance(v, (list, tuple)):
        out: list[float] = []
        for x in v:
            out.extend(_flat_floats(x))
        return out
    try:
        return [float(v)]
    except (TypeError, ValueError):
        return []


def _sqnr_db(aot_vals: list[float], rt_vals: list[float]) -> float:
    """Signal-to-quantization-noise ratio in dB. Higher is better.

    Thin wrapper around `torch.ao.ns.fx.utils.compute_sqnr` (the canonical
    implementation already used by `backends/test/harness/error_statistics.py`).
    """
    from torch.ao.ns.fx.utils import compute_sqnr

    n = min(len(aot_vals), len(rt_vals))
    if n == 0:
        return float("nan")
    aot_t = torch.tensor(aot_vals[:n], dtype=torch.float32)
    rt_t = torch.tensor(rt_vals[:n], dtype=torch.float32)
    return float(compute_sqnr(rt_t, aot_t))


def compare_aot_runtime_dataframe(
    specs: Sequence[TapSpec],
    aot_flat: Sequence[Any],
    rt_flat: Sequence[Any],
) -> pd.DataFrame:
    """
    Build a side-by-side AOT-vs-runtime DataFrame from the flat outputs of
    the *tapped* ExportedProgram (eager) and the post-strip runtime program.

    Both `aot_flat[spec.output_index]` and `rt_flat[spec.output_index]` already
    contain the *reduced* tap value, since `tap.Tensor`'s eager impl applies
    the named reducer (see `custom_ops_lib.py`).

    Output columns per spec:
      - For non-FULL_TENSOR reducers: one `aot_<field>` and `rt_<field>` per
        reducer field (e.g. `aot_min`, `rt_min`, ...).
      - For FULL_TENSOR: `sqnr_db` (signal-to-noise of aot vs rt over the
        whole tensor, in dB)
    """
    rows: list[dict[str, Any]] = []
    for spec in specs:
        aot_vals = _flat_floats(aot_flat[spec.output_index])
        rt_vals = _flat_floats(rt_flat[spec.output_index])

        row: dict[str, Any] = {
            "node_name": spec.node_name,
            "module_path": spec.module_path,
            "module_class": spec.module_class,
            "op_target": spec.op_target,
            "reducer_name": spec.reducer_name,
            "output_index": spec.output_index,
        }

        if spec.reducer_name == "FULL_TENSOR":
            row["sqnr_db"] = _sqnr_db(aot_vals, rt_vals)
            row["aot_numel"] = len(aot_vals)
            row["rt_numel"] = len(rt_vals)
        else:
            fields = (
                list(spec.fields)
                if spec.fields
                else [f"v{i}" for i in range(max(len(aot_vals), len(rt_vals)))]
            )
            for i, f in enumerate(fields):
                row[f"aot_{f}"] = aot_vals[i] if i < len(aot_vals) else float("nan")
                row[f"rt_{f}"] = rt_vals[i] if i < len(rt_vals) else float("nan")

        rows.append(row)
    return pd.DataFrame(rows)
