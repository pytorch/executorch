# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generate WebGPU op-test artifacts from the declarative cases.

Per case: export the module to `<id>.pte`, write its inputs + torch golden as raw
little-endian fp32, and emit `manifest.json` for the C++ gtest driver to consume.
Run: `python -m ...generate_op_tests --output <dir> [--ops add,sigmoid,rms_norm]`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

# Importing cases populates op_test_registry via the @register_op_test decorators.
from executorch.backends.webgpu.test.op_tests import cases  # noqa: F401
from executorch.backends.webgpu.test.op_tests.test_suite import (
    InputSpec,
    op_test_registry,
    WebGPUTestSuite,
)
from executorch.exir import to_edge_transform_and_lower


def _materialize(spec) -> torch.Tensor:
    """Produce a forward-input tensor from a bare shape tuple or an InputSpec."""
    if isinstance(spec, InputSpec):
        shape, gen = spec.shape, spec.gen
    else:
        shape, gen = spec, "randn"
    if callable(gen):
        return gen(shape).to(torch.float32)
    if gen == "randn":
        return torch.randn(*shape)
    if gen == "ramp":
        n = 1
        for d in shape:
            n *= d
        return torch.linspace(-1.0, 1.0, n, dtype=torch.float32).reshape(shape)
    raise ValueError(f"unknown input gen: {gen!r}")


def export_case(suite: WebGPUTestSuite, case) -> tuple[torch.nn.Module, tuple, object]:
    """Build the module + forward inputs and export to an ExecuTorch program."""
    module = suite.module_factory(**case.construct)
    # Seed so an unseeded-randn input is reproducible across generations (the golden uses
    # the SAME tensor, so this only affects which bytes a case sees, never pass/fail).
    torch.manual_seed(0)
    inputs = tuple(_materialize(s) for s in case.inputs)
    ep = torch.export.export(module, inputs)
    prog = to_edge_transform_and_lower(
        ep, partitioner=[VulkanPartitioner()]
    ).to_executorch()
    return module, inputs, prog


def _has_vulkan_delegate(prog) -> bool:
    return any(
        d.id == "VulkanBackend"
        for p in prog.executorch_program.execution_plan
        for d in p.delegates
    )


def _write_fp32(t: torch.Tensor, path: str) -> None:
    t.detach().contiguous().cpu().numpy().astype("<f4").tofile(path)


def generate_case(op: str, suite: WebGPUTestSuite, case, out_dir: str) -> dict:
    """Export one case, write its .pte + input/golden .bin, return its manifest entry."""
    module, inputs, prog = export_case(suite, case)
    if not _has_vulkan_delegate(prog):
        msg = (
            f"{op}/{case.name or 'case'} produced NO VulkanBackend delegate "
            "(silent CPU fallback) — this case would not exercise the WebGPU path."
        )
        if case.required:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}")
    golden_dtype = getattr(suite, "golden_dtype", "float64")
    out_index = 0
    with torch.no_grad():
        # fp32 eager is the dual-oracle reference; compute it BEFORE any .double().
        eager = module(*inputs)
        eager_t = eager[out_index] if isinstance(eager, (tuple, list)) else eager
        if case.golden_fn is not None:
            golden = case.golden_fn(module, inputs)
        elif golden_dtype == "float64":
            # fp64 oracle (~1e-15); deepcopy keeps the original module fp32.
            double_module = copy.deepcopy(module).double()
            golden = double_module(*[x.double() for x in inputs])
        else:
            golden = eager  # gather/copy: fp64 is bit-identical, skip it
    golden_t = golden[out_index] if isinstance(golden, (tuple, list)) else golden
    out_t = golden_t.to(torch.float32)
    atol = case.atol if case.atol is not None else suite.atol
    rtol = case.rtol if case.rtol is not None else suite.rtol
    # Dual-oracle gate: the fp64 golden must match the fp32 eager within tol — proves
    # the oracle isn't itself buggy. Skipped for float32 suites.
    if golden_dtype == "float64" and case.golden_fn is None:
        torch.testing.assert_close(
            eager_t.to(torch.float32), out_t, atol=atol, rtol=rtol
        )

    case_id = f"{op}__{case.name or 'case'}"
    pte_rel = f"{case_id}.pte"
    with open(os.path.join(out_dir, pte_rel), "wb") as f:
        f.write(prog.buffer)

    input_entries: list[dict] = []
    for i, t in enumerate(inputs):
        rel = f"{case_id}.in{i}.bin"
        _write_fp32(t, os.path.join(out_dir, rel))
        input_entries.append({"path": rel, "shape": list(t.shape), "dtype": "float32"})

    golden_rel = f"{case_id}.golden.bin"
    _write_fp32(out_t, os.path.join(out_dir, golden_rel))

    return {
        "op": op,
        "case": case.name or "case",
        "pte": pte_rel,
        "inputs": input_entries,
        "golden": {
            "path": golden_rel,
            "shape": list(out_t.shape),
            "dtype": "float32",
            "output_index": out_index,
        },
        "atol": atol,
        "rtol": rtol,
        "required": case.required,
        "heavy": case.heavy,
    }


def generate(
    out_dir: str, ops: list[str] | None = None, include_unverified: bool = False
) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    if ops:
        # Explicit selection is honored verbatim (lets you target a single
        # not-yet-verified op once its handler builds).
        selected = ops
    else:
        # Default = the green set only (verified suites). `--all` includes the
        # authored-but-unverified ones.
        selected = [
            op
            for op, suite in op_test_registry.items()
            if include_unverified or suite.verified
        ]
    heavy_enabled = bool(os.environ.get("WEBGPU_TEST_HEAVY"))
    entries: list[dict] = []
    for op in selected:
        suite = op_test_registry[op]
        for case in suite.cases:
            if case.heavy and not heavy_enabled:
                # Export-gated: omitted from the default manifest (set WEBGPU_TEST_HEAVY).
                continue
            entries.append(generate_case(op, suite, case, out_dir))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(entries, f, indent=2)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WebGPU op-test artifacts.")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--ops",
        default=None,
        help="comma-separated op names (default: verified suites only)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include authored-but-unverified suites (handler-pending ops)",
    )
    args = parser.parse_args()
    ops = args.ops.split(",") if args.ops else None
    entries = generate(args.output, ops, include_unverified=args.all)
    print(
        f"Generated {len(entries)} cases -> {os.path.join(args.output, 'manifest.json')}"
    )


if __name__ == "__main__":
    main()
