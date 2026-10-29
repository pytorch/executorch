# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Export the ESP-NN backend's test models as BundleIO programs (.bpte) for the
ESP32 executor runner (examples/espressif/project with -DET_ESPNN_BACKEND=ON).

Each program bundles one input set and the expected outputs from the host
reference (ProgramEmulator), which reproduces the delegate bit-exactly.
Fully delegated programs must therefore match to float precision; programs
with CPU regions get a tolerance for the portable kernels' float rounding.

Writes, into --output-dir:
  <name>.bpte      one program per case
  manifest.json    per-program atol/rtol, delegate and CPU operator counts
  operators.txt    the CPU operators the programs need, for
                   EXECUTORCH_SELECT_OPS_LIST

    python examples/espressif/export_espnn_test_programs.py --output-dir espnn_tests
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# The test models are loaded from the checkout: a non-editable install ships
# only the test modules that other shipped code imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backends.espressif.test import (  # noqa: E402
    test_cpu_fallback as fallback_cases,
    test_int8_lowering as lowering_cases,
    test_int8_ops as op_cases,
)

from executorch.backends.espressif import to_espnn_edge  # noqa: E402
from executorch.backends.espressif.reference import ProgramEmulator  # noqa: E402
from executorch.devtools.bundled_program.config import (  # noqa: E402
    MethodTestCase,
    MethodTestSuite,
)
from executorch.devtools.bundled_program.core import BundledProgram  # noqa: E402
from executorch.devtools.bundled_program.serialize import (  # noqa: E402
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.exir.lowered_backend_module import LoweredBackendModule  # noqa: E402

EXACT_TOLERANCE = 1e-5
FALLBACK_TOLERANCE = 0.01


def cases():
    yield "conv-act-pool-linear", lowering_cases.ConvActModel(), [(1, 3, 16, 16)]
    yield "concat-mean-linear", lowering_cases.ConcatModel(), [(1, 4, 6, 6)]
    yield "se-lut-mul-softmax", lowering_cases.SEBlockModel(), [(1, 3, 16, 16)]
    for name, model, shapes, _ in op_cases.CASES:
        yield name, model, shapes
    for name, model, shapes, _ in fallback_cases.CASES:
        yield name, model, shapes


def as_list(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def export_case(name, model, shapes, seed):
    _, program = op_cases.quantize(model, shapes)
    edge, report = to_espnn_edge(program)
    exported = edge.exported_program()
    generator = torch.Generator().manual_seed(seed)
    inputs = [torch.randn(shape, generator=generator) for shape in shapes]
    expected = [
        t.to(torch.float32) for t in as_list(ProgramEmulator(exported)(*inputs))
    ]

    executorch_program = edge.to_executorch()
    plan = executorch_program.executorch_program.execution_plan[0]
    operators = sorted({f"{op.name}.{op.overload}" for op in plan.operators})
    bundle = BundledProgram(
        executorch_program,
        [MethodTestSuite("forward", [MethodTestCase(inputs, expected)])],
    )

    exact = not report["cpu_ops"]
    if exact:
        atol = rtol = EXACT_TOLERANCE
    else:
        peak = max(float(t.abs().max()) for t in expected)
        atol, rtol = FALLBACK_TOLERANCE * max(peak, 1.0), FALLBACK_TOLERANCE
    delegates = sum(
        isinstance(module, LoweredBackendModule)
        for _, module in exported.graph_module.named_children()
    )
    entry = {
        "name": name,
        "file": f"{name}.bpte",
        "atol": float(f"{atol:.6g}"),
        "rtol": rtol,
        "exact": exact,
        "delegates": delegates,
        "delegated_ops": report["delegated_ops"],
        "cpu_ops": report["cpu_ops"],
        "operators": operators,
    }
    return serialize_from_bundled_program_to_flatbuffer(bundle), entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--only", help="comma separated case names (default: every case)"
    )
    args = parser.parse_args()
    selected = set(args.only.split(",")) if args.only else None
    torch.set_grad_enabled(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest, operators = [], set()
    for name, model, shapes in cases():
        if selected is not None and name not in selected:
            continue
        if any(entry["name"] == name for entry in manifest):
            raise SystemExit(f"duplicate case name: {name}")
        blob, entry = export_case(name, model, shapes, args.seed)
        (args.output_dir / entry["file"]).write_bytes(blob)
        manifest.append(entry)
        operators.update(entry["operators"])
        print(
            f"{name:34s} delegates={entry['delegates']} "
            f"cpu_ops={len(entry['cpu_ops'])} atol={entry['atol']} "
            f"({len(blob)} bytes)"
        )
    if selected is not None and len(manifest) != len(selected):
        missing = selected - {entry["name"] for entry in manifest}
        raise SystemExit(f"unknown case names: {sorted(missing)}")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "operators.txt").write_text(",".join(sorted(operators)) + "\n")
    print(f"wrote {len(manifest)} programs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
