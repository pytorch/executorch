# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Build the on-board operator test bundles from the backend's unit-test models.

  op_suite.xtb   every XTS3 opcode as a standalone int8 graph (bit-exact check)
  pte_suite.xtp  ExecuTorch programs: full delegation, multiple inputs and
                 outputs, and CPU fallback (cumsum, transformer, LSTM)

    python examples/espressif/scripts/make_test_bundles.py --output-dir examples/espressif/espnn_runner/bundles
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from espnn_bundle import sqnr_db, write_bundle, xte1_vectors, xtv2_vectors  # noqa: E402

from executorch.backends.espressif import to_espnn_edge  # noqa: E402
from executorch.backends.espressif.reference import ProgramEmulator  # noqa: E402
from executorch.backends.espressif.serialization import (  # noqa: E402
    build_graph,
    parse_graph,
    serialize_graph,
)
from executorch.backends.espressif.test import (  # noqa: E402  # noqa: E402  # noqa: E402
    test_cpu_fallback as fallback_cases,
    test_int8_lowering as lowering_cases,
    test_int8_ops as op_cases,
)
from executorch.exir.lowered_backend_module import LoweredBackendModule  # noqa: E402


def standalone_cases():
    yield "conv-act-pool-linear", lowering_cases.ConvActModel(), [(1, 3, 16, 16)]
    yield "concat-mean-linear", lowering_cases.ConcatModel(), [(1, 4, 6, 6)]
    yield "se-lut-mul-softmax", lowering_cases.SEBlockModel(), [(1, 3, 16, 16)]
    for name, model, shapes, _ in op_cases.CASES:
        yield name, model, shapes


def program_cases():
    yield "se-block-full-delegate", lowering_cases.SEBlockModel(), [
        (1, 3, 16, 16)
    ], 99.0
    yield "detection-head-2-outputs", op_cases.DetectionHead(), [(1, 3, 16, 16)], 99.0
    yield "two-inputs-two-outputs", op_cases.TwoInputTwoOutput(), [
        (1, 3, 12, 12),
        (1, 6),
    ], 99.0
    for name, model, shapes, _ in fallback_cases.CASES:
        yield name, model, shapes, 30.0


def random_inputs(shapes, count, seed):
    generator = torch.Generator().manual_seed(seed)
    return [[torch.randn(s, generator=generator) for s in shapes] for _ in range(count)]


def build_op_suite(samples: int):
    entries, reports = [], []
    for name, model, shapes in standalone_cases():
        _, program = op_cases.quantize(model, shapes)
        blob = serialize_graph(build_graph(program))
        graph = parse_graph(blob)
        vectors = xtv2_vectors(
            graph, [[x.numpy() for x in s] for s in random_inputs(shapes, samples, 7)]
        )
        operations = dict(Counter(op.opcode.name for op in graph.operations))
        entries.append((name, blob, vectors))
        reports.append(
            {
                "name": name,
                "operations": operations,
                "blob_bytes": len(blob),
                "inputs": [list(graph.tensors[i].shape) for i in graph.inputs],
                "outputs": [list(graph.tensors[i].shape) for i in graph.outputs],
            }
        )
        print(f"op_suite  {name:34s} {sum(operations.values()):3d} ops {operations}")
    return entries, reports


def build_pte_suite(samples: int):
    entries, reports = [], []
    for name, model, shapes, min_sqnr in program_cases():
        converted, program = op_cases.quantize(model, shapes)
        edge, report = to_espnn_edge(program)
        exported = edge.exported_program()
        emulator = ProgramEmulator(exported)
        rows, worst = [], math.inf
        for inputs in random_inputs(shapes, samples, 11):
            outputs = emulator(*inputs)
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            expected = converted(*inputs)
            expected = (
                list(expected) if isinstance(expected, (tuple, list)) else [expected]
            )
            worst = min(
                [worst]
                + [sqnr_db(w, g.to(torch.float32)) for w, g in zip(expected, outputs)]
            )
            rows.append((inputs, outputs))
        exact = not report["cpu_ops"]
        entries.append(
            (name, edge.to_executorch().buffer, xte1_vectors(rows, exact, min_sqnr))
        )
        delegates = sum(
            isinstance(m, LoweredBackendModule)
            for _, m in exported.graph_module.named_children()
        )
        reports.append(
            {
                "name": name,
                "delegates": delegates,
                "partitions": report["partitions"],
                "delegated_ops": report["delegated_ops"],
                "cpu_ops": report["cpu_ops"],
                "exact_required": exact,
                "min_sqnr_db": min_sqnr,
                "host_sqnr_vs_pytorch_db": round(worst, 1),
                "inputs": [list(t.shape) for t in rows[0][0]],
                "outputs": [list(t.shape) for t in rows[0][1]],
            }
        )
        print(
            f"pte_suite {name:34s} delegates={delegates} cpu_ops={len(report['cpu_ops'])}"
        )
    return entries, reports


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "espnn_runner" / "bundles",
    )
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    for filename, magic, builder in (
        ("op_suite.xtb", b"XTB1", build_op_suite),
        ("pte_suite.xtp", b"XTP1", build_pte_suite),
    ):
        entries, reports = builder(args.samples)
        path = args.output_dir / filename
        size = write_bundle(path, magic, entries)
        path.with_suffix(".json").write_text(json.dumps(reports, indent=2))
        print(f"wrote {path} ({size} bytes, {len(entries)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
