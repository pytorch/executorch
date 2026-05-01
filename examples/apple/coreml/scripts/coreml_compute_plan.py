# Copyright © 2026 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.

"""Report which CoreML operations would dispatch to ANE / GPU / CPU.

The CoreML runtime decides at compile/load time which compute device each
MIL operation will run on; that decision is exposed by ``MLComputePlan``
in coremltools 9.0+.  This script wraps that API so users can answer
"why isn't my model running on the ANE?" without writing Swift.

Usage::

    # Analyze a CoreML model directly (mlpackage or compiled mlmodelc).
    python coreml_compute_plan.py --model_path path/to/model.mlpackage

    # Analyze every Core ML partition embedded in an ExecuTorch .pte.
    python coreml_compute_plan.py --model_path path/to/program.pte

    # Show ops that fell off the ANE, grouped by op type.
    python coreml_compute_plan.py --model_path model.mlpackage --show_non_ane

    # Pick which devices the runtime is allowed to consider.
    python coreml_compute_plan.py --model_path model.mlpackage \\
        --compute_units cpu_and_ne
"""

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import coremltools as ct
from coremltools.models.compute_device import (
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)
from coremltools.models.compute_plan import MLComputePlan


_DEVICE_NAMES: List[Tuple[type, str]] = [
    (MLNeuralEngineComputeDevice, "ANE"),
    (MLGPUComputeDevice, "GPU"),
    (MLCPUComputeDevice, "CPU"),
]

_COMPUTE_UNIT_CHOICES = {
    "all": ct.ComputeUnit.ALL,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}


def _device_name(device) -> str:
    if device is None:
        return "unknown"
    for cls, name in _DEVICE_NAMES:
        if isinstance(device, cls):
            return name
    return type(device).__name__


def _iter_operations(block) -> Iterable:
    for op in block.operations:
        yield op
        for nested in getattr(op, "blocks", None) or []:
            yield from _iter_operations(nested)


def _ensure_compiled(model_path: str, tmpdir: str) -> str:
    """Return a `.mlmodelc` path; compile from `.mlpackage` if needed."""
    if model_path.endswith(".mlmodelc"):
        return model_path
    if model_path.endswith(".mlpackage"):
        dest = os.path.join(
            tmpdir, os.path.basename(model_path).replace(".mlpackage", ".mlmodelc")
        )
        return str(ct.models.utils.compile_model(model_path, destination_path=dest))
    raise ValueError(
        f"Expected a .mlpackage or .mlmodelc path, got: {model_path}"
    )


def _extract_models_from_pte(pte_path: str, out_dir: str) -> List[str]:
    """Pull every CoreML partition out of a .pte into `out_dir`.

    Returns a list of paths to the extracted model directories (which
    `MLComputePlan.load_from_path` accepts directly).
    """
    # Imported lazily so the script still runs against a plain .mlpackage
    # without requiring the executorch package.
    from executorch.backends.apple.coreml import executorchcoreml
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import (
        BackendDelegateDataReference,
        DataLocation,
    )
    import json

    COREML_BACKEND_ID = "CoreMLBackend"
    MAGIC_NUMBER = b"CMJR"

    with open(pte_path, "rb") as f:
        pte_data = f.read()
    pte_file = deserialize_pte_binary(pte_data)
    program = pte_file.program

    named_data = {}
    if pte_file.named_data is not None:
        for key, entry in pte_file.named_data.pte_data.items():
            named_data[key] = pte_file.named_data.buffers[entry.buffer_index]

    delegates = sum((p.delegates for p in program.execution_plan), [])
    coreml_delegates = [d for d in delegates if d.id == COREML_BACKEND_ID]
    if not coreml_delegates:
        return []

    extracted: List[str] = []
    seen_keys: set = set()
    for i, delegate in enumerate(coreml_delegates):
        ref: BackendDelegateDataReference = delegate.processed
        if ref.location != DataLocation.INLINE:
            continue
        raw = program.backend_delegate_data[ref.index].data
        model_bytes: Optional[bytes] = None
        name: Optional[str] = None
        if raw.startswith(MAGIC_NUMBER):
            reference = json.loads(raw[len(MAGIC_NUMBER) :].decode("utf-8"))
            key = reference.get("key")
            if key in seen_keys or key not in named_data:
                continue
            seen_keys.add(key)
            model_bytes = named_data[key]
            name = key
        else:
            model_bytes = raw
            name = f"model_{i + 1}"
        if model_bytes is None:
            continue
        out_path = os.path.join(out_dir, name)
        os.makedirs(out_path, exist_ok=True)
        if executorchcoreml.unflatten_directory_contents(model_bytes, out_path):
            extracted.append(out_path)
    return extracted


def analyze_one(model_path: str, compute_units: ct.ComputeUnit) -> List[Tuple[str, str, str]]:
    """Return [(function, operator_name, device)] for every op that has a plan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        compiled = _ensure_compiled(model_path, tmpdir)
        plan = MLComputePlan.load_from_path(compiled, compute_units=compute_units)
        program = plan.model_structure.program
        if program is None:
            raise RuntimeError(
                f"{model_path} is not an MLProgram model; this tool only supports "
                "the MLProgram backend (the CoreML backend executorch produces today)."
            )

        rows: List[Tuple[str, str, str]] = []
        for fname, fn in program.functions.items():
            for op in _iter_operations(fn.block):
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if usage is None:
                    # Constants and similar non-dispatched ops don't have a plan.
                    continue
                rows.append(
                    (fname, op.operator_name, _device_name(usage.preferred_compute_device))
                )
        return rows


def _print_report(label: str, rows: List[Tuple[str, str, str]], show_non_ane: bool) -> None:
    print(f"\n=== {label} ===")
    if not rows:
        print("  (no dispatched operations found)")
        return
    by_device = Counter(device for _, _, device in rows)
    total = sum(by_device.values())
    for device in ("ANE", "GPU", "CPU", "unknown"):
        count = by_device.get(device, 0)
        if count == 0:
            continue
        pct = 100.0 * count / total
        print(f"  {device}: {count:5d} / {total} ({pct:5.1f}%)")

    if show_non_ane:
        non_ane = [(fn, op_name) for fn, op_name, dev in rows if dev != "ANE"]
        if non_ane:
            print("\n  Non-ANE op types:")
            for op_name, count in Counter(op for _, op in non_ane).most_common():
                print(f"    {count:5d}  {op_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to a .pte, .mlpackage, or .mlmodelc.",
    )
    parser.add_argument(
        "--compute_units",
        default="cpu_and_ne",
        choices=sorted(_COMPUTE_UNIT_CHOICES),
        help="Which devices the runtime may use when planning dispatch.",
    )
    parser.add_argument(
        "--show_non_ane",
        action="store_true",
        help="List op types that did not get assigned to the ANE.",
    )
    args = parser.parse_args()

    compute_units = _COMPUTE_UNIT_CHOICES[args.compute_units]
    model_path = args.model_path

    if model_path.endswith(".pte"):
        with tempfile.TemporaryDirectory() as out_dir:
            extracted = _extract_models_from_pte(model_path, out_dir)
            if not extracted:
                print(
                    f"{model_path} does not contain any CoreML delegate partitions.",
                    file=sys.stderr,
                )
                return 1
            for path in extracted:
                rows = analyze_one(path, compute_units)
                _print_report(os.path.basename(path), rows, args.show_non_ane)
    else:
        rows = analyze_one(model_path, compute_units)
        _print_report(os.path.basename(model_path.rstrip("/")), rows, args.show_non_ane)
    return 0


if __name__ == "__main__":
    sys.exit(main())
