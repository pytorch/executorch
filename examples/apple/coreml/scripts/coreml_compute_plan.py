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
import sys
import tempfile
from collections import Counter
from typing import Iterable, List, Tuple

import coremltools as ct
from coremltools.models.compute_device import (
    MLCPUComputeDevice,
    MLGPUComputeDevice,
    MLNeuralEngineComputeDevice,
)
from coremltools.models.compute_plan import MLComputePlan

from executorch.examples.apple.coreml.scripts.extract_coreml_models import (
    extract_coreml_models,
)


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
    raise ValueError(f"Expected a .mlpackage or .mlmodelc path, got: {model_path}")


def analyze_one(
    model_path: str, compute_units: ct.ComputeUnit
) -> List[Tuple[str, str, str]]:
    """Return [(function, operator_name, device)] for every op that has a plan.

    coremltools 9.0's ``MLComputePlan.load_from_path`` only exposes usage for
    the default function of a multifunction package, so a multifunction
    .mlpackage is analyzed function-by-function by projecting each function
    as the ``main`` of a temp single-function copy.
    """
    function_names = _mlpackage_function_names(model_path)
    if len(function_names) <= 1:
        return _analyze_compiled(model_path, compute_units)
    rows: List[Tuple[str, str, str]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in function_names:
            projected = _project_to_single(model_path, fname, tmpdir)
            for _, op_name, device in _analyze_compiled(projected, compute_units):
                rows.append((fname, op_name, device))
    return rows


def _analyze_compiled(
    model_path: str, compute_units: ct.ComputeUnit
) -> List[Tuple[str, str, str]]:
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
                    (
                        fname,
                        op.operator_name,
                        _device_name(usage.preferred_compute_device),
                    )
                )
        return rows


def _mlpackage_function_names(model_path: str) -> List[str]:
    """Names of the MLProgram functions inside an .mlpackage, or [] otherwise."""
    if not model_path.endswith(".mlpackage"):
        return []
    spec = ct.models.MLModel(model_path, skip_model_load=True).get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        return []
    return list(spec.mlProgram.functions.keys())


def _project_to_single(src_mlpackage: str, function_name: str, tmpdir: str) -> str:
    """Re-save ``src_mlpackage`` with only ``function_name`` exposed as ``main``."""
    from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

    dest = os.path.join(tmpdir, f"{function_name}.mlpackage")
    desc = MultiFunctionDescriptor()
    desc.add_function(
        src_mlpackage,
        src_function_name=function_name,
        target_function_name="main",
    )
    desc.default_function_name = "main"
    save_multifunction(desc, dest)
    return dest


def _print_report(
    label: str, rows: List[Tuple[str, str, str]], show_non_ane: bool
) -> None:
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
        with open(model_path, "rb") as f:
            pte_data = f.read()
        with tempfile.TemporaryDirectory() as out_dir:
            extracted = extract_coreml_models(pte_data, out_dir=out_dir)
            if not extracted:
                print(
                    f"{model_path} does not contain any CoreML delegate partitions.",
                    file=sys.stderr,
                )
                return 1
            for path in extracted:
                rows = analyze_one(str(path), compute_units)
                _print_report(path.name, rows, args.show_non_ane)
    else:
        rows = analyze_one(model_path, compute_units)
        _print_report(os.path.basename(model_path.rstrip("/")), rows, args.show_non_ane)
    return 0


if __name__ == "__main__":
    sys.exit(main())
