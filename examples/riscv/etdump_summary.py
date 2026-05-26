# Copyright 2026 The ExecuTorch Authors.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render a per-XNNPACK-op summary from an ETDump file."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from executorch.devtools import Inspector


# "Convolution (NHWC, F32) IGEMM #3" -> ("Convolution (NHWC, F32) IGEMM", 3)
_SEQ_RE = re.compile(r"^(.*?)\s+#(\d+)$")

# Wrappers around per-op events; kept separate to avoid double-counting children.
FRAMEWORK_EVENTS = frozenset(
    {
        "Method::execute",
        "Method::init",
        "Program::load_method",
        "DELEGATE_CALL",
        "OPERATOR_CALL",
    }
)

_REG_LOG_RE = re.compile(r"Note \(XNNPACK\):.*microkernel '([^']+)'")


def parse_run_log(path: Path):
    syms = set()
    with open(path, errors="ignore") as f:
        for line in f:
            m = _REG_LOG_RE.search(line)
            if m:
                syms.add(m.group(1))
    return sorted(syms)


# Two-source mapping from an ETDump op name to a symbol-substring pattern.
# When the operator type uses xnn_microkernel_type_default, runtime.c does NOT
# append a category suffix, so we fall back to matching on the base op name.
_OP_NAME_RE = re.compile(r"^(.*?)\s*\(([^)]*)\)\s*(.*)$")
_DTYPE_TOKENS = frozenset(
    {
        "F32",
        "F16",
        "QS8",
        "QU8",
        "QC8",
        "QC4",
        "QD8",
        "QC8W",
        "QC4W",
        "X8",
        "X16",
        "X24",
        "X32",
        "X64",
    }
)
# Infix between the kind token and `_ukernel_`: zero or more `<word>_`
# segments (e.g. `_gemm_ukernel_`, `_gemm_minmax_ukernel_`,
# `_gemm_minmax_fp32_ukernel_`, ...).
_INFIX = r"(?:[a-z0-9]+_)*"
_KIND_PATTERN = {
    # Microkernel categories appended by runtime.c (xnn_microkernel_type_to_string).
    "GEMM": r"_gemm_" + _INFIX + r"ukernel_",
    "IGEMM": r"_igemm_" + _INFIX + r"ukernel_",
    "DWConv": r"_dwconv_" + _INFIX + r"ukernel_",
    "Transpose": r"_transposec?_" + _INFIX + r"ukernel_",
    "Reduce": r"_(?:rsum|rmax|rminmax|rdmax|rdsum)_" + _INFIX + r"ukernel_",
    "Reduce2": r"_(?:rdmax|rdsum)_" + _INFIX + r"ukernel_",
    "VMulCAddC": r"_vmulcaddc_" + _INFIX + r"ukernel_",
    "Average Pooling": r"_(?:avgpool|gavgpool)_" + _INFIX + r"ukernel_",
    "Pixelwise Average Pooling": r"_pavgpool_" + _INFIX + r"ukernel_",
    "Conv2D HWC2CHW": r"_conv_hwc2chw_" + _INFIX + r"ukernel_",
    "SPMM": r"_spmm_" + _INFIX + r"ukernel_",
    "Subconv2D": r"_subconv2d_" + _INFIX + r"ukernel_",
    # Base op names (default microkernel type, no category suffix in the ETDump name).
    "Add": r"_v(?:add|addc)_" + _INFIX + r"ukernel_",
    "Subtract": r"_v(?:sub|subc|rsubc)_" + _INFIX + r"ukernel_",
    "Multiply": r"_v(?:mul|mulc)_" + _INFIX + r"ukernel_",
    "Divide": r"_v(?:div|divc|rdivc)_" + _INFIX + r"ukernel_",
    "Maximum": r"_v(?:max|maxc)_" + _INFIX + r"ukernel_",
    "Minimum": r"_v(?:min|minc)_" + _INFIX + r"ukernel_",
    "Clamp": r"_vclamp_" + _INFIX + r"ukernel_",
    "Sigmoid": r"_vsigmoid_" + _INFIX + r"ukernel_",
    "Tanh": r"_vtanh_" + _INFIX + r"ukernel_",
    "Negate": r"_vneg_" + _INFIX + r"ukernel_",
    "Abs": r"_vabs_" + _INFIX + r"ukernel_",
    "Square": r"_vsqr_" + _INFIX + r"ukernel_",
    "Square Root": r"_vsqrt_" + _INFIX + r"ukernel_",
    "Reciprocal Square Root": r"_vrsqrt_" + _INFIX + r"ukernel_",
    "Convert": r"_vcvt_" + _INFIX + r"ukernel_",
    "Copy": r"_(?:copy|memcpy)_" + _INFIX + r"ukernel_",
    "Constant Pad": r"_xx_pad_" + _INFIX + r"ukernel_",
    "Softmax": r"_(?:raddstoreexpminusmax|rmax)_" + _INFIX + r"ukernel_",
    "Max Pooling": r"_maxpool_" + _INFIX + r"ukernel_",
}


def op_kernels(op_name, kernels):
    m = _OP_NAME_RE.match(op_name)
    if not m:
        return []
    base, inside, tail = m.group(1).strip(), m.group(2), m.group(3).strip()
    key = tail if tail in _KIND_PATTERN else (base if base in _KIND_PATTERN else None)
    if key is None:
        return []
    dtype_tokens = [
        s.strip().lower() for s in inside.split(",") if s.strip() in _DTYPE_TOKENS
    ]
    cat_re = re.compile(_KIND_PATTERN[key])
    return [
        sym
        for sym in kernels
        if cat_re.search(sym) and all(d in sym for d in dtype_tokens)
    ]


def aggregate(etdump_path: Path):
    insp = Inspector(etdump_path=str(etdump_path))
    per_op = defaultdict(lambda: {"count": 0, "raw": []})
    framework = defaultdict(lambda: {"count": 0, "raw": []})
    for block in insp.event_blocks:
        for ev in block.events:
            m = _SEQ_RE.match(ev.name or "")
            base = m.group(1) if m else (ev.name or "<unnamed>")
            bucket = framework if base in FRAMEWORK_EVENTS else per_op
            bucket[base]["count"] += 1
            bucket[base]["raw"].extend(ev.perf_data.raw if ev.perf_data else [])
    return per_op, framework


def render(per_op, framework, etdump_path, kernels):
    def rows_of(d):
        rows = []
        for name, v in d.items():
            raw = v["raw"]
            s = sum(raw)
            rows.append(
                {
                    "op": name,
                    "count": v["count"],
                    "sum_ms": s,
                    "avg_ms": (s / len(raw)) if raw else 0.0,
                    "max_ms": max(raw) if raw else 0.0,
                    "kernels": op_kernels(name, kernels) if kernels else [],
                }
            )
        rows.sort(key=lambda r: r["sum_ms"], reverse=True)
        return rows

    op_rows = rows_of(per_op)
    fw_rows = rows_of(framework)
    ops_total = sum(r["sum_ms"] for r in op_rows)
    fw_total = sum(r["sum_ms"] for r in fw_rows)

    def fmt_table(label, rows, total):
        print(f"\n[etdump_summary] {label}  total={total:.3f} ms")
        print(
            f"{'%':>5}  {'sum_ms':>10}  {'count':>6}  {'avg_ms':>10}  {'max_ms':>10}  op"
        )
        for r in rows:
            pct = (r["sum_ms"] / total * 100.0) if total else 0.0
            print(
                f"{pct:5.1f}  {r['sum_ms']:10.3f}  {r['count']:6d}  "
                f"{r['avg_ms']:10.3f}  {r['max_ms']:10.3f}  {r['op']}"
            )

    print(f"[etdump_summary] {etdump_path}")
    fmt_table(f"XNNPACK ops ({len(op_rows)} unique)", op_rows, ops_total)
    fmt_table(f"Framework wrappers ({len(fw_rows)})", fw_rows, fw_total)
    if kernels:
        print(f"\n[etdump_summary] Registered XNNPACK microkernels ({len(kernels)}):")
        for sym in kernels:
            print(f"  {sym}")

    return op_rows, fw_rows, ops_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etdump", type=Path)
    parser.add_argument("--run-log", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.etdump.exists():
        print(f"[etdump_summary] missing {args.etdump}", file=sys.stderr)
        sys.exit(1)

    kernels = []
    if args.run_log is not None:
        if not args.run_log.exists():
            print(f"[etdump_summary] missing run log {args.run_log}", file=sys.stderr)
            sys.exit(1)
        kernels = parse_run_log(args.run_log)

    per_op, framework = aggregate(args.etdump)
    op_rows, fw_rows, ops_total = render(per_op, framework, args.etdump, kernels)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "etdump": str(args.etdump),
                    "run_log": str(args.run_log) if args.run_log else None,
                    "ops_total_ms": ops_total,
                    "registered_kernels": kernels,
                    "ops": op_rows,
                    "framework": fw_rows,
                },
                indent=2,
            )
        )
        print(f"[etdump_summary] wrote {args.json}")


if __name__ == "__main__":
    main()
