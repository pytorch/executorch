#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from parity import compare_trace_payloads


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Voxtral parity traces from eager and runner paths."
    )
    parser.add_argument("--reference", required=True, help="Path to reference JSON trace.")
    parser.add_argument("--candidate", required=True, help="Path to candidate JSON trace.")
    parser.add_argument(
        "--hidden-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for hidden-state comparisons.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the comparison result as JSON.",
    )
    args = parser.parse_args()

    reference = json.loads(Path(args.reference).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    result = compare_trace_payloads(
        reference,
        candidate,
        hidden_atol=args.hidden_atol,
    )

    for check in result["checks"]:
        status = "PASS" if check["ok"] else "FAIL"
        print(f"{status} {check['name']}: {json.dumps(check, sort_keys=True)}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
