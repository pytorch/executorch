# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Build examples/espressif/espnn_runner for the connected chip, flash a bundle
(.xtb int8 graphs or .xtp ExecuTorch programs) and report each case.

    python examples/espressif/scripts/run_bundle.py --port /dev/ttyUSB0 \
        examples/espressif/espnn_runner/bundles/op_suite.xtb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import espnn_board as board  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--target", help="override the detected chip")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--build-root", type=Path, default=board.RUNNER_DIR / "build")
    parser.add_argument("--executorch-build-dir", type=Path)
    parser.add_argument("--executorch-source-dir", type=Path)
    parser.add_argument("--json", type=Path, help="write the parsed results here")
    args = parser.parse_args()

    chip = board.detect_chip(args.port)
    target = args.target or chip.target
    executorch = args.bundle.read_bytes()[:4] == b"XTP1"
    mode = "executorch" if executorch else "standalone"
    print(
        f"{chip.description}: {chip.flash_mb} MB flash; building {mode} runner for {target}",
        flush=True,
    )
    build = board.RunnerBuild(
        target=target,
        executorch=executorch,
        build_dir=args.build_root / f"{target}-{mode}",
        executorch_build_dir=args.executorch_build_dir,
        executorch_source_dir=args.executorch_source_dir,
    )
    result = board.run_bundle(
        build,
        args.bundle,
        args.port,
        args.iterations,
        args.timeout,
        build.build_dir / "logs",
        on_line=lambda line: print(line, flush=True),
    )
    if args.json:
        args.json.write_text(
            json.dumps(
                {"chip": result.chip, "cases": result.cases, "passed": result.passed},
                indent=2,
            )
        )
    print(f"RESULT: {'PASS' if result.passed else 'FAIL'} ({len(result.cases)} cases)")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
