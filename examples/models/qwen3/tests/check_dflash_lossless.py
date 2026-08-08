# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that DFlash produces the exact same output as normal greedy decoding.

A command failure or an output mismatch both FAIL this check.
"""

import re
import subprocess
import sys

PROMPT = "Write a Python function that takes a list of integers and returns the second largest number in the list."
N = 96


def run(script, extra):
    result = subprocess.run(
        [
            sys.executable,
            f"examples/models/qwen3/{script}",
            "--prompt",
            PROMPT,
            "--max-new-tokens",
            str(N),
        ]
        + extra,
        capture_output=True,
        text=True,
        cwd=".",
    )
    if result.returncode != 0:
        print(f"FAIL: {script} exited with code {result.returncode}")
        print(f"--- stderr ---\n{result.stderr}")
        print(f"--- stdout ---\n{result.stdout}")
        sys.exit(1)

    m = re.search(r"Generated \([^)]*\): (.*?)\n\n", result.stdout, re.DOTALL)
    if m is None:
        print(f"FAIL: could not find a 'Generated (...):' line in {script}'s output")
        print(f"--- stdout ---\n{result.stdout}")
        sys.exit(1)
    return m.group(1)


baseline = run("run_baseline.py", [])
dflash = run("run_dflash.py", [])

print("BASELINE:\n", baseline[:400])
print("\nDFLASH:\n", dflash[:400])
print("\nRESULT:")
if baseline.strip() == dflash.strip():
    print("PASS: DFlash output is token-for-token identical to baseline (LOSSLESS)")
else:
    for i, (a, b) in enumerate(zip(baseline, dflash)):
        if a != b:
            print(
                f"DIVERGE at char {i}: baseline={baseline[i:i+30]!r} dflash={dflash[i:i+30]!r}"
            )
            break
    print("FAIL: outputs differ -- speculative loop is not lossless")
    sys.exit(1)
