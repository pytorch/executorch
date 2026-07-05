"""Verification (doc Part 8, criterion 1): greedy DFlash must be LOSSLESS --
identical tokens to greedy autoregressive baseline, since the target verifies
every accepted token. Any divergence means the speculative loop is broken."""
import subprocess, sys, re

PROMPT = "Write a Python function that takes a list of integers and returns the second largest number in the list."
N = 96

def run(script, extra):
    out = subprocess.run(
        [sys.executable, f"examples/models/qwen3/{script}",
         "--prompt", PROMPT, "--max-new-tokens", str(N)] + extra,
        capture_output=True, text=True, cwd=".",
    ).stdout
    m = re.search(r"Generated \([^)]*\): (.*?)\n\n", out, re.DOTALL)
    return m.group(1) if m else out

baseline = run("run_baseline.py", [])
dflash = run("run_dflash.py", [])

print("=== BASELINE ===\n", baseline[:400])
print("\n=== DFLASH ===\n", dflash[:400])
print("\n=== RESULT ===")
if baseline.strip() == dflash.strip():
    print("PASS: DFlash output is token-for-token identical to baseline (LOSSLESS)")
else:
    # find first divergence
    for i, (a, b) in enumerate(zip(baseline, dflash)):
        if a != b:
            print(f"DIVERGE at char {i}: baseline={baseline[i:i+30]!r} dflash={dflash[i:i+30]!r}")
            break
    print("FAIL: outputs differ -- speculative loop is not lossless")
