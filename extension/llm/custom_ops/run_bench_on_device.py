#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Build SDPA benchmark binaries for Android, deploy to device via adb,
run them, and produce formatted comparison tables.

Supports bench_sdpa (flash attention vs GEMM-based SDPA vs OnnxGQA)
and bench_transposed_cache (standard vs transposed KV cache layout).

Devices can be specified by serial (--device_serial) or leased from the
Enkaku device pool (--device_model s25).

Usage:
    # Lease an S25 and run both benchmarks
    python run_bench_on_device.py --device_model s25 --benchmark both

    # Use a specific device serial
    python run_bench_on_device.py --device_serial <serial> --benchmark bench_sdpa
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

DEVICE_TMP_DIR = "/data/local/tmp"
DEVICE_BINARY_PATH = f"{DEVICE_TMP_DIR}/bench_program"
DEVICE_RESULTS_PATH = f"{DEVICE_TMP_DIR}/bench_results.json"

# xplat cxx_binary targets that can be cross-compiled for Android
BENCHMARK_TARGETS = {
    "bench_sdpa": "fbsource//xplat/executorch/extension/llm/custom_ops:bench_sdpa",
    "bench_transposed_cache": (
        "fbsource//xplat/executorch/extension/llm/custom_ops:bench_transposed_cache"
    ),
}

BUILD_MODE = "@arvr/mode/android/linux/opt"

NDK_STRIP_REL_PATH = (
    "third-party/toolchains/android-ndk/r17c/toolchains/"
    "aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin"
)

# Enkaku device model mapping (matches vizard/utils/device_utils.py)
ANDROID_PHONE_TYPE = {
    "s25": "hammerhead-sm-s931u1",
    "s24": "hammerhead-sm-s921u1",
    "s23": "hammerhead-sm-s911u1",
    "s22": "hammerhead-sm-s901u1",
    "pixel_9_pro": "meta watch-pixel 9 pro",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def find_fbsource_root() -> str:
    """Locate the fbsource repo root via sl/hg or directory walk."""
    for cmd in ["sl root", "hg root"]:
        try:
            result = subprocess.run(
                cmd.split(), capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    path = os.getcwd()
    while path != "/":
        if os.path.isdir(os.path.join(path, ".hg")) or os.path.isdir(
            os.path.join(path, ".sl")
        ):
            return path
        path = os.path.dirname(path)
    raise RuntimeError("Cannot find fbsource root. Run from within the repo.")


def find_ndk_strip(fbsource_root: str) -> Optional[str]:
    """Find the NDK strip binary for ARM64 Android targets."""
    bin_dir = os.path.join(fbsource_root, NDK_STRIP_REL_PATH)
    if not os.path.isdir(bin_dir):
        return None
    for filename in os.listdir(bin_dir):
        if "strip" in filename:
            return os.path.join(bin_dir, filename)
    return None


def run_cmd(
    cmd: List[str],
    cwd: Optional[str] = None,
    check: bool = True,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Run a shell command, printing it for visibility."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=check, timeout=timeout
    )


def adb_cmd(serial: str, *args: str) -> List[str]:
    """Construct an adb command list with the given device serial."""
    return ["adb", "-s", serial] + list(args)


# ---------------------------------------------------------------------------
# Device leasing via Enkaku
# ---------------------------------------------------------------------------


def ensure_adb() -> None:
    """Ensure adb is on PATH, fetching via fbpkg if needed."""
    try:
        subprocess.run(
            ["adb", "version"], capture_output=True, text=True, timeout=5
        )
    except FileNotFoundError:
        print("[Lease] adb not found, fetching via fbpkg...")
        adb_dir = "/tmp/adb"
        subprocess.run(
            ["fbpkg", "fetch", "adb:stable", "-d", adb_dir],
            check=True, timeout=120,
        )
        os.environ["PATH"] = adb_dir + ":" + os.environ.get("PATH", "")
        print(f"  adb installed to {adb_dir}")


def lease_device(
    device_model: str,
    runtime_limit_sec: int = 3600,
) -> str:
    """Lease an Android phone via Enkaku. Returns the device serial.

    Sets up os.environ so subsequent adb commands route through
    Enkaku's tunnel to the remote device.
    """
    if device_model not in ANDROID_PHONE_TYPE:
        raise ValueError(
            f"Unknown device model '{device_model}'. "
            f"Choose from: {', '.join(ANDROID_PHONE_TYPE)}"
        )

    ensure_adb()

    ek_model = ANDROID_PHONE_TYPE[device_model]
    print(f"\n[Lease] Leasing {device_model} ({ek_model})...")

    # Start the Enkaku daemon
    subprocess.run(["ek", "start"], capture_output=True, timeout=30)

    lease_args = [
        "ek", "re", "lease",
        "-J", "platform=riot",
        "-J", f"device_model={ek_model}",
        "-J", "host_os=linux",
        "-J", "adb_device=true",
        "--wait-for-android-device",
        "--wait-for-android-device-args="
        "--expected-local-adbserver-devices=1 "
        "--expected-remote-adbserver-devices=1",
        "--runtime-limit-in-secs", str(runtime_limit_sec),
        "--json",
        "--priority", "1",
        "--use-case-id", "wearables-human-lease",
        "--auth", "appid",
    ]

    print(f"  $ {' '.join(lease_args)}")
    result = subprocess.run(
        lease_args, capture_output=True, text=True, timeout=1800
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Device lease failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[:1000]}\nstderr: {result.stderr[:1000]}"
        )

    # Parse the JSON block from ek output
    leased_info = _parse_lease_json(result.stdout)
    if leased_info is None:
        raise RuntimeError(
            f"Could not parse lease JSON from output:\n{result.stdout[:2000]}"
        )

    # Extract device serial
    device_serial = _extract_device_serial(device_model, leased_info)
    if device_serial is None:
        release_device(env=os.environ.copy())
        raise RuntimeError(
            f"No matching device serial found in lease info:\n"
            f"{json.dumps(leased_info, indent=2)[:1000]}"
        )

    print(f"  Leased device: {device_serial}")

    # Set up ADB environment so adb routes through Enkaku's tunnel
    _setup_adb_env(device_serial)

    return device_serial


def _parse_lease_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON block from Enkaku lease stdout."""
    json_lines = []
    in_json = False
    for line in stdout.splitlines():
        if line.strip() == "{":
            in_json = True
        if in_json:
            json_lines.append(line)
        if line.strip() == "}":
            in_json = False

    if not json_lines:
        return None
    try:
        return json.loads("\n".join(json_lines))
    except json.JSONDecodeError:
        return None


def _extract_device_serial(
    device_model: str, leased_info: Dict[str, Any]
) -> Optional[str]:
    """Get the adb serial from Enkaku lease info for the requested model."""
    android_devices = leased_info.get("android_devices", [])
    if not android_devices:
        return None

    # For phones, match by product_model substring
    ek_model = ANDROID_PHONE_TYPE.get(device_model, "")
    match = re.search(r"(sm-[a-z0-9]+|pixel .+)", ek_model)
    expected = match.group(1).lower() if match else ek_model.lower()

    for dev in android_devices:
        product = dev.get("adb_prop", {}).get("product_model", "").lower()
        if expected in product:
            return dev["serial"]

    # Fallback: return the first device
    return android_devices[0].get("serial")


def _setup_adb_env(device_serial: str) -> None:
    """Run 'ek ae' to get ADB routing env vars and merge into os.environ."""
    result = subprocess.run(
        ["ek", "ae", "-j", "-x", device_serial],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to get ADB env for {device_serial}: {result.stderr}"
        )

    device_env = json.loads(result.stdout)
    for key, value in device_env.items():
        os.environ[key] = str(value)
    print(f"  ADB environment configured for {device_serial}")


def release_device(
    device_serial: Optional[str] = None, env: Optional[Dict[str, Any]] = None
) -> None:
    """Release a leased device via Enkaku."""
    use_env = env or os.environ.copy()

    if device_serial is None:
        # Release all leased devices
        try:
            result = subprocess.run(
                ["ek", "al", "-j", "-x"],
                capture_output=True, text=True, timeout=10, env=use_env,
            )
            devices = json.loads(result.stdout)
            for dev in devices:
                release_device(dev["serial"], use_env)
        except Exception as e:
            print(f"  Warning: could not list devices for release: {e}")
        return

    try:
        result = subprocess.run(
            ["ek", "ae", "-j", "-x", device_serial],
            capture_output=True, text=True, timeout=10, env=use_env,
        )
        device_env = json.loads(result.stdout)
        session_id = device_env.get("ENKAKU_PEER_SESSION_ID")
        if session_id:
            subprocess.run(
                ["ek", "re", "release", session_id],
                capture_output=True, text=True, timeout=30, env=use_env,
            )
            print(f"  Released device {device_serial}")
        else:
            print(f"  Warning: no session ID for {device_serial}")
    except Exception as e:
        print(f"  Warning: failed to release {device_serial}: {e}")


# ---------------------------------------------------------------------------
# Build / deploy / run
# ---------------------------------------------------------------------------


def build_benchmark(
    benchmark_name: str, fbsource_root: str, no_strip: bool = False
) -> str:
    """Build the benchmark for Android and optionally strip the binary.

    Returns the path to the (possibly stripped) binary.
    """
    target = BENCHMARK_TARGETS[benchmark_name]
    print(f"\n[Build] Building {benchmark_name} for Android...")

    result = run_cmd(
        ["buck2", "build", BUILD_MODE, target, "--show-full-output"],
        cwd=fbsource_root,
        timeout=1200,
    )

    # Parse "target path" output to extract binary location
    binary_path = None
    for line in result.stdout.strip().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            binary_path = parts[1].strip()
            break

    if not binary_path or not os.path.exists(binary_path):
        raise RuntimeError(
            f"Build succeeded but binary not found. Output:\n{result.stdout}"
        )

    print(f"  Built: {binary_path}")

    if not no_strip:
        strip_tool = find_ndk_strip(fbsource_root)
        if strip_tool:
            tmp_binary = os.path.join(
                tempfile.gettempdir(), f"{benchmark_name}_stripped"
            )
            shutil.copy2(binary_path, tmp_binary)
            run_cmd([strip_tool, tmp_binary])
            orig_mb = os.path.getsize(binary_path) / (1024 * 1024)
            new_mb = os.path.getsize(tmp_binary) / (1024 * 1024)
            print(f"  Stripped: {orig_mb:.1f}MB -> {new_mb:.1f}MB")
            return tmp_binary
        else:
            print("  Warning: NDK strip tool not found, skipping strip")

    return binary_path


def deploy_to_device(serial: str, binary_path: str) -> None:
    """Push the benchmark binary to the Android device."""
    print(f"\n[Deploy] Pushing to device {serial}...")
    run_cmd(adb_cmd(serial, "shell", "rm", "-f", DEVICE_BINARY_PATH))
    run_cmd(adb_cmd(serial, "push", binary_path, DEVICE_BINARY_PATH))
    run_cmd(adb_cmd(serial, "shell", "chmod", "755", DEVICE_BINARY_PATH))
    print(f"  Deployed to {DEVICE_BINARY_PATH}")


def run_on_device(
    serial: str,
    benchmark_filter: Optional[str] = None,
    benchmark_repetitions: Optional[int] = None,
    cpu_threads: Optional[int] = None,
) -> Tuple[str, str]:
    """Execute the benchmark on device. Returns (stdout, stderr)."""
    print(f"\n[Run] Executing benchmark on device...")

    bench_args = [
        f"--benchmark_out={DEVICE_RESULTS_PATH}",
        "--benchmark_out_format=json",
    ]
    if benchmark_filter:
        bench_args.append(f"--benchmark_filter={benchmark_filter}")
    if benchmark_repetitions:
        bench_args.append(f"--benchmark_repetitions={benchmark_repetitions}")

    env_prefix = ""
    if cpu_threads:
        env_prefix = f"export OMP_NUM_THREADS={cpu_threads}; "

    shell_cmd = f"{env_prefix}{DEVICE_BINARY_PATH} {' '.join(bench_args)}"
    result = run_cmd(
        adb_cmd(serial, "shell", shell_cmd), check=False, timeout=1800
    )

    if result.returncode != 0:
        print(f"  Warning: benchmark exited with code {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")

    return result.stdout, result.stderr


def pull_results(serial: str, local_dir: str) -> str:
    """Pull the JSON results file from the device. Returns local path."""
    local_path = os.path.join(local_dir, "bench_results.json")
    print("\n[Pull] Fetching results from device...")
    run_cmd(adb_cmd(serial, "pull", DEVICE_RESULTS_PATH, local_path))
    return local_path


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_benchmark_name(name: str) -> Dict[str, Any]:
    """Parse a Google Benchmark name into fixture, benchmark, and named args.

    Handles both named args ("B:1/Hq:32") and positional args ("1/32").
    """
    parts = name.split("/")
    fixture = parts[0] if parts else name
    benchmark = parts[1] if len(parts) > 1 else ""
    args: Dict[str, int] = {}

    for part in parts[2:]:
        if ":" in part:
            key, val = part.split(":", 1)
            try:
                args[key] = int(val)
            except ValueError:
                pass
        else:
            try:
                args[f"arg{len(args)}"] = int(part)
            except ValueError:
                pass

    return {"fixture": fixture, "benchmark": benchmark, "args": args}


def normalize_to_us(time_val: float, time_unit: str) -> float:
    """Convert a time value to microseconds."""
    multipliers = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
    return time_val * multipliers.get(time_unit, 1.0)


def format_time_us(us: float) -> str:
    """Format a microsecond value with the most readable unit."""
    if us < 1.0:
        return f"{us * 1000:.0f} ns"
    elif us < 1000.0:
        return f"{us:.1f} us"
    elif us < 1e6:
        return f"{us / 1000:.2f} ms"
    else:
        return f"{us / 1e6:.3f} s"


def format_speedup(baseline: float, comparison: float) -> str:
    """Format the speedup of comparison relative to baseline.

    > 1.0 means comparison is faster than baseline.
    """
    ratio = baseline / comparison
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    else:
        return f"{1.0 / ratio:.2f}x slower"


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def print_table(headers: List[str], rows: List[List[str]], title: str = "") -> None:
    """Print a simple aligned ASCII table."""
    if title:
        print(f"\n{title}")
    if not rows:
        print("  (no data)")
        return

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print(f"  {fmt.format(*headers)}")
    print(f"  {sep}")
    for row in rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        print(f"  {fmt.format(*padded)}")


def config_sort_key(key: str, field: str = "StartPos") -> int:
    """Extract a numeric sort key from a config string."""
    m = re.search(rf"{field}=(\d+)", key)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Result grouping: collect benchmark entries into {config -> {label -> time_us}}
# ---------------------------------------------------------------------------


def group_results(
    benchmarks: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """Group benchmark JSON entries by config key.

    Returns {config_key: {label: time_us}} where label encodes
    the implementation name and layout, e.g. "StandardSDPA(std)".
    """
    results: Dict[str, Dict[str, float]] = defaultdict(dict)

    for bm in benchmarks:
        # When repetitions > 1, prefer the _mean aggregate
        if bm.get("run_type") == "aggregate" and "_mean" not in bm.get("name", ""):
            continue

        parsed = parse_benchmark_name(bm["name"])
        bench_name = parsed["benchmark"]
        args = parsed["args"]
        time_us = normalize_to_us(bm["real_time"], bm.get("time_unit", "ns"))

        trans = args.get("Trans")
        layout = "trans" if trans else "std"

        # Build the config key from the shape args (excluding Trans)
        shape_parts = []
        for k in ["B", "Hq", "Hkv", "H", "D", "MaxS", "StartPos", "SeqLen"]:
            if k in args:
                shape_parts.append(f"{k}={args[k]}")
        config_key = ", ".join(shape_parts)

        # Label includes layout when Trans arg exists
        if trans is not None:
            label = f"{bench_name}({layout})"
        else:
            label = bench_name

        results[config_key][label] = time_us

    return dict(results)


def split_decode_prefill(
    results: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Split grouped results into decode (SeqLen=1) and prefill buckets."""
    decode = {}
    prefill = {}
    for key, times in results.items():
        seq_len = config_sort_key(key, "SeqLen")
        if seq_len == 1:
            decode[key] = times
        else:
            prefill[key] = times
    return decode, prefill


# ---------------------------------------------------------------------------
# bench_sdpa summary
# ---------------------------------------------------------------------------


def summarize_sdpa_results(benchmarks: List[Dict]) -> None:
    """Print comparison tables for bench_sdpa results."""
    results = group_results(benchmarks)
    decode, prefill = split_decode_prefill(results)

    _print_sdpa_comparison(
        decode, "SDPA Comparison — Decode (seq_len=1)", sort_field="StartPos"
    )

    _print_layout_effect(
        decode,
        ["StandardSDPA", "OnnxGQA"],
        "Transposed Layout Effect — Decode",
        sort_field="StartPos",
    )

    _print_sdpa_comparison(
        prefill, "SDPA Comparison — Prefill", sort_field="SeqLen"
    )


def _print_sdpa_comparison(
    configs: Dict[str, Dict[str, float]], title: str, sort_field: str
) -> None:
    """Print a table comparing CustomSDPA vs StandardSDPA vs OnnxGQA."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    headers = [
        sort_field,
        "CustomSDPA",
        "StdSDPA(std)",
        "StdSDPA(trans)",
        "OnnxGQA(std)",
        "OnnxGQA(trans)",
        "Std(s)/Custom",
        "Onnx(s)/Custom",
    ]
    rows = []

    for key in sorted(configs, key=lambda k: config_sort_key(k, sort_field)):
        t = configs[key]
        pos = str(config_sort_key(key, sort_field))

        custom = t.get("CustomSDPA") or t.get("CustomSDPA(std)")
        std_s = t.get("StandardSDPA(std)")
        std_t = t.get("StandardSDPA(trans)")
        onnx_s = t.get("OnnxGQA(std)")
        onnx_t = t.get("OnnxGQA(trans)")

        row = [
            pos,
            format_time_us(custom) if custom else "-",
            format_time_us(std_s) if std_s else "-",
            format_time_us(std_t) if std_t else "-",
            format_time_us(onnx_s) if onnx_s else "-",
            format_time_us(onnx_t) if onnx_t else "-",
            format_speedup(std_s, custom) if (custom and std_s) else "-",
            format_speedup(onnx_s, custom) if (custom and onnx_s) else "-",
        ]
        rows.append(row)

    print_table(headers, rows)


def _print_layout_effect(
    configs: Dict[str, Dict[str, float]],
    bench_names: List[str],
    title: str,
    sort_field: str,
) -> None:
    """For each benchmark, compare standard vs transposed layout."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    for bench in bench_names:
        headers = [sort_field, "Standard", "Transposed", "Speedup (trans)"]
        rows = []
        for key in sorted(configs, key=lambda k: config_sort_key(k, sort_field)):
            t = configs[key]
            pos = str(config_sort_key(key, sort_field))
            std_t = t.get(f"{bench}(std)")
            trs_t = t.get(f"{bench}(trans)")
            if std_t and trs_t:
                rows.append([
                    pos,
                    format_time_us(std_t),
                    format_time_us(trs_t),
                    format_speedup(std_t, trs_t),
                ])
        if rows:
            print_table(headers, rows, title=f"  {bench}:")


# ---------------------------------------------------------------------------
# bench_transposed_cache summary
# ---------------------------------------------------------------------------


def summarize_transposed_cache_results(benchmarks: List[Dict]) -> None:
    """Print comparison tables for bench_transposed_cache results."""
    results = group_results(benchmarks)
    decode, prefill = split_decode_prefill(results)

    for layout, label in [("std", "Standard Layout"), ("trans", "Transposed Layout")]:
        _print_tc_comparison(
            decode,
            layout,
            f"Transposed Cache — Decode, {label}",
            sort_field="StartPos",
        )

    _print_layout_effect(
        decode,
        ["CustomSDPA", "StandardSDPA", "OnnxGQA", "CombinedUpdateSDPA", "UpdateCache"],
        "Layout Effect — Decode",
        sort_field="StartPos",
    )

    for layout, label in [("std", "Standard Layout"), ("trans", "Transposed Layout")]:
        _print_tc_comparison(
            prefill,
            layout,
            f"Transposed Cache — Prefill, {label}",
            sort_field="SeqLen",
        )


def _print_tc_comparison(
    configs: Dict[str, Dict[str, float]],
    layout: str,
    title: str,
    sort_field: str,
) -> None:
    """Print a table for one layout of the transposed-cache benchmark."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    headers = [
        sort_field,
        "CustomSDPA",
        "StandardSDPA",
        "OnnxGQA",
        "Combined",
        "UpdateCache",
    ]
    rows = []
    for key in sorted(configs, key=lambda k: config_sort_key(k, sort_field)):
        t = configs[key]
        pos = str(config_sort_key(key, sort_field))

        custom = t.get(f"CustomSDPA({layout})")
        std = t.get(f"StandardSDPA({layout})")
        onnx = t.get(f"OnnxGQA({layout})")
        combined = t.get(f"CombinedUpdateSDPA({layout})")
        update = t.get(f"UpdateCache({layout})")

        if not any([custom, std, onnx, combined, update]):
            continue

        rows.append([
            pos,
            format_time_us(custom) if custom else "-",
            format_time_us(std) if std else "-",
            format_time_us(onnx) if onnx else "-",
            format_time_us(combined) if combined else "-",
            format_time_us(update) if update else "-",
        ])

    print_table(headers, rows)


# ---------------------------------------------------------------------------
# Validation output parsing
# ---------------------------------------------------------------------------


def print_validation_output(stdout: str) -> None:
    """Extract and display validation test results from console output."""
    lines = [
        l.strip()
        for l in stdout.splitlines()
        if any(kw in l.upper() for kw in ["PASS", "FAIL", "VALIDATION", "TEST"])
        and "benchmark" not in l.lower()
    ]
    if lines:
        print("\n[Validation Tests]")
        for l in lines:
            print(f"  {l}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SDPA benchmarks on an Android device"
    )
    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument(
        "--device_serial",
        help="ADB device serial (from 'adb devices')",
    )
    device_group.add_argument(
        "--device_model",
        choices=list(ANDROID_PHONE_TYPE.keys()),
        help="Lease a device by model name via Enkaku (e.g. s25)",
    )
    parser.add_argument(
        "--benchmark",
        choices=["bench_sdpa", "bench_transposed_cache", "both"],
        default="both",
        help="Which benchmark to run (default: both)",
    )
    parser.add_argument(
        "--benchmark_filter",
        help="Google Benchmark filter regex (e.g. '.*1/32/8/128.*')",
    )
    parser.add_argument(
        "--benchmark_repetitions",
        type=int,
        help="Number of repetitions for statistical confidence",
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip build step (requires --binary_path)",
    )
    parser.add_argument(
        "--binary_path",
        help="Path to a pre-built binary (use with --skip_build)",
    )
    parser.add_argument(
        "--no_strip",
        action="store_true",
        help="Skip NDK binary stripping",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        help="Set OMP_NUM_THREADS on device before running",
    )
    args = parser.parse_args()

    benchmarks_to_run = (
        ["bench_sdpa", "bench_transposed_cache"]
        if args.benchmark == "both"
        else [args.benchmark]
    )

    # Resolve device: either use provided serial or lease one
    leased_serial = None
    if args.device_model:
        leased_serial = lease_device(args.device_model)
        device_serial = leased_serial
    else:
        device_serial = args.device_serial
        ensure_adb()

    fbsource_root = None
    if not args.skip_build:
        fbsource_root = find_fbsource_root()
        print(f"fbsource root: {fbsource_root}")

    try:
        with tempfile.TemporaryDirectory(prefix="bench_sdpa_") as tmpdir:
            for bench_name in benchmarks_to_run:
                print(f"\n{'#' * 80}")
                print(f"  Benchmark: {bench_name}")
                print(f"{'#' * 80}")

                # 1. Build (or reuse pre-built binary)
                if args.skip_build:
                    if not args.binary_path:
                        sys.exit("Error: --skip_build requires --binary_path")
                    binary_path = args.binary_path
                else:
                    binary_path = build_benchmark(
                        bench_name, fbsource_root, args.no_strip
                    )

                # 2. Deploy
                deploy_to_device(device_serial, binary_path)

                # 3. Run
                stdout, stderr = run_on_device(
                    device_serial,
                    args.benchmark_filter,
                    args.benchmark_repetitions,
                    args.cpu_threads,
                )
                print_validation_output(stdout)

                # 4. Pull JSON results
                results_path = pull_results(device_serial, tmpdir)

                # 5. Parse and summarize
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"\nError reading results: {e}")
                    print("Raw device stdout:\n" + stdout[:2000])
                    continue

                bm_list = data.get("benchmarks", [])
                if not bm_list:
                    print("\nNo benchmark entries in JSON output.")
                    print("Raw device stdout:\n" + stdout[:2000])
                    continue

                print(f"\n  Parsed {len(bm_list)} benchmark entries")

                if bench_name == "bench_sdpa":
                    summarize_sdpa_results(bm_list)
                else:
                    summarize_transposed_cache_results(bm_list)
    finally:
        # Always release the leased device
        if leased_serial:
            print(f"\n[Release] Releasing device {leased_serial}...")
            release_device(leased_serial)

    print("\nDone.")


if __name__ == "__main__":
    main()
