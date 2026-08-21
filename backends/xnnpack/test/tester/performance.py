# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import math
import os
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from torch.utils._pytree import tree_flatten


logger = logging.getLogger(__name__)

PERF_ENABLE_ENV = "EXECUTORCH_XNNPACK_PYTEST_PERF"
PERF_UPDATE_ENV = "EXECUTORCH_XNNPACK_PYTEST_PERF_UPDATE"
PERF_RUNS_ENV = "EXECUTORCH_XNNPACK_PYTEST_PERF_RUNS"
PERF_WARMUP_ENV = "EXECUTORCH_XNNPACK_PYTEST_PERF_WARMUP_RUNS"
PERF_THRESHOLD_ENV = "EXECUTORCH_XNNPACK_PYTEST_PERF_THRESHOLD_PCT"

_DEFAULT_RUNS = 10
_DEFAULT_WARMUP_RUNS = 2
_DEFAULT_THRESHOLD_PCT = 10.0
_FORCED_THREAD_COUNT = 1
_LINUX_ML_FEATURES = (
    "fp",
    "asimd",
    "asimddp",
    "asimdhp",
    "avx",
    "avx2",
    "avx512_bf16",
    "avx512_vnni",
    "avx512bw",
    "avx512f",
    "bf16",
    "fphp",
    "i8mm",
    "sme",
    "sme2",
    "sse",
    "sse2",
    "sse4_1",
    "sse4_2",
    "sve",
    "sve2",
)
_MACOS_ML_FEATURES = (
    ("hw.optional.arm.FEAT_BF16", "bf16"),
    ("hw.optional.arm.FEAT_DotProd", "asimddp"),
    ("hw.optional.arm.FEAT_FP16", "fphp"),
    ("hw.optional.arm.FEAT_I8MM", "i8mm"),
    ("hw.optional.arm.FEAT_SME", "sme"),
    ("hw.optional.arm.FEAT_SME2", "sme2"),
)
_MACOS_ML_FEATURE_KEYS = tuple(key for key, _ in _MACOS_ML_FEATURES)


def maybe_run_performance_test(
    *,
    serialized_buffer: bytes,
    inputs: Tuple[Any, ...],
    results_path: Optional[str],
) -> None:
    """Run the optional XNNPACK pytest performance stage.

    The stage is skipped unless ``EXECUTORCH_XNNPACK_PYTEST_PERF`` is set.
    In update mode it records a result entry and Markdown summary; otherwise
    it compares the current median latency against the recorded result.
    """
    if not _env_flag(PERF_ENABLE_ENV):
        return
    if not isinstance(serialized_buffer, bytes):
        raise RuntimeError("XNNPACK pytest perf requires a serialized PTE buffer.")

    test_id = _current_pytest_test_id()
    timed_runs = _env_int(PERF_RUNS_ENV, _DEFAULT_RUNS)
    warmup_runs = _env_int(PERF_WARMUP_ENV, _DEFAULT_WARMUP_RUNS)
    threshold_pct = _env_float(PERF_THRESHOLD_ENV, _DEFAULT_THRESHOLD_PCT)

    record = _measure_latency(
        serialized_buffer=serialized_buffer,
        inputs=inputs,
        test_id=test_id,
        timed_runs=timed_runs,
        warmup_runs=warmup_runs,
        threshold_pct=threshold_pct,
    )

    path = Path(results_path) if results_path else _default_results_path(test_id)
    results = _load_results(path)
    runtime_key = record["timing_runtime"]["runtime_key"]
    test_entries = results.setdefault("entries", {}).setdefault(test_id, {})
    recorded_entry = test_entries.get(runtime_key)

    if _env_flag(PERF_UPDATE_ENV):
        test_entries[runtime_key] = record
        _write_results(path, results)
        _write_results_summary(path.with_suffix(".md"), results, path.name)
        logger.info("Updated XNNPACK pytest perf results for %s", runtime_key)
        return

    if recorded_entry is None:
        logger.warning(
            "No XNNPACK pytest perf result for %s / %s. Set %s=1 to record one.",
            test_id,
            runtime_key,
            PERF_UPDATE_ENV,
        )
        return

    _assert_within_threshold(
        test_id=test_id,
        runtime_key=runtime_key,
        current=record,
        recorded=recorded_entry,
        threshold_pct=threshold_pct,
    )


def _measure_latency(
    *,
    serialized_buffer: bytes,
    inputs: Tuple[Any, ...],
    test_id: str,
    timed_runs: int,
    warmup_runs: int,
    threshold_pct: float,
) -> Dict[str, Any]:
    """Measure in-process pybinding runtime latency for a serialized PTE."""
    from executorch.extension.pybindings import _portable_lib as portable_native
    from executorch.extension.pybindings.portable_lib import (
        _load_for_executorch_from_buffer,
        _threadpool_get_thread_count,
        _unsafe_reset_threadpool,
        Verification,
    )

    native_path = Path(portable_native.__file__).resolve()
    host = _host_identity()
    timing_runtime = _timing_runtime_identity(native_path, host)
    inputs_flattened, _ = tree_flatten(inputs)

    original_thread_count = _threadpool_get_thread_count()
    _unsafe_reset_threadpool(_FORCED_THREAD_COUNT)
    observed_thread_count = _threadpool_get_thread_count()

    try:
        module = _load_for_executorch_from_buffer(
            serialized_buffer, program_verification=Verification.Minimal
        )
        for _ in range(warmup_runs):
            module.run_method("forward", tuple(inputs_flattened))

        elapsed_ms = []
        for _ in range(timed_runs):
            start_ns = time.perf_counter_ns()
            module.run_method("forward", tuple(inputs_flattened))
            elapsed_ms.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
    finally:
        _unsafe_reset_threadpool(original_thread_count)

    mean_ms = statistics.fmean(elapsed_ms)
    stdev_ms = statistics.pstdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0

    return {
        "test_id": test_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "timing_runtime": timing_runtime,
        "kernel_path": _kernel_path_identity(host, timing_runtime),
        "metrics": {
            "load_mode": "buffer",
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "thread_count_requested": _FORCED_THREAD_COUNT,
            "thread_count_observed": observed_thread_count,
            "median_ms": statistics.median(elapsed_ms),
            "mean_ms": mean_ms,
            "min_ms": min(elapsed_ms),
            "max_ms": max(elapsed_ms),
            "p90_ms": _percentile(elapsed_ms, 0.90),
            "cv_pct": 0.0 if mean_ms == 0.0 else stdev_ms / mean_ms * 100.0,
            "samples_ms": elapsed_ms,
        },
        "threshold_pct": threshold_pct,
    }


def _assert_within_threshold(
    *,
    test_id: str,
    runtime_key: str,
    current: Dict[str, Any],
    recorded: Dict[str, Any],
    threshold_pct: float,
) -> None:
    """Assert the current median latency is not a thresholded regression."""
    current_median = current["metrics"]["median_ms"]
    recorded_median = recorded["metrics"]["median_ms"]
    if recorded_median == 0.0:
        logger.warning(
            "Skipping XNNPACK pytest perf comparison against zero recorded median"
        )
        return

    delta_pct = (current_median - recorded_median) / recorded_median * 100.0
    if delta_pct <= threshold_pct:
        logger.info(
            "XNNPACK pytest perf %s / %s: %.3f ms vs %.3f ms recorded (%+.2f%%)",
            test_id,
            runtime_key,
            current_median,
            recorded_median,
            delta_pct,
        )
        return

    raise AssertionError(
        f"XNNPACK pytest perf regression for {test_id} / {runtime_key} exceeded "
        f"{threshold_pct:.2f}%: current median {current_median:.3f} ms, "
        f"recorded median {recorded_median:.3f} ms, delta {delta_pct:+.2f}%."
    )


def _default_results_path(test_id: str) -> Path:
    """Return the default JSON result path for a derived XNNPACK test id."""
    parts = test_id.split(".")
    if len(parts) < 3 or parts[0] != "xnnpack":
        raise RuntimeError(
            "XNNPACK pytest perf test ids must look like "
            "'xnnpack.<suite>.<name>.<variant>'."
        )

    suite = parts[1]
    name = parts[2]
    return (
        Path(__file__).resolve().parents[1]
        / suite
        / (f"{name}_pytest_perf_results.json")
    )


def _current_pytest_test_id() -> str:
    """Derive a stable perf record id from ``PYTEST_CURRENT_TEST``."""
    current_test = os.environ.get("PYTEST_CURRENT_TEST", "").strip()
    current_test = re.sub(r"\s+\([^)]*\)$", "", current_test)
    if not current_test:
        raise RuntimeError(
            "XNNPACK pytest perf requires PYTEST_CURRENT_TEST to derive the "
            "perf record id."
        )

    test_path, *node_parts = current_test.split("::")
    if not node_parts:
        raise RuntimeError(
            f"Could not derive XNNPACK pytest perf id from {current_test}"
        )

    namespace = _pytest_module_namespace(Path(test_path))
    test_name = node_parts[-1]
    return ".".join(_slug(part) for part in ("xnnpack", *namespace, test_name))


def _pytest_module_namespace(path: Path) -> list[str]:
    """Map a pytest file path under XNNPACK tests to an id namespace."""
    parts = path.with_suffix("").parts
    marker = ("backends", "xnnpack", "test")
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return list(parts[index + len(marker) :])
    return [path.stem]


def _timing_runtime_identity(native_path: Path, host: Dict[str, Any]) -> Dict[str, Any]:
    """Build the runtime identity used to separate recorded perf results."""
    return {
        "kind": "pybinding",
        "runtime_key": _runtime_key(native_path, host),
        "native_path": _display_path(native_path),
        "native_mtime_utc": datetime.fromtimestamp(
            native_path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "pte_load_mode": "buffer",
        "thread_count": _FORCED_THREAD_COUNT,
        **_runtime_symbol_hints(native_path),
    }


def _runtime_symbol_hints(native_path: Path) -> Dict[str, bool]:
    """Inspect linked native symbols that hint at available kernel families."""
    return {
        "xnnpack_symbols_present": _binary_contains(
            native_path, (b"XnnpackBackend", b"xnn_")
        ),
        "neon_symbols_present": _binary_contains(native_path, (b"neon", b"NEON")),
        "sme_symbols_present": _binary_contains(
            native_path, (b"neonsme", b"_sme", b"-sme")
        ),
        "sme2_symbols_present": _binary_contains(
            native_path, (b"neonsme2", b"_sme2", b"-sme2")
        ),
        "kleidiai_symbols_present": _binary_contains(native_path, (b"kai_", b"Kleidi")),
        "neonsme2_symbols_present": _binary_contains(native_path, (b"neonsme2",)),
        "sse_symbols_present": _binary_contains(
            native_path, (b"_sse", b"-sse", b"sse2", b"sse4", b"SSE2", b"SSE4")
        ),
        "avx_symbols_present": _binary_contains(
            native_path, (b"_avx", b"-avx", b"AVX")
        ),
        "avx2_symbols_present": _binary_contains(
            native_path, (b"_avx2", b"-avx2", b"avx2", b"AVX2")
        ),
        "avx512_symbols_present": _binary_contains(
            native_path, (b"avx512", b"avx-512", b"AVX512", b"AVX-512")
        ),
    }


def _kernel_path_identity(
    host: Dict[str, Any], timing_runtime: Dict[str, Any]
) -> Dict[str, Any]:
    """Describe the expected XNNPACK hardware path for this host/runtime."""
    symbol_hints = {
        key: value
        for key, value in timing_runtime.items()
        if key.endswith("_symbols_present")
    }
    backend = "xnnpack" if symbol_hints["xnnpack_symbols_present"] else "unknown"
    expected_hardware_path = _expected_hardware_path(host, symbol_hints)
    return {
        "backend": backend,
        "expected_hardware_path": expected_hardware_path,
        "confidence": "expected_not_proven",
        "basis": [
            "host.ml_features",
            "runtime.linked_symbols",
        ],
        "host_ml_features": host["ml_features"],
        "runtime_symbol_hints": symbol_hints,
    }


def _expected_hardware_path(host: Dict[str, Any], symbol_hints: Dict[str, bool]) -> str:
    """Deduce the expected hardware path from host features and symbols."""
    features = set(host["ml_features"])
    arm_path = _expected_arm_hardware_path(features, host, symbol_hints)
    if arm_path != "unknown":
        return arm_path

    return _expected_x86_hardware_path(features, symbol_hints)


def _expected_arm_hardware_path(
    features: set[str],
    host: Dict[str, Any],
    symbol_hints: Dict[str, bool],
) -> str:
    """Deduce the expected Arm path such as NEON, SME, or SME2."""
    if _host_has_sme2(features, host):
        return _expected_sme2_path(symbol_hints)

    if _host_has_sme(features):
        return _expected_sme_path(symbol_hints)

    if "asimd" in features and symbol_hints["neon_symbols_present"]:
        return "kleidiai+neon" if symbol_hints["kleidiai_symbols_present"] else "neon"

    return "unknown"


def _host_has_sme2(features: set[str], host: Dict[str, Any]) -> bool:
    """Return whether the host feature set reports SME2 support."""
    return (
        "sme2" in features
        or host["sme2_available"]
        or "hw.optional.arm.FEAT_SME2" in features
    )


def _host_has_sme(features: set[str]) -> bool:
    """Return whether the host feature set reports SME support."""
    return "sme" in features or "hw.optional.arm.FEAT_SME" in features


def _expected_sme2_path(symbol_hints: Dict[str, bool]) -> str:
    """Choose the SME2 path name from linked runtime symbol hints."""
    if symbol_hints["kleidiai_symbols_present"]:
        return "kleidiai+sme2"
    if symbol_hints["sme2_symbols_present"] or symbol_hints["neonsme2_symbols_present"]:
        return "sme2"
    return "unknown"


def _expected_sme_path(symbol_hints: Dict[str, bool]) -> str:
    """Choose the SME path name from linked runtime symbol hints."""
    if symbol_hints["kleidiai_symbols_present"]:
        return "kleidiai+sme"
    if symbol_hints["sme_symbols_present"]:
        return "sme"
    return "unknown"


def _expected_x86_hardware_path(
    features: set[str], symbol_hints: Dict[str, bool]
) -> str:
    """Deduce the expected x86 SIMD path from features and symbols."""
    candidates = (
        ("avx512f", "avx512_symbols_present", "avx512"),
        ("avx2", "avx2_symbols_present", "avx2"),
        ("avx", "avx_symbols_present", "avx"),
    )
    for feature, symbol_key, hardware_path in candidates:
        if feature in features and symbol_hints[symbol_key]:
            return hardware_path

    if {"sse", "sse2", "sse4_1", "sse4_2"} & features and symbol_hints[
        "sse_symbols_present"
    ]:
        return "sse"

    return "unknown"


def _runtime_key(native_path: Path, host: Dict[str, Any]) -> str:
    """Build a key that separates host, runtime, load mode, and threads."""
    stem = f"{host['system']}-{host['machine']}-{host['cpu_id']}"
    sme2 = "sme2" if host["sme2_available"] else "nosme2"
    runtime = f"pybinding-buffer-threads{_FORCED_THREAD_COUNT}"
    return _slug(f"{stem}-{sme2}-{runtime}-{native_path.name}")


def _host_identity() -> Dict[str, Any]:
    """Collect host details that affect perf comparability and path choice."""
    cpu_info = _read_linux_cpuinfo()
    mac_brand = _sysctl("machdep.cpu.brand_string")
    mac_model = _sysctl("hw.model")
    mac_features = {key: _sysctl(key) for key in _MACOS_ML_FEATURE_KEYS}
    mac_features = {key: value for key, value in mac_features.items() if value}

    linux_features = (
        cpu_info.get("Features")
        or cpu_info.get("flags")
        or cpu_info.get("Features".lower())
        or ""
    ).split()
    system = platform.system()
    machine = platform.machine()
    ml_features = _ml_features(linux_features, mac_features, system, machine)
    cpu_model = (
        cpu_info.get("model name")
        or cpu_info.get("Hardware")
        or cpu_info.get("Processor")
        or mac_brand
        or mac_model
        or platform.processor()
        or "unknown-cpu"
    )
    cpu_id = _slug(
        " ".join(
            filter(
                None,
                (
                    cpu_model,
                    mac_model,
                    cpu_info.get("CPU implementer"),
                    cpu_info.get("CPU part"),
                ),
            )
        )
    )

    return {
        "system": system,
        "release": platform.release(),
        "machine": machine,
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "cpu_id": cpu_id,
        "linux_features": linux_features,
        "ml_features": ml_features,
        "macos_model": mac_model,
        "macos_features": mac_features,
        "sme2_available": "sme2" in ml_features,
    }


def _read_linux_cpuinfo() -> Dict[str, str]:
    """Read the first processor entry from Linux ``/proc/cpuinfo``."""
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return {}

    result = {}
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            if result:
                break
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _ml_features(
    linux_features: list[str],
    mac_features: Dict[str, str],
    system: str,
    machine: str,
) -> list[str]:
    """Normalize Linux and macOS CPU features into one ML feature list."""
    relevant = [feature for feature in _LINUX_ML_FEATURES if feature in linux_features]
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        relevant.extend(("fp", "asimd"))
    relevant.extend(
        feature
        for key, feature in _MACOS_ML_FEATURES
        if mac_features.get(key) in ("1", "true")
    )
    return list(dict.fromkeys(relevant))


def _display_path(path: Path) -> str:
    """Display paths relative to the ExecuTorch repository when possible."""
    try:
        return str(path.relative_to(Path(__file__).resolve().parents[4]))
    except ValueError:
        return str(path)


def _sysctl(key: str) -> Optional[str]:
    """Read a macOS sysctl value, returning ``None`` when unavailable."""
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _binary_contains(path: Path, needles: Tuple[bytes, ...]) -> bool:
    """Return whether a binary contains any of the given byte sequences."""
    with path.open("rb") as handle:
        overlap = b""
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return False
            haystack = overlap + chunk
            if any(needle in haystack for needle in needles):
                return True
            overlap = haystack[-64:]


def _load_results(path: Path) -> Dict[str, Any]:
    """Load a perf results JSON file, or return an empty schema."""
    if not path.exists():
        return {"version": 1, "entries": {}}
    with path.open("r") as handle:
        return json.load(handle)


def _write_results(path: Path, results: Dict[str, Any]) -> None:
    """Write the complete machine-readable perf results JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    results["version"] = 1
    with path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_results_summary(
    path: Path, results: Dict[str, Any], source_name: str
) -> None:
    """Write a compact Markdown summary of the perf results JSON."""
    headers = [
        "Test",
        "Host",
        "Processor",
        "ML features",
        "Expected HW path",
        "Threads",
        "Mean ms",
        "Median ms",
        "CV %",
        "Recorded",
    ]
    rows = _results_summary_rows(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# XNNPACK Pytest Perf Results\n\n")
        handle.write(f"Generated from `{source_name}`.\n\n")
        _write_markdown_table(handle, headers, rows, {5, 6, 7, 8})


def _write_markdown_table(
    handle: Any,
    headers: list[str],
    rows: list[list[str]],
    right_aligned_columns: set[int],
) -> None:
    """Write a GitHub-flavored Markdown table with raw column alignment."""
    widths = [
        max(len(row[index]) for row in [headers] + rows)
        for index in range(len(headers))
    ]

    def _format_row(row: list[str]) -> str:
        cells = []
        for index, cell in enumerate(row):
            if index in right_aligned_columns:
                cells.append(cell.rjust(widths[index]))
            else:
                cells.append(cell.ljust(widths[index]))
        return "| " + " | ".join(cells) + " |\n"

    separators = []
    for index, width in enumerate(widths):
        if index in right_aligned_columns:
            separators.append("-" * (width - 1) + ":")
        else:
            separators.append("-" * width)

    handle.write(_format_row(headers))
    handle.write("| " + " | ".join(separators) + " |\n")
    for row in rows:
        handle.write(_format_row(row))


def _results_summary_rows(results: Dict[str, Any]) -> list[list[str]]:
    """Build Markdown table rows from the nested perf results schema."""
    rows = []
    entries = results.get("entries", {})
    for test_id in sorted(entries):
        for runtime_key in sorted(entries[test_id]):
            record = entries[test_id][runtime_key]
            host = record.get("host", {})
            kernel_path = record.get("kernel_path", {})
            metrics = record.get("metrics", {})
            rows.append(
                [
                    _markdown_table_cell(test_id),
                    _markdown_table_cell(_host_summary(host)),
                    _markdown_table_cell(_processor_summary(host)),
                    _markdown_table_cell(", ".join(host.get("ml_features", []))),
                    _markdown_table_cell(
                        kernel_path.get("expected_hardware_path", "unknown")
                    ),
                    str(metrics.get("thread_count_observed", "unknown")),
                    _format_metric(metrics.get("mean_ms")),
                    _format_metric(metrics.get("median_ms")),
                    _format_metric(metrics.get("cv_pct")),
                    _markdown_table_cell(_recorded_date(record)),
                ]
            )
    return rows


def _host_summary(host: Dict[str, Any]) -> str:
    """Return a compact host label for the Markdown summary."""
    parts = [
        host.get("system"),
        host.get("machine"),
        host.get("macos_model"),
    ]
    return " ".join(part for part in parts if part) or "unknown"


def _processor_summary(host: Dict[str, Any]) -> str:
    """Return the most readable processor label available in host details."""
    return (
        host.get("cpu_model")
        or host.get("processor")
        or host.get("cpu_id")
        or "unknown"
    )


def _recorded_date(record: Dict[str, Any]) -> str:
    """Format a recorded UTC timestamp for the Markdown summary."""
    recorded_at = record.get("recorded_at_utc", "")
    match = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})", recorded_at)
    return f"{match.group(1)} {match.group(2)} UTC" if match else recorded_at


def _format_metric(value: Any) -> str:
    """Format numeric summary metrics to three decimal places."""
    return f"{value:.3f}" if isinstance(value, (float, int)) else "unknown"


def _markdown_table_cell(value: Any) -> str:
    """Escape a value for use inside a Markdown table cell."""
    return str(value).replace("|", "\\|") if value is not None else ""


def _percentile(values: list[float], fraction: float) -> float:
    """Compute an interpolated percentile for a non-empty list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _env_flag(name: str) -> bool:
    """Return whether an environment variable is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable with a default."""
    value = os.environ.get(name)
    if value is None:
        return default
    return max(1, int(value))


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a default."""
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _slug(value: str) -> str:
    """Convert a string to a stable lowercase identifier component."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower()).strip("-")
