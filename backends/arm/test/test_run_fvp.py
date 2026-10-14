# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess  # nosec B404 - invokes a copied repository script
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_FVP = _REPO_ROOT / "backends" / "arm" / "scripts" / "run_fvp.sh"


@pytest.fixture
def fvp_test_environment(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """Create a run_fvp.sh environment with a controllable fake FVP."""
    script = tmp_path / "backends" / "arm" / "scripts" / "run_fvp.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(_RUN_FVP, script)

    setup_path = tmp_path / "examples" / "arm" / "arm-scratch" / "setup_path.sh"
    setup_path.parent.mkdir(parents=True)
    setup_path.write_text("")

    elf = tmp_path / "executor_runner"
    elf.write_bytes(b"")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in (
        "FVP_Corstone_SSE-300_Ethos-U55",
        "FVP_Corstone_SSE-300_Ethos-U65",
        "FVP_Corstone_SSE-320",
    ):
        fvp = bin_dir / name
        fvp.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"${FVP_TEST_OUTPUT-}\"\n"
            'exit "${FVP_TEST_STATUS-0}"\n'
        )
        fvp.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return script, elf, env


def run_fvp(
    test_environment: tuple[Path, Path, dict[str, str]],
    output: str,
    status: int = 0,
    *,
    target: str = "ethos-u55-128",
    args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run the test copy of run_fvp.sh with configured fake FVP output."""
    script, elf, env = test_environment
    env.update(FVP_TEST_OUTPUT=output, FVP_TEST_STATUS=str(status))
    return subprocess.run(  # nosec B603 - fixed local script and arguments
        [
            "/bin/bash",
            str(script),
            f"--elf={elf}",
            f"--target={target}",
            *args,
        ],
        capture_output=True,
        env=env,
        text=True,
    )


def test_fvp_launch_failure_is_reported(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    """A nonzero FVP exit must propagate through the logging pipeline."""
    result = run_fvp(
        fvp_test_environment,
        "docker: command not found",
        status=127,
    )

    assert result.returncode == 127
    assert "Simulation complete, 127" in result.stdout
    assert "FVP launch or execution failed" in result.stdout
    assert "No problems found!" not in result.stdout


def test_fvp_output_requires_success_marker(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    """A zero exit without evidence of execution must fail."""
    result = run_fvp(fvp_test_environment, "Simulator started")

    assert result.returncode == 1
    assert "Simulation complete, 0" in result.stdout
    assert "No successful model execution found in log" in result.stdout
    assert "No problems found!" not in result.stdout


def test_fvp_generic_program_exit_is_not_success(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    """Generic program termination must not prove model execution."""
    result = run_fvp(fvp_test_environment, "Program complete, exiting.")

    assert result.returncode == 1
    assert "No successful model execution found in log" in result.stdout
    assert "No problems found!" not in result.stdout


def test_fvp_success_is_reported(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    """A zero exit with a model success marker must pass."""
    result = run_fvp(fvp_test_environment, "Model run: 1")

    assert result.returncode == 0
    assert "Simulation complete, 0" in result.stdout
    assert "No problems found!" in result.stdout


def test_fvp_classic_ml_success_is_reported(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    """The Classic ML runner's inference marker must pass."""
    result = run_fvp(fvp_test_environment, "Inference complete: 1 output(s)")

    assert result.returncode == 0
    assert "Simulation complete, 0" in result.stdout
    assert "No problems found!" in result.stdout


_U55_REDUCE_SUM_DIAGNOSTIC = (
    "[critical][u_dut.u_npu.u_npu_cc.u_npu_cc_tsu.proc][10730081 ns] "
    "Op=4: Invalid data type, ifm=5 ofm=1"
)


def test_fvp_known_u55_diagnostic_and_banner_are_accepted(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    result = run_fvp(
        fvp_test_environment,
        "Ethos-U rev 136b7d75 --- Apr  8 2024 21:45:17\n"
        f"{_U55_REDUCE_SUM_DIAGNOSTIC}\nModel run: 1",
    )

    assert result.returncode == 0
    assert (
        _U55_REDUCE_SUM_DIAGNOSTIC.replace("[critical]", "[warning]") in result.stdout
    )


def test_fvp_known_diagnostic_does_not_hide_other_errors(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    result = run_fvp(
        fvp_test_environment,
        f"{_U55_REDUCE_SUM_DIAGNOSTIC}\nE [executorch] Incorrect output\nModel run: 1",
    )

    assert result.returncode == 1


_MONITOR_PREFIX = "I [executorch:arm_perf_monitor.cpp:300 StopMeasurements()] "
_CPU_REPORT = (
    "Profiler report, CPU cycles per operator:",
    "ethos-u : cycle_cnt : 42 cycles (42.00 per inference)",
    "Operator(s) total: 42 CPU cycles (42.00 per inference)",
    "Inference runtime: 100 CPU cycles total (100.00 per inference)",
    "Inference CPU ratio: 100.00 %",
    "Inference NPU ratio: 0.00 %",
    "cpu_wait_for_npu_cntr : 0 CPU cycles (0.00 per inference)",
    "NOTE: CPU/NPU ratios use CPU-clock time inside NPU callbacks, not NPU utilization",
    "CPU timing and CPU/NPU ratios: N/A (CPU PMU is not present)",
)
_NPU_REPORT = (
    "Ethos-U PMU report:",
    "ethosu_pmu_cycle_cntr : 0 (0.00 per inference)",
    "ethosu_pmu_cntr4 : 123 (123.00 per inference)",
    "Ethos-U per-delegate PMU report:",
    "Ethos-U delegate 0 PMU cycles: 0",
    "Ethos-U delegate 0 PMU counter 4: 123",
)
_EXECUTION_REPORT = (
    "NPU Inferences : 1",
    "NPU delegations: 1 (1.00 per inference)",
    "Ethos-U delegate 0: 1 backend invocations, 1 NPU invocations",
    "Ethos-U IO copy calls: 2 (2.00 per inference)",
    "Ethos-U IO copy bytes: 1024 bytes (1024.00 per inference)",
    "Ethos-U delegate 0 input copy: 1 calls, 512 bytes",
)
_FAST_BANNER = "FVP CPU timing, CPU/NPU ratios and NPU PMU counters: N/A (--fast)."
_TIMED_BANNER = (
    "FVP CPU timing and CPU/NPU ratios: N/A; NPU PMU counters are simulated."
)


@pytest.mark.parametrize("target", ["ethos-u55-128", "ethos-u65-256", "ethos-u85-256"])
@pytest.mark.parametrize("fast", [False, True])
def test_fvp_reports_only_available_measurements(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
    target: str,
    fast: bool,
) -> None:
    """Generic runners need no CLI mode; timed NPU zeros remain valid values."""
    unrelated = "I [executorch:other.cpp:300 StopMeasurements()] Inference runtime: 100"
    output = "\n".join(
        _MONITOR_PREFIX + message
        for message in (*_CPU_REPORT, *_NPU_REPORT, *_EXECUTION_REPORT)
    )
    result = run_fvp(
        fvp_test_environment,
        f"{output}\n{unrelated}\nModel run: 1",
        target=target,
        args=("--fast",) if fast else (),
    )

    assert result.returncode == 0
    assert result.stdout.count(_FAST_BANNER if fast else _TIMED_BANNER) == 1
    for message in _CPU_REPORT:
        assert _MONITOR_PREFIX + message not in result.stdout
    for message in _NPU_REPORT:
        assert (_MONITOR_PREFIX + message in result.stdout) is not fast
    for message in _EXECUTION_REPORT:
        assert _MONITOR_PREFIX + message in result.stdout
    assert unrelated in result.stdout
    assert "Model run: 1" in result.stdout


def test_cortex_m_reports_its_forced_fast_mode(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
) -> None:
    """The CPU-only path forces --fast even when the caller omits the flag."""
    _, elf, _ = fvp_test_environment
    bundle = elf.parent / "model.bpte"
    bundle.write_bytes(b"")
    result = run_fvp(
        fvp_test_environment,
        f"{_MONITOR_PREFIX}{_NPU_REPORT[1]}\nTest_result: PASS",
        target="cortex-m55",
        args=(f"--bundle={bundle}",),
    )

    assert result.returncode == 0
    assert _FAST_BANNER in result.stdout
    assert _TIMED_BANNER not in result.stdout
    assert _MONITOR_PREFIX + _NPU_REPORT[1] not in result.stdout
    assert "Test_result: PASS" in result.stdout


@pytest.mark.parametrize("level", ["E", "F"])
def test_fvp_measurement_filter_preserves_errors(
    fvp_test_environment: tuple[Path, Path, dict[str, str]],
    level: str,
) -> None:
    """A metric-shaped diagnostic must still fail the run and remain visible."""
    diagnostic = f"{level}{_MONITOR_PREFIX[1:]}{_NPU_REPORT[1]}"
    result = run_fvp(
        fvp_test_environment, f"{diagnostic}\nModel run: 1", args=("--fast",)
    )

    assert result.returncode == 1
    assert diagnostic in result.stdout
