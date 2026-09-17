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
    fvp = bin_dir / "FVP_Corstone_SSE-300_Ethos-U55"
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
) -> subprocess.CompletedProcess[str]:
    """Run the test copy of run_fvp.sh with configured fake FVP output."""
    script, elf, env = test_environment
    env.update(FVP_TEST_OUTPUT=output, FVP_TEST_STATUS=str(status))
    return subprocess.run(  # nosec B603 - fixed local script and arguments
        [
            "/bin/bash",
            str(script),
            f"--elf={elf}",
            "--target=ethos-u55-128",
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
