# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Run Arm public API backward-compatibility scenarios."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from subprocess import run  # nosec B404


REPO_ROOT = Path(__file__).resolve().parents[4]
PYTEST_CONFIG = Path("backends/arm/test/pytest.ini")
SCENARIO_FILES = (
    Path("backends/arm/test/public_api_bc/test_ethosu_flow.py"),
    Path("backends/arm/test/public_api_bc/test_vgf_fp_flow.py"),
    Path("backends/arm/test/public_api_bc/test_vgf_int_flow.py"),
)
# Leave empty until a release contains these scenario files. The release epic
# should update this to the oldest still-supported release ref.
OLDEST_SUPPORTED_REF = ""


def _resolve_git() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Could not find git in PATH")
    return git


GIT = _resolve_git()


def _materialize_file(repo_relative_path: Path, output_root: Path) -> Path:
    destination_path = output_root / repo_relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if OLDEST_SUPPORTED_REF:
        result = run(  # nosec B603
            [
                GIT,
                "show",
                f"{OLDEST_SUPPORTED_REF}:{repo_relative_path.as_posix()}",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        destination_path.write_text(result.stdout, encoding="utf-8")
        return destination_path

    source_path = REPO_ROOT / repo_relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing scenario file: {source_path}")

    shutil.copy2(source_path, destination_path)
    return destination_path


def _run_pytest(entrypoints: list[Path], output_root: Path) -> None:
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    run(  # nosec B603
        [
            sys.executable,
            "-m",
            "pytest",
            "--config-file",
            str(REPO_ROOT / PYTEST_CONFIG),
            *[str(path) for path in entrypoints],
        ],
        cwd=output_root,
        env=env,
        check=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="arm-public-api-bc-",
        ignore_cleanup_errors=True,
    ) as temporary_dir:
        materialized_root = Path(temporary_dir)
        entrypoints = [
            _materialize_file(repo_relative_path, materialized_root)
            for repo_relative_path in SCENARIO_FILES
        ]

        source_name = OLDEST_SUPPORTED_REF or "workspace snapshot"
        print("Materialized Arm public API BC scenarios:")
        print(f"  source: {source_name}")
        print(f"  root:   {materialized_root}")

        _run_pytest(entrypoints, materialized_root)


if __name__ == "__main__":
    main()
