#!/usr/bin/env python3
"""Lintrunner adapter for pyrefly.

Runs pyrefly in project mode (reads scope from pyproject.toml) and converts
its JSON output to the NDJSON format lintrunner expects. The file list passed
by lintrunner is intentionally ignored — pyrefly's project-excludes handle
scope, and the inline # pyrefly: ignore comments handle pre-existing errors.

pyrefly is told explicitly which site-packages to use (via --site-package-path)
so it resolves imports from the same Python environment that runs this script,
not whatever Python 3.10 interpreter it auto-discovers on the system. This is
critical on machines that have a separate Python 3.10 install (e.g. a venv or
pyenv) that lacks torch and other project dependencies.

Prerequisites: pyrefly and all project dependencies (torch, etc.) must be
installed in the active Python environment. Run `lintrunner init` to install
pyrefly via requirements-lintrunner.txt; install torch separately per
https://pytorch.org/get-started/locally/.
"""

import json
import subprocess
import sys
import sysconfig


def main() -> None:
    site_packages = sysconfig.get_path("purelib")

    cmd = [sys.executable, "-m", "pyrefly", "check", "--output-format", "json"]
    if site_packages:
        cmd += ["--site-package-path", site_packages]

    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        # pyrefly failed to start or produced invalid output (not type errors).
        # Propagate stderr so CI shows the actual failure, then exit non-zero.
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        sys.exit(result.returncode if result.returncode != 0 else 1)

    for error in data.get("errors", []):
        severity = error.get("severity", "error")
        if severity not in ("error", "warning", "advice"):
            severity = "error"
        print(
            json.dumps(
                {
                    "path": error["path"],
                    "line": error["line"],
                    "char": error.get("column", 1),
                    "code": "PYREFLY",
                    "severity": severity,
                    "name": error.get("name", "type-error"),
                    "description": error.get(
                        "concise_description", error.get("description", "")
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
