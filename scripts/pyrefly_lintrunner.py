#!/usr/bin/env python3
"""Lintrunner adapter for pyrefly.

Runs pyrefly in project mode (reads scope from pyproject.toml) and converts
its JSON output to the NDJSON format lintrunner expects. The file list passed
by lintrunner is intentionally ignored — pyrefly's project-excludes handle
scope, and the inline # pyrefly: ignore comments handle pre-existing errors.

Prerequisites: pyrefly must be installed in the active Python environment.
Run `lintrunner init` to install it via requirements-lintrunner.txt.
"""

import json
import subprocess
import sys


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pyrefly", "check", "--output-format", "json"],
        capture_output=True,
        text=True,
    )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return

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
