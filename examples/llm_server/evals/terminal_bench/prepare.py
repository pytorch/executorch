# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fetch the pinned task subset without modifying an existing task checkout."""

import fcntl
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from .suite import HARNESS_CACHE, SUITES, TASK_ROOT


def prepare_tasks(destination: Path = TASK_ROOT) -> None:
    dataset = SUITES["dataset"]
    tasks = sorted({task for suite in SUITES["suites"].values() for task in suite})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
                checkout = Path(temp) / "tasks"
                checkout.mkdir()
                commands = [
                    ["init", "--quiet"],
                    ["remote", "add", "origin", dataset["repository"]],
                    ["sparse-checkout", "init", "--cone"],
                    ["sparse-checkout", "set", *tasks],
                    [
                        "fetch",
                        "--depth=1",
                        "origin",
                        dataset["revision"],
                    ],
                    ["checkout", "--detach", "FETCH_HEAD"],
                ]
                for command in commands:
                    subprocess.run(["git", "-C", str(checkout), *command], check=True)
                checkout.rename(destination)
        head = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(destination), "status", "--porcelain"], text=True
        ).strip()
        if head != dataset["revision"] or dirty:
            raise RuntimeError(
                f"Task cache is modified: {destination}. Use a fresh EXECUTORCH_EVAL_CACHE."
            )
        for task in tasks:
            for filename in ("task.toml", "instruction.md"):
                if not (destination / task / filename).is_file():
                    raise RuntimeError(
                        f"Incomplete task cache: {destination / task / filename}"
                    )
    print(f"Pinned tasks ready: {destination}")


def main() -> None:
    prepare_tasks()
    environment = {
        "dataset": SUITES["dataset"],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "docker": json.loads(
            subprocess.check_output(
                ["docker", "version", "--format", "{{json .}}"], text=True
            )
        ),
        "python": subprocess.check_output(
            [str(HARNESS_CACHE / "venv/bin/python"), "--version"], text=True
        ).strip(),
        "packages": subprocess.check_output(
            ["uv", "pip", "freeze", "--python", str(HARNESS_CACHE / "venv/bin/python")],
            text=True,
        ).splitlines(),
        "docker_context": subprocess.check_output(
            ["docker", "context", "show"], text=True
        ).strip(),
        "docker_config": os.environ.get("DOCKER_CONFIG"),
    }
    (HARNESS_CACHE / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
