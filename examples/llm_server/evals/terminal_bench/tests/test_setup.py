# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from executorch.examples.llm_server.evals.terminal_bench import prepare

EVALS = Path(__file__).resolve().parents[2]


def test_task_cache_pins_revision_and_preserves_local_edits(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    task = origin / "sample"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("original task")
    subprocess.run(["git", "init", "--quiet", str(origin)], check=True)
    subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(origin),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "task",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(origin), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(
        prepare,
        "SUITES",
        {
            "dataset": {"repository": str(origin), "revision": revision},
            "suites": {"smoke": ["sample"]},
        },
    )
    destination = tmp_path / "cache/tasks"
    prepare.prepare_tasks(destination)
    prepare.prepare_tasks(destination)
    assert (destination / "sample/instruction.md").read_text() == "original task"
    (destination / "sample/instruction.md").write_text("developer edit")
    with pytest.raises(RuntimeError, match="modified"):
        prepare.prepare_tasks(destination)
    assert (destination / "sample/instruction.md").read_text() == "developer edit"


@pytest.mark.parametrize(
    "platform,context,fresh_group,agent_host",
    [
        ("Darwin", "colima", False, "host.lima.internal"),
        ("Darwin", "desktop-linux", False, None),
        ("Linux", "default", False, None),
        ("Linux", "default", True, None),
    ],
)
def test_run_script_uses_checkout_and_preserves_arguments(
    tmp_path, platform, context, fresh_group, agent_host
):
    cache = tmp_path / "cache with spaces"
    binary = cache / "terminal-bench/venv/bin/python"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nprint(json.dumps({{'argv':sys.argv[1:], 'path':os.environ['PYTHONPATH'], 'server':os.environ['EXECUTORCH_EVAL_SERVER_PYTHON'], 'agent_host':os.getenv('EXECUTORCH_EVAL_AGENT_HOST'), 'sg':os.getenv('EVAL_TEST_SG')}}))\n"
    )
    binary.chmod(0o755)
    commands = {
        "uname": f"print({platform!r})",
        "docker": f"print({context!r}) if sys.argv[1:] == ['context', 'show'] else sys.exit(0 if not {fresh_group!r} or os.getenv('EVAL_TEST_SG') else 1)",
        "id": "print('testuser' if sys.argv[1:] == ['-un'] else ('users docker' if len(sys.argv) == 3 or os.getenv('EVAL_TEST_SG') else 'users'))",
        "sg": "assert sys.argv[1:3] == ['docker', '-c']; os.environ['EVAL_TEST_SG']='1'; os.execv('/bin/bash', ['bash', '-c', sys.argv[3]])",
    }
    for name, code in commands.items():
        executable = tmp_path / name
        executable.write_text(f"#!{sys.executable}\nimport os, sys\n{code}\n")
        executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "EXECUTORCH_EVAL_CACHE": str(cache),
        "EXECUTORCH_EVAL_SERVER_PYTHON": sys.executable,
    }
    env.pop("EXECUTORCH_EVAL_AGENT_HOST", None)
    env.pop("EVAL_TEST_SG", None)
    result = subprocess.run(
        [
            "bash",
            str(EVALS / "run.sh"),
            "terminal-bench",
            "--config",
            "model with spaces.toml",
            "--server-arg=--assistant-header=custom",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    value = json.loads(result.stdout)
    assert value["argv"] == [
        "-m",
        "executorch.examples.llm_server.evals.terminal_bench",
        "--config",
        "model with spaces.toml",
        "--server-arg=--assistant-header=custom",
    ]
    assert value["path"].split(os.pathsep)[0] == str(EVALS.parents[2] / "src")
    assert value["server"] == sys.executable
    assert value["agent_host"] == agent_host
    assert bool(value["sg"]) == fresh_group


def test_ci_setup_does_not_try_to_install_docker(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 1\n")
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(EVALS / "setup.sh"), "terminal-bench", "--ci"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "CI requires a running Docker" in result.stderr


@pytest.mark.parametrize("script", ["run.sh", "setup.sh"])
def test_unknown_harness_is_rejected(script):
    result = subprocess.run(
        ["bash", str(EVALS / script), "not-a-harness"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "Unsupported evaluation" in result.stderr
