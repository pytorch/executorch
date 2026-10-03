# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run Terminal-Bench tasks through the ExecuTorch LLM server."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .suite import HARNESS_CACHE, SUITES, SUITES_PATH, TASK_ROOT

HARBOR_VERSION = "0.22.0"
AGENT_VERSION = "2.4.6"
PROBE_IMAGE = "curlimages/curl:8.12.1"
SERVER_MODULE = "executorch.examples.llm_server.python.server"
REPO_ROOT = Path(__file__).resolve().parents[4]
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class Trial:
    key: str
    task: str
    attempt: int
    session_id: str


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, help="TOML configuration; CLI values override it"
    )
    p.add_argument(
        "--task",
        type=Path,
        action="append",
        help="Local Harbor task directory; repeat for a subset",
    )
    p.add_argument("--suite", choices=tuple(SUITES["suites"]))
    p.add_argument("--task-root", type=Path, default=TASK_ROOT)
    p.add_argument(
        "--run-dir", type=Path, help="Existing directory for --resume, or a new run"
    )
    p.add_argument("--output-root", type=Path, default=HARNESS_CACHE / "runs")
    p.add_argument("--executorch-root", type=Path, default=REPO_ROOT)
    p.add_argument(
        "--server-python",
        default=os.environ.get("EXECUTORCH_EVAL_SERVER_PYTHON", sys.executable),
    )
    p.add_argument("--server-module", default=SERVER_MODULE)
    p.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra launcher option as --name=value; repeat for model-specific settings",
    )
    p.add_argument(
        "--session-affinity",
        action="store_true",
        help="Reuse a named session across turns; requires a worker with named-session support",
    )
    p.add_argument("--worker-bin", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--tokenizer-path", type=Path, required=True)
    p.add_argument(
        "--hf-tokenizer",
        required=True,
        help="Prefer a pinned local tokenizer directory",
    )
    p.add_argument("--model-id", default="qwen3")
    p.add_argument("--max-context", type=int, required=True)
    p.add_argument("--max-output-tokens", type=int, default=512)
    p.add_argument("--step-limit", type=int, default=100)
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Keep the model template's thinking default",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server listen address, reachable from task containers",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--agent-base-url",
        help="Container-visible URL; defaults to host.docker.internal (host.lima.internal with the Colima launcher)",
    )
    p.add_argument("--harbor-bin", default=str(HARNESS_CACHE / "venv/bin/harbor"))
    p.add_argument("--startup-timeout", type=float, default=180)
    p.add_argument("--resume", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest and exact launch plans only",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check prerequisites without starting a trial",
    )
    return p


def config_arguments(p: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    config, _ = probe.parse_known_args(argv)
    if config.config is None:
        return argv
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    path = config.config.expanduser().resolve()
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    actions = {action.dest: action for action in p._actions}
    overrides = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    result = []
    for key, value in values.items():
        if key not in actions or key in {
            "help",
            "config",
            "resume",
            "check",
            "dry_run",
        }:
            raise ValueError(f"unknown or command-only configuration key: {key}")
        action = actions[key]
        option = action.option_strings[0]
        if (
            option in overrides
            or (key == "task" and "--suite" in overrides)
            or (key == "suite" and "--task" in overrides)
        ):
            continue
        result.extend(config_value_arguments(key, value, action, path.parent))
    return result + argv


def config_value_arguments(key, value, action, base: Path) -> list[str]:
    option = action.option_strings[0]
    paths = {
        "task",
        "task_root",
        "run_dir",
        "output_root",
        "executorch_root",
        "worker_bin",
        "model_path",
        "tokenizer_path",
    }
    if isinstance(action, argparse._StoreTrueAction):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        if value:
            return [option]
        return []
    result = []
    items = value if isinstance(value, list) else [value]
    if not items or any(isinstance(item, (bool, dict, list)) for item in items):
        raise ValueError(f"invalid configuration value: {key}")
    strings = []
    for item in items:
        text = str(item)
        if key in paths or (
            key in {"server_python", "harbor_bin", "hf_tokenizer"}
            and (text.startswith(("~", ".", "/")) or (base / text).exists())
        ):
            text = str((base / Path(text).expanduser()).resolve())
        strings.append(text)
    if isinstance(action, argparse._AppendAction):
        result.extend(f"{option}={item}" for item in strings)
    elif action.nargs == "+":
        result.extend([option, *strings])
    elif len(strings) == 1:
        result.append(f"{option}={strings[0]}")
    else:
        raise ValueError(f"{key} expects a single value")
    return result


def validate(args) -> None:
    if args.resume and args.run_dir is None:
        raise ValueError("--resume requires an explicit --run-dir")
    args.output_root = args.output_root.expanduser().resolve()
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.run_dir = args.output_root / f"{stamp}-{uuid.uuid4().hex[:8]}"
    args.run_dir = args.run_dir.expanduser().resolve()
    args.executorch_root = args.executorch_root.resolve()
    args.worker_bin = args.worker_bin.resolve()
    args.model_path = args.model_path.resolve()
    args.tokenizer_path = args.tokenizer_path.resolve()
    resolve_tasks(args)
    if not 0 < args.max_output_tokens < args.max_context:
        raise ValueError(
            "output reserve must be positive and smaller than --max-context"
        )
    if min(args.step_limit, args.attempts) < 1:
        raise ValueError("step and attempt limits must be positive")
    if not 0 < args.port < 65536 or args.startup_timeout <= 0:
        raise ValueError("invalid port or startup timeout")
    if len(set(args.task)) != len(args.task):
        raise ValueError("duplicate tasks")
    validate_server_options(args)


def resolve_tasks(args) -> None:
    args.task_root = args.task_root.expanduser().resolve()
    if args.task and args.suite:
        raise ValueError("select either --task or --suite")
    if not args.task:
        args.suite = args.suite or "smoke"
        args.task = [args.task_root / name for name in SUITES["suites"][args.suite]]
    args.task = [path.resolve() for path in args.task]
    for task in args.task:
        if (
            not (task / "task.toml").is_file()
            or not (task / "instruction.md").is_file()
        ):
            raise ValueError(
                f"not a Harbor task directory: {task}; run evals/setup.sh terminal-bench or supply --task"
            )


def validate_server_options(args) -> None:
    if args.agent_base_url is None:
        agent_host = os.environ.get(
            "EXECUTORCH_EVAL_AGENT_HOST", "host.docker.internal"
        )
        args.agent_base_url = f"http://{agent_host}:{args.port}/v1"
    url = urllib.parse.urlsplit(args.agent_base_url)
    if (
        url.scheme != "http"
        or not url.hostname
        or url.path.rstrip("/") != "/v1"
        or url.query
        or url.fragment
        or url.username
    ):
        raise ValueError("--agent-base-url must be an HTTP URL ending in /v1")
    if (url.port or 80) != args.port:
        raise ValueError("agent URL port must match the managed server port")
    controlled = {
        "worker_bin",
        "model_path",
        "tokenizer_path",
        "hf_tokenizer",
        "model_id",
        "host",
        "port",
        "max_context",
        "no_think",
        "num_runners",
    }
    for flag in args.server_arg:
        name = flag.split("=", 1)[0].lstrip("-").replace("-", "_")
        if name in controlled or not flag.startswith("--") or "=" not in flag:
            raise ValueError(
                f"server argument is controlled by the benchmark or malformed: {flag}"
            )


def source_revision(root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--"])
        return {
            "commit": commit,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None}


def task_digest(task: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(task.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(task)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_manifest(args) -> dict:
    settings = {
        k: v for k, v in vars(args).items() if k not in {"dry_run", "check", "resume"}
    }
    settings = json.loads(json.dumps(settings, default=str))
    assets = {}
    for key, path in [
        ("worker_bin", args.worker_bin),
        ("model_path", args.model_path),
        ("tokenizer_path", args.tokenizer_path),
    ]:
        assets[key] = (
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": file_digest(path),
            }
            if path.is_file()
            else {"path": str(path), "missing": True}
        )
    setup_record = HARNESS_CACHE / "environment.json"
    uses_setup_environment = (
        Path(args.harbor_bin).resolve() == (HARNESS_CACHE / "venv/bin/harbor").resolve()
    )
    return {
        "schema_version": 1,
        "settings": settings,
        "harbor_version": HARBOR_VERSION,
        "agent_version": AGENT_VERSION,
        "suites_sha256": file_digest(SUITES_PATH),
        "task_source": (
            SUITES["dataset"] if args.task_root == TASK_ROOT and args.suite else None
        ),
        "setup_environment": (
            json.loads(setup_record.read_text())
            if uses_setup_environment and setup_record.is_file()
            else None
        ),
        "executorch": source_revision(args.executorch_root),
        "benchmark": source_revision(Path(__file__).parent),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets": assets,
        "tasks": [
            {"path": str(task), "sha256": task_digest(task)} for task in args.task
        ],
    }


def trials(args) -> list[Trial]:
    result = []
    run_tag = hashlib.sha256(str(args.run_dir).encode()).hexdigest()[:10]
    for index, task in enumerate(args.task):
        name = re.sub(r"[^a-zA-Z0-9_-]", "-", task.name)[:48]
        for attempt in range(1, args.attempts + 1):
            key = f"{index:03d}-{name}-a{attempt}"
            result.append(Trial(key, str(task), attempt, f"tb-{run_tag}-{key}"))
    return result


def agent_config(args, trial: Trial) -> dict:
    return {
        "agent": {"step_limit": args.step_limit},
        "model": {
            "model_kwargs": {
                "temperature": 0,
                "extra_headers": (
                    {"x-session-affinity": trial.session_id}
                    if args.session_affinity
                    else {}
                ),
            }
        },
    }


def server_environment(args) -> dict[str, str]:
    env = os.environ.copy()
    source = args.executorch_root / "src"
    package_parent = (
        source if (source / "executorch").is_dir() else args.executorch_root.parent
    )
    env["PYTHONPATH"] = str(package_parent) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def server_command(args, trial: Trial) -> list[str]:
    command = [
        args.server_python,
        "-m",
        args.server_module,
        "--worker-bin",
        str(args.worker_bin),
        "--model-path",
        str(args.model_path),
        "--tokenizer-path",
        str(args.tokenizer_path),
        "--hf-tokenizer",
        args.hf_tokenizer,
        "--model-id",
        args.model_id,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-context",
        str(args.max_context),
    ]
    if not args.thinking:
        command.append("--no-think")
    command.extend(args.server_arg)
    return command


def harbor_command(args, trial: Trial, execution: Path) -> list[str]:
    no_proxy = ",".join(
        [
            "localhost",
            "127.0.0.1",
            "host.docker.internal",
            "host.lima.internal",
            "172.17.0.1",
            urllib.parse.urlsplit(args.agent_base_url).hostname or "",
        ]
    )
    command = [
        args.harbor_bin,
        "run",
        "--path",
        trial.task,
        "--agent",
        "mini-swe-agent",
        "--model",
        f"openai/{args.model_id}",
        "--ak",
        f"version={AGENT_VERSION}",
        "--ak",
        f"max_tokens={args.max_output_tokens}",
        "--ak",
        f"config_file={execution / 'mini-swe.yaml'}",
        "--ae",
        "MSWEA_API_KEY=dummy-local-key",
        "--ae",
        "OPENAI_API_KEY=dummy-local-key",
        "--ae",
        f"OPENAI_BASE_URL={args.agent_base_url.rstrip('/')}",
        "--ae",
        f"NO_PROXY={no_proxy}",
        "--ae",
        f"no_proxy={no_proxy}",
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--jobs-dir",
        str(execution / "jobs"),
        "--job-name",
        "trial",
    ]
    if needs_host_mapping(args):
        command.extend(["--extra-docker-compose", str(execution / "host-gateway.yaml")])
    return command


def needs_host_mapping(args) -> bool:
    return (
        sys.platform == "linux"
        and urllib.parse.urlsplit(args.agent_base_url).hostname
        == "host.docker.internal"
    )


def check_container_connection(args, execution: Path) -> None:
    name = f"executorch-eval-probe-{uuid.uuid4().hex}"
    command = ["docker", "run", "--rm", "--name", name]
    if needs_host_mapping(args):
        command.extend(["--add-host", "host.docker.internal:host-gateway"])
    command.extend(
        [
            PROBE_IMAGE,
            "--noproxy",
            "*",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "15",
            args.agent_base_url.rstrip("/").removesuffix("/v1") + "/health",
        ]
    )
    json_write(execution / "container-check-command.json", command)
    try:
        with (execution / "container-check.log").open("w") as log:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=log, text=True, timeout=120
            )
            log.write(result.stdout)
            if result.returncode or json.loads(result.stdout).get("status") != "ok":
                raise RuntimeError(
                    f"Task-container route to the server failed; see {execution / 'container-check.log'}"
                )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def check_prerequisites(args) -> list[str]:
    errors = []
    for label, command in [
        ("Harbor", args.harbor_bin),
        ("server Python", args.server_python),
        ("Docker", "docker"),
    ]:
        if shutil.which(command) is None:
            errors.append(f"{label} executable not found: {command}")
    for path in [args.worker_bin, args.model_path, args.tokenizer_path]:
        if not path.is_file():
            errors.append(f"missing runtime asset: {path}")
    if args.worker_bin.is_file() and not os.access(args.worker_bin, os.X_OK):
        errors.append(f"worker is not executable: {args.worker_bin}")
    checks = []
    if shutil.which(args.harbor_bin):
        checks.append(
            ("Harbor version", [args.harbor_bin, "--version"], None, HARBOR_VERSION)
        )
    if shutil.which("docker"):
        checks.append(("Docker Compose", ["docker", "compose", "version"], None, None))
        checks.append(
            (
                "Docker daemon",
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                None,
                None,
            )
        )
    if shutil.which(args.server_python):
        checks.append(
            (
                "ExecuTorch server",
                [args.server_python, "-m", args.server_module, "--help"],
                server_environment(args),
                "--max-context",
            )
        )
    return errors + run_prerequisite_checks(checks)


def run_prerequisite_checks(checks) -> list[str]:
    errors = []
    for label, command, env, expected in checks:
        try:
            completed = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            matches = (
                completed.stdout.strip() == expected
                if label == "Harbor version"
                else expected is None or expected in completed.stdout
            )
            if completed.returncode or not matches:
                errors.append(f"{label} check failed: {completed.stdout[-2000:]}")
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"{label} check failed: {error}")
    return errors


def local_url(args) -> str:
    host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    return f"http://{host}:{args.port}"


def request_json(url: str, method: str = "GET") -> dict:
    with HTTP.open(urllib.request.Request(url, method=method), timeout=5) as response:
        return json.load(response)


def wait_ready(args, process: subprocess.Popen) -> float:
    started = time.monotonic()
    last_error = "no health response"
    while time.monotonic() - started < args.startup_timeout:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with code {process.returncode}; see server.log"
            )
        try:
            if request_json(local_url(args) + "/health").get("status") == "ok":
                return time.monotonic() - started
        except (OSError, ValueError) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(f"server readiness timed out: {last_error}")


def stop_process(process: subprocess.Popen | None, stop_signal=signal.SIGTERM) -> None:
    if process is None:
        return
    # The server can exit before its worker, so also terminate its process group.
    try:
        os.killpg(process.pid, stop_signal)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def parse_service_log(
    path: Path, session_id: str, max_context: int, output_reserve: int
) -> dict:
    turns = []
    integer_fields = {
        "prompt_tokens",
        "reused_prompt_tokens",
        "prefilled_prompt_tokens",
        "completion_tokens",
    }
    for line in path.read_text(errors="replace").splitlines():
        marker = "llm_turn_stats "
        if marker not in line:
            continue
        row = dict(
            part.split("=", 1)
            for part in line.split(marker, 1)[1].split()
            if "=" in part
        )
        if row.get("session_id") != session_id:
            continue
        for key in integer_fields & row.keys():
            row[key] = int(row[key])
        for key in ("prefill_ms", "decode_ms", "total_ms"):
            if key in row:
                row[key] = float(row[key])
        turns.append(row)
    input_tokens = sum(t["prompt_tokens"] for t in turns)
    reused = sum(t["reused_prompt_tokens"] for t in turns)
    return {
        "turns": turns,
        "completed_turns": len(turns),
        "peak_prompt_tokens": max((t["prompt_tokens"] for t in turns), default=0),
        "input_tokens": input_tokens,
        "reused_tokens": reused,
        "prefilled_tokens": sum(t["prefilled_prompt_tokens"] for t in turns),
        "completion_tokens": sum(t["completion_tokens"] for t in turns),
        "prefix_cache_hits": sum(t.get("reason") == "prefix_cache" for t in turns),
        "continuation_hits": sum(t.get("reason") == "exact_prefix" for t in turns),
        "reuse_fraction": reused / input_tokens if input_tokens else None,
        "prefill_ms": sum(t.get("prefill_ms", 0) for t in turns),
        "decode_ms": sum(t.get("decode_ms", 0) for t in turns),
        "generation_ms": sum(t.get("total_ms", 0) for t in turns),
        "context_violations": sum(
            t["prompt_tokens"] + output_reserve > max_context for t in turns
        ),
    }


def read_harbor_result(execution: Path, returncode: int) -> dict:
    files = sorted((execution / "jobs" / "trial").glob("*/result.json"))
    if len(files) != 1:
        return {
            "status": "harness_error",
            "error": f"expected one trial result, found {len(files)}",
            "harbor_returncode": returncode,
        }
    data = json.loads(files[0].read_text())
    reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    exception = data.get("exception_info") or {}
    result = {
        "status": (
            "trial_error"
            if exception or returncode
            else ("scored" if reward is not None else "harness_error")
        ),
        "reward": reward,
        "exception": exception.get("exception_type"),
        "harbor_returncode": returncode,
        "harbor_result": str(files[0]),
        "agent_metrics": data.get("agent_result"),
    }
    trajectory = files[0].parent / "agent" / "mini-swe-agent.trajectory.json"
    if trajectory.exists():
        data = json.loads(trajectory.read_text())
        info = data.get("info", {})
        messages = data.get("messages", [])
        result["trajectory"] = str(trajectory)
        result["tool_observations"] = sum(m.get("role") == "tool" for m in messages)
        result["format_errors"] = sum(
            m.get("extra", {}).get("interrupt_type") == "FormatError" for m in messages
        )
        result["api_calls"] = info.get("model_stats", {}).get("api_calls")
        result["agent_exit"] = info.get("exit_status")
    return result


def run_trial(args, trial: Trial, execution: Path) -> dict:
    execution.mkdir(parents=True)
    json_write(execution / "mini-swe.yaml", agent_config(args, trial))
    if needs_host_mapping(args):
        json_write(
            execution / "host-gateway.yaml",
            {
                "services": {
                    "main": {"extra_hosts": ["host.docker.internal:host-gateway"]}
                }
            },
        )
    commands = {
        "server": server_command(args, trial),
        "harbor": harbor_command(args, trial, execution),
    }
    json_write(execution / "commands.json", commands)
    result = {
        "status": "harness_error",
        "trial": asdict(trial),
        "execution": str(execution),
    }
    server = harbor = None
    ready = False
    started = time.monotonic()
    with (execution / "server.log").open("w") as server_log, (
        execution / "harbor.log"
    ).open("w") as harbor_log:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((args.host, args.port))
            server = subprocess.Popen(
                commands["server"],
                env=server_environment(args),
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result["startup_seconds"] = wait_ready(args, server)
            ready = True
            check_container_connection(args, execution)
            if args.session_affinity:
                request_json(
                    local_url(args) + f"/v1/sessions/{trial.session_id}/reset", "POST"
                )
            trial_started = time.monotonic()
            harbor = subprocess.Popen(
                commands["harbor"],
                stdin=subprocess.DEVNULL,
                stdout=harbor_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            returncode = harbor.wait()
            result["trial_seconds"] = time.monotonic() - trial_started
            result.update(read_harbor_result(execution, returncode))
        except KeyboardInterrupt:
            result["status"] = "interrupted"
            raise
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            stop_process(harbor, signal.SIGINT)
            if ready and args.session_affinity:
                try:
                    result["session_cleanup"] = request_json(
                        local_url(args) + f"/v1/sessions/{trial.session_id}", "DELETE"
                    )
                except (OSError, ValueError) as error:
                    result["cleanup_error"] = str(error)
            stop_process(server)
            server_log.flush()
            result["wall_seconds"] = time.monotonic() - started
            result["service"] = parse_service_log(
                execution / "server.log",
                trial.session_id if args.session_affinity else "<scratch>",
                args.max_context,
                args.max_output_tokens,
            )
            service = result["service"]
            result["validation"] = {
                "context_within_limit": (
                    service["context_violations"] == 0
                    if service["completed_turns"]
                    else None
                ),
                "observed_prefix_reuse": service["prefix_cache_hits"] > 0,
                "cleanup_succeeded": ready
                and (not args.session_affinity or "session_cleanup" in result),
            }
            if result["status"] == "scored" and (
                service["context_violations"]
                or not service["completed_turns"]
                or "cleanup_error" in result
            ):
                result["status"] = "harness_error"
                result["error"] = (
                    "missing turn metrics, context violation, or session cleanup failure; inspect validation and logs"
                )
            json_write(execution / "outcome.json", result)
    return result


def write_results(run_dir: Path, planned: list[Trial]) -> None:
    results = []
    for trial in planned:
        path = run_dir / "trials" / trial.key / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    json_write(run_dir / "results.json", results)
    fields = [
        "task",
        "attempt",
        "status",
        "reward",
        "api_calls",
        "agent_exit",
        "tool_observations",
        "format_errors",
        "trial_seconds",
        "completed_turns",
        "peak_prompt_tokens",
        "reused_tokens",
        "prefilled_tokens",
        "prefix_cache_hits",
        "continuation_hits",
        "prefill_ms",
        "context_violations",
        "execution",
    ]
    with (run_dir / "results.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({**result, **result["trial"], **result["service"]})
    summary = [
        "# Terminal-Bench results",
        "",
        f"Completed {len(results)} of {len(planned)} planned trials. Each row is one task attempt; this subset is not a full Terminal-Bench score.",
        "",
        "| Task | Attempt | Status | Reward | Agent exit | Tool observations | Prefill (ms) | Decode (ms) |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        row = result["trial"]
        service = result["service"]
        cells = [
            Path(row["task"]).name,
            row["attempt"],
            result["status"],
            result.get("reward"),
            result.get("agent_exit"),
            result.get("tool_observations"),
            service["prefill_ms"],
            service["decode_ms"],
        ]
        summary.append(
            "| "
            + " | ".join(
                str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells
            )
            + " |"
        )
    summary.extend(
        [
            "",
            "See results.json for token usage, timings, validation, and evidence paths.",
            "",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(summary))


def run_matrix(args, p: argparse.ArgumentParser) -> int:
    manifest = make_manifest(args)
    manifest_path = args.run_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            p.error(
                "run directory already has a manifest; use --resume or a new directory"
            )
        if json.loads(manifest_path.read_text()) != manifest:
            p.error(
                "configuration, sources, or assets changed; use a new run directory"
            )
    elif any(path.name != ".lock" for path in args.run_dir.iterdir()):
        p.error("run directory is nonempty without a benchmark manifest")
    else:
        json_write(manifest_path, manifest)
    planned = trials(args)
    print(f"Results: {args.run_dir}", flush=True)
    write_results(args.run_dir, planned)
    json_write(
        args.run_dir / "plan.json",
        [
            {
                "trial": asdict(trial),
                "agent_config": agent_config(args, trial),
                "server": server_command(args, trial),
                "harbor": harbor_command(
                    args, trial, args.run_dir / "trials" / trial.key / "execution-1"
                ),
            }
            for trial in planned
        ],
    )
    if args.dry_run:
        print(f"Planned {len(planned)} trials: {args.run_dir / 'plan.json'}")
        return 0
    errors = check_prerequisites(args)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2
    try:
        for trial in planned:
            root = args.run_dir / "trials" / trial.key
            if (root / "result.json").exists():
                continue
            attempt = 1
            while (root / f"execution-{attempt}").exists():
                attempt += 1
            print(f"Running {trial.key}", flush=True)
            result = run_trial(args, trial, root / f"execution-{attempt}")
            json_write(root / "result.json", result)
            write_results(args.run_dir, planned)
            service = result["service"]
            print(
                f"{trial.key}: {result['status']}, reward={result.get('reward')}, "
                f"agent_exit={result.get('agent_exit')}, tool_observations={result.get('tool_observations')}, "
                f"peak_prompt={service['peak_prompt_tokens']} + reserve={args.max_output_tokens} "
                f"<= context={args.max_context}\nEvidence: {result['execution']}",
                flush=True,
            )
    except KeyboardInterrupt:
        write_results(args.run_dir, planned)
        print(
            "Interrupted; completed trials are preserved for --resume.", file=sys.stderr
        )
        return 130
    return int(
        any(
            json.loads(
                (args.run_dir / "trials" / trial.key / "result.json").read_text()
            )["status"]
            != "scored"
            for trial in planned
        )
    )


def main(argv=None) -> int:
    p = parser()
    try:
        args = p.parse_args(config_arguments(p, sys.argv[1:] if argv is None else argv))
    except (OSError, ValueError) as error:
        p.error(str(error))
    try:
        validate(args)
    except ValueError as error:
        p.error(str(error))
    if args.check:
        errors = check_prerequisites(args)
        print("\n".join(errors) if errors else "Prerequisites ready.")
        return int(bool(errors))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("another benchmark process owns this run directory")
        return run_matrix(args, p)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    raise SystemExit(main())
