# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run Terminal-Bench through Harbor against a local LLM server on macOS."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE = Path(
    os.environ.get("EXECUTORCH_EVAL_CACHE", Path.home() / ".cache/executorch-evals")
)
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_config(path):
    config = tomllib.loads(path.read_text())
    server = config["server"]
    server.setdefault("python", sys.executable)
    server.setdefault("module", "executorch.examples.llm_server.python.server")
    server.setdefault("host", "0.0.0.0")
    server.setdefault("port", 8000)
    for key in ("worker_bin", "model_path", "tokenizer_path", "hf_tokenizer", "python"):
        value = server[key]
        if key in {"worker_bin", "model_path", "tokenizer_path"} or value.startswith(
            ("/", "~", ".")
        ):
            server[key] = str((path.parent / Path(value).expanduser()).resolve())
    defaults = {
        "tasks": ["fix-git"],
        "attempts": 1,
        "step_limit": 100,
        "max_output_tokens": 512,
        "agent_host": None,
    }
    options = config.get("terminal_bench", {})
    if unknown := options.keys() - defaults.keys():
        raise ValueError(f"Unknown Terminal-Bench settings: {sorted(unknown)}")
    config["terminal_bench"] = options = defaults | options
    if not 0 < options["max_output_tokens"] < server["max_context"]:
        raise ValueError(
            "max_output_tokens must be positive and smaller than max_context"
        )
    if min(options["attempts"], options["step_limit"]) < 1 or not options["tasks"]:
        raise ValueError("Choose tasks and positive attempt/step limits")
    return config


def commands(config, output):
    server, options = config["server"], config["terminal_bench"]
    serve = [server["python"], "-m", server["module"]]
    for key, value in server.items():
        if key in {"python", "module"} or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            serve.append(flag)
        else:
            serve.extend(
                f"{flag}={item}"
                for item in (value if isinstance(value, list) else [value])
            )
    agent_host = options["agent_host"]
    if not agent_host:
        context = subprocess.check_output(
            ["docker", "context", "show"], text=True
        ).strip()
        agent_host = (
            "host.lima.internal"
            if context.startswith("colima")
            else "host.docker.internal"
        )
    url = f"http://{agent_host}:{server['port']}"
    harbor = [
        str(CACHE / "terminal-bench/venv/bin/harbor"),
        "run",
        "--dataset",
        "terminal-bench@2.0",
        "--agent",
        "mini-swe-agent",
        "--model",
        f"openai/{server['model_id']}",
        "--ak",
        "version=2.4.6",
        "--ak",
        f"max_tokens={options['max_output_tokens']}",
        "--ak",
        f"config_file={output / 'agent.yaml'}",
        "--ae",
        "MSWEA_API_KEY=local",
        "--ae",
        "OPENAI_API_KEY=local",
        "--ae",
        f"OPENAI_BASE_URL={url}/v1",
        "--ae",
        f"NO_PROXY={agent_host}",
        "--ae",
        f"no_proxy={agent_host}",
        "--n-concurrent",
        "1",
        "--n-attempts",
        str(options["attempts"]),
        "--max-retries",
        "0",
        "--jobs-dir",
        str(output),
        "--job-name",
        "harbor",
    ]
    for task in options["tasks"]:
        harbor.extend(["--include-task-name", task])
    probe = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"executorch-eval-{os.getpid()}",
        "curlimages/curl:8.12.1",
        "--noproxy",
        "*",
        "--fail",
        "--max-time",
        "15",
        url + "/health",
    ]
    return serve, harbor, probe


def wait_ready(process, host, port, timeout=180):
    host = "127.0.0.1" if host == "0.0.0.0" else host
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited; see server.log")
        try:
            with HTTP.open(f"http://{host}:{port}/health", timeout=2) as response:
                if json.load(response).get("status") == "ok":
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    raise TimeoutError("Server did not become ready; see server.log")


def stop(process, stop_signal=signal.SIGTERM):
    if process is None:
        return
    # The worker can outlive its Python server; terminate the whole process group.
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


def metrics(path, context, reserve):
    turns = []
    for line in path.read_text(errors="replace").splitlines():
        if "llm_turn_stats " in line:
            turns.append(
                dict(
                    item.split("=", 1)
                    for item in line.split("llm_turn_stats ", 1)[1].split()
                    if "=" in item
                )
            )
    result = {
        key: sum((float if key.endswith("_ms") else int)(turn[key]) for turn in turns)
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "prefilled_prompt_tokens",
            "reused_prompt_tokens",
            "prefill_ms",
            "decode_ms",
        )
    }
    result["requests"] = len(turns)
    result["peak_prompt_tokens"] = max(
        (int(turn["prompt_tokens"]) for turn in turns), default=0
    )
    result["context_violations"] = sum(
        int(turn["prompt_tokens"]) + reserve > context for turn in turns
    )
    return result


def run(config, output, serve, harbor, probe):
    server = config["server"]
    process = job = None
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with (output / "server.log").open("w") as server_log, (output / "harbor.log").open(
        "w"
    ) as harbor_log:
        try:
            with socket.socket() as port:
                port.bind((server["host"], server["port"]))
            process = subprocess.Popen(
                serve,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_ready(process, server["host"], server["port"])
            try:
                with (output / "connection.log").open("w") as log:
                    subprocess.run(
                        probe,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=120,
                    )
            finally:
                subprocess.run(
                    ["docker", "rm", "--force", probe[probe.index("--name") + 1]],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                )
            job = subprocess.Popen(
                harbor,
                stdout=harbor_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if job.wait():
                raise RuntimeError("Harbor failed; see harbor.log")
            result = json.loads((output / "harbor/result.json").read_text())
            stats = result["stats"]
            if (
                not result["n_total_trials"]
                or stats["n_completed_trials"] != result["n_total_trials"]
                or stats["n_errored_trials"]
            ):
                raise RuntimeError(
                    "Harbor reported incomplete or errored trials; see harbor/result.json"
                )
            print(json.dumps(stats, indent=2))
        finally:
            stop(job, signal.SIGINT)
            stop(process)
            server_log.flush()
            timing = metrics(
                output / "server.log",
                server["max_context"],
                config["terminal_bench"]["max_output_tokens"],
            )
            (output / "metrics.json").write_text(json.dumps(timing, indent=2) + "\n")
    if not timing["requests"] or timing["context_violations"]:
        raise RuntimeError(
            "Missing server metrics or context limit exceeded; see metrics.json"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--task", action="append", help="Task name or glob; repeat to select a subset"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CACHE
        / "terminal-bench/runs"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.error("Terminal-Bench evaluations currently support macOS only.")
    try:
        config = load_config(args.config.expanduser().resolve())
        if args.task:
            config["terminal_bench"]["tasks"] = args.task
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        serve, harbor, probe = commands(config, output)
        (output / "agent.yaml").write_text(
            json.dumps(
                {
                    "agent": {"step_limit": config["terminal_bench"]["step_limit"]},
                    "model": {"model_kwargs": {"temperature": 0}},
                }
            )
        )
        revision = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        (output / "run.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "revision": revision,
                    "server": serve,
                    "harbor": harbor,
                    "probe": probe,
                },
                indent=2,
            )
            + "\n"
        )
        (output / "source.diff").write_bytes(
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "diff", "HEAD"])
        )
        print(f"Results: {output}", flush=True)
        if not args.dry_run:
            run(config, output, serve, harbor, probe)
        return 0
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    raise SystemExit(main())
