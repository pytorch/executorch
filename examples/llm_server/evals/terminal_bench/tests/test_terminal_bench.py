# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from executorch.examples.llm_server.evals.terminal_bench import runner as bench

PROCESS = Path(__file__).parent / "fixtures" / "benchmark_process.py"
SERVER_COMMAND = bench.server_command


@pytest.fixture
def argv(tmp_path):
    task = tmp_path / "fix-git"
    task.mkdir()
    (task / "instruction.md").write_text("Fix the repository.")
    (task / "task.toml").write_text('version = "1.0"\n')
    for name in ("worker", "model.pte", "tokenizer.json"):
        (tmp_path / name).write_text("fixture")
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        number = port.getsockname()[1]
    return [
        "--session-affinity",
        "--attempts",
        "3",
        "--task",
        str(task),
        "--run-dir",
        str(tmp_path / "run"),
        "--executorch-root",
        str(tmp_path / "executorch"),
        "--worker-bin",
        str(tmp_path / "worker"),
        "--model-path",
        str(tmp_path / "model.pte"),
        "--tokenizer-path",
        str(tmp_path / "tokenizer.json"),
        "--hf-tokenizer",
        "test",
        "--max-context",
        "1600",
        "--max-output-tokens",
        "4",
        "--host",
        "127.0.0.1",
        "--port",
        str(number),
        "--agent-base-url",
        f"http://127.0.0.1:{number}/v1",
    ]


def arguments(argv):
    args = bench.parser().parse_args(argv)
    bench.validate(args)
    return args


@pytest.fixture
def processes(monkeypatch, tmp_path):
    events = tmp_path / "events"
    actual_harbor = bench.harbor_command
    monkeypatch.setattr(bench, "check_prerequisites", lambda args: [])
    monkeypatch.setattr(
        bench, "check_container_connection", lambda args, execution: None
    )
    monkeypatch.setattr(
        bench,
        "server_command",
        lambda args, trial: [
            sys.executable,
            str(PROCESS),
            "server",
            str(args.port),
            str(events),
        ],
    )
    monkeypatch.setattr(
        bench,
        "harbor_command",
        lambda args, trial, execution: [
            sys.executable,
            str(PROCESS),
            "harbor",
            *actual_harbor(args, trial, execution)[2:],
        ],
    )
    return events


def test_trials_and_context_controls(argv):
    args = arguments(argv + ["--attempts", "2"])
    planned = bench.trials(args)
    assert len(planned) == 2
    assert len({t.session_id for t in planned}) == 2
    for trial in planned:
        command = bench.server_command(args, trial)
        assert command[command.index("--max-context") + 1] == "1600"
        assert "--no-think" in command
        assert not any("worker-arg" in arg or "cliff" in arg for arg in command)
        assert "max_tokens=4" in bench.harbor_command(args, trial, args.run_dir)
        config = bench.agent_config(args, trial)
        assert config["model"]["model_kwargs"]["temperature"] == 0
        assert (
            config["model"]["model_kwargs"]["extra_headers"]["x-session-affinity"]
            == trial.session_id
        )


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-output-tokens", "1600"],
        ["--server-arg=--max-context=8192"],
        ["--server-arg=--port=9000"],
        ["--agent-base-url", "http://localhost:9000/v1"],
        ["--step-limit", "0"],
    ],
)
def test_reject_invalid_controls(argv, extra):
    with pytest.raises(ValueError):
        arguments(argv + extra)


def test_parse_metrics_separates_context_from_accumulated_input(tmp_path):
    path = tmp_path / "server.log"
    path.write_text(
        "INFO llm_turn_stats session_id=s reason=prefix_cache prompt_tokens=100 reused_prompt_tokens=40 prefilled_prompt_tokens=60 completion_tokens=4 prefill_ms=1 decode_ms=2 total_ms=3\n"
        "INFO llm_turn_stats session_id=s reason=exact_prefix prompt_tokens=150 reused_prompt_tokens=104 prefilled_prompt_tokens=46 completion_tokens=4 prefill_ms=2 decode_ms=3 total_ms=5\n"
        "INFO llm_turn_stats session_id=other prompt_tokens=99999\n"
    )
    stats = bench.parse_service_log(path, "s", 200, 20)
    assert stats["input_tokens"] == 250
    assert stats["peak_prompt_tokens"] == 150
    assert stats["context_violations"] == 0
    assert stats["reused_tokens"] == 144
    assert stats["prefilled_tokens"] == 106
    assert stats["prefix_cache_hits"] == stats["continuation_hits"] == 1
    assert bench.parse_service_log(path, "s", 160, 20)["context_violations"] == 1


def test_process_lifecycle_and_resume(argv, processes):
    assert bench.main(argv) == 0
    args = arguments(argv)
    rows = json.loads((args.run_dir / "results.json").read_text())
    assert len(rows) == 3
    assert (
        processes.read_text().splitlines()
        == ["start", "reset", "chat", "chat", "delete"] * 3
    )
    assert all(row["reward"] == 1 and row["status"] == "scored" for row in rows)
    assert all(row["validation"]["cleanup_succeeded"] for row in rows)
    assert bench.main(argv + ["--resume"]) == 0
    assert len(processes.read_text().splitlines()) == 15
    with pytest.raises(SystemExit):
        bench.main(argv + ["--resume", "--step-limit", "10"])


def test_cleanup_when_harbor_fails(argv, processes, monkeypatch):
    monkeypatch.setenv("BENCH_TEST_FAIL", "1")
    assert bench.main(argv + ["--attempts", "1"]) == 1
    args = arguments(argv)
    row = json.loads((args.run_dir / "results.json").read_text())[0]
    assert row["status"] == "harness_error"
    assert row["harbor_returncode"] == 29
    assert row["validation"]["cleanup_succeeded"]
    assert processes.read_text().splitlines()[-1] == "delete"


def test_dry_run_does_not_launch_processes(argv, processes):
    assert bench.main(argv + ["--dry-run"]) == 0
    assert not processes.exists()
    args = arguments(argv)
    assert len(json.loads((args.run_dir / "plan.json").read_text())) == 3
    assert bench.main(argv + ["--resume"]) == 0


def test_startup_failure_is_recorded(argv, processes, monkeypatch):
    monkeypatch.setattr(
        bench,
        "server_command",
        lambda args, trial: [sys.executable, "-c", "raise SystemExit(7)"],
    )
    assert bench.main(argv + ["--attempts", "1"]) == 1
    args = arguments(argv)
    row = json.loads((args.run_dir / "results.json").read_text())[0]
    assert "server exited with code 7" in row["error"]
    assert row["validation"]["context_within_limit"] is None


def test_occupied_port_does_not_launch_or_contact_existing_service(argv, processes):
    args = arguments(argv)
    with socket.socket() as occupied:
        occupied.bind((args.host, args.port))
        occupied.listen()
        assert bench.main(argv + ["--attempts", "1"]) == 1
    assert not processes.exists()


def test_resume_rejects_changed_task(argv, processes):
    assert bench.main(argv + ["--dry-run"]) == 0
    args = arguments(argv)
    (args.task[0] / "instruction.md").write_text("Different task")
    with pytest.raises(SystemExit):
        bench.main(argv + ["--resume"])
    assert not processes.exists()


def test_interruption_preserves_evidence_and_can_resume(argv, processes, monkeypatch):
    original = bench.read_harbor_result

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(bench, "read_harbor_result", interrupt)
    argv = argv + ["--attempts", "1"]
    assert bench.main(argv) == 130
    args = arguments(argv)
    trial = bench.trials(args)[0]
    root = args.run_dir / "trials" / trial.key
    assert (
        json.loads((root / "execution-1" / "outcome.json").read_text())["status"]
        == "interrupted"
    )
    assert not (root / "result.json").exists()
    assert processes.read_text().splitlines()[-1] == "delete"
    monkeypatch.setattr(bench, "read_harbor_result", original)
    assert bench.main(argv + ["--resume"]) == 0
    assert (root / "execution-2" / "outcome.json").exists()


@pytest.mark.parametrize("affinity", [False, True])
def test_managed_python_server(argv, processes, monkeypatch, affinity):
    # Use the real launcher's argv, HTTP endpoints, SessionRuntime, and logs.
    monkeypatch.setattr(
        bench,
        "server_command",
        lambda args, trial: [
            sys.executable,
            str(PROCESS),
            "python-server",
            *SERVER_COMMAND(args, trial)[3:],
        ],
    )
    options = [arg for arg in argv if arg != "--session-affinity"] + ["--attempts", "1"]
    if affinity:
        options.append("--session-affinity")
    assert bench.main(options) == 0
    args = arguments(options)
    row = json.loads((args.run_dir / "results.json").read_text())[0]
    assert row["validation"]["context_within_limit"]
    assert row["validation"]["cleanup_succeeded"]
    assert row["service"]["completed_turns"] == 2


def test_config_paths_and_cli_override(argv, tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        'worker_bin = "worker"\n'
        'model_path = "model.pte"\n'
        'tokenizer_path = "tokenizer.json"\n'
        'hf_tokenizer = "./local-tokenizer"\n'
        'task = ["fix-git"]\n'
        "max_context = 1600\n"
        "attempts = 2\n"
        'server_arg = ["--assistant-header=custom"]\n'
        "thinking = true\n"
        "session_affinity = true\n"
        'agent_base_url = "http://host.docker.internal:8000/v1"\n'
    )
    p = bench.parser()
    args = p.parse_args(
        bench.config_arguments(p, ["--config", str(path), "--attempts", "4"])
    )
    bench.validate(args)
    assert args.worker_bin == tmp_path / "worker"
    assert args.task == [tmp_path / "fix-git"]
    assert args.hf_tokenizer == str(tmp_path / "local-tokenizer")
    assert args.attempts == 4
    assert args.server_arg == ["--assistant-header=custom"]
    assert args.thinking
    assert args.session_affinity


@pytest.mark.parametrize(
    "content",
    ["typo = 2", "resume = true", 'thinking = "yes"', 'model_id = ["a", "b"]'],
)
def test_invalid_config_is_rejected(tmp_path, content):
    path = tmp_path / "bad.toml"
    path.write_text(content)
    with pytest.raises(ValueError):
        bench.config_arguments(bench.parser(), ["--config", str(path)])


def test_fresh_directories_and_explicit_resume(argv, tmp_path):
    index = argv.index("--run-dir")
    argv = argv[:index] + argv[index + 2 :] + ["--output-root", str(tmp_path / "runs")]
    first, second = arguments(argv), arguments(argv)
    assert first.run_dir != second.run_dir
    assert first.run_dir.parent == second.run_dir.parent == tmp_path / "runs"
    with pytest.raises(ValueError, match="explicit --run-dir"):
        arguments(argv + ["--resume"])


def test_summary_records_format_failure_without_claiming_tool_execution(tmp_path):
    task = tmp_path / "jobs/trial/task"
    (task / "agent").mkdir(parents=True)
    (task / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 0}}})
    )
    (task / "agent/mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "info": {
                    "exit_status": "RepeatedFormatError",
                    "model_stats": {"api_calls": 1},
                },
                "messages": [
                    {
                        "role": "user",
                        "extra": {
                            "interrupt_type": "FormatError",
                            "response": "<tool_call>bad call</tool_call>",
                        },
                    }
                ],
            }
        )
    )
    result = bench.read_harbor_result(tmp_path, 0)
    assert result["status"] == "scored" and result["reward"] == 0
    assert result["agent_exit"] == "RepeatedFormatError"
    assert result["format_errors"] == 1 and result["tool_observations"] == 0
    assert Path(result["trajectory"]).is_file()


def test_resume_rejects_asset_content_change_with_same_size_and_timestamp(
    argv, processes
):
    assert bench.main(argv + ["--dry-run"]) == 0
    args = arguments(argv)
    stat = args.model_path.stat()
    args.model_path.write_text("changed")
    os.utime(args.model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(SystemExit):
        bench.main(argv + ["--resume"])


def test_suite_overrides_configured_tasks(argv, tmp_path):
    config = tmp_path / "model.toml"
    config.write_text('task = ["does-not-exist"]\n')
    index = argv.index("--task")
    options = (
        argv[:index]
        + argv[index + 2 :]
        + ["--config", str(config), "--suite", "smoke", "--task-root", str(tmp_path)]
    )
    p = bench.parser()
    args = p.parse_args(bench.config_arguments(p, options))
    bench.validate(args)
    assert args.task == [tmp_path / "fix-git"]
    assert args.suite == "smoke"
    config.write_text('suite = "nightly"\n')
    args = p.parse_args(bench.config_arguments(p, argv + ["--config", str(config)]))
    bench.validate(args)
    assert args.suite is None and args.task == [tmp_path / "fix-git"]


def test_linux_container_mapping_matches_harbor_and_probe(argv, tmp_path, monkeypatch):
    monkeypatch.setattr(bench.sys, "platform", "linux")
    args = arguments(
        argv
        + ["--agent-base-url", f"http://host.docker.internal:{arguments(argv).port}/v1"]
    )
    command = bench.harbor_command(args, bench.trials(args)[0], tmp_path)
    assert command[-2:] == [
        "--extra-docker-compose",
        str(tmp_path / "host-gateway.yaml"),
    ]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '{"status":"ok"}')

    monkeypatch.setattr(bench.subprocess, "run", run)
    bench.check_container_connection(args, tmp_path)
    assert "host.docker.internal:host-gateway" in calls[0]
    assert calls[-1][:3] == ["docker", "rm", "--force"]
    assert calls[-1][-1] == calls[0][calls[0].index("--name") + 1]


def test_failed_container_route_cleans_up_without_running_agent(
    argv, processes, monkeypatch
):
    def fail(args, execution):
        raise RuntimeError("container route failed")

    monkeypatch.setattr(bench, "check_container_connection", fail)
    assert bench.main(argv + ["--attempts", "1"]) == 1
    args = arguments(argv)
    result = json.loads((args.run_dir / "results.json").read_text())[0]
    assert result["status"] == "harness_error"
    assert "container route failed" in result["error"]
    assert result["validation"]["cleanup_succeeded"]
    assert processes.read_text().splitlines() == ["start", "delete"]
    assert "harness_error" in (args.run_dir / "summary.md").read_text()


def test_probe_timeout_removes_its_container(argv, tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 120)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bench.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        bench.check_container_connection(arguments(argv), tmp_path)
    assert calls[-1][:3] == ["docker", "rm", "--force"]
