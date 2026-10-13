# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import yaml

from executorch.examples.llm_server.evals.terminal_bench import runner
from executorch.examples.llm_server.evals.terminal_bench.sessions import (
    SessionCleanup,
    SessionMiniSweAgent,
)
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.cli.job_plugins import attach_job_plugin
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import TrialConfig
from harbor.trial.hooks import TrialEvent
from harbor.trial.queue import TrialQueue


class ConfigurationTest(unittest.TestCase):
    def load(self, options="", limits=""):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[server]\nworker_bin="worker"\nmodel_path="model.pte"\n'
                'tokenizer_path="tokenizer.json"\nhf_tokenizer="org/model"\n'
                f"max_context=8192\n{limits}\n[terminal_bench]\n"
                f'agent_host="host.lima.internal"\n{options}\n'
            )
            return runner.load_config(path)

    def test_serial_command_unchanged(self):
        config = self.load('session_affinity="fixed"')
        serve, command, _ = runner.commands(config, Path("/results"))
        self.assertEqual(command[command.index("--n-concurrent") + 1], "1")
        self.assertEqual(command[command.index("--agent") + 1], "mini-swe-agent")
        self.assertNotIn("--plugin", command)
        for key in ("max_sessions", "max_inflight_requests", "max_decode_sequences"):
            self.assertNotIn(key, config["server"])
            self.assertFalse(
                any(arg.startswith("--" + key.replace("_", "-")) for arg in serve)
            )
        configured = self.load("concurrency=1", "max_sessions=2")
        self.assertEqual(configured["server"]["max_sessions"], 2)

    def test_concurrency_wires_agent_and_cleanup(self):
        config = self.load("concurrency=2")
        serve, command, _ = runner.commands(config, Path("/results"))
        self.assertEqual(command[command.index("--n-concurrent") + 1], "2")
        self.assertTrue(
            command[command.index("--agent") + 1].endswith(":SessionMiniSweAgent")
        )
        self.assertTrue(
            command[command.index("--plugin") + 1].endswith(":SessionCleanup")
        )
        self.assertIn("base_url=http://127.0.0.1:8000/v1", command)
        for key in ("max_sessions", "max_inflight_requests", "max_decode_sequences"):
            self.assertEqual(config["server"][key], 2)
            self.assertIn("--" + key.replace("_", "-") + "=2", serve)

    def test_explicit_limits_override_defaults(self):
        config = self.load(
            "concurrency=4",
            "max_sessions=8\nmax_inflight_requests=6\nmax_decode_sequences=2",
        )
        serve, _, _ = runner.commands(config, Path("/results"))
        for key, value in {
            "max_sessions": 8,
            "max_inflight_requests": 6,
            "max_decode_sequences": 2,
        }.items():
            self.assertEqual(config["server"][key], value)
            self.assertIn("--" + key.replace("_", "-") + f"={value}", serve)

    def test_rejects_invalid_concurrency_and_shared_affinity(self):
        for value in ("0", "-1", "true", "1.5", '"2"'):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                self.load(f"concurrency={value}")
        with self.assertRaisesRegex(ValueError, "session_affinity"):
            self.load('concurrency=2\nsession_affinity="shared"')

    def test_requires_sufficient_capacity(self):
        for limits, missing in [
            ("max_sessions=1\nmax_inflight_requests=2", "max_sessions"),
            ("max_sessions=2\nmax_inflight_requests=1", "max_inflight_requests"),
            ('max_sessions="2"\nmax_inflight_requests=2', "max_sessions"),
            ("max_decode_sequences=0", "max_decode_sequences"),
            ("max_decode_sequences=true", "max_decode_sequences"),
            ("max_decode_sequences=1.5", "max_decode_sequences"),
        ]:
            with self.subTest(limits=limits), self.assertRaisesRegex(
                ValueError, missing
            ):
                self.load("concurrency=2", limits)


class SessionLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_file = self.root / "agent.yaml"
        self.config = {
            "agent": {"step_limit": 100},
            "model": {
                "model_kwargs": {
                    "temperature": 0.8,
                    "extra_headers": {"custom": "kept"},
                }
            },
        }
        self.config_file.write_text(json.dumps(self.config))
        self.open_sessions = {}
        self.observed = set()
        self.stopped = set()
        self.peak = 0
        self.fail_cleanup = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                sid = self.headers["X-ExecuTorch-Session-ID"]
                if sid not in owner.open_sessions and len(owner.open_sessions) == 2:
                    self.send_response(429)
                    self.end_headers()
                    return
                previous = owner.open_sessions.get(sid, 0)
                owner.open_sessions[sid] = previous + 1
                owner.observed.add(sid)
                owner.peak = max(owner.peak, len(owner.open_sessions))
                self.send_response(200 if body["turn"] == previous else 409)
                self.end_headers()
                self.wfile.write(json.dumps({"previous_turns": previous}).encode())

            def do_DELETE(self):
                sid = self.path.rsplit("/", 1)[-1]
                if owner.fail_cleanup or sid not in owner.stopped:
                    self.send_response(500)
                else:
                    owner.open_sessions.pop(sid, None)
                    self.send_response(200)
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    async def asyncTearDown(self):
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        self.thread.join()

    def make_agent(self, name):
        return SessionMiniSweAgent(
            logs_dir=self.root / name / "agent",
            config_file=str(self.config_file),
            model_name="openai/test-model",
        )

    def test_adapter_preserves_agent_behavior_and_config(self):
        first = self.make_agent("task__attempt1")
        second = self.make_agent("task__attempt2")
        self.assertIs(SessionMiniSweAgent.run, MiniSweAgent.run)
        self.assertIs(SessionMiniSweAgent.setup, MiniSweAgent.setup)
        for agent, name in [(first, "task__attempt1"), (second, "task__attempt2")]:
            config = yaml.safe_load(agent._config_yaml)
            headers = config["model"]["model_kwargs"]["extra_headers"]
            self.assertEqual(headers["X-ExecuTorch-Session-ID"], name + "__agent")
            self.assertEqual(headers["custom"], "kept")
            self.assertEqual(config["model"]["model_kwargs"]["temperature"], 0.8)
            self.assertEqual(config["agent"], self.config["agent"])
        self.assertEqual(json.loads(self.config_file.read_text()), self.config)

    async def test_six_trials_reuse_two_slots_including_failure_and_cancellation(self):
        owner = self
        queue = TrialQueue(n_concurrent=2, retry_config=RetryConfig(max_retries=0))
        await attach_job_plugin(
            queue,
            "executorch.examples.llm_server.evals.terminal_bench.sessions:SessionCleanup",
            kwargs={"base_url": self.base_url},
        )

        class FakeTrial:
            def __init__(self, config):
                self.config = config
                self.hooks = {event: [] for event in TrialEvent}

            def add_hook(self, event, callback):
                self.hooks[event].append(callback)

            async def run(self):
                name = self.config.trial_name
                agent = owner.make_agent(name)
                headers = yaml.safe_load(agent._config_yaml)["model"]["model_kwargs"][
                    "extra_headers"
                ]
                result = SimpleNamespace(exception_info=None, trial_name=name)
                try:
                    async with httpx.AsyncClient(trust_env=False) as client:
                        for turn in range(2):
                            response = await client.post(
                                owner.base_url + "/chat/completions",
                                headers=headers,
                                json={"turn": turn},
                            )
                            response.raise_for_status()
                            owner.assertEqual(response.json()["previous_turns"], turn)
                            await asyncio.sleep(0.02)
                            if name == "task__attempt1":
                                result.exception_info = SimpleNamespace(
                                    exception_type="AgentTimeoutError"
                                )
                                return result
                            if name == "task__attempt3":
                                raise asyncio.CancelledError()
                    return result
                finally:
                    owner.stopped.add(name + "__agent")
                    event = SimpleNamespace(trial_name=name, result=result)
                    for hook in self.hooks[TrialEvent.END]:
                        await hook(event)

        async def create(config):
            return FakeTrial(config)

        configs = [
            TrialConfig(trial_name=f"task__attempt{i}", task={"path": str(self.root)})
            for i in range(6)
        ]
        with patch("harbor.trial.trial.Trial.create", side_effect=create):
            results = await asyncio.gather(
                *queue.submit_batch(configs), return_exceptions=True
            )
        self.assertEqual(self.peak, 2)
        self.assertEqual(len(self.observed), 6)
        self.assertEqual(self.open_sessions, {})
        self.assertEqual(results[1].exception_info.exception_type, "AgentTimeoutError")
        self.assertIsInstance(results[3], asyncio.CancelledError)
        self.assertTrue(all(results[i].exception_info is None for i in [0, 2, 4, 5]))

    async def test_cleanup_failure_is_reported(self):
        self.fail_cleanup = True
        plugin = SessionCleanup(self.base_url)
        with self.assertRaises(httpx.HTTPStatusError):
            await plugin.on_trial_end(SimpleNamespace(trial_name="failed"))


if __name__ == "__main__":
    unittest.main()
