# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Subprocess doubles for runner tests; these do not evaluate a benchmark task."""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen


def event(path, name):
    with Path(path).open("a") as stream:
        stream.write(name + "\n")


def fake_server(port, events):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, payload):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def do_GET(self):
            self.respond({"status": "ok"})

        def do_POST(self):
            if self.path.endswith("/reset"):
                event(events, "reset")
                self.respond({"reset": True})
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            session = self.headers.get("x-session-affinity", "<scratch>")
            later = len(body["messages"]) > 2
            reused = later and session != "<scratch>"
            reason = "exact_prefix" if reused else "new"
            print(
                f"INFO llm_turn_stats session_id={session} reason={reason} prompt_tokens=100 reused_prompt_tokens={25 if reused else 0} prefilled_prompt_tokens={75 if reused else 100} completion_tokens=2 prefill_ms=10 decode_ms=5 total_ms=15 finish=stop",
                flush=True,
            )
            event(events, "chat")
            self.respond(
                {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
            )

        def do_DELETE(self):
            event(events, "delete")
            self.respond({"closed": True})

    event(events, "start")
    HTTPServer(("127.0.0.1", int(port)), Handler).serve_forever()


def python_server(argv):
    from executorch.examples.llm_server.python import server
    from executorch.examples.llm_server.python.chat_template import ChatTemplate
    from executorch.examples.llm_server.python.tests.conftest import FakeRunner

    class ByteTemplate(ChatTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(allow_fallback=True)

        def count_tokens(self, text):
            return len(text.encode())

    server.ChatTemplate = ByteTemplate
    server._spawn = lambda args: FakeRunner(["OK"], max_named_sessions=2)
    sys.argv = ["server", *argv]
    server.main()


def fake_harbor(argv):
    options = dict(zip(argv[::2], argv[1::2]))
    kwargs = dict(
        value.split("=", 1)
        for flag, value in zip(argv[::2], argv[1::2])
        if flag == "--ak"
    )
    env = dict(
        value.split("=", 1)
        for flag, value in zip(argv[::2], argv[1::2])
        if flag == "--ae"
    )
    config = json.loads(Path(kwargs["config_file"]).read_text())
    headers = {
        "Content-Type": "application/json",
        **config["model"]["model_kwargs"]["extra_headers"],
    }
    history = [
        {"role": "system", "content": "Help with the task."},
        {"role": "user", "content": "Inspect a.py."},
    ]
    for turn in range(2):
        request = Request(
            env["OPENAI_BASE_URL"] + "/chat/completions",
            headers=headers,
            data=json.dumps(
                {
                    "model": options["--model"].removeprefix("openai/"),
                    "messages": history,
                    "max_tokens": 4,
                    "temperature": 0,
                }
            ).encode(),
        )
        with urlopen(request, timeout=15) as response:
            history.append(json.load(response)["choices"][0]["message"])
        if turn == 0:
            history.extend(
                [
                    {"role": "user", "content": "Command finished successfully."},
                    {"role": "assistant", "content": "Now inspect b.py"},
                    {"role": "user", "content": "done"},
                ]
            )
    if os.environ.get("BENCH_TEST_FAIL"):
        return 29
    task = Path(options["--jobs-dir"]) / options["--job-name"] / "fake-task"
    (task / "agent").mkdir(parents=True)
    (task / "result.json").write_text(
        json.dumps(
            {"verifier_result": {"rewards": {"reward": 1.0}}, "exception_info": None}
        )
    )
    (task / "agent" / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {"info": {"model_stats": {"api_calls": 2}, "exit_status": "Submitted"}}
        )
    )
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "server":
        fake_server(*args)
    elif mode == "python-server":
        python_server(args)
    else:
        raise SystemExit(fake_harbor(args))
