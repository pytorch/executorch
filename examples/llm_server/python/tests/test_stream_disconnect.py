# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Real-socket coverage for HTTP disconnect -> worker cancellation."""

import json
import os
import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.worker_client import spawn_worker


_FAKE_WORKER = r"""
import json
import os
from pathlib import Path
import select
import sys

control_fd = int(os.environ["EXECUTORCH_LLM_WORKER_CONTROL_FD"])
marker = Path(sys.argv[1])
print(json.dumps({"ready": True, "supports_cancel": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if "op" in request:
        op = request["op"]
        print(json.dumps({op + "ed": True, "session_id": request["session_id"]}), flush=True)
        continue
    request_id = request["cancel_request_id"]
    print(json.dumps({"token": "worker-token-visible-content"}), flush=True)
    readable, _, _ = select.select([control_fd], [], [], 10)
    if not readable:
        marker.write_text(json.dumps({"request_id": request_id, "error": "timeout"}))
        sys.exit(2)
    frame = b""
    while len(frame) < 8:
        chunk = os.read(control_fd, 8 - len(frame))
        if not chunk:
            marker.write_text(json.dumps({"request_id": request_id, "error": "eof"}))
            sys.exit(3)
        frame += chunk
    cancel_id = int.from_bytes(frame, "little")
    marker.write_text(json.dumps({"request_id": request_id, "cancel_id": cancel_id}))
    print(
        json.dumps(
            {
                "done": True,
                "finish_reason": "stop",
                "cancelled": True,
                "prompt_tokens": 1,
                "completion_tokens": 1,
            }
        ),
        flush=True,
    )
"""


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.skipif(
    os.name != "posix", reason="cancellation control pipe is POSIX-only"
)
def test_socket_disconnect_cancels_matching_worker_request(tmp_path):
    worker_script = tmp_path / "fake_worker.py"
    marker = tmp_path / "cancelled.json"
    worker_script.write_text(_FAKE_WORKER)

    worker = spawn_worker([sys.executable, "-u", str(worker_script), str(marker)])
    runtime = SessionRuntime(worker, cancel_grace_seconds=1.0)
    serving = ServingChat(
        runtime,
        ChatTemplate(hf_tokenizer_path=None, allow_fallback=True),
        "test-model",
    )
    app = build_app(serving, "test-model")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="off", access_log=False)
    )
    server_thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    server_thread.start()

    client = None
    try:
        assert _wait_until(lambda: server.started), "Uvicorn did not start"
        body = json.dumps(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 8,
            }
        ).encode()
        request = (
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode() + body

        client = socket.create_connection(("127.0.0.1", port), timeout=5)
        client.sendall(request)
        response = b""
        while b"worker-token" not in response:
            try:
                chunk = client.recv(4096)
            except TimeoutError as error:
                phase = marker.read_text() if marker.exists() else "no worker marker"
                raise AssertionError(
                    f"stream stalled before worker token; marker={phase}; "
                    f"response={response!r}"
                ) from error
            assert chunk, "stream ended before the worker token"
            response += chunk
        client.shutdown(socket.SHUT_RDWR)
        client.close()
        client = None

        observed = None

        def read_cancel_marker():
            nonlocal observed
            try:
                candidate = json.loads(marker.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return False
            if "cancel_id" not in candidate:
                return False
            observed = candidate
            return True

        assert _wait_until(read_cancel_marker), "worker did not receive cancellation"
        assert observed["cancel_id"] == observed["request_id"]
        assert observed["request_id"] > 0

        health = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5)
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
    finally:
        if client is not None:
            client.close()
        server.should_exit = True
        server_thread.join(timeout=5)
        listener.close()
        runtime.close_worker()
        assert not server_thread.is_alive()
