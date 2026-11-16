# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shutdown regressions with an external timeout; only the stdlib is needed."""

import asyncio
import os
import signal
import subprocess
import sys
import unittest
from pathlib import Path

_WORKER = """
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "generate":
        print(json.dumps({"request_id": request["request_id"], "token": "a"}), flush=True)
"""


async def _shutdown_queued(queued):
    from executorch.examples.llm_server.python.multiplexed_worker_client import (
        MultiplexedWorkerClient,
    )
    from executorch.examples.llm_server.python.session_runtime import (
        GenerationOptions,
        PromptInput,
        SessionRuntime,
    )
    from executorch.examples.llm_server.python.worker_client import WorkerError

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _WORKER,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        runtime = SessionRuntime(MultiplexedWorkerClient(proc))
        generation = runtime.generate_stream(
            "s", PromptInput(text="hi"), GenerationOptions(max_new_tokens=8)
        )
        await generation.wait_ready()
        operations = (runtime.open, runtime.reset, runtime.close)
        waiters = [
            asyncio.create_task(operations[i % len(operations)]("s"))
            for i in range(queued)
        ]
        await asyncio.sleep(0)
        assert runtime._admitted == queued + 1
        await asyncio.wait_for(runtime.aclose_worker(), timeout=5)
        assert proc.returncode is not None
        assert not runtime._operations and not runtime._native_tasks
        assert runtime._admitted == 0 and not runtime._session_locks._entries
        assert all(waiter.done() for waiter in waiters)
        errors = await asyncio.gather(*waiters, return_exceptions=True)
        assert all(isinstance(error, WorkerError) for error in errors)
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


class SessionRuntimeShutdownTest(unittest.TestCase):
    def test_queued_lifecycle_shutdown(self):
        path = Path(__file__).resolve()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(path.parents[5]), env.get("PYTHONPATH")))
        )
        # Queue depth changes which callbacks are pending at the final gather.
        for queued in (*range(9), 12):
            with self.subTest(queued=queued):
                with subprocess.Popen(
                    [sys.executable, str(path), "--scenario", str(queued)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=os.name == "posix",
                ) as child:
                    try:
                        output, _ = child.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        # asyncio timeouts cannot interrupt a non-yielding loop.
                        if os.name == "posix":
                            os.killpg(child.pid, signal.SIGKILL)
                        else:
                            child.kill()
                        output, _ = child.communicate()
                        self.fail(
                            f"shutdown stalled with {queued} queued operations\n{output}"
                        )
                    self.assertEqual(child.returncode, 0, output)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--scenario"]:
        asyncio.run(_shutdown_queued(int(sys.argv[2])))
    else:
        unittest.main()
