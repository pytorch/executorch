# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible HTTP server for Qwen3.5 MoE (process-isolated).

This is the CONTROL PLANE only: FastAPI/uvicorn + OpenAI protocol, chat
templating, tool parsing, request validation. It runs NO CUDA model code. Model
execution lives in a separate worker subprocess (worker.py) that this process
talks to over JSONL on stdin/stdout.

Why two processes: executing the AOTI CUDA model inside a live asyncio server
process segfaults in the int4 matmul (validated by elimination — not thread
affinity, GIL, signals, or executor offload; the trigger is CUDA execution while
a live asyncio loop is resident). Isolating CUDA in a plain (no-asyncio) worker
process is the reliable shape, and it still loads weights once.

V1 constraints:
  * serving_capacity == 1: one worker, one session; concurrent HTTP requests
    queue (RunnerPool num_runners=1).
  * prefix cache off (Qwen seek() is NotSupported).
  * The control plane only does blocking pipe I/O on its executor thread (no
    CUDA), which is safe under asyncio.

Launch (LD_LIBRARY_PATH shim is forwarded to the worker for the CUDA blob):

    LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \\
      python -m executorch.examples.models.qwen3_5_moe.serve \\
        --model-path  qwen35_moe_exports/model.pte \\
        --data-path   qwen35_moe_exports/aoti_cuda_blob.ptd \\
        --tokenizer-path ~/models/Qwen3.5-35B-A3B/tokenizer.json \\
        --hf-tokenizer   ~/models/Qwen3.5-35B-A3B \\
        --model-id qwen3.5-moe --no-think
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.runner_pool import RunnerPool
from executorch.extension.llm.server.python.serving_chat import ServingChat
from executorch.extension.llm.server.python.tool_parsers import QwenFunctionCallDetector

logger = logging.getLogger(__name__)

_CAPACITY = {"max_physical_sessions_without_weight_duplication": 1}


class _WorkerStats:
    __slots__ = ("num_prompt_tokens", "num_generated_tokens")

    def __init__(self, prompt_tokens: int, generated_tokens: int):
        self.num_prompt_tokens = prompt_tokens
        self.num_generated_tokens = generated_tokens


class WorkerRunner:
    """Drives the model worker subprocess over JSONL, exposing the RunnerPool
    generate() surface. One worker = one session; calls are serialized by a lock
    (and by RunnerPool's single slot). The control plane never executes CUDA."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._lock = threading.Lock()

    def reset(self) -> None:
        # The worker resets its session per request; nothing to do here.
        pass

    def stop(self) -> None:
        # Best-effort only: the worker request is synchronous and not
        # interruptible mid-generation in V1.
        pass

    def generate(self, prompt, config, token_callback=None, stats_callback=None):
        request = {
            "prompt": prompt,
            "max_new_tokens": getattr(config, "max_new_tokens", -1),
            "temperature": getattr(config, "temperature", 0.0),
        }
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Qwen worker exited (code {self._proc.returncode}); restart the server"
                )
            try:
                self._proc.stdin.write(json.dumps(request) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise RuntimeError("Qwen worker stdin is closed") from e

            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("Qwen worker exited mid-request")
                msg = json.loads(line)
                if "token" in msg:
                    if token_callback is not None:
                        token_callback(msg["token"])
                elif msg.get("done"):
                    if stats_callback is not None:
                        stats_callback(
                            _WorkerStats(
                                msg.get("prompt_tokens", 0),
                                msg.get("completion_tokens", 0),
                            )
                        )
                    return
                elif "error" in msg:
                    raise RuntimeError(f"Qwen worker error: {msg['error']}")


def _spawn_worker(args) -> subprocess.Popen:
    """Start the model worker subprocess and block until it reports ready."""
    env = dict(os.environ)
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        # The AOTI CUDA blob needs the conda libstdc++; forward it to the worker.
        env["LD_LIBRARY_PATH"] = f"{conda}/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        sys.executable,
        "-m",
        "executorch.examples.models.qwen3_5_moe.worker",
        "--model-path",
        args.model_path,
        "--tokenizer-path",
        args.tokenizer_path,
        "--hf-tokenizer",
        args.hf_tokenizer,
    ]
    if args.data_path:
        cmd += ["--data-path", args.data_path]
    if args.ext_dir:
        cmd += ["--ext-dir", args.ext_dir]

    logger.info("Starting Qwen worker subprocess (loads the model once)...")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    line = proc.stdout.readline()
    if not line:
        raise SystemExit("Qwen worker failed to start (no output; check stderr).")
    msg = json.loads(line)
    if not msg.get("ready"):
        raise SystemExit(f"Qwen worker did not report ready: {msg}")
    logger.info("Qwen worker ready; serving single-slot, concurrent requests queue.")
    return proc


def build_app_from_args(args):
    """Construct the FastAPI app + the model worker. Returns (app, model_id)."""
    default_template_kwargs = {"enable_thinking": False} if args.no_think else None
    template = ChatTemplate(
        args.hf_tokenizer, default_template_kwargs=default_template_kwargs
    )

    proc = _spawn_worker(args)
    worker_runner = WorkerRunner(proc)

    # tokenizer=None -> prefix cache disabled (Qwen seek() is NotSupported).
    # serving_capacity passed so the factory path is clamped to 1.
    pool = RunnerPool(
        runner_factory=lambda: worker_runner,
        num_runners=1,
        tokenizer=None,
        serving_capacity=_CAPACITY,
    )
    serving = ServingChat(
        pool,
        template,
        args.model_id,
        max_context=args.max_context,
        # Qwen3.5-MoE emits the XML <function=…><parameter=…> tool format.
        tool_detector_cls=QwenFunctionCallDetector,
    )

    from executorch.extension.llm.server.python.server import build_app

    app = build_app(serving, args.model_id)

    @app.on_event("shutdown")
    def _stop_worker():
        if proc.poll() is None:
            proc.terminate()

    return app, args.model_id


def main() -> None:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible LLM server for Qwen3.5 MoE (process-isolated, V1)"
    )
    p.add_argument("--model-path", required=True, help="Path to the .pte model")
    p.add_argument(
        "--data-path", default=None, help="Path to the .ptd CUDA delegate blob"
    )
    p.add_argument(
        "--tokenizer-path", required=True, help="Path to the HuggingFace tokenizer.json"
    )
    p.add_argument(
        "--hf-tokenizer",
        required=True,
        help="HF tokenizer id/dir for the model's chat template + encoding",
    )
    p.add_argument("--model-id", default="qwen3.5-moe")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Context window; prompts exceeding it are rejected with 400.",
    )
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Default reasoning off (enable_thinking=False).",
    )
    p.add_argument(
        "--num-runners",
        type=int,
        default=1,
        help="V1 supports 1 only (serving_capacity=1).",
    )
    p.add_argument(
        "--ext-dir",
        default=None,
        help="Directory with the built _qwen35_moe module (for the worker).",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.num_runners != 1:
        p.error(
            "Qwen3.5 MoE V1 is single-slot: serving_capacity=1. One worker serves "
            "one session; concurrent requests queue."
        )

    app, _ = build_app_from_args(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
