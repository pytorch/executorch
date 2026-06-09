# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible HTTP server for Qwen3.5 MoE (process-isolated).

This is the CONTROL PLANE only: FastAPI/uvicorn + OpenAI protocol, chat
templating, tool parsing, request validation. It runs NO CUDA model code and
imports no model pybind. Model execution lives in a separate C++ worker
process (qwen3_5_moe_worker) that this process drives over JSONL via the generic
WorkerClient — the same protocol the generic text_llm_worker speaks.

Why two processes: executing the AOTI CUDA model inside a live asyncio server
process segfaults in the int4 matmul (validated by elimination — the trigger is
CUDA execution while a live asyncio loop is resident). Isolating CUDA in a plain
(no-asyncio) C++ worker process is the reliable shape, and it loads weights once.

Sessions and constraints:
  * One worker hosts many isolated sessions on a single ~18GB weight load (CUDA
    per-session mutable rebinding); requests route by session_id (anonymous
    requests share a scratch session). See --max-sessions.
  * Execution is synchronous: one in-flight request at a time, concurrent HTTP
    requests queue. Sessions provide isolation, not concurrent throughput.
  * Warm append-only resume is on by default (--warm-resume): a named session
    reuses its resident context across turns when the prompt is an exact-token
    extension, including tool-call turns via token-ID prompt segments. Anonymous
    (scratch) requests always reset.
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
import logging
import os
from pathlib import Path

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.serving_chat import ServingChat
from executorch.extension.llm.server.python.session_runtime import SessionRuntime
from executorch.extension.llm.server.python.tool_parsers import QwenFunctionCallDetector
from executorch.extension.llm.server.python.worker_client import spawn_worker

logger = logging.getLogger(__name__)


def _default_worker_bin() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(
        repo_root
        / "cmake-out"
        / "examples"
        / "models"
        / "qwen3_5_moe"
        / "qwen3_5_moe_worker"
    )


def _spawn(args):
    """Spawn the C++ Qwen worker and return a ready WorkerClient."""
    env = dict(os.environ)
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        # The AOTI CUDA blob needs the conda libstdc++; forward it to the worker.
        env["LD_LIBRARY_PATH"] = f"{conda}/lib:" + env.get("LD_LIBRARY_PATH", "")
    worker_bin = args.worker_bin or _default_worker_bin()
    cmd = [
        worker_bin,
        "--model_path",
        args.model_path,
        "--tokenizer_path",
        args.tokenizer_path,
    ]
    if args.data_path:
        cmd += ["--data_path", args.data_path]
    cmd += ["--max_sessions", str(args.max_sessions)]
    cmd += [f"--warm_resume={'true' if args.warm_resume else 'false'}"]
    logger.info("Starting Qwen worker subprocess (loads the model once)...")
    return spawn_worker(cmd, env=env)


def build_app_from_args(args):
    """Construct the FastAPI app + the model worker. Returns (app, model_id)."""
    default_template_kwargs = {"enable_thinking": False} if args.no_think else None
    template = ChatTemplate(
        args.hf_tokenizer, default_template_kwargs=default_template_kwargs
    )

    worker = _spawn(args)  # one worker, weights once, many isolated sessions
    runtime = SessionRuntime(worker)
    serving = ServingChat(
        runtime,
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
        runtime.close_worker()

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
        help="HF tokenizer id/dir for the model's chat template",
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
        help="Workers (processes). 1 only: a worker hosts many isolated sessions "
        "on one weight load; more workers would duplicate the ~18GB weights.",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=1,
        help="Isolated sessions the one worker hosts on a single weight load "
        "(CUDA per-session mutable rebinding); clamped to 1 if the backend "
        "cannot rebind. One slot is reserved for anonymous requests, so the "
        "number of addressable session_ids is max-sessions - 1.",
    )
    p.add_argument(
        "--warm-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm append-only resume for named sessions: a request whose tokens "
        "extend the session's resident context prefills only the suffix. "
        "--no-warm-resume resets every request (for A/B measurement).",
    )
    p.add_argument(
        "--worker-bin",
        default=None,
        help="Path to the qwen3_5_moe_worker binary "
        "(default: cmake-out/examples/models/qwen3_5_moe/qwen3_5_moe_worker).",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.num_runners != 1:
        p.error(
            "Only 1 worker process is supported (it hosts many isolated sessions "
            "on one ~18GB weight load); more workers would duplicate the weights."
        )

    app, _ = build_app_from_args(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
