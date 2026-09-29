# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible HTTP server for Qwen3.5 MoE.

The server process is Python control plane only. It handles OpenAI-compatible
HTTP, Qwen chat templating, tool parsing, and request validation. Model
execution lives in the C++ `qwen3_5_moe_worker` process and is driven over the
generic `examples/llm_server` JSONL protocol.
"""

import argparse
import logging
import os
from pathlib import Path

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.tool_parsers import QwenFunctionCallDetector
from executorch.examples.llm_server.python.worker_client import spawn_worker

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    starts = [Path(__file__).resolve(), Path(__file__).absolute(), Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in starts:
        cur = start if start.is_dir() else start.parent
        for path in (cur, *cur.parents):
            if path in seen:
                continue
            seen.add(path)
            if (path / "CMakeLists.txt").exists() and (
                path / "examples" / "models" / "qwen3_5_moe"
            ).is_dir():
                return path
    raise RuntimeError(
        "Could not locate the ExecuTorch source checkout; pass --worker-bin "
        "explicitly."
    )


def _default_worker_bin() -> str:
    return str(
        _repo_root()
        / "cmake-out"
        / "examples"
        / "models"
        / "qwen3_5_moe"
        / "qwen3_5_moe_worker"
    )


def _spawn(args):
    env = dict(os.environ)
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{conda}/lib:{existing}" if existing else f"{conda}/lib"
        )
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
    logger.info("Starting Qwen worker subprocess (loads the model once).")
    return spawn_worker(cmd, env=env)


def build_app_from_args(args):
    default_template_kwargs = {"enable_thinking": False} if args.no_think else None
    template = ChatTemplate(
        args.hf_tokenizer, default_template_kwargs=default_template_kwargs
    )

    worker = _spawn(args)
    runtime = SessionRuntime(worker)
    serving = ServingChat(
        runtime,
        template,
        args.model_id,
        max_context=args.max_context,
        tool_detector_cls=QwenFunctionCallDetector,
    )
    app = build_app(serving, args.model_id)

    @app.on_event("shutdown")
    def _stop_worker():
        runtime.close_worker()

    return app, args.model_id


def main() -> None:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible LLM server for Qwen3.5 MoE"
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
        help="Worker processes. 1 only: a worker hosts isolated sessions on one "
        "weight load; more workers would duplicate the weights.",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=1,
        help="Physical sessions the worker can host on one weight load. One slot "
        "is reserved for anonymous requests, so addressable named sessions are "
        "max-sessions - 1.",
    )
    p.add_argument(
        "--warm-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm append-only resume for named sessions. --no-warm-resume "
        "resets every request.",
    )
    p.add_argument(
        "--worker-bin",
        default=None,
        help="Path to qwen3_5_moe_worker. Defaults to the CMake output path.",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.num_runners != 1:
        p.error(
            "Only 1 worker process is supported; more workers would duplicate "
            "the model weights."
        )

    app, _ = build_app_from_args(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
