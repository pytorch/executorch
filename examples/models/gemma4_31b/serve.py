# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible HTTP server for Gemma 4 31B.

The server process is Python control plane only. It handles HTTP, Gemma chat
templating, tool parsing, and request validation. Model execution lives in the
C++ `gemma4_31b_worker` process and is driven over the generic
`examples/llm_server` JSONL protocol.
"""

import argparse
import logging
import os
import re
from pathlib import Path

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.llm_server.python.tool_parsers import (
    GemmaToolCallDetector,
    HermesDetector,
    QwenFunctionCallDetector,
)
from executorch.examples.llm_server.python.worker_client import spawn_worker

logger = logging.getLogger(__name__)

_GEMMA_CHANNEL_SPECIALS = {"<|channel>", "<channel|>", "<|think|>"}
_GEMMA_CHANNEL_BLOCK = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)


def _strip_gemma_channels(text: str) -> str:
    text = _GEMMA_CHANNEL_BLOCK.sub("", text)
    open_idx = text.find("<|channel>")
    if open_idx != -1:
        text = text[:open_idx]
    return text.replace("<channel|>", "").replace("<|think|>", "").strip()


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
                path / "examples" / "models" / "gemma4_31b"
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
        / "gemma4_31b"
        / "gemma4_31b_worker"
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
        "--max_sessions",
        str(args.max_sessions),
        f"--warm_resume={'true' if args.warm_resume else 'false'}",
        "--bos_id",
        str(args.bos_id),
        "--eos_id",
        str(args.eos_id),
    ]
    if args.data_path:
        cmd += ["--data_path", args.data_path]
    logger.info("Starting Gemma4 31B worker subprocess (loads the model once).")
    return spawn_worker(cmd, env=env)


def _tool_detector(name: str):
    if name == "gemma":
        return GemmaToolCallDetector
    if name == "hermes":
        return HermesDetector
    if name == "qwen":
        return QwenFunctionCallDetector
    if name == "none":
        return None
    raise ValueError(f"unknown tool parser: {name}")


def build_app_from_args(args):
    template = ChatTemplate(
        args.hf_tokenizer,
        assistant_header="<|turn>model\n",
        strip_rendered_bos=True,
        append_generation_prompt_after_tool_response=True,
    )
    worker = _spawn(args)
    runtime = SessionRuntime(worker)
    serving = ServingChat(
        runtime,
        template,
        args.model_id,
        max_context=args.max_context,
        tool_detector_cls=_tool_detector(args.tool_parser),
        prompt_token_offset=1,
        content_filter=_strip_gemma_channels,
        content_filter_specials=_GEMMA_CHANNEL_SPECIALS,
    )
    app = build_app(serving, args.model_id)

    @app.on_event("shutdown")
    def _stop_worker():
        runtime.close_worker()

    return app, args.model_id


def main() -> None:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible LLM server for Gemma 4 31B"
    )
    p.add_argument("--model-path", required=True, help="Path to the .pte model")
    p.add_argument("--data-path", default=None, help="Path to the delegate blob")
    p.add_argument("--tokenizer-path", required=True, help="Path to tokenizer.json")
    p.add_argument(
        "--hf-tokenizer",
        required=True,
        help="HF tokenizer id/dir for the model's chat template",
    )
    p.add_argument("--model-id", default="gemma4_31b")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-context", type=int, default=None)
    p.add_argument(
        "--num-runners",
        type=int,
        default=1,
        help="Worker processes. 1 only; more would duplicate the weights.",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=1,
        help="Physical sessions the worker can host on one weight load. One "
        "slot is reserved for anonymous requests, so addressable named "
        "sessions are max-sessions - 1.",
    )
    p.add_argument(
        "--warm-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm append-only resume for named sessions when available.",
    )
    p.add_argument(
        "--tool-parser",
        choices=("gemma", "hermes", "qwen", "none"),
        default="gemma",
        help="Tool-call format parser to apply to model output.",
    )
    p.add_argument(
        "--bos-id",
        type=int,
        default=2,
        help="BOS token id to prepend in the worker. The launcher strips the "
        "HF template's literal BOS before C++ tokenization.",
    )
    p.add_argument("--eos-id", type=int, default=1)
    p.add_argument(
        "--worker-bin",
        default=None,
        help="Path to gemma4_31b_worker. Defaults to the CMake output path.",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.num_runners != 1:
        p.error("Only 1 worker process is supported; more would duplicate weights.")

    app, _ = build_app_from_args(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
