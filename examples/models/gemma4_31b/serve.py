# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible HTTP server for Gemma 4 31B on CUDA."""

import argparse
import logging
import os
from pathlib import Path

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.serving_chat import ServingChat
from executorch.extension.llm.server.python.session_runtime import SessionRuntime
from executorch.extension.llm.server.python.tool_parsers import (
    HermesDetector,
    QwenFunctionCallDetector,
)
from executorch.extension.llm.server.python.worker_client import spawn_worker

logger = logging.getLogger(__name__)


def _default_worker_bin() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(
        repo_root
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
        env["LD_LIBRARY_PATH"] = f"{conda}/lib:" + env.get("LD_LIBRARY_PATH", "")
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
    logger.info("Starting Gemma4 31B worker subprocess...")
    return spawn_worker(cmd, env=env)


def _tool_detector(name: str):
    if name == "hermes":
        return HermesDetector
    if name == "qwen":
        return QwenFunctionCallDetector
    if name == "none":
        return None
    raise ValueError(f"unknown tool parser: {name}")


def build_app_from_args(args):
    template = ChatTemplate(args.hf_tokenizer)
    worker = _spawn(args)
    runtime = SessionRuntime(worker)
    serving = ServingChat(
        runtime,
        template,
        args.model_id,
        max_context=args.max_context,
        tool_detector_cls=_tool_detector(args.tool_parser),
        prompt_token_offset=1,
    )

    from executorch.extension.llm.server.python.server import build_app

    app = build_app(serving, args.model_id)

    @app.on_event("shutdown")
    def _stop_worker():
        runtime.close_worker()

    return app, args.model_id


def main() -> None:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible CUDA LLM server for Gemma 4 31B"
    )
    p.add_argument("--model-path", required=True, help="Path to the .pte model")
    p.add_argument("--data-path", default=None, help="Path to the .ptd delegate blob")
    p.add_argument("--tokenizer-path", required=True, help="Path to the tokenizer.json")
    p.add_argument(
        "--hf-tokenizer",
        required=True,
        help="HF tokenizer id/dir for the model's chat template",
    )
    p.add_argument("--model-id", default="gemma4-31b")
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
        help="Isolated sessions the CUDA worker may host when the export has "
        "mutable-buffer metadata.",
    )
    p.add_argument(
        "--warm-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm append-only resume for named sessions when available.",
    )
    p.add_argument(
        "--tool-parser",
        choices=("hermes", "qwen", "none"),
        default="hermes",
        help="Tool-call format parser to apply to model output.",
    )
    p.add_argument("--bos-id", type=int, default=2)
    p.add_argument("--eos-id", type=int, default=1)
    p.add_argument(
        "--worker-bin",
        default=None,
        help="Path to the gemma4_31b_worker binary.",
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
