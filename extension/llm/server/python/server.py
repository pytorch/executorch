# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible HTTP server for ExecuTorch LLMs.

Point any OpenAI-compatible agent harness (pi, opencode, ...) at
``http://<host>:<port>/v1``.

This process is the CONTROL PLANE only: FastAPI/uvicorn + OpenAI protocol, chat
templating, tool parsing, request validation. It runs NO model code and imports
no runtime pybind. Model execution lives in a separate C++ worker process
(``text_llm_worker``) driven over JSONL via WorkerClient.

V1 is single-slot: one worker hosts one session, so concurrent requests queue.
There is no prefix cache in V1 serving; caching, if it returns, lives inside the
worker/session, not the control plane.

Example:
    python -m executorch.extension.llm.server.python.server \\
        --model-path model.pte --tokenizer-path tokenizer.bin \\
        --hf-tokenizer Qwen/Qwen2.5-Coder-7B-Instruct --model-id qwen2.5-coder
"""

import argparse
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from .chat_template import ChatTemplate
from .errors import APIError
from .protocol import ChatCompletionRequest, ModelCard, ModelList
from .runner_pool import RunnerPool
from .serving_chat import ServingChat
from .tool_parsers import HermesDetector
from .worker_client import spawn_worker

logger = logging.getLogger(__name__)


def _default_worker_bin() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    return str(
        repo_root
        / "cmake-out"
        / "extension"
        / "llm"
        / "server"
        / "cpp"
        / "text_llm_worker"
    )


def _spawn(args):
    """Spawn the C++ text_llm_worker and return a ready WorkerClient."""
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
    ]
    logger.info("Starting model worker subprocess (loads the model once)...")
    return spawn_worker(cmd, env=env)


def build_app(serving: ServingChat, model_id: str) -> FastAPI:
    app = FastAPI(title="ExecuTorch LLM Server")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=model_id)])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        # Typed param → FastAPI validates the body and returns 422 on bad input.
        # APIError (e.g. context_length_exceeded) → structured 4xx/5xx, never a
        # dropped connection. Mid-stream failures are handled inside the stream.
        try:
            result = await serving.create(req)
        except APIError as e:
            return JSONResponse(e.body(), status_code=e.status)
        if req.stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return JSONResponse(result.model_dump(exclude_none=True))

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="ExecuTorch OpenAI-compatible LLM server")
    p.add_argument(
        "--model-path",
        required=True,
        help="Path to the self-contained .pte model (external .ptd weights are not "
        "supported by the generic text worker; use a model-specific launcher).",
    )
    p.add_argument("--tokenizer-path", required=True, help="Path to the tokenizer")
    p.add_argument(
        "--hf-tokenizer",
        default=None,
        help="HF tokenizer id/dir for model-correct chat templating (required unless "
        "--allow-chatml-fallback).",
    )
    p.add_argument(
        "--allow-chatml-fallback",
        action="store_true",
        help="Allow approximate generic ChatML templating when --hf-tokenizer is absent. "
        "Off by default: the fallback can't reproduce model-specific controls.",
    )
    p.add_argument(
        "--model-id", default="executorch", help="Model id reported on /v1/models"
    )
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Default reasoning off (sends enable_thinking=False to the chat template, "
        "e.g. Qwen3). Per-request chat_template_kwargs still override this.",
    )
    p.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Model context window; if set (and a tokenizer is available), prompts that "
        "exceed it are rejected with 400 context_length_exceeded instead of failing mid-generation. "
        "Set this to match the value used at export.",
    )
    p.add_argument(
        "--num-runners",
        type=int,
        default=1,
        help="V1 supports 1 only (single-slot: one worker serves one session).",
    )
    p.add_argument(
        "--worker-bin",
        default=None,
        help="Path to the text_llm_worker binary "
        "(default: cmake-out/extension/llm/server/cpp/text_llm_worker).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.num_runners != 1:
        p.error(
            "V1 is single-slot: one worker serves one session; concurrent requests queue."
        )

    default_template_kwargs = {"enable_thinking": False} if args.no_think else None
    # Requires --hf-tokenizer unless --allow-chatml-fallback (raises otherwise).
    template = ChatTemplate(
        args.hf_tokenizer,
        default_template_kwargs=default_template_kwargs,
        allow_fallback=args.allow_chatml_fallback,
    )
    worker = _spawn(args)  # one worker == one session (single-slot V1)
    pool = RunnerPool([worker])
    serving = ServingChat(
        pool,
        template,
        args.model_id,
        max_context=args.max_context,
        # Hermes JSON is the generic default; a model-specific server (e.g. a
        # Qwen launcher) selects the Qwen XML detector instead.
        tool_detector_cls=HermesDetector,
    )

    app = build_app(serving, args.model_id)

    @app.on_event("shutdown")
    def _stop_worker():
        pool.close()

    import uvicorn  # imported here so build_app() is usable without the ASGI server

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
