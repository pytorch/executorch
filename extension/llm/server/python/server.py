# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible HTTP server for ExecuTorch LLMs.

Point any OpenAI-compatible agent harness (pi, opencode, ...) at
``http://<host>:<port>/v1``.

Example:
    python -m executorch.extension.llm.server.python.server \\
        --model-path model.pte --tokenizer-path tokenizer.bin \\
        --hf-tokenizer Qwen/Qwen2.5-Coder-7B-Instruct --model-id qwen2.5-coder
"""

import argparse
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from .chat_template import ChatTemplate
from .errors import APIError
from .protocol import ChatCompletionRequest, ModelCard, ModelList
from .runner_pool import RunnerPool
from .serving_chat import ServingChat
from .tool_parsers import HermesDetector

logger = logging.getLogger(__name__)


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
    p.add_argument("--model-path", required=True, help="Path to the .pte model")
    p.add_argument("--tokenizer-path", required=True, help="Path to the tokenizer")
    p.add_argument("--data-path", default=None, help="Optional .ptd weights file")
    p.add_argument(
        "--hf-tokenizer",
        default=None,
        help="HF tokenizer id/dir for model-correct chat templating (required unless "
        "--allow-chatml-fallback). Also required for --enable-prefix-cache.",
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
        "--num-runners", type=int, default=1, help="KV-cache instances (N x memory)"
    )
    p.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        help="Enable conservative per-runner turn-to-turn KV prefix reuse. Off by default; "
        "requires --hf-tokenizer and a non-sliding-window model (falls back to full prefill "
        "on any reuse failure).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.enable_prefix_cache and not args.hf_tokenizer:
        p.error(
            "--enable-prefix-cache requires --hf-tokenizer (token-level prefix matching)."
        )

    default_template_kwargs = {"enable_thinking": False} if args.no_think else None
    # Requires --hf-tokenizer unless --allow-chatml-fallback (raises otherwise).
    template = ChatTemplate(
        args.hf_tokenizer,
        default_template_kwargs=default_template_kwargs,
        allow_fallback=args.allow_chatml_fallback,
    )
    cache_tokenizer = template.tokenizer() if args.enable_prefix_cache else None
    if cache_tokenizer is not None:
        logger.info("KV prefix caching enabled (conservative, per-runner).")
    pool = RunnerPool(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        num_runners=args.num_runners,
        tokenizer=cache_tokenizer,
    )
    serving = ServingChat(
        pool,
        template,
        args.model_id,
        max_context=args.max_context,
        tool_detector_cls=HermesDetector,
    )

    import uvicorn  # imported here so build_app() is usable without the ASGI server

    uvicorn.run(build_app(serving, args.model_id), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
