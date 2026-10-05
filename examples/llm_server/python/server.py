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
no runtime pybind. Model execution lives in a separate C++ worker process driven
over JSONL via WorkerClient.

One worker process: explicit multiplexing permits bounded concurrent requests;
legacy workers serialize execution. Session capacity is set by the worker/engine.
Extra worker processes would duplicate weights, so `--num-runners` accepts 1.

Example:
    python -m executorch.examples.llm_server.python.server \\
        --worker-bin /path/to/model_worker \\
        --model-path model.pte --tokenizer-path tokenizer.bin \\
        --hf-tokenizer Qwen/Qwen2.5-Coder-7B-Instruct --model-id qwen2.5-coder
"""

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from typing import AsyncContextManager, Awaitable, Callable, Optional

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from .chat_template import ChatTemplate
from .errors import APIError
from .protocol import ChatCompletionRequest, ModelCard, ModelList
from .request_body import BoundedChatBody
from .serving_chat import ServingChat
from .session_runtime import SessionRuntime
from .tool_parsers import HermesDetector
from .worker_client import spawn_worker

logger = logging.getLogger(__name__)


def _spawn(args):
    """Spawn the C++ model worker and return a ready WorkerClient."""
    env = dict(os.environ)
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        env["LD_LIBRARY_PATH"] = f"{conda}/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        args.worker_bin,
        "--model_path",
        args.model_path,
        "--tokenizer_path",
        args.tokenizer_path,
    ]
    logger.info("Starting model worker subprocess (loads the model once)...")
    return spawn_worker(cmd, env=env)


def _resolve_session_id(
    req: ChatCompletionRequest,
    x_executorch_session_id: Optional[str],
    session_id_header: Optional[str],
    x_session_affinity: Optional[str],
) -> Optional[str]:
    # Session id precedence: body field wins, else the X-ExecuTorch-Session-ID /
    # session_id / x-session-affinity headers (in that order). Aliases let clients
    # that already emit a stable per-conversation id for cache affinity (e.g. pi's
    # sendSessionAffinityHeaders) route to a session with no extra config.
    if req.session_id is not None:
        return req.session_id
    return x_executorch_session_id or session_id_header or x_session_affinity


async def _session_op(
    op: Callable[[str], Awaitable[None]], session_id: str, ok: dict
) -> JSONResponse:
    # Shared shape for the session vendor-extension routes (close/reset): run the
    # operation, mapping APIError to a structured JSON error.
    try:
        await op(session_id)
    except APIError as e:
        return JSONResponse(e.body(), status_code=e.status)
    return JSONResponse(ok)


async def _create_with_disconnect(
    request: Request, serving: ServingChat, req: ChatCompletionRequest
):
    async def wait_disconnect():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    creation = asyncio.create_task(serving.create(req))
    disconnect = asyncio.create_task(wait_disconnect())
    transferred = False

    async def cleanup():
        disconnect.cancel()
        creation.cancel()
        result, _ = await asyncio.gather(creation, disconnect, return_exceptions=True)
        if req.stream and not transferred and not isinstance(result, BaseException):
            await result.aclose()

    try:
        done, _ = await asyncio.wait(
            (creation, disconnect), return_when=asyncio.FIRST_COMPLETED
        )
        if disconnect in done:
            raise ClientDisconnect()
        result = await creation
        transferred = True
        return result
    finally:
        await SessionRuntime._finish_cleanup(asyncio.create_task(cleanup()))


class _ClosingStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await SessionRuntime._finish_cleanup(
                asyncio.create_task(self.body_iterator.aclose())
            )


def _create_serving_app(serving, serving_factory):
    if (serving is None) == (serving_factory is None):
        raise ValueError("exactly one of serving or serving_factory must be supplied")
    if serving_factory is not None and not callable(serving_factory):
        raise TypeError("serving_factory must be an async context manager factory")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            async with serving_factory() as serving:
                if serving is None:
                    raise RuntimeError("serving_factory yielded no serving adapter")
                app.state.serving = serving
                try:
                    yield
                finally:
                    app.state.serving = None
        finally:
            app.state.serving = None

    if serving_factory is None:
        app = FastAPI(title="ExecuTorch LLM Server")
    else:
        app = FastAPI(title="ExecuTorch LLM Server", lifespan=lifespan)
    app.state.serving = serving
    return app


def _current_serving(app: FastAPI) -> ServingChat:
    serving = app.state.serving
    if serving is None:
        raise RuntimeError(
            "serving adapter is unavailable; run the application lifespan "
            "before requests and do not use it after shutdown"
        )
    return serving


def build_app(
    serving: Optional[ServingChat],
    model_id: str,
    *,
    serving_factory: Optional[Callable[[], AsyncContextManager[ServingChat]]] = None,
) -> FastAPI:
    """Build an app with either an injected adapter or a lifespan-owned factory.

    The factory owns partial-startup cleanup and awaited runtime shutdown. Direct
    instances retain FastAPI's default lifespan and are not closed by this app.
    """
    app = _create_serving_app(serving, serving_factory)
    app.add_middleware(
        BoundedChatBody,
        # Resolve at request time: lifespan factories populate app.state only
        # after build_app. Legacy adapters keep their existing body limits.
        enabled=lambda: getattr(app.state.serving, "uses_native_transport", False)
        is True,
    )

    @app.get("/health")
    async def health():
        if not _current_serving(app).healthy:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=model_id)])

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        request: Request,
        # FastAPI dependency: the Header() call in the default is required.
        # `session_id` is matched verbatim (underscore).
        x_executorch_session_id: Optional[str] = Header(default=None),  # noqa: B008
        session_id_header: Optional[str] = Header(  # noqa: B008
            default=None, alias="session_id"
        ),
        x_session_affinity: Optional[str] = Header(default=None),  # noqa: B008
    ):
        # Typed param → FastAPI validates the body and returns 422 on bad input.
        # APIError (e.g. context_length_exceeded) → structured 4xx/5xx, never a
        # dropped connection. Mid-stream failures are handled inside the stream.
        req.session_id = _resolve_session_id(
            req, x_executorch_session_id, session_id_header, x_session_affinity
        )
        try:
            result = await _create_with_disconnect(request, _current_serving(app), req)
        except ClientDisconnect:
            return Response(status_code=499)
        except APIError as e:
            return JSONResponse(e.body(), status_code=e.status)
        if req.stream:
            return _ClosingStreamingResponse(result, media_type="text/event-stream")
        return JSONResponse(result.model_dump(exclude_none=True))

    @app.delete("/v1/sessions/{session_id}")
    async def close_session(session_id: str):
        # Free a named session's state + capacity slot (vendor extension).
        return await _session_op(
            _current_serving(app).close_session,
            session_id,
            {"closed": True, "session_id": session_id},
        )

    @app.post("/v1/sessions/{session_id}/reset")
    async def reset_session(session_id: str):
        # Replace a named session's context under its public ID (vendor extension).
        return await _session_op(
            _current_serving(app).reset_session,
            session_id,
            {"reset": True, "session_id": session_id},
        )

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
        "--assistant-header",
        default="<|im_start|>assistant\n",
        help="Exact template text ending the assistant generation header, including "
        "any trailing whitespace. Defaults to the ChatML header.",
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
        help="Worker processes. 1 only: one worker hosts many isolated sessions "
        "on a single weight load; more workers would duplicate the weights.",
    )
    p.add_argument(
        "--worker-bin",
        required=True,
        help="Path to a model worker binary that speaks the llm_server JSONL protocol "
        "and accepts --model_path / --tokenizer_path.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.num_runners != 1:
        p.error(
            "Only 1 worker process is supported (it hosts many isolated sessions "
            "on one weight load); more workers would duplicate the weights."
        )

    default_template_kwargs = {"enable_thinking": False} if args.no_think else None
    # Requires --hf-tokenizer unless --allow-chatml-fallback (raises otherwise).
    template = ChatTemplate(
        args.hf_tokenizer,
        default_template_kwargs=default_template_kwargs,
        allow_fallback=args.allow_chatml_fallback,
        assistant_header=args.assistant_header,
    )
    template.generation_preamble()
    worker = _spawn(args)  # one worker hosting many isolated sessions
    runtime = SessionRuntime(worker)
    serving = ServingChat(
        runtime,
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
        runtime.close_worker()

    import uvicorn  # imported here so build_app() is usable without the ASGI server

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
