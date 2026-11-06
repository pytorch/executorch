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
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from .chat_template import ChatTemplate
from .errors import APIError
from .hud import install as install_hud
from .hud.metrics import record_error, Turn
from .protocol import ChatCompletionRequest, ModelCard, ModelList
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
    op: Callable[[str], Awaitable[None]],
    session_id: str,
    ok: dict,
    serving: ServingChat,
) -> JSONResponse:
    # Shared shape for the session vendor-extension routes (close/reset): run the
    # operation, mapping APIError to a structured JSON error.
    try:
        await op(session_id)
    except APIError as e:
        return JSONResponse(serving.error_body(e), status_code=e.status)
    return JSONResponse(ok)


async def _create_with_disconnect(
    request: Request,
    serving: ServingChat,
    req: ChatCompletionRequest,
    observer: Optional[Turn] = None,
):
    async def wait_disconnect():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    creation = asyncio.create_task(serving.create(req, observer=observer))
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
    def __init__(
        self, content, *, heartbeat_interval=15.0, metrics=None, turn=None, **kwargs
    ):
        self._source = content
        super().__init__(_with_heartbeats(content, heartbeat_interval), **kwargs)
        self.metrics = metrics
        self.turn = turn

    async def __call__(self, scope, receive, send):
        disconnected = False
        completed = False

        async def observed_receive():
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def observed_send(message):
            nonlocal completed
            await send(message)
            if message["type"] == "http.response.body" and (
                not message.get("more_body", False)
                or message.get("body", b"").endswith(b"data: [DONE]\n\n")
            ):
                # Clients may disconnect after [DONE], before iterator cleanup.
                completed = True

        try:
            await super().__call__(scope, observed_receive, observed_send)
        except (asyncio.CancelledError, ClientDisconnect):
            disconnected = True
            raise
        except Exception as error:
            record_error(self.turn, type(error).__name__)
            raise
        finally:
            try:
                await SessionRuntime._finish_cleanup(
                    asyncio.create_task(self.body_iterator.aclose())
                )
            finally:
                try:
                    await SessionRuntime._finish_cleanup(
                        asyncio.create_task(self._source.aclose())
                    )
                finally:
                    if self.turn is not None:
                        self.metrics.finish(
                            self.turn,
                            "cancelled" if disconnected and not completed else "done",
                        )


async def _with_heartbeats(stream, interval=15.0):
    chunks = asyncio.Queue(maxsize=1)
    ready = asyncio.Event()

    async def produce():
        try:
            async for chunk in stream:
                await chunks.put(chunk)
                ready.set()
        finally:
            await stream.aclose()

    producer = asyncio.create_task(produce())
    producer.add_done_callback(lambda _: ready.set())
    try:
        while True:
            ready.clear()
            try:
                chunk = chunks.get_nowait()
            except asyncio.QueueEmpty:
                if producer.done():
                    await producer
                    return
                try:
                    async with asyncio.timeout(interval):
                        await ready.wait()
                except TimeoutError:
                    yield ": keep-alive\n\n"
            else:
                yield chunk
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)


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


async def _chat_response(app, request, req):
    try:
        if req.session_id is not None:
            ServingChat._validate_session_id(req.session_id)
    except APIError as error:
        return JSONResponse(
            _current_serving(app).error_body(error), status_code=error.status
        )
    metrics = app.state.telemetry
    turn = metrics.begin(req.session_id) if metrics is not None else None
    transferred = False
    state = "done"
    try:
        result = await _create_with_disconnect(
            request, _current_serving(app), req, turn
        )
        if req.stream:
            response = _ClosingStreamingResponse(
                result, media_type="text/event-stream", metrics=metrics, turn=turn
            )
            transferred = True
            return response
        return JSONResponse(result.model_dump(exclude_none=True))
    except ClientDisconnect:
        state = "cancelled"
        return Response(status_code=499)
    except asyncio.CancelledError:
        state = "cancelled"
        raise
    except APIError as e:
        record_error(turn, e.code or e.err_type)
        return JSONResponse(_current_serving(app).error_body(e), status_code=e.status)
    except Exception as error:
        record_error(turn, type(error).__name__)
        raise
    finally:
        if turn is not None and not transferred:
            metrics.finish(turn, state)


def build_app(
    serving: Optional[ServingChat],
    model_id: str,
    *,
    serving_factory: Optional[Callable[[], AsyncContextManager[ServingChat]]] = None,
    model_path: Optional[str] = None,
    max_context: Optional[int] = None,
    max_concurrent_requests: Optional[int] = None,
    hud: bool = False,
    listen_address: Optional[str] = None,
) -> FastAPI:
    """Build an app with either an injected adapter or a lifespan-owned factory.

    The factory owns partial-startup cleanup and awaited runtime shutdown. Direct
    instances retain FastAPI's default lifespan and are not closed by this app.
    HUD-enabled launchers must capture worker stderr; embedders default to plain logs.
    """
    app = _create_serving_app(serving, serving_factory)
    app.state.telemetry = None
    if hud:
        install_hud(
            app,
            model_id,
            model_path=model_path,
            context=max_context,
            capacity=max_concurrent_requests,
            listen_address=listen_address,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        api_error = APIError(
            422, f"{location}: {first['msg']}", "invalid_request_error", "invalid_value"
        )
        return JSONResponse(
            _current_serving(app).error_body(api_error), status_code=422
        )

    @app.get("/health")
    async def health():
        if not _current_serving(app).healthy:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return {"status": "ok"}

    @app.get("/v1/models", response_model_exclude_none=True)
    async def list_models() -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=model_id,
                    context_length=_current_serving(app).max_context,
                )
            ]
        )

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
        return await _chat_response(app, request, req)

    @app.delete("/v1/sessions/{session_id}")
    async def close_session(session_id: str):
        # Free a named session's state + capacity slot (vendor extension).
        closed = False
        try:
            serving = _current_serving(app)
            response = await _session_op(
                serving.close_session,
                session_id,
                {"closed": True, "session_id": session_id},
                serving,
            )
            closed = response.status_code == 200
            return response
        finally:
            if app.state.telemetry is not None:
                if closed:
                    app.state.telemetry.close_session(session_id)
                else:
                    app.state.telemetry.reset_context(session_id, None)

    @app.post("/v1/sessions/{session_id}/reset")
    async def reset_session(session_id: str):
        # Replace a named session's context under its public ID (vendor extension).
        context = None
        try:
            serving = _current_serving(app)
            response = await _session_op(
                serving.reset_session,
                session_id,
                {"reset": True, "session_id": session_id},
                serving,
            )
            if response.status_code == 200:
                context = 0
            return response
        finally:
            if app.state.telemetry is not None:
                app.state.telemetry.reset_context(session_id, context)

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
