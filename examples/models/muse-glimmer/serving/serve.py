# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenAI-compatible HTTP server for Muse Glimmer.

The Python control plane drives ``muse_glimmer_worker`` over JSONL.
"""

import argparse
import base64
import logging
import os
import re
from pathlib import Path

from executorch.examples.llm_server.python.errors import APIError, ContextLengthExceeded
from executorch.examples.llm_server.python.hud.metrics import Turn
from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat, ToolContext
from executorch.examples.llm_server.python.session_runtime import (
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.worker_client import spawn_worker
from executorch.examples.models.muse_glimmer.serving.chat_template import (
    MuseGlimmerChatTemplate as ChatTemplate,
    resolve_tool_result_names,
)
from executorch.examples.models.muse_glimmer.serving.framing import (
    ASSISTANT_HEADER as _ASSISTANT_HEADER,
    body_spans,
    clean_body,
    CONTROLS,
    recipient_bodies,
)
from executorch.examples.models.muse_glimmer.serving.tool_parsers import (
    AtemToolCallDetector,
)
from executorch.examples.models.muse_glimmer.serving.tool_parsers.atem import (
    parameter_value_spans,
)
from executorch.examples.models.muse_glimmer.serving.tool_parsers.schema import (
    ToolOutputBudget,
    ToolSchema,
)

logger = logging.getLogger(__name__)

# The template ends at this prefix; the model emits the recipient and message
# delimiter.
_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_ERROR_CODE = "invalid_image"
_IMAGE_PLACEHOLDER = "\x00muse_glimmer_image_placeholder\x00"
_IMAGE_DATA_URL_RE = re.compile(
    r"\Adata:(image/(?:jpeg|png));base64,([A-Za-z0-9+/]*={0,2})\Z"
)

# Harmony messages may recur within one assistant turn. The framing parser
# handles their delimiters without interpreting quoted headers inside a body.
_MUSE_GLIMMER_HEADER_SPECIALS = set(CONTROLS)


def _extract_muse_glimmer_reasoning(text: str) -> tuple[str | None, str]:
    """Split a Harmony turn into private `to=self` bodies and visible messages.

    Nonempty thinking bodies retain their whitespace and are joined with a
    newline without adding channel framing. Stray controls are removed while
    quoted headers remain literal text. Warm resume compares supplied reasoning
    against the returned text before splicing stored generated token ids, preserving the original
    channel framing for unchanged echoes or omitted/null reasoning.
    """
    protected = parameter_value_spans(text)
    bodies = list(recipient_bodies(text, protected))
    matches = [header for header, _, _ in bodies]
    if not matches:
        return None, "".join(
            text[a:b] for a, b in body_spans(text, protected_spans=protected)
        )

    reasoning: list[str] = []
    visible: list[str] = []
    prefix = clean_body(text[: matches[0].start()]).strip()
    if prefix:
        visible.append(prefix)

    for match, start, end in bodies:
        body = "".join(text[a:b] for a, b in body_spans(text, start, end, protected))
        if not body.strip():
            continue
        if match.group(1) == "self":
            reasoning.append(body)
        else:
            visible.append(body.strip())

    return "\n".join(reasoning) or None, "\n\n".join(visible)


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
                path / "examples" / "models" / "muse-glimmer"
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
        / "muse-glimmer"
        / "muse_glimmer_worker"
    )


def _invalid_image(message: str) -> APIError:
    return APIError(400, message, "invalid_request_error", _IMAGE_ERROR_CODE)


def _normalize_image_request(  # noqa: C901
    req: ChatCompletionRequest, max_image_bytes: int
) -> tuple[ChatCompletionRequest, dict | None]:
    """Clone the request, replace its image with a marker, and return its payload."""
    normalized = req.model_copy(deep=True)
    image_segment = None

    for message in normalized.messages:
        if not isinstance(message.content, list):
            continue
        text_parts = []
        for part in message.content:
            if part.get("type") != "image_url" and "image_url" not in part:
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                continue
            if image_segment is not None:
                raise _invalid_image(
                    "Only one image_url part is supported per request."
                )
            if set(part) != {"type", "image_url"} or part.get("type") != "image_url":
                raise _invalid_image(
                    "image_url parts must contain only type and image_url."
                )
            image_url = part.get("image_url")
            if not isinstance(image_url, dict) or set(image_url) != {"url"}:
                raise _invalid_image("image_url must be an object containing only url.")
            url = image_url.get("url")
            if not isinstance(url, str):
                raise _invalid_image("image_url.url must be a data URL string.")
            match = _IMAGE_DATA_URL_RE.fullmatch(url)
            if match is None:
                raise _invalid_image(
                    "image_url.url must be a base64 JPEG or PNG data URL."
                )
            mime_type, data = match.groups()
            if not data:
                raise _invalid_image("Image data must not be empty.")
            try:
                decoded = base64.b64decode(data, validate=True)
            except ValueError as error:
                raise _invalid_image("Image data is not valid base64.") from error
            if not decoded:
                raise _invalid_image("Image data must not be empty.")
            if len(decoded) > max_image_bytes:
                raise _invalid_image(
                    f"Decoded image exceeds the {max_image_bytes}-byte limit."
                )
            image_segment = {
                "image": {
                    "encoding": "base64",
                    "mime_type": mime_type,
                    "data": data,
                }
            }
            text_parts.append(_IMAGE_PLACEHOLDER)
        # Muse Glimmer's HF template expects string content rather than OpenAI's
        # structured content-part list. The marker preserves the image's position
        # until the rendered Harmony prompt is split into worker segments.
        message.content = "".join(text_parts)

    return normalized, image_segment


class MuseGlimmerServingChat(ServingChat):
    def __init__(
        self, *args, max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._max_image_bytes = max_image_bytes

    def _visible_content(self, text: str) -> str:
        # The recipient-aware extractor already removed framing. Parsing the
        # resulting bodies again would reinterpret a quoted header at offset 0.
        cut = min(
            (
                text.find(token)
                for token in self._content_specials
                if token not in CONTROLS and token in text
            ),
            default=len(text),
        )
        return text[:cut].strip()

    @staticmethod
    def _with_image(prompt: PromptInput, image_segment: dict | None) -> PromptInput:
        if image_segment is None:
            return prompt
        segments = (
            [{"text": prompt.text}]
            if prompt.text is not None
            else list(prompt.segments or [])
        )
        output = []
        found = False
        for segment in segments:
            text = segment.get("text")
            if not isinstance(text, str) or _IMAGE_PLACEHOLDER not in text:
                output.append(segment)
                continue
            if found or text.count(_IMAGE_PLACEHOLDER) != 1:
                raise RuntimeError(
                    "rendered prompt must contain exactly one image marker"
                )
            before, after = text.split(_IMAGE_PLACEHOLDER)
            if before:
                output.append({"text": before})
            output.append(image_segment)
            if after:
                output.append({"text": after})
            found = True
        if not found:
            raise RuntimeError("rendered prompt is missing its image marker")
        return PromptInput(segments=output)

    def _count_prompt_tokens(self, prompt: PromptInput) -> int | None:
        if prompt.text is not None:
            return super()._count_prompt_tokens(prompt)
        if any("image" in segment for segment in prompt.segments or []):
            # Only native preparation knows the image-expanded decoder positions.
            return None
        return super()._count_prompt_tokens(prompt)

    def _parse_tools(
        self, req, text, finish_reason, raw_text, context, request_stopped
    ):
        if self._tool_detector_cls is not None and issubclass(
            self._tool_detector_cls, AtemToolCallDetector
        ):
            parsed = self._tool_detector_cls().detect_and_parse(
                raw_text,
                context.schemas,
                finish_reason=finish_reason,
                truncated=request_stopped,
            )
            parsed.normal_text = self._split_reasoning(parsed.normal_text)[1]
            return parsed
        return super()._parse_tools(
            req, text, finish_reason, raw_text, context, request_stopped
        )

    @staticmethod
    def _validate_reasoning_options(reasoning):
        unsupported = set(reasoning) - {"effort", "enabled", "exclude"}
        if unsupported:
            raise APIError(
                400,
                "Unsupported reasoning keys: " + ", ".join(sorted(unsupported)),
                "invalid_request_error",
                "unsupported_parameter",
            )
        for key in ("enabled", "exclude"):
            if reasoning.get(key) is not None and not isinstance(reasoning[key], bool):
                raise APIError(
                    400,
                    f"reasoning.{key} must be a boolean",
                    "invalid_request_error",
                    "invalid_value",
                )
        if reasoning.get("enabled") is False:
            raise APIError(
                400,
                "Muse does not support disabling reasoning generation",
                "invalid_request_error",
                "unsupported_parameter",
            )

    def _normalize_reasoning(self, req):
        if req.reasoning is None and req.reasoning_effort is None:
            return req
        reasoning = req.reasoning or {}
        self._validate_reasoning_options(reasoning)
        effort = reasoning.get("effort")
        if effort is None:
            effort = req.reasoning_effort
        if effort is not None and effort not in ("low", "medium", "high"):
            raise APIError(
                400,
                "Muse reasoning effort must be low, medium or high",
                "invalid_request_error",
                "invalid_value",
            )
        kwargs = dict(req.chat_template_kwargs or {})
        if effort is not None:
            if (
                req.reasoning_effort not in (None, effort)
                or kwargs.get("reasoning_strength", effort) != effort
            ):
                raise APIError(
                    400,
                    "Conflicting reasoning effort controls",
                    "invalid_request_error",
                    "invalid_value",
                )
            kwargs["reasoning_strength"] = effort
        return req.model_copy(
            update={
                "reasoning": None,
                "reasoning_effort": None,
                "chat_template_kwargs": kwargs,
            }
        )

    async def _create(
        self, req: ChatCompletionRequest, lease, admission, observer: Turn | None = None
    ):
        # The inherited create() owns the transaction around preparation and
        # response cleanup; only image-aware prompt construction differs here.
        req = self._normalize_reasoning(req)
        self._reject_unsupported_params(req)
        tool_context = ToolContext()
        if self._tools_active(req):
            budget = ToolOutputBudget()
            tool_context.schemas = {
                name: ToolSchema(schema, budget=budget)
                for name, schema in self._tool_schemas(req).items()
            }
        if req.session_id is not None:
            self._validate_session_id(req.session_id)

        history_req = req
        req, image_segment = _normalize_image_request(req, self._max_image_bytes)
        req = req.model_copy(
            update={"messages": resolve_tool_result_names(req.messages)}
        )
        template_tools = None if req.tool_choice == "none" else req.tools
        template_kwargs = dict(req.chat_template_kwargs or {})
        template_kwargs.pop("return_reasoning", None)
        prompt = self._template.render(
            req.messages, tools=template_tools, template_kwargs=template_kwargs
        )
        prompt_input = self._transcript.build_prompt_input(
            session_id=req.session_id,
            messages=req.messages,
            rendered_prompt=prompt,
            tools=template_tools,
            template_kwargs=template_kwargs,
            history_messages=history_req.messages,
        )
        prompt_input = self._with_image(prompt_input, image_segment)
        if self._max_context:
            count = self._count_prompt_tokens(prompt_input)
            if count is not None:
                requested = req.resolved_max_tokens()
                if count >= self._max_context:
                    raise ContextLengthExceeded(count, self._max_context, requested)
                if requested > 0 and count + requested > self._max_context:
                    raise ContextLengthExceeded(count, self._max_context, requested)

        gen_stops = self._generation_stops(req)
        options = self._options(req, gen_stops)
        preamble = self._template.generation_preamble(
            template_kwargs, tools=template_tools
        )
        return await self._create_response(
            history_req,
            prompt_input,
            options,
            preamble,
            gen_stops,
            lease,
            admission,
            tool_context=tool_context,
            observer=observer,
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
        "--artifact_mode",
        args.artifact_mode,
        "--dflash_block_length",
        str(args.dflash_block_length),
        "--dflash_n_draft",
        str(args.dflash_n_draft),
        f"--dflash_draft_argmax={'true' if args.dflash_draft_argmax else 'false'}",
        f"--cuda_graph={'true' if args.cuda_graph else 'false'}",
    ]
    if args.data_path:
        cmd += ["--data_path", args.data_path]
    cmd += ["--max_image_bytes", str(args.max_image_bytes)]
    if args.pos_embed_path:
        cmd += ["--pos_embed_path", args.pos_embed_path]
    logger.info("Starting Muse Glimmer worker subprocess (loads the model once).")
    return spawn_worker(cmd, env=env)


def _tool_detector(name: str):
    if name == "none":
        return None
    if name == "atem":
        # The Hugging Face chat template supplies the ATEM tool definitions.
        return AtemToolCallDetector
    raise ValueError(f"unknown tool parser: {name}")


def build_app_from_args(args):
    template = ChatTemplate(
        args.hf_tokenizer,
        assistant_header=_ASSISTANT_HEADER,
        strip_rendered_bos=True,
        append_generation_prompt_after_tool_response=True,
    )
    worker = _spawn(args)
    runtime = SessionRuntime(worker)
    from .stream_parser import MuseGlimmerToolReasoningParser

    serving = MuseGlimmerServingChat(
        runtime,
        template,
        args.model_id,
        max_context=args.max_context,
        max_image_bytes=args.max_image_bytes,
        tool_detector_cls=_tool_detector(args.tool_parser),
        tool_reasoning_parser_factory=MuseGlimmerToolReasoningParser,
        reasoning_token_counter=template.count_reasoning_tokens,
        # The worker prepends the numeric BOS the HF template renders as text and
        # serve.py strips (strip_rendered_bos), so token counts include it.
        prompt_token_offset=1,
        content_filter_specials=_MUSE_GLIMMER_HEADER_SPECIALS,
        reasoning_extractor=_extract_muse_glimmer_reasoning,
    )
    app = build_app(serving, args.model_id)

    @app.on_event("shutdown")
    def _stop_worker():
        runtime.close_worker()

    return app, args.model_id


def main() -> None:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible LLM server for Muse Glimmer"
    )
    p.add_argument("--model-path", required=True, help="Path to the .pte model")
    p.add_argument("--data-path", default=None, help="Path to the delegate blob (.ptd)")
    p.add_argument(
        "--pos-embed-path",
        default=None,
        help="Optional path to precomputed image position embeddings.",
    )
    p.add_argument(
        "--max-image-bytes",
        type=int,
        default=_DEFAULT_MAX_IMAGE_BYTES,
        help="Maximum decoded inline image size forwarded to the worker.",
    )
    p.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to the Hugging Face tokenizer.json",
    )
    p.add_argument(
        "--hf-tokenizer",
        required=True,
        help="HF tokenizer id/dir for the model's chat template",
    )
    p.add_argument("--model-id", default="muse_glimmer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Context window; prompts exceeding it are rejected with 400.",
    )
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
        help="Physical sessions the worker can host on one weight load. One slot "
        "is reserved for anonymous requests, so addressable named sessions are "
        "max-sessions - 1.",
    )
    p.add_argument(
        "--warm-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm append-only resume for named sessions when available.",
    )
    p.add_argument(
        "--tool-parser",
        choices=("atem", "none"),
        default="none",
        help="Tool-call format parser to apply to model output. 'atem' parses the "
        "Muse Glimmer `to=NAME<|message|><atem:function_calls>...` format "
        "into OpenAI tool_calls.",
    )
    p.add_argument(
        "--bos-id",
        type=int,
        default=200000,
        help="BOS token id to prepend in the worker. The launcher strips the HF "
        "template's literal BOS before C++ tokenization.",
    )
    p.add_argument("--eos-id", type=int, default=200001)
    p.add_argument(
        "--artifact-mode",
        choices=("auto", "autoregressive", "dflash"),
        default="auto",
        help="Worker artifact mode. Auto detects the exported method contract.",
    )
    p.add_argument(
        "--dflash-block-length",
        type=int,
        default=4,
        help="DFlash block length; zero uses the artifact's exported maximum.",
    )
    p.add_argument(
        "--dflash-n-draft",
        type=int,
        default=3,
        help="DFlash candidates per iteration; zero uses block_length - 1.",
    )
    p.add_argument(
        "--dflash-draft-argmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use exact deterministic argmax proposals for stochastic DFlash drafting.",
    )
    p.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture CUDA graphs for supported methods. This disables CUDA "
        "multi-session state rebinding (default: off).",
    )
    p.add_argument(
        "--worker-bin",
        default=None,
        help="Path to muse_glimmer_worker. Defaults to the CMake output path.",
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
