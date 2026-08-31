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

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.errors import (
    APIError,
    ContextLengthExceeded,
    ModelNotFound,
)
from executorch.examples.llm_server.python.protocol import ChatCompletionRequest
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import (
    PromptInput,
    SessionRuntime,
)
from executorch.examples.llm_server.python.worker_client import spawn_worker
from executorch.examples.models.muse_glimmer.serving.tool_parsers import (
    AtemToolCallDetector,
)

logger = logging.getLogger(__name__)

# The template ends at this prefix; the model emits the recipient and message
# delimiter.
_ASSISTANT_HEADER = "<|start|>assistant"
_DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_ERROR_CODE = "invalid_image"
_IMAGE_PLACEHOLDER = "\x00muse_glimmer_image_placeholder\x00"
_IMAGE_DATA_URL_RE = re.compile(
    r"\Adata:(image/(?:jpeg|png));base64,([A-Za-z0-9+/]*={0,2})\Z"
)

# Harmony recipient headers may recur within one assistant turn. Treat their
# delimiters as content, then strip every header before returning visible text.
# The optional leading space handles text already trimmed by the tool parser;
# the lookbehind avoids matching prose such as "goto=".
_MUSE_GLIMMER_HEADER_SPECIALS = {"<|start|>", "<|message|>", "<|eom|>"}
_MUSE_GLIMMER_IDENT = r"[a-zA-Z_][a-zA-Z0-9_-]*"
_MUSE_GLIMMER_RECIPIENT = rf"{_MUSE_GLIMMER_IDENT}(?:\.{_MUSE_GLIMMER_IDENT})*"
_MUSE_GLIMMER_HEADER_RE = re.compile(
    rf"(?<![A-Za-z0-9_])"
    rf"(?:<\|start\|>assistant)?"
    rf"\s?to={_MUSE_GLIMMER_RECIPIENT}"
    rf"(?: constrain={_MUSE_GLIMMER_IDENT})?"
    rf"<\|message\|>"
)
_MUSE_GLIMMER_ADDRESSED_HEADER_RE = re.compile(
    rf"(?<![A-Za-z0-9_])"
    rf"(?:<\|start\|>assistant)?"
    rf"\s?to=({_MUSE_GLIMMER_RECIPIENT})"
    rf"(?: constrain={_MUSE_GLIMMER_IDENT})?"
    rf"<\|message\|>"
)
# Remove control tokens left without a complete recipient header.
_MUSE_GLIMMER_STRAY_SPECIAL_RE = re.compile(
    r"<\|(?:start|message|channel|constrain|end|eom|eot)\|>"
)


def _strip_muse_glimmer_header(text: str) -> str:
    text = _MUSE_GLIMMER_HEADER_RE.sub("", text)
    text = _MUSE_GLIMMER_STRAY_SPECIAL_RE.sub("", text)
    return text.strip()


def _extract_muse_glimmer_reasoning(text: str) -> tuple[str | None, str]:
    """Split a Harmony turn into private `to=self` and remaining messages."""
    matches = list(_MUSE_GLIMMER_ADDRESSED_HEADER_RE.finditer(text))
    if not matches:
        return None, text

    reasoning: list[str] = []
    visible: list[str] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        visible.append(prefix)

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        body = re.sub(r"(?:<\|eom\|>|<\|eot\|>)\s*$", "", body).strip()
        if not body:
            continue
        if match.group(1) == "self":
            reasoning.append(body)
        else:
            visible.append(body)

    return "\n\n".join(reasoning) or None, "\n\n".join(visible)


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
                path / "examples" / "models" / "fb" / "muse_glimmer"
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
        / "fb"
        / "muse_glimmer"
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
        text_segments = [
            segment for segment in prompt.segments or [] if "image" not in segment
        ]
        if not text_segments:
            return self._prompt_token_offset
        return super()._count_prompt_tokens(PromptInput(segments=text_segments))

    async def create(self, req: ChatCompletionRequest):
        # This intentionally tracks ServingChat.create closely. The generic class
        # has no prompt-segment hook, while all response work remains inherited.
        if req.model is not None and req.model != self._model_id:
            raise ModelNotFound(req.model, self._model_id)
        self._reject_invalid_values(req)
        self._reject_unsupported_params(req)
        if req.session_id is not None:
            self._validate_session_id(req.session_id)

        req, image_segment = _normalize_image_request(req, self._max_image_bytes)
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
        )
        prompt_input = self._with_image(prompt_input, image_segment)
        if self._max_context:
            count = self._count_prompt_tokens(prompt_input)
            if count is not None:
                if count >= self._max_context:
                    raise ContextLengthExceeded(count, self._max_context)
                requested = req.resolved_max_tokens()
                if requested > 0 and count + requested > self._max_context:
                    raise ContextLengthExceeded(count, self._max_context, requested)

        if self._tools_active(req):
            gen_stops = self._stops + self._request_stops(req)
        else:
            gen_stops = self._stops + self._content_specials + self._request_stops(req)
        options = self._options(req, gen_stops)
        preamble = self._template.generation_preamble(
            template_kwargs, tools=template_tools
        )
        if req.session_id is not None:
            await self._preflight_session(req.session_id)
        if req.stream:
            return self._stream(req, prompt_input, options, preamble, gen_stops)
        return await self._complete(req, prompt_input, options, preamble, gen_stops)


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
    serving = MuseGlimmerServingChat(
        runtime,
        template,
        args.model_id,
        max_context=args.max_context,
        max_image_bytes=args.max_image_bytes,
        tool_detector_cls=_tool_detector(args.tool_parser),
        # The worker prepends the numeric BOS the HF template renders as text and
        # serve.py strips (strip_rendered_bos), so token counts include it.
        prompt_token_offset=1,
        content_filter=_strip_muse_glimmer_header,
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
