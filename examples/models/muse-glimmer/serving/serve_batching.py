# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Serve Muse Glimmer text/images with the batching multiplexed worker.

Requires an off-graph batching export: export_solo.py --use-offgraph-kv-cache
for MLX; add --backend cuda for CUDA. Both serve text and images when exported
with --mmproj. DFlash is opt-in and requires a matching export; the legacy
launcher and runners remain separate.
"""

import argparse
import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from executorch.examples.llm_server.python.hud.worker_logs import capture_worker_logs
from executorch.examples.llm_server.python.multiplexed_worker_client import (
    spawn_multiplexed_worker,
)
from executorch.examples.llm_server.python.serve import (
    _parse_preflight_timeout,
    _validate_limits,
)
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.models.muse_glimmer.serving import serve as legacy
from executorch.examples.models.muse_glimmer.serving.chat_template import (
    MuseGlimmerChatTemplate as ChatTemplate,
)
from executorch.examples.models.muse_glimmer.serving.stream_parser import (
    MuseGlimmerStreamParser,
    MuseGlimmerToolReasoningParser,
)

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_MAX_MESSAGE_BYTES = 1024 * 1024
_INT32_MAX = (1 << 31) - 1
_UINT64_MAX = (1 << 64) - 1


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-bin", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--hf-tokenizer", required=True)
    parser.add_argument("--pos-embed-path")
    parser.add_argument("--backend", choices=("mlx", "cuda"), default="mlx")
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument(
        "--dflash-n-draft",
        type=int,
        default=0,
        help="Draft proposals per round; 0 uses the exported maximum minus one.",
    )
    parser.add_argument("--model-id", default="muse_glimmer")
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument(
        "--evict-idle-sessions",
        action="store_true",
        help="Evict least recently used idle sessions when logical session slots are exhausted; "
        "clients must send complete history.",
    )
    parser.add_argument("--max-decode-sequences", type=int, default=4)
    parser.add_argument("--max-inflight-requests", type=int, default=4)
    parser.add_argument(
        "--prefix-cache-entries",
        type=int,
        help="Retained snapshots per named session (default: 8); shared with prompt-end checkpoints disabled (default: 0)",
    )
    parser.add_argument(
        "--prefix-capture-slots",
        type=int,
        default=0,
        help="Transient capture rows; 0 (default): size from session/request limits",
    )
    parser.add_argument("--max-vision-patches", type=int, default=4096)
    parser.add_argument(
        "--prefix-checkpoint-prompt-end",
        nargs="?",
        choices=("true", "false"),
        const="true",
        default="true",
        help="Retain prompt-end checkpoints separately for each named session (default: true; false disables)",
    )
    parser.add_argument(
        "--prefix-checkpoint-interval",
        type=int,
        default=0,
        help="Periodic checkpoint interval in tokens; 0 disables periodic capture.",
    )
    parser.add_argument("--max-image-bytes", type=int, default=_MAX_IMAGE_BYTES)
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=_MAX_REQUEST_BYTES,
        help="Input JSONL frame limit, including base64 image data and newline.",
    )
    parser.add_argument("--bos-id", type=int, default=200000)
    parser.add_argument("--eos-id", type=int, default=200001)
    parser.add_argument("--tool-parser", choices=("atem", "none"), default="none")
    parser.add_argument(
        "--streaming-preflight-timeout",
        type=_parse_preflight_timeout,
        default=None,
        help="Seconds to wait for first output (including prefill); none waits indefinitely. "
        "Later errors are SSE events.",
    )
    parser.add_argument(
        "--api-dialect",
        choices=("openai", "openrouter"),
        default="openai",
        help="OpenRouter clients require usage on the finish chunk; OpenAI uses "
        "a separate usage-only chunk. OpenRouter also selects "
        "context-overflow error metadata.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    args.prefix_checkpoint_prompt_end = args.prefix_checkpoint_prompt_end == "true"
    if args.prefix_cache_entries is None:
        args.prefix_cache_entries = 8 if args.prefix_checkpoint_prompt_end else 0
    _validate_limits(
        parser,
        args,
        prefix_capture_slots=args.prefix_capture_slots,
        prefix_checkpoint_prompt_end=args.prefix_checkpoint_prompt_end,
    )
    if not 0 <= args.dflash_n_draft < _INT32_MAX:
        parser.error("--dflash-n-draft must be nonnegative and below INT32_MAX")
    if args.dflash_n_draft and not args.dflash:
        parser.error("--dflash-n-draft requires --dflash")
    if not 4 <= args.max_vision_patches <= _INT32_MAX:
        parser.error(f"--max-vision-patches must be between 4 and {_INT32_MAX}")
    if args.max_context <= 1:
        parser.error("--max-context must be greater than 1")
    for name, maximum in (
        ("max_image_bytes", _MAX_IMAGE_BYTES),
        ("max_request_bytes", _MAX_REQUEST_BYTES),
    ):
        if not 1 <= getattr(args, name) <= maximum:
            parser.error(f"--{name.replace('_', '-')} must be between 1 and {maximum}")
    if not (0 <= args.bos_id <= _UINT64_MAX and 0 <= args.eos_id <= _UINT64_MAX):
        parser.error("BOS and EOS IDs must be uint64 values")
    if 4 * ((args.max_image_bytes + 2) // 3) + 1024 > args.max_request_bytes:
        parser.error("--max-request-bytes must fit the base64 image plus JSON framing")
    _validate_backend(parser, args)
    _validate_paths(parser, args)
    return args


def _validate_backend(parser, args):
    if args.backend != "cuda":
        return
    if args.data_path is None:
        parser.error("--backend cuda requires --data-path (aoti_cuda_blob.ptd)")


def _validate_paths(parser, args):
    args.worker_bin = os.path.expanduser(args.worker_bin)
    if shutil.which(args.worker_bin) is None:
        parser.error("--worker-bin must name an executable file or command on PATH")
    for name in ("model_path", "data_path", "tokenizer_path", "pos_embed_path"):
        value = getattr(args, name)
        if value is None:
            continue
        path = Path(value).expanduser()
        # Preserve HF tokenizer configuration, including EOS metadata, when present.
        if (
            name == "tokenizer_path"
            and path.name == "tokenizer.json"
            and path.is_file()
            and path.with_name("tokenizer_config.json").is_file()
        ):
            path = path.parent
        if name == "tokenizer_path" and path.is_dir():
            if not (path / "tokenizer.json").is_file():
                parser.error("tokenizer directory must contain tokenizer.json")
        elif not path.is_file():
            parser.error(f"--{name.replace('_', '-')} must name an existing file")
        setattr(args, name, str(path))


def _worker_command(args):
    command = [args.worker_bin]
    for flag, value in (
        ("pte", args.model_path),
        ("tokenizer", args.tokenizer_path),
        ("backend", args.backend),
        ("dflash", str(args.dflash).lower()),
        ("dflash_n_draft", args.dflash_n_draft),
        ("max_sessions", args.max_sessions),
        ("max_session_tokens", args.max_context),
        ("max_decode_sequences", args.max_decode_sequences),
        ("max_inflight_requests", args.max_inflight_requests),
        ("prefix_cache_entries", args.prefix_cache_entries),
        (
            "prefix_capture_slots",
            (args.prefix_capture_slots if args.prefix_cache_entries else None),
        ),
        ("prefix_checkpoint_interval", args.prefix_checkpoint_interval or None),
        (
            "prefix_checkpoint_prompt_end",
            str(args.prefix_checkpoint_prompt_end).lower(),
        ),
        ("max_vision_patches", args.max_vision_patches),
        ("max_image_bytes", args.max_image_bytes),
        ("max_input_frame_bytes", args.max_request_bytes),
        ("bos_id", args.bos_id),
        ("eos_id", args.eos_id),
        ("data_path", args.data_path),
        ("pos_embed_path", args.pos_embed_path),
    ):
        if value is not None:
            # gflags requires '=' to pass false for boolean flags.
            command.append(f"--{flag}={value}")
    return command


def build_app_from_args(args):
    template = ChatTemplate(
        args.hf_tokenizer,
        assistant_header=legacy._ASSISTANT_HEADER,
        strip_rendered_bos=True,
        append_generation_prompt_after_tool_response=True,
    )
    template.generation_preamble()

    @asynccontextmanager
    async def serving_factory():
        async with capture_worker_logs() as stderr:
            worker = await spawn_multiplexed_worker(
                _worker_command(args),
                stderr=stderr,
                max_request_bytes=args.max_request_bytes,
                max_message_bytes=_MAX_MESSAGE_BYTES,
            )
            runtime = None
            try:
                runtime = SessionRuntime(worker)
                yield legacy.MuseGlimmerServingChat(
                    runtime,
                    template,
                    args.model_id,
                    max_context=args.max_context,
                    evict_idle_sessions=args.evict_idle_sessions,
                    streaming_preflight_timeout=args.streaming_preflight_timeout,
                    api_dialect=args.api_dialect,
                    max_image_bytes=args.max_image_bytes,
                    tool_detector_cls=legacy._tool_detector(args.tool_parser),
                    prompt_token_offset=1,
                    content_filter_specials=legacy._MUSE_GLIMMER_HEADER_SPECIALS,
                    reasoning_extractor=legacy._extract_muse_glimmer_reasoning,
                    streaming_parser_factory=MuseGlimmerStreamParser,
                    tool_reasoning_parser_factory=MuseGlimmerToolReasoningParser,
                    reasoning_token_counter=template.count_reasoning_tokens,
                )
            finally:
                if runtime is None:
                    await SessionRuntime._finish_cleanup(
                        asyncio.create_task(worker.close())
                    )
                else:
                    await runtime.aclose_worker()

    host = f"[{args.host}]" if ":" in args.host else args.host
    return build_app(
        None,
        args.model_id,
        serving_factory=serving_factory,
        model_path=args.model_path,
        max_context=args.max_context,
        max_concurrent_requests=args.max_inflight_requests,
        hud=True,
        listen_address=f"http://{host}:{args.port}",
    )


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    uvicorn.run(build_app_from_args(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
