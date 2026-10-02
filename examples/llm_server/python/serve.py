# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch the batching-backed OpenAI server with one multiplexed worker."""

import argparse
import logging
import os
import shutil
from pathlib import Path

from .chat_template import ChatTemplate
from .server import build_app
from .serving_chat import ServingChat
from .session_runtime import SessionRuntime
from .tool_parsers import HermesDetector
from .worker_client import spawn_worker

_MAX_INT32 = (1 << 31) - 1


def _validate_limits(parser, args) -> None:
    for flag, value in (
        ("--max-context", args.max_context),
        ("--max-sessions", args.max_sessions),
        ("--max-decode-sequences", args.max_decode_sequences),
        ("--max-inflight-requests", args.max_inflight_requests),
    ):
        if not 1 <= value <= _MAX_INT32:
            parser.error(f"{flag} must be between 1 and {_MAX_INT32}")
    if not 0 <= args.prefix_cache_entries <= _MAX_INT32:
        parser.error(f"--prefix-cache-entries must be between 0 and {_MAX_INT32}")
    physical_sessions = (
        args.max_sessions
        + args.prefix_cache_entries
        + (1 if args.prefix_cache_entries else 0)
    )
    if physical_sessions > _MAX_INT32:
        parser.error("session and prefix-cache capacity exceeds the row limit")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-bin", required=True, help="Multiplexed worker executable"
    )
    parser.add_argument(
        "--model-path", required=True, help="Packed off-graph-cache PTE"
    )
    parser.add_argument(
        "--tokenizer-path", required=True, help="Tokenizer file or HF directory"
    )
    parser.add_argument(
        "--hf-tokenizer",
        help="HF chat-template source; defaults to the tokenizer directory",
    )
    parser.add_argument("--model-id", default="executorch")
    parser.add_argument(
        "--assistant-header",
        default="<|im_start|>assistant\n",
        help="Exact assistant generation header, including trailing whitespace",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        required=True,
        help="Context limit for sessions and HTTP validation",
    )
    parser.add_argument("--max-sessions", type=int, default=16)
    parser.add_argument("--max-decode-sequences", type=int, default=8)
    parser.add_argument("--max-inflight-requests", type=int, default=64)
    parser.add_argument("--prefix-cache-entries", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    _validate_limits(parser, args)

    args.worker_bin = os.path.expanduser(args.worker_bin)
    if shutil.which(args.worker_bin) is None:
        parser.error("--worker-bin must name an executable file or command on PATH")
    args.model_path = str(Path(args.model_path).expanduser())
    if not Path(args.model_path).is_file():
        parser.error("--model-path must name an existing file")

    tokenizer = Path(args.tokenizer_path).expanduser()
    # File-only HF loading skips tokenizer_config.json, including its EOS token.
    if (
        tokenizer.name == "tokenizer.json"
        and tokenizer.is_file()
        and tokenizer.with_name("tokenizer_config.json").is_file()
    ):
        tokenizer = tokenizer.parent
    if tokenizer.is_dir():
        if not (tokenizer / "tokenizer.json").is_file():
            parser.error("--tokenizer-path directory must contain tokenizer.json")
        if args.hf_tokenizer is None:
            args.hf_tokenizer = str(tokenizer)
    elif not tokenizer.is_file():
        parser.error("--tokenizer-path must name an existing file or HF directory")
    args.tokenizer_path = str(tokenizer)
    if not args.hf_tokenizer:
        parser.error("--hf-tokenizer is required when --tokenizer-path is a file")
    return args


def main(argv=None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    import uvicorn

    template = ChatTemplate(
        hf_tokenizer_path=args.hf_tokenizer,
        assistant_header=args.assistant_header,
    )
    template.generation_preamble()
    worker = spawn_worker(
        [
            args.worker_bin,
            "--pte",
            args.model_path,
            "--tokenizer",
            args.tokenizer_path,
            "--max_sessions",
            str(args.max_sessions),
            "--max_session_tokens",
            str(args.max_context),
            "--max_decode_sequences",
            str(args.max_decode_sequences),
            "--max_inflight_requests",
            str(args.max_inflight_requests),
            "--prefix_cache_entries",
            str(args.prefix_cache_entries),
        ],
        require_multiplexing=True,
    )
    runtime = None
    try:
        runtime = SessionRuntime(worker)
        serving = ServingChat(
            runtime,
            template,
            args.model_id,
            max_context=args.max_context,
            tool_detector_cls=HermesDetector,
        )
        uvicorn.run(build_app(serving, args.model_id), host=args.host, port=args.port)
    finally:
        if runtime is None:
            worker.close()
        else:
            runtime.close_worker()


if __name__ == "__main__":
    main()
