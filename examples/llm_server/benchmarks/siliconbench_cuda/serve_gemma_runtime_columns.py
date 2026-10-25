# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license in the root LICENSE.

"""Gemma 4 OpenAI API using the generic batching multiplexed worker."""

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.multiplexed_worker_client import spawn_multiplexed_worker
from executorch.examples.llm_server.python.server import build_app
from executorch.examples.llm_server.python.serving_chat import ServingChat
from executorch.examples.llm_server.python.session_runtime import SessionRuntime
from executorch.examples.models.gemma4_31b.serve import (
    _GEMMA_CHANNEL_SPECIALS, _strip_gemma_channels, _tool_detector,
)


def build_app_from_args(args):
    template = ChatTemplate(args.hf_tokenizer, assistant_header="<|turn>model\n",
        strip_rendered_bos=False, append_generation_prompt_after_tool_response=False)
    template.generation_preamble()
    if args.prefix_cache_entries == 0:
        args.prefix_checkpoint_interval = 0
    command = [args.worker_bin, f"--pte={args.model_path}", f"--data_path={args.data_path}",
        f"--tokenizer={args.tokenizer_path}", f"--max_sessions={args.max_sessions}",
        f"--max_session_tokens={args.max_context}",
        f"--initial_kv_capacity={args.initial_kv_capacity}",
        f"--max_decode_sequences={args.max_decode_sequences}",
        f"--max_inflight_requests={args.max_inflight_requests}",
        f"--prefix_cache_entries={args.prefix_cache_entries}",
        f"--prefix_checkpoint_interval={args.prefix_checkpoint_interval}",
        f"--idle_kv_compaction_ms={args.idle_kv_compaction_ms}",
        f"--idle_kv_spare_columns={args.idle_kv_spare_columns}"]

    @asynccontextmanager
    async def serving_factory():
        worker = await spawn_multiplexed_worker(command,
            max_request_bytes=32 * 1024 * 1024, max_message_bytes=1024 * 1024)
        runtime = None
        try:
            runtime = SessionRuntime(worker)
            yield ServingChat(runtime, template, args.model_id,
                max_context=args.max_context, prompt_token_offset=0,
                tool_detector_cls=_tool_detector(args.tool_parser),
                content_filter=_strip_gemma_channels,
                content_filter_specials=_GEMMA_CHANNEL_SPECIALS)
        finally:
            if runtime is None:
                await SessionRuntime._finish_cleanup(asyncio.create_task(worker.close()))
            else:
                await runtime.aclose_worker()

    return build_app(None, args.model_id, serving_factory=serving_factory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["worker-bin", "model-path", "data-path", "tokenizer-path", "hf-tokenizer"]:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--model-id", default="gemma4-12b-it")
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--initial-kv-capacity", type=int, default=-1)
    parser.add_argument("--max-sessions", type=int, default=16)
    parser.add_argument("--max-decode-sequences", type=int, default=16)
    parser.add_argument("--max-inflight-requests", type=int, default=16)
    parser.add_argument("--prefix-cache-entries", type=int, default=16)
    parser.add_argument("--prefix-checkpoint-interval", type=int, default=512)
    parser.add_argument("--idle-kv-compaction-ms", type=int, default=250)
    parser.add_argument("--idle-kv-spare-columns", type=int, default=1)
    parser.add_argument("--tool-parser", choices=["none", "gemma"], default="none")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    for name in [args.worker_bin, args.model_path, args.data_path]:
        if not Path(name).is_file():
            parser.error(f"Missing file: {name}")
    if min(args.max_context, args.max_sessions, args.max_decode_sequences,
           args.max_inflight_requests) <= 0 or args.prefix_cache_entries < 0:
        parser.error("Invalid batching limits")
    if args.initial_kv_capacity != -1 and not 0 < args.initial_kv_capacity <= args.max_context:
        parser.error("Initial KV capacity must fit max context")
    import uvicorn
    uvicorn.run(build_app_from_args(args), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
