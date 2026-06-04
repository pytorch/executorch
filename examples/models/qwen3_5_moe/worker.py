# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-isolated Qwen3.5 MoE generation worker.

Runs the CUDA/AOTI model in a dedicated process with NO asyncio HTTP server.
The OpenAI control plane (serve.py) talks to this worker over JSONL on
stdin/stdout. This isolation is required: executing the AOTI CUDA model inside a
live asyncio server process segfaults in the int4 matmul (validated). Here the
model runs like the CLI — a plain synchronous loop — which is reliable.

Protocol (one JSON object per line):
  worker -> stdout, once at startup:  {"ready": true}
  serve  -> stdin,  per request:      {"prompt": str, "max_new_tokens": int,
                                       "temperature": float}
  worker -> stdout, per request:      {"token": str} *   (streamed)
                                      {"done": true, "prompt_tokens": int,
                                       "completion_tokens": int}
                                  or  {"error": str}

stdout carries ONLY protocol JSON; all logs go to stderr. One request at a time.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.session_generate import (
    SessionGenerateAdapter,
)


def _load_ext(explicit_dir):
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "cmake-out" / "examples" / "models" / "qwen3_5_moe")
    for d in candidates:
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    return importlib.import_module("_qwen35_moe")


class _Config:
    __slots__ = ("max_new_tokens", "temperature")

    def __init__(self, max_new_tokens, temperature):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature


def _emit(obj):
    sys.stdout.write(json.dumps(obj))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="Qwen3.5 MoE generation worker")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", default=None)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--hf-tokenizer", required=True)
    p.add_argument("--ext-dir", default=None)
    args = p.parse_args()

    ext = _load_ext(args.ext_dir)
    # HF tokenizer for prompt encoding (the model's own template tokenizer).
    hf_tokenizer = ChatTemplate(args.hf_tokenizer).tokenizer()
    engine = ext.create_engine(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        cuda_graph=False,
    )
    adapter = SessionGenerateAdapter(engine.create_session(), hf_tokenizer)

    _emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            _handle_request(adapter, json.loads(line))
        except Exception as e:  # noqa: BLE001 - report to the control plane
            _emit({"error": repr(e)})


def _handle_request(adapter, req) -> None:
    """Run one generation request and stream the JSONL result. In a function (not
    the read loop) so the callbacks don't close over loop variables."""
    config = _Config(
        int(req.get("max_new_tokens") or -1),
        float(req.get("temperature", 0.0)),
    )
    stats = {"prompt": 0, "gen": 0}

    def stats_cb(s):
        stats["prompt"] = s.num_prompt_tokens
        stats["gen"] = s.num_generated_tokens

    adapter.reset()
    adapter.generate(
        req["prompt"],
        config,
        token_callback=lambda t: _emit({"token": t}),
        stats_callback=stats_cb,
    )
    _emit(
        {
            "done": True,
            "prompt_tokens": stats["prompt"],
            "completion_tokens": stats["gen"],
        }
    )


if __name__ == "__main__":
    main()
