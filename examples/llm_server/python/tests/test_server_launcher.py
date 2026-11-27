# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import SimpleNamespace

import pytest
import uvicorn

from executorch.examples.llm_server.python import server


@pytest.mark.parametrize("configured", [False, True])
def test_launcher_checks_assistant_header_before_starting_worker(
    monkeypatch, caplog, configured
):
    header = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    class Tokenizer:
        chat_template = "llama"
        all_special_tokens = []

        def apply_chat_template(self, messages, **kwargs):
            return "<|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>" + header

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: Tokenizer())
        ),
    )
    argv = [
        "server",
        "--worker-bin",
        "worker",
        "--model-path",
        "model.pte",
        "--tokenizer-path",
        "tokenizer.json",
        "--hf-tokenizer",
        "local-tokenizer",
    ]
    if configured:
        argv.extend(["--assistant-header", header])
    monkeypatch.setattr(sys, "argv", argv)

    def spawn(args):
        warnings = [
            record for record in caplog.records if "Assistant header" in record.message
        ]
        assert len(warnings) == (0 if configured else 1)
        return object()

    monkeypatch.setattr(server, "_spawn", spawn)
    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: apps.append(app))
    server.main()
    assert len(apps) == 1
