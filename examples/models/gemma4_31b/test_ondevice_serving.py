# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import urllib.request

import pytest

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.protocol import ChatMessage

_SERVER = os.environ.get("GEMMA_SERVER_URL")
_HF_DIR = os.environ.get("GEMMA_HF_DIR")

pytestmark = pytest.mark.skipif(
    not _SERVER or not _HF_DIR or not os.path.isdir(_HF_DIR),
    reason="set GEMMA_SERVER_URL and GEMMA_HF_DIR to run Gemma on-device tests",
)


def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _SERVER.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_prompt_tokens_match_real_template_with_numeric_bos_prefix():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    template = ChatTemplate(
        _HF_DIR,
        assistant_header="<|turn>model\n",
        strip_rendered_bos=True,
        append_generation_prompt_after_tool_response=True,
    )
    tok = AutoTokenizer.from_pretrained(_HF_DIR)
    messages = [ChatMessage(role="user", content="Say ok.")]
    rendered = template.render(messages)
    expected_ids = [tok.bos_token_id] + tok.encode(rendered, add_special_tokens=False)

    body = _post(
        "/v1/chat/completions",
        {
            "model": "gemma4_31b",
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 1,
            "temperature": 0,
            "session_id": "gemma-bos-regression",
        },
    )
    assert body["usage"]["prompt_tokens"] == len(expected_ids)
