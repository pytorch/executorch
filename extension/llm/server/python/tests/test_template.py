# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contract tests for chat-template kwargs (e.g. enable_thinking) passthrough."""

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.protocol import (
    ChatMessage,
    FunctionCall,
    ToolCall,
)


class _FakeHF:
    def __init__(self):
        self.seen_kwargs = None
        self.seen_messages = None
        self.encode_add_special = None

    def apply_chat_template(
        self, messages, tools, add_generation_prompt, tokenize, **kwargs
    ):
        self.seen_kwargs = kwargs
        self.seen_messages = messages
        return "PROMPT"

    # Default add_special_tokens=True mirrors real HF tokenizers (so a caller
    # that forgets to disable specials would over-count).
    def encode(self, text, add_special_tokens=True):
        self.encode_add_special = add_special_tokens
        return list(range(len(text)))  # 1 id per char, deterministic


def _template_with_fake(defaults=None):
    t = ChatTemplate(
        hf_tokenizer_path=None, allow_fallback=True, default_template_kwargs=defaults
    )
    fake = _FakeHF()
    t._hf = fake
    return t, fake


def test_count_tokens_excludes_special_tokens():
    # The rendered prompt already carries control tokens, so count_tokens must
    # encode with add_special_tokens=False (matching the session/prefix-cache
    # paths) — not the tokenizer's default True, which double-counts BOS/EOS and
    # can falsely reject near-limit requests under --max-context.
    t, fake = _template_with_fake()
    n = t.count_tokens("PROMPT")
    assert fake.encode_add_special is False
    assert n == len("PROMPT")


def test_default_template_kwargs_applied():
    t, fake = _template_with_fake(defaults={"enable_thinking": False})
    t.render([ChatMessage(role="user", content="hi")])
    assert fake.seen_kwargs == {"enable_thinking": False}


def test_per_request_kwargs_override_defaults():
    t, fake = _template_with_fake(defaults={"enable_thinking": False})
    t.render(
        [ChatMessage(role="user", content="hi")],
        template_kwargs={"enable_thinking": True},
    )
    assert fake.seen_kwargs["enable_thinking"] is True


def test_no_kwargs_when_none():
    t, fake = _template_with_fake(defaults=None)
    t.render([ChatMessage(role="user", content="hi")])
    assert fake.seen_kwargs == {}


def test_fallback_ignores_kwargs_without_hf():
    # No HF tokenizer → ChatML fallback, must not raise on kwargs.
    t = ChatTemplate(
        hf_tokenizer_path=None,
        allow_fallback=True,
        default_template_kwargs={"enable_thinking": False},
    )
    out = t.render([ChatMessage(role="user", content="hi")], template_kwargs={"x": 1})
    assert "<|im_start|>user" in out and out.endswith("<|im_start|>assistant\n")


# (5) Chat-template behaviors: multi-turn ordering, system message, roles.
def test_multi_turn_order_preserved():
    t = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    out = t.render(
        [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="second"),
            ChatMessage(role="user", content="third"),
        ]
    )
    assert out.index("first") < out.index("second") < out.index("third")
    assert out.endswith("<|im_start|>assistant\n")  # generation prompt appended


def test_system_message_rendered():
    t = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    out = t.render(
        [
            ChatMessage(role="system", content="You are terse."),
            ChatMessage(role="user", content="hi"),
        ]
    )
    assert "<|im_start|>system\nYou are terse." in out


def test_each_role_labeled():
    t = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    out = t.render(
        [
            ChatMessage(role="system", content="s"),
            ChatMessage(role="user", content="u"),
            ChatMessage(role="assistant", content="a"),
        ]
    )
    for role in ("system", "user", "assistant"):
        assert f"<|im_start|>{role}" in out


# Tool round-trip: a turn-2 request (assistant tool_call + tool result) must
# serialize into the shape any HF chat template consumes — the multi-turn loop
# breaks at turn 2 otherwise.
def test_tool_call_roundtrip_messages_passthrough():
    t, fake = _template_with_fake()
    t.render(
        [
            ChatMessage(role="user", content="weather?"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        index=0,
                        id="c1",
                        type="function",
                        function=FunctionCall(
                            name="get_weather", arguments='{"city": "Paris"}'
                        ),
                    )
                ],
            ),
            ChatMessage(role="tool", tool_call_id="c1", content='{"temp_c": 18}'),
        ]
    )
    msgs = fake.seen_messages
    asst = next(m for m in msgs if m["role"] == "assistant")
    assert asst["tool_calls"][0]["function"]["name"] == "get_weather"
    assert asst["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["tool_call_id"] == "c1" and "temp_c" in tool["content"]
