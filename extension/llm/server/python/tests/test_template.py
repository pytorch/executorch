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
# breaks at turn 2 otherwise. OpenAI sends tool-call arguments as a JSON string;
# HF templates expect a mapping (Qwen renders `arguments|items`), so the server
# decodes it before templating.
def test_tool_call_arguments_decoded_for_template():
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
    # Decoded from the JSON string into a mapping the template can iterate.
    assert asst["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["tool_call_id"] == "c1" and "temp_c" in tool["content"]


def test_tool_call_non_json_arguments_left_as_string():
    # A non-JSON arguments value must not crash; it passes through unchanged.
    t, fake = _template_with_fake()
    t.render(
        [
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        index=0,
                        id="c1",
                        type="function",
                        function=FunctionCall(name="f", arguments="not json"),
                    )
                ],
            )
        ]
    )
    asst = next(m for m in fake.seen_messages if m["role"] == "assistant")
    assert asst["tool_calls"][0]["function"]["arguments"] == "not json"


class _HFSpecials:
    """Minimal fake HF tokenizer exposing all_special_tokens / eos_token."""

    def __init__(self, all_special_tokens, eos_token="<|im_end|>"):
        self.all_special_tokens = list(all_special_tokens)
        self.eos_token = eos_token


def _template_with_specials(all_special_tokens, eos_token="<|im_end|>"):
    t = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    t._hf = _HFSpecials(all_special_tokens, eos_token)
    return t


def test_turn_stop_excludes_tool_delimiters():
    # A tokenizer that marks BOTH a turn terminator and tool/structural delimiters
    # as special: the stop set must keep the terminator and drop the delimiters,
    # else generation halts at <tool_call> before the parser sees the call.
    t = _template_with_specials(
        ["<|im_end|>", "<tool_call>", "</tool_call>", "<|box_start|>"],
        eos_token="<|im_end|>",
    )
    stops = t.turn_stop_sequences()
    assert "<|im_end|>" in stops
    assert "<tool_call>" not in stops
    assert "</tool_call>" not in stops
    assert "<|box_start|>" not in stops


def test_turn_stop_includes_eos_and_known_terminators():
    t = _template_with_specials(
        ["<|endoftext|>", "<|eot_id|>", "<tool_call>"], eos_token="<|endoftext|>"
    )
    stops = t.turn_stop_sequences()
    assert "<|endoftext|>" in stops  # the tokenizer EOS
    assert "<|eot_id|>" in stops  # allowlisted terminator registered as special
    assert "<tool_call>" not in stops


def test_turn_stop_drops_whitespace_only_specials():
    t = _template_with_specials(["<|im_end|>", "  ", "\n", ""], eos_token="<|im_end|>")
    stops = t.turn_stop_sequences()
    assert all(s.strip() for s in stops)
    assert "  " not in stops and "\n" not in stops


def test_turn_stop_fallback_without_hf_is_narrow():
    t = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)  # no _hf
    stops = t.turn_stop_sequences()
    assert "<|im_end|>" in stops
    assert "<tool_call>" not in stops


def test_fallback_extracts_text_parts_not_repr():
    # The ChatML fallback renders list-content text parts, not a Python repr.
    t = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)  # no _hf
    msg = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {}},
        ],
    )
    out = t.render([msg])
    assert "hello" in out
    assert "image_url" not in out and "{'type'" not in out  # no repr leak
