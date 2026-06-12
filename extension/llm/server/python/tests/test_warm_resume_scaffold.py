# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Warm-resume generation-scaffold reproduction.

Qwen3's template prefills a deterministic ``<think>`` scaffold into the
generation prompt (so it lands in resident KV) but strips it when re-rendering a
turn as history *before* the last user message, while *preserving* it (as the
empty block) for turns after. The token-ID splice must reproduce each turn's
exact resident scaffold ahead of its generated ids, normalizing whatever the
history render put there -- inserting when stripped, replacing when a different
form was preserved -- so the worker's exact-token prefix check lands.
"""

import os

import pytest

from executorch.extension.llm.server.python.chat_template import ChatTemplate
from executorch.extension.llm.server.python.openai_transcript import (
    OpenAITranscriptState,
)
from executorch.extension.llm.server.python.protocol import (
    ChatMessage,
    FunctionCall,
    ToolCall,
)

HDR = "<|im_start|>assistant\n"
NOTHINK = "<think>\n\n</think>\n\n"  # no-think generation preamble / preserved block
THINK = "<think>\n"  # think-mode generation preamble
GEMMA_HDR = "<|turn>model\n"
GEMMA_PREAMBLE = "<|channel>thought\n<channel|>"


def _msgs(*pairs):
    return [ChatMessage(role=r, content=c) for r, c in pairs]


class _FakeQwen:
    """Mimics Qwen3 scaffold behavior in render(): the generation prompt appends
    the mode scaffold after the assistant header; history strips the scaffold for
    assistant turns before the last user message and preserves the empty block
    for turns after it (true in both modes -- the case that needs normalize)."""

    def __init__(self, default_thinking=False):
        self._default_thinking = default_thinking

    def _gen(self, kw):
        thinking = (kw or {}).get("enable_thinking", self._default_thinking)
        return THINK if thinking else NOTHINK

    def render(self, messages, tools=None, template_kwargs=None):
        last_user = max(
            (i for i, m in enumerate(messages) if m.role == "user"), default=-1
        )
        out = []
        for i, m in enumerate(messages):
            c = m.content if isinstance(m.content, str) else ""
            if m.role == "assistant" and i > last_user:
                out.append(f"{HDR}{NOTHINK}{c}<|im_end|>\n")  # preserved empty block
            else:
                out.append(f"<|im_start|>{m.role}\n{c}<|im_end|>\n")
        out.append(HDR + self._gen(template_kwargs))
        return "".join(out)


class _FakePlain:
    """No-scaffold ChatML template (preamble '')."""

    def render(self, messages, tools=None, template_kwargs=None):
        out = [
            f"<|im_start|>{m.role}\n"
            f"{m.content if isinstance(m.content, str) else ''}<|im_end|>\n"
            for m in messages
        ]
        out.append(HDR)
        return "".join(out)


class _FakeOtherHeader:
    """No-scaffold template whose assistant header is NOT the Qwen/ChatML one
    (Llama-style), to prove token-id splicing isn't disabled for templates that
    don't use ``<|im_start|>assistant\\n`` when the preamble is ''."""

    OHDR = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def render(self, messages, tools=None, template_kwargs=None):
        out = []
        for m in messages:
            c = m.content if isinstance(m.content, str) else ""
            if m.role == "assistant":
                out.append(f"{self.OHDR}{c}<|eot_id|>")
            else:
                out.append(
                    f"<|start_header_id|>{m.role}<|end_header_id|>\n\n{c}<|eot_id|>"
                )
        out.append(self.OHDR)
        return "".join(out)


class _FakeGemma:
    def assistant_header(self):
        return GEMMA_HDR

    def render(self, messages, tools=None, template_kwargs=None):
        out = ["<bos>"]
        for m in messages:
            role = "model" if m.role == "assistant" else m.role
            content = m.content if isinstance(m.content, str) else ""
            out.append(f"<|turn>{role}\n{content}<turn|>\n")
        out.append(GEMMA_HDR + GEMMA_PREAMBLE)
        return "".join(out)


def _ids_index(segs, ids):
    for i, s in enumerate(segs):
        if s.get("ids") == ids:
            return i
    return -1


def _text_before_ids(segs, ids):
    i = _ids_index(segs, ids)
    assert i > 0 and "text" in segs[i - 1], "expected a {text} segment before {ids}"
    return segs[i - 1]["text"]


def _scaffold_before(segs, ids):
    """The scaffold region: text after the last assistant header preceding ids."""
    return _text_before_ids(segs, ids).rsplit(HDR, 1)[-1]


# --- 5a. Hermetic unit tests (no model) -------------------------------------


def test_nothink_ordinary_append_inserts_scaffold():
    st = OpenAITranscriptState(_FakeQwen(default_thinking=False))
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[10, 11, 12],
        prior_turns=0,
        preamble=NOTHINK,
    )
    msgs = _msgs(("user", "u1"), ("assistant", "a1"), ("user", "u2"))
    kw = {"enable_thinking": False}
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=st._template.render(msgs, template_kwargs=kw),
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is not None
    # History stripped the scaffold; the fix inserts exactly one copy.
    assert _scaffold_before(pi.segments, [10, 11, 12]) == NOTHINK
    assert _text_before_ids(pi.segments, [10, 11, 12]).count(NOTHINK) == 1


def test_think_ordinary_append_inserts_open_scaffold():
    st = OpenAITranscriptState(_FakeQwen(default_thinking=True))
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[1, 2],
        prior_turns=0,
        preamble=THINK,
    )
    msgs = _msgs(("user", "u1"), ("assistant", "a1"), ("user", "u2"))
    kw = {"enable_thinking": True}
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=st._template.render(msgs, template_kwargs=kw),
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is not None
    assert _scaffold_before(pi.segments, [1, 2]) == THINK


def test_think_toolloop_normalizes_preserved_scaffold():
    # Turn generated in THINK mode (preamble open-think) but rendered as a
    # post-last-user turn, where history preserves the *empty* block. The fix
    # must REPLACE that block with the stored open-think preamble -- not keep it
    # (wrong scaffold) and not append a second one (double-insert).
    st = OpenAITranscriptState(_FakeQwen(default_thinking=True))
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[7, 8, 9],
        prior_turns=0,
        preamble=THINK,
    )
    msgs = _msgs(("user", "u1"), ("assistant", "a1"))  # a1 AFTER last user
    kw = {"enable_thinking": True}
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=st._template.render(msgs, template_kwargs=kw),
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is not None
    assert _scaffold_before(pi.segments, [7, 8, 9]) == THINK
    # the preserved empty block was replaced, not kept and not doubled
    assert NOTHINK not in _text_before_ids(pi.segments, [7, 8, 9])


def test_no_scaffold_template_is_unchanged():
    st = OpenAITranscriptState(_FakePlain())
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[5],
        prior_turns=0,
        preamble="",
    )
    msgs = _msgs(("user", "u1"), ("assistant", "a1"), ("user", "u2"))
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=st._template.render(msgs),
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None
    assert _scaffold_before(pi.segments, [5]) == ""  # nothing inserted, no regression


def test_non_qwen_header_no_scaffold_still_splices():
    # Regression: a no-scaffold template whose assistant header isn't the
    # Qwen/ChatML one must still get token-id splicing (the normalization is a
    # no-op when preamble == "", not a hard requirement for the Qwen header).
    st = OpenAITranscriptState(_FakeOtherHeader())
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[9, 9],
        prior_turns=0,
        preamble="",
    )
    msgs = _msgs(("user", "u1"), ("assistant", "a1"), ("user", "u2"))
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=st._template.render(msgs),
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None  # splicing NOT disabled by the missing header
    assert any(s.get("ids") == [9, 9] for s in pi.segments)  # ids actually spliced


def test_custom_assistant_header_inserts_scaffold():
    st = OpenAITranscriptState(_FakeGemma())
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[4, 5],
        prior_turns=0,
        preamble=GEMMA_PREAMBLE,
    )
    msgs = _msgs(("user", "u1"), ("assistant", "a1"), ("user", "u2"))
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=st._template.render(msgs),
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None
    before = _text_before_ids(pi.segments, [4, 5])
    assert before.rsplit(GEMMA_HDR, 1)[-1] == GEMMA_PREAMBLE


def test_stop_trimmed_turn_falls_back_to_text():
    st = OpenAITranscriptState(_FakeQwen())
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[],  # stop-trimmed -> ids None -> not resumable
        prior_turns=0,
        preamble=NOTHINK,
    )
    msgs = _msgs(("user", "u1"), ("assistant", "a1"), ("user", "u2"))
    kw = {"enable_thinking": False}
    rendered = st._template.render(msgs, template_kwargs=kw)
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is None and pi.text == rendered


def test_fingerprint_mismatch_falls_back_to_text():
    st = OpenAITranscriptState(_FakeQwen())
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[10],
        prior_turns=0,
        preamble=NOTHINK,
    )
    msgs = _msgs(("user", "u1"), ("assistant", "EDITED"), ("user", "u2"))
    kw = {"enable_thinking": False}
    rendered = st._template.render(msgs, template_kwargs=kw)
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is None and pi.text == rendered


def test_mode_switch_uses_per_turn_scaffold():
    st = OpenAITranscriptState(_FakeQwen())
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[1],
        prior_turns=0,
        preamble=NOTHINK,  # turn 1 generated no-think
    )
    st.record_assistant_turn(
        session_id="s",
        content="a2",
        tool_calls=None,
        generated_token_ids=[2],
        prior_turns=1,
        preamble=THINK,  # turn 2 generated think
    )
    msgs = _msgs(
        ("user", "u1"),
        ("assistant", "a1"),
        ("user", "u2"),
        ("assistant", "a2"),
        ("user", "u3"),
    )
    kw = {"enable_thinking": True}
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=st._template.render(msgs, template_kwargs=kw),
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is not None
    assert _scaffold_before(pi.segments, [1]) == NOTHINK
    assert _scaffold_before(pi.segments, [2]) == THINK


# --- Tool-call argument fingerprint canonicalization ------------------------


def _fp(content, tool_calls):
    return OpenAITranscriptState._assistant_fingerprint(content, tool_calls)


def _dtc(name, args):
    return {"function": {"name": name, "arguments": args}}


def test_fingerprint_ignores_tool_arg_whitespace():
    assert _fp(None, [_dtc("bash", '{"command": "echo hi"}')]) == _fp(
        None, [_dtc("bash", '{"command":"echo hi"}')]
    )


def test_fingerprint_ignores_tool_arg_key_order():
    assert _fp(None, [_dtc("f", '{"x": 1, "y": 2}')]) == _fp(
        None, [_dtc("f", '{"y": 2, "x": 1}')]
    )


def test_fingerprint_invalid_json_args_stay_byte_sensitive():
    # Non-JSON arguments can't be canonicalized, so they stay literal: a
    # genuinely different string remains a different turn.
    assert _fp(None, [_dtc("f", "not json {")]) != _fp(
        None, [_dtc("f", "not json {  ")]
    )


def test_fingerprint_non_string_args_match_equivalent_json_string():
    # Already-structured args hash stably and match the equivalent JSON string.
    assert _fp(None, [_dtc("f", {"x": 1})]) == _fp(None, [_dtc("f", '{"x": 1}')])


def test_tool_turn_splices_despite_reserialized_args():
    # End-to-end: the server recorded a spaced arguments string; the client echoes
    # the same call back compact (the real pi behavior). The turn must still
    # fingerprint-match and splice -- not prune to a text fallback.
    st = OpenAITranscriptState(_FakeQwen())
    st.record_assistant_turn(
        session_id="s",
        content=None,
        tool_calls=[
            ToolCall(
                index=0,
                id="c1",
                type="function",
                function=FunctionCall(name="bash", arguments='{"command": "echo hi"}'),
            )
        ],
        generated_token_ids=[1, 2, 3],
        prior_turns=0,
        preamble=NOTHINK,
    )
    echoed = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                index=0,
                id="c1",
                type="function",
                function=FunctionCall(name="bash", arguments='{"command":"echo hi"}'),
            )
        ],
    )
    msgs = [
        ChatMessage(role="user", content="u1"),
        echoed,
        ChatMessage(role="user", content="u2"),
    ]
    kw = {"enable_thinking": False}
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=st._template.render(msgs, template_kwargs=kw),
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is not None  # matched + spliced, not pruned to text
    assert any(s.get("ids") == [1, 2, 3] for s in pi.segments)


def test_gemma_tool_span_ignores_close_marker_inside_string():
    st = OpenAITranscriptState(_FakeGemma())
    call = ToolCall(
        id="call_1",
        function=FunctionCall(
            name="bash", arguments='{"command":"printf <tool_call|> ok"}'
        ),
    )
    st.record_assistant_turn(
        session_id="s",
        content="",
        tool_calls=[call],
        generated_token_ids=[101, 102, 103],
        prior_turns=0,
        preamble="",
    )
    msgs = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="", tool_calls=[call]),
        ChatMessage(role="tool", tool_call_id="call_1", content="done"),
    ]
    rendered = (
        "<bos><|turn>user\nu1<turn|>\n"
        "<|turn>model\n"
        '<|tool_call>call:bash{command:<|"|>printf <tool_call|> ok<|"|>}<tool_call|>'
        "<|tool_response>done<tool_response|>"
    )
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None
    assert any(s.get("ids") == [101, 102, 103] for s in pi.segments)
    suffix = "".join(
        s.get("text", "")
        for s in pi.segments[_ids_index(pi.segments, [101, 102, 103]) + 1 :]
    )
    assert suffix == "<|tool_response>done<tool_response|>"


# --- 5b. Token-level fidelity against the real tokenizer (gated/skipped) -----

_MODEL = os.environ.get(
    "QWEN_HF_DIR", "/home/mnachin/local/scripts/models/Qwen3.5-35B-A3B-HQQ-INT4"
)
_HAVE_MODEL = os.path.isdir(_MODEL)
_skip = pytest.mark.skipif(
    not _HAVE_MODEL, reason=f"real Qwen tokenizer dir not present: {_MODEL}"
)


def _real_template_and_enc():
    pytest.importorskip("transformers")
    from executorch.extension.llm.server.python.chat_template import ChatTemplate
    from transformers import AutoTokenizer

    tmpl = ChatTemplate(hf_tokenizer_path=_MODEL)
    tok = AutoTokenizer.from_pretrained(_MODEL)
    # Encode the way the worker does: no extra special tokens (the rendered text
    # already contains the literal <|im_*|> / <think> control strings).
    return tmpl, (lambda s: tok.encode(s, add_special_tokens=False))


def _assemble(segs, enc):
    out = []
    for seg in segs:
        out += seg["ids"] if "ids" in seg else enc(seg["text"])
    return out


_GEMMA_MODEL = os.environ.get(
    "GEMMA_HF_DIR", "/home/mnachin/local/scripts/models/gemma-4-31B-it-HQQ-INT4"
)
_HAVE_GEMMA = os.path.isdir(_GEMMA_MODEL)
_skip_gemma = pytest.mark.skipif(
    not _HAVE_GEMMA, reason=f"real Gemma tokenizer dir not present: {_GEMMA_MODEL}"
)


def _real_gemma_template_and_enc(strip_bos=True):
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tmpl = ChatTemplate(
        hf_tokenizer_path=_GEMMA_MODEL,
        assistant_header=GEMMA_HDR,
        strip_rendered_bos=strip_bos,
    )
    tok = AutoTokenizer.from_pretrained(_GEMMA_MODEL)
    return tmpl, tok, (lambda s: tok.encode(s, add_special_tokens=False))


@_skip_gemma
def test_gemma_real_template_bos_strip_matches_full_render_ids():
    stripped, tok, enc = _real_gemma_template_and_enc(strip_bos=True)
    full, _, _ = _real_gemma_template_and_enc(strip_bos=False)
    msgs = _msgs(("user", "What is the capital of France?"))
    full_render = full.render(msgs)
    stripped_render = stripped.render(msgs)

    assert full_render.startswith(tok.bos_token)
    assert not stripped_render.startswith(tok.bos_token)
    assert [tok.bos_token_id] + enc(stripped_render) == enc(full_render)


@_skip_gemma
def test_gemma_real_template_warm_resume_prefix_with_bos_prefix():
    tmpl, tok, enc = _real_gemma_template_and_enc(strip_bos=True)
    st = OpenAITranscriptState(tmpl)
    content = "The capital is Paris."
    gen_ids = enc(content)
    first_prompt = tmpl.render(_msgs(("user", "u1")))
    resident = [tok.bos_token_id] + enc(first_prompt) + gen_ids
    st.record_assistant_turn(
        session_id="s",
        content=content,
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble=tmpl.generation_preamble(),
    )
    msgs = _msgs(("user", "u1"), ("assistant", content), ("user", "u2"))
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=tmpl.render(msgs),
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None
    assembled = [tok.bos_token_id] + _assemble(pi.segments, enc)
    assert assembled[: len(resident)] == resident


@_skip_gemma
def test_gemma_real_template_post_tool_splices_call_and_keeps_tool_response():
    tmpl, tok, enc = _real_gemma_template_and_enc(strip_bos=True)
    st = OpenAITranscriptState(tmpl)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    call = ToolCall(
        id="call_1",
        function=FunctionCall(name="bash", arguments='{"command":"echo hello42"}'),
    )
    raw_call = '<|tool_call>call:bash{command:<|"|>echo hello42<|"|>}<tool_call|>'
    gen_ids = enc(raw_call)
    first_prompt = tmpl.render(_msgs(("user", "Run echo hello42")), tools=tools)
    resident = [tok.bos_token_id] + enc(first_prompt) + gen_ids
    st.record_assistant_turn(
        session_id="s",
        content="",
        tool_calls=[call],
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble=tmpl.generation_preamble(tools=tools),
    )
    msgs = [
        ChatMessage(role="user", content="Run echo hello42"),
        ChatMessage(role="assistant", content="", tool_calls=[call]),
        ChatMessage(role="tool", tool_call_id="call_1", content="hello42"),
    ]
    rendered = tmpl.render(msgs, tools=tools)
    assert rendered.endswith("<tool_response|>")

    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=tools,
        template_kwargs=None,
    )
    assert pi.segments is not None
    assembled = [tok.bos_token_id] + _assemble(pi.segments, enc)
    assert assembled[: len(resident)] == resident

    suffix_text = "".join(seg.get("text", "") for seg in pi.segments)
    assert "<|tool_response>" in suffix_text
    assert "hello42" in suffix_text


@_skip
@pytest.mark.parametrize("thinking", [False, True])
def test_token_level_exact_prefix_ordinary(thinking):
    tmpl, enc = _real_template_and_enc()
    kw = {"enable_thinking": thinking}
    st = OpenAITranscriptState(tmpl)
    content = "Mercury, Venus, Earth."
    gen_ids = enc(content)  # stand-in for the worker's generated_token_ids
    gen_prompt1 = tmpl.render(_msgs(("user", "u1")), template_kwargs=kw)
    resident = enc(gen_prompt1) + gen_ids
    st.record_assistant_turn(
        session_id="s",
        content=content,
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble=tmpl.generation_preamble(kw),
    )
    msgs = _msgs(("user", "u1"), ("assistant", content), ("user", "u2"))
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=tmpl.render(msgs, template_kwargs=kw),
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is not None
    assembled = _assemble(pi.segments, enc)
    # resident is an exact token prefix => plan_prefill returns exact_prefix and
    # reuses exactly len(resident) tokens.
    assert assembled[: len(resident)] == resident


@_skip
def test_token_level_exact_prefix_toolloop_think():
    # Mandatory: post-last-user turn where the template preserves a think block
    # before the sentinel; the fix must normalize it to the stored open-think
    # preamble so the token prefix still lands.
    tmpl, enc = _real_template_and_enc()
    kw = {"enable_thinking": True}
    st = OpenAITranscriptState(tmpl)
    content = "result is 42"
    gen_ids = enc(content)
    gen_prompt1 = tmpl.render(_msgs(("user", "u1")), template_kwargs=kw)
    resident = enc(gen_prompt1) + gen_ids
    st.record_assistant_turn(
        session_id="s",
        content=content,
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble=tmpl.generation_preamble(kw),
    )
    msgs = _msgs(("user", "u1"), ("assistant", content))  # a1 AFTER last user
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=tmpl.render(msgs, template_kwargs=kw),
        tools=None,
        template_kwargs=kw,
    )
    assert pi.segments is not None
    assembled = _assemble(pi.segments, enc)
    assert assembled[: len(resident)] == resident


# --- generation_preamble threads tools ------------------------------------


def test_generation_preamble_threads_tools():
    # generation_preamble must pass `tools` to the render probe and key the cache
    # on them, so a template whose post-header scaffold depends on tools gets the
    # right preamble for each (and never serves a stale cached one).
    class _ToolScaffoldTok:
        eos_token = "<|im_end|>"
        all_special_tokens = ["<|im_end|>"]

        def encode(self, text, add_special_tokens=False):
            return [0]

        def apply_chat_template(
            self, messages, tools, add_generation_prompt, tokenize, **kwargs
        ):
            scaffold = "<tools-on>" if tools else "<tools-off>"
            return "<|im_start|>assistant\n" + scaffold

    t = ChatTemplate(hf_tokenizer_path=None, allow_fallback=True)
    t._hf = _ToolScaffoldTok()
    assert t.generation_preamble(tools=None) == "<tools-off>"
    assert (
        t.generation_preamble(tools=[{"type": "function", "function": {"name": "f"}}])
        == "<tools-on>"
    )
    # cached separately -> the no-tool value is not shadowed by the tool one
    assert t.generation_preamble(tools=None) == "<tools-off>"
