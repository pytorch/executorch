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

from executorch.examples.llm_server.python.chat_template import ChatTemplate
from executorch.examples.llm_server.python.openai_transcript import (
    OpenAITranscriptState,
)
from executorch.examples.llm_server.python.protocol import (
    ChatMessage,
    FunctionCall,
    ToolCall,
)
from tokenizers import decoders, models, Tokenizer

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

    def assistant_header(self):
        return self.OHDR

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
    def __init__(self, header=GEMMA_HDR):
        self._header = header

    def assistant_header(self):
        return self._header

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
    # A correctly configured non-ChatML header still supports splicing.
    fake = _FakeOtherHeader()
    st = OpenAITranscriptState(fake)
    enc = _ByteTokenizer.encode
    gen_ids = enc("a1")
    resident = enc(fake.render(_msgs(("user", "u1")))) + gen_ids
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=gen_ids,
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
    assembled = _assemble(pi.segments, enc)
    assert assembled == enc(fake.render(msgs))
    assert assembled[: len(resident)] == resident
    assert any(s.get("ids") == gen_ids for s in pi.segments)


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


@pytest.mark.parametrize("header", [GEMMA_HDR, HDR], ids=["matched", "mismatched"])
def test_gemma_tool_span_ignores_close_marker_inside_string(header):
    st = OpenAITranscriptState(_FakeGemma(header))
    enc = _ByteTokenizer.encode
    raw = '<|tool_call>call:bash{command:<|"|>printf <tool_call|> ok<|"|>}<tool_call|>'
    gen_ids = enc(raw)
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
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble="",
    )
    msgs = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="", tool_calls=[call]),
        ChatMessage(role="tool", tool_call_id="call_1", content="done"),
    ]
    first_prompt = "<bos><|turn>user\nu1<turn|>\n<|turn>model\n"
    trailing = "<|tool_response>done<tool_response|>"
    rendered = first_prompt + raw + trailing
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assembled = enc(pi.text) if pi.segments is None else _assemble_ids(pi.segments)
    assert assembled == enc(first_prompt) + gen_ids + enc(trailing)
    if header != GEMMA_HDR:
        assert pi.segments is None
        assert pi.text == rendered
        return
    assert pi.segments is not None
    assert any(s.get("ids") == gen_ids for s in pi.segments)
    suffix = "".join(
        s.get("text", "") for s in pi.segments[_ids_index(pi.segments, gen_ids) + 1 :]
    )
    assert suffix == "<|tool_response>done<tool_response|>"


# --- 5b. Token-level fidelity against the real tokenizer (gated/skipped) -----

_MODEL = os.environ.get("QWEN_HF_DIR", "")
_HAVE_MODEL = bool(_MODEL) and os.path.isdir(_MODEL)
_skip = pytest.mark.skipif(
    not _HAVE_MODEL, reason="set QWEN_HF_DIR to a local Qwen tokenizer directory"
)


def _real_template_and_enc():
    pytest.importorskip("transformers")
    from executorch.examples.llm_server.python.chat_template import ChatTemplate
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


_GEMMA_MODEL = os.environ.get("GEMMA_HF_DIR", "")
_HAVE_GEMMA = bool(_GEMMA_MODEL) and os.path.isdir(_GEMMA_MODEL)
_skip_gemma = pytest.mark.skipif(
    not _HAVE_GEMMA, reason="set GEMMA_HF_DIR to a local Gemma tokenizer directory"
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


# --- Harmony/ATEM thinking turns (no generation scaffold) ------------------


class _FakeHarmony:
    """Mimics a Harmony/ATEM template: the generation prompt ends at the bare
    assistant header (preamble ''), and an echoed thinking turn re-renders its
    reasoning block first. The raw generation opened with the thinking
    recipient (` to=self...`), but the sentinel substitution drops
    reasoning_content, so its re-render frames the turn as `to=user` -- the
    splice must drop that framing for the stored raw ids to extend the
    resident prefix exactly."""

    HDR = "<|start|>assistant"

    def assistant_header(self):
        return self.HDR

    def render(self, messages, tools=None, template_kwargs=None):
        out = ["<bos>"]
        for m in messages:
            c = m.content if isinstance(m.content, str) else ""
            if m.role == "assistant":
                r = m.reasoning_content
                if r:
                    out.append(f"{self.HDR} to=self<|message|>{r}<|eom|>")
                out.append(f"{self.HDR} to=user<|message|>{c}<|eot|>")
            else:
                out.append(f"<|start|>{m.role}<|message|>{c}<|eot|>")
        out.append(self.HDR)
        return "".join(out)


class _ByteTokenizer:
    """Deterministic encoding for assembly assertions: distinct strings get
    distinct id sequences, so id equality implies string equality."""

    @staticmethod
    def encode(text):
        return [ord(c) for c in text]


def _assemble_ids(segments):
    out = []
    for seg in segments:
        out += _ByteTokenizer.encode(seg["text"]) if "text" in seg else list(seg["ids"])
    return out


def _harmony_resident(fake, raw):
    # What the worker holds after turn 1: the rendered prompt plus the exact
    # raw turn bytes, as token ids. The test double's render is the setup;
    # assertions below compare id sequences only.
    first = fake.render(_msgs(("user", "u1")))
    gen_ids = _ByteTokenizer.encode(raw)
    return _ByteTokenizer.encode(first) + gen_ids, gen_ids


def test_thinking_turn_splice_reproduces_resident_prefix():
    # The raw turn opened with ` to=self...` while the sentinel re-render
    # frames as `to=user`; the assembled prompt must still reproduce the
    # resident prefix exactly (no segment peeking -- pure id equality).
    # The recorded ids are the worker's non-terminal generated ids (the
    # terminal <|eot|> is NOT generated -- worker_loop.h); the trailing
    # turn-2 text supplies the terminator. Full-prompt equality (not just
    # a prefix check) so a duplicated/missing terminator fails loudly.
    fake = _FakeHarmony()
    st = OpenAITranscriptState(fake)
    raw = (
        " to=self<|message|>\nthink\n<|eom|>" "<|start|>assistant to=user<|message|>a1"
    )
    resident, gen_ids = _harmony_resident(fake, raw)
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble="",
        reasoning_content="\nthink\n",
    )
    msgs = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="a1", reasoning_content="\nthink\n"),
        ChatMessage(role="user", content="u2"),
    ]
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=fake.render(msgs),
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None
    trailing = "<|eot|><|start|>user<|message|>u2<|eot|><|start|>assistant"
    assert _assemble_ids(pi.segments) == resident + _ByteTokenizer.encode(trailing)
    # Realism pins (worker_loop.h: generated ids exclude the terminal EOS):
    # the recorded raw carries no terminator; the trailing text supplies it.
    # Without these, eot-including test data would bless a duplicated <|eot|>.
    assert not raw.endswith("<|eot|>")
    assert trailing.startswith("<|eot|>")


def test_plain_turn_splice_reproduces_resident_prefix():
    # A non-thinking turn's raw opening matches the template framing; same
    # assembly assertion as above, proving the framing drop is a no-op there.
    # Realistic non-terminal ids + full-prompt equality, as above.
    fake = _FakeHarmony()
    st = OpenAITranscriptState(fake)
    raw = " to=user<|message|>a1"
    resident, gen_ids = _harmony_resident(fake, raw)
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble="",
    )
    msgs = _msgs(("user", "u1"), ("assistant", "a1"), ("user", "u2"))
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=fake.render(msgs),
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None
    trailing = "<|eot|><|start|>user<|message|>u2<|eot|><|start|>assistant"
    assert _assemble_ids(pi.segments) == resident + _ByteTokenizer.encode(trailing)
    # Realism pins (worker_loop.h: generated ids exclude the terminal EOS):
    # the recorded raw carries no terminator; the trailing text supplies it.
    # Without these, eot-including test data would bless a duplicated <|eot|>.
    assert not raw.endswith("<|eot|>")
    assert trailing.startswith("<|eot|>")


@pytest.mark.parametrize(
    "recorded_reasoning, echoed_fields, matches",
    [
        pytest.param("ORIGINAL", {}, True, id="omitted"),
        pytest.param(
            "ORIGINAL", {"reasoning_content": "ORIGINAL"}, True, id="unchanged"
        ),
        pytest.param("ORIGINAL", {"reasoning_content": "EDITED"}, False, id="changed"),
        pytest.param(
            "ORIGINAL", {"reasoning_content": "ORIGINAL "}, False, id="whitespace"
        ),
        pytest.param("ORIGINAL", {"reasoning_content": ""}, False, id="empty"),
        pytest.param("ORIGINAL", {"reasoning_content": None}, True, id="null"),
        pytest.param(
            "line 1\nline 2",
            {"reasoning_content": "line 1\r\nline 2"},
            False,
            id="line-endings",
        ),
        pytest.param(None, {"reasoning_content": None}, True, id="unchanged-null"),
        pytest.param(None, {"reasoning_content": ""}, False, id="null-to-empty"),
    ],
)
def test_reasoning_echo_must_match_when_explicit(
    recorded_reasoning, echoed_fields, matches
):
    fake = _FakeHarmony()
    st = OpenAITranscriptState(fake)
    raw = " to=user<|message|>a1"
    if recorded_reasoning is not None:
        raw = (
            " to=self<|message|>" + recorded_reasoning + "<|eom|>"
            "<|start|>assistant" + raw
        )
    resident, gen_ids = _harmony_resident(fake, raw)
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble="",
        reasoning_content=recorded_reasoning,
    )
    msgs = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="a1", **echoed_fields),
        ChatMessage(role="user", content="u2"),
    ]
    rendered = fake.render(msgs)
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assembled = (
        _ByteTokenizer.encode(pi.text)
        if pi.segments is None
        else _assemble_ids(pi.segments)
    )
    if matches:
        assert pi.segments is not None
        trailing = "<|eot|><|start|>user<|message|>u2<|eot|><|start|>assistant"
        assert assembled == resident + _ByteTokenizer.encode(trailing)
    else:
        assert assembled == _ByteTokenizer.encode(rendered)
        assert pi.segments is None
        assert pi.text == rendered


def test_reasoning_edit_invalidates_stored_tail_without_later_resurrection():
    fake = _FakeHarmony()
    st = OpenAITranscriptState(fake)
    enc = _ByteTokenizer.encode
    msgs = []
    generated = []
    for i in range(1, 4):
        reasoning = f"reasoning {i}"
        raw = (
            " to=self<|message|>" + reasoning + "<|eom|>"
            f"<|start|>assistant to=user<|message|>a{i}"
        )
        gen_ids = enc(raw)
        generated.append(gen_ids)
        st.record_assistant_turn(
            session_id="s",
            content=f"a{i}",
            tool_calls=None,
            generated_token_ids=gen_ids,
            prior_turns=i - 1,
            preamble="",
            reasoning_content=reasoning,
        )
        msgs.extend(
            [
                ChatMessage(role="user", content=f"u{i}"),
                ChatMessage(
                    role="assistant", content=f"a{i}", reasoning_content=reasoning
                ),
            ]
        )
    msgs.append(ChatMessage(role="user", content="u4"))
    original = msgs[3]
    msgs[3] = original.model_copy(update={"reasoning_content": "EDITED"})
    for second_turn in (msgs[3], original):
        msgs[3] = second_turn
        rendered = fake.render(msgs)
        pi = st.build_prompt_input(
            session_id="s",
            messages=msgs,
            rendered_prompt=rendered,
            tools=None,
            template_kwargs=None,
        )
        assert pi.segments is not None
        assert _assemble_ids(pi.segments) == enc(rendered)
        assert [s["ids"] for s in pi.segments if "ids" in s] == [generated[0]]


@pytest.mark.parametrize("thinking", [False, True])
def test_harmony_bpe_splice_preserves_resident_ids_and_full_prompt(thinking):
    # A tiny real BPE with a merge crossing the prompt/generation boundary.
    # Assets and training are unnecessary; every token and merge is explicit.
    specials = ["<bos>", "<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
    vocab = {chr(i): i for i in range(128)}
    vocab.update({token: len(vocab) + i for i, token in enumerate(specials)})
    vocab["t "] = len(vocab)
    vocab["ab"] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab, merges=[("t", " "), ("a", "b")]))
    tokenizer.add_special_tokens(specials)
    tokenizer.decoder = decoders.Fuse()

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    fake = _FakeHarmony()
    first_prompt = fake.render(_msgs(("user", "u1")))
    reasoning = "\nthink\n" if thinking else None
    raw_prefix = " to=user<|message|>"
    if thinking:
        raw_prefix = (
            " to=self<|message|>" + reasoning + "<|eom|>"
            "<|start|>assistant" + raw_prefix
        )
    raw = raw_prefix + "ab"
    # Valid generated tokens need not use the encoder's canonical segmentation.
    # These decode faithfully, but re-encoding would merge the final a + b.
    gen_ids = enc(raw_prefix) + [vocab["a"], vocab["b"]]
    assert tokenizer.decode(gen_ids, skip_special_tokens=False) == raw
    assert gen_ids != enc(raw)
    assert vocab["<|eot|>"] not in gen_ids
    assert enc(first_prompt + " ") != enc(first_prompt) + enc(" ")
    resident = enc(first_prompt) + gen_ids

    st = OpenAITranscriptState(fake)
    st.record_assistant_turn(
        session_id="s",
        content="ab",
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble="",
        reasoning_content=reasoning,
    )
    msgs = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="ab", reasoning_content=reasoning),
        ChatMessage(role="user", content="u2"),
    ]
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=fake.render(msgs),
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None
    trailing = "<|eot|><|start|>user<|message|>u2<|eot|><|start|>assistant"
    assembled = _assemble(pi.segments, enc)
    assert assembled[: len(resident)] == resident
    assert assembled == resident + enc(trailing)
    full_text = first_prompt + raw + trailing
    assert tokenizer.decode(assembled, skip_special_tokens=False) == full_text
    assert enc(full_text) != assembled  # Text equality cannot establish reuse.


class _FakeMismatchedHeader:
    """Mirrors the production failure mode: the adapter provides
    assistant_header() (production adapters always do) but it returns the
    default ChatML header while the template renders Harmony-style framing.
    A header-like literal inside *message content* must never be treated as
    the turn boundary.
    """

    # The default header: correct for ChatML, wrong for this template.
    HDR = "<|im_start|>assistant\n"

    def __init__(self, header=HDR):
        self._header = header

    def assistant_header(self):
        return self._header

    def render(self, messages, tools=None, template_kwargs=None):
        out = ["<bos>"]
        for m in messages:
            c = m.content if isinstance(m.content, str) else ""
            if m.role == "assistant":
                out.append(f"<|start|>assistant to=user<|message|>{c}<|eot|>")
            else:
                out.append(f"<|start|>user<|message|>{c}<|eot|>")
        out.append("<|start|>assistant")
        return "".join(out)


class _FakeMistralNemo:
    """Mistral-Nemo-style template (cf. llama.cpp's
    mistralai-Mistral-Nemo-Instruct-2407.jinja): user turns render as
    `[INST]{content}[/INST]`, assistant turns as `{content}</s>` -- there is
    no assistant header at all. With the default ChatML header configured, a
    quoted header literal leaves a `KEEP[/INST]` tail that must fall back
    rather than delete KEEP and the instruction boundary.
    """

    HDR = "<|im_start|>assistant\n"  # default, wrong for this template

    def assistant_header(self):
        return self.HDR

    def render(self, messages, tools=None, template_kwargs=None):
        out = ["<bos>"]
        for m in messages:
            c = m.content if isinstance(m.content, str) else ""
            if m.role == "assistant":
                out.append(f"{c}</s>")
            else:
                out.append(f"[INST]{c}[/INST]")
        return "".join(out)


def test_user_header_literal_does_not_truncate_conversation():
    # A user message quoting the literal '<|im_start|>assistant\n' matches
    # rfind() on the default header; the remainder after it --
    # `KEEP<|eot|><|start|>assistant to=user<|message|>` -- is short and
    # single-line, so length/newline guards pass. It still must not truncate:
    # the remainder carries real turn structure (terminator + role boundary),
    # so the splice falls back to plain text (correct output, no reuse)
    # instead of deleting KEEP and the role boundary while splicing.
    st = OpenAITranscriptState(_FakeMismatchedHeader())
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=[5, 6],
        prior_turns=0,
        preamble="",
    )
    u1 = "Explain <|im_start|>assistant\nKEEP"
    msgs = _msgs(("user", u1), ("assistant", "a1"), ("user", "u2"))
    rendered = st._template.render(msgs)
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is None and pi.text == rendered
    assert "KEEP" in pi.text


@pytest.mark.parametrize(
    "header",
    [HDR, "", _FakeHarmony.HDR],
    ids=["mismatched", "empty", "matched"],
)
def test_splice_requires_verified_header_to_preserve_full_prompt(header):
    fake = _FakeMismatchedHeader(header)
    st = OpenAITranscriptState(fake)
    enc = _ByteTokenizer.encode
    # Generated IDs include recipient framing, but exclude the terminal EOT.
    # Keeping the rendered recipient as well would duplicate model input even
    # if the worker's resident-prefix check subsequently chooses a cold prefill.
    gen_ids = enc(" to=user<|message|>a1")
    resident = enc(fake.render(_msgs(("user", "plain question")))) + gen_ids
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble="",
    )
    msgs = _msgs(("user", "plain question"), ("assistant", "a1"), ("user", "u2"))
    rendered = fake.render(msgs)
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assembled = enc(pi.text) if pi.segments is None else _assemble_ids(pi.segments)
    trailing = "<|eot|><|start|>user<|message|>u2<|eot|><|start|>assistant"
    assert assembled == resident + enc(trailing)
    assert assembled == enc(rendered)
    if header == _FakeHarmony.HDR:
        assert pi.segments is not None
        assert any(s.get("ids") == gen_ids for s in pi.segments)
    else:
        assert pi.segments is None
        assert pi.text == rendered


@pytest.mark.parametrize(
    "u1", ["plain question", "Explain <|im_start|>assistant\nKEEP"]
)
def test_mistral_bracket_terminator_tail_falls_back(u1):
    # Nemo-style `[/INST]` terminators are turn structure no blocklist can
    # enumerate exhaustively: the `KEEP[/INST]` tail is short and single-line
    # yet must fall back, preserving KEEP and the instruction boundary.
    fake = _FakeMistralNemo()
    st = OpenAITranscriptState(fake)
    enc = _ByteTokenizer.encode
    gen_ids = enc("a1")
    resident = enc(fake.render(_msgs(("user", u1)))) + gen_ids
    st.record_assistant_turn(
        session_id="s",
        content="a1",
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble="",
    )
    msgs = _msgs(("user", u1), ("assistant", "a1"), ("user", "u2"))
    rendered = st._template.render(msgs)
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is None and pi.text == rendered
    assert enc(pi.text) == resident + enc("</s>[INST]u2[/INST]")


@pytest.mark.parametrize(
    "template_cls, header",
    [(_FakeOtherHeader, _FakeOtherHeader.OHDR), (_FakeMistralNemo, "[/INST]")],
    ids=["llama", "mistral"],
)
@pytest.mark.parametrize("configured", [False, True])
def test_non_chatml_header_configuration_preserves_generated_bpe_tokens(
    template_cls, header, configured
):
    vocab = {chr(i): i for i in range(128)}
    vocab["ab"] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab, merges=[("a", "b")]))
    tokenizer.decoder = decoders.Fuse()
    fake = template_cls()

    class TemplateTokenizer:
        def apply_chat_template(self, messages, tools, **kwargs):
            return fake.render([ChatMessage(**m) for m in messages], tools=tools)

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    template = ChatTemplate(
        allow_fallback=True, assistant_header=header if configured else HDR
    )
    template._hf = TemplateTokenizer()
    state = OpenAITranscriptState(template)
    first_prompt = template.render(_msgs(("user", "u1")))
    # Decoding preserves the answer, but re-encoding merges these two tokens.
    generated = [vocab["a"], vocab["b"]]
    assert tokenizer.decode(generated) == "ab"
    assert generated != enc("ab")
    resident = enc(first_prompt) + generated
    state.record_assistant_turn(
        session_id="s",
        content="ab",
        tool_calls=None,
        generated_token_ids=generated,
        prior_turns=0,
        preamble=template.generation_preamble(),
    )
    messages = _msgs(("user", "u1"), ("assistant", "ab"), ("user", "u2"))
    rendered = template.render(messages)
    prompt = state.build_prompt_input(
        session_id="s",
        messages=messages,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assembled = (
        enc(prompt.text) if prompt.text is not None else _assemble(prompt.segments, enc)
    )
    assert tokenizer.decode(assembled) == rendered
    if configured:
        assert prompt.segments is not None
        suffix = rendered[len(first_prompt + "ab") :]
        assert assembled == resident + enc(suffix)
    else:
        assert prompt.text == rendered
        assert assembled[: len(resident)] != resident


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


# --- Muse Glimmer real-template thinking-turn prefix (gated/skipped) --------

_GLIMMER_MODEL = os.environ.get("MUSE_GLIMMER_HF_DIR", "")
_HAVE_GLIMMER = bool(_GLIMMER_MODEL) and os.path.isdir(_GLIMMER_MODEL)
_skip_glimmer = pytest.mark.skipif(
    not _HAVE_GLIMMER,
    reason="set MUSE_GLIMMER_HF_DIR to a local Muse Glimmer tokenizer directory",
)


def _real_glimmer_template_and_enc():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tmpl = ChatTemplate(
        hf_tokenizer_path=_GLIMMER_MODEL,
        assistant_header="<|start|>assistant",
        strip_rendered_bos=True,
    )
    tok = AutoTokenizer.from_pretrained(_GLIMMER_MODEL)
    return tmpl, tok, (lambda s: tok.encode(s, add_special_tokens=False))


@_skip_glimmer
def test_glimmer_real_template_thinking_turn_splice_reproduces_resident():
    # End-to-end warm-resume contract for a thinking turn with the real ATEM
    # template + tokenizer: the spliced segments must assemble to the exact
    # resident prompt plus the trailing turn-2 text. Pins the framing fix:
    # the raw turn opened with ` to=self` while the sentinel re-render
    # frames as `to=user`; dropping that framing lets the stored raw ids
    # extend the resident prefix exactly. Before, the worker reported
    # "mismatch" and re-prefilled every turn. The recorded ids are the
    # worker's non-terminal generated ids (no terminal <|eot|>); full-prompt
    # equality so a duplicated/missing terminator fails loudly.
    tmpl, tok, enc = _real_glimmer_template_and_enc()
    assert tmpl.generation_preamble() == ""
    st = OpenAITranscriptState(tmpl)
    u1 = "What is 12*13?"
    first_prompt = tmpl.render(_msgs(("user", u1)))
    assert first_prompt.endswith("<|start|>assistant")
    reasoning = "\nLet me think.\n"
    visible = "156"
    raw = (
        " to=self<|message|>" + reasoning + "<|eom|>"
        "<|start|>assistant to=user<|message|>" + visible
    )
    gen_ids = enc(raw)
    st.record_assistant_turn(
        session_id="s",
        content=visible,
        tool_calls=None,
        generated_token_ids=gen_ids,
        prior_turns=0,
        preamble=tmpl.generation_preamble(),
        reasoning_content=reasoning,
    )
    msgs = [
        ChatMessage(role="user", content=u1),
        ChatMessage(role="assistant", content=visible, reasoning_content=reasoning),
        ChatMessage(role="user", content="And 14*15?"),
    ]
    rendered = tmpl.render(msgs)
    pi = st.build_prompt_input(
        session_id="s",
        messages=msgs,
        rendered_prompt=rendered,
        tools=None,
        template_kwargs=None,
    )
    assert pi.segments is not None
    # Full-prompt assertion using the worker's segment encoding: each text
    # segment encoded standalone, id runs spliced verbatim -- exactly how the
    # worker assembles segments before its exact-prefix check. This verifies
    # token ordering and splice placement, not just bag-of-segments: moving
    # the generated ids elsewhere in the prompt fails loudly.
    # The trailing text (terminator + turn 2 + generation header) is whatever
    # follows the echoed a1 turn's visible content in the template's own
    # render. The anchor is unique in this render (verified); the slice keeps
    # the terminator so a missing/duplicated <|eot|> fails the comparison.
    anchor = visible + "<|eot|>"
    assert rendered.count(anchor) == 1
    trailing = "<|eot|>" + rendered.split(anchor, 1)[1]
    assembled = [tok.bos_token_id] + _assemble(pi.segments, enc)
    expected = [tok.bos_token_id] + enc(first_prompt) + gen_ids + enc(trailing)
    assert assembled == expected
    # Test-data realism pin: the worker reports non-terminal generated ids
    # (worker_loop.h), so the recorded raw must not end with the terminator
    # the trailing text already supplies -- otherwise the test would bless a
    # duplicated <|eot|> in the resident prefix.
    assert not raw.endswith("<|eot|>")
    assert trailing.startswith("<|eot|>")
