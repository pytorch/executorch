# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-client regressions derived from SGLang and vLLM Muse-Glimmer contracts."""

import json

import pytest

pytest.importorskip("jsonschema", reason="requires llm_server serving dependencies")
pytest.importorskip("regex", reason="requires llm_server serving dependencies")
from executorch.examples.models.muse_glimmer.serving.chat_template import (  # noqa: E402
    MuseGlimmerChatTemplate,
)
from executorch.examples.models.muse_glimmer.serving.stream_parser import (  # noqa: E402
    MuseGlimmerStreamParser,
)
from executorch.examples.models.muse_glimmer.tests.test_atem_tool_parser import (  # noqa: E402
    _atem,
)
from executorch.examples.models.muse_glimmer.tests.test_migration_contracts import (  # noqa: E402
    client_for,
    request,
    response_message,
)


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [1, 7, 10000])
@pytest.mark.parametrize(
    "prefix,suffix", [("", ""), ("after prose:\n", ""), ("Example:\n```xml\n", "\n```")]
)
@pytest.mark.parametrize("real_call", [False, True])
def test_quoted_tool_header_remains_user_content(
    dialect, stream, split, prefix, suffix, real_call
):
    # SGLang: test_pipeline_quoted_header_in_answer_is_not_a_call. A header
    # inside the ongoing user-directed body does not start another message.
    example = (
        prefix
        + "<|start|>assistant to=edit.edit<|message|>"
        + _atem("edit.edit", [("q", "example")])
        + suffix
    )
    raw = (
        " to=self<|message|>Think.<|eom|><|start|>assistant to=user<|message|>"
        + example
    )
    if real_call:
        raw += "<|eom|><|start|>assistant to=edit.edit<|message|>" + _atem(
            "edit.edit", [("q", "real")]
        )
    client, serving = client_for(raw, split)
    serving.api_dialect = dialect
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"q": {"type": "string"}}}, stream),
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert message["content"] == example
    assert message["reasoning_content"] == "Think."
    assert finish == ("tool_calls" if real_call else "stop")
    calls = message.get("tool_calls") or []
    assert [
        (c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls
    ] == ([("edit", {"q": "real"})] if real_call else [])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("recipient", ["self", "user", "edit.edit"])
def test_quoted_header_preserved_without_tools(stream, recipient):
    example = f"<|start|>assistant to={recipient}<|message|>example"
    client, serving = client_for(" to=user<|message|>" + example)
    serving._streaming_parser_factory = MuseGlimmerStreamParser
    body = request({}, stream)
    body.pop("tools")
    response = client.post("/v1/chat/completions", json=body)
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert message["content"] == example
    assert not message.get("reasoning_content")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "value", ["\n```\n", "\n~~~\n", "<|eom|>to=self<|message|>quoted"]
)
def test_skipped_invocation_parameters_cannot_hide_later_real_call(
    stream, wrapped, value
):
    example = _atem("edit.edit", [("q", value)])
    if not wrapped:
        example = example.removeprefix("<atem:function_calls>\n").removesuffix(
            "\n</atem:function_calls>"
        )
    example = "see " + example
    client, _ = client_for(example + "\n" + _atem("edit.edit", [("q", "real")]))
    response = client.post("/v1/chat/completions", json=request({}, stream))
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert message["content"] == example
    assert [json.loads(c["function"]["arguments"]) for c in message["tool_calls"]] == [
        {"q": "real"}
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("placement", ["fence", "parameter"])
def test_plain_chat_keeps_nonterminal_controls_literal(stream, placement):
    from executorch.examples.models.muse_glimmer.serving.framing import CONTROLS
    from executorch.examples.models.muse_glimmer.tests.test_serve import _StubTemplate

    class Template(_StubTemplate):
        def special_tokens(self):
            return list(CONTROLS)

    literal = "<|end|><|channel|><|constrain|>"
    content = (
        "```\n" + literal + "\n```"
        if placement == "fence"
        else _atem("edit.edit", [("q", literal)])
    )
    client, serving = client_for(" to=user<|message|>" + content, template=Template())
    serving._streaming_parser_factory = MuseGlimmerStreamParser
    body = request({}, stream)
    body.pop("tools")
    response = client.post("/v1/chat/completions", json=body)
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert message["content"] == content


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("include", [None, False, True])
@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
def test_null_exclude_leaves_visibility_unset(stream, include, dialect):
    client, serving = client_for(
        " to=self<|message|>thought<|eom|> to=user<|message|>answer"
    )
    serving.api_dialect = dialect
    serving._streaming_parser_factory = MuseGlimmerStreamParser
    body = request({}, stream)
    body.pop("tools")
    body.update(reasoning={"exclude": None}, include_reasoning=include)
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    message, _ = response_message(response, stream)
    assert message["content"] == "answer"
    assert (message.get("reasoning_content") or "") == (
        "" if include is False else "thought"
    )


@pytest.mark.parametrize("stream", [False, True])
def test_rejected_self_header_in_visible_body_remains_literal(stream):
    content = "hi to=self<|message|>quoted thought"
    client, serving = client_for(" to=final<|message|>" + content + "<|eot|>")
    serving._streaming_parser_factory = MuseGlimmerStreamParser
    body = request({}, stream)
    body.pop("tools")
    body["reasoning"] = {"exclude": True}
    response = client.post("/v1/chat/completions", json=body)
    message, _ = response_message(response, stream)
    assert message["content"] == content
    assert not message.get("reasoning_content")


def test_local_muse_tokenizer_reports_reasoning_tokens():
    import os

    path = os.environ.get("MUSE_GLIMMER_HF_DIR")
    if not path:
        pytest.skip("set MUSE_GLIMMER_HF_DIR to a local Muse tokenizer")
    template = MuseGlimmerChatTemplate(path)
    hf = template._hf
    thought_ids = hf.encode("thinking", add_special_tokens=False)
    ids = (
        hf.encode(" to=self<|message|>", add_special_tokens=False)
        + thought_ids
        + hf.encode("<|eom|> to=user<|message|>answer", add_special_tokens=False)
    )
    assert template.count_reasoning_tokens(ids) == len(thought_ids)


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"include_reasoning": False},
        {"reasoning": {"exclude": True}},
        {
            "include_reasoning": False,
            "reasoning": {"exclude": True},
            "chat_template_kwargs": {"return_reasoning": False},
        },
        {"include_reasoning": True},
    ],
)
def test_reasoning_aliases_across_clients(dialect, stream, options):
    client, serving = client_for(
        " to=self<|message|>Think.<|eom|><|start|>assistant to=user<|message|>42"
    )
    serving.api_dialect = dialect
    serving._streaming_parser_factory = MuseGlimmerStreamParser
    body = request({}, stream)
    body.pop("tools")
    body.update(options)
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    message, _ = response_message(response, stream)
    assert message["content"] == "42"
    assert (message.get("reasoning_content") or "") == (
        "Think." if options.get("include_reasoning") is True else ""
    )


@pytest.mark.parametrize(
    "options",
    [
        {"include_reasoning": True, "reasoning": {"exclude": True}},
        {"include_reasoning": False, "reasoning": {"exclude": False}},
        {
            "include_reasoning": False,
            "chat_template_kwargs": {"return_reasoning": True},
        },
        {"include_reasoning": True, "chat_template_kwargs": {"return_reasoning": 1}},
        {"include_reasoning": False, "chat_template_kwargs": {"return_reasoning": 0}},
    ],
)
def test_conflicting_visibility_rejected_before_generation(options):
    client, serving = client_for("unused")
    body = request({})
    body.update(options)
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_value"
    assert not serving._runtime.prompts


@pytest.fixture
def counting_template(tmp_path):
    from tokenizers import decoders, models, pre_tokenizers, Tokenizer
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {token: i for i, token in enumerate(alphabet + ["ab"])}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"],
    )
    hf.save_pretrained(tmp_path)
    return MuseGlimmerChatTemplate(str(tmp_path))


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("include", [False, True])
@pytest.mark.parametrize("tools", [False, True])
def test_reasoning_usage_counts_generated_ids_when_hidden(
    counting_template, dialect, stream, include, tools
):
    template = counting_template
    hf = template._hf
    prefix = " to=self<|message|>"
    suffix = "<|eom|><|start|>assistant to=user<|message|>42"
    # Forced generation of separate a/b IDs: re-encoding "ab" would count one.
    ids = (
        hf.encode(prefix, add_special_tokens=False)
        + [hf.get_vocab()["a"], hf.get_vocab()["b"]]
        + hf.encode(suffix, add_special_tokens=False)
    )
    raw = hf.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    client, serving = client_for(raw, 1)
    serving.api_dialect = dialect
    serving._streaming_parser_factory = MuseGlimmerStreamParser
    serving._reasoning_token_counter = template.count_reasoning_tokens
    serving._runtime._outputs[0].update(gen_ids=ids, completion_tokens=len(ids))
    body = request({}, stream)
    if not tools:
        body.pop("tools")
    body.update(include_reasoning=include, stream_options={"include_usage": True})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    message, _ = response_message(response, stream)
    assert message["content"] == "42"
    assert (message.get("reasoning_content") or "") == ("ab" if include else "")
    if stream:
        usages = [
            json.loads(line[6:])["usage"]
            for line in response.text.splitlines()
            if line.startswith("data: ")
            and line != "data: [DONE]"
            and json.loads(line[6:]).get("usage")
        ]
        assert len(usages) == 1
        usage = usages[0]
    else:
        usage = response.json()["usage"]
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 2
    assert usage["completion_tokens"] == len(ids)
    assert usage["total_tokens"] == usage["prompt_tokens"] + len(ids)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" to=self<|message|>abc", 2),
        (" to=self<|message|>é", 2),
        (" to=self<|message|>abc<|eom|> to=self<|message|>de<|eot|>", 4),
        (" to=user<|message|>abc", 0),
        (" to=user<|message|>Example: <|start|>assistant to=self<|message|>abc", 0),
        (
            " to=edit.edit<|message|>"
            + _atem(
                "edit.edit", [("q", "<|eom|><|start|>assistant to=self<|message|>abc")]
            ),
            0,
        ),
    ],
)
def test_reasoning_count_handles_truncation_and_quoted_headers(
    counting_template, raw, expected
):
    ids = counting_template._hf.encode(raw, add_special_tokens=False)
    assert counting_template.count_reasoning_tokens(ids) == expected


def test_unknown_generated_ids_do_not_claim_zero_reasoning(counting_template):
    client, serving = client_for(
        " to=self<|message|>Think.<|eom|> to=user<|message|>42"
    )
    serving._reasoning_token_counter = counting_template.count_reasoning_tokens
    response = client.post("/v1/chat/completions", json=request({}))
    assert response.status_code == 200
    assert response.json()["usage"].get("completion_tokens_details") is None


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("first", ["first", "```", "<|eom|> to=user<|message|>"])
def test_adjacent_tool_blocks_isolate_parameter_framing(dialect, stream, first):
    raw = _atem("edit.edit", [("q", first)]) + _atem("edit.edit", [("q", "second")])
    client, serving = client_for(raw, 1)
    serving.api_dialect = dialect
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"q": {"type": "string"}}}, stream),
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert not message.get("content")
    assert [json.loads(c["function"]["arguments"]) for c in message["tool_calls"]] == [
        {"q": first},
        {"q": "second"},
    ]


@pytest.mark.parametrize("dialect", ["openai", "openrouter"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("split", [1, 7, 10000])
@pytest.mark.parametrize("prefix", ["Example:\n", ""])
def test_fenced_end_markers_never_execute_examples(dialect, stream, split, prefix):
    example = (
        prefix
        + "```\n<|start|>assistant to=self<|message|>plan<|eom|><|start|>assistant to=edit.edit<|message|>"
        + _atem("edit.edit", [("q", "example")])
        + "\n```"
    )
    client, serving = client_for(" to=user<|message|>" + example, split)
    serving.api_dialect = dialect
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"q": {"type": "string"}}}, stream),
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert not message.get("tool_calls")
    assert not message.get("reasoning_content")
    assert message["content"] == example


@pytest.mark.parametrize("stream", [False, True])
def test_tool_turn_scrubs_specials_and_preserves_quoted_headers(stream):
    quote = "<|start|>assistant to=self<|message|>quoted"
    raw = (
        " to=user<|message|>hello "
        + quote
        + "<|leaked_special|>discard<|eom|><|start|>assistant to=edit.edit<|message|>"
        + _atem("edit.edit", [("q", "ok")])
    )
    client, serving = client_for(raw)
    serving._content_specials = ["<|leaked_special|>"]
    response = client.post("/v1/chat/completions", json=request({}, stream))
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert message["content"] == "hello " + quote


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" to=self<|message|>abc<|eom|>def", 5),
        (" to=self<|message|>abc<|eom|><|start|>assistant to=user<|message|>def", 2),
    ],
)
def test_reasoning_counter_uses_extracted_body_boundaries(
    counting_template, raw, expected
):
    ids = counting_template._hf.encode(raw, add_special_tokens=False)
    assert counting_template.count_reasoning_tokens(ids) == expected


def test_reasoning_counter_omits_non_atomic_markers(counting_template, monkeypatch):
    encode = counting_template._hf.encode
    monkeypatch.setattr(
        counting_template._hf,
        "encode",
        lambda text, **kwargs: (
            [1, 2] if text == "<|message|>" else encode(text, **kwargs)
        ),
    )
    assert counting_template.count_reasoning_tokens([1, 2]) is None


def test_reasoning_counter_omits_unaligned_body_boundary(counting_template):
    hf = counting_template._hf
    hf.add_tokens([" to=self<|message|>"], special_tokens=True)
    ids = hf.encode(" to=self<|message|>abc", add_special_tokens=False)
    assert counting_template.count_reasoning_tokens(ids) is None


@pytest.mark.parametrize("stream", [False, True])
def test_live_inline_backticks_do_not_hide_following_tool_call(stream):
    raw = ' to=self<|message|>So first call report.report with q = "```". That is three backticks.' "<|eom|><|start|>assistant to=edit.edit<|message|>" + _atem(
        "edit.edit", [("q", "```")]
    ) + "<|eom|><|start|>assistant to=edit.edit<|message|>" + _atem(
        "edit.edit", [("q", "second")]
    )
    client, _ = client_for(raw, 1)
    response = client.post(
        "/v1/chat/completions",
        json=request({"properties": {"q": {"type": "string"}}}, stream),
    )
    assert response.status_code == 200
    message, finish = response_message(response, stream)
    assert finish == "tool_calls"
    assert not message.get("content")
    assert [json.loads(c["function"]["arguments"]) for c in message["tool_calls"]] == [
        {"q": "```"},
        {"q": "second"},
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("recipient", ["user", "edit.edit"])
def test_line_start_inline_backticks_allow_following_recipient(stream, recipient):
    answer = "answer" if recipient == "user" else _atem("edit.edit", [("q", "real")])
    thought = "Use\n```q``` here"
    raw = (
        " to=self<|message|>"
        + thought
        + f"<|eom|><|start|>assistant to={recipient}<|message|>"
        + answer
    )
    client, _ = client_for(raw)
    response = client.post("/v1/chat/completions", json=request({}, stream))
    message, finish = response_message(response, stream)
    assert message["reasoning_content"] == thought
    assert finish == ("stop" if recipient == "user" else "tool_calls")
    if recipient == "user":
        assert message["content"] == answer
    else:
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
            "q": "real"
        }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("nested", ["```py", "``` \\t extra", "~~~py"])
def test_fence_with_info_cannot_close_quoted_example(stream, nested):
    example = (
        "Example:\n```\nouter\n"
        + nested
        + "\n<|eom|><|start|>assistant to=edit.edit<|message|>"
        + _atem("edit.edit", [("q", "quoted")])
        + "\n```"
    )
    client, _ = client_for(" to=user<|message|>" + example)
    response = client.post("/v1/chat/completions", json=request({}, stream))
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert not message.get("tool_calls")
    assert message["content"] == example


@pytest.mark.parametrize("stream", [False, True])
def test_real_special_tokens_inside_fenced_tool_turn_stay_literal(stream):
    from executorch.examples.models.muse_glimmer.serving.framing import CONTROLS

    example = "Example:\n```\nfoo" + "bar".join(CONTROLS) + "tail\n```\nafter"
    client, serving = client_for(" to=user<|message|>" + example)
    serving._content_specials = list(CONTROLS)
    response = client.post("/v1/chat/completions", json=request({}, stream))
    message, finish = response_message(response, stream)
    assert finish == "stop"
    assert message["content"] == example


@pytest.mark.parametrize("stream", [False, True])
def test_ignore_eos_does_not_report_unaligned_reasoning_usage(
    counting_template, stream
):
    raw = " to=self<|message|>before<|eot|>hidden after EOS"
    ids = counting_template._hf.encode(raw, add_special_tokens=False)
    client, serving = client_for(raw)
    serving._runtime.supports_ignore_eos = True
    serving._native_stops = ["<|eot|>"]
    serving._reasoning_token_counter = counting_template.count_reasoning_tokens
    serving._runtime._outputs[0].update(
        gen_ids=ids, completion_tokens=len(ids), finish_reason="length"
    )
    body = request({}, stream)
    body.pop("tools")
    body.update(ignore_eos=True, stream_options={"include_usage": True})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        usage = [c["usage"] for c in chunks if c.get("usage")][-1]
    else:
        usage = response.json()["usage"]
    assert usage["completion_tokens"] == len(ids)
    assert "completion_tokens_details" not in usage


def test_cleaned_empty_reasoning_body_has_zero_tokens(counting_template):
    counting_template._hf.add_special_tokens(
        {
            "additional_special_tokens": counting_template._hf.all_special_tokens
            + ["<|channel|>"]
        }
    )
    raw = " to=self<|message|> <|channel|> <|eom|>"
    assert (
        counting_template.count_reasoning_tokens(
            counting_template._hf.encode(raw, add_special_tokens=False)
        )
        == 0
    )
