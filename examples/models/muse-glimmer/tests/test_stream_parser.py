# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure incremental-parser coverage; no model or native worker is loaded."""

import pytest

pytest.importorskip("pydantic", reason="requires llm_server serving dependencies")
pytest.importorskip("jsonschema", reason="requires llm_server serving dependencies")
pytest.importorskip("regex", reason="requires llm_server serving dependencies")

from executorch.examples.llm_server.python.protocol import DeltaMessage  # noqa: E402
from executorch.examples.models.muse_glimmer.serving.serve import (  # noqa: E402
    _extract_muse_glimmer_reasoning,
)
from executorch.examples.models.muse_glimmer.serving.stream_parser import (  # noqa: E402
    MuseGlimmerStreamParser,
)


def _check_chunks(trace, expected):
    chunks = [[trace[:split], trace[split:]] for split in range(len(trace) + 1)]
    chunks.append(list(trace))
    for parts in chunks:
        parser = MuseGlimmerStreamParser()
        deltas = [delta for part in parts for delta in parser.feed(part)]
        deltas.extend(parser.finish())
        assert all(isinstance(delta, DeltaMessage) for delta in deltas)
        actual = (
            "".join(delta.reasoning_content or "" for delta in deltas),
            "".join(delta.content or "" for delta in deltas),
        )
        assert actual == expected, parts


def test_batch_equivalence_across_chunk_boundaries():
    traces = [
        " to=self<|message|>  think\nthrough this \t<|eom|>  \n"
        "<|start|>assistant to=user<|message|>  First answer. \n<|eom|>"
        " to=self<|message|> \n<|eom|>"
        " to=self<|message|>more thought\n<|eom|>"
        " to=user<|message|> \t<|eom|>"
        " to=user<|message|> Second answer.  <|eot|>\n",
        " Ordinary prefix: goto=self<|message|>is prose.\n"
        "<|start|>assistant\nto=tools.lookup constrain=json<|message|> Result. "
        "<|eom|>\n<|start|>assistant to=self.notes<|message|> Also visible. "
        "<|eot|>",
        "<|start|>assistant to=self constrain=analysis<|message|>\n"
        "keep <|unknown|> and goto=user verbatim \n"
        "<|start|>assistant to=self<|message|> next \t"
        "<|start|>assistant to=user<|message|>\nline one\nline two \n",
        "set x to=value",
        "set x to=",
        "to=value",
        "plain text to=self",
        "to=user<|message|>set x to=value",
        "to=self<|message|>set x to=value",
    ]
    for trace in traces:
        reasoning, visible = _extract_muse_glimmer_reasoning(trace)
        expected = (reasoning or "", visible.strip())
        _check_chunks(trace, expected)


@pytest.mark.parametrize(
    "trace, streamed, buffered",
    [
        ("answer <|sta", ("", "answer"), ("", "answer <|sta")),
        (
            "to=self<|message|>before<|eom|>after to=user<|message|>answer",
            ("beforeafter to=user<|message|>answer", ""),
            ("beforeafter to=user<|message|>answer", ""),
        ),
        (
            "to=user<|message|>A to=user<|message|><|start|> to=user<|message|>B",
            ("", "A to=user<|message|> to=user<|message|>B"),
            ("", "A to=user<|message|> to=user<|message|>B"),
        ),
    ],
)
def test_intentional_framing_differences(trace, streamed, buffered):
    reasoning, visible = _extract_muse_glimmer_reasoning(trace)
    assert (reasoning or "", visible.strip()) == buffered
    _check_chunks(trace, streamed)


def test_immediate_emission_and_truncated_framing():
    parser = MuseGlimmerStreamParser()
    assert list(parser.feed(" to=se")) == []
    # The first body must arrive when its header resolves, not at the next header.
    assert list(parser.feed("lf<|message|> think")) == [
        DeltaMessage(reasoning_content=" think")
    ]
    assert list(parser.feed(" now<|eom|>  \n")) == [
        DeltaMessage(reasoning_content=" now")
    ]
    assert list(parser.feed("<|start|>assistant to=user<|message|> answer  ")) == [
        DeltaMessage(content="answer")
    ]
    assert list(parser.feed("<|eo")) == []
    assert list(parser.finish()) == []

    for suffix in (
        "<|start|>assistant to=tools.lookup constrain=js",
        "<|message",
    ):
        parser = MuseGlimmerStreamParser()
        deltas = list(parser.feed("plain text " + suffix)) + list(parser.finish())
        assert "".join(delta.content or "" for delta in deltas) == "plain text"
        assert not any(delta.reasoning_content for delta in deltas)

    for text in (" ordinary\nplain text ", "go to", "a <"):
        parser = MuseGlimmerStreamParser()
        deltas = [delta for char in text for delta in parser.feed(char)]
        deltas.extend(parser.finish())
        assert "".join(delta.content or "" for delta in deltas) == text.strip()


def test_tool_reasoning_stream_never_interprets_parameter_headers():
    from executorch.examples.models.muse_glimmer.serving.stream_parser import (
        MuseGlimmerToolReasoningParser,
    )

    raw = (
        " to=self<|message|>Thinking.<|eom|><|start|>assistant to=f<|message|>"
        '<atem:function_calls><atem:invoke name="f"><atem:parameter name="value">'
        " to=self<|message|>MUST_NOT_STREAM</atem:parameter>"
        "</atem:invoke></atem:function_calls>"
    )
    for width in [1, 3, 7, len(raw)]:
        parser = MuseGlimmerToolReasoningParser()
        output = []
        for i in range(0, len(raw), width):
            output.extend(parser.feed(raw[i : i + width]))
        output.extend(parser.finish())
        assert "".join(d.reasoning_content or "" for d in output) == "Thinking."


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("```\n to=self<|message|>x\n```", ("", "```\n to=self<|message|>x\n```")),
        (
            " to=self<|message|>t<|eom|><|end|><|start|>assistant to=user<|message|>42",
            ("t<|start|>assistant to=user<|message|>42", ""),
        ),
        (
            " to=user<|message|>```\n<|start|>assistant to=self<|message|>x<|eom|><|start|>assistant to=user<|message|>y\n```",
            (
                "",
                "```\n<|start|>assistant to=self<|message|>x<|eom|><|start|>assistant to=user<|message|>y\n```",
            ),
        ),
    ],
)
def test_quoted_framing_matches_buffered_for_every_split(raw, expected):
    reasoning, visible = _extract_muse_glimmer_reasoning(raw)
    assert (reasoning or "", visible) == expected
    _check_chunks(raw, expected)
