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


@pytest.mark.parametrize("recipient", ["self", "user"])
@pytest.mark.parametrize(
    "syntax",
    [
        '<atem:invoke name="f">',
        '<atem:invoke name="f"><atem:parameter name="q">value</atem:parameter>',
        '<atem:invoke name="f"><atem:parameter name="q"><|eom|>to=self<|message|>literal',
    ],
)
def test_atem_body_text_blocks_channel_switch(recipient, syntax):
    trace = (
        f" to={recipient}<|message|>before<|eot|>" + syntax + "to=user<|message|>after"
    )
    reasoning, content = _extract_muse_glimmer_reasoning(trace)
    _check_chunks(trace, (reasoning or "", content))


def test_composed_framing_is_independent_of_chunk_boundaries():
    from itertools import product

    fragments = [
        "plain text ",
        "<|eom|>",
        "\n```xml\nquoted\n```\n",
        '<atem:invoke name="f">',
        '<atem:parameter name="q">\n```\n<|eom|>to=user<|message|>quoted</atem:parameter></atem:invoke>',
        "Use `<atem:parameter` tags. ",
        "to=user<|message|>visible",
        "to=self<|message|>thought",
        "<|end|>",
    ]
    for recipient, left, right in product(("self", "user"), fragments, fragments):
        trace = f" to={recipient}<|message|>before<|eom|>" + left + right + " tail"
        reasoning, content = _extract_muse_glimmer_reasoning(trace)
        _check_chunks(trace, (reasoning or "", content))


@pytest.mark.parametrize("marker", ["<atem:invoke", "<atem:parameter"])
def test_bare_atem_mentions_do_not_hold_subsequent_prose(marker):
    parser = MuseGlimmerStreamParser()
    text = "Use `" + marker + "` tags to explain the format."
    deltas = [delta for char in text for delta in parser.feed(char)]
    assert "".join(d.content or "" for d in deltas) == text
    assert list(parser.finish()) == []


def test_long_valid_tag_names_and_unclosed_parameters_have_linear_holds():
    import subprocess
    import sys

    code = """
from executorch.examples.models.muse_glimmer.serving.stream_parser import MuseGlimmerStreamParser
for trace in [
    '<atem:invoke name="' + 'a' * 64000 + '">done',
    '<atem:invoke name="f"><atem:parameter name="q">' + 'x' * 64000,
    'Use `<atem:parameter` tags. ' + 'x' * 64000,
]:
    parser = MuseGlimmerStreamParser()
    deltas = [d for char in trace for d in parser.feed(char)] + list(parser.finish())
    assert ''.join(d.content or '' for d in deltas) == trace
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_tool_reasoning_is_a_safe_prefix_without_assuming_eof_at_atem():
    from itertools import product

    from executorch.examples.models.muse_glimmer.serving.stream_parser import (
        MuseGlimmerToolReasoningParser,
    )

    for prefix, suffix in product(
        ["Think. ", "\n```", " to=", "\n~~~", "<|eom|>\n```"],
        [
            "quoted`\n to=user<|message|>example",
            "\n</atem:invoke>",
            "\n```\n<|eom|> to=user<|message|>answer",
        ],
    ):
        trace = " to=self<|message|>" + prefix + '<atem:invoke name="f">' + suffix
        reasoning, _ = _extract_muse_glimmer_reasoning(trace)
        for split in range(len(trace) + 1):
            parser = MuseGlimmerToolReasoningParser()
            deltas = (
                list(parser.feed(trace[:split]))
                + list(parser.feed(trace[split:]))
                + list(parser.finish())
            )
            emitted = "".join(d.reasoning_content or "" for d in deltas)
            assert (reasoning or "").startswith(emitted), (
                trace,
                split,
                emitted,
                reasoning,
            )


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


@pytest.mark.parametrize("inline", ["```q``` here", "   ```q``` here", "``q``", "`q`"])
def test_line_start_inline_code_preserves_answer_at_every_split(inline):
    thought = "Use\n" + inline
    raw = (
        " to=self<|message|>"
        + thought
        + "<|eom|><|start|>assistant to=user<|message|>answer"
    )
    assert _extract_muse_glimmer_reasoning(raw) == (thought, "answer")
    _check_chunks(raw, (thought, "answer"))


@pytest.mark.parametrize("nested", ["```py", "``` extra", "~~~~"])
def test_invalid_closing_fence_keeps_headers_quoted_at_every_split(nested):
    example = (
        "```\nouter\n"
        + nested
        + "\n<|eom|><|start|>assistant to=self<|message|>quoted\n```"
    )
    raw = " to=user<|message|>" + example
    assert _extract_muse_glimmer_reasoning(raw) == (None, example)
    _check_chunks(raw, ("", example))


def test_parameter_headers_stay_content_without_tools_at_every_split():
    from executorch.examples.models.muse_glimmer.tests.test_atem_tool_parser import (
        _atem,
    )

    example = _atem(
        "edit.edit",
        [("q", "prefix<|eom|><|start|>assistant to=self<|message|>quoted reasoning")],
    )
    raw = " to=edit.edit<|message|>" + example
    assert _extract_muse_glimmer_reasoning(raw) == (None, example)
    _check_chunks(raw, ("", example))


def test_literal_parameter_close_stays_data_at_every_split():
    from executorch.examples.models.muse_glimmer.tests.test_atem_tool_parser import (
        _atem,
    )

    example = _atem(
        "edit.edit",
        [
            (
                "q",
                "before </atem:parameter> more <|eom|><|start|>assistant to=self<|message|>quoted",
            )
        ],
    )
    raw = " to=edit.edit<|message|>" + example
    assert _extract_muse_glimmer_reasoning(raw) == (None, example)
    _check_chunks(raw, ("", example))
