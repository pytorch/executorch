# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure incremental-parser coverage; no model or native worker is loaded."""

import pytest

pytest.importorskip("pydantic", reason="requires llm_server serving dependencies")

from executorch.examples.llm_server.python.protocol import DeltaMessage  # noqa: E402
from executorch.examples.models.muse_glimmer.serving.serve import (  # noqa: E402
    _extract_muse_glimmer_reasoning,
    _strip_muse_glimmer_header,
)
from executorch.examples.models.muse_glimmer.serving.stream_parser import (  # noqa: E402
    MuseGlimmerStreamParser,
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
    ]
    for trace in traces:
        reasoning, visible = _extract_muse_glimmer_reasoning(trace)
        expected = (reasoning or "", _strip_muse_glimmer_header(visible))
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
        " to=self",
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
