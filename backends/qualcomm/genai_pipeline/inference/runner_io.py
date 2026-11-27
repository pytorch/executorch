# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference requests a runner adapter executes, and the outputs it collects.

A request names one thing the runner can do -- run a compiled method, generate
text, or forward tokens -- and says only *what* is wanted. *How* it is run --
runner flags, file transfer, KV-cache mode, dequantization -- is the adapter's.
Each request is answered by exactly one output type, defined right after it.

Requests and outputs are frozen: a request is planned by its caller and
executed unchanged, and an output is read, never edited, once the adapter
returns it. Outputs carry decoded data -- text, float logits -- rather than raw
runner files to parse, so a consumer needs no knowledge of the runner.

``RunnerMetadata`` describes the deployed model the requests run against. The
adapter reads it from the PTE, so a consumer needs no knowledge of the PTE
either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ExecuteRequest:
    """Run a compiled method through the generic executor runner on data samples.

    Answered with an ``ExecuteOutput``.

    Attributes:
        inputs: The samples to run, one positional-input tuple each; the runner
            reads them as one input list and runs them in a single invocation.
        method_index: Index of the PTE method to run.
        iteration: Number of times to run it.
    """

    inputs: Tuple[Tuple[Any, ...], ...]
    method_index: int = 0
    iteration: int = 1

    def __post_init__(self) -> None:
        """Reject no samples, a negative method index, or a non-positive iteration count.

        Raises:
            ValueError: If ``inputs`` is empty, ``method_index`` is negative, or
                ``iteration`` is below one.
        """
        if not self.inputs:
            raise ValueError("ExecuteRequest requires at least one input sample")
        if self.method_index < 0:
            raise ValueError(f"method_index must be >= 0, got {self.method_index}")
        if self.iteration < 1:
            raise ValueError(f"iteration must be >= 1, got {self.iteration}")


@dataclass(frozen=True)
class ExecuteOutput:
    """Files pulled after an ``ExecuteRequest``.

    The generic executor runner writes raw output tensors whose layout only the
    caller knows, so they are returned as files rather than decoded.

    Attributes:
        files: Host paths of the pulled output folder's files.
    """

    files: List[Path] = field(default_factory=list)


@dataclass(frozen=True)
class GenerateRequest:
    """Generate text for prompts; the runner applies the chat template.

    Answered with a ``GenerateOutput``. Generation settings -- temperature,
    sequence length, lookahead parameters -- are runner-level and fixed when
    the adapter deploys, so a request carries only the prompts.

    Attributes:
        prompts: Raw user prompts, generated in order within one runner
            invocation.
    """

    prompts: List[str]

    def __post_init__(self) -> None:
        """Reject a request without prompts.

        Raises:
            ValueError: If ``prompts`` is empty.
        """
        if not self.prompts:
            raise ValueError("GenerateRequest needs at least one prompt")


@dataclass(frozen=True)
class GenerateOutput:
    """Generated text for a ``GenerateRequest``.

    Attributes:
        texts: The runner's text output, one entry per runner invocation. A
            ``GenerateRequest`` is one invocation, so this holds a single
            string covering every prompt's response.
        tokens_per_sec: Decode speed the runner reported for the invocation.
        files: Host paths of the collected runner outputs, kept for inspection.
    """

    texts: List[str]
    tokens_per_sec: float
    files: List[Path] = field(default_factory=list)


@dataclass(frozen=True)
class ForwardRequest:
    """Forward the given tokens and return the logits at each of their positions.

    Answered with a ``ForwardOutput``. Only the context length of the deployed
    PTE bounds ``token_ids``; the adapter checks it, since only it reads the PTE.

    Attributes:
        token_ids: Token IDs fed to the runner as-is, already templated.
        seq_len: Runner decode bound. ``None`` stops right after the given
            tokens, which is all that their logits need; a larger bound only
            spends decode steps whose logits are discarded.
    """

    token_ids: List[int]
    seq_len: Optional[int] = None

    def __post_init__(self) -> None:
        """Reject an empty request or a decode bound shorter than its tokens.

        Raises:
            ValueError: If ``token_ids`` is empty, or ``seq_len`` does not
                exceed ``len(token_ids)``.
        """
        if not self.token_ids:
            raise ValueError("ForwardRequest needs at least one token")
        if self.seq_len is not None and self.seq_len <= len(self.token_ids):
            raise ValueError(
                f"seq_len ({self.seq_len}) must exceed the token count "
                f"({len(self.token_ids)})"
            )


@dataclass(frozen=True)
class ForwardOutput:
    """Dequantized logits for a ``ForwardRequest``.

    Attributes:
        logits: Float32 logits shaped ``(1, len(token_ids), vocab)``: one row
            per given token, dequantized with the PTE's logits scale and zero
            point, and cut to the given tokens even when the runner decoded
            further.
        tokens_per_sec: Speed the runner reported for the invocation.
        files: Host paths of the collected runner outputs, kept for inspection;
            consumers read ``logits`` instead of parsing them.
    """

    logits: "torch.Tensor"
    tokens_per_sec: float
    files: List[Path] = field(default_factory=list)


#: Every request an adapter may execute.
RunnerRequest = Union[ExecuteRequest, GenerateRequest, ForwardRequest]

#: Every output an adapter may collect; its type follows the request's.
RunnerOutput = Union[ExecuteOutput, GenerateOutput, ForwardOutput]


@dataclass(frozen=True)
class RunnerMetadata:
    """Metadata the deployed decoder PTE was compiled with.

    Attributes:
        max_context_len: The decoder's context length, which bounds a
            ``ForwardRequest``.
        max_seq_len: The decoder's maximum sequence length.
        vocab_size: Size of the output vocabulary.
        logits_scale: Scale of the quantized output logits; 1 for float.
        logits_zero_point: Zero point of the quantized output logits; 0 for
            float.
        kv_io_bit_width: Bit width of the KV-cache I/O, 32 for float.
    """

    max_context_len: int
    max_seq_len: int
    vocab_size: int
    logits_scale: float
    logits_zero_point: int
    kv_io_bit_width: int
