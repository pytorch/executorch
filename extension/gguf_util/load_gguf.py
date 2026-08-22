# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gguf
from gguf import GGUFValueType, ReaderTensor


@dataclass
class AttentionArgs:
    head_count: int
    head_count_kv: int
    layer_norm_rms_epsilon: float


@dataclass
class RopeArgs:
    freq_base: float


@dataclass
class GGUFModelArgs:
    arch: str
    embedding_length: int
    block_count: int
    feed_forward_length: int
    vocab_size: int
    attention: AttentionArgs
    rope: RopeArgs


@dataclass
class GGUFWeights:
    tensors: list[ReaderTensor]


def _get_metadata(reader: gguf.GGUFReader) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for idx, field in enumerate(reader.fields.values()):
        val = None
        if field.types[:1] == [GGUFValueType.ARRAY]:
            itype = field.types[-1]
            if itype == GGUFValueType.STRING:
                val = [
                    str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data
                ]
            else:
                val = [pv for idx in field.data for pv in field.parts[idx].tolist()]
        elif field.types[0] == GGUFValueType.STRING:
            val = str(bytes(field.parts[-1]), encoding="utf-8")
        else:
            val = field.parts[-1].tolist()[0]

        metadata[field.name] = val

    return metadata


def _build_model_args(metadata: dict[str, Any]) -> GGUFModelArgs:
    arch = metadata["general.architecture"]

    return GGUFModelArgs(
        arch=arch,
        embedding_length=metadata[f"{arch}.embedding_length"],
        block_count=metadata[f"{arch}.block_count"],
        feed_forward_length=metadata[f"{arch}.feed_forward_length"],
        vocab_size=len(metadata["tokenizer.ggml.tokens"]),
        attention=AttentionArgs(
            head_count=metadata[f"{arch}.attention.head_count"],
            head_count_kv=metadata[f"{arch}.attention.head_count_kv"],
            layer_norm_rms_epsilon=metadata[f"{arch}.attention.layer_norm_rms_epsilon"],
        ),
        rope=RopeArgs(
            # default value from llama2 model definition
            freq_base=metadata.get(f"{arch}.rope.freq_base", 1e4),
        ),
    )


def load_file(gguf_file: str) -> (GGUFModelArgs, GGUFWeights):
    """
    Load a GGUF file and return the model arguments and weights.
    """
    if not Path(gguf_file).is_file():
        raise ValueError(f"Could not find file {gguf_file}")

    reader = gguf.GGUFReader(gguf_file, "r")

    # Step 1: Build GGUFModelArgs
    metadata = _get_metadata(reader)
    model_args = _build_model_args(metadata)

    # Step 2: Build GGUFWeights
    gguf_weights = GGUFWeights(tensors=reader.tensors)

    return (model_args, gguf_weights)
