# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a HuggingFace decoder-only LM through the native backend.

Builds the model with ``AutoModelForCausalLM`` from a HuggingFace model
directory (or only its config, with random weights), wraps it with the static
cache of ``transformers.integrations.executorch``, and exports the
``(input_ids [1, T], cache_position [T])`` signature. Reports which operators
the native partitioner leaves behind.
"""

import argparse
import collections
import logging
import operator

import torch
import torch.nn as nn
from executorch.backends.native import get_default_compile_config
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_submodules,
)
from executorch.exir.native import to_native
from torch.export import Dim, export
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM


def _load(path: str, layers: int, random_weights: bool) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(path)
    text_config = config.get_text_config()
    if layers:
        text_config.num_hidden_layers = layers
        if getattr(text_config, "layer_types", None):
            text_config.layer_types = text_config.layer_types[:layers]
    kwargs = {"torch_dtype": torch.float32, "attn_implementation": "sdpa"}
    if random_weights:
        model = AutoModelForCausalLM.from_config(config, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, config=config, **kwargs)
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    return model.eval()


class LastTokenLM(nn.Module):
    """The transformers static-cache module, returning only the last position's
    logits; its own forward computes them for every prompt token."""

    def __init__(self, model: PreTrainedModel, max_cache_len: int) -> None:
        super().__init__()
        # TODO(PTN): attention exports as masked, non-causal GQA aten.sdpa over
        # the whole static cache; engines must pattern-match it back to causal
        # SDPA.
        self.cached = TorchExportableModuleForDecoderOnlyLM(
            model, batch_size=1, max_cache_len=max_cache_len
        ).model
        self.cache = (
            self.cached.static_cache
            if hasattr(self.cached, "static_cache")
            else self.cached.cache
        )

    def forward(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        return self.cached.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits


# TEMPORARY: reproduces the ET-VK llama recipe (`-qmode 8da4w -G 128 -E 4,32`)
# with torchao configs, until PTN export has its own quantization flow.
def _quantize_8da4w(
    model: PreTrainedModel, group_size: int, embedding_group_size: int
) -> None:
    quantize_(
        model,
        IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(embedding_group_size),
            intx_choose_qparams_algorithm="hqq_scale_only",
        ),
        filter_fn=lambda m, _: isinstance(m, nn.Embedding),
    )
    quantize_(
        model,
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(group_size),
            intx_choose_qparams_algorithm="hqq_scale_only",
        ),
        filter_fn=lambda m, _: isinstance(m, nn.Linear)
        and m.in_features % group_size == 0,
    )


def _histogram(gm: torch.fx.GraphModule) -> collections.Counter:
    return collections.Counter(
        str(n.target)
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target not in (executorch_call_delegate, operator.getitem)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="HuggingFace model directory or hub ID."
    )
    parser.add_argument(
        "--random_weights",
        action="store_true",
        help="Use only the model's config, with random weights.",
    )
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--max_context_length", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--non_strict", action="store_true")
    parser.add_argument("--quantize", choices=["none", "8da4w"], default="none")
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--embedding_group_size", type=int, default=32)
    args = parser.parse_args()
    cache_len = args.max_context_length or args.max_seq_length
    if cache_len < args.max_seq_length - 1:
        parser.error("the context must hold a full prompt of max_seq_length - 1 tokens")
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    model = _load(args.model, args.layers, args.random_weights)
    if args.quantize == "8da4w":
        _quantize_8da4w(model, args.group_size, args.embedding_group_size)
    wrapper = LastTokenLM(model, cache_len).eval()

    input_ids = torch.zeros((1, 6), dtype=torch.long)
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.long)
    seq = Dim("seq", min=1, max=args.max_seq_length - 1)
    with torch.no_grad():
        ep = export(
            wrapper,
            (input_ids, cache_position),
            dynamic_shapes={"input_ids": {1: seq}, "cache_position": {0: seq}},
            strict=not args.non_strict,
        )
    print("== exported aten ops ==", flush=True)
    for k, v in sorted(_histogram(ep.graph_module).items()):
        print(f"  {v:4d} {k}")

    edge = to_edge_transform_and_lower(
        ep,
        transform_passes=get_default_passes(),
        partitioner=[
            NativePartitioner(external_constants_tag=None, _serialize_as_ptn=True)
        ],
        compile_config=get_default_compile_config(),
    )
    gm = edge.exported_program().graph_module
    lowered = get_lowered_submodules(gm)
    print(f"== {len(lowered)} native delegate(s) ==")
    delegated = collections.Counter()
    for _, module, _ in lowered:
        delegated += _histogram(module.original_module.graph_module)
    for k, v in sorted(delegated.items()):
        print(f"  {v:4d} {k}")
    print("== residual (not delegated) ==")
    for k, v in sorted(_histogram(gm).items()):
        print(f"  {v:4d} {k}")

    if args.output:
        to_native(ep).save(args.output)
        print(f"saved {args.output}")
