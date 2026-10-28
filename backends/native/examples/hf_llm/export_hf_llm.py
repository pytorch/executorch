# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a supported Hugging Face decoder-only LM through the native backend.

Builds the model with ``AutoModelForCausalLM`` from a Hugging Face model
directory (or only its config, with random weights), wraps it with the cache
adapter from ``transformers.integrations.executorch``, and exports the
``(input_ids [1, T], cache_position [T])`` signature. Reports which operators
the native partitioner leaves behind. ``--gguf`` takes the weights from a GGUF
file of an architecture ``gguf_hf`` supports instead, keeping its quantized blocks
packed in the PTN. ``--samples`` appends a ``golden`` sample set: eager logits
for a prefill and a few greedy decode steps per prompt.
"""

import argparse
import collections
import logging
import operator

import torch
import torch.nn as nn
from executorch.backends.native import get_default_compile_config
from executorch.backends.native.examples.hf_llm.gguf_hf import (
    gguf_architecture,
    gguf_to_hf,
)
from executorch.backends.native.partitioner import NativePartitioner
from executorch.backends.native.passes import get_default_passes
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.lowered_backend_module import (
    executorch_call_delegate,
    get_lowered_submodules,
)
from executorch.exir.native import Call, Sample, SampleSet, to_native, write_samples
from executorch.extension.llm.export.load import assign_one, iter_checkpoint, RawIter
from torch.export import Dim, export
from torchao.quantization import (
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup
from torchao.utils import TorchAOBaseTensor
from transformers import (
    AttentionInterface,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.cache_utils import StaticLayer
from transformers.integrations.accelerate import init_empty_weights
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    AttentionMaskInterface,
)


def _load(
    path: str, layers: int, random_weights: bool, empty_weights: bool
) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(path)
    text_config = config.get_text_config()
    if layers:
        text_config.num_hidden_layers = layers
        if getattr(text_config, "layer_types", None):
            text_config.layer_types = text_config.layer_types[:layers]
    kwargs = {"torch_dtype": torch.float32, "attn_implementation": "sdpa"}
    if empty_weights:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, **kwargs)
    elif random_weights:
        model = AutoModelForCausalLM.from_config(config, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, config=config, **kwargs)
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    return model.eval()


def load_gguf(
    model: PreTrainedModel,
    path: str,
    architecture: str | None = None,
    raw_iter: RawIter | None = None,
) -> None:
    """Replace every parameter with a GGUF file's weights; quantized ones stay
    ``ExportableGGUFTensor`` so the PTN carries the GGUF blocks. ``architecture``
    defaults to the file's ``general.architecture``."""
    key_map, convert = gguf_to_hf(model, architecture or gguf_architecture(path))
    shapes = {n: p.shape for n, p in model.named_parameters(remove_duplicate=False)}
    expected = set(shapes)
    loaded = set()
    for fqn, value in iter_checkpoint(
        path,
        key_map=key_map,
        convert=convert,
        tie_map={"model.embed_tokens.weight": ("lm_head.weight", False)},
        dtype=torch.float32,
        raw_iter=raw_iter,
    ):
        if fqn not in expected:
            raise ValueError(f"GGUF tensor maps to unknown parameter {fqn!r}")
        want = tuple(shapes[fqn])
        if fqn in ("model.embed_tokens.weight", "lm_head.weight"):
            # llama.cpp can trim the vocabulary to the tokenizer's size.
            want = (value.shape[0], *want[1:])
        if tuple(value.shape) != want:
            raise ValueError(f"{fqn}: GGUF shape {tuple(value.shape)} != model {want}")
        assign_one(model, fqn, value)
        loaded.add(fqn)
    if loaded != expected:
        raise ValueError(f"GGUF is missing {sorted(expected - loaded)}")


# TEMPORARY: an FP16 KV cache for an FP32 model. Remove this knob once PTN moves
# to ExecuTorch's off-graph KV cache (extension/llm/cache), where the KV dtype is
# runtime cache configuration rather than part of the exported graph.
_CACHE_DTYPE_SDPA = "sdpa_in_cache_dtype"


class _CastingStaticLayer(StaticLayer):
    """A static cache layer that casts K/V writes to its own dtype."""

    def update(self, key_states, value_states, cache_kwargs=None):
        return super().update(
            key_states.to(self.dtype), value_states.to(self.dtype), cache_kwargs
        )


def _sdpa_in_cache_dtype(module, query, key, value, attention_mask, **kwargs):
    output, weights = sdpa_attention_forward(
        module, query.to(key.dtype), key, value, attention_mask, **kwargs
    )
    return output.to(query.dtype), weights


class LastTokenLM(nn.Module):
    """Use the Transformers ExecuTorch cache wrapper and return last-position logits."""

    def __init__(
        self,
        model: PreTrainedModel,
        max_cache_len: int,
        kv_cache_dtype: torch.dtype = torch.float32,
    ) -> None:
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
        if kv_cache_dtype != model.dtype:
            self._store_kv_as(model, kv_cache_dtype)

    def _store_kv_as(self, model: PreTrainedModel, dtype: torch.dtype) -> None:
        for i, old in enumerate(self.cache.layers):
            if type(old) is not StaticLayer:
                raise ValueError(f"cannot change the KV dtype of {type(old).__name__}")
            layer = _CastingStaticLayer(old.max_cache_len)
            layer.lazy_initialization(old.keys[:, :, :1].to(dtype))
            self.cache.layers[i] = layer
            self.cached.register_buffer(f"key_cache_{i}", layer.keys, persistent=False)
            self.cached.register_buffer(
                f"value_cache_{i}", layer.values, persistent=False
            )
        # Keep the vmap-free mask TorchExportableModuleForDecoderOnlyLM selected.
        mask = ALL_MASK_ATTENTION_FUNCTIONS[model.config._attn_implementation]
        AttentionInterface.register(_CACHE_DTYPE_SDPA, _sdpa_in_cache_dtype)
        AttentionMaskInterface.register(_CACHE_DTYPE_SDPA, mask)
        model.config._attn_implementation = _CACHE_DTYPE_SDPA

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


# TEMPORARY: use torchao-based 8da4w quantization until PTN export has its own
# quantization flow.
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


_PROMPTS = (
    "The capital of France is",
    "Once upon a time, in a small village by the sea, there lived a",
    "def fibonacci(n):",
    "The three primary colors are red,",
    "Water boils at a temperature of",
)
_DECODE_STEPS = 4
# Advisory (atol, rtol) for engine logits against eager ones. Engines quantize
# activations their own way, not torchao's, so quantized weights get a looser bound.
_FLOAT_TOL = (1e-3, 1e-3)
_QUANTIZED_TOL = (2.0, 5e-2)


def _sample_prompts(
    model: str, count: int, max_seq_length: int, cache_len: int
) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [tokenizer.encode(p) for p in _PROMPTS[:count]]
    longest = max(map(len, prompts))
    if longest > max_seq_length - 1 or longest + _DECODE_STEPS > cache_len:
        raise ValueError("a sample prompt and its decode steps do not fit")
    return prompts


def golden_samples(
    wrapper: LastTokenLM, prompts: list[list[int]], decode_steps: int
) -> SampleSet:
    """Per tokenized prompt, from an empty cache: a prefill call, then
    `decode_steps` calls each fed the previous call's eager argmax."""
    quantized = any(isinstance(p, TorchAOBaseTensor) for p in wrapper.parameters())
    samples = []
    with torch.no_grad():
        for ids in prompts:
            wrapper.cache.reset()
            input_ids = torch.tensor([ids], dtype=torch.long)
            positions = torch.arange(len(ids), dtype=torch.long)
            calls = []
            for _ in range(decode_steps + 1):
                logits = wrapper(input_ids, positions)
                calls.append(Call("forward", [input_ids, positions], [logits]))
                input_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
                positions = positions[-1:] + 1
            samples.append(Sample(calls))
    wrapper.cache.reset()
    tol = _QUANTIZED_TOL if quantized else _FLOAT_TOL
    return SampleSet(samples, source="eager/cpu", tol=tol)


def _histogram(gm: torch.fx.GraphModule) -> collections.Counter:
    return collections.Counter(
        str(n.target)
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target not in (executorch_call_delegate, operator.getitem)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Hugging Face model directory or hub ID."
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
    parser.add_argument(
        "--gguf",
        default="",
        help="GGUF file with the model's weights; --model supplies the config.",
    )
    parser.add_argument("--kv_cache_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help=f"Append eager logits for the first N of {len(_PROMPTS)} built-in "
        f"prompts, each a prefill and {_DECODE_STEPS} decode calls, as forward's "
        "'golden' sample set; needs --output and a tokenizer in --model.",
    )
    args = parser.parse_args()
    if not 0 <= args.samples <= len(_PROMPTS) or (args.samples and not args.output):
        parser.error(f"--samples takes 0 to {len(_PROMPTS)} and needs --output")
    if args.gguf and (args.layers or args.random_weights or args.quantize != "none"):
        parser.error(
            "--gguf loads every weight; it excludes --layers, --random_weights "
            "and --quantize"
        )
    if args.max_seq_length < 7:
        parser.error(
            "--max_seq_length must be at least 7: the example prompt has 6 tokens"
        )
    cache_len = args.max_context_length or args.max_seq_length
    if cache_len < args.max_seq_length - 1:
        parser.error("the context must hold a full prompt of max_seq_length - 1 tokens")
    return args


def main() -> None:
    args = _parse_args()
    cache_len = args.max_context_length or args.max_seq_length
    prompts = (
        _sample_prompts(args.model, args.samples, args.max_seq_length, cache_len)
        if args.samples
        else []
    )
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    model = _load(
        args.model, args.layers, args.random_weights, empty_weights=bool(args.gguf)
    )
    if args.gguf:
        load_gguf(model, args.gguf)
    if args.quantize == "8da4w":
        _quantize_8da4w(model, args.group_size, args.embedding_group_size)
    kv_cache_dtype = torch.float16 if args.kv_cache_dtype == "fp16" else torch.float32
    wrapper = LastTokenLM(model, cache_len, kv_cache_dtype).eval()

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
    if prompts:
        samples = golden_samples(wrapper, prompts, _DECODE_STEPS)
        write_samples(args.output, "golden", samples)
        print(f"wrote {len(prompts)} golden samples")
