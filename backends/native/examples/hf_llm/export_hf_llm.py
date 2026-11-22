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
packed in the PTN. ``--offgraph_cache`` exports attention as
``kvcache::update_and_attend`` with no KV cache in the graph; the runner owns the
cache. ``--samples`` appends a ``golden`` sample set: eager logits
for a prefill and a few greedy decode steps per prompt.
"""

import argparse
import contextlib
import logging
from collections.abc import Iterator

import torch
import torch.nn as nn
from executorch.backends.native.examples.hf_llm.gguf_hf import (
    check_config_tensors,
    gguf_architecture,
    gguf_to_hf,
)
from executorch.backends.native.examples.hf_llm.hf_utils import (
    PositionedStaticLayer,
    report_lowering,
)
from executorch.backends.native.examples.hf_llm.muse_glimmer import MuseGlimmerTextLM
from executorch.exir.native import Call, Sample, SampleSet, to_native, write_samples
from executorch.extension.llm.cache.hf_attention import register_offgraph_attention
from executorch.extension.llm.cache.reference_cache import (
    CacheConfig,
    LayerPolicy,
    SequenceReferenceCache,
)
from executorch.extension.llm.cache.update_and_attend import REGISTRY
from executorch.extension.llm.export.gguf import iter_gguf
from executorch.extension.llm.export.load import assign_one, iter_checkpoint, RawIter
from executorch.extension.llm.export.model_metadata import (
    write_cache_geometry,
    write_logits_to_keep_mode,
    write_max_context_len,
)
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
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.cache_utils import StaticCache, StaticLayer, StaticSlidingWindowLayer
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import (
    and_masks,
    AttentionMaskInterface,
    causal_mask_function,
    chunked_overlay,
    sliding_window_overlay,
)


_OFFGRAPH_ATTENTION = "offgraph"


@contextlib.contextmanager
def _parameters_on_meta() -> Iterator[None]:
    """Create parameters on the meta device and buffers as usual, so computed
    buffers such as RoPE frequencies keep their values. Weights are assigned
    later. What accelerate's ``init_empty_weights`` did, which transformers no
    longer re-exports."""
    register = nn.Module.register_parameter

    def register_on_meta(module, name, param):
        register(module, name, param)
        if param is not None:
            module._parameters[name] = nn.Parameter(
                param.to("meta"), requires_grad=param.requires_grad
            )

    nn.Module.register_parameter = register_on_meta
    try:
        yield
    finally:
        nn.Module.register_parameter = register


# Models Transformers builds only with their vision tower: the class that
# builds one, and the text-only module to export instead.
_TEXT_ONLY = {"muse_glimmer": (AutoModelForImageTextToText, MuseGlimmerTextLM)}
# Per-layer config lists, kept for the layers --layers keeps.
_PER_LAYER = ("layer_types", "layer_rope_theta")


def _load(
    path: str,
    layers: int,
    random_weights: bool,
    empty_weights: bool,
    offgraph_cache: bool = False,
) -> nn.Module:
    config = AutoConfig.from_pretrained(path)
    text_config = config.get_text_config()
    if layers:
        text_config.num_hidden_layers = layers
        for name in _PER_LAYER:
            if getattr(text_config, name, None):
                setattr(text_config, name, getattr(text_config, name)[:layers])
    auto_class, text_only = _TEXT_ONLY.get(
        config.model_type, (AutoModelForCausalLM, None)
    )
    if text_only and not offgraph_cache:
        raise ValueError(f"{config.model_type} exports only with --offgraph_cache")
    if offgraph_cache:
        register_offgraph_attention(_OFFGRAPH_ATTENTION)
    attn = _OFFGRAPH_ATTENTION if offgraph_cache else "sdpa"
    kwargs = {"dtype": torch.float32, "attn_implementation": attn}
    if empty_weights:
        with _parameters_on_meta():
            model = auto_class.from_config(config, **kwargs)
    elif random_weights:
        model = auto_class.from_config(config, **kwargs)
    else:
        model = auto_class.from_pretrained(path, config=config, **kwargs)
    if text_only:
        model = text_only(model)
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    return model.eval()


def load_gguf(
    model: nn.Module,
    path: str,
    architecture: str | None = None,
    raw_iter: RawIter | None = None,
    truncated: bool = False,
) -> None:
    """Replace every parameter with a GGUF file's weights; quantized ones stay
    ``ExportableGGUFTensor`` so the PTN carries the GGUF blocks. ``architecture``
    defaults to the file's ``general.architecture``. ``truncated`` loads a model
    built with fewer layers from the file's first blocks."""
    architecture = architecture or gguf_architecture(path)
    key_map, convert = gguf_to_hf(model, architecture, truncated)
    read = raw_iter or iter_gguf
    shapes = {n: p.shape for n, p in model.named_parameters(remove_duplicate=False)}
    expected = set(shapes)
    loaded = set()
    for fqn, value in iter_checkpoint(
        path,
        key_map=key_map,
        convert=convert,
        tie_map={"model.embed_tokens.weight": ("lm_head.weight", False)},
        dtype=torch.float32,
        raw_iter=lambda p: check_config_tensors(model, architecture, read(p)),
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


_EXPORT_ATTENTION = "native_export"
# Attention arguments `_attention` implements besides query, key, value, the
# mask and its own `softcap`: both paths apply the scale, and the mask, always
# materialized, carries causality and the sliding window, so `is_causal` changes
# nothing. `dropout` passes only when zero.
_IMPLEMENTED_ATTENTION_ARGS = frozenset({"scaling", "is_causal", "sliding_window"})
# Model-level flags transformers forwards to every attention call; they select
# outputs, not the attention itself. Packed-sequence arguments (`cu_seq_lens_*`,
# `max_length_*`, `seq_idx`) are not here: export runs one sequence and leaves
# them None.
_INERT_ATTENTION_ARGS = frozenset(
    {
        "position_ids",
        "use_cache",
        "output_attentions",
        "output_hidden_states",
        "output_router_logits",
        "num_items_in_batch",
        "deterministic",
        # transformers 5 forwards it to every attention; the mask encodes it.
        "cache_position",
    }
)


def _check_attention_args(module: nn.Module, kwargs: dict) -> None:
    """Raise for an attention argument export would drop: SDPA ignores what it
    does not know, e.g. attention sinks, and the PTN would then differ from
    eager."""
    for name, value in kwargs.items():
        if value is None or name in _INERT_ATTENTION_ARGS:
            continue
        if name in _IMPLEMENTED_ATTENTION_ARGS:
            continue
        if name == "dropout" and not value:
            continue
        raise NotImplementedError(
            f"{type(module).__name__} (layer {getattr(module, 'layer_idx', '?')}) "
            f"passes {name}={_describe(value)} to attention, which export does "
            "not implement"
        )


def _describe(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return f"<tensor {tuple(value.shape)}>"
    return repr(value)


def _attention(module, query, key, value, attention_mask, softcap=None, **kwargs):
    _check_attention_args(module, kwargs)
    if softcap is not None:
        raise ValueError(
            "export does not support attention logit softcapping: SDPA cannot "
            "cap the attention logits"
        )
    if key.dtype == query.dtype:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )
    # TEMPORARY: an FP16 KV cache for an FP32 model. Remove this once PTN moves
    # to ExecuTorch's off-graph KV cache (extension/llm/cache), where the KV
    # dtype is runtime cache configuration rather than part of the graph.
    output, weights = sdpa_attention_forward(
        module, query.to(key.dtype), key, value, attention_mask, **kwargs
    )
    return output.to(query.dtype), weights


# Mask arguments transformers passes that `_layer_mask` reads (`config`,
# `device`) or can ignore: it rebuilds the causal, sliding-window or chunked mask
# transformers builds from `mask_function`, as a boolean mask SDPA takes as is.
_KNOWN_MASK_ARGS = frozenset(
    {"config", "device", "dtype", "mask_function", "allow_is_causal_skip", "use_vmap"}
)


def _check_mask_args(kwargs: dict) -> None:
    """Raise for a mask `_layer_mask` would not reproduce: transformers passes
    ``use_vmap=True`` when a model composes its own overlay onto the mask, e.g.
    Gemma 3's bidirectional attention, and ``allow_is_bidirectional_skip`` only
    for a bidirectional mask; refuse any other argument it does not know."""
    if kwargs.get("use_vmap"):
        raise NotImplementedError(
            "export does not implement a model's own attention mask overlay"
        )
    for name, value in kwargs.items():
        if value is not None and name not in _KNOWN_MASK_ARGS:
            raise NotImplementedError(
                f"export does not implement the attention mask argument {name}"
            )


def _code(function: object) -> object:
    # Every function one def statement creates shares that def's code object, so
    # code identity tells which factory built a closure.
    return getattr(function, "__code__", None)


def _closure(function: object) -> dict[str, object]:
    """The values a nested function closes over, by name."""
    cells = function.__closure__ or ()
    return {
        name: cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, cells)
    }


def _local_overlay(mask_function: object) -> tuple[str, int | None]:
    """What ``mask_function`` adds to causality: ``("causal", None)``,
    ``("sliding", window)`` or ``("chunked", size)``, the masks transformers
    builds from its factories. Raise for any other function, e.g. one composed
    with packed sequences, bidirectional blocks or a bidirectional mask, which
    `_layer_mask` would drop. A chunked mask's left padding is zero: export
    takes no padding mask."""
    if mask_function is causal_mask_function:
        return "causal", None
    if _code(mask_function) is _code(and_masks(causal_mask_function)):
        parts = _closure(mask_function)["mask_functions"]
        if len(parts) == 2 and parts[1] is causal_mask_function:
            overlay = _code(parts[0])
            if overlay is _code(sliding_window_overlay(0)):
                return "sliding", _closure(parts[0])["sliding_window"]
            if overlay is _code(chunked_overlay(0, torch.zeros(1))):
                return "chunked", _closure(parts[0])["chunk_size"]
    raise NotImplementedError(
        f"export does not implement the attention mask function {mask_function!r}"
    )


def _layer_mask(
    batch_size: int,
    cache_position: torch.Tensor | None = None,
    kv_length: int = 0,
    kv_offset: int = 0,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    q_length: int | None = None,
    q_offset: int | torch.Tensor = 0,
    **kwargs,
) -> torch.Tensor:
    """One layer's mask, always materialized from comparisons. transformers 4's
    ``sdpa_mask_without_vmap`` applies the config's sliding window or chunks to
    every layer; transformers 5's ``sdpa_mask`` drops masks that SDPA's
    ``is_causal`` could stand in for and builds windows from a scalar ``full``.
    transformers 5 passes ``q_length`` and ``q_offset`` instead of
    ``cache_position``. Only sliding-window and chunked masks pass a
    ``local_size``."""
    _check_mask_args(kwargs)
    if attention_mask is not None:
        raise ValueError("export takes no padding mask")
    kind, size = _local_overlay(kwargs.get("mask_function", causal_mask_function))
    if size != local_size:
        raise NotImplementedError(
            f"the mask function's local size {size} is not local_size {local_size}"
        )
    if cache_position is None:
        cache_position = torch.arange(q_length, device=kwargs.get("device")) + q_offset
    kv_arange = torch.arange(kv_length, device=cache_position.device)
    kv_arange += kv_offset
    positions = cache_position.view(-1, 1)
    mask = kv_arange <= positions
    if kind == "sliding":
        mask *= (positions - size) < kv_arange
    elif kind == "chunked":
        mask *= (kv_arange // size) == (positions // size)
    return mask[None, None, :, :].expand(batch_size, -1, -1, -1)


class LastTokenLM(nn.Module):
    """Use the Transformers ExecuTorch cache wrapper and return last-position logits."""

    def __init__(
        self,
        model: PreTrainedModel,
        max_cache_len: int,
        kv_cache_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        # transformers' export wrapper turns sliding-window layers into plain
        # StaticLayers and the rewrite below keeps only keys and values, so
        # refuse static layers that keep more, e.g. an indexer's keys or a
        # linear attention's, before either runs.
        layers = StaticCache(config=model.config, max_cache_len=max_cache_len).layers
        for layer in layers:
            if isinstance(layer, StaticLayer) and type(layer) not in (
                StaticLayer,
                StaticSlidingWindowLayer,
            ):
                raise NotImplementedError(
                    f"export does not support {type(layer).__name__} cache layers"
                )
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
        self._write_cache_at_input_positions(
            max_cache_len, kv_cache_dtype, kv_cache_dtype != model.dtype
        )
        self._use_export_attention(model)

    def _write_cache_at_input_positions(
        self, max_cache_len: int, dtype: torch.dtype, change_dtype: bool
    ) -> None:
        # Sliding-window and chunked layers keep every position too: their
        # attention mask, not a rolling buffer, limits what they attend.
        for i, old in enumerate(self.cache.layers):
            if not isinstance(old, StaticLayer):
                if change_dtype:
                    raise ValueError(
                        f"cannot change the KV dtype of {type(old).__name__}"
                    )
                continue
            layer = PositionedStaticLayer.replacing(old, dtype, max_cache_len)
            self.cache.layers[i] = layer
            self.cached.register_buffer(f"key_cache_{i}", layer.keys, persistent=False)
            self.cached.register_buffer(
                f"value_cache_{i}", layer.values, persistent=False
            )

    def _use_export_attention(self, model: PreTrainedModel) -> None:
        # Whatever attention the model was configured with: transformers keeps
        # registrations process-wide, so every wrapper registers the same
        # functions, and `_attention` takes the boolean mask `_layer_mask` builds.
        AttentionInterface.register(_EXPORT_ATTENTION, _attention)
        AttentionMaskInterface.register(_EXPORT_ATTENTION, _layer_mask)
        model.config._attn_implementation = _EXPORT_ATTENTION

    def forward(
        self, input_ids: torch.Tensor, cache_position: torch.Tensor
    ) -> torch.Tensor:
        # transformers 5 takes positions from the cache's length, not cache_position.
        for layer in self.cache.layers:
            layer.cumulative_length = cache_position[:1].clone()
            if isinstance(layer, PositionedStaticLayer):
                layer.positions = cache_position
        return self.cached.model(
            input_ids=input_ids,
            cache_position=cache_position,
            attention_mask=None,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits

    def call_inputs(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> list[torch.Tensor]:
        return [input_ids, positions]

    def reset_cache(self) -> None:
        self.cache.reset()

    def cache_scope(self) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()


def _cache_windows(config) -> list[int]:
    """Per layer, the sliding window HF attends over, or 0 for full attention.

    Without layer_types (e.g. Mistral), HF windows every layer when
    sliding_window is set.
    """
    window = getattr(config, "sliding_window", None)
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        return [window or 0] * config.num_hidden_layers
    unsupported = set(layer_types) - {"full_attention", "sliding_attention"}
    if unsupported:
        raise ValueError(
            f"--offgraph_cache does not support {sorted(unsupported)} layers "
            f"({config.model_type})"
        )
    if "sliding_attention" in layer_types and not window:
        raise ValueError(f"{config.model_type} has sliding layers but no window")
    return [window if t == "sliding_attention" else 0 for t in layer_types]


class OffGraphLastTokenLM(nn.Module):
    """Last-position logits with the KV cache outside the graph, or with
    ``selected`` the logits of the token indices a third input names, which a
    batched forward over several sequences needs.

    Each attention layer emits ``kvcache::update_and_attend`` over this step's
    K/V; the cache, not the graph, applies a sliding layer's window, so the
    runner learns the windows from ``metadata()``. Eager calls (golden
    samples) run inside ``cache_scope()``, which makes a reference cache of
    ``capacity`` positions the active one.
    """

    def __init__(self, model: nn.Module, capacity: int, selected: bool = False) -> None:
        super().__init__()
        self.model = model
        self.selected = selected
        config = model.config.get_text_config()
        self._windows = _cache_windows(config)
        self._head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self._kv_heads = config.num_key_value_heads
        # The longest context the model supports, else the reference cache's.
        self._max_context_len = (
            getattr(config, "max_position_embeddings", None) or capacity
        )
        self._reference = SequenceReferenceCache(
            CacheConfig(
                n_layers=config.num_hidden_layers,
                n_kv_heads=self._kv_heads,
                head_dim=self._head_dim,
                capacity=capacity,
                layers=[
                    LayerPolicy.ring(w) if w else LayerPolicy.flat()
                    for w in self._windows
                ],
            )
        )
        self._key = f"export_hf_llm/{id(self)}"

    def metadata(self) -> dict[str, int | list[int]]:
        """The cache geometry and the model's context length, under the names
        LLM runners read them by."""
        n = len(self._windows)
        geometry = write_cache_geometry(
            [self._kv_heads] * n, [self._head_dim] * n, self._windows
        )
        return (
            {
                name: value.tolist() if isinstance(value, torch.Tensor) else value
                for name, value in geometry.items()
            }
            | write_max_context_len(self._max_context_len)
            | write_logits_to_keep_mode("selected" if self.selected else "last")
        )

    def call_inputs(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> list[torch.Tensor]:
        """A call's inputs, selecting the last token's logits."""
        last = torch.tensor([input_ids.shape[1] - 1], dtype=torch.long)
        return [input_ids, positions, last] if self.selected else [input_ids, positions]

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        logits_to_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            cache_position=cache_position,
            # The model would derive positions from its (absent) cache's length.
            position_ids=cache_position.unsqueeze(0),
            use_cache=False,
            past_key_values=None,
            logits_to_keep=logits_to_keep if self.selected else 1,
        ).logits

    def reset_cache(self) -> None:
        self._reference.reset()

    @contextlib.contextmanager
    def cache_scope(self) -> Iterator[None]:
        REGISTRY.install(self._key, self._reference)
        try:
            with REGISTRY.active(self._key):
                yield
        finally:
            REGISTRY.uninstall(self._key)


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
    wrapper: LastTokenLM | OffGraphLastTokenLM,
    prompts: list[list[int]],
    decode_steps: int,
) -> SampleSet:
    """Per tokenized prompt, from an empty cache: a prefill call, then
    `decode_steps` calls each fed the previous call's eager argmax."""
    quantized = any(isinstance(p, TorchAOBaseTensor) for p in wrapper.parameters())
    samples = []
    with torch.no_grad(), wrapper.cache_scope():
        for ids in prompts:
            wrapper.reset_cache()
            input_ids = torch.tensor([ids], dtype=torch.long)
            positions = torch.arange(len(ids), dtype=torch.long)
            calls = []
            for _ in range(decode_steps + 1):
                inputs = wrapper.call_inputs(input_ids, positions)
                logits = wrapper(*inputs)
                calls.append(Call("forward", inputs, [logits]))
                input_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
                positions = positions[-1:] + 1
            samples.append(Sample(calls))
    wrapper.reset_cache()
    tol = _QUANTIZED_TOL if quantized else _FLOAT_TOL
    return SampleSet(samples, source="eager/cpu", tol=tol)


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
    parser.add_argument(
        "--layers",
        type=int,
        default=0,
        help="Keep only the first N layers; with --gguf, load their weights.",
    )
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
        "--offgraph_cache",
        action="store_true",
        help="Export attention as kvcache::update_and_attend with no KV cache in "
        "the graph; the runner owns the cache and sets its capacity and dtype.",
    )
    parser.add_argument(
        "--logits_to_keep",
        choices=["last", "selected"],
        default="last",
        help="With --offgraph_cache: 'selected' adds a third int64[K] input of "
        "token indices whose logits to return, for a batching runtime that "
        "puts several sequences in one forward.",
    )
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
    if args.gguf and (args.random_weights or args.quantize != "none"):
        parser.error(
            "--gguf loads every weight; it excludes --random_weights and --quantize"
        )
    if args.logits_to_keep == "selected" and not args.offgraph_cache:
        parser.error("--logits_to_keep selected needs --offgraph_cache")
    if args.offgraph_cache and (
        args.max_context_length or args.kv_cache_dtype != "fp32"
    ):
        parser.error(
            "--offgraph_cache leaves the KV cache to the runner; it excludes "
            "--max_context_length and --kv_cache_dtype"
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
        args.model,
        args.layers,
        args.random_weights,
        empty_weights=bool(args.gguf),
        offgraph_cache=args.offgraph_cache,
    )
    if args.gguf:
        load_gguf(model, args.gguf, truncated=bool(args.layers))
    if args.quantize == "8da4w":
        _quantize_8da4w(model, args.group_size, args.embedding_group_size)
    if args.offgraph_cache:
        wrapper = OffGraphLastTokenLM(
            model, cache_len, selected=args.logits_to_keep == "selected"
        ).eval()
    else:
        kv_cache_dtype = (
            torch.float16 if args.kv_cache_dtype == "fp16" else torch.float32
        )
        wrapper = LastTokenLM(model, cache_len, kv_cache_dtype).eval()

    input_ids = torch.zeros((1, 6), dtype=torch.long)
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.long)
    seq = Dim("seq", min=1, max=args.max_seq_length - 1)
    inputs = wrapper.call_inputs(input_ids, cache_position)
    dynamic_shapes = {"input_ids": {1: seq}, "cache_position": {0: seq}}
    if len(inputs) == 3:
        # Trace two rows: export specializes a size of 1.
        inputs[2] = torch.tensor([0, input_ids.shape[1] - 1])
        dynamic_shapes["logits_to_keep"] = {
            0: Dim("rows", min=1, max=args.max_seq_length - 1)
        }
    with torch.no_grad():
        ep = export(
            wrapper,
            tuple(inputs),
            dynamic_shapes=dynamic_shapes,
            strict=not args.non_strict,
        )
    report_lowering(ep, "forward")

    if args.output:
        metadata = wrapper.metadata() if args.offgraph_cache else None
        to_native(ep, metadata=metadata).save(args.output)
        print(f"saved {args.output}")
    if prompts:
        samples = golden_samples(wrapper, prompts, _DECODE_STEPS)
        write_samples(args.output, "golden", samples)
        print(f"wrote {len(prompts)} golden samples")


if __name__ == "__main__":
    main()
