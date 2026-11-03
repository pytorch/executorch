# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rewrite an exported decoder step into QNN's KV-cache-as-I/O contract.

Hugging Face exports a decoder step with its static cache as plain inputs: each
layer's full K and V cache goes in, the step writes the new token into it with
``index_copy_`` and attends over the whole buffer, and the updated buffers come
back out. HTP wants a different contract, the one ``qnn_llama_runner`` drives:

    (input_tokens, atten_mask, pos_ids, past_k_0..N-1, past_v_0..N-1)
        -> (logits, new_k_0..N-1, new_v_0..N-1)

where ``past_k_i`` is ``[B, H, D, S-1]`` (K stored transposed, so attention is a
plain matmul), ``past_v_i`` is ``[B, H, S-1, D]``, only the new token's K/V slice
is returned, and the mask is an additive float tensor managed by the runner.

``to_kv_cache_io`` does that rewrite on the traced graph, after a normal export:
no model wrapper and nothing model-specific. It matches the cache writes and the
attention that reads them, so it applies to any decoder Hugging Face exports this
way, and it leaves models without a cache alone.
"""

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from executorch.extension.llm.export.cache_graph import (
    fold_position_tables,
    fold_weight_arithmetic,
    norm_after_head_transpose,
    producer_cache_write,
    split_rotate_half,
)
from torch.export import ExportedProgram

aten = torch.ops.aten

# Label stored in the program so a runtime can pick the I/O manager that speaks
# this contract.
KV_CACHE_IO_CONTRACT = "kvcache_io"
KV_CACHE_IO_VERSION = 1
METHOD_NAME = "kv_forward"
PREFILL_METHOD_NAME = "prefill_forward"
# Masked-slot value seen in calibration. Quantized, the runner writes code 0
# for a masked slot and the mask's calibrated encoding decides what it means.
# At -65535 the softmax input's step is ~1.0: scores lose precision and HTP's
# softmax runs ~10x slower. -255 is what Qualcomm's native flow uses.
CALIBRATION_MASK_VALUE = -255.0

_CACHE_WRITES = (aten.index_copy_.default, aten.index_copy.default)
_SDPA = aten.scaled_dot_product_attention.default
# Steps SeqMSE searches over: about one calibration batch of Qualcomm's native
# flow, which searches on its first batch.
_SEQ_MSE_STEPS = 32
# The R3 rotation's Hadamard matrix, shared by every layer.
_R3_WEIGHT = "_qnn_r3"
# Ops that only reshape or narrow a cache read between the write and the
# attention: the GQA head repeat, and the slice to the filled length some
# attention paths read (with ``is_causal``). The rewritten attention reads the
# whole window under the runner's mask instead, so both are dropped.
_REPEAT_OPS = (
    aten.slice.Tensor,
    aten.unsqueeze.default,
    aten.expand.default,
    aten.reshape.default,
    aten.view.default,
    aten.clone.default,
    aten._unsafe_view.default,
)


@dataclass
class KvCacheIoProgram:
    """A decoder step rewritten into the QNN KV-cache-as-I/O contract."""

    module: torch.fx.GraphModule
    example_inputs: tuple
    num_layers: int
    max_seq_len: int
    k_shapes: List[torch.Size]
    v_shapes: List[torch.Size]
    vocab_size: int
    ar_len: int = 1
    logits_shape: Optional[torch.Size] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Encoder-decoder models: each cross-attention layer's keys (transposed,
    # ``[1, H, D, S]``) and values (``[1, H, S, D]``), fixed for a whole
    # generation, are inputs after the self-attention caches.
    cross_k_shapes: List[torch.Size] = field(default_factory=list)
    cross_v_shapes: List[torch.Size] = field(default_factory=list)

    @property
    def method_name(self) -> str:
        # The names qnn_llama_runner loads: a single-token decode step and the
        # multi-token prompt step of hybrid mode.
        return METHOD_NAME if self.ar_len == 1 else PREFILL_METHOD_NAME

    def export(self, strict: bool = False) -> ExportedProgram:
        return torch.export.export(self.module, self.example_inputs, strict=strict)

    def constant_methods(self) -> Dict[str, Any]:
        """Model metadata ``qnn_llama_runner`` reads, plus the contract label."""
        return {
            **{k: v for k, v in self.metadata.items() if v is not None},
            "get_max_seq_len": self.max_seq_len,
            "get_n_layers": self.num_layers,
            **(
                {"get_n_cross_layers": len(self.cross_k_shapes)}
                if self.cross_k_shapes
                else {}
            ),
            "get_vocab_size": self.vocab_size,
            "use_kv_cache": True,
            "get_io_contract": KV_CACHE_IO_CONTRACT,
            "get_io_contract_version": KV_CACHE_IO_VERSION,
        }

    def quant_io_dtype_fn(self, kv_dtype: torch.dtype, io_dtype: torch.dtype):
        """``TagQuantIO`` callback: keep the cache, mask and logits quantized at the
        graph boundary, which is the I/O the runner exchanges."""
        ar = self.ar_len
        cache_in = {tuple(s) for s in self.k_shapes} | {tuple(s) for s in self.v_shapes}
        cache_out = {(*s[:3], ar) for s in self.k_shapes} | {
            (s[0], s[1], ar, s[3]) for s in self.v_shapes
        }
        mask_shape = tuple(self.example_inputs[1].shape)
        logits_shape = tuple(self.logits_shape)

        def fn(node):
            from executorch.backends.qualcomm.builders.utils import is_graph_output

            val = node.meta.get("val")
            if not isinstance(val, torch.Tensor):
                return None
            shape = tuple(val.shape)
            if node.op == "placeholder":
                if shape in cache_in:
                    return kv_dtype
                if shape == mask_shape and val.is_floating_point():
                    return io_dtype
            if is_graph_output(node):
                if shape in cache_out:
                    return kv_dtype
                if shape == logits_shape:
                    return io_dtype
            return None

        return fn

    def calibration_fn(  # noqa: C901
        self,
        prompts: Sequence[Sequence[int]],
        eos_ids: Sequence[int] = (),
        seq_mse_candidates: int = 0,
        cross_kv: Optional[Sequence[Sequence[torch.Tensor]]] = None,
        seq_mse_steps: Optional[Dict[str, float]] = None,
    ) -> Callable[[torch.nn.Module], None]:
        """PT2E calibration that drives the step the way the runner does.

        Each prompt runs token by token; every step's new K/V slice is written
        into the past buffers fed to the next step, and generation continues
        greedily after the prompt until EOS or the context is full, so the
        observers see real cache contents instead of an empty cache.

        ``cross_kv`` gives each prompt's cross-attention inputs (keys, then
        values) for an encoder-decoder step.

        With ``seq_mse_candidates``, each convolution's weight scale is then
        searched (SeqMSE) for the least output error over every step of the
        first prompt, as Qualcomm's native flow does for its first batch.
        ``seq_mse_steps`` shares the result between graphs (see ``seq_mse``).
        """
        layers, ar, total = self.num_layers, self.ar_len, self.max_seq_len
        past = total - ar

        def step(module, chunk, start, past_k, past_v, cross=()):
            """Run ``chunk`` (at most ``ar`` tokens) at position ``start`` and
            write its new K/V into the past buffers, as the runner does."""
            n = len(chunk)
            mask = torch.full((1, 1, ar, total), CALIBRATION_MASK_VALUE)
            for row in range(ar):
                mask[..., row, :start] = 0.0
                mask[..., row, past : past + min(row, n - 1) + 1] = 0.0
            tokens = torch.tensor([chunk + [chunk[-1]] * (ar - n)], dtype=torch.int32)
            positions = torch.tensor(
                [[min(start + i, total - 1) for i in range(ar)]], dtype=torch.int32
            )
            outputs = module(tokens, mask, positions, *past_k, *past_v, *cross)
            for i in range(layers):
                past_k[i][..., start : start + n] = outputs[1 + i][..., :n]
                past_v[i][..., start : start + n, :] = outputs[1 + layers + i][
                    ..., :n, :
                ]
            return outputs[0]

        def run(module: torch.nn.Module, index: int) -> None:
            tokens = list(prompts[index])
            cross = tuple(cross_kv[index]) if cross_kv else ()
            past_k = [torch.zeros(s) for s in self.k_shapes]
            past_v = [torch.zeros(s) for s in self.v_shapes]
            with torch.no_grad():
                pos = 0
                while pos < past:
                    if ar > 1:
                        # The prompt step only ever sees prompt chunks.
                        if pos >= len(tokens):
                            break
                        chunk = tokens[pos : min(pos + ar, past)]
                        step(module, chunk, pos, past_k, past_v, cross)
                        pos += len(chunk)
                        continue
                    logits = step(module, [tokens[pos]], pos, past_k, past_v, cross)
                    pos += 1
                    if pos >= len(tokens):
                        nxt = int(torch.argmax(logits[0, -1]).item())
                        if nxt in eos_ids:
                            break
                        tokens.append(nxt)

        def calibrate(module: torch.nn.Module) -> None:
            for index in range(len(prompts)):
                run(module, index)
            if seq_mse_candidates and prompts:
                seq_mse(
                    module,
                    seq_mse_candidates,
                    lambda: run(module, 0),
                    seq_mse_steps,
                )

        return calibrate


def _hadamard(n: int, dtype: torch.dtype) -> torch.Tensor:
    """The ``n`` x ``n`` Hadamard matrix scaled to be orthonormal."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"R3 needs a power-of-two head dim, got {n}.")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)])
    return (h / n**0.5).to(dtype)


def seq_mse(
    module: torch.fx.GraphModule,
    num_candidates: int,
    run: Callable[[], None],
    steps: Optional[Dict[str, float]] = None,
) -> None:
    """SeqMSE over every forward ``run`` makes: QNN's SeqMSE searches on each
    forward it sees and keeps the last result, so with one token per step the
    inputs are gathered first and searched once.

    ``steps`` maps each weight to its chosen scale step: filled when empty,
    applied without searching otherwise, so graphs sharing weights (a prompt
    and a decode step) quantize them identically."""
    from executorch.backends.qualcomm._passes.seq_mse import (
        InsertSeqMse,
        RemoveSeqMse,
        SeqMseModule,
    )

    # The rewritten program holds its weights as constants, not parameters.
    module.get_parameter = lambda target: functools.reduce(
        getattr, target.split("."), module
    )
    try:
        InsertSeqMse(num_candidates)(module)
    finally:
        del module.get_parameter
    # Each search, keyed by its weight (conv -> weight observer -> weight).
    searches = {
        node.args[1].args[1].args[0].target: getattr(module, node.target)
        for node in module.graph.nodes
        if node.op == "call_module"
        and isinstance(getattr(module, node.target), SeqMseModule)
    }
    # The Hadamard matrix's entries are all +-1/sqrt(d): any smaller scale
    # would clip it.
    searches.pop(_R3_WEIGHT, None)
    if steps:
        for weight, m in searches.items():
            m.best_candidate_step = steps.get(weight, 1.0)
    else:
        seen = {weight: ([], []) for weight in searches}
        for weight, m in searches.items():
            m.forward = lambda x, y, io=seen[weight]: (
                len(io[0]) < _SEQ_MSE_STEPS
                and (io[0].append(x.detach()), io[1].append(y.detach()))
            )
        run()
        for weight, m in searches.items():
            inputs, outputs = seen[weight]
            if inputs:
                m.best_candidate_step = m._find_best_candidate(
                    torch.cat(inputs), torch.cat(outputs)
                )
            if steps is not None:
                steps[weight] = m.best_candidate_step
    RemoveSeqMse()(module)


def has_kv_cache(exported_program: ExportedProgram) -> bool:
    """Whether the program writes a cache it takes as an input."""
    return any(
        n.op == "call_function" and n.target in _CACHE_WRITES
        for n in exported_program.graph.nodes
    )


def _source_input(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The program input a cache read comes from, if nothing writes it."""
    while node.op == "call_function" and node.target in _REPEAT_OPS:
        node = node.args[0]
    return node if node.op == "placeholder" else None


def _write_counters(layers) -> List[torch.fx.Node]:
    """The scalar inputs (cache fill counters) the cache writes' indices derive
    from."""
    counters, stack, seen = [], [], set()
    for _, k_write, v_write in layers:
        stack += [k_write.args[2], v_write.args[2]]
    while stack:
        node = stack.pop()
        if not isinstance(node, torch.fx.Node) or node in seen:
            continue
        seen.add(node)
        val = node.meta.get("val")
        if node.op == "placeholder":
            if isinstance(val, torch.Tensor) and val.dim() == 0:
                counters.append(node)
            continue
        stack.extend(node.all_input_nodes)
    return counters


def to_kv_cache_io(  # noqa: C901
    exported_program: ExportedProgram,
    metadata: Optional[Dict[str, Any]] = None,
    masked_softmax: bool = False,
    r3: bool = False,
) -> KvCacheIoProgram:
    """Rewrite a Hugging Face decoder step into QNN's KV-cache-as-I/O contract.

    Args:
        exported_program: A single-token decoder step exported with its static
            cache as inputs (``past_key_values_*``), as Hugging Face exporters
            produce it.
        metadata: Constant-method metadata to carry, e.g. ``get_bos_id`` and
            ``get_eos_ids``.
        masked_softmax: Mask with ``where(mask == 0, scores, min(scores) - 20)``
            instead of adding the mask, which HTP fuses into its MaskedSoftmax
            op. For quantized programs: unlike an additive mask, it holds
            whatever the scores' range.
        r3: Rotate queries and keys by a normalized Hadamard matrix after RoPE
            (SpinQuant's R3, as Qualcomm's native flow applies it). Scores are
            unchanged; outliers in the keys, and so in the cache, are spread
            across the head dimension, which quantizes better.

    Returns:
        A ``KvCacheIoProgram``: the rewritten module, its example inputs, and the
        constant methods, quantized-I/O tagging and calibration loop that go with
        the contract.
    """
    gm = exported_program.module()
    graph = gm.graph
    placeholders = {n.name: n for n in graph.nodes if n.op == "placeholder"}

    # The attention calls, in layer order, and the cache writes they read.
    # Attention reading caches that are only inputs (an encoder-decoder's
    # cross-attention) takes those as fixed inputs.
    layers, cross = [], []
    for node in graph.nodes:
        if node.op == "call_function" and node.target is _SDPA:
            k_write = producer_cache_write(node.args[1])
            v_write = producer_cache_write(node.args[2])
            if k_write is not None and v_write is not None:
                layers.append((node, k_write, v_write))
                continue
            k_in, v_in = _source_input(node.args[1]), _source_input(node.args[2])
            mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask")
            if k_in is None or v_in is None or mask is not None:
                raise ValueError(
                    f"{node.name} does not read a cache this rewrite understands."
                )
            cross.append((node, k_in, v_in))
    if not layers:
        raise ValueError("No attention reading a written cache was found.")

    query = layers[0][0].args[0].meta["val"]
    if query.shape[0] != 1:
        raise ValueError(f"Expected batch 1, got query {tuple(query.shape)}.")
    ar = query.shape[2]
    cache = layers[0][1].args[0].meta["val"]
    max_seq_len = cache.shape[2]
    dtype = cache.dtype

    tokens = placeholders.get("input_ids", placeholders.get("decoder_input_ids"))
    position = placeholders.get("position_ids")
    # Without position inputs, positions come from the caches' fill counters.
    counters = [] if position is not None else _write_counters(layers)
    if tokens is None or (position is None and not counters):
        raise ValueError(
            "Expected token ids and positions (position_ids or cache counters)."
        )

    first = next(iter(graph.nodes))
    with graph.inserting_before(first):
        new_tokens = graph.placeholder("input_tokens")
        new_mask = graph.placeholder("atten_mask")
        new_position = graph.placeholder("pos_ids")
        past_k, past_v = [], []
        for i in range(len(layers)):
            past_k.append(graph.placeholder(f"past_k_{i}"))
        for i in range(len(layers)):
            past_v.append(graph.placeholder(f"past_v_{i}"))
        cross_k = [graph.placeholder(f"cross_k_{i}") for i in range(len(cross))]
        cross_v = [graph.placeholder(f"cross_v_{i}") for i in range(len(cross))]

    # Tokens and positions come in as int32, as the runner feeds them.
    tokens.replace_all_uses_with(new_tokens)
    if counters:
        # Each counter holds the step's first position.
        new_position.meta["val"] = torch.zeros(1, ar, dtype=torch.int32)
        body = next(n for n in graph.nodes if n.op != "placeholder")
        with graph.inserting_before(body):
            row = graph.call_function(aten.select.int, (new_position, 0, 0))
            start = graph.call_function(aten.select.int, (row, 0, 0))
            for counter in counters:
                value = graph.call_function(
                    aten._to_copy.default,
                    (start,),
                    {"dtype": counter.meta["val"].dtype},
                )
                counter.replace_all_uses_with(value)
        position = new_position
    fold_position_tables(gm, position, new_position, max_seq_len, dtype)
    split_rotate_half(graph)
    fold_weight_arithmetic(gm)
    norm_after_head_transpose(graph)
    if position is not new_position and position.users:
        with graph.inserting_after(new_position):
            widened = graph.call_function(
                aten._to_copy.default,
                (new_position,),
                {"dtype": position.meta["val"].dtype},
            )
        position.replace_all_uses_with(widened)

    hadamard = None
    if r3:
        gm.register_buffer(
            _R3_WEIGHT, _hadamard(cache.shape[3], dtype), persistent=False
        )
        with graph.inserting_before(first):
            hadamard = graph.get_attr(_R3_WEIGHT)

    k_shapes, v_shapes, new_ks, new_vs = [], [], [], []
    for i, (sdpa, k_write, v_write) in enumerate(layers):
        q = sdpa.args[0]
        k_new, v_new = k_write.args[3], v_write.args[3]
        b, h_kv, s, d = k_write.args[0].meta["val"].shape
        h_q = q.meta["val"].shape[1]
        scale = sdpa.kwargs.get("scale", None) or d ** -0.5
        k_shapes.append(torch.Size([b, h_kv, d, s - ar]))
        v_shapes.append(torch.Size([b, h_kv, s - ar, d]))
        with graph.inserting_before(sdpa):
            if hadamard is not None:
                # As a linear layer, HTP runs it from the shared weights like the
                # projections; a matmul with a constant operand runs much slower.
                # H is symmetric, so x @ H == linear(x, H).
                q = graph.call_function(aten.linear.default, (q, hadamard))
                k_new = graph.call_function(aten.linear.default, (k_new, hadamard))
            k_t = graph.call_function(aten.transpose.int, (k_new, 2, 3))
            k_all = graph.call_function(aten.cat.default, ([past_k[i], k_t], 3))
            v_all = graph.call_function(aten.cat.default, ([past_v[i], v_new], 2))
            if h_q != h_kv:
                rep = h_q // h_kv
                k_all = graph.call_function(aten.unsqueeze.default, (k_all, 2))
                k_all = graph.call_function(
                    aten.expand.default, (k_all, [b, h_kv, rep, d, s])
                )
                k_all = graph.call_function(
                    aten.reshape.default, (k_all, [b, h_q, d, s])
                )
                v_all = graph.call_function(aten.unsqueeze.default, (v_all, 2))
                v_all = graph.call_function(
                    aten.expand.default, (v_all, [b, h_kv, rep, s, d])
                )
                v_all = graph.call_function(
                    aten.reshape.default, (v_all, [b, h_q, s, d])
                )
            scores = graph.call_function(aten.matmul.default, (q, k_all))
            scores = graph.call_function(aten.mul.Tensor, (scores, scale))
            if masked_softmax:
                low = graph.call_function(aten.amin.default, (scores, [-1], True))
                low = graph.call_function(aten.add.Tensor, (low, -20.0))
                visible = graph.call_function(aten.eq.Scalar, (new_mask, 0))
                scores = graph.call_function(aten.where.self, (visible, scores, low))
            else:
                scores = graph.call_function(aten.add.Tensor, (scores, new_mask))
            probs = graph.call_function(aten._softmax.default, (scores, -1, False))
            attn = graph.call_function(aten.matmul.default, (probs, v_all))
        sdpa.replace_all_uses_with(attn)
        new_ks.append(k_t)
        new_vs.append(v_new)

    cross_k_shapes, cross_v_shapes = [], []
    for i, (sdpa, k_in, _) in enumerate(cross):
        q = sdpa.args[0]
        b, h_kv, s, d = k_in.meta["val"].shape
        h_q = q.meta["val"].shape[1]
        scale = sdpa.kwargs.get("scale", None) or d ** -0.5
        cross_k_shapes.append(torch.Size([b, h_kv, d, s]))
        cross_v_shapes.append(torch.Size([b, h_kv, s, d]))
        with graph.inserting_before(sdpa):
            k_all, v_all = cross_k[i], cross_v[i]
            if h_q != h_kv:
                rep = h_q // h_kv
                k_all = graph.call_function(
                    aten.repeat_interleave.self_int, (k_all, rep, 1)
                )
                v_all = graph.call_function(
                    aten.repeat_interleave.self_int, (v_all, rep, 1)
                )
            scores = graph.call_function(aten.matmul.default, (q, k_all))
            scores = graph.call_function(aten.mul.Tensor, (scores, scale))
            probs = graph.call_function(aten._softmax.default, (scores, -1, False))
            attn = graph.call_function(aten.matmul.default, (probs, v_all))
        sdpa.replace_all_uses_with(attn)

    # New outputs: logits, then each layer's new K slice, then each new V slice.
    output = next(n for n in reversed(graph.nodes) if n.op == "output")
    logits = (
        output.args[0][0]
        if isinstance(output.args[0], (list, tuple))
        else output.args[0]
    )
    logits_shape = list(logits.meta["val"].shape)
    if ar > 1 and logits_shape[1] != ar:
        # Hugging Face keeps only the last position's logits; the runner reads the
        # row of the last real token in a padded prompt chunk, so keep them all.
        node = logits
        while node.op == "call_function" and node.args:
            if node.target is aten.slice.Tensor and node.args[1] == 1:
                node.replace_all_uses_with(node.args[0])
                logits_shape[1] = ar
                break
            node = node.args[0]
    output.args = ((logits, *new_ks, *new_vs),)

    # Keep exactly what the new outputs are computed from. Everything else served
    # the old contract only: the in-place cache writes, the cache-length counters
    # and their updates, the old mask and the shape asserts on them. Some of those
    # are side-effecting, so dead-code elimination alone would keep them.
    live = set()
    stack = [output]
    while stack:
        node = stack.pop()
        if node in live:
            continue
        live.add(node)
        stack.extend(node.all_input_nodes)
    keep = {new_tokens, new_mask, new_position, *past_k, *past_v, *cross_k, *cross_v}
    for node in reversed(list(graph.nodes)):
        if node not in live and node not in keep:
            graph.erase_node(node)
    graph._codegen = torch.fx.graph.CodeGen()
    graph.lint()
    # A fresh module keeps only what the rewritten graph references, and none of
    # the unlifted module's input-spec hooks for the old signature.
    gm = torch.fx.GraphModule(gm, graph)

    vocab_size = logits.meta["val"].shape[-1]
    example_inputs = (
        torch.zeros(1, ar, dtype=torch.int32),
        torch.zeros(1, 1, ar, max_seq_len, dtype=dtype),
        torch.zeros(1, ar, dtype=torch.int32),
        *[torch.zeros(s, dtype=dtype) for s in k_shapes],
        *[torch.zeros(s, dtype=dtype) for s in v_shapes],
        *[torch.zeros(s, dtype=dtype) for s in cross_k_shapes],
        *[torch.zeros(s, dtype=dtype) for s in cross_v_shapes],
    )
    return KvCacheIoProgram(
        module=gm,
        example_inputs=example_inputs,
        num_layers=len(layers),
        max_seq_len=max_seq_len,
        k_shapes=k_shapes,
        v_shapes=v_shapes,
        vocab_size=vocab_size,
        ar_len=ar,
        logits_shape=torch.Size(logits_shape),
        metadata=dict(metadata or {}),
        cross_k_shapes=cross_k_shapes,
        cross_v_shapes=cross_v_shapes,
    )
