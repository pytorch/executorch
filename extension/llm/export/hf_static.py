# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""Backend-neutral rewrites of Hugging Face's per-step static-cache signature.

The program has fixed ``prefill`` and ``decode`` methods. Runners supply masks,
absolute positions and cache counters; padding rows in window masks are false.
The program attends over incoming caches and current-token rows, then updates the caches.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.utils._pytree as pytree
from executorch.exir.dialects._ops import ops
from executorch.exir.operator.convert import unwrap_op_overload
from executorch.exir.program._program import _get_updated_graph_signature
from executorch.extension.llm.export import cache_graph
from torch.export import ExportedProgram

aten = torch.ops.aten
IO_CONTRACT = "hf_static"
# 1: float caches, fixed prefill/decode methods, absolute counters and padding masks.
# 2: adds keys stored [1, H, D, L] (get_key_layout) and integer caches whose
#    contents the runner treats as opaque.
IO_CONTRACT_VERSION = 2
PREFILL_METHOD = "prefill"
DECODE_METHOD = "decode"
CACHE_PREFIX = "past_key_values_"
CPU_META = "hf_static_cpu"
_MASK_FILL = -10000.0


@dataclass
class HfStaticLayout:
    """What a runner and the calibration loop need to know about the inputs."""

    input_names: List[str]
    query_len: int
    max_seq_len: int
    n_layers: int
    cache_shapes: List[torch.Size]
    # The cache inputs' shapes in order: per layer keys, values and counter (rank 0).
    state_shapes: List[torch.Size]
    dtype: torch.dtype
    # Each attention mask's width: the length of the caches it masks.
    mask_widths: Dict[str, int]
    state_dtypes: List[torch.dtype] = field(default_factory=list)
    token_dtype: torch.dtype = torch.int64
    position_dtype: torch.dtype = torch.int64
    window_masks: List[str] = field(default_factory=list)
    # "hf": keys [1, H, L, D] as HF stores them; "transposed": [1, H, D, L].
    key_layout: str = "hf"

    @classmethod
    def of(cls, exported_program: ExportedProgram) -> "HfStaticLayout":
        names, vals = [], []
        for spec in exported_program.graph_signature.input_specs:
            if spec.kind.name != "USER_INPUT":
                continue
            node = exported_program.graph.find_nodes(
                op="placeholder", target=spec.arg.name
            )[0]
            names.append(spec.arg.name)
            vals.append(node.meta["val"])
        state = [v for n, v in zip(names, vals) if n.startswith(CACHE_PREFIX)]
        caches = [v for v in state if v.dim() == 4]
        if len(state) != 3 * (len(caches) // 2) or any(
            v.dim() not in (0, 4) for v in state
        ):
            raise ValueError(
                "Expected per layer keys, values and a counter as the cache inputs."
            )
        meta = exported_program.graph_module.meta
        context = meta.get(
            "hf_static_max_seq_len", max(c.shape[2] for c in caches[1::2])
        )
        return cls(
            input_names=names,
            query_len=vals[names.index("input_ids")].shape[1],
            max_seq_len=context,
            n_layers=len(caches) // 2,
            cache_shapes=[c.shape for c in caches],
            state_shapes=[v.shape for v in state],
            dtype=caches[0].dtype,
            mask_widths={
                n: v.shape[-1]
                for n, v in zip(names, vals)
                if n.startswith("attention_mask")
            },
            state_dtypes=[v.dtype for v in state],
            token_dtype=vals[names.index("input_ids")].dtype,
            position_dtype=vals[names.index("position_ids")].dtype,
            window_masks=meta.get("hf_static_window_masks", []),
            key_layout=meta.get("hf_static_key_layout", "hf"),
        )


def sliding_windows(config) -> List[Optional[int]]:
    """Per-layer windows from the model config, never inferred from cache size."""
    layers = getattr(config, "per_layer_config", [config] * config.num_hidden_layers)
    types = getattr(config, "layer_types", None)
    if types is not None and len(types) != len(layers):
        raise ValueError("layer_types must describe every cache layer.")
    windows = []
    for i, layer in enumerate(layers):
        kind = types[i] if types is not None else None
        window = getattr(layer, "sliding_window", None)
        if kind == "full_attention":
            window = None
        elif kind not in (None, "sliding_attention"):
            raise ValueError(f"Unsupported cache layer type {kind!r}.")
        elif kind == "sliding_attention" and window is None:
            window = getattr(config, "sliding_window", None)
            if window is None:
                raise ValueError("A sliding layer requires sliding_window.")
        if window is not None and window <= 0:
            raise ValueError("sliding_window must be positive.")
        windows.append(window)
    return windows


def specialize(  # noqa: C901
    exported_program: ExportedProgram,
    query_len: int,
    max_seq_len: int,
    *,
    config,
    sliding: str = "window",
) -> ExportedProgram:
    """Re-export at fixed lengths. Sliding layers keep HF's window-sized caches
    (``window``) or take context-length caches with the window masked (``full``)."""
    if sliding not in ("window", "full"):
        raise ValueError(f"Unknown sliding mode {sliding!r}.")
    if not 0 < query_len <= max_seq_len:
        raise ValueError("query_len must be positive and no longer than max_seq_len.")
    windows = sliding_windows(config)
    if (
        getattr(config, "sliding_window", None) is not None
        and config.sliding_window < max_seq_len
        and not any(windows)
    ):
        raise ValueError(
            "sliding_window is shorter than context but no sliding layer was detected."
        )
    if sliding == "window" and any(
        query_len > min(w, max_seq_len) for w in windows if w is not None
    ):
        raise ValueError("query_len exceeds the smallest sliding cache length.")
    args, kwargs = exported_program.example_inputs
    flat, spec = pytree.tree_flatten((args, kwargs))
    user = [
        s.arg.name
        for s in exported_program.graph_signature.input_specs
        if s.kind.name == "USER_INPUT"
    ]
    if len(user) != len(flat):
        raise ValueError("The program's example inputs don't match its user inputs.")
    lengths = [
        max_seq_len if w is None or sliding == "full" else min(w, max_seq_len)
        for w in windows
    ]
    gm = exported_program.module()
    attention_layers = _cache_attention(gm.graph)
    if len(attention_layers) != len(windows):
        raise ValueError(
            f"Expected {len(windows)} cache attention layers, matched {len(attention_layers)}."
        )
    masks = {}
    window_masks = []
    for i, (sdpa, _, _) in enumerate(attention_layers):
        mask = sdpa.args[3] if len(sdpa.args) > 3 else sdpa.kwargs.get("attn_mask")
        if mask is None or mask.op != "placeholder":
            raise ValueError("Expected attention to use a mask input directly.")
        if mask.name in masks and masks[mask.name] != lengths[i]:
            raise ValueError(
                "Layers with different cache lengths require separate masks."
            )
        masks[mask.name] = lengths[i]
        if windows[i] is not None and sliding == "window":
            window_masks.append(mask.name)
    static = []
    state_index = 0
    for name, tensor in zip(user, flat):
        if name == "input_ids":
            static.append(torch.zeros(1, query_len, dtype=tensor.dtype))
        elif name == "position_ids":
            static.append(torch.arange(query_len, dtype=tensor.dtype)[None])
        elif name in masks:
            static.append(
                torch.ones(1, 1, query_len, masks[name], dtype=torch.bool).tril()
            )
        elif name.startswith(CACHE_PREFIX):
            if tensor.dim() == 4:
                static.append(
                    torch.zeros(
                        *tensor.shape[:2],
                        lengths[state_index // 3],
                        tensor.shape[3],
                        dtype=tensor.dtype,
                    )
                )
            else:
                static.append(torch.zeros((), dtype=tensor.dtype))
            state_index += 1
        else:
            raise ValueError(f"Unexpected hf_static input {name!r}.")
    if state_index != 3 * len(windows):
        raise ValueError("Expected three cache state inputs per layer.")
    args, kwargs = pytree.tree_unflatten(static, spec)
    program = torch.export.export(gm, args, kwargs, strict=False)
    program.graph_module.meta.update(
        hf_static_max_seq_len=max_seq_len,
        hf_static_windows=windows,
        hf_static_window_masks=list(dict.fromkeys(window_masks)),
        hf_static_sliding=sliding,
    )
    return program


def _cpu_call(graph):
    def call(target, args, kwargs=None):
        node = graph.call_function(target, args, kwargs)
        node.meta.setdefault("custom", {})[CPU_META] = True
        return node

    return call


def _masked(
    call, scores: torch.fx.Node, keep: torch.fx.Node, masked_softmax: bool
) -> torch.fx.Node:
    if masked_softmax:
        # HTP fuses this into MaskedSoftmax; it holds for any score range once
        # quantized, where an additive mask would set the scores' encoding.
        low = call(
            aten.add.Tensor, (call(aten.amin.default, (scores, [-1], True)), -20.0)
        )
        return call(aten.where.self, (keep, scores, low))
    bias = call(
        aten.where.self,
        (
            keep,
            call(aten.zeros_like.default, (scores,)),
            call(aten.full_like.default, (scores, _MASK_FILL)),
        ),
    )
    return call(aten.add.Tensor, (scores, bias))


def _constant(
    gm: torch.fx.GraphModule, name: str, value: torch.Tensor
) -> torch.fx.Node:
    if not hasattr(gm, name):
        gm.register_buffer(name, value)
    return gm.graph.get_attr(name)


def _step_mask(gm, mask, index, start, length: int, steps: int) -> torch.fx.Node:
    """Visibility over incoming cache slots followed by the step's token rows."""
    call = _cpu_call(gm.graph)
    slots = _constant(
        gm, f"_hf_static_slots_{length}", torch.arange(length).view(1, 1, 1, length)
    )
    end = call(aten.add.Tensor, (start, steps))
    unwritten = call(
        aten.logical_or.default,
        (call(aten.lt.Tensor, (slots, start)), call(aten.ge.Tensor, (slots, end))),
    )
    past = call(aten.logical_and.default, (mask, unwritten))
    new = call(aten.index_select.default, (mask, 3, index))
    return call(aten.cat.default, ([past, new], -1))


def _window_mask(gm, start, window: int, steps: int) -> torch.fx.Node:
    """Which of [a window-sized cache's slots, the step's token rows] each query
    row sees, as HF's sliding mask lets it: the cache holds the last
    ``window`` positions before ``start`` in order (see ``_sliding_writes``),
    and a query sees the positions less than ``window`` behind it."""
    call = _cpu_call(gm.graph)
    slots = _constant(
        gm, f"_hf_static_slots_{window}", torch.arange(window).view(1, 1, 1, window)
    )
    rows = _constant(
        gm, f"_hf_static_rows_{steps}", torch.arange(steps).view(1, 1, steps, 1)
    )
    cols = _constant(
        gm, f"_hf_static_cols_{steps}", torch.arange(steps).view(1, 1, 1, steps)
    )
    first = call(aten.clamp.default, (call(aten.sub.Tensor, (start, window)), 0))
    # Query row i is at start + i; slot j holds position first + j.
    behind = call(
        aten.sub.Tensor,
        (call(aten.add.Tensor, (rows, start)), call(aten.add.Tensor, (slots, first))),
    )
    past = call(
        aten.logical_and.default,
        (call(aten.lt.Tensor, (slots, start)), call(aten.lt.Scalar, (behind, window))),
    )
    new = call(
        aten.logical_and.default,
        (
            call(aten.le.Tensor, (cols, rows)),
            call(aten.lt.Scalar, (call(aten.sub.Tensor, (rows, cols)), window)),
        ),
    )
    return call(aten.cat.default, ([past, new], -1))


def _sliding_writes(gm, cache, rows, start, valid, window: int, steps: int):
    """Shift to keep the latest window, then write real rows in place.

    Padding writes repeat the final real row's index AND data, so colliding
    indices store identical values irrespective of kernel write order.
    """
    assert 0 < steps <= window
    call = _cpu_call(gm.graph)
    slots = _constant(gm, f"_hf_static_slot_index_{window}", torch.arange(window))
    steps_ = _constant(gm, f"_hf_static_steps_{steps}", torch.arange(steps))
    start, valid = (call(aten.view.default, (t, [1])) for t in (start, valid))
    first = call(aten.clamp.default, (call(aten.sub.Tensor, (start, window)), 0))
    end = call(aten.add.Tensor, (start, valid))
    first_after = call(aten.clamp.default, (call(aten.sub.Tensor, (end, window)), 0))
    shift = call(aten.sub.Tensor, (first_after, first))
    source = call(
        aten.clamp.default, (call(aten.add.Tensor, (slots, shift)), None, window - 1)
    )
    call(
        aten.index_copy_.default,
        (cache, 2, slots, call(aten.index_select.default, (cache, 2, source))),
    )
    row = call(aten.minimum.default, (steps_, call(aten.sub.Tensor, (valid, 1))))
    slot = call(aten.sub.Tensor, (call(aten.add.Tensor, (row, start)), first_after))
    return call(
        aten.index_copy_.default,
        (cache, 2, slot, call(aten.index_select.default, (rows, 2, row))),
    )


def _window_full_mask(gm, start, length: int, steps: int, window: int):
    """[context-length cache slots | the step's rows] each query row sees
    within ``window`` positions behind it (slot j holds position j)."""
    call = _cpu_call(gm.graph)
    slots = _constant(
        gm, f"_hf_static_slots_{length}", torch.arange(length).view(1, 1, 1, length)
    )
    rows = _constant(
        gm, f"_hf_static_rows_{steps}", torch.arange(steps).view(1, 1, steps, 1)
    )
    cols = _constant(
        gm, f"_hf_static_cols_{steps}", torch.arange(steps).view(1, 1, 1, steps)
    )
    behind = call(aten.sub.Tensor, (call(aten.add.Tensor, (rows, start)), slots))
    past = call(
        aten.logical_and.default,
        (call(aten.lt.Tensor, (slots, start)), call(aten.lt.Scalar, (behind, window))),
    )
    new = call(
        aten.logical_and.default,
        (
            call(aten.le.Tensor, (cols, rows)),
            call(aten.lt.Scalar, (call(aten.sub.Tensor, (rows, cols)), window)),
        ),
    )
    return call(aten.cat.default, ([past, new], -1))


def _attention_concat(
    call, q, cache_kt, cache_v, k_new, v_new, keep, scale, masked_softmax
):
    """One QK and one PV matmul over the cache and current-token rows, with
    grouped-query heads expanded in K and V for QNN's per-head split (mha2sha)."""
    _, hq, steps, d = q.meta["val"].shape
    _, hkv, length, _ = cache_v.meta["val"].shape
    rep, total = hq // hkv, length + steps
    k_all = call(
        aten.cat.default,
        ([cache_kt, call(aten.transpose.int, (k_new, 2, 3))], 3),
    )
    v_all = call(aten.cat.default, ([cache_v, v_new], 2))
    if rep > 1:

        def expand(t, shape):
            t = call(
                aten.expand.default,
                (call(aten.unsqueeze.default, (t, 2)), [1, hkv, rep, *shape]),
            )
            return call(aten.reshape.default, (t, [1, hq, *shape]))

        k_all, v_all = expand(k_all, [d, total]), expand(v_all, [total, d])
    scores = call(aten.mul.Tensor, (call(aten.matmul.default, (q, k_all)), scale))
    probs = call(
        aten._softmax.default, (_masked(call, scores, keep, masked_softmax), -1, False)
    )
    return call(aten.matmul.default, (probs, v_all))


def _attention_sha(
    call, q, cache_kt, cache_v, k_new, v_new, keep, scale, masked_softmax
):
    """One single-head attention per query head over [cache, the step's rows],
    each KV head's keys and values concatenated once for its query heads: the
    per-head graph QNN's mha2sha makes of the concat form, written out (QNN
    cannot expand quantized 16-bit K/V for grouped-query heads)."""
    _, hq, steps, d = q.meta["val"].shape
    _, hkv, _, _ = cache_v.meta["val"].shape
    rep = hq // hkv

    def head(t, i):
        return call(aten.slice.Tensor, (t, 1, i, i + 1))

    outs = []
    for g in range(hkv):
        kg = call(
            aten.cat.default,
            ([head(cache_kt, g), call(aten.transpose.int, (head(k_new, g), 2, 3))], 3),
        )
        vg = call(aten.cat.default, ([head(cache_v, g), head(v_new, g)], 2))
        for r in range(rep):
            scores = call(
                aten.mul.Tensor,
                (call(aten.matmul.default, (head(q, g * rep + r), kg)), scale),
            )
            probs = call(
                aten._softmax.default,
                (_masked(call, scores, keep, masked_softmax), -1, False),
            )
            outs.append(call(aten.matmul.default, (probs, vg)))
    return call(aten.cat.default, (outs, 1))


def _attention_split(
    call, q, cache_kt, cache_v, k_new, v_new, keep, scale, masked_softmax
):
    """The cache and current-token rows in separate matmuls, grouped-query heads folded
    into the query rows, so each KV head's cache is read once and never copied:

        scores = [q @ K_cache^T, q @ K_new^T]
        out    = softmax(scores)[:L] @ V_cache + softmax(scores)[L:] @ V_new

    Quantized, the two probability slices and partial products are calibrated
    separately, so this assumes min/max observers (as the LLM recipes use):
    how attention divides between the cache and current-token rows varies from step
    to step, and a moving average would clip the steps it doesn't favor.

    ``keep`` is the step mask repeated for each query head of a KV group."""
    _, hq, steps, d = q.meta["val"].shape
    _, hkv, length, _ = cache_v.meta["val"].shape
    rep = hq // hkv
    qg = call(aten.reshape.default, (q, [1, hkv, rep * steps, d]))
    scores = call(
        aten.cat.default,
        (
            [
                call(aten.matmul.default, (qg, cache_kt)),
                call(
                    aten.matmul.default, (qg, call(aten.transpose.int, (k_new, 2, 3)))
                ),
            ],
            -1,
        ),
    )
    scores = call(aten.mul.Tensor, (scores, scale))
    probs = call(
        aten._softmax.default, (_masked(call, scores, keep, masked_softmax), -1, False)
    )
    p_past = call(aten.slice.Tensor, (probs, -1, 0, length))
    p_new = call(aten.slice.Tensor, (probs, -1, length, length + steps))
    out = call(
        aten.add.Tensor,
        (
            call(aten.matmul.default, (p_past, cache_v)),
            call(aten.matmul.default, (p_new, v_new)),
        ),
    )
    return call(aten.reshape.default, (out, [1, hq, steps, d]))


def _cache_attention(graph: torch.fx.Graph):
    """Each attention reading a cache the step writes, with its K and V writes."""
    layers = []
    for sdpa in graph.nodes:
        if (
            sdpa.op != "call_function"
            or sdpa.target != aten.scaled_dot_product_attention.default
        ):
            continue
        k_write = cache_graph.producer_cache_write(sdpa.args[1])
        v_write = cache_graph.producer_cache_write(sdpa.args[2])
        if k_write is not None and v_write is not None:
            layers.append((sdpa, k_write, v_write))
    if not layers:
        raise ValueError("No attention reading a written cache was found.")
    return layers


def _fold_small_constants(gm: torch.fx.GraphModule, max_numel: int = 1 << 16) -> None:
    """Evaluate small values computed from constants alone (e.g. RoPE's
    ``inv_freq`` cast and reshape) into constants, so that the position tables
    take in all of RoPE: otherwise its angle is computed on HTP, which loses
    precision once quantized. Large values (weights) are left alone."""
    graph = gm.graph
    count = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or not node.all_input_nodes:
            continue
        if any(i.op != "get_attr" for i in node.all_input_nodes):
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.numel() > max_numel:
            continue

        def value(a):
            if isinstance(a, torch.fx.Node):
                obj = gm
                for atom in a.target.split("."):
                    obj = getattr(obj, atom)
                return obj
            return a

        with torch.no_grad():
            folded = node.target(
                *torch.fx.node.map_arg(node.args, value),
                **torch.fx.node.map_arg(node.kwargs, value),
            )
        if not isinstance(folded, torch.Tensor):
            continue
        with graph.inserting_before(node):
            constant = _constant(
                gm, f"_hf_static_constant_{count}", folded.contiguous()
            )
        count += 1
        constant.meta = dict(node.meta)
        node.replace_all_uses_with(constant)
        graph.erase_node(node)


def _name_tables_by_shape(gm: torch.fx.GraphModule) -> None:
    """Methods lowered together share constants by name; the position tables of
    a prompt chunk and of a single token can differ in shape under the same name."""
    renamed = set()
    for node in gm.graph.nodes:
        if node.op != "get_attr" or not node.target.startswith("_qnn_position_table_"):
            continue
        table = getattr(gm, node.target)
        name = f"{node.target}_{'x'.join(str(d) for d in table.shape)}"
        if not hasattr(gm, name):
            gm.register_buffer(name, table)
        renamed.add(node.target)
        node.target = name
    for name in renamed:
        del gm._buffers[name]


def _rewrite_attention(  # noqa: C901
    gm: torch.fx.GraphModule,
    masked_softmax: bool,
    form: str,
    windows: Sequence[Optional[int]],
    key_layout: str = "hf",
    sliding_mode: str = "window",
) -> int:
    """Attend over incoming caches and current-token rows, then update caches on CPU."""
    graph = gm.graph
    call, cpu = graph.call_function, _cpu_call(graph)
    layers = _cache_attention(graph)
    if len(layers) != len(windows):
        raise ValueError(
            f"Expected {len(windows)} cache attention layers, matched {len(layers)}."
        )
    attention = {
        "concat": _attention_concat,
        "split": _attention_split,
        "sha": _attention_sha,
    }[form]
    masks, windowed = {}, {}
    for i, (sdpa, k_write, v_write) in enumerate(layers):
        q = sdpa.args[0]
        mask = sdpa.args[3] if len(sdpa.args) > 3 else sdpa.kwargs.get("attn_mask")
        if mask is None or mask.meta["val"].dtype != torch.bool:
            raise ValueError(f"{sdpa.name}: expected a bool attention mask.")
        cache_k, cache_v = k_write.args[0], v_write.args[0]
        length, steps = cache_v.meta["val"].shape[2], q.meta["val"].shape[2]
        rep = q.meta["val"].shape[1] // cache_v.meta["val"].shape[1]
        sliding = windows[i] is not None
        full_window = sliding and sliding_mode == "full"
        if sliding and not full_window and key_layout != "hf":
            raise ValueError("Window-sized sliding caches require HF's key layout.")
        mask.meta.setdefault("custom", {})[CPU_META] = True
        for write in (k_write, v_write):
            write.meta.setdefault("custom", {})[CPU_META] = True
            stack = [write.args[2]]
            seen = set()
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                node.meta.setdefault("custom", {})[CPU_META] = True
                stack.extend(node.all_input_nodes)
        with graph.inserting_before(sdpa):
            if mask not in masks:
                index = k_write.args[2]
                start = cpu(
                    aten.view.default,
                    (cpu(aten.slice.Tensor, (index, 0, 0, 1)), [1, 1, 1, 1]),
                )
                if full_window:
                    keep = _window_full_mask(gm, start, length, steps, windows[i])
                elif sliding:
                    keep = _window_mask(gm, start, length, steps)
                else:
                    keep = _step_mask(gm, mask, index, start, length, steps)
                masks[mask] = (keep, start)
            keep, start = masks[mask]
            if form == "split" and rep > 1:
                if (mask, rep) not in masks:
                    masks[(mask, rep)] = cpu(
                        aten.repeat.default, (keep, [1, 1, rep, 1])
                    )
                keep = masks[(mask, rep)]
            scale = sdpa.args[6] if len(sdpa.args) > 6 else sdpa.kwargs.get("scale")
            cache_kt = (
                cache_k
                if key_layout == "transposed"
                else call(aten.transpose.int, (cache_k, 2, 3))
            )
            out = attention(
                call,
                q,
                cache_kt,
                cache_v,
                k_write.args[3],
                v_write.args[3],
                keep,
                scale or q.meta["val"].shape[3] ** -0.5,
                masked_softmax,
            )
        out.meta = dict(sdpa.meta)
        sdpa.replace_all_uses_with(out)
        graph.erase_node(sdpa)
        if sliding and not full_window:
            windowed[k_write] = windowed[v_write] = (mask, start, length, steps)
    graph.eliminate_dead_code()
    output = next(n for n in graph.nodes if n.op == "output")
    valid = {}
    for _, k_write, v_write in layers:
        if key_layout == "transposed":
            with graph.inserting_before(output):
                rows = call(aten.transpose.int, (k_write.args[3], 2, 3))
                k_t = _cpu_call(graph)(
                    aten.index_copy_.default,
                    (k_write.args[0], 3, k_write.args[2], rows),
                )
            k_write.replace_all_uses_with(k_t)
            graph.erase_node(k_write)
            k_write = k_t
        for write in (k_write, v_write):
            if write not in windowed:
                output.prepend(write)
                continue
            mask, start, length, steps = windowed[write]
            with graph.inserting_before(output):
                if mask not in valid:
                    rows = cpu(aten.any.dim, (mask, -1, True))
                    valid[mask] = cpu(aten.sum.dim_IntList, (rows, [2], True))
                shifted = _sliding_writes(
                    gm, write.args[0], write.args[3], start, valid[mask], length, steps
                )
            write.replace_all_uses_with(shifted)
            graph.erase_node(write)
    # Counter updates are part of the mutable-input protocol, not model compute.
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and node.target == aten.add_.Tensor
            and node.args[0].meta.get("custom", {}).get(CPU_META)
        ):
            node.meta.setdefault("custom", {})[CPU_META] = True
    return len(layers)


def to_hf_static(
    exported_program: ExportedProgram,
    masked_softmax: bool = False,
    attention: str = "split",
    key_layout: str = "hf",
    int32_token_ids: bool = False,
) -> ExportedProgram:
    """Rewrite a specialized graph keeping its per-step signature; with
    ``key_layout="transposed"`` the key caches are [1, H, D, L]. With
    ``int32_token_ids`` the embedding gathers by int32 ids, converted on CPU
    (the input stays HF's int64)."""
    if attention not in ("split", "concat", "sha"):
        raise ValueError(f"Unknown attention form {attention!r}.")
    if key_layout not in ("hf", "transposed"):
        raise ValueError(f"Unknown key layout {key_layout!r}.")
    reshaped = key_layout != "hf"
    # A plain module: the exported one checks inputs against the old shapes.
    gm = exported_program.module(check_guards=not reshaped)
    if reshaped:
        gm = torch.fx.GraphModule(gm, gm.graph)
    graph = gm.graph
    position_ids = next(
        n for n in graph.nodes if n.op == "placeholder" and n.name == "position_ids"
    )
    layout = HfStaticLayout.of(exported_program)
    if int32_token_ids:
        for node in list(graph.nodes):
            if node.target == aten.embedding.default:
                with graph.inserting_before(node):
                    ids = _cpu_call(graph)(
                        aten._to_copy.default, (node.args[1],), {"dtype": torch.int32}
                    )
                node.replace_input_with(node.args[1], ids)
    _fold_small_constants(gm)
    # Only RoPE depends on the position alone; the cache writes follow the counters.
    position_ids.meta.setdefault("custom", {})[CPU_META] = True
    cache_graph.fold_position_tables(
        gm,
        position_ids,
        position_ids,
        layout.max_seq_len,
        layout.dtype,
        index_meta={"custom": {CPU_META: True}},
    )
    cache_graph.split_rotate_half(graph)
    cache_graph.fold_weight_arithmetic(gm)
    cache_graph.norm_after_head_transpose(graph)
    # Unlifting the program later (calibration, lowering) fails on folded
    # values registered as non-persistent buffers.
    gm._non_persistent_buffers_set.clear()
    _name_tables_by_shape(gm)
    windows = exported_program.graph_module.meta.get(
        "hf_static_windows", [None] * layout.n_layers
    )
    matched = _rewrite_attention(
        gm,
        masked_softmax,
        attention,
        windows,
        key_layout,
        exported_program.graph_module.meta.get("hf_static_sliding", "window"),
    )
    if matched != layout.n_layers:
        raise ValueError(
            f"Expected {layout.n_layers} cache attention layers, matched {matched}."
        )
    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    args, kwargs = copy.deepcopy(exported_program.example_inputs)
    flat, spec = pytree.tree_flatten((args, kwargs))
    keys = set([n for n in layout.input_names if n.startswith(CACHE_PREFIX)][0::3])
    for i, name in enumerate(layout.input_names):
        if key_layout == "transposed" and name in keys:
            flat[i] = flat[i].transpose(2, 3).contiguous()
    args, kwargs = pytree.tree_unflatten(flat, spec)
    with torch.fx.traceback.preserve_node_meta():
        program = torch.export.export(gm, args, kwargs, strict=False)
    program.graph_module.meta.update(exported_program.graph_module.meta)
    program.graph_module.meta.update(hf_static_key_layout=key_layout)
    return program


def tag_cache_mutations(program: ExportedProgram) -> None:
    """Tag copies introduced by functionalization of the cache protocol."""
    state_names = [
        s.arg.name
        for s in program.graph_signature.input_specs
        if s.kind.name == "USER_INPUT" and s.arg.name.startswith(CACHE_PREFIX)
    ]
    counters = set(state_names[2::3])
    nodes = {n.name: n for n in program.graph.nodes}
    outputs = [
        (spec, spec.target)
        for spec in program.graph_signature.output_specs
        if spec.kind.name == "USER_INPUT_MUTATION" and spec.target in state_names
    ]
    user_outputs = [
        spec
        for spec in program.graph_signature.output_specs
        if spec.kind.name == "USER_OUTPUT"
    ]
    if len(user_outputs) == len(state_names) + 1:
        outputs.extend(zip(user_outputs[1:], state_names))
    for spec, target in outputs:
        stack = [nodes[spec.arg.name]]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.op != "call_function":
                continue
            try:
                op = unwrap_op_overload(node.target)
            except TypeError:
                continue
            if target in counters:
                node.meta.setdefault("custom", {})[CPU_META] = True
                stack.extend(node.all_input_nodes)
            elif op in (aten.copy.default, aten.copy_.default):
                node.meta.setdefault("custom", {})[CPU_META] = True
                stack.append(node.args[1])
            elif op in (
                aten.index_put.default,
                aten.index_put_.default,
                aten.index_copy.default,
                aten.index_copy_.default,
            ):
                node.meta.setdefault("custom", {})[CPU_META] = True
                stack.append(node.args[0])


def inplace_input_writes(edge_program: ExportedProgram) -> int:
    """Write each mutated user input in place, on the caller's buffer.

    Functionalization turns a cache write into an ``index_put`` producing a new
    tensor plus ``copy`` nodes back to the input, which ExecuTorch emits as
    full-cache copies. Each such write (or chain of writes) becomes
    ``index_put_`` on the input itself, and a whole new value (a counter)
    becomes ``copy_``. A write is left alone if the input is read after it.
    Returns the number of inputs now written in place.
    """
    graph = edge_program.graph
    order = {n: i for i, n in enumerate(graph.nodes)}
    output = next(n for n in graph.nodes if n.op == "output")
    outs = list(output.args[0])
    placeholders = {n.name: n for n in graph.nodes if n.op == "placeholder"}
    rewritten = 0
    for i, spec in enumerate(edge_program.graph_signature.output_specs):
        if spec.kind.name != "USER_INPUT_MUTATION":
            continue
        target, node, chain = placeholders.get(spec.target), outs[i], []
        while (
            node.op == "call_function"
            and node.target == ops.edge.aten.copy.default
            and node.args[0] is target
        ):
            chain.append(node)
            node = node.args[1]
        if node.op != "call_function":
            continue
        if node.target in (
            ops.edge.aten.index_put_.default,
            ops.edge.aten.copy_.default,
        ):
            if node.args[0] is target:
                rewritten += 1
            continue
        writes = [node]
        while (
            writes[-1].target == ops.edge.aten.index_put.default
            and writes[-1].args[0] is not target
            and writes[-1].args[0].op == "call_function"
            and writes[-1].args[0].target == ops.edge.aten.index_put.default
            and len(writes[-1].args[0].users) == 1
        ):
            writes.append(writes[-1].args[0])
        first = writes[-1]
        if any(
            order[user] > order[first]
            for user in target.users
            if user is not first and user not in chain and user.op != "output"
        ):
            continue
        if first.target == ops.edge.aten.index_put.default and first.args[0] is target:
            inplace = target
            for write in reversed(writes):
                with graph.inserting_before(write):
                    inplace = graph.call_function(
                        ops.edge.aten.index_put_.default, (inplace, *write.args[1:])
                    )
                inplace.meta = dict(write.meta)
            node = writes[0]
        else:
            with graph.inserting_after(node):
                inplace = graph.call_function(
                    ops.edge.aten.copy_.default, (target, node)
                )
            inplace.meta = dict(node.meta)
        outs = [inplace if o in set(chain) | {node} else o for o in outs]
        rewritten += 1
    output.args = (tuple(outs),)
    graph.eliminate_dead_code()
    edge_program.graph_module.recompile()
    edge_program._graph_signature = _get_updated_graph_signature(
        edge_program.graph_signature, edge_program.graph_module
    )
    return rewritten


def flat_call(
    exported_program: ExportedProgram, module: Optional[torch.nn.Module] = None
):
    """The program (or ``module``, e.g. its PT2E-prepared form) as a function of
    its flat user inputs, returning its flat outputs."""
    module = module if module is not None else exported_program.module()
    spec = exported_program.call_spec.in_spec

    def call(*flat):
        args, kwargs = pytree.tree_unflatten(list(flat), spec)
        return pytree.tree_flatten(module(*args, **kwargs))[0]

    return call
