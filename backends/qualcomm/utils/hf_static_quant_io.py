# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized cache I/O for hf_static programs (ported from the pre-split
stack 2 branch, commit 24ce101bb7): each cache keeps one integer encoding in
every method, so HTP reads it without converting it each step and the step's
rows leave the delegate in that encoding for the CPU write."""

import operator
from typing import Dict, List, Optional

import torch
from executorch.exir.pass_base import PassResult
from executorch.extension.llm.export.hf_static import CACHE_PREFIX as _CACHE_PREFIX
from torch.export import ExportedProgram

aten = torch.ops.aten
_PER_TENSOR_QDQ = (
    torch.ops.quantized_decomposed.quantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
)
_CACHE_WRITES = {
    aten.index_copy.default: 3,
    aten.index_copy_.default: 3,
    aten.index_put.default: 2,
    aten.index_put_.default: 2,
}
# A cache's integer type by its quantization range (signed encodings keep the
# unsigned type of their width: a runner treats the contents as opaque).
QUANT_RANGES = {
    (0, 65535): torch.uint16,
    (-32767, 32767): torch.uint16,
    (-32768, 32767): torch.uint16,
    (0, 255): torch.uint8,
    (-127, 127): torch.uint8,
    (-128, 127): torch.uint8,
}


def _write_requantizations(graph_module: torch.fx.GraphModule) -> PassResult:
    """Materialize the requantization of the rows a cache write stores (see
    ``_rows_encodings``) as a conversion inside the delegate, so the rows
    leave it in the cache's encoding: QNN keeps a requantization as a note on
    the producer, by consumer, and adds the conversion only when it compiles
    the delegate, where a consumer outside it (the CPU write) isn't seen."""
    from executorch.backends.qualcomm.utils.constants import (
        QCOM_QUANT_ATTRS,
        QCOM_QUANTIZED_IO,
        QCOM_REQUANTIZE,
    )
    from executorch.exir.dialects._ops import ops as exir_ops

    graph = graph_module.graph
    for _, write, index in list(_cache_writes(graph)):
        rows = write.args[index]
        attrs = rows.meta.get(QCOM_REQUANTIZE, {}).pop(write.name, None)
        if attrs is None:
            continue
        with graph.inserting_after(rows):
            converted = graph.call_function(
                exir_ops.edge.aten._to_copy.default, (rows,)
            )
        converted.meta = {
            "val": rows.meta["val"],
            QCOM_QUANT_ATTRS: attrs,
            QCOM_QUANTIZED_IO: rows.meta.get(QCOM_QUANTIZED_IO),
        }
        write.replace_input_with(rows, converted)
        if not rows.meta[QCOM_REQUANTIZE]:
            del rows.meta[QCOM_REQUANTIZE]
    graph_module.recompile()
    return PassResult(graph_module, True)


def _quant_io_passes(dtype_fn):
    from executorch.backends.qualcomm._passes import TagQuantIO
    from executorch.backends.qualcomm._passes.qnn_pass_manager import (
        get_qnn_pass_manager_cls,
    )
    from executorch.backends.qualcomm.utils.constants import (
        QCOM_PASS_ACTIVATE_KEY,
        QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY,
    )

    passes_job = get_qnn_pass_manager_cls().get_capture_program_passes()
    passes_job[TagQuantIO][QCOM_PASS_ACTIVATE_KEY] = True
    passes_job[TagQuantIO][QCOM_PASS_ARGS_KWARGS_DEFAULTS_KEY][
        "get_quant_io_dtype_fn"
    ] = dtype_fn
    return passes_job


def _cache_writes(graph: torch.fx.Graph):
    """Each cache input written by the step (ATen or edge graph), with its
    write and the write's argument index of the new rows."""
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        # An edge op wraps its ATen op.
        index = _CACHE_WRITES.get(
            node.target, _CACHE_WRITES.get(getattr(node.target, "_op", None))
        )
        cache = node.args[0] if index is not None else None
        if (
            isinstance(cache, torch.fx.Node)
            and cache.op == "placeholder"
            and cache.name.startswith(_CACHE_PREFIX)
        ):
            yield cache, node, index


# Ops that only move or pick values: a cache keeps one encoding through them
# (through a concatenation with the new rows too, as kv_cache_io's shares it).
_SAME_VALUES = (
    aten.cat.default,
    aten.transpose.int,
    aten.permute.default,
    aten.view.default,
    aten.reshape.default,
    aten.unsqueeze.default,
    aten.squeeze.dim,
    aten.slice.Tensor,
    aten.select.int,
    aten.expand.default,
)


def _encoding(node: torch.fx.Node) -> tuple:
    """A quantize/dequantize node's (scale, zero point, quant min, quant max)."""
    return tuple(node.args[1:5])


def _value_encodings(value: torch.fx.Node):
    """The quantize/dequantize nodes of ``value`` and of what only moves its
    values (``_SAME_VALUES``), up to the ops that compute with it; and the
    quantizes whose values those ops read."""
    quantize, dequantize = _PER_TENSOR_QDQ
    group, read, seen, stack = [], [], set(), list(value.users)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.target in _PER_TENSOR_QDQ:
            group.append(node)
            if node.target == dequantize and any(
                u.target not in (*_SAME_VALUES, quantize) for u in node.users
            ):
                read.append(node.args[0])
        elif node.target not in _SAME_VALUES:
            continue
        stack.extend(node.users)
    return group, read


def _union(a: tuple, b: tuple) -> tuple:
    """An encoding of ``a``'s type covering both ranges: what one min/max
    observer over both values would give. Symmetric (``a``'s zero point at
    the middle of its range) stays symmetric."""
    (sa, za, qmin, qmax), (sb, zb, bmin, bmax) = a, b
    lo = min((qmin - za) * sa, (bmin - zb) * sb, 0.0)
    hi = max((qmax - za) * sa, (bmax - zb) * sb, 0.0)
    if za == (qmin + qmax + 1) // 2:
        scale = max(-lo, hi) / ((qmax - qmin) / 2)
        return (scale, za, qmin, qmax)
    scale = (hi - lo) / (qmax - qmin)
    zero = int(min(max(round(qmin - lo / scale), qmin), qmax))
    return (scale, zero, qmin, qmax)


def _cache_encoding_groups(program: ExportedProgram):
    """Per cache input: the quantize/dequantize nodes of the cache as stored
    and as attention reads it, of the rows its write stores and of the write's
    result; and the quantize whose type attention reads the cache in."""
    groups = {}
    for cache, write, index in _cache_writes(program.graph):
        rows = write.args[index]
        if rows.target != _PER_TENSOR_QDQ[1]:
            # E.g. a window-sized sliding cache's shift (_sliding_writes).
            raise ValueError(
                f"{cache.name}: the cache's write isn't of quantized rows; "
                "export with cache_dtype='float' (or sliding='full')."
            )
        source = rows
        while source.target in _PER_TENSOR_QDQ:
            source = source.args[0]
        cache_group, read = _value_encodings(cache)
        kinds = {(*_encoding(q)[2:], q.args[5]) for q in read}
        if len(kinds) != 1:
            raise ValueError(
                f"{cache.name}: attention reads the cache in {len(kinds)} types; "
                "export with cache_dtype='float'."
            )
        groups[cache.name] = (
            cache_group
            + _rows_encodings(program, write, index, source, read[0])
            + _value_encodings(write)[0],
            read[0],
        )
    return groups


def _rows_encodings(program, write, index, source, cache_q) -> List[torch.fx.Node]:
    """Requantize the rows a cache write stores into the cache's encoding: the
    rows' producer keeps its own (an op computing them may not produce the
    cache's type), and the write and the ops that only move the rows (e.g.
    into the concatenation with the cache) read them requantized, each through
    its own dequantize (QNN records one requantization per consumer; see
    ``HfStaticPartitioner`` for the write's). Returns the quantize/dequantize
    nodes that take the cache's encoding."""
    quantize, dequantize = _PER_TENSOR_QDQ
    rows, graph = write.args[index], program.graph
    encoding = (*_encoding(cache_q), cache_q.args[5])
    consumers = [(write, rows)]
    for q_rows in [u for u in source.users if u.target == quantize]:
        for dq_rows in [u for u in q_rows.users if u.target == dequantize]:
            consumers += [
                (u, dq_rows) for u in dq_rows.users if u.target in _SAME_VALUES
            ]
    with graph.inserting_after(rows):
        q = graph.call_function(quantize, (rows, *encoding))
    q.meta["val"] = rows.meta["val"].to(cache_q.args[5])
    group = [q]
    for user, old in consumers:
        with graph.inserting_after(q):
            dq = graph.call_function(dequantize, (q, *encoding))
        dq.meta["val"] = rows.meta["val"]
        user.replace_input_with(old, dq)
        group.append(dq)
        if user is not write:
            group += _value_encodings(user)[0]
    return group


def _cache_quant_io(programs: Dict[str, ExportedProgram]) -> Dict[str, tuple]:
    """Give each cache one encoding in every method: as stored, as attention
    reads it, and for the rows each step writes. So on HTP the cache is read
    without requantizing it each step, and a step's rows leave the graph
    quantized as the cache holds them, for the CPU write to copy bytes
    (``kv_cache_io`` shares the encodings of the cache and the new rows in its
    concat the same way). The encoding is of the type attention reads the
    cache in (e.g. the symmetric values a 16-bit matmul needs, or 8-bit keys)
    and covers every range the quantizer observed for the cache and the rows,
    in every method (the runner hands the same buffers to all of them).
    Returns each cache input's encoding (scale, zero point, quant min, quant
    max)."""
    per_method = [_cache_encoding_groups(p) for p in programs.values()]
    encodings = {}
    for name in per_method[0]:
        reads = [groups[name][1] for groups in per_method]
        if len({(*_encoding(q)[2:], q.args[5]) for q in reads}) != 1:
            raise ValueError(f"{name}: the methods read the cache in different types.")
        nodes = [n for groups in per_method for n in groups[name][0]]
        union = _encoding(reads[0])
        for node in nodes:
            union = _union(union, _encoding(node))
        for node in nodes:
            node.args = (node.args[0], *union, reads[0].args[5], *node.args[6:])
        (qmin, qmax) = union[2:]
        if (qmin, qmax) not in QUANT_RANGES:
            raise ValueError(f"{name}: unsupported cache range [{qmin}, {qmax}].")
        encodings[name] = union
    for program in programs.values():
        program.graph.eliminate_dead_code()
        program.graph_module.recompile()
    return encodings


def _check_cache_writes(edge, encodings: Dict[str, tuple]) -> None:
    """Every cache write of a lowered program stores rows that leave the
    delegate in the cache's encoding: the CPU write copies bytes."""
    from executorch.backends.qualcomm.utils.constants import (
        QCOM_QUANT_ATTRS,
        QCOM_QUANTIZED_IO,
    )

    for method in edge.methods:
        program = edge.exported_program(method)
        for cache, write, index in _cache_writes(program.graph):
            rows = write.args[index]
            delegate = rows.args[0] if rows.target == operator.getitem else None
            if delegate is None or not rows.meta.get(QCOM_QUANTIZED_IO):
                raise RuntimeError(f"{method}: {cache.name}'s rows aren't quantized.")
            lowered = getattr(program.graph_module, delegate.args[0].target)
            graph = lowered.original_module.graph
            produced = graph.output_node().args[0][rows.args[1]]
            # Through the cast BuildQuantIo may have added.
            while produced.op == "call_function" and not produced.meta.get(
                QCOM_QUANT_ATTRS
            ):
                produced = produced.args[0]
            attrs = produced.meta[QCOM_QUANT_ATTRS]
            got = (
                attrs["scale"],
                attrs["zero_point"],
                attrs["quant_min"],
                attrs["quant_max"],
            )
            if not all(
                abs(a - b) <= 1e-9 * max(1.0, abs(b))
                for a, b in zip(got, encodings[cache.name])
            ):
                raise RuntimeError(
                    f"{method}: {cache.name}'s rows leave HTP in {got}, the cache "
                    f"holds {encodings[cache.name]}."
                )


def _logits(graph: torch.fx.Graph) -> Optional[torch.fx.Node]:
    """The [1, N, V] logits value the graph returns."""
    output = graph.output_node()
    values = [a for a in output.args[0] if isinstance(a, torch.fx.Node)]
    writes = {w for _, w, _ in _cache_writes(graph)}
    logits = [
        a
        for a in values
        if a not in writes
        and isinstance(a.meta.get("val"), torch.Tensor)
        and a.meta["val"].dim() == 3
        and a.meta["val"].is_floating_point()
    ]
    return logits[0] if len(logits) == 1 else None


def logits_encoding(program: ExportedProgram) -> tuple:
    """(scale, zero point, quant min, quant max) of a quantized program's logits."""
    logits = _logits(program.graph)
    if logits is None or logits.target != _PER_TENSOR_QDQ[1]:
        raise ValueError("The program's logits aren't dequantized at the output.")
    return _encoding(logits)


def _quant_io_dtype_fn(dtypes: Dict[str, torch.dtype], logits: bool = False):
    """``TagQuantIO`` callback for quantized caches: the cache inputs, the
    step's rows leaving HTP for the write, and the (CPU) write keep the
    cache's integer type, so nothing converts the caches."""

    tagged: Dict[int, Dict[torch.fx.Node, torch.dtype]] = {}

    def fn(node: torch.fx.Node) -> Optional[torch.dtype]:
        if node.op == "placeholder":
            return dtypes.get(node.name)
        if id(node.graph) not in tagged:
            tagged[id(node.graph)] = {}
            for cache, write, index in _cache_writes(node.graph):
                if cache.name in dtypes:
                    for n in (write, write.args[index]):
                        tagged[id(node.graph)][n] = dtypes[cache.name]
            out = _logits(node.graph) if logits else None
            if out is not None:
                tagged[id(node.graph)][out] = torch.uint16
        return tagged[id(node.graph)].get(node)

    return fn


def _unsigned_storage(
    programs: Dict[str, ExportedProgram], encodings: Dict[str, tuple]
) -> Dict[str, tuple]:
    """Store each symmetric 16-bit cache as unsigned 16-bit with a 32768
    offset (the same grid), as QNN's own decoders store their caches: the
    graph reads HTP's native unsigned type and converts it to the symmetric
    values attention needs inside the delegate, and the step's rows leave in
    the unsigned encoding. Returns each cache's stored encoding."""
    quantize, dequantize = _PER_TENSOR_QDQ
    stored = {}
    for name, (scale, zero, qmin, qmax) in encodings.items():
        symmetric = zero == 0 and (qmin, qmax) in ((-32767, 32767), (-32768, 32767))
        stored[name] = (scale, 32768, 0, 65535) if symmetric else encodings[name]
    for program in programs.values():
        graph = program.graph
        for cache, write, index in list(_cache_writes(graph)):
            store, read = stored[cache.name], encodings[cache.name]
            if store == read:
                continue
            read_dtype = next(u for u in cache.users if u.target == quantize).args[5]

            def as_stored(node, store=store):
                node.args = (node.args[0], *store, torch.int32, *node.args[6:])

            for q in [u for u in cache.users if u.target == quantize]:
                as_stored(q)
                for dq in [u for u in q.users if u.target == dequantize]:
                    as_stored(dq)
                    users = list(dq.users)
                    with graph.inserting_after(dq):
                        rq = graph.call_function(quantize, (dq, *read, read_dtype))
                    with graph.inserting_after(rq):
                        rdq = graph.call_function(dequantize, (rq, *read, read_dtype))
                    rq.meta["val"] = dq.meta["val"].to(read_dtype)
                    rdq.meta["val"] = dq.meta["val"]
                    for user in users:
                        user.replace_input_with(dq, rdq)
            rows = write.args[index]
            source = rows.args[0].args[0]
            with graph.inserting_after(source):
                q_rows = graph.call_function(quantize, (source, *store, torch.int32))
            with graph.inserting_after(q_rows):
                dq_rows = graph.call_function(dequantize, (q_rows, *store, torch.int32))
            q_rows.meta["val"] = source.meta["val"].to(torch.int32)
            dq_rows.meta["val"] = source.meta["val"]
            write.replace_input_with(rows, dq_rows)
            for node in _value_encodings(write)[0]:
                as_stored(node)
        graph.eliminate_dead_code()
        program.graph_module.recompile()
    return stored
