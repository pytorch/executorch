# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Remember PyTorch's analysis of which tensors a user Triton kernel accesses.

Tracing a call of a user-defined Triton kernel (functionalization in
``torch.export``, and Inductor's mutation propagation and lowering) runs
``torch._higher_order_ops.triton_kernel_wrap.identify_accessed_tensors``: it
generates the kernel's TTIR and walks it to find the tensors the kernel reads
and writes. Nothing caches it, so a model whose every layer launches the same
kernels repeats it for every call site of every exported method; for the Muse
Glimmer batching export that is about a sixth of the export.

The analysis depends only on the kernel and on what ``generate_ttir`` turns
the arguments into: a fake or Inductor tensor becomes a fresh tensor of its
dtype, a symbolic value becomes 2, and every other value is used as is. The
first call with a given (kernel, normalized arguments) goes through PyTorch;
repeats return a copy of its result. Calls this cannot key exactly (TMA
descriptors, unhashable values) always go through PyTorch.
"""

from __future__ import annotations

import collections
import copy
import functools
import threading
from typing import Any, Dict, Optional, Tuple

import torch

# Entries kept, least recently used dropped first, so a long-running process
# does not keep every kernel and specialization it ever traced.
MAX_ENTRIES = 16384

_LOCK = threading.Lock()
# key -> (kernel, result). Holding the kernel keeps its id from being reused
# while its entry exists.
_CACHE: "collections.OrderedDict[Tuple, Tuple[Any, Any]]" = collections.OrderedDict()
# key -> lock held while that key's first call goes through PyTorch, so a
# concurrent caller of the same key waits for it instead of repeating it.
_PENDING: Dict[Tuple, threading.Lock] = {}
_STATS = {"hits": 0, "misses": 0, "uncached": 0}
_installed = False

_UNKEYABLE = object()


def _argument_key(value: Any) -> Any:
    """What ``generate_ttir`` makes of ``value``, as a hashable key."""
    import sympy
    from torch._inductor.ir import TensorBox
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool, sympy.Expr)):
        return ("symbolic",)
    if isinstance(value, (FakeTensor, TensorBox)):
        # generate_ttir replaces these with torch.empty(2, dtype=value.dtype):
        # their shape, strides and device do not reach the analysis.
        return ("tensor", value.dtype)
    if isinstance(value, torch.Tensor):
        # Passed through as is: Triton specializes on its pointer's alignment.
        return ("real tensor", value.dtype, value.data_ptr() % 16)
    try:
        hash(value)
    except TypeError:
        return _UNKEYABLE
    return (type(value).__module__, type(value).__qualname__, value)


def _call_key(kernel: Any, kwargs: Dict[str, Any], tma_descriptor_metadata) -> Optional[Tuple]:
    if tma_descriptor_metadata:
        return None
    arguments = []
    for name in sorted(kwargs):
        key = _argument_key(kwargs[name])
        if key is _UNKEYABLE:
            return None
        arguments.append((name, key))
    return (id(kernel), tuple(arguments))


def _count(name: str) -> None:
    with _LOCK:
        _STATS[name] += 1


def _lookup(key: Tuple) -> Optional[Tuple[Any, Any]]:
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None:
            _CACHE.move_to_end(key)
            _STATS["hits"] += 1
        return entry


def _cached(identify_accessed_tensors):
    @functools.wraps(identify_accessed_tensors)
    def wrapper(kernel, kwargs, tma_descriptor_metadata):
        key = _call_key(kernel, kwargs, tma_descriptor_metadata)
        if key is None:
            _count("uncached")
            return identify_accessed_tensors(kernel, kwargs, tma_descriptor_metadata)
        entry = _lookup(key)
        if entry is not None:
            return copy.deepcopy(entry[1])
        with _LOCK:
            pending = _PENDING.setdefault(key, threading.Lock())
        with pending:
            # Another caller may have finished this key while we waited.
            entry = _lookup(key)
            if entry is not None:
                return copy.deepcopy(entry[1])
            try:
                result = identify_accessed_tensors(
                    kernel, kwargs, tma_descriptor_metadata
                )
                with _LOCK:
                    _STATS["misses"] += 1
                    _CACHE[key] = (kernel, copy.deepcopy(result))
                    while len(_CACHE) > MAX_ENTRIES:
                        _CACHE.popitem(last=False)
            finally:
                with _LOCK:
                    _PENDING.pop(key, None)
        return result

    wrapper.__wrapped_uncached__ = identify_accessed_tensors
    return wrapper


def install() -> None:
    """Caches ``identify_accessed_tensors`` for this process. Idempotent."""
    global _installed
    with _LOCK:
        if _installed:
            return
        from torch._higher_order_ops import triton_kernel_wrap

        original = getattr(triton_kernel_wrap, "identify_accessed_tensors", None)
        if original is None:
            # A PyTorch without this analysis: nothing to cache.
            return
        if hasattr(original, "__wrapped_uncached__"):
            _installed = True
            return
        # Both callers (get_mutated_tensors here and Inductor's
        # UserDefinedTritonKernel) look the function up on this module.
        triton_kernel_wrap.identify_accessed_tensors = _cached(original)
        _installed = True


def clear() -> None:
    """Drops every cached analysis and resets the counters ``stats`` reports."""
    with _LOCK:
        _CACHE.clear()
        for name in _STATS:
            _STATS[name] = 0


def stats() -> Dict[str, int]:
    """A snapshot of the hit, miss and uncached call counts since ``clear``."""
    with _LOCK:
        return dict(_STATS)
