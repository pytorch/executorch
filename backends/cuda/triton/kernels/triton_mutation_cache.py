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
kernels repeats it for every call site of every exported method.

The analysis depends only on the traced kernel and on what ``generate_ttir``
turns the arguments into: a fake or Inductor tensor becomes a fresh tensor of
its dtype, a symbolic value becomes 2, and every other value is used as is.
``enabled()`` caches it for the duration of a compile. Calls this cannot key
exactly (TMA descriptors, unhashable values) always go through PyTorch.
"""

from __future__ import annotations

import contextlib
import copy
import threading
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import triton

_UNKEYABLE = object()

_lazy_types: Optional[Tuple[Tuple[type, ...], Tuple[type, ...]]] = None


def _symbolic_and_traced_tensor_types() -> Tuple[Tuple[type, ...], Tuple[type, ...]]:
    """The symbolic-value and fake/Inductor tensor types, imported on first use
    so that tracing does not load Inductor before it is needed."""
    global _lazy_types
    if _lazy_types is None:
        import sympy
        from torch._inductor.ir import TensorBox
        from torch._subclasses.fake_tensor import FakeTensor

        _lazy_types = (
            (torch.SymInt, torch.SymFloat, torch.SymBool, sympy.Expr),
            (FakeTensor, TensorBox),
        )
    return _lazy_types


def _argument_key(value: Any) -> Any:
    """What ``generate_ttir`` makes of ``value``, as a hashable key."""
    symbolic, traced_tensor = _symbolic_and_traced_tensor_types()
    if isinstance(value, symbolic):
        return ("symbolic",)
    if isinstance(value, traced_tensor):
        return ("tensor", value.dtype)
    if isinstance(value, torch.Tensor):
        # Triton specializes on the pointer's alignment, so that is what matters.
        return ("real tensor", value.dtype, value.data_ptr() % 16)
    try:
        hash(value)
    except TypeError:
        return _UNKEYABLE
    return (type(value).__module__, type(value).__qualname__, value)


def _traced_kernel_and_kwargs(kernel: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict]:
    """The JIT function ``generate_ttir`` traces, and the arguments it traces it
    with: an autotuner's first config is merged into the arguments."""
    if isinstance(kernel, triton.runtime.Autotuner) and kernel.configs:
        return kernel.fn, {**kwargs, **kernel.configs[0].kwargs}
    return kernel, kwargs


def _call_key(
    kernel: Any, kwargs: Dict[str, Any], tma_descriptor_metadata
) -> Optional[Tuple[Tuple, Any]]:
    """The cache key of a call and the traced kernel it belongs to, or None."""
    if tma_descriptor_metadata:
        return None
    traced, merged = _traced_kernel_and_kwargs(kernel, kwargs)
    arguments = []
    for name in sorted(merged):
        key = _argument_key(merged[name])
        if key is _UNKEYABLE:
            return None
        arguments.append((name, key))
    return (id(traced), tuple(arguments)), traced


@contextlib.contextmanager
def enabled() -> Iterator[Dict[str, int]]:
    """Caches PyTorch's access analysis while the context is active. Yields the
    hit, miss and uncached call counts. Contexts must not overlap: each one
    patches the function PyTorch calls."""
    from torch._higher_order_ops import triton_kernel_wrap

    original = triton_kernel_wrap.identify_accessed_tensors
    # key -> (traced kernel, result). The kernel is held so its id stays unique.
    entries: Dict[Tuple, Tuple[Any, Any]] = {}
    counts = {"hits": 0, "misses": 0, "uncached": 0}
    lock = threading.RLock()

    def cached(kernel, kwargs, tma_descriptor_metadata):
        keyed = _call_key(kernel, kwargs, tma_descriptor_metadata)
        if keyed is None:
            with lock:
                counts["uncached"] += 1
            return original(kernel, kwargs, tma_descriptor_metadata)
        key, traced = keyed
        with lock:
            entry = entries.get(key)
            if entry is not None:
                counts["hits"] += 1
                return copy.deepcopy(entry[1])
            counts["misses"] += 1
            result = original(kernel, kwargs, tma_descriptor_metadata)
            entries[key] = (traced, copy.deepcopy(result))
            return result

    # Both callers (get_mutated_tensors and Inductor's UserDefinedTritonKernel)
    # look the function up on this module.
    triton_kernel_wrap.identify_accessed_tensors = cached
    try:
        yield counts
    finally:
        triton_kernel_wrap.identify_accessed_tensors = original
