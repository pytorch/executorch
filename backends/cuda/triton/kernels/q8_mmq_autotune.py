# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Autotune support used only by the large-row Q8 MMQ kernels."""
from __future__ import annotations

import concurrent.futures
import contextlib
import os
import threading
import weakref
from typing import Optional, Sequence

import torch
import triton
from executorch.backends.cuda.autotune.cuda_graph_timing import time_with_cuda_graph_us

try:
    from triton.runtime import _async_compile
except ImportError:
    _async_compile = None

_MAX_COMPILE_THREADS = 32


def _tuning_key(kernel, args: Sequence, kwargs: dict) -> tuple:
    # Match Triton Autotuner.run so a concrete cached pick can be reused.
    named = {**dict(zip(kernel.arg_names, args)), **kwargs}
    named = {k: v for k, v in named.items() if k in kernel.arg_names}
    key = [named[k] for k in kernel.keys if k in named]
    key += [str(v.dtype) for v in named.values() if hasattr(v, "dtype")]
    return tuple(key)


def _has_untuned_configs(kernel, args: Sequence, kwargs: dict) -> bool:
    """Whether ``kernel`` is about to benchmark configs it could compile ahead:
    an autotuner with several configs, a key it has not tuned, and tensors with
    real CUDA storage (tracing passes fake or functional tensors, which the
    launch itself handles)."""
    if not isinstance(kernel, triton.runtime.Autotuner) or len(kernel.configs) <= 1:
        return False
    for arg in (*args, *kwargs.values()):
        if isinstance(arg, torch.Tensor) and (
            type(arg) not in (torch.Tensor, torch.nn.Parameter)
            or arg.device.type != "cuda"
        ):
            return False
    return _tuning_key(kernel, args, kwargs) not in kernel.cache


_COMPILE_THREADS_LOCK = threading.Lock()
_compile_thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _compile_threads() -> concurrent.futures.ThreadPoolExecutor:
    """The compile threads, started on first use and shared by every shape."""
    global _compile_thread_pool
    with _COMPILE_THREADS_LOCK:
        if _compile_thread_pool is None:
            _compile_thread_pool = concurrent.futures.ThreadPoolExecutor(
                min(_MAX_COMPILE_THREADS, os.cpu_count() or 1),
                thread_name_prefix="quantized_gemm_compile",
            )
        return _compile_thread_pool


@contextlib.contextmanager
def _compile_pool():
    with _async_compile.AsyncCompileMode(_compile_threads(), ignore_errors=True):
        yield


def _precompile_untuned_configs(kernel, args: Sequence, kwargs: dict) -> None:
    """Compiles every config an autotuned kernel is about to benchmark, in parallel.

    triton.autotune compiles each config on its first benchmark call, one after
    another, and at compile time that is most of a shape's autotune. Compiling
    the pruned configs on a thread pool first leaves the benchmark, and so the
    pick, unchanged; it then times kernels that are already compiled. A config
    that fails to compile fails again in the benchmark, which skips it as before.
    """
    if _async_compile is None or _async_compile.active_mode.get() is not None:
        return
    if not _has_untuned_configs(kernel, args, kwargs):
        return
    with _compile_pool():
        kernel.warmup(*args, grid=(1,), **kwargs)


# kernel -> {config key: the kernel restricted to that one config}. One object
# per (kernel, config), so every call site of a tuned shape traces the same
# kernel and PyTorch's per-kernel caches keep working.
_SINGLE_CONFIG_KERNELS: weakref.WeakKeyDictionary[object, dict] = (
    weakref.WeakKeyDictionary()
)


def _tuned_kernel(kernel, args: Sequence, kwargs: dict):
    """``kernel`` restricted to the config its autotuner already picked for
    this call's key, or ``kernel`` itself if it has not tuned the key.

    Traced, an autotuned kernel hands Inductor every config, and AOTI then
    compiles and benchmarks all of them again for the same shapes; restricted
    to the pick, Inductor compiles that one config and benchmarks nothing.
    Called eagerly it launches the same config the autotuner would.
    """
    if not isinstance(kernel, triton.runtime.Autotuner) or len(kernel.configs) <= 1:
        return kernel
    try:
        config = kernel.cache.get(_tuning_key(kernel, args, kwargs))
    except TypeError:
        # A symbolic size in the key (a dynamic M the kernel tunes on): no
        # concrete pick to restrict to.
        return kernel
    if config is None or config.pre_hook is not None:
        return kernel
    config_key = tuple(sorted(config.all_kwargs().items()))
    variants = _SINGLE_CONFIG_KERNELS.setdefault(kernel, {})
    tuned = variants.get(config_key)
    if tuned is None:
        tuned = triton.autotune(configs=[config], key=[])(kernel.fn)
        variants[config_key] = tuned
    return tuned


def autotune_do_bench(kernel_call, quantiles):
    """Use the original experiment's CUDA graph timing and fallback windows."""
    if torch.cuda.current_stream().cuda_stream != 0:
        try:
            ms = time_with_cuda_graph_us(kernel_call) / 1000.0
        except RuntimeError:
            pass
        else:
            return [ms] * len(quantiles) if quantiles else ms
    return triton.testing.do_bench(kernel_call, warmup=5, rep=20, quantiles=quantiles)
