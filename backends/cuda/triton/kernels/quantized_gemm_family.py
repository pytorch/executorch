# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bucketed quantized GEMM custom ops and shared launch infrastructure.

A quantized weight format (INT4, INT5, INT6, INT8, ...) describes its GEMM to
``QuantizedGemmFamily``: the op signature, a launcher, a fake implementation,
and one function that says why an input cannot be served. The family registers
one ``triton::<name>_m{M}`` op per supported row count ("bucket"); each is a
separate graph node, so AOTInductor compiles and tunes each bucket's kernel on
its own.

Legality lives with the format, in one place: ``supports`` (never raises) and
``validate`` (raises) both derive from the format's ``unsupported_reason``, and
every op validates its inputs before launching. Choosing a bucket for an
activation is the dispatcher's job (``quantize_op_dispatch``).

``launch_split_k_gemm`` is the launch skeleton the formats share: output and
split-K workspace allocation, the grid, and the reduce after the main kernel.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import importlib
import inspect
import logging
import os
import threading
import typing
import weakref
from typing import Callable, Optional, Sequence

import torch
import triton
from executorch.backends.cuda.triton.kernels.quantized_gemm_utils import splitk_reduce
from torch.library import triton_op, wrap_triton

try:
    from triton.runtime import _async_compile
except (
    ImportError
):  # A Triton without parallel compile compiles each config on first use.
    _async_compile = None

# Concurrent compiles of autotune configs. Each runs in a worker process: in
# one process Triton's MLIR and LLVM passes hold the GIL, so threads alone keep
# only a few compiles busy.
_MAX_COMPILE_THREADS = 32

logger: logging.Logger = logging.getLogger(__name__)


def _resolved_signature(prototype) -> inspect.Signature:
    """The prototype's signature with its annotations evaluated in the
    prototype's own module, so schema inference sees real types even when that
    module uses postponed (string) annotations."""
    hints = typing.get_type_hints(prototype)
    signature = inspect.signature(prototype)
    return signature.replace(
        parameters=[
            p.replace(annotation=hints.get(p.name, p.annotation))
            for p in signature.parameters.values()
        ],
        return_annotation=hints.get("return", signature.return_annotation),
    )


class QuantizedGemmFamily:
    """The per-M ops of one quantized GEMM format.

    ``prototype`` declares the op signature, activation first, e.g.
    ``def prototype(x: Tensor, qdata: Tensor, ..., group_size: int) -> Tensor``.
    ``launch``, ``fake`` and ``unsupported_reason`` take ``(bucket, *args)`` with
    the prototype's arguments. ``unsupported_reason`` returns None when the
    bucket's kernel can serve the arguments, else a short reason; it must not
    raise for any arguments matching the prototype.
    """

    def __init__(
        self,
        name: str,
        buckets: Sequence[int],
        prototype: Callable[..., torch.Tensor],
        launch: Callable[..., torch.Tensor],
        fake: Callable[..., torch.Tensor],
        unsupported_reason: Callable[..., Optional[str]],
    ) -> None:
        if (
            not buckets
            or any(b < 1 for b in buckets)
            or len(set(buckets)) != len(buckets)
        ):
            raise ValueError(
                f"buckets must be distinct positive row counts, got {buckets}"
            )
        self.name = name
        self.buckets: tuple[int, ...] = tuple(sorted(buckets))
        self._unsupported_reason = unsupported_reason
        self._ops = {
            bucket: self._register(bucket, prototype, launch, fake)
            for bucket in self.buckets
        }

    def _register(self, bucket, prototype, launch, fake):
        op_name = f"{self.name}_m{bucket}"
        signature = _resolved_signature(prototype)

        def impl(*args):
            self.validate(bucket, *args)
            return launch(bucket, *args)

        def impl_fake(*args):
            return fake(bucket, *args)

        for fn, name in ((impl, op_name), (impl_fake, f"_{op_name}_fake")):
            fn.__name__ = fn.__qualname__ = name
            fn.__signature__ = signature
        op = triton_op(f"triton::{op_name}", mutates_args={})(impl)
        op.register_fake(impl_fake)
        return op

    def op(self, bucket: int) -> Callable[..., torch.Tensor]:
        try:
            return self._ops[bucket]
        except KeyError as error:
            raise RuntimeError(
                f"unsupported {self.name} bucket {bucket}; expected one of {self.buckets}"
            ) from error

    def supports(self, bucket: int, *args) -> bool:
        """Whether ``op(bucket)`` can serve these arguments. Never raises."""
        return bucket in self._ops and self._unsupported_reason(bucket, *args) is None

    def validate(self, bucket: int, *args) -> None:
        """Raises RuntimeError with the reason ``op(bucket)`` cannot serve these arguments."""
        if bucket not in self._ops:
            raise RuntimeError(
                f"unsupported {self.name} bucket {bucket}; expected one of {self.buckets}"
            )
        reason = self._unsupported_reason(bucket, *args)
        if reason is not None:
            raise RuntimeError(f"{self.name}_m{bucket}: {reason}")


def _tuning_key(kernel, args: Sequence, kwargs: dict) -> tuple:
    # Mirrors the key triton.runtime.Autotuner.run caches its pick under (built
    # the same way in the Triton 3.5 to 3.9 in fbsource). Should Triton change
    # it, lookups miss and work is redone, never skipped;
    # test_the_tuning_key_is_the_autotuners_cache_key catches the change.
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


# Tuning keys whose configs a candidate prepass compiled. Weak kernel keys and
# one-shot values keep this bookkeeping bounded and prevent stale object-id hits.
_COMPILED_AHEAD: weakref.WeakKeyDictionary[object, set[tuple]] = (
    weakref.WeakKeyDictionary()
)


def _compile_workers() -> int:
    return min(_MAX_COMPILE_THREADS, os.cpu_count() or 1)


_PROCESS_POOL_LOCK = threading.Lock()
_process_pool = None
_process_pool_failed = False


def _compile_process_pool():
    """The worker processes, started on first use and kept for later shapes;
    None once they have failed. Inductor's ``SubprocPool``: a fresh Python
    process that forks the workers, so they neither re-run the caller's main
    module (as ``spawn`` would) nor inherit its CUDA state (as ``fork`` would)."""
    global _process_pool
    with _PROCESS_POOL_LOCK:
        if _process_pool is None and not _process_pool_failed:
            from torch._inductor.compile_worker.subproc_pool import (
                SubprocKind,
                SubprocPool,
            )

            _process_pool = SubprocPool(_compile_workers(), kind=SubprocKind.FORK)
        return _process_pool


def _stop_using_processes(error: BaseException) -> None:
    global _process_pool, _process_pool_failed
    with _PROCESS_POOL_LOCK:
        if _process_pool_failed:
            return
        _process_pool_failed = True
        pool, _process_pool = _process_pool, None
    logger.warning(
        "Compiling quantized GEMM configs in this process from now on: worker "
        "processes failed (%s)",
        error,
    )
    if pool is not None:
        pool.shutdown()


def _compile_in_process(
    cache_dir: str,
    module: str,
    qualname: str,
    source_module: str,
    source_qualname: str,
    signature: dict,
    constants: dict,
    attrs: dict,
    target,
    options: dict,
    env_vars: dict,
) -> None:
    """Compiles one config in a worker process into the shared Triton cache.
    Returns ``_UNIMPORTABLE`` if the worker cannot find the kernel or source
    class, which no other config would find either."""
    # Compiling needs no GPU; keep the worker from initializing one.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    triton.knobs.cache.dir = cache_dir
    try:
        fn = importlib.import_module(module)
        for part in qualname.split("."):
            fn = getattr(fn, part)
        while not isinstance(fn, triton.runtime.JITFunction):
            fn = fn.fn
        source = importlib.import_module(source_module)
        for part in source_qualname.split("."):
            source = getattr(source, part)
    except (ImportError, AttributeError):
        return _UNIMPORTABLE
    triton.compile(
        source(fn, signature, constants, attrs),
        target=target,
        options=options,
        _env_vars=env_vars,
    )
    return None


def _process_compile_request(compile_fn) -> Optional[tuple]:
    """The arguments of the ``triton.compile`` call that Triton's async-compile
    closure makes, so a worker process can make the same call. None when the
    closure or kernel is not the expected shape (it then compiles in-process)."""
    names = compile_fn.__code__.co_freevars
    if set(names) != {"env_vars", "options", "self", "src", "target"}:
        return None
    cells = dict(zip(names, (cell.cell_contents for cell in compile_fn.__closure__)))
    src = cells["src"]
    jit_fn = getattr(src, "fn", None)
    module = getattr(jit_fn, "module", None)
    qualname = getattr(getattr(jit_fn, "fn", None), "__qualname__", "")
    if module is None or not qualname or "<locals>" in qualname:
        return None
    return (
        triton.knobs.cache.dir,
        module,
        qualname,
        type(src).__module__,
        type(src).__qualname__,
        src.signature,
        src.constants,
        src.attrs,
        cells["target"],
        cells["options"].__dict__,
        cells["env_vars"],
    )


_UNIMPORTABLE = "unimportable"


def _workers_failed(result=None, error: Optional[BaseException] = None) -> bool:
    """Whether a worker's answer means the workers themselves are unusable, as
    opposed to one config failing to compile (which ``SubprocPool`` reports as
    a ``SubprocException``): they cannot import the kernel, or the pool closed
    or broke."""
    from torch._inductor.compile_worker.subproc_pool import SubprocException

    if error is not None:
        return not isinstance(error, SubprocException)
    return result == _UNIMPORTABLE


def _compiled_in_a_process_first(compile_fn):
    request = _process_compile_request(compile_fn)
    if request is None:
        return compile_fn

    def compile_then_load():
        pool = _compile_process_pool()
        if pool is not None:
            try:
                result = pool.submit(_compile_in_process, *request).result()
            except Exception as error:
                if _workers_failed(error=error):
                    _stop_using_processes(error)
                # Otherwise this config also fails to compile in-process, where
                # the benchmark skips it as before.
            else:
                if _workers_failed(result=result):
                    _stop_using_processes(
                        RuntimeError(f"workers cannot import {request[1]}")
                    )
        # Loads what the worker cached (same source, options and key), or
        # compiles here if it did not.
        return compile_fn()

    return compile_then_load


if _async_compile is not None:

    class _ProcessCompileMode(_async_compile.AsyncCompileMode):
        """Triton's async compile, with each compile done in a worker process."""

        def submit(self, key, compile_fn, finalize_fn):
            return super().submit(
                key, _compiled_in_a_process_first(compile_fn), finalize_fn
            )


_COMPILE_THREADS_LOCK = threading.Lock()
_compile_thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _compile_threads() -> concurrent.futures.ThreadPoolExecutor:
    """The compile threads, started on first use and shared by every shape."""
    global _compile_thread_pool
    with _COMPILE_THREADS_LOCK:
        if _compile_thread_pool is None:
            _compile_thread_pool = concurrent.futures.ThreadPoolExecutor(
                _compile_workers(),
                thread_name_prefix="quantized_gemm_compile",
            )
        return _compile_thread_pool


@contextlib.contextmanager
def _compile_pool():
    with _ProcessCompileMode(_compile_threads(), ignore_errors=True):
        yield


def _queue_untuned_configs(kernel, args: Sequence, kwargs: dict) -> bool:
    """Queues the configs ``kernel`` is about to benchmark on the active compile
    pool; False if there is no pool or nothing to compile."""
    if _async_compile is None or _async_compile.active_mode.get() is None:
        return False
    if not _has_untuned_configs(kernel, args, kwargs):
        return False
    key = _tuning_key(kernel, args, kwargs)
    try:
        kernel.warmup(*args, grid=(1,), **kwargs)
    except Exception as error:  # noqa: BLE001 - precompiling only saves time
        # The timed launch compiles this candidate's configs itself.
        logger.warning(
            "Not precompiling %s's configs: %s", kernel.base_fn.__name__, error
        )
        return False
    _COMPILED_AHEAD.setdefault(kernel, set()).add(key)
    return True


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
    key = _tuning_key(kernel, args, kwargs)
    compiled_keys = _COMPILED_AHEAD.get(kernel)
    if compiled_keys is not None and key in compiled_keys:
        compiled_keys.remove(key)
        if not compiled_keys:
            del _COMPILED_AHEAD[kernel]
        return
    try:
        with _compile_pool():
            kernel.warmup(*args, grid=(1,), **kwargs)
    except Exception as error:  # noqa: BLE001 - precompiling only saves time
        # The launch compiles and benchmarks the configs itself, as without
        # precompiling, and reports a real problem there.
        logger.warning(
            "Not precompiling %s's configs: %s", kernel.base_fn.__name__, error
        )


# Set while every @autotune_launch_param candidate makes one untimed call: the
# launches then only queue their configs on one shared compile pool.
_QUEUING_ONLY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "quantized_gemm_queuing_only", default=False
)


@contextlib.contextmanager
def _compile_every_candidate_at_once():
    """Compiles what all launch-parameter candidates (e.g. every split-K) will
    benchmark on one pool, rather than one candidate's configs at a time; leaving
    waits for all of them. Timing then runs as before on compiled kernels."""
    if _async_compile is None or _async_compile.active_mode.get() is not None:
        yield
        return
    with _compile_pool():
        token = _QUEUING_ONLY.set(True)
        try:
            yield
        finally:
            _QUEUING_ONLY.reset(token)


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


def launch_split_k_gemm(
    kernel,
    *,
    bucket: int,
    m,
    n: int,
    device: torch.device,
    split_k: int,
    block_m: int,
    inputs: Sequence,
    shape_args: Sequence,
    split_k_on_axis_1: bool = False,
    reduce_block_n: int = 64,
    **constexprs,
) -> torch.Tensor:
    """Runs a format's main kernel and returns the BF16 [m, n] output.

    The kernel is called as ``kernel(*inputs, out, *shape_args, stride_os,
    stride_om, stride_on, SPLIT_K=split_k, **constexprs)`` on a grid of
    ``(cdiv(n, BLOCK_N), cdiv(bucket, BLOCK_M), split_k)``, both blocks coming
    from its config or launch constexprs. Kernels without a ``BLOCK_M``
    constexpr retain a size-one M grid. Tile kernels may instead request
    ``(cdiv(n, BLOCK_N), split_k, cdiv(bucket, BLOCK_M))`` to keep split CTAs
    adjacent in launch order. With split-K it writes FP32 partials
    to a ``(split_k, bucket, n)`` workspace and reduces with fixed
    ``block_m`` x ``reduce_block_n`` tiles,
    reduced deterministically afterwards. The workspace is sized by the bucket,
    not the runtime M, so its strides (constexprs) stay static under a dynamic
    M; rows at or above M are never read. Without split-K the split stride is
    unused and the kernel writes ``out`` directly.
    """
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    if split_k > 1:
        partial = torch.empty((split_k, bucket, n), dtype=torch.float32, device=device)
        stride_os = partial.stride(0)
    else:
        partial = out
        stride_os = 0

    def grid(meta):
        n_tiles = triton.cdiv(n, meta["BLOCK_N"])
        m_tiles = triton.cdiv(bucket, meta["BLOCK_M"]) if "BLOCK_M" in meta else 1
        if split_k_on_axis_1:
            if m_tiles == 1:
                return (n_tiles, split_k) if split_k > 1 else (n_tiles,)
            if split_k > 1:
                return (n_tiles, split_k, m_tiles)
            return (n_tiles, m_tiles)
        return (n_tiles, m_tiles, split_k)

    args = (
        *inputs,
        partial,
        *shape_args,
        stride_os,
        partial.stride(-2),
        partial.stride(-1),
    )
    kwargs = {"SPLIT_K": split_k, **constexprs}
    if _QUEUING_ONLY.get():
        if _async_compile is None or _async_compile.active_mode.get() is None:
            raise RuntimeError("candidate prepass lost its active compile pool")
        _queue_untuned_configs(kernel, args, kwargs)
        return out
    _precompile_untuned_configs(kernel, args, kwargs)
    wrap_triton(_tuned_kernel(kernel, args, kwargs))[grid](*args, **kwargs)
    if split_k > 1:
        splitk_reduce(partial, out, block_m, reduce_block_n)
    return out


__all__ = ["QuantizedGemmFamily", "launch_split_k_gemm"]
