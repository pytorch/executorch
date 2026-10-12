# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA-graph timing for AOTInductor's compile-time autotuning.

Inductor picks the launch config of every Triton kernel it generates, the
implementation of every matmul (its Triton templates), and the config of every
user ``triton.autotune`` kernel (our custom ops) by timing the candidates
through ``torch._inductor.runtime.benchmarking.benchmarker``, which dispatches
on the device type through a registry. The default CUDA timing brackets each
call of a candidate with a pair of CUDA events. During an export the CPU is
saturated (Inductor compiles kernels in parallel worker processes), and we
measured those readings becoming bimodal there: a few-microsecond kernel reads
either its own time or tens of microseconds more, so candidates become
indistinguishable and the autotuner picks configs up to several times slower
than the best one.

``cuda_graph_timing()`` registers a CUDA timing that captures several
calls of the candidate into one CUDA graph, each preceded by an L2 flush (as
Inductor's default timing does) and bracketed by timing events recorded inside
the graph. A replay is a single submission, so the GPU runs the flushes, calls
and events back to back with no host involvement, and a busy CPU cannot add to
the measured time.

The candidates Inductor hands over launch on the raw stream that was current
when they were built, so the context also makes a non-default stream current
and captures on that same stream (the legacy default stream cannot be
captured).

To keep captures succeeding:
- each capture's graph and its memory pool are released right after timing,
  and the allocator's cache is emptied once it holds a lot of unused memory;
- on CUDA OOM the cache is emptied and the capture retried; if that fails too,
  the tensors given as ``offload`` (the model's weights moved to the GPU for
  the compile) are parked on the CPU while the candidate is timed and restored
  afterwards. This is safe because AOTI autotunes on random inputs, never on
  the model's tensors;
- a candidate that cannot be captured at all is timed with Inductor's default.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import torch
from torch._inductor.runtime import benchmarking

logger: logging.Logger = logging.getLogger(__name__)

_TARGET_GRAPH_US = 2000.0
_MIN_CALLS = 4
_MAX_CALLS = 32
_EMPTY_CACHE_THRESHOLD_BYTES = 1 << 30


@dataclass
class CudaGraphTimingStats:
    captured: int = 0
    fallbacks: int = 0
    oom_retries: int = 0
    offloads: int = 0


class _State(threading.local):
    def __init__(self) -> None:
        self.stats: CudaGraphTimingStats | None = None
        self.offload: Callable[[], Iterable[torch.Tensor]] | None = None


_STATE = _State()
_INSTALL_LOCK = threading.Lock()
_installed_previous: Callable[..., Any] | None = None
_install_count = 0


def _default_cuda_bench(
    bench: benchmarking.Benchmarker,
    fn: Callable[[], Any],
    *,
    warmup: int,
    rep: int,
    **kwargs: Any,
) -> Any:
    kwargs.setdefault("device_type", "cuda")
    return bench.benchmark_gpu(fn, warmup=warmup, rep=rep, **kwargs)


def _release_cached_memory_if_large() -> None:
    if (
        torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        > _EMPTY_CACHE_THRESHOLD_BYTES
    ):
        torch.cuda.empty_cache()


_l2_flush_buffers: dict[int, torch.Tensor] = {}


def _l2_flush_buffer() -> torch.Tensor:
    device = torch.cuda.current_device()
    if device not in _l2_flush_buffers:
        size = torch.cuda.get_device_properties(device).L2_cache_size
        _l2_flush_buffers[device] = torch.empty(
            max(size, 1) // 4, dtype=torch.int32, device="cuda"
        )
    return _l2_flush_buffers[device]


def time_with_cuda_graph_us(fn: Callable[[], Any]) -> float:
    """Device time of one ``fn()`` call with a cold L2, in microseconds.

    ``fn`` must launch its work on the current stream, which must not be the
    legacy default stream.
    """
    stream = torch.cuda.current_stream()
    if stream.cuda_stream == 0:
        raise RuntimeError("cannot capture on the legacy default CUDA stream")
    flush = _l2_flush_buffer()
    fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    # Only sizes the graph: a busy CPU can inflate it, which just means fewer calls.
    estimate_us = max(start.elapsed_time(end) * 1000.0, 1.0)
    calls = min(max(math.ceil(_TARGET_GRAPH_US / estimate_us), _MIN_CALLS), _MAX_CALLS)
    events = [
        (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for _ in range(calls)
    ]
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            for call_start, call_end in events:
                flush.zero_()
                call_start.record()
                fn()
                call_end.record()
        graph.replay()
        events[-1][1].synchronize()
        return min(s.elapsed_time(e) for s, e in events) * 1000.0
    finally:
        # A failed capture leaves nothing to release; that must not mask its error.
        with contextlib.suppress(RuntimeError):
            graph.reset()
        del graph
        _release_cached_memory_if_large()


class ParkedTensorsNotRestored(RuntimeError):
    """Model tensors parked on the CPU could not be moved back to the GPU.

    Never treated as a timing failure: compiling on would package empty weights.
    """


def _restore_storage(storage: torch.UntypedStorage, host: torch.UntypedStorage) -> None:
    try:
        storage.resize_(host.nbytes())
    except torch.OutOfMemoryError:
        _cleanup_after_oom()
        storage.resize_(host.nbytes())
    storage.copy_(host)


@contextlib.contextmanager
def _parked_on_cpu(tensors: Iterable[torch.Tensor]) -> Iterator[int]:
    """Moves the CUDA storages behind ``tensors`` to host memory, then back.

    Works on storages so aliases (e.g. tied weights) stay aliased and every
    tensor object stays valid; only the device addresses change.
    """
    parked: list[tuple[torch.UntypedStorage, torch.UntypedStorage]] = []
    seen: set[int] = set()
    try:
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                continue
            storage = tensor.untyped_storage()
            if storage.nbytes() == 0 or storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            host = storage.cpu()
            storage.resize_(0)
            parked.append((storage, host))
        torch.cuda.empty_cache()
        yield sum(host.nbytes() for _, host in parked)
    finally:
        failed = 0
        for storage, host in parked:
            try:
                _restore_storage(storage, host)
            except Exception:  # noqa: BLE001 - try every storage, then raise
                failed += 1
        torch.cuda.synchronize()
        if failed:
            raise ParkedTensorsNotRestored(
                f"{failed} of {len(parked)} model tensors parked on the CPU for "
                "autotune timing could not be moved back to the GPU"
            )


def _cleanup_after_oom() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _time_capturing_robustly(
    fn: Callable[[], Any], stats: CudaGraphTimingStats
) -> float:
    try:
        return time_with_cuda_graph_us(fn)
    except torch.OutOfMemoryError:
        stats.oom_retries += 1
        _cleanup_after_oom()
    try:
        return time_with_cuda_graph_us(fn)
    except torch.OutOfMemoryError:
        stats.oom_retries += 1
        _cleanup_after_oom()
        if _STATE.offload is None:
            raise
    with _parked_on_cpu(_STATE.offload()) as parked_bytes:
        stats.offloads += 1
        logger.info(
            "CUDA-graph autotune timing: parked %.1f GiB of model tensors on the CPU to capture a candidate",
            parked_bytes / (1 << 30),
        )
        return time_with_cuda_graph_us(fn)


def _time_candidate_with_cuda_graph(
    bench: benchmarking.Benchmarker,
    fn: Callable[[], Any],
    *,
    warmup: int,
    rep: int,
    **kwargs: Any,
) -> Any:
    previous = _installed_previous or _default_cuda_bench
    stats = _STATE.stats
    plain_result = "quantiles" in kwargs or kwargs.get("return_mode", "min") not in (
        "min",
        "mean",
        "median",
    )
    if stats is None or plain_result:
        return previous(bench, fn, warmup=warmup, rep=rep, **kwargs)
    if not kwargs.get("is_vetted_benchmarking", False):
        benchmarking.may_ban_benchmarking()
    try:
        # Same lock as Inductor's own GPU timing (a no-op unless a harness set one).
        with benchmarking.maybe_gpu_benchmark_lock():
            us = _time_capturing_robustly(fn, stats)
    except ParkedTensorsNotRestored:
        raise
    except (
        Exception
    ) as error:  # noqa: BLE001 - any failure falls back to the default timing
        stats.fallbacks += 1
        logger.debug(
            "CUDA-graph autotune timing fell back to the default timing: %s", error
        )
        _cleanup_after_oom()
        return previous(bench, fn, warmup=warmup, rep=rep, **kwargs)
    stats.captured += 1
    return us / 1000.0


def _install() -> None:
    global _installed_previous, _install_count
    with _INSTALL_LOCK:
        if _install_count == 0:
            _installed_previous = benchmarking._BENCHMARK_DISPATCH.get("cuda")
            benchmarking.register_benchmarker(
                "cuda", _time_candidate_with_cuda_graph, override=True
            )
        _install_count += 1


def _uninstall() -> None:
    global _installed_previous, _install_count
    with _INSTALL_LOCK:
        _install_count -= 1
        if _install_count == 0:
            benchmarking.register_benchmarker(
                "cuda", _installed_previous or _default_cuda_bench, override=True
            )
            _installed_previous = None


@contextlib.contextmanager
def cuda_graph_timing(
    offload: Callable[[], Iterable[torch.Tensor]] | None = None,
) -> Iterator[CudaGraphTimingStats]:
    """Times Inductor's autotune candidates on this thread with CUDA graphs.

    ``offload`` returns the CUDA tensors that may be parked on the CPU when a
    capture runs out of memory.
    """
    if not torch.cuda.is_available():
        yield CudaGraphTimingStats()
        return
    previous = (_STATE.stats, _STATE.offload)
    stats = CudaGraphTimingStats()
    outer_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    # Work queued on the outer stream (e.g. moving weights to the device) must
    # finish before this stream uses it, and vice versa on exit.
    capture_stream.wait_stream(outer_stream)
    _install()
    _STATE.stats = stats
    _STATE.offload = offload
    try:
        with torch.cuda.stream(capture_stream):
            yield stats
    finally:
        try:
            outer_stream.wait_stream(capture_stream)
        finally:
            _STATE.stats, _STATE.offload = previous
            _uninstall()
        if stats.captured or stats.fallbacks:
            logger.info(
                "CUDA-graph autotune timing: %d captured, %d fell back, %d OOM retries, %d offloads",
                stats.captured,
                stats.fallbacks,
                stats.oom_retries,
                stats.offloads,
            )
