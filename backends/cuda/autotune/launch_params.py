# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch-time parameters picked by timing every candidate during AOTI compile.

Some parameters of a Triton op are fixed before ``triton.autotune`` sees a
config, because they decide what the launch function allocates or launches.
Split-K is one: it sizes the FP32 workspace and decides whether a reduce runs.
Instead of a rule (which would encode one GPU's balance of compute, bandwidth
and SM count), the launch function's author lists the candidates::

    @autotune_launch_param("split_k", (1, 2, 4, 8, 16))
    def _launch(bucket, x, qdata, ..., *, split_k):
        check_split_k(split_k, ...)        # raises InvalidLaunchParam if illegal
        ...

Inside ``autotune_launch_params()``, which the CUDA backend enters for its AOTI
compile, a call that does not pass ``split_k`` times every candidate on the
device and passes the fastest:

- the call's tensors are replaced by random tensors of the same shapes, dtypes
  and strides (export traces with fake tensors; the timing needs real ones).
  Their values carry no meaning, so a launch function whose cost depends on
  the values of an input (e.g. a KV length read from a tensor) cannot be timed
  this way yet;
- each candidate runs once (its Triton autotune picks a config), then all are
  timed with CUDA-graph replays over a few interleaved rounds (medians);
- a candidate the function rejects with ``InvalidLaunchParam`` is skipped. Any
  other exception is a bug in the launch function or its kernel and stops the
  compile; it is not taken as a rejection;
- within 2% of the fastest, the smallest candidate wins (for split-K: fewer
  partials, less memory, steadier picks across exports);
- the pick is cached per (decorated function, device, arguments' shapes,
  strides, dtypes and non-tensor values), so each shape is timed once;
- if the device runs out of memory, its cache is freed and the timing retried,
  then retried again with the context's ``offload`` tensors (e.g. the model's
  weights) parked on the CPU; if that fails too, the first candidate is used
  and a warning is logged.

The call then proceeds with the picked value, which export traces into the
graph as a constant. Outside the context (eager, ``torch.compile``, tests) the
first candidate is used, which must be legal for every input; nothing is timed.
A call that passes the parameter explicitly always uses it.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import inspect
import logging
import math
import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

import torch
from executorch.backends.cuda.autotune.cuda_graph_timing import (
    _parked_on_cpu,
    time_with_cuda_graph_us,
)
from torch._library.triton import set_wrap_triton_enabled
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.utils._python_dispatch import _disable_current_modes

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Candidates within this ratio of the fastest tie; the smallest of them wins.
_TIE_RATIO = 1.02
# Interleaved timing rounds per candidate; each candidate keeps its median.
_ROUNDS = 3


class InvalidLaunchParam(ValueError):
    """Raised by a launch function for a candidate that is illegal for its inputs."""


@dataclass
class LaunchParamPick:
    function: str
    param: str
    times_us: dict[Any, float]
    rejected: dict[Any, str]
    choice: Any


@dataclass
class LaunchParamStats:
    measured: int = 0
    cache_hits: int = 0
    # Timings that needed the offload tensors parked on the CPU.
    offloads: int = 0
    # Calls that could not be timed for lack of device memory; they used the
    # first candidate.
    out_of_memory: int = 0
    picks: list[LaunchParamPick] = field(default_factory=list)


class _State(threading.local):
    def __init__(self) -> None:
        self.stats: LaunchParamStats | None = None
        self.offload: Callable[[], Iterable[torch.Tensor]] | None = None


_STATE = _State()
_CACHE: dict[tuple, Any] = {}
# Held while timing, so concurrent compiles do not time against each other
# (re-entrant: a decorated function may call another one).
_TIMING_LOCK = threading.RLock()


@contextlib.contextmanager
def autotune_launch_params(
    offload: Callable[[], Iterable[torch.Tensor]] | None = None,
) -> Iterator[LaunchParamStats]:
    """Times ``@autotune_launch_param`` candidates on this thread.

    ``offload`` returns the CUDA tensors that may be parked on the CPU when the
    timing runs out of memory.
    """
    previous = (_STATE.stats, _STATE.offload)
    _STATE.stats = stats = LaunchParamStats()
    _STATE.offload = offload
    try:
        yield stats
    finally:
        _STATE.stats, _STATE.offload = previous


def clear_launch_param_cache() -> None:
    with _TIMING_LOCK:
        _CACHE.clear()


def _size_hint(size: Any) -> int:
    if isinstance(size, int):
        return size
    node = size.node
    if node.hint is not None:
        return int(node.hint)
    return int(node.shape_env.size_hint(node.expr))


def _argument_key(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            tuple(_size_hint(s) for s in value.shape),
            tuple(_size_hint(s) for s in value.stride()),
            str(value.dtype),
        )
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    return repr(value)


def _first_device(values: Iterable[Any]) -> torch.device:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    raise ValueError("autotune_launch_param needs a tensor argument to time on")


def _random_like(value: Any) -> Any:
    """A random tensor with ``value``'s shape, strides, dtype and device.

    Filled through a flat buffer, so overlapping layouts (e.g. a stride-0
    expand) work too.
    """
    if not isinstance(value, torch.Tensor):
        return value
    shape = [_size_hint(s) for s in value.shape]
    stride = [_size_hint(s) for s in value.stride()]
    numel = (
        0
        if math.prod(shape) == 0
        else 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    )
    flat = torch.empty(numel, dtype=value.dtype, device=value.device)
    if value.dtype.is_floating_point:
        flat.normal_(0.0, 0.5)
    elif value.dtype == torch.bool:
        flat.zero_()
    else:
        info = torch.iinfo(value.dtype)
        flat.random_(max(info.min, -128), min(info.max, 127) + 1)
    return flat.as_strided(shape, stride)


def _time_candidates(
    fn: Callable[..., Any],
    param: str,
    candidates: tuple[Any, ...],
    args: tuple,
    kwargs: dict,
) -> tuple[dict[Any, float], dict[Any, str]]:
    # Leave export's fake / functional / proxy modes, and launch the Triton
    # kernels directly, as triton_op's eager path does: through wrap_triton's
    # higher-order op every call builds a new Triton autotuner, so each call
    # would re-tune (and fail inside a CUDA-graph capture).
    with unset_fake_temporarily(), _disable_current_modes(), set_wrap_triton_enabled(
        False
    ):
        device = _first_device((*args, *kwargs.values()))
        stream = torch.cuda.Stream(device)
        # The inputs are made on the stream that times them.
        stream.wait_stream(torch.cuda.current_stream(device))
        runs: dict[Any, Callable[[], Any]] = {}
        rejected: dict[Any, str] = {}
        with torch.cuda.device(device), torch.cuda.stream(stream):
            real_args = tuple(_random_like(a) for a in args)
            real_kwargs = {k: _random_like(v) for k, v in kwargs.items()}
            for value in candidates:

                def run(value: Any = value) -> Any:
                    return fn(*real_args, **real_kwargs, **{param: value})

                try:
                    run()
                except InvalidLaunchParam as error:
                    rejected[value] = str(error)
                    continue
                runs[value] = run
            samples: dict[Any, list[float]] = {value: [] for value in runs}
            for _ in range(_ROUNDS):
                for value, run in runs.items():
                    samples[value].append(time_with_cuda_graph_us(run))
        stream.synchronize()
    return {value: statistics.median(ts) for value, ts in samples.items()}, rejected


def _free_device_memory() -> None:
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _time_candidates_within_memory(
    fn: Callable[..., Any],
    name: str,
    param: str,
    candidates: tuple[Any, ...],
    args: tuple,
    kwargs: dict,
    stats: LaunchParamStats,
) -> tuple[dict[Any, float] | None, dict[Any, str]]:
    """``_time_candidates``, retried after freeing the device's cache and then
    with the offload tensors parked on the CPU. Returns no times when the device
    still runs out of memory."""
    try:
        return _time_candidates(fn, param, candidates, args, kwargs)
    except torch.OutOfMemoryError:
        _free_device_memory()
    try:
        return _time_candidates(fn, param, candidates, args, kwargs)
    except torch.OutOfMemoryError:
        _free_device_memory()
    if _STATE.offload is not None:
        try:
            with _parked_on_cpu(_STATE.offload()):
                stats.offloads += 1
                return _time_candidates(fn, param, candidates, args, kwargs)
        except torch.OutOfMemoryError:
            _free_device_memory()
    logger.warning(
        "%s: out of device memory while timing %s candidates; using %s=%s",
        name,
        param,
        param,
        candidates[0],
    )
    return None, {}


def autotune_launch_param(param: str, candidates: Sequence[Any]) -> Callable[[F], F]:
    """Fills keyword argument ``param`` of the decorated launch function with
    the fastest of ``candidates`` during AOTI compile (see the module doc)."""
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("autotune_launch_param needs at least one candidate")

    def decorate(fn: F) -> F:
        name = f"{fn.__module__}.{fn.__qualname__}"
        if param not in inspect.signature(fn).parameters:
            raise TypeError(f"{name} has no parameter {param!r}")
        # Identifies this decoration in the cache, whatever the function's name.
        token = object()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            stats = _STATE.stats
            if param in kwargs or stats is None or len(candidates) == 1:
                kwargs.setdefault(param, candidates[0])
                return fn(*args, **kwargs)
            device = _first_device((*args, *kwargs.values()))
            key = (
                token,
                torch.cuda.get_device_name(device),
                tuple(_argument_key(a) for a in args),
                tuple(sorted((k, _argument_key(v)) for k, v in kwargs.items())),
            )
            # Only the cache lookup and the timing hold the lock; the call that
            # follows (export-side tracing) runs without it.
            with _TIMING_LOCK:
                choice = _CACHE.get(key, _MISSING)
                if choice is not _MISSING:
                    stats.cache_hits += 1
                    times = None
                else:
                    times, rejected = _time_candidates_within_memory(
                        fn, name, param, candidates, args, kwargs, stats
                    )
                    if times is None:
                        stats.out_of_memory += 1
                        choice = candidates[0]
                    elif not times:
                        raise ValueError(
                            f"{name}: no legal {param} among {candidates}: {rejected}"
                        )
                    else:
                        fastest = min(times.values())
                        choice = min(
                            v for v, t in times.items() if t <= fastest * _TIE_RATIO
                        )
                        _CACHE[key] = choice
            if times:
                stats.measured += 1
                stats.picks.append(
                    LaunchParamPick(name, param, times, rejected, choice)
                )
                logger.info(
                    "%s: %s=%s (%s)",
                    name,
                    param,
                    choice,
                    ", ".join(f"{v}: {t:.1f} us" for v, t in times.items()),
                )
            return fn(*args, **kwargs, **{param: choice})

        return wrapper  # type: ignore[return-value]

    return decorate


_MISSING = object()

__all__ = [
    "InvalidLaunchParam",
    "LaunchParamPick",
    "LaunchParamStats",
    "autotune_launch_param",
    "autotune_launch_params",
    "clear_launch_param_cache",
]
