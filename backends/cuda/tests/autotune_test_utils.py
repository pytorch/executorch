# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers to check AOTInductor's autotune picks while the CPU is saturated."""

import contextlib
import multiprocessing
import os
import signal
import statistics
import time
from typing import Iterator, NamedTuple

from executorch.backends.cuda.autotune import cuda_graph_timing as cgt
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.select_algorithm import AlgorithmSelectorCache


class Pick(NamedTuple):
    kind: str  # "inductor", "template" (matmul choice) or "custom" (our triton.autotune)
    name: str
    candidates: int
    regret: float  # reference time of the pick / reference time of the best candidate


def _spin(stop_at: float) -> None:
    while time.time() < stop_at:
        pass


class SaturatedCpu:
    """One busy-loop process per core; can be paused to measure a reference."""

    def __init__(self) -> None:
        stop_at = time.time() + 3600
        self.workers = [
            multiprocessing.Process(target=_spin, args=(stop_at,), daemon=True)
            for _ in range(os.cpu_count() or 1)
        ]

    def __enter__(self) -> "SaturatedCpu":
        for w in self.workers:
            w.start()
        return self

    def __exit__(self, *exc) -> None:
        for w in self.workers:
            with contextlib.suppress(ProcessLookupError):
                os.kill(w.pid, signal.SIGCONT)
            w.terminate()
        for w in self.workers:
            w.join(timeout=10)

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        for w in self.workers:
            os.kill(w.pid, signal.SIGSTOP)
        try:
            yield
        finally:
            for w in self.workers:
                os.kill(w.pid, signal.SIGCONT)


@contextlib.contextmanager
def record_autotune_picks(cpu: SaturatedCpu, picks: list) -> Iterator[None]:
    """For each autotune decision taken inside the context, re-time every
    candidate with the CPU paused (CUDA-graph timing, best of two) and append
    a ``Pick`` with what the pick costs against the best candidate."""
    orig_kernel = CachingAutotuner.benchmark_all_configs
    orig_choices = AlgorithmSelectorCache.benchmark_choices

    def record(kind, name, timings, reference):
        valid = {c: t for c, t in reference.items() if t != float("inf")}
        used = {c: t for c, t in timings.items() if c in valid and t != float("inf")}
        if len(valid) < 2 or not used:
            return
        pick = min(used, key=used.get)
        picks.append(Pick(kind, str(name), len(valid), valid[pick] / min(valid.values())))

    def kernel_hook(self, *args, **kwargs):
        timings = orig_kernel(self, *args, **kwargs)
        with cpu.paused(), cgt.cuda_graph_timing():
            reference = {
                l: min(self.bench(l, *args, **kwargs) for _ in range(2)) for l in timings
            }
        self.reset_to_zero_args(*args, **kwargs)
        kind = "custom" if self.custom_kernel else "inductor"
        record(kind, self.inductor_meta.get("kernel_name"), timings, reference)
        return timings

    def choices_hook(cls, choices, autotune_args, *args, **kwargs):
        timings = orig_choices.__func__(cls, choices, autotune_args, *args, **kwargs)
        with cpu.paused(), cgt.cuda_graph_timing():
            reference = {
                c: min(cls.benchmark_choice(c, autotune_args) for _ in range(2))
                for c in timings
            }
        record("template", type(next(iter(timings))).__name__, timings, reference)
        return timings

    CachingAutotuner.benchmark_all_configs = kernel_hook
    AlgorithmSelectorCache.benchmark_choices = classmethod(choices_hook)
    try:
        yield
    finally:
        CachingAutotuner.benchmark_all_configs = orig_kernel
        AlgorithmSelectorCache.benchmark_choices = orig_choices


def summarize(picks: list) -> str:
    if not picks:
        return "no autotune decisions recorded"
    regrets = [p.regret for p in picks]
    kinds = {k: sum(1 for p in picks if p.kind == k) for k in ("inductor", "template", "custom")}
    return (
        f"{len(picks)} decisions {kinds}: mean regret {statistics.mean(regrets):.3f}x, "
        f"worst {max(regrets):.3f}x, >1.10x: {sum(r > 1.10 for r in regrets)}"
    )
