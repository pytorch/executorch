# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Picks, among the implementations that can run a call, the one that is
fastest on the export GPU.

The replacement pass records the plan of every planned call that several
implementations can run (see ``routing.py``), the regular lowering
(Inductor) and the family's replacement, for a family that opts in. Two
passes follow it:

- ``MeasureChoicesPass`` measures each such call in a worker process, so
  nothing the exporting process has loaded can interfere: inside the
  backend's AOTI compile context, on inputs made from the nodes' fake values,
  it runs the replacement once (its compile-time autotune picks its own
  schedule and config) and compiles the original op with Inductor's C++
  wrapper under the backend's AOTI options, after the backend's
  decompositions, while ``collect_unsupported_fallback_kernels`` records the
  kernels the ExecuTorch CUDA runtime does not provide. A candidate that
  cannot run drops out; if both can, they are timed: a few CUDA-graph
  timings first, then, unless those settle it beyond the GPU's run-to-run
  variation, interleaved samples of both.
  A replacement is chosen only if it is faster by the margin and its slow
  samples (90th percentile) still beat the regular lowering's fast ones
  (10th).
- ``ApplyChoicesPass`` replaces each call whose choice is a replacement and
  writes one JSON log per export.

The worker compiles the regular lowering with the C++ wrapper. A process
that cannot load such a module (a build that links torch into its executable
without exporting torch's C shims) measures nothing: every call takes its
static choice, with one warning per export.

With two or more calls to measure, the worker compiles them in parallel
processes, largest first (``MeasuredRoutingConfig.workers``, default 8; 0 to
skip), so that the measurement, one call at a time, finds their kernels
compiled. It starts them at once, while it checks that it can measure, and
stops them if it cannot. Those processes decide nothing: they publish no
result and leave no autotune timings.

Results are cached on disk by both calls' metadata, the precision settings,
the GPU, driver and versions, and the backend's sources. Only outcomes that
are properties of the call are cached (timings, kernels the runtime lacks,
Inductor finding no valid choice), not failures that may not recur. A call
whose inputs cannot be described (symbolic sizes, a storage offset, an input
passed twice), or that the worker does not report, takes its plan's static
choice.

Each call is compared on its own: fusion with neighbouring ops is not
measured.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import gc
import hashlib
import importlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import torch
import torch.utils._pytree as pytree
from executorch.backends.cuda.autotune.cuda_graph_timing import (
    cuda_graph_call_times_us,
    time_with_cuda_graph_us,
)
from executorch.backends.cuda.triton.routing import (
    REGULAR_LOWERING,
    replace,
    ReplacementError,
    ReplacementPlan,
)
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg
from torch.fx.passes.infra.pass_base import PassBase, PassResult

logger: logging.Logger = logging.getLogger(__name__)

META_KEY = "measured_routing"
SCHEMA_VERSION = 1
_FAST_TIMINGS = 5
_INTERLEAVED_ROUNDS = 10
# The run-to-run movement of one kernel's time on a GPU (clock states)
# reaches 20-25%; the fast timings decide alone only beyond it.
_NOISE_BAND = 1.25
# A worker that reports no result and writes no output for this long is hung;
# a slow one that keeps making progress (measuring every call of a large model)
# runs to the end.
_WORKER_IDLE_TIMEOUT_S = 1800
_WORKER_POLL_S = 5.0
# The devices this process measured on: the worker could load modules
# compiled with Inductor's C++ wrapper there.
_capable_devices: set[str] = set()
_PRECISION_KEYS = [("generic", "all")] + [
    (backend, op)
    for backend in ("cuda", "mkldnn")
    for op in ("all", "matmul", "conv", "rnn")
]


@dataclass(frozen=True)
class MeasuredRoutingConfig:
    margin: float = 0.15
    cache_dir: Optional[str] = None
    # Processes that compile the uncached calls in parallel before they are
    # measured; 0 compiles each as it is measured.
    workers: int = 8

    def directory(self) -> Path:
        if self.cache_dir is not None:
            return Path(self.cache_dir)
        from torch._inductor.runtime.cache_dir_utils import cache_dir

        return Path(cache_dir()) / "executorch_measured_routing"


class Unmeasurable(Exception):
    """The call's inputs cannot be described for a measurement."""


def record_plan(node: Node, plan: ReplacementPlan) -> None:
    node.meta[META_KEY] = {"plan": plan, "name": node.name}


def _encode(value: Any, seen: set[Node]) -> Any:
    """``value`` as JSON, each tensor as its dtype, size and strides."""
    if isinstance(value, Node):
        if value in seen:
            raise Unmeasurable(f"{value.name} is passed more than once")
        seen.add(value)
        value = value.meta.get("val")
    if isinstance(value, torch.Tensor):
        dims = [*value.shape, *value.stride(), value.storage_offset()]
        if not all(type(d) is int for d in dims):
            raise Unmeasurable(f"symbolic size, stride or offset {dims}")
        if value.storage_offset() != 0:
            raise Unmeasurable(f"storage offset {value.storage_offset()}")
        dtype = str(value.dtype).removeprefix("torch.")
        return {"tensor": [dtype, list(value.shape), list(value.stride())]}
    if isinstance(value, (list, tuple)):
        return [_encode(v, seen) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise Unmeasurable(f"cannot describe {value!r}")


def _materialize(value: Any, device: torch.device) -> Any:
    """``value`` with each encoded tensor made the way AOTI makes its
    autotune inputs."""
    from torch._dynamo.testing import rand_strided

    if isinstance(value, dict):
        dtype, size, stride = value["tensor"]
        return rand_strided(size, stride, getattr(torch, dtype), device)
    if isinstance(value, list):
        return [_materialize(v, device) for v in value]
    return value


def _op_name(op: Any) -> str:
    """The name of an op overload; an edge op is named by its aten op."""
    return f"{op._schema.name}.{op._schema.overload_name or 'default'}"


def _op(name: str) -> Any:
    qualified, overload = name.rsplit(".", 1)
    namespace, op = qualified.split("::")
    return getattr(getattr(getattr(torch.ops, namespace), op), overload)


def _spec(node: Node, plan: ReplacementPlan) -> dict[str, Any]:
    """Both implementations of a recorded call, as JSON."""
    seen: set[Node] = set()
    args = _encode(node.args, seen)
    kwargs = {k: _encode(v, seen) for k, v in node.kwargs.items()}
    (replacement,) = [c for c in plan.candidates if c.name != REGULAR_LOWERING]
    # The replacement's own inputs: one may alias another the same way.
    replacement_seen: set[Node] = set()
    replacement_args = _encode(replacement.args, replacement_seen)
    replacement_kwargs = {
        k: _encode(v, replacement_seen) for k, v in replacement.kwargs.items()
    }
    return {
        "original": _op_name(node.target),
        "args": args,
        "kwargs": kwargs,
        "output": _encode(node.meta.get("val"), set()),
        "replacement": {
            "name": replacement.name,
            "op": _op_name(replacement.kernel),
            "args": replacement_args,
            **({"kwargs": replacement_kwargs} if replacement_kwargs else {}),
        },
    }


def source_files() -> list[Path]:
    """The backend's sources a measurement depends on: this module, the
    kernels, their autotune and the options AOTI compiles with."""
    root = Path(__file__).resolve().parents[1]
    return sorted(
        path
        for path in root.rglob("*.py")
        if not {"tests", "benchmarks"} & set(path.relative_to(root).parts)
    )


_sources_digest: Optional[str] = None


def _sources_hash() -> str:
    global _sources_digest
    if _sources_digest is None:
        root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for path in source_files():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        _sources_digest = digest.hexdigest()
    return _sources_digest


def _driver_version() -> int:
    driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    version = ctypes.c_int()
    driver.cuDriverGetVersion(ctypes.byref(version))
    return version.value


def _precision_state() -> list[list[str]]:
    return [
        [b, o, torch._C._get_fp32_precision_getter(b, o)] for b, o in _PRECISION_KEYS
    ]


def cache_key(spec: dict[str, Any], options: dict[str, Any], device) -> str:
    import triton

    properties = torch.cuda.get_device_properties(device)
    key = {
        "schema_version": SCHEMA_VERSION,
        **spec,
        "precision": _precision_state(),
        "gpu": [
            properties.name,
            properties.major,
            properties.minor,
            properties.total_memory,
            properties.multi_processor_count,
        ],
        "cuda": [torch.version.cuda, _driver_version()],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "inductor_options": json.dumps(options, sort_keys=True, default=str),
        "sources": _sources_hash(),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _valid_samples(stage: Any) -> bool:
    return isinstance(stage, dict) and all(
        isinstance(stage.get(side), list)
        and len(stage[side]) >= 2
        and all(
            type(t) in (int, float) and math.isfinite(t) and t > 0 for t in stage[side]
        )
        for side in ("ours", "inductor")
    )


def _read_entry(path: Path, plan: ReplacementPlan) -> Optional[dict[str, Any]]:
    """A cached measurement of ``plan``'s candidates, or None for a missing,
    unreadable, foreign or inconsistent entry."""
    try:
        entry = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("schema_version") != SCHEMA_VERSION:
        return None
    candidates = entry.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {
        c.name for c in plan.candidates
    }:
        return None
    for status in candidates.values():
        if not isinstance(status, dict):
            return None
        # Only failures that are properties of the call are cached.
        if status.get("status") != "available" and (
            status.get("status") != "unavailable"
            or not isinstance(status.get("reason"), str)
            or status.get("cacheable") is not True
        ):
            return None
    timings = entry.get("timings")
    if not all(c["status"] == "available" for c in candidates.values()):
        return entry if timings is None else None
    if not isinstance(timings, dict) or not _valid_samples(timings.get("fast")):
        return None
    interleaved = timings.get("interleaved")
    if interleaved is not None and not _valid_samples(interleaved):
        return None
    return entry


def _publish(path: Path, content: dict[str, Any]) -> None:
    """Writes ``path`` atomically: concurrent exports may each publish a
    complete file, and the last one stays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(content, f, indent=1, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


def _summary(samples: list[float]) -> dict[str, float]:
    deciles = statistics.quantiles(samples, n=10)
    return {
        "median_us": statistics.median(samples),
        "p10_us": deciles[0],
        "p90_us": deciles[-1],
    }


def verdict(timings: dict[str, Any], margin: float) -> Optional[tuple[bool, str]]:
    """(keep the replacement, outcome) from its timings and the regular
    lowering's at ``margin``, or None if that needs interleaved samples the
    timings do not have."""
    threshold = 1 + margin
    fast = timings["fast"]
    ratio = statistics.median(fast["inductor"]) / statistics.median(fast["ours"])
    if ratio >= threshold * _NOISE_BAND and max(fast["ours"]) < min(fast["inductor"]):
        return True, "faster"
    if ratio * _NOISE_BAND < threshold:
        return False, "slower"
    if timings.get("interleaved") is None:
        return None
    ours = _summary(timings["interleaved"]["ours"])
    theirs = _summary(timings["interleaved"]["inductor"])
    if theirs["median_us"] < ours["median_us"] * threshold:
        return False, "slower"
    if ours["p90_us"] < theirs["p10_us"]:
        return True, "faster"
    return False, "inconclusive"


# The measurement, which runs in the worker process.


@contextlib.contextmanager
def _timing() -> Iterator[None]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # A collection during a capture could unload a Triton module, which a
    # capturing stream does not permit.
    gc.collect()
    gc.disable()
    try:
        with torch.cuda.stream(stream):
            yield
    finally:
        gc.enable()
        torch.cuda.current_stream().wait_stream(stream)


def _compile_replacement(
    replacement: dict[str, Any], device: torch.device
) -> Callable[[], Any]:
    """A replacement, run once so its compile-time autotune (active in the
    backend's AOTI context) picks its schedule and config."""
    op = _op(replacement["op"])
    args = _materialize(replacement["args"], device)
    kwargs = {
        k: _materialize(v, device) for k, v in replacement.get("kwargs", {}).items()
    }
    op(*args, **kwargs)
    torch.cuda.synchronize(device)
    return lambda: op(*args, **kwargs)


def _compile_inductor(
    backend: Any, options: dict[str, Any], spec: dict[str, Any], device: torch.device
) -> tuple[Callable[[], Any], list[str]]:
    """Inductor's lowering of the original op, decomposed the way the backend
    decomposes it before Inductor sees it, compiled with the C++ wrapper; and
    the kernels it calls that the runtime does not provide."""
    from torch.fx.experimental.proxy_tensor import make_fx

    op = _op(spec["original"])
    op = backend.get_decomposition_table().get(op, op)
    args = _materialize(spec["args"], device)
    kwargs = {k: _materialize(v, device) for k, v in spec["kwargs"].items()}
    positions = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
    names = [k for k, v in kwargs.items() if isinstance(v, torch.Tensor)]
    tensors = [args[i] for i in positions] + [kwargs[k] for k in names]

    def call(*inputs):
        call_args, call_kwargs = list(args), dict(kwargs)
        for i, t in zip(positions, inputs):
            call_args[i] = t
        for k, t in zip(names, inputs[len(positions) :]):
            call_kwargs[k] = t
        return op(*call_args, **call_kwargs)

    # Traced first, so Dynamo sees only the aten ops AOTI compiles, not the
    # backend's decomposition functions.
    graph = make_fx(call, tracing_mode="fake")(*tensors)
    missing: set[str] = set()
    torch._dynamo.reset()
    with backend.collect_unsupported_fallback_kernels(
        missing
    ), torch._inductor.config.patch(options):
        compiled = torch.compile(graph, fullgraph=True, dynamic=False)
        compiled(*tensors)
    torch.cuda.synchronize(device)
    return (lambda: compiled(*tensors)), sorted(missing)


def _no_valid_choices(error: Optional[BaseException]) -> bool:
    from torch._inductor.select_algorithm import NoValidChoicesError

    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, NoValidChoicesError):
            return True
        seen.add(id(error))
        error = error.__cause__ or getattr(error, "inner_exception", None)
    return False


def _unavailable(phase: str, error: BaseException, cacheable: bool) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": repr(error),
        "error": type(error).__name__,
        "phase": phase,
        "cacheable": cacheable,
    }


def _sample(
    ours: Callable[[], Any], theirs: Callable[[], Any], margin: float
) -> dict[str, Any]:
    """Fast timings of both, plus interleaved samples unless the fast ones
    settle the decision at ``margin``."""
    with _timing():
        timings: dict[str, Any] = {
            "fast": {
                side: [time_with_cuda_graph_us(fn) for _ in range(_FAST_TIMINGS)]
                for side, fn in (("ours", ours), ("inductor", theirs))
            },
            "interleaved": None,
        }
        if verdict(timings, margin) is None:
            samples: dict[str, list[float]] = {"ours": [], "inductor": []}
            sides = [("ours", ours), ("inductor", theirs)]
            for i in range(_INTERLEAVED_ROUNDS):
                for side, fn in sides if i % 2 == 0 else sides[::-1]:
                    samples[side] += cuda_graph_call_times_us(fn)
            timings["interleaved"] = samples
    return timings


def measure(
    backend: Any,
    options: dict[str, Any],
    spec: dict[str, Any],
    device: torch.device,
    margin: float,
) -> dict[str, Any]:
    """Both implementations of a call: whether each can run, and how fast.
    Runs inside the backend's AOTI compile context."""
    from executorch.backends.cuda.autotune.launch_params import autotune_launch_params

    replacement = spec["replacement"]["name"]
    candidates: dict[str, dict[str, Any]] = {}
    ours = theirs = None
    try:
        with autotune_launch_params() as stats:
            ours = _compile_replacement(spec["replacement"], device)
        candidates[replacement] = {
            "status": "available",
            # Its own pick, which the exporting process makes again for AOTI.
            "picks": [
                {
                    "function": p.function,
                    "param": p.param,
                    "choice": p.choice,
                    "times_us": {str(k): v for k, v in p.times_us.items()},
                }
                for p in stats.picks
            ],
        }
    except Exception as e:  # noqa: BLE001 - unavailable for this export
        candidates[replacement] = _unavailable("compile", e, False)
    try:
        # Compiles and runs it, so a fallback the runtime provides has to load.
        theirs, missing = _compile_inductor(backend, options, spec, device)
    except Exception as e:  # noqa: BLE001 - cached only if deterministic
        candidates[REGULAR_LOWERING] = _unavailable("compile", e, _no_valid_choices(e))
    else:
        if missing:
            theirs = None
            candidates[REGULAR_LOWERING] = {
                "status": "unavailable",
                "reason": f"calls {', '.join(missing)}, which the runtime does not "
                "provide",
                "phase": "compile",
                "cacheable": True,
            }
        else:
            candidates[REGULAR_LOWERING] = {"status": "available"}
    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidates": candidates,
        "timings": None,
    }
    if ours is not None and theirs is not None:
        try:
            entry["timings"] = _sample(ours, theirs, margin)
        except Exception as e:  # noqa: BLE001 - nothing was timed
            return {"unmeasurable": f"timing raised {e!r}"}
    entry["cacheable"] = all(
        c["status"] == "available" or c["cacheable"] for c in candidates.values()
    )
    return entry


def _inductor_options(backend: Any, compile_specs: list[Any]) -> dict[str, Any]:
    options = {
        k: v
        for k, v in backend.get_aoti_compile_options(compile_specs).items()
        if not k.startswith("aot_inductor.")
    }
    # The wrapper AOTI emits, so the regular lowering makes AOTI's choices and
    # the collector sees the kernels AOTI would call; a cached graph would
    # skip that codegen.
    options.update(cpp_wrapper=True, fx_graph_cache=False)
    return options


def _cpp_wrapper_unavailable(device: torch.device) -> Optional[str]:
    """Why this process cannot run a module compiled with Inductor's C++
    wrapper, or None if it can. A build that links torch into its executable
    without exporting torch's C shims (e.g. some packaged binaries) cannot
    load one."""
    x = torch.ones(8, device=device)
    torch._dynamo.reset()
    try:
        with torch._inductor.config.patch(cpp_wrapper=True, fx_graph_cache=False):
            torch.compile(lambda t: t + 1, fullgraph=True, dynamic=False)(x)
    except Exception as e:  # noqa: BLE001 - reported, nothing is measured
        return repr(e)
    return None


def _module() -> str:
    # The worker runs this module as ``python -m``, where it is ``__main__``.
    return __spec__.name if __spec__ is not None else __name__


@contextlib.contextmanager
def _without_autotune_cache_writes() -> Iterator[None]:
    """Inductor saves its autotune timings even with ``autotune_local_cache``
    off; a warm-up process's are taken on a shared GPU and must not reach the
    measurement."""
    from torch._inductor.codecache import PersistentCache

    update = PersistentCache.update_local_cache
    PersistentCache.update_local_cache = lambda self, cache: None
    try:
        with torch._inductor.config.patch(autotune_local_cache=False):
            yield
    finally:
        PersistentCache.update_local_cache = update


def _warm_one(
    backend: Any,
    compile_specs: list[Any],
    options: dict[str, Any],
    item: dict[str, Any],
) -> None:
    """Compiles both implementations of one call, for their compiled kernels
    only."""
    device = torch.device(item["device"])
    torch.cuda.set_device(device)
    with backend.get_extra_aoti_compile_context_manager(
        compile_specs
    ), _without_autotune_cache_writes():
        try:
            _compile_replacement(item["spec"]["replacement"], device)
        finally:
            _compile_inductor(backend, options, item["spec"], device)


def _size(spec: dict[str, Any]) -> int:
    return sum(
        math.prod(value["tensor"][1])
        for value in spec["args"]
        if isinstance(value, dict)
    )


class _WarmUp:
    """Compiles each call in a process of its own, largest first, several at
    a time, so that the measurement, one call at a time, finds the kernels
    compiled. The processes decide nothing: they publish no result and leave
    no autotune timings."""

    def __init__(self, job: dict[str, Any], directory: Path) -> None:
        self._job = job
        self._directory = directory
        self._lock = threading.Lock()
        self._running: set[subprocess.Popen[bytes]] = set()
        self._stopped = False
        self._executor = ThreadPoolExecutor(1)
        self._future = self._executor.submit(self._run)

    def result(self) -> dict[str, Any]:
        """Waits for the processes; their statistics."""
        return self._future.result()

    def stop(self) -> None:
        """Kills the running processes and starts no more."""
        with self._lock:
            self._stopped = True
            for proc in self._running:
                proc.kill()
        self._executor.shutdown()

    def _run(self) -> dict[str, Any]:
        from torch._inductor.async_compile import get_compile_threads
        from torch._inductor.utils import python_subprocess_env

        signatures = self._job["signatures"]
        threads = get_compile_threads()
        workers = min(self._job["workers"], len(signatures), threads)
        if len(signatures) < 2 or workers < 1:
            return {}
        env = {
            **python_subprocess_env(),
            "PYTHONUNBUFFERED": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": str(max(1, threads // workers)),
        }
        start = time.perf_counter()
        order = sorted(signatures, key=lambda key: -_size(signatures[key]["spec"]))
        with ThreadPoolExecutor(workers) as pool:
            calls = dict(pool.map(lambda key: self._compile(key, env), order))
        return {
            "workers": workers,
            "wall_s": time.perf_counter() - start,
            "calls": calls,
        }

    def _compile(self, key: str, env: dict[str, str]) -> tuple[str, dict[str, Any]]:
        start = time.perf_counter()
        log = self._directory / f"warm_{key}.log"
        try:
            proc = self._launch(key, env, log)
        except OSError as e:
            # The warm-up is optional: the measurement compiles what it lacks.
            return key, {
                "rc": None,
                "wall_s": time.perf_counter() - start,
                "error": f"could not start: {e!r}",
            }
        if proc is None:
            return key, {"rc": None, "wall_s": 0.0, "error": "stopped"}
        rc = _wait_for_worker(proc, log)
        with self._lock:
            self._running.discard(proc)
        stats: dict[str, Any] = {"rc": rc, "wall_s": time.perf_counter() - start}
        if rc != 0:
            output = log.read_text(errors="replace").splitlines()
            stats["error"] = next((x for x in reversed(output) if x.strip()), "")
        return key, stats

    def _launch(
        self, key: str, env: dict[str, str], log: Path
    ) -> Optional[subprocess.Popen[bytes]]:
        """Starts the process compiling ``key``; None once stopped."""
        path = self._directory / f"warm_{key}.json"
        path.write_text(
            json.dumps(
                {
                    **self._job,
                    "warm": True,
                    "signatures": {key: self._job["signatures"][key]},
                }
            )
        )
        with open(log, "w") as f, self._lock:
            if self._stopped:
                return None
            proc = subprocess.Popen(
                [sys.executable, "-m", _module(), str(path)],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            self._running.add(proc)
        return proc


def _worker_main(job_path: str) -> None:
    from executorch.exir.backend.compile_spec_schema import CompileSpec
    from torch._inductor.async_compile import AsyncCompile

    job = json.loads(Path(job_path).read_text())
    for backend_name, op_name, precision in job["precision"]:
        torch._C._set_fp32_precision_setter(backend_name, op_name, precision)
    module, name = job["backend"].split(":")
    backend = getattr(importlib.import_module(module), name)
    compile_specs = [CompileSpec(k, bytes.fromhex(v)) for k, v in job["compile_specs"]]
    options = _inductor_options(backend, compile_specs)
    out = Path(job_path).parent
    if job.get("warm"):
        # Compiled the same way as in the measurement, the kernels have the
        # same cache keys.
        AsyncCompile.wait_process_pool_ready()
        (item,) = job["signatures"].values()
        _warm_one(backend, compile_specs, options, item)
        return
    warm_up = _WarmUp(job, out)
    try:
        AsyncCompile.wait_process_pool_ready()
        reason = _cpp_wrapper_unavailable(torch.device(job["device"]))
        _publish(out / "capability.json", {"unavailable": reason})
        if reason is not None:
            return
        _publish(out / "warm.json", warm_up.result())
    finally:
        warm_up.stop()
    with backend.get_extra_aoti_compile_context_manager(compile_specs):
        for key, item in job["signatures"].items():
            device = torch.device(item["device"])
            torch.cuda.set_device(device)
            result = measure(backend, options, item["spec"], device, job["margin"])
            _publish(out / f"{key}.json", result)


def _progress(written: Path) -> tuple[tuple[str, int], ...]:
    """What a worker has written: its job directory's results and log, or one
    log file."""
    files = []
    for path in written.iterdir() if written.is_dir() else [written]:
        try:
            files.append((path.name, path.stat().st_size))
        except FileNotFoundError:
            continue
    return tuple(sorted(files))


def _wait_for_worker(proc: subprocess.Popen[bytes], written: Path) -> Optional[int]:
    """The worker's return code, or None after it was killed for writing
    nothing to ``written`` (a directory or a file) for
    ``_WORKER_IDLE_TIMEOUT_S``."""
    seen, since = _progress(written), time.monotonic()
    while True:
        try:
            return proc.wait(timeout=_WORKER_POLL_S)
        except subprocess.TimeoutExpired:
            pass
        now = _progress(written)
        if now != seen:
            seen, since = now, time.monotonic()
        elif time.monotonic() - since > _WORKER_IDLE_TIMEOUT_S:
            proc.kill()
            proc.wait()
            return None


def _run_worker(
    backend: Any,
    compile_specs: list[Any],
    config: MeasuredRoutingConfig,
    device: str,
    signatures: dict[str, dict[str, Any]],
) -> tuple[Optional[str], dict[str, dict[str, Any]]]:
    """Measures ``signatures`` (key -> spec and device) in a worker process.
    Returns why the worker cannot measure in this process (None if it can)
    and the results it reported."""
    from torch._inductor.utils import python_subprocess_env

    jobs = config.directory() / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=jobs))
    try:
        job = {
            "backend": f"{backend.__module__}:{backend.__qualname__}",
            "compile_specs": [[s.key, s.value.hex()] for s in compile_specs],
            "precision": _precision_state(),
            "margin": config.margin,
            "workers": config.workers,
            "device": device,
            "signatures": signatures,
        }
        (directory / "job.json").write_text(json.dumps(job))
        log = directory / "worker.log"
        with open(log, "w") as f:
            proc = subprocess.Popen(
                [sys.executable, "-m", _module(), str(directory / "job.json")],
                # Unbuffered, so the log shows the worker's progress as it goes.
                env={**python_subprocess_env(), "PYTHONUNBUFFERED": "1"},
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            rc = _wait_for_worker(proc, directory)
        output = log.read_text().splitlines()
        capability = directory / "capability.json"
        if capability.exists():
            reason = json.loads(capability.read_text())["unavailable"]
        else:
            last = next((line for line in reversed(output) if line.strip()), "")
            reason = f"the measurement worker failed to start (rc={rc}): {last}"
        warm = directory / "warm.json"
        warm = json.loads(warm.read_text()) if warm.exists() else {}
        results = {}
        for key in signatures:
            path = directory / f"{key}.json"
            if path.exists():
                results[key] = json.loads(path.read_text())
                if warm:
                    results[key]["warm"] = {
                        "workers": warm["workers"],
                        "batch_wall_s": warm["wall_s"],
                        **warm["calls"][key],
                    }
        if rc != 0:
            logger.error(
                "The measurement worker failed (rc=%s) after reporting %d of %d "
                "calls:\n%s",
                rc,
                len(results),
                len(signatures),
                "\n".join(output[-20:]),
            )
        return reason, results
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _decide(
    plan: ReplacementPlan, entry: dict[str, Any], margin: float
) -> tuple[str, str]:
    """(chosen candidate, outcome) for a measured call."""
    statuses = entry["candidates"]
    available = [
        c.name for c in plan.candidates if statuses[c.name]["status"] == "available"
    ]
    if not available:
        reasons = "; ".join(
            f"{name}: {status.get('reason')}" for name, status in statuses.items()
        )
        raise ReplacementError(f"{plan.signature}: nothing can run it ({reasons})")
    if len(available) == 1:
        return available[0], "only_available"
    decided = verdict(entry["timings"], margin)
    if decided is None:
        raise AssertionError("timings without the samples their decision needs")
    keep, outcome = decided
    return (plan.candidates[1].name if keep else REGULAR_LOWERING), outcome


class MeasureChoicesPass(PassBase):
    """Measures the implementations of each recorded call and stores the
    choice in ``node.meta[META_KEY]``."""

    def __init__(
        self, backend: Any, compile_specs: list[Any], config: MeasuredRoutingConfig
    ) -> None:
        super().__init__()
        self._backend = backend
        self._compile_specs = compile_specs
        self._config = config

    def call(self, graph_module: GraphModule) -> PassResult:  # noqa: C901
        nodes = [n for n in graph_module.graph.nodes if META_KEY in n.meta]
        if not nodes:
            return PassResult(graph_module, False)
        start = time.perf_counter()
        options = _inductor_options(self._backend, self._compile_specs)
        cache = self._config.directory() / "cache"
        prepared: dict[Node, tuple[Optional[str], Optional[dict[str, Any]], str]] = {}
        entries: dict[str, Optional[dict[str, Any]]] = {}
        pending: dict[str, dict[str, Any]] = {}
        for node in nodes:
            try:
                spec = _spec(node, node.meta[META_KEY]["plan"])
            except Unmeasurable as e:
                prepared[node] = (None, None, str(e))
                continue
            device = next(
                v.device
                for v in pytree.tree_leaves(node.meta["val"])
                if isinstance(v, torch.Tensor)
            )
            key = cache_key(spec, options, device)
            prepared[node] = (key, spec, "")
            if key not in entries:
                plan = node.meta[META_KEY]["plan"]
                entries[key] = self._cached(cache / f"{key}.json", plan)
                if entries[key] is None:
                    pending[key] = {"spec": spec, "device": str(device)}
        hits = {key for key, entry in entries.items() if entry is not None}
        warm: dict[str, Optional[dict[str, Any]]] = {}
        logger.info(
            "Measuring %d calls: %d signatures, %d cached (margin %.2f, cache %s)",
            len(nodes),
            len(entries),
            len(hits),
            self._config.margin,
            self._config.directory(),
        )
        # Whether this process can measure at all decides before any cached
        # entry is used; a process found able to stays able.
        device = str(
            next(
                v.device
                for n in nodes
                for v in pytree.tree_leaves(n.meta["val"])
                if isinstance(v, torch.Tensor)
            )
        )
        unavailable = None
        if pending or device not in _capable_devices:
            unavailable, results = _run_worker(
                self._backend, self._compile_specs, self._config, device, pending
            )
            if unavailable is None:
                _capable_devices.add(device)
                for key, result in results.items():
                    warm[key] = result.pop("warm", None)
                    if result.pop("cacheable", False):
                        _publish(cache / f"{key}.json", result)
                    entries[key] = result
        if unavailable is not None:
            logger.warning(
                "Measured routing cannot run in this process, so all %d calls take "
                "their static choice: the measurement worker cannot load modules "
                "compiled with Inductor's C++ wrapper (%s)",
                len(nodes),
                unavailable,
            )
        for node, (key, spec, unmeasurable) in prepared.items():
            if unavailable is not None:
                entry = {"unmeasurable": unavailable, "environment": True}
                hit = False
            else:
                entry = None if key is None else entries.get(key)
                if entry is None:
                    entry = {
                        "unmeasurable": unmeasurable or "the worker did not report it"
                    }
                hit = key in hits
            result = self._result(node, spec, entry, hit)
            result["warm"] = warm.get(key)
            node.meta[META_KEY]["result"] = result
        logger.info("Measured in %.1f s", time.perf_counter() - start)
        return PassResult(graph_module, False)

    def _cached(self, path: Path, plan: ReplacementPlan) -> Optional[dict[str, Any]]:
        entry = _read_entry(path, plan)
        timings = None if entry is None else entry.get("timings")
        if timings is not None and verdict(timings, self._config.margin) is None:
            return None
        return entry

    def _result(
        self,
        node: Node,
        spec: Optional[dict[str, Any]],
        entry: dict[str, Any],
        cache_hit: bool,
    ) -> dict[str, Any]:
        plan = node.meta[META_KEY]["plan"]
        if entry.get("environment"):
            choice, outcome = plan.static_choice, "unmeasurable"
        elif "unmeasurable" in entry:
            choice, outcome = plan.static_choice, "unmeasurable"
            logger.error(
                "Cannot measure %s (%s); it takes its static choice, %s (%s)",
                node.name,
                entry["unmeasurable"],
                choice,
                plan.static_reason,
            )
        else:
            choice, outcome = _decide(plan, entry, self._config.margin)
        timings = entry.get("timings")
        candidates = []
        for c in plan.candidates:
            status = {"status": "illegal", "reason": c.reason}
            if c.reason is None:
                status = dict(entry.get("candidates", {}).get(c.name, {}))
            if timings is not None:
                side = "inductor" if c.name == REGULAR_LOWERING else "ours"
                status["fast_us"] = _summary(timings["fast"][side])
            candidates.append({"name": c.name, "label": c.label, **status})
        interleaved = None if timings is None else timings["interleaved"]
        return {
            "node": node.meta[META_KEY]["name"],
            "consumers": [str(u.target) for u in node.users],
            "spec": spec,
            "candidates": candidates,
            "interleaved_us": (
                None
                if interleaved is None
                else {side: _summary(s) for side, s in interleaved.items()}
            ),
            "timings": timings,
            "static_choice": plan.static_choice,
            "choice": choice,
            "outcome": outcome,
            "margin": self._config.margin,
            "cache_hit": cache_hit,
        }


class ApplyChoicesPass(PassBase):
    """Replaces each measured call whose choice is a replacement and writes
    the export's log."""

    def __init__(self, method_name: str, config: MeasuredRoutingConfig) -> None:
        super().__init__()
        self._method_name = method_name
        self._config = config

    def call(self, graph_module: GraphModule) -> PassResult:
        graph = graph_module.graph
        recorded = []
        for node in graph.nodes:
            info = node.meta.pop(META_KEY, None)
            if info is not None and "result" in info:
                recorded.append((node, info))
        # A recorded call may take an earlier one, which may be replaced here.
        replaced: dict[Node, Node] = {}
        for node, info in recorded:
            plan, choice = info["plan"], info["result"]["choice"]
            if choice == REGULAR_LOWERING:
                continue
            candidate = plan.candidate(choice)
            args, kwargs = map_arg(
                (candidate.args, candidate.kwargs), lambda n: replaced.get(n, n)
            )
            new_node = replace(
                graph,
                node,
                plan,
                dataclasses.replace(candidate, args=args, kwargs=kwargs),
            )
            info["result"]["replaced"] = new_node is not None
            if new_node is not None:
                replaced[node] = new_node
        modified = bool(replaced)
        if modified:
            graph.lint()
            graph_module.recompile()
        if recorded:
            self._write_log([info["result"] for _, info in recorded])
        return PassResult(graph_module, modified)

    def _write_log(self, results: list[dict[str, Any]]) -> None:
        outcomes = [r["outcome"] for r in results]
        choices = [r["choice"] for r in results]
        summary = {
            "method": self._method_name,
            "config": {
                "margin": self._config.margin,
                "cache_dir": str(self._config.directory()),
                "workers": self._config.workers,
            },
            "measured": len(results),
            "choices": {c: choices.count(c) for c in sorted(set(choices))},
            "outcomes": {o: outcomes.count(o) for o in sorted(set(outcomes))},
            "cache_hits": sum(r["cache_hit"] for r in results),
            "warm_wall_s": max(
                (r["warm"]["batch_wall_s"] for r in results if r.get("warm")),
                default=0.0,
            ),
            "nodes": results,
        }
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"{self._method_name}_{stamp}_{os.getpid()}_{os.urandom(4).hex()}.json"
        path = self._config.directory() / "logs" / name
        _publish(path, summary)
        logger.info(
            "Measured routing (margin %.2f): %s of %d calls, outcomes %s, %d "
            "cached; log %s",
            self._config.margin,
            summary["choices"],
            summary["measured"],
            summary["outcomes"],
            summary["cache_hits"],
            path,
        )


if __name__ == "__main__":
    _worker_main(sys.argv[1])
