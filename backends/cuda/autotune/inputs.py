# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Representative values for the data-dependent arguments of autotuned kernels.

AOTInductor autotunes every kernel at compile time on inputs it fabricates from
shapes and dtypes (``torch._dynamo.testing.rand_strided``): floating tensors are
random, integer and bool tensors are all zeros. A kernel whose amount of work
comes from an integer tensor, such as the number of filled KV-cache positions,
is then timed doing no work, and the config that wins does not win at runtime.

The kernel's author declares what such arguments hold at runtime, next to the
kernel::

    @autotune_inputs(KV_LEN_ptr=lambda args: [1024, args["Lk"] // 2, args["Lk"]])
    @triton.autotune(configs=[...], key=[...])
    @triton.jit
    def kernel(..., KV_LEN_ptr, ..., Lk, ...): ...

A provider receives the kernel's arguments by name (tensors, scalars and
constexprs, as autotuning sees them) and returns one value per scenario: a
number fills the tensor, a tensor replaces it (same shape, dtype and device).
Several declared arguments pair up by position, so their lists must have the
same length. An argument passed as something other than a tensor (e.g. an
unused pointer passed as 0) is left alone.

Inside ``autotune_input_scenarios()``, which the CUDA backend enters for its
AOTI compile, every timing of a declared kernel runs once per scenario with the
declared arguments replaced by private tensors (Inductor's buffers are shared
with other kernels and stay untouched) and reports the sum, so the autotuner
ranks configs by their total cost over the declared values. Kernels nobody
declared are timed as before. Eager ``triton.autotune`` runs on real arguments
and is not affected.
"""

from __future__ import annotations

import contextlib
import inspect
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence, Union

import torch

Scenario = Union[int, float, torch.Tensor]
ValueProvider = Callable[[Mapping[str, Any]], Sequence[Scenario]]


class _Declaration:
    __slots__ = ("origin", "providers")

    def __init__(self, origin: str, providers: dict[str, ValueProvider]) -> None:
        self.origin = origin
        self.providers = providers


_REGISTRY: dict[str, _Declaration] = {}


def _python_function(kernel: Any) -> Callable[..., Any]:
    fn = kernel
    while hasattr(fn, "fn"):  # triton.autotune / heuristics -> JITFunction -> function
        fn = fn.fn
    return fn


def autotune_inputs(**providers: ValueProvider) -> Callable[[Any], Any]:
    """Declares representative values for some of a kernel's arguments.

    Apply outermost, above ``@triton.autotune``; the kernel is returned as is.
    Kernels are matched by function name at compile time (Inductor recompiles
    them from source), so declared kernel names must be unique.
    """
    if not providers:
        raise ValueError("autotune_inputs needs at least one argument declaration")

    def register(kernel: Any) -> Any:
        fn = _python_function(kernel)
        params = list(inspect.signature(fn).parameters)
        unknown = sorted(set(providers) - set(params))
        if unknown:
            raise ValueError(
                f"{fn.__name__} has no parameter(s) {unknown}; its parameters are {params}"
            )
        origin = f"{fn.__module__}.{fn.__qualname__}"
        existing = _REGISTRY.get(fn.__name__)
        if existing is not None and existing.origin != origin:
            raise ValueError(
                f"kernel name {fn.__name__} is declared by both {existing.origin} and {origin}"
            )
        _REGISTRY[fn.__name__] = _Declaration(origin, dict(providers))
        return kernel

    return register


def _scenario_tensor(
    name: str, original: torch.Tensor, value: Scenario
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if (
            value.shape != original.shape
            or value.dtype != original.dtype
            or value.device != original.device
        ):
            raise ValueError(
                f"autotune value for {name} must match its shape/dtype/device "
                f"{tuple(original.shape)}/{original.dtype}/{original.device}, got "
                f"{tuple(value.shape)}/{value.dtype}/{value.device}"
            )
        return value
    return torch.full_like(original, value)


def scenario_arguments(
    kernel_name: str,
    arg_names: Sequence[str],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    constants: Mapping[str, Any],
) -> list[tuple[tuple[Any, ...], dict[str, Any]]] | None:
    """The (args, kwargs) to time a declared kernel with, one pair per scenario,
    or None when the kernel is not declared or none of its declared arguments
    is a tensor in this call."""
    declaration = _REGISTRY.get(kernel_name)
    if declaration is None:
        return None
    named = {**constants, **dict(zip(arg_names, args)), **kwargs}
    values: dict[str, Sequence[Scenario]] = {}
    for name, provider in declaration.providers.items():
        if not isinstance(named.get(name), torch.Tensor):
            continue
        values[name] = list(provider(named))
    if not values:
        return None
    counts = {len(v) for v in values.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError(
            f"{kernel_name}: declared arguments must give the same, non-zero number "
            f"of scenarios, got { {n: len(v) for n, v in values.items()} }"
        )
    positions = {name: i for i, name in enumerate(arg_names[: len(args)])}
    runs = []
    for i in range(counts.pop()):
        run_args, run_kwargs = list(args), dict(kwargs)
        for name, scenario_values in values.items():
            tensor = _scenario_tensor(name, named[name], scenario_values[i])
            if name in run_kwargs:
                run_kwargs[name] = tensor
            else:
                run_args[positions[name]] = tensor
        runs.append((tuple(run_args), run_kwargs))
    return runs


class _State(threading.local):
    def __init__(self) -> None:
        self.depth = 0


_STATE = _State()
_INSTALL_LOCK = threading.Lock()
_install_count = 0
_original_bench: Callable[..., Any] | None = None


def _bench_over_scenarios(self, launcher, *args, **kwargs):
    assert _original_bench is not None
    if _STATE.depth == 0 or not getattr(self, "custom_kernel", False):
        return _original_bench(self, launcher, *args, **kwargs)
    runs = scenario_arguments(
        getattr(self.fn, "__name__", ""),
        self.fn.arg_names,
        args,
        kwargs,
        self.triton_meta.get("constants", {}),
    )
    if runs is None:
        return _original_bench(self, launcher, *args, **kwargs)
    return sum(
        _original_bench(self, launcher, *run_args, **run_kwargs)
        for run_args, run_kwargs in runs
    )


@contextlib.contextmanager
def autotune_input_scenarios() -> Iterator[None]:
    """Times declared kernels over their declared values on this thread."""
    global _install_count, _original_bench
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    with _INSTALL_LOCK:
        if _install_count == 0:
            # Kept after uninstalling: a call another thread already routed
            # through the wrapper must still reach the original.
            if _original_bench is None:
                _original_bench = CachingAutotuner.bench
            CachingAutotuner.bench = _bench_over_scenarios
        _install_count += 1
    _STATE.depth += 1
    try:
        yield
    finally:
        _STATE.depth -= 1
        with _INSTALL_LOCK:
            _install_count -= 1
            if _install_count == 0:
                CachingAutotuner.bench = _original_bench


__all__ = ["autotune_input_scenarios", "autotune_inputs", "scenario_arguments"]
