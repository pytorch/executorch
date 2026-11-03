# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Op cases: a model and the inputs of each sample it is recorded with, or of
the calls of one sample when its shapes are dynamic."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import torch
from torch.export import Dim

if TYPE_CHECKING:
    from facto.inputgen.specs.model import Spec

Inputs = tuple[torch.Tensor, ...]

# Caps a suite case's input and output elements over its samples, keeping at
# least one sample: a large output alone has enough elements for statistics.
_ELEMENT_BUDGET = 1 << 16


@dataclass
class Case:
    name: str
    model: torch.nn.Module
    # Positional float32 inputs of each sample; the count may vary by case.
    inputs: list[Inputs]
    # `{dim: Dim}` of each positional input. A dynamic case's inputs are the
    # calls of one sample, which run in order on one executable.
    dynamic_shapes: tuple[dict[int, Dim], ...] | None = None


def outputs(result) -> list[torch.Tensor]:
    return list(result) if isinstance(result, (tuple, list)) else [result]


def draws(count: int, *shapes: tuple[int, ...]) -> list[Inputs]:
    """`count` samples of standard-normal inputs of the given shapes, from a fixed
    seed, so equal calls return equal samples."""
    generator = torch.Generator().manual_seed(0)
    return [
        tuple(torch.randn(shape, generator=generator) for shape in shapes)
        for _ in range(count)
    ]


def chain(name: str, model: torch.nn.Module, *shapes: tuple[int | Dim, ...]) -> Case:
    """A dynamic case with standard-normal inputs of `shapes`, from a fixed seed.
    Each Dim starts mid-range, which export traces, then visits both bounds and
    returns, every other Dim in the opposite order."""
    dims = list(dict.fromkeys(d for s in shapes for d in s if not isinstance(d, int)))
    walks = {}
    for k, d in enumerate(dims):
        mid = (d.min + d.max + 1) // 2
        walks[d] = (mid, d.min, d.max, mid) if k % 2 == 0 else (mid, d.max, d.min, mid)
    generator = torch.Generator().manual_seed(0)
    inputs = [
        tuple(
            torch.randn(
                [walks[d][step] if d in walks else d for d in s], generator=generator
            )
            for s in shapes
        )
        for step in range(4)
    ]
    dynamic_shapes = tuple(
        {i: d for i, d in enumerate(s) if not isinstance(d, int)} for s in shapes
    )
    return Case(name, model, inputs, dynamic_shapes)


class _SuiteRunner:
    """Stands in for the backend test suite's runner, keeping what a test lowers."""

    _flow = None

    def __init__(self) -> None:
        self.calls: list[tuple[torch.nn.Module, Inputs]] = []

    def lower_and_run_model(self, model, inputs, **_) -> None:
        self.calls.append((model, tuple(inputs)))


def suite_cases(module: ModuleType, samples: int) -> list[Case]:
    """One case per model a backend test suite operator module lowers, with
    inputs from up to `samples` runs under different seeds."""
    cases = []
    for test_name, test in sorted(vars(module).items()):
        if not test_name.startswith("test_"):
            continue
        runs = []
        for seed in range(samples):
            torch.manual_seed(seed)
            runner = _SuiteRunner()
            if "dtype" in inspect.signature(test).parameters:
                test(runner, torch.float32)
            else:
                test(runner)
            runs.append(runner.calls)
        lowered = [[(type(m), [x.shape for x in xs]) for m, xs in run] for run in runs]
        if any(seed_lowered != lowered[0] for seed_lowered in lowered):
            raise ValueError(f"{test_name} lowers different models across seeds")
        for i, (model, inputs) in enumerate(runs[0]):
            name = f"{test_name.removeprefix('test_')}_{i}"
            try:
                results = outputs(model(*inputs))
            except RuntimeError as e:
                e.add_note(f"while recording suite case {name}")
                raise
            numel = sum(t.numel() for t in (*inputs, *results))
            count = max(1, min(samples, _ELEMENT_BUDGET // max(1, numel)))
            cases.append(Case(name, model, [run[i][1] for run in runs[:count]]))
    if not cases:
        raise ValueError(f"no test in {module.__name__} lowers a model")
    return cases


def facto_cases(
    spec: "Spec", build: Callable[..., Case], keep: Callable[[int, tuple], bool]
) -> list[Case]:
    """One-sample cases `build(name, *args)` from FACTO's argument tuples for
    `spec`, named by position, keeping those where `keep(position, args)`."""
    # FACTO is on the path for the sample generator only, not for pytest.
    from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
    from facto.inputgen.utils.random_manager import seeded_random_manager

    seeded_random_manager.seed(0)
    return [
        build(f"facto_{i}", *args)
        for i, (args, _, _) in enumerate(ArgumentTupleGenerator(spec).gen())
        if keep(i, args)
    ]
