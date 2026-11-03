# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Op cases: a model and the inputs of each sample it is recorded with."""

import inspect
from dataclasses import dataclass
from types import ModuleType

import torch

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
