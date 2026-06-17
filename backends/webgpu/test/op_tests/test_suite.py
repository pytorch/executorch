# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative op-test schema for the WebGPU backend.

Mirrors the authoring ergonomics of the Vulkan op-test framework
(backends/vulkan/test/op_tests/cases.py) — a per-op suite of input cases
registered via a decorator — but the reference engine is a torch golden
loaded in C++ (the WebGPU native binary has no ATen), not an inline ATen call.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

# Prime-number dim sizes, mirroring backends/vulkan/test/op_tests/cases.py.
XL = 113
L = 89
M2 = 41
M1 = 37
M = 29
S2 = 11
S1 = 7
S = 5
XS = 3


@dataclass
class InputSpec:
    """A single forward input. The materialized tensor feeds both the torch golden
    and the exported .pte (no C++ reconstruction)."""

    shape: Tuple[int, ...]
    dtype: str = "float32"
    gen: Union[str, Callable] = "randn"  # "randn"/"ramp" or a callable(shape)->Tensor


# A forward input is either a bare shape tuple (-> default-randn fp32) or an InputSpec.
Input = Union[Tuple[int, ...], InputSpec]


@dataclass
class Case:
    """One generated test. `construct` kwargs go to `module_factory` and are baked into
    the .pte as constants (never serialized inputs); `inputs` are the forward inputs only.
    `atol`/`rtol` override suite defaults; `required` -> missing .pte FAILs (not skips);
    `heavy` -> export-gated behind WEBGPU_TEST_HEAVY; `golden_fn` overrides the fp64 oracle.
    """

    construct: Dict[str, object] = field(default_factory=dict)
    inputs: Tuple[Input, ...] = ()
    name: Optional[str] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    required: bool = True
    heavy: bool = False
    golden_fn: Optional[Callable] = None

    def __post_init__(self) -> None:
        # Mirror kQ4gswConfigs: every heavy config is required=False (export-gated, never FAILs on absence).
        if self.heavy:
            self.required = False


@dataclass
class WebGPUTestSuite:
    """A per-op suite: a module factory + the list of cases to generate."""

    module_factory: Callable[..., object]
    cases: List[Case]
    atol: float = 1e-3
    rtol: float = 1e-3
    # Golden oracle dtype. "float64" (default) computes the golden via an fp64 forward;
    # gather/copy ops set "float32" since .double() is bit-identical (skips the dual-oracle gate).
    golden_dtype: str = "float64"
    # `verified=False` declares a suite NOT in the default green run yet (handler unbuilt
    # or GPU numerics unconfirmed); the generator skips it unless named via --ops or --all.
    verified: bool = True


op_test_registry: Dict[str, WebGPUTestSuite] = {}


def register_op_test(name_or_names: Union[str, List[str]]) -> Callable:
    """Decorator: register the suite returned by `fn()` under one or more op names
    (list-aware, mirroring Vulkan's register_test_suite)."""

    def decorator(fn: Callable[[], WebGPUTestSuite]) -> Callable:
        suite = fn()
        names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        for name in names:
            op_test_registry[name] = suite
        return fn

    return decorator
