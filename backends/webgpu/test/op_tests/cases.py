# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""WebGPU op-test cases.

Declarative per-op suites for the manifest-driven op-test framework, mirroring the
Vulkan op-test authoring ergonomics. Each op reuses its `nn.Module` + generators from
the per-op `test_*.py`; new ops append a `@register_op_test` entry in their own tests-diff.
"""

import torch

from executorch.backends.webgpu.test.op_tests.test_suite import (
    Case,
    InputSpec,
    M,
    M1,
    M2,
    register_op_test,
    S,
    S1,
    S2,
    WebGPUTestSuite,
    XS,
)
from executorch.backends.webgpu.test.ops.add.test_add import (
    AddChainedModule,
    AddModule,
    AddSelfModule,
)
from executorch.backends.webgpu.test.ops.rms_norm.test_rms_norm import (
    _CASES,
    _linspace_weight,
    _ramp,
    RmsNormModule,
)
from executorch.backends.webgpu.test.ops.sigmoid.test_sigmoid import (
    _sigmoid_range,
    _sigmoid_wide_range,
    SigmoidChainedModule,
    SigmoidModule,
)

# rms_norm coverage is exactly the 15 cases the native test covered.
RMS_NORM_CASES = _CASES


def _add_factory(variant: str = "regular") -> torch.nn.Module:
    return {
        "regular": AddModule,
        "self": AddSelfModule,
        "chained": AddChainedModule,
    }[variant]()


def _sigmoid_factory(variant: str = "regular") -> torch.nn.Module:
    return {
        "regular": SigmoidModule,
        "chained": SigmoidChainedModule,
    }[variant]()


@register_op_test("add")
def _add_suite() -> WebGPUTestSuite:
    # Same-shape numeric coverage only: broadcast adds stay export-smoke in
    # ops/add/test_add.py because the kernel can't broadcast.
    return WebGPUTestSuite(
        module_factory=_add_factory,
        cases=[
            Case(
                name="regular_2d",
                construct={"variant": "regular"},
                inputs=((M1, M2), (M1, M2)),
            ),
            Case(
                name="regular_3d",
                construct={"variant": "regular"},
                inputs=((S, S1, S2), (S, S1, S2)),
            ),
            Case(
                name="regular_4d",
                construct={"variant": "regular"},
                inputs=((XS, S, S1, S2), (XS, S, S1, S2)),
            ),
            Case(name="self", construct={"variant": "self"}, inputs=((M1, M2),)),
            # "scalar" (x+3.0) is intentionally OMITTED — the WebGPU add kernel can't
            # do scalar/broadcast adds (0x30 at runtime); it stays export-smoke.
            Case(
                name="chained",
                construct={"variant": "chained"},
                inputs=((M1, M2), (M1, M2)),
            ),
        ],
    )


@register_op_test("sigmoid")
def _sigmoid_suite() -> WebGPUTestSuite:
    return WebGPUTestSuite(
        module_factory=_sigmoid_factory,
        cases=[
            Case(
                name="regular_1d",
                construct={"variant": "regular"},
                inputs=(InputSpec(shape=(M,), gen=_sigmoid_range),),
            ),
            Case(
                name="regular_2d",
                construct={"variant": "regular"},
                inputs=(InputSpec(shape=(M1, M2), gen=_sigmoid_range),),
            ),
            Case(
                name="regular_4d",
                construct={"variant": "regular"},
                inputs=(InputSpec(shape=(XS, S, S1, S2), gen=_sigmoid_range),),
            ),
            Case(
                name="wide_range",
                construct={"variant": "regular"},
                inputs=(InputSpec(shape=(M1, M2), gen=_sigmoid_wide_range),),
            ),
            Case(
                name="chained",
                construct={"variant": "chained"},
                inputs=(InputSpec(shape=(M1, M2), gen=_sigmoid_range),),
            ),
        ],
    )


def _rms_norm_factory(hidden: int, eps: float, weight_fn) -> torch.nn.Module:
    model = RmsNormModule(hidden, eps=eps)
    with torch.no_grad():
        model.weight.copy_(weight_fn(hidden))
    return model


@register_op_test("rms_norm")
def _rms_norm_suite() -> WebGPUTestSuite:
    cases = []
    for c in RMS_NORM_CASES:
        shape = c["shape"]
        hidden = shape[-1]
        weight_fn = c.get("weight_fn", _linspace_weight)
        input_fn = c.get("input_fn", _ramp)
        cases.append(
            Case(
                name=c["name"],
                construct={"hidden": hidden, "eps": 1e-6, "weight_fn": weight_fn},
                inputs=(InputSpec(shape=shape, gen=input_fn),),
            )
        )
    return WebGPUTestSuite(module_factory=_rms_norm_factory, cases=cases)
