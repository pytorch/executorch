# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.webgpu.test.op_tests.test_suite import (
    Case,
    M,
    op_test_registry,
    register_op_test,
    S,
    S1,
    S2,
    WebGPUTestSuite,
    XL,
)
from executorch.backends.webgpu.test.tester import WEBGPU_SUPPORTED_OPS
from executorch.exir.dialects._ops import ops as exir_ops


def test_decorator_registers():
    @register_op_test("dummy")
    def _dummy():
        return WebGPUTestSuite(
            module_factory=lambda **kw: None,  # kw = per-case construction params
            cases=[
                Case(construct={}, inputs=((M, M), (M, M))),
                Case(construct={}, inputs=((S, S1, S2),)),
            ],
            atol=1e-3,
            rtol=1e-3,
        )

    assert "dummy" in op_test_registry
    suite = op_test_registry["dummy"]
    assert len(suite.cases) == 2 and suite.cases[0].construct == {}
    assert suite.atol == 1e-3 and isinstance(XL, int)


def test_add_mul_sigmoid_rms_norm_registered():
    from executorch.backends.webgpu.test.op_tests import cases  # registers

    assert {"add", "mul", "sigmoid", "rms_norm"} <= set(op_test_registry)
    assert len(op_test_registry["add"].cases) >= 3  # regular/self/chained
    assert len(op_test_registry["mul"].cases) >= 6  # regular/broadcast/self/chained
    assert len(op_test_registry["sigmoid"].cases) >= 4  # regular/range/chained
    # Exact parity, no hardcoded literal (real _CASES == 15; import so it can't drift):
    assert len(op_test_registry["rms_norm"].cases) == len(cases.RMS_NORM_CASES)
    # weight is a construction param, NOT a forward input:
    rms0 = op_test_registry["rms_norm"].cases[0]
    assert "weight_fn" in rms0.construct and len(rms0.inputs) == 1


def test_add_cases_are_same_shape_no_scalar():
    # Contract: the WebGPU add kernel only handles same-shape adds (no scalar/broadcast
    # -> 0x30 at runtime), so the numeric add suite must contain only same-shape cases.
    from executorch.backends.webgpu.test.op_tests import cases  # noqa: F401  registers

    for c in op_test_registry["add"].cases:
        assert c.construct.get("variant") != "scalar", f"{c.name}: scalar add can't run"
        shapes = {tuple(s if isinstance(s, tuple) else s.shape) for s in c.inputs}
        assert len(shapes) == 1, f"{c.name}: not same-shape (broadcast?): {shapes}"


def test_mul_cases_cover_broadcast_without_scalar():
    from executorch.backends.webgpu.test.op_tests import cases  # noqa: F401  registers

    saw_broadcast = False
    for c in op_test_registry["mul"].cases:
        shapes = [tuple(s if isinstance(s, tuple) else s.shape) for s in c.inputs]
        if len(shapes) == 2 and shapes[0] != shapes[1]:
            saw_broadcast = True
        assert c.construct.get("variant") != "scalar", f"{c.name}: scalar mul can't run"
    assert saw_broadcast


def test_webgpu_supported_ops_include_registered_elementwise_ops():
    assert exir_ops.edge.aten.add.Tensor in WEBGPU_SUPPORTED_OPS
    assert exir_ops.edge.aten.mul.Tensor in WEBGPU_SUPPORTED_OPS
    assert exir_ops.edge.aten.sigmoid.default in WEBGPU_SUPPORTED_OPS
    assert exir_ops.edge.et_vk.rms_norm.default in WEBGPU_SUPPORTED_OPS


def test_case_required_heavy_golden_fn_defaults():
    c = Case(inputs=((M, M),))
    assert c.required is True and c.heavy is False and c.golden_fn is None


def test_heavy_forces_not_required():
    # __post_init__ invariant: a heavy case is export-gated, so it is never required
    # (mirrors kQ4gswConfigs, every heavy config required=False).
    c = Case(inputs=((M, M),), heavy=True)
    assert c.heavy is True and c.required is False


def test_golden_dtype_default():
    from executorch.backends.webgpu.test.op_tests import cases  # noqa: F401  registers

    # fp64 oracle is the default; these compute ops keep it. (Per-op golden_dtype
    # for gather/copy ops is asserted in each op's own tests-diff.)
    assert op_test_registry["add"].golden_dtype == "float64"
    assert op_test_registry["mul"].golden_dtype == "float64"
    assert op_test_registry["sigmoid"].golden_dtype == "float64"
    assert op_test_registry["rms_norm"].golden_dtype == "float64"
