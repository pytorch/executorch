# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import tosa_serializer as ts
from executorch.backends.arm.tosa.constant_pool import TosaSerializerWithConstantPool


def _serializer(path_prefix=""):
    return TosaSerializerWithConstantPool(
        path_prefix,
        targetMajor=1,
        targetMinor=0,
        targetPatch=0,
        targetDraft=False,
    )


def test_identical_constants_are_reused():
    serializer = _serializer()

    first = serializer.addConst([1], ts.DType.INT8, [0], name="first")
    second = serializer.addConst([1], ts.DType.INT8, [0], name="second")

    assert isinstance(serializer, ts.TosaSerializer)
    assert second is first
    assert first.name == "first"
    assert len(serializer.currRegion.currBasicBlock.operators) == 1
    assert list(serializer.currRegion.currBasicBlock.tensors.keys()) == ["first"]


def test_unpooled_constants_are_not_reused():
    serializer = _serializer()

    first = serializer.addUnpooledConst([1], ts.DType.INT8, [0], name="first")
    second = serializer.addUnpooledConst([1], ts.DType.INT8, [0], name="second")

    assert second is not first
    assert len(serializer.currRegion.currBasicBlock.operators) == 2
    assert list(serializer.currRegion.currBasicBlock.tensors.keys()) == [
        "first",
        "second",
    ]


def test_unnamed_constant_uses_serializer_generated_name():
    serializer = _serializer()

    constant = serializer.addConst([1], ts.DType.INT8, [0])

    assert constant.name


@pytest.mark.parametrize(
    "first,second",
    [
        (([1], ts.DType.INT8, [0]), ([1], ts.DType.INT16, [0])),
        (([1], ts.DType.INT8, [0]), ([2], ts.DType.INT8, [0, 0])),
        (([1], ts.DType.INT8, [0]), ([1], ts.DType.INT8, [1])),
    ],
)
def test_constants_with_different_keys_remain_separate(first, second):
    serializer = _serializer()

    first_const = serializer.addConst(*first, name="first")
    second_const = serializer.addConst(*second, name="second")

    assert second_const is not first_const
    assert len(serializer.currRegion.currBasicBlock.operators) == 2


def test_float_constants_use_exact_serialized_values():
    serializer = _serializer()

    positive_zero = serializer.addConst(
        [1], ts.DType.FP32, np.array([0.0]), name="positive_zero"
    )
    negative_zero = serializer.addConst(
        [1], ts.DType.FP32, np.array([-0.0]), name="negative_zero"
    )
    repeated_negative_zero = serializer.addConst(
        [1],
        ts.DType.FP32,
        np.array([-0.0]),
        name="repeated_negative_zero",
    )

    assert negative_zero is not positive_zero
    assert repeated_negative_zero is negative_zero
    assert len(serializer.currRegion.currBasicBlock.operators) == 2


def test_constants_are_scoped_to_basic_blocks():
    serializer = _serializer()
    first = serializer.addConst([1], ts.DType.INT8, [0], name="first")

    serializer.startRegion("main")
    serializer.currRegion.addBasicBlock("main")
    second = serializer.addConst([1], ts.DType.INT8, [0], name="second")

    assert second is not first
    assert second.name == "second"


def test_start_region_preserves_path_prefix():
    serializer = _serializer("artifacts")

    serializer.startRegion("other")

    assert serializer.currRegion.pathPrefix == "artifacts"


def test_constant_pool_serialization_is_deterministic():
    def serialize():
        serializer = _serializer()
        serializer.addConst([1], ts.DType.INT8, [0], name="first")
        serializer.addConst([1], ts.DType.INT8, [0], name="duplicate")
        serializer.addConst([1], ts.DType.INT8, [1], name="second")
        return bytes(serializer.serialize())

    assert serialize() == serialize()
