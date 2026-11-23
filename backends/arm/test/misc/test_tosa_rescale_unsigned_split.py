# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import pytest
import torch
from executorch.backends.arm._passes.tosa_lowering_utils import (
    create_rescale_node,
    rescale_tensor,
)
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch._subclasses.fake_tensor import FakeTensorMode


def _rescale_unsigned_output(output_zp: int, dtype=torch.int8):
    graph = torch.fx.Graph()
    source = graph.placeholder("wide_input")
    with FakeTensorMode():
        source.meta["val"] = torch.empty(7, dtype=torch.int32)
    output = create_rescale_node(
        graph,
        (source, dtype, [1.0], 0, output_zp),
        kwargs={"output_unsigned": True},
        from_node=source,
    )
    graph.output(output)
    graph.lint()
    return [
        node
        for node in graph.nodes
        if node.target == exir_ops.backend.tosa.RESCALE.default
    ]


def test_wide_to_unsigned_int16_rescale_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="Wide-to-unsigned RESCALE only supports int8 output"
    ):
        _rescale_unsigned_output(0, dtype=torch.int16)


def _clip(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _direct_unsigned_rescale(value: int, output_zp: int) -> int:
    return _clip(value + output_zp, 0, 255)


def _split_unsigned_rescale(value: int, output_zp: int) -> int:
    signed = _clip(value + output_zp - 128, -128, 127)
    return _clip(signed - (-128), 0, 255)


def test_wide_to_unsigned_rescale_rebases_to_signed_domain() -> None:
    first, second = _rescale_unsigned_output(output_zp=0)
    assert first.args[0].name == "wide_input"
    assert second.args[0] is first
    assert first.kwargs["output_unsigned"] is False
    assert second.kwargs["output_unsigned"] is True
    assert first.args[4] == -128
    assert second.args[3] == -128
    assert second.args[4] == 0


@pytest.mark.parametrize("output_zp", [0, 17, 255])
def test_tensor_helper_legalizes_wide_to_unsigned(output_zp):
    with TosaLoweringContext(
        TosaSpecification.create_from_string("TOSA-1.1+INT")
    ), FakeTensorMode():
        output = rescale_tensor(
            torch.empty(7, dtype=torch.int32),
            torch.int8,
            [1.0],
            0,
            output_zp,
            output_unsigned=True,
        )
    assert output.shape == (7,)
    assert output.dtype == torch.int8


@pytest.mark.parametrize("output_zp", [0, 17, 255])
def test_wide_to_unsigned_rescale_preserves_unsigned_range(output_zp: int) -> None:
    first, second = _rescale_unsigned_output(output_zp)
    assert first.args[4] == output_zp - 128
    assert second.args[3] == -128
    assert second.args[4] == 0
    for value in (-1, 0, 1, 126, 127, 128, 200, 254, 255, 256):
        assert _split_unsigned_rescale(value, output_zp) == _direct_unsigned_rescale(
            value, output_zp
        )
