# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from executorch.examples.models.mlperf_tiny.streaming_wakeword import StreamingWakeWord
from executorch.extension.export_util.utils import export_to_exec_prog
from executorch.runtime import Runtime


@pytest.fixture
@torch.no_grad()
def model():
    torch.manual_seed(42)
    model = StreamingWakeWord().eval()
    for block in model.blocks:
        block.bn.bias.uniform_(-0.1, 0.1)
        block.bn.running_mean.uniform_(-0.2, 0.2)
        block.bn.running_var.uniform_(0.5, 1.5)
    model.classifier.weight.mul_(32)
    return model


@pytest.mark.parametrize("apply_softmax", [False, True])
@torch.no_grad()
def test_stream_and_reload(model, apply_softmax):
    model.apply_softmax = apply_softmax
    frames = torch.randn(100, *model.FRAME_SHAPE)
    program = export_to_exec_prog(model, (frames[0],))
    assert all(not plan.delegates for plan in program.executorch_program.execution_plan)
    buffer = program.buffer
    expected = [model(frame).clone() for frame in frames]

    for _ in range(2):
        loaded = Runtime.get().load_program(buffer)
        assert loaded.method_names == {"forward"}
        forward = loaded.load_method("forward")
        for frame, output in zip(frames, expected):
            torch.testing.assert_close(
                forward.execute((frame,))[0], output, atol=1e-6, rtol=1e-5
            )


@torch.no_grad()
def test_independent_programs(model):
    frames = torch.randn(80, *model.FRAME_SHAPE)
    buffer = export_to_exec_prog(model, (frames[0],)).buffer
    first = Runtime.get().load_program(buffer)
    second = Runtime.get().load_program(buffer)
    first_forward = first.load_method("forward")
    second_forward = second.load_method("forward")
    other_reference = copy.deepcopy(model)

    for frame in frames:
        torch.testing.assert_close(
            first_forward.execute((frame,))[0], model(frame), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            second_forward.execute((-frame,))[0],
            other_reference(-frame),
            atol=1e-6,
            rtol=1e-5,
        )
