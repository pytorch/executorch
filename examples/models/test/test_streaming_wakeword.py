# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.mlperf_tiny.streaming_wakeword import StreamingWakeWord
from executorch.examples.models.model_factory import EagerModelFactory
from torch import nn


@pytest.fixture
def model():
    torch.manual_seed(42)
    model = StreamingWakeWord().eval()
    with torch.no_grad():
        for block in model.blocks:
            block.bn.bias.uniform_(-0.1, 0.1)
            block.bn.running_mean.uniform_(-0.2, 0.2)
            block.bn.running_var.uniform_(0.5, 1.5)
    return model


@pytest.mark.parametrize("signal", ["random", "zero", "impulse", "alternating"])
@torch.no_grad()
def test_matches_full_windows(model, signal):
    layers = []
    for block in model.blocks:
        layers.extend(
            copy.deepcopy(layer)
            for layer in (block.depthwise, block.pointwise, block.bn, block.relu)
        )
    reference = nn.Sequential(*layers, nn.Flatten(1), copy.deepcopy(model.classifier))
    frames = torch.randn(1, 40, 70, 1)
    if signal == "zero":
        frames.zero_()
    elif signal == "impulse":
        frames.zero_()
        frames[:, :, 30, :] = 1
    elif signal == "alternating":
        frames[:, :, ::2, :] = 1
        frames[:, :, 1::2, :] = -1

    for index in range(frames.shape[2]):
        actual = model(frames[:, :, index : index + 1, :])
        if index >= model.RECEPTIVE_FIELD - 1:
            window = frames[:, :, index - model.RECEPTIVE_FIELD + 1 : index + 1, :]
            torch.testing.assert_close(actual, reference(window), atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_reset_replay_and_independent_streams(model):
    inputs = [torch.randn(model.FRAME_SHAPE) for _ in range(100)]
    independent = StreamingWakeWord().eval()
    independent.load_state_dict(model.state_dict())
    expected = [model(frame).clone() for frame in inputs]

    for frame, result in zip(inputs[:35], expected[:35]):
        torch.testing.assert_close(independent(frame), result)

    model.reset()
    assert all(torch.count_nonzero(block.history) == 0 for block in model.blocks)
    for frame, result in zip(inputs[35:], expected[35:]):
        torch.testing.assert_close(independent(frame), result)

    for frame, result in zip(inputs, expected):
        torch.testing.assert_close(model(frame), result)


@torch.no_grad()
def test_factory_returns_probabilities():
    example, inputs, _, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL["streaming_wakeword"]
    )
    assert not example.training
    output = example(*inputs)
    assert output.shape == (1, 3)
    torch.testing.assert_close(output.sum(dim=-1), torch.ones(1))


def test_rejects_multi_frame_input(model):
    with pytest.raises(ValueError, match="LFBE frame"):
        model(torch.zeros(1, 40, 2, 1))
