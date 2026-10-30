# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check cached export preparation without an external model checkout."""

from types import SimpleNamespace

import pytest
import torch

from examples.arm.QAT_example import qat_loop
from examples.arm.QAT_example.qat_loop import extract_encoder, prepare_cached_model


class _FlowNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = torch.nn.Conv2d(3, 4, 1)
        self.blend = torch.nn.Conv2d(4, 3, 1)


class _CachedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flownet = _FlowNet()

    def forward(self, previous, current, features):
        # The image depends only on cached features, so the QAT test detects
        # whether gradients reach the encoder called for the previous frame.
        image = torch.sigmoid(self.flownet.blend(features))
        return image, self.flownet.encode(current)


@pytest.mark.parametrize("io_quantization", ("int8", "uint8"))
def test_cache_quantization_and_bootstrap(io_quantization):
    torch.manual_seed(2026)
    frames = [torch.rand(1, 3, 8, 8) for _ in range(3)]
    model = prepare_cached_model(
        _CachedModel().eval(),
        [((frames[0], frames[1]), frames[2])],
        is_qat=False,
        io_quantization=io_quantization,
        steps=1,
        lr=1e-3,
    )
    cache_input = list(model.graph.find_nodes(op="placeholder"))[2]
    cache_quantize = next(iter(cache_input.users))
    cache_output = next(iter(model.graph.find_nodes(op="output"))).args[0][1]
    assert cache_quantize.args[1:] == cache_output.args[1:]
    assert cache_quantize.args[-1] == torch.int8
    encoder = extract_encoder(model)
    with torch.no_grad():
        for seed_value in (0.0, 127.0):
            _, features = model(
                frames[0], frames[0], torch.full((1, 4, 8, 8), seed_value)
            )
            torch.testing.assert_close(features, encoder(frames[0]), rtol=0, atol=0)
            for previous, current in zip(frames, frames[1:]):
                expected, _ = model(previous, current, encoder(previous))
                actual, features = model(previous, current, features)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(features, encoder(current), rtol=0, atol=0)


def test_qat_backpropagates_through_previous_frame_encoder():
    torch.manual_seed(2026)
    previous = torch.rand(1, 3, 8, 8, requires_grad=True)
    current = torch.rand(1, 3, 8, 8)
    target = torch.zeros(1, 3, 8, 8)
    prepare_cached_model(
        _CachedModel().eval(),
        [((previous, current), target)],
        is_qat=True,
        io_quantization="uint8",
        steps=1,
        lr=1e-3,
    )
    assert previous.grad is not None
    assert torch.isfinite(previous.grad).all()
    assert previous.grad.abs().sum() > 0


def test_cached_runtime_primes_with_each_frames_input_qparams():
    frames = (torch.full((1, 3, 8, 8), 0.7), torch.full((1, 3, 8, 8), 0.9))
    seed = torch.zeros(1, 4, 8, 8)
    qparams = [
        {"scale": 0.1, "zero_point": 2, "qmin": 0, "qmax": 255, "dtype": "torch.uint8"},
        {"scale": 0.2, "zero_point": 9, "qmin": 0, "qmax": 255, "dtype": "torch.uint8"},
        {
            "scale": 0.25,
            "zero_point": -3,
            "qmin": -127,
            "qmax": 127,
            "dtype": "torch.int8",
        },
    ]
    calls = []

    def execute(inputs):
        calls.append(tuple(tensor.clone() for tensor in inputs))
        return torch.ones_like(inputs[0]), torch.full_like(inputs[2], 17)

    output, runtime_inputs = qat_loop.execute_runtime_artifact(
        SimpleNamespace(execute=execute),
        qparams,
        [qparams[0], qparams[2]],
        frames,
        cache=seed,
    )
    assert len(calls) == 2
    torch.testing.assert_close(
        calls[0][0], qat_loop.quantize_tensor_for_runtime(frames[0], qparams[0])
    )
    torch.testing.assert_close(
        calls[0][1], qat_loop.quantize_tensor_for_runtime(frames[0], qparams[1])
    )
    assert torch.all(calls[0][2] == -3)
    torch.testing.assert_close(
        calls[1][1], qat_loop.quantize_tensor_for_runtime(frames[1], qparams[1])
    )
    assert torch.all(calls[1][2] == 17)
    assert torch.all(runtime_inputs[2] == 17)
    torch.testing.assert_close(output, torch.full_like(frames[0], -0.1))
