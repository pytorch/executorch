# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the MLPerf Tiny streaming wake word model, int8 through the Cadence
quantizer, as a bundled program whose test cases are consecutive frames.

The model keeps a history of past frames, so the test cases only match when run
in order from zeroed state. The expected outputs come from the PyTorch model
after quantization. The weights are random, so the match shows the kernels are
correct, not that the model detects anything.
"""

import argparse
from pathlib import Path

import torch
from executorch.backends.cadence.aot.compiler import (
    _lower_ep_to_cadence_gen_etrecord,
    apply_pre_edge_transform_passes,
    get_fake_quant_model,
)
from executorch.backends.cadence.aot.quantizer.quantizer import CadenceDefaultQuantizer
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.examples.models.mlperf_tiny.streaming_wakeword import StreamingWakeWord


def random_frames(count, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = StreamingWakeWord.FRAME_SHAPE
    return [torch.rand(shape, generator=generator) * 2 - 1 for _ in range(count)]


def reset_history(module):
    for name in StreamingWakeWord.HISTORY_NAMES:
        module.get_buffer(name).zero_()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    model = StreamingWakeWord(apply_softmax=True).eval()
    # Spread the random weights so that every layer's activations vary.
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.uniform_(module.weight, 0.5, 1.5)
            torch.nn.init.normal_(module.bias, std=0.1)
            module.running_mean.normal_(std=0.1)
            module.running_var.uniform_(0.5, 1.5)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.5)
            torch.nn.init.normal_(module.bias, std=0.5)

    example = tuple(random_frames(1, seed=0))
    quantizer = CadenceDefaultQuantizer()
    quantized = get_fake_quant_model(
        model,
        example,
        quantizer,
        calibration_data=[(frame,) for frame in random_frames(64, seed=2)],
    )

    frames = random_frames(args.frames, seed=1)
    reset_history(quantized)
    cases = [MethodTestCase((frame,), quantized(frame)) for frame in frames]
    reset_history(quantized)

    program = _lower_ep_to_cadence_gen_etrecord(
        apply_pre_edge_transform_passes(
            torch.export.export(quantized, example, strict=True), quantizer
        )
    )
    bundle = BundledProgram(program, [MethodTestSuite("forward", cases)])
    args.output.write_bytes(serialize_from_bundled_program_to_flatbuffer(bundle))
    print(f"Exported {args.output} with {len(cases)} frames")


if __name__ == "__main__":
    main()
