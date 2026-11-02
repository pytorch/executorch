# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run an exported FP32 classifier using only executorch-native and NumPy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from executorch_native import Model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--categories", type=Path)
    parser.add_argument("--method", default="forward")
    parser.add_argument("--engine", choices=("ggml", "vulkan"), default="ggml")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    model = Model.load(args.package)
    info = model.method_info(args.method)
    if len(info.inputs) != 1 or len(info.outputs) != 1:
        parser.error("expected one input and one output")
    if info.inputs[0].dtype != np.float32 or info.outputs[0].dtype != np.float32:
        parser.error("this example requires FP32 inputs and outputs")
    image = np.fromfile(args.input, dtype=np.float32).reshape(info.inputs[0].shape)
    session = model.create_session(engine=args.engine)
    (output,) = session.run([image], method=args.method)
    logits = output.reshape(-1)
    labels = args.categories.read_text().splitlines() if args.categories else []
    print(f"engine: {args.engine}; method: {args.method}")
    for index in np.argsort(logits)[::-1][:5]:
        label = labels[index] if index < len(labels) else f"class_{index}"
        print(f"{index:4d} {label:28s} {logits[index]:10.4f}")
    if args.expected:
        reference = np.fromfile(args.expected, dtype=np.float32).reshape(output.shape)
        np.testing.assert_allclose(output, reference, rtol=args.rtol, atol=args.atol)
        print("PASS: logits match the reference")


if __name__ == "__main__":
    main()
