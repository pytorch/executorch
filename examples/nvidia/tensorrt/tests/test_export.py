# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export correctness tests — exports each model and verifies against eager PyTorch.

    buck test $GPU_FLAGS fbcode//executorch/examples/nvidia/tensorrt:test_export
    buck test $GPU_FLAGS fbcode//executorch/examples/nvidia/tensorrt:test_export -- test_add
"""

import io
import logging
import unittest

import torch

from executorch.examples.nvidia.tensorrt.export import (
    _verify_correctness,
    export_model,
)


def _export_and_verify(model_name: str) -> None:
    """Export a model and verify its outputs against eager PyTorch."""
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    with torch.no_grad():
        model, example_inputs, exec_prog = export_model(
            model_name, ".", True, logger,
        )
        buf = io.BytesIO()
        exec_prog.write_to_file(buf)
        _verify_correctness(model_name, model, example_inputs, buf.getvalue(), logger)


class ExportCorrectnessTest(unittest.TestCase):
    """Export each model with TensorRT and verify outputs against eager PyTorch."""

    def test_add(self) -> None:
        _export_and_verify("add")

    def test_add_mul(self) -> None:
        _export_and_verify("add_mul")

    def test_conv1d(self) -> None:
        _export_and_verify("conv1d")

    def test_dl3(self) -> None:
        _export_and_verify("dl3")

    def test_edsr(self) -> None:
        _export_and_verify("edsr")

    def test_emformer_join(self) -> None:
        _export_and_verify("emformer_join")

    def test_emformer_transcribe(self) -> None:
        _export_and_verify("emformer_transcribe")

    def test_ic3(self) -> None:
        _export_and_verify("ic3")

    def test_ic4(self) -> None:
        _export_and_verify("ic4")

    def test_linear(self) -> None:
        _export_and_verify("linear")

    def test_mul(self) -> None:
        _export_and_verify("mul")

    def test_mv2(self) -> None:
        _export_and_verify("mv2")

    def test_mv3(self) -> None:
        _export_and_verify("mv3")

    def test_resnet18(self) -> None:
        _export_and_verify("resnet18")

    def test_resnet50(self) -> None:
        _export_and_verify("resnet50")

    def test_softmax(self) -> None:
        _export_and_verify("softmax")

    def test_w2l(self) -> None:
        _export_and_verify("w2l")
