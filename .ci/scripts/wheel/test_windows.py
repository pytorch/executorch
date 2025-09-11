#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.examples.models import Backend, Model, MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory
from executorch.examples.xnnpack import MODEL_NAME_TO_OPTIONS
from executorch.examples.xnnpack.quantization.utils import quantize as quantize_xnn
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch_from_buffer,
)
from test_base import ModelTest


def test_model_xnnpack(model: Model, quantize: bool) -> None:
    model_instance, example_inputs, _, _ = EagerModelFactory.create_model(
        *MODEL_NAME_TO_MODEL[str(model)]
    )

    model_instance.eval()
    ref_outputs = model_instance(*example_inputs)

    if quantize:
        quant_type = MODEL_NAME_TO_OPTIONS[str(model)].quantization
        model_instance = torch.export.export_for_training(
            model_instance, example_inputs
        )
        model_instance = quantize_xnn(
            model_instance.module(), example_inputs, quant_type
        )

    lowered = to_edge_transform_and_lower(
        torch.export.export(model_instance, example_inputs),
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    ).to_executorch()

    loaded_model = _load_for_executorch_from_buffer(lowered.buffer)
    et_outputs = loaded_model([*example_inputs])

    if isinstance(ref_outputs, torch.Tensor):
        ref_outputs = (ref_outputs,)

    assert len(ref_outputs) == len(et_outputs)
    for i in range(len(ref_outputs)):
        torch.testing.assert_close(ref_outputs[i], et_outputs[i], atol=1e-4, rtol=1e-5)


def run_tests(model_tests: List[ModelTest]) -> None:
    for model_test in model_tests:
        if model_test.backend == Backend.Xnnpack:
            test_model_xnnpack(model_test.model, quantize=False)
        else:
            raise RuntimeError(f"Unsupported backend {model_test.backend}.")


if __name__ == "__main__":
    run_tests(
        model_tests=[
            ModelTest(
                model=Model.Mv3,
                backend=Backend.Xnnpack,
            ),
        ]
    )
