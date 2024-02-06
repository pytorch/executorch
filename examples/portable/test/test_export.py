# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from typing import Any, Callable

import torch
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

from executorch.examples.portable.utils import export_to_edge

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)


class ExportTest(unittest.TestCase):
    def _assert_eager_lowered_same_result(
        self,
        eager_model: torch.nn.Module,
        example_inputs,
        validation_fn: Callable[[Any, Any], bool],
    ):
        """
        Asserts that the given model has the same result as the eager mode
        lowered model, with example_inputs, validated by validation_fn, which
        takes the eager mode output and ET output, and returns True if they
        match.
        """
        eager_model = eager_model.eval()
        model = torch._export.capture_pre_autograd_graph(eager_model, example_inputs)
        edge_model = export_to_edge(model, example_inputs)

        executorch_prog = edge_model.to_executorch()

        pte_model = _load_for_executorch_from_buffer(executorch_prog.buffer)

        with torch.no_grad():
            eager_output = eager_model(*example_inputs)
        with torch.no_grad():
            executorch_output = pte_model.run_method("forward", example_inputs)

        self.assertTrue(validation_fn(eager_output, executorch_output))

    @staticmethod
    def validate_tensor_allclose(eager_output, executorch_output, rtol=1e-5, atol=1e-5):
        result = torch.allclose(
            eager_output,
            executorch_output[0],
            rtol=rtol,
            atol=atol,
        )
        if not result:
            print(f"eager output: {eager_output}")
            print(f"executorch output: {executorch_output}")
        return result

    def test_mv3_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["mv3"]
        )
        eager_model = eager_model.eval()

        # TODO(T166083470): Fix accuracy issue
        self._assert_eager_lowered_same_result(
            eager_model,
            example_inputs,
            lambda x, y: self.validate_tensor_allclose(x, y, rtol=1e-3, atol=1e-5),
        )

    def test_mv2_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["mv2"]
        )
        eager_model = eager_model.eval()

        self._assert_eager_lowered_same_result(
            eager_model, example_inputs, self.validate_tensor_allclose
        )

    def test_vit_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["vit"]
        )
        eager_model = eager_model.eval()

        self._assert_eager_lowered_same_result(
            eager_model, example_inputs, self.validate_tensor_allclose
        )

    def test_w2l_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["w2l"]
        )
        eager_model = eager_model.eval()

        self._assert_eager_lowered_same_result(
            eager_model, example_inputs, self.validate_tensor_allclose
        )

    def test_ic3_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["ic3"]
        )
        eager_model = eager_model.eval()

        self._assert_eager_lowered_same_result(
            eager_model, example_inputs, self.validate_tensor_allclose
        )

    def test_resnet18_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["resnet18"]
        )
        eager_model = eager_model.eval()

        self._assert_eager_lowered_same_result(
            eager_model, example_inputs, self.validate_tensor_allclose
        )

    def test_resnet50_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["resnet50"]
        )
        eager_model = eager_model.eval()

        self._assert_eager_lowered_same_result(
            eager_model, example_inputs, self.validate_tensor_allclose
        )

    def test_dl3_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["dl3"]
        )
        eager_model = eager_model.eval()

        self._assert_eager_lowered_same_result(
            eager_model, example_inputs, self.validate_tensor_allclose
        )
