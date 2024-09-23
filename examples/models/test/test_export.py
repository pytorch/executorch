# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.examples.models import MODEL_NAME_TO_MODEL
from executorch.examples.models.model_factory import EagerModelFactory

from executorch.extension.export_util.utils import export_to_edge

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)


class ExportTest(unittest.TestCase):
    def collect_executorch_and_eager_outputs(
        self,
        eager_model: torch.nn.Module,
        example_inputs,
    ):
        """
        Compares the output of the given eager mode PyTorch model with the output
        of the equivalent executorch model, both provided with example inputs.
        Returns a tuple containing the outputs of the eager mode model and the executorch mode model.
        """
        eager_model = eager_model.eval()
        model = torch.export.export_for_training(eager_model, example_inputs).module()
        edge_model = export_to_edge(model, example_inputs)

        executorch_prog = edge_model.to_executorch()

        pte_model = _load_for_executorch_from_buffer(executorch_prog.buffer)

        with torch.no_grad():
            eager_output = eager_model(*example_inputs)
        with torch.no_grad():
            executorch_output = pte_model.run_method("forward", example_inputs)

        return (eager_output, executorch_output)

    def validate_tensor_allclose(
        self, eager_output, executorch_output, rtol=1e-5, atol=1e-5
    ):
        self.assertTrue(
            isinstance(eager_output, type(executorch_output)),
            f"Outputs are not of the same type: eager type: {type(eager_output)}, executorch type: {type(executorch_output)}",
        )
        self.assertTrue(
            len(eager_output) == len(executorch_output),
            f"len(eager_output)={len(eager_output)}, len(executorch_output)={len(executorch_output)}",
        )
        result = True
        for i in range(len(eager_output)):
            result = torch.allclose(
                eager_output[i],
                executorch_output[i],
                rtol=rtol,
                atol=atol,
            )
            if not result:
                print(f"eager output[{i}]: {eager_output[i]}")
                print(f"executorch output[{i}]: {executorch_output[i]}")
                break
        return self.assertTrue(result)

    def test_mv3_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["mv3"]
        )
        eager_output, executorch_output = self.collect_executorch_and_eager_outputs(
            eager_model, example_inputs
        )
        # TODO(T166083470): Fix accuracy issue
        self.validate_tensor_allclose(
            eager_output, executorch_output[0], rtol=1e-3, atol=1e-5
        )

    def test_mv2_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["mv2"]
        )
        eager_output, executorch_output = self.collect_executorch_and_eager_outputs(
            eager_model, example_inputs
        )
        self.validate_tensor_allclose(eager_output, executorch_output[0])

    def test_vit_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["vit"]
        )
        eager_output, executorch_output = self.collect_executorch_and_eager_outputs(
            eager_model, example_inputs
        )
        # TODO(T166083470): Fix accuracy, detected on Arm64
        self.validate_tensor_allclose(
            eager_output, executorch_output[0], rtol=1e-2, atol=1e-2
        )

    def test_w2l_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["w2l"]
        )
        eager_output, executorch_output = self.collect_executorch_and_eager_outputs(
            eager_model, example_inputs
        )
        self.validate_tensor_allclose(eager_output, executorch_output[0])

    def test_ic3_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["ic3"]
        )
        eager_output, executorch_output = self.collect_executorch_and_eager_outputs(
            eager_model, example_inputs
        )
        # TODO(T166083470): Fix accuracy issue
        self.validate_tensor_allclose(
            eager_output, executorch_output[0], rtol=1e-3, atol=1e-5
        )

    def test_resnet18_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["resnet18"]
        )
        eager_output, executorch_output = self.collect_executorch_and_eager_outputs(
            eager_model, example_inputs
        )
        self.validate_tensor_allclose(eager_output, executorch_output[0])

    def test_resnet50_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["resnet50"]
        )
        eager_output, executorch_output = self.collect_executorch_and_eager_outputs(
            eager_model, example_inputs
        )
        self.validate_tensor_allclose(eager_output, executorch_output[0])

    def test_dl3_export_to_executorch(self):
        eager_model, example_inputs, _ = EagerModelFactory.create_model(
            *MODEL_NAME_TO_MODEL["dl3"]
        )
        eager_output, executorch_output = self.collect_executorch_and_eager_outputs(
            eager_model, example_inputs
        )
        self.validate_tensor_allclose(list(eager_output.values()), executorch_output)
