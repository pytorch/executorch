# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest
from unittest.mock import patch

import torch

from executorch.examples.models.model_factory import EagerModelFactory


class _FakeModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def get_eager_model(self):
        return self

    def get_example_inputs(self):
        return (torch.ones(1),)

    def get_example_kwarg_inputs(self):
        return {"input_pos": torch.tensor([0])}

    def get_dynamic_shapes(self):
        return {"tokens": None}


class ModelFactoryTest(unittest.TestCase):
    def test_create_model_imports_from_package_root(self) -> None:
        fake_module = types.SimpleNamespace(AddModule=_FakeModel)

        with patch(
            "executorch.examples.models.model_factory.importlib.import_module",
            return_value=fake_module,
        ) as mock_import:
            model, example_inputs, example_kwarg_inputs, dynamic_shapes = (
                EagerModelFactory.create_model("toy_model", "AddModule", foo="bar")
            )

        mock_import.assert_called_once_with("executorch.examples.models.toy_model")
        self.assertEqual(model.kwargs, {"foo": "bar"})
        self.assertEqual(len(example_inputs), 1)
        self.assertTrue(torch.equal(example_inputs[0], torch.ones(1)))
        self.assertEqual(set(example_kwarg_inputs.keys()), {"input_pos"})
        self.assertTrue(
            torch.equal(example_kwarg_inputs["input_pos"], torch.tensor([0]))
        )
        self.assertEqual(dynamic_shapes, {"tokens": None})

    def test_create_model_loads_real_toy_model(self) -> None:
        model, example_inputs, example_kwarg_inputs, dynamic_shapes = (
            EagerModelFactory.create_model("toy_model", "AddModule")
        )

        self.assertEqual(type(model).__name__, "AddModule")
        self.assertEqual(len(example_inputs), 2)
        self.assertIsNone(example_kwarg_inputs)
        self.assertIsNone(dynamic_shapes)
