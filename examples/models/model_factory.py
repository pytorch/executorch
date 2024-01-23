# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
from typing import Any, Tuple

import torch


class EagerModelFactory:
    """
    A factory class for dynamically creating instances of classes implementing EagerModelBase.
    """

    @staticmethod
    def create_model(
        module_name, model_class_name, **kwargs
    ) -> Tuple[torch.nn.Module, Any, Any]:
        """
        Create an instance of a model class that implements EagerModelBase and retrieve related data.

        Args:
            module_name (str): The name of the module containing the model class.
            model_class_name (str): The name of the model class to create an instance of.

        Returns:
            Tuple[nn.Module, Any]: A tuple containing the eager PyTorch model instance and example inputs,
              and any dynamic shape information for those inputs.

        Raises:
            ValueError: If the provided model class is not found in the module.
        """
        package_prefix = "executorch." if not os.getcwd().endswith("executorch") else ""
        module = importlib.import_module(
            f"{package_prefix}examples.models.{module_name}"
        )

        if hasattr(module, model_class_name):
            model_class = getattr(module, model_class_name)
            model = model_class(**kwargs)
            if hasattr(model, "get_dynamic_shapes"):
                return (
                    model.get_eager_model(),
                    model.get_example_inputs(),
                    model.get_dynamic_shapes(),
                )
            else:
                return model.get_eager_model(), model.get_example_inputs(), None

        raise ValueError(
            f"Model class '{model_class_name}' not found in module '{module_name}'."
        )
