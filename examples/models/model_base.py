# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch


class EagerModelBase(ABC):
    """
    Abstract base class for eager mode models.

    This abstract class defines the interface that eager mode model classes should adhere to.
    Eager mode models inherit from this class to ensure consistent behavior and structure.
    """

    @abstractmethod
    def __init__(self):
        """
        Constructor for EagerModelBase.

        This initializer may be overridden in derived classes to provide additional setup if needed.
        """
        pass

    @abstractmethod
    def get_eager_model(self) -> torch.nn.Module:
        """
        Abstract method to return an eager PyTorch model instance.

        Returns:
            nn.Module: An instance of a PyTorch model, suitable for eager execution.
        """
        raise NotImplementedError("get_eager_model")

    @abstractmethod
    def get_example_inputs(self):
        """
        Abstract method to provide example inputs for the model.

        Returns:
            Any: Example inputs that can be used for testing and tracing.
        """
        raise NotImplementedError("get_example_inputs")
