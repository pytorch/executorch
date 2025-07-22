# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import typing

import torch


class CIFAR10Model(torch.nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(CIFAR10Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        """
        The forward function takes the input image and applies the
        convolutional layers and the fully connected layers to
        extract the features and classify the image respectively.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ModelWithLoss(torch.nn.Module):
    """
    NOTE: A wrapper class that combines a model and the loss function
    into a single module. Used for capturing the entire computational
    graph, i.e. forward pass and the loss calculation, to be captured
    during export. Our objective is to enable on-device training, so
    the loss calculation should also be included in the exported graph.
    """

    def __init__(
        self, model: torch.nn.Module, criterion: torch.nn.CrossEntropyLoss
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass through the model
        output = self.model(x)
        # Calculate loss
        loss = self.criterion(output, target)
        # Return loss and predicted class
        return loss, output.detach().argmax(dim=1)
