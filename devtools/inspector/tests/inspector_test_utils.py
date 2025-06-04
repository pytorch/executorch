# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvlLinearModel(nn.Module):
    """
    A neural network model with a convolutional layer followed by a linear layer.
    """

    def __init__(self):
        super(ConvlLinearModel, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.conv_layer.weight = nn.Parameter(
            torch.tensor([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]])
        )
        self.conv_layer.bias = nn.Parameter(torch.tensor([0.0]))

        self.linear_layer = nn.Linear(in_features=4, out_features=2)
        self.linear_layer.weight = nn.Parameter(
            torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        )
        self.linear_layer.bias = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.additional_bias = nn.Parameter(
            torch.tensor([0.5, -0.5]), requires_grad=False
        )
        self.scale_factor = nn.Parameter(torch.tensor([2.0, 0.5]), requires_grad=False)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        x = x + self.additional_bias
        x = x - 0.1
        x = x * self.scale_factor
        x = x / (self.scale_factor + 1.0)
        x = F.relu(x)
        x = torch.sigmoid(x)
        output1, output2 = torch.split(x, 1, dim=1)
        return output1, output2

    @staticmethod
    def get_input():
        """
        Returns the pre-defined input tensor for this model.
        """
        return torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

    @staticmethod
    def get_expected_intermediate_outputs():
        """
        Returns the expected outputs of the debug handles and intermediate output mapping for this model for the given input.
        """
        return {
            (10,): torch.tensor([[[[7.7000, 6.7000], [4.7000, 3.7000]]]]),
            (11,): torch.tensor([[7.7000, 6.7000, 4.7000, 3.7000]]),
            (12,): torch.tensor(
                [
                    [0.1000, 0.5000],
                    [0.2000, 0.6000],
                    [0.3000, 0.7000],
                    [0.4000, 0.8000],
                ]
            ),
            (13,): torch.tensor([[5.0000, 14.1200]]),
            (14,): torch.tensor([[5.5000, 13.6200]]),
            (15,): torch.tensor([[5.4000, 13.5200]]),
            (16,): torch.tensor([[10.8000, 6.7600]]),
            (17,): torch.tensor([3.0000, 1.5000]),
            (18,): torch.tensor([[3.6000, 4.5067]]),
            (19,): torch.tensor([[3.6000, 4.5067]]),
            (20,): torch.tensor([[0.9734, 0.9891]]),
            (21,): [torch.tensor([[0.9734]]), torch.tensor([[0.9891]])],
        }


# Global model registry
model_registry = {
    "ConvLinearModel": ConvlLinearModel,
    # Add new models here
}


def check_if_final_outputs_match(model_name, actual_outputs_with_handles):
    """
    Checks if the actual outputs match the expected outputs for the specified model.
    Returns True if all outputs match, otherwise returns False.
    """
    model_instance = model_registry[model_name]
    expected_outputs_with_handles = model_instance.get_expected_intermediate_outputs()
    if len(actual_outputs_with_handles) != len(expected_outputs_with_handles):
        return False
    for debug_handle, expected_output in expected_outputs_with_handles.items():
        actual_output = actual_outputs_with_handles.get(debug_handle)
        if actual_output is None:
            return False
        if isinstance(expected_output, list):
            if not isinstance(actual_output, list):
                return False
            if len(actual_output) != len(expected_output):
                return False
            for actual, expected in zip(actual_output, expected_output):
                if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                    return False
        else:
            if not torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-5):
                return False
    return True
