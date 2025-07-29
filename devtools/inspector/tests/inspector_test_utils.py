# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir.debug_handle_utils import UNSET_DEBUG_HANDLE


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
        x = x.to(x.dtype)
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
    def get_edge_dialect_expected_intermediate_outputs():
        """
        Returns the expected outputs of the debug handles and intermediate output mapping for edge dialect graph of this model for the given input.
        """
        return {
            (1,): torch.tensor([[[[7.7000, 6.7000], [4.7000, 3.7000]]]]),
            (2,): torch.tensor([[7.7000, 6.7000, 4.7000, 3.7000]]),
            (3,): torch.tensor([[5.0000, 14.1200]]),
            (4,): torch.tensor([[5.5000, 13.6200]]),
            (5,): torch.tensor([[5.4000, 13.5200]]),
            (6,): torch.tensor([[10.8000, 6.7600]]),
            (7,): torch.tensor([3.0000, 1.5000]),
            (8,): torch.tensor([[3.6000, 4.5067]]),
            (9,): torch.tensor([[3.6000, 4.5067]]),
            (10,): torch.tensor([[0.9734, 0.9891]]),
            (11,): [torch.tensor([[0.9734]]), torch.tensor([[0.9891]])],
        }

    @staticmethod
    def get_edge_dialect_expected_debug_handle_to_op_names():
        """
        Returns the expected debug handle and op names mapping for this model for the given input.
        """
        return {
            (1,): ["aten_convolution_default"],
            (2,): ["aten_view_copy_default"],
            (3,): ["aten_permute_copy_default", "aten_addmm_default"],
            (4,): ["aten_add_tensor"],
            (5,): ["aten_sub_tensor"],
            (6,): ["aten_mul_tensor"],
            (7,): ["aten_add_tensor_1"],
            (8,): ["aten_div_tensor"],
            (9,): ["aten_relu_default"],
            (10,): ["aten_sigmoid_default"],
            (11,): ["aten_split_with_sizes_copy_default"],
        }

    @staticmethod
    def get_exported_program_expected_intermediate_outputs():
        """
        Returns the expected outputs of the debug handles and intermediate output mapping for export graph of this model for the given input.
        """
        return {
            (UNSET_DEBUG_HANDLE,): torch.tensor([[5.4000, 13.5200]]),
            (1,): torch.tensor([[[[7.7000, 6.7000], [4.7000, 3.7000]]]]),
            (2,): torch.tensor([[7.7000, 6.7000, 4.7000, 3.7000]]),
            (3,): torch.tensor([[5.0000, 14.1200]]),
            (4,): torch.tensor([[5.5000, 13.6200]]),
            (5,): torch.tensor([[5.4000, 13.5200]]),
            (6,): torch.tensor([[10.8000, 6.7600]]),
            (7,): torch.tensor([3.0000, 1.5000]),
            (8,): torch.tensor([[3.6000, 4.5067]]),
            (9,): torch.tensor([[3.6000, 4.5067]]),
            (10,): torch.tensor([[0.9734, 0.9891]]),
            (11,): [torch.tensor([[0.9734]]), torch.tensor([[0.9891]])],
        }

    @staticmethod
    def get_exported_program_expected_debug_handle_to_op_names():
        """
        Returns the expected debug handle and op name mapping for this model for the given input.
        """
        return {
            (UNSET_DEBUG_HANDLE,): ["_assert_tensor_metadata_default", "to"],
            (1,): ["conv2d"],
            (2,): ["view"],
            (3,): ["linear"],
            (4,): ["add"],
            (5,): ["sub"],
            (6,): ["mul"],
            (7,): ["add_1"],
            (8,): ["div"],
            (9,): ["relu"],
            (10,): ["sigmoid"],
            (11,): ["split"],
        }


# Global model registry
model_registry = {
    "ConvLinearModel": ConvlLinearModel,
    # Add new models here
}


def check_if_intermediate_outputs_match(
    actual_outputs_with_handles, expected_outputs_with_handles
):
    """
    Checks if the actual outputs match the expected outputs for the specified model.
    Returns True if all outputs match, otherwise returns False.
    """

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


def check_if_debug_handle_to_op_names_match(
    actual_debug_handle_to_op_name, expected_debug_handle_to_op_name
):
    """
    Checks if the actual op names match the expected op names for the specified model.
    Returns True if all match, otherwise returns False.
    """
    if len(actual_debug_handle_to_op_name) != len(expected_debug_handle_to_op_name):
        return False
    for debug_handle, expected_op_name in expected_debug_handle_to_op_name.items():
        actual_op_name = actual_debug_handle_to_op_name.get(debug_handle)
        if actual_op_name != expected_op_name:
            return False
    return True
