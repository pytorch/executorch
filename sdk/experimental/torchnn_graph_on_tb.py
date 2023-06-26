# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

# Simple torch nn module with just one aten add operator
class AddModule(nn.Module):
    def __init__(self):
        super(AddModule, self).__init__()

    def forward(self, x, y, alpha=1):
        return torch.add(x, alpha * y)


# Create sample inputs
input_1 = torch.randn(3, 4)
input_2 = torch.randn(3, 4)
alpha = torch.randn(1)

add_module = AddModule()

# Write to TensorBoard
writer = SummaryWriter()
# Call the add_graph function that takes a torch nn module and sample inputs
writer.add_graph(add_module, [input_1, input_2, alpha])
writer.close()
