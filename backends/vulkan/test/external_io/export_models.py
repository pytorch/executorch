# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Exports the programs that external_io_test.cpp loads, one <name>.pte each.
# Every input and output has shape SIZES.

import argparse
import os

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.backends.vulkan.serialization.vulkan_graph_schema import (
    VkMemoryLayout,
    VkStorageType,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export

SIZES = (1, 8, 16, 24)


class Affine(torch.nn.Module):
    def forward(self, x):
        return x * 2.0 + 1.0


class Chain(torch.nn.Module):
    def forward(self, y):
        return y * 3.0 - 0.5


class ConvAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, 1, 1)

    def forward(self, x, y):
        return self.conv(x) + y


class Recurrent(torch.nn.Module):
    def forward(self, x, s):
        return 0.9 * s + 0.1 * torch.tanh(x + s)


class TwoDelegates(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, 1, 1)

    def forward(self, x):
        # cumsum has no Vulkan implementation, so it splits the graph in two.
        return self.conv2(torch.cumsum(self.conv1(x), dim=-1) * 0.05)


# Skipping memory metadata tagging makes every tensor take the overrides.
TEXTURE = {
    "storage_type_override": VkStorageType.TEXTURE_3D,
    "memory_layout_override": VkMemoryLayout.TENSOR_CHANNELS_PACKED,
    "skip_tag_memory_metadata": True,
}
BUFFER = {
    "storage_type_override": VkStorageType.BUFFER,
    "memory_layout_override": VkMemoryLayout.TENSOR_WIDTH_PACKED,
    "skip_tag_memory_metadata": True,
}

PROGRAMS = {
    "affine_texture": (Affine, 1, TEXTURE),
    "chain_texture": (Chain, 1, TEXTURE),
    "affine_buffer": (Affine, 1, BUFFER),
    "chain_buffer": (Chain, 1, BUFFER),
    "conv_add": (ConvAdd, 2, TEXTURE),
    "recurrent": (Recurrent, 2, TEXTURE),
    "two_delegates": (TwoDelegates, 1, TEXTURE),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(0)
    for name, (module, num_inputs, options) in PROGRAMS.items():
        inputs = tuple(torch.randn(SIZES) for _ in range(num_inputs))
        program = to_edge_transform_and_lower(
            export(module().eval(), inputs, strict=True),
            compile_config=EdgeCompileConfig(_skip_dim_order=False),
            partitioner=[VulkanPartitioner(options)],
        ).to_executorch()
        with open(os.path.join(args.outdir, name + ".pte"), "wb") as f:
            f.write(program.buffer)


if __name__ == "__main__":
    main()
