# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of showcasing registering custom operator through torch library API."""
import torch

from examples.export.export_example import export_to_ff

torch.ops.load_library("cmake-out/examples/custom_ops/libcustom_ops_aot_lib.so")

# example model
class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul4.default(a)


def main():
    m = Model()
    input = torch.randn(2, 3)
    # capture and lower
    export_to_ff("custom_ops_2", m, (input,))


if __name__ == "__main__":
    main()
