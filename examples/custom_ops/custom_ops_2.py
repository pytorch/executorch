# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of showcasing registering custom operator through loading a shared
library that calls PyTorch C++ op registration API.
"""

import argparse

import torch

from examples.export.export_example import export_to_pte

# example model
class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul4.default(a)


def main():
    m = Model()
    input = torch.randn(2, 3)

    # capture and lower
    export_to_pte("custom_ops_2", m, (input,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--so_library",
        required=True,
        help="Provide path to so library. E.g., cmake-out/examples/custom_ops/libcustom_ops_aot_lib.so",
    )
    args = parser.parse_args()
    torch.ops.load_library(args.so_library)
    print(args.so_library)

    main()
