# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of showcasing registering custom operator through loading a shared
library that calls PyTorch C++ op registration API.
"""

import torch

from examples.export.export_example import export_to_pte


# example model
class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul4.default(a)


def main():
    m = Model()
    input = torch.randn(2, 3)
    # load shared library
    from sys import platform

    if platform == "linux" or platform == "linux2":
        extension = ".so"
    elif platform == "darwin":
        extension = ".dylib"
    else:
        raise RuntimeError(f"Unsupported platform {platform}")
    torch.ops.load_library(
        f"cmake-out/examples/custom_ops/libcustom_ops_aot_lib{extension}"
    )
    # capture and lower
    export_to_pte("custom_ops_2", m, (input,))


if __name__ == "__main__":
    main()
