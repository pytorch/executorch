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

from examples.export.export_example import export_to_pte, save_pte_program


# example model
class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul4.default(a)


def main():
    m = Model()
    input = torch.randn(2, 3)

    # capture and lower
    model_name = "custom_ops_2"
    buffer = export_to_pte(model_name, m, (input,))
    save_pte_program(buffer, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--so_library",
        required=True,
        help="Provide path to so library. E.g., cmake-out/examples/custom_ops/libcustom_ops_aot_lib.so",
    )
    args = parser.parse_args()
    # See if we have custom op my_ops::mul4.out registered
    has_out_ops = True
    try:
        op = torch.ops.my_ops.mul4.out
    except AttributeError:
        print("No registered custom op my_ops::mul4.out")
        has_out_ops = False
    if not has_out_ops:
        if args.so_library:
            torch.ops.load_library(args.so_library)
        else:
            raise RuntimeError(
                "Need to specify shared library path to register custom op my_ops::mul4.out into"
                "EXIR. The required shared library is defined as `custom_ops_aot_lib` in "
                "examples/custom_ops/CMakeLists.txt if you are using CMake build, or `custom_ops_aot_lib_2` in "
                "examples/custom_ops/targets.bzl for buck2. One example path would be cmake-out/examples/custom_ops/"
                "libcustom_ops_aot_lib.[so|dylib]."
            )
    print(args.so_library)

    main()
