# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of showcasing registering custom operator through torch library API."""
import torch
from examples.portable.scripts.export import export_to_exec_prog, save_pte_program

from executorch.exir import EdgeCompileConfig
from torch.library import impl, Library

my_op_lib = Library("my_ops", "DEF")

# registering an operator that multiplies input tensor by 3 and returns it.
my_op_lib.define("mul3(Tensor input) -> Tensor")  # should print 'mul3'


@impl(my_op_lib, "mul3", dispatch_key="CompositeExplicitAutograd")
def mul3_impl(a: torch.Tensor) -> torch.Tensor:
    return a * 3


# registering the out variant.
my_op_lib.define(
    "mul3.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)"
)  # should print 'mul3.out'


@impl(my_op_lib, "mul3.out", dispatch_key="CompositeExplicitAutograd")
def mul3_out_impl(a: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(a)
    out.mul_(3)
    return out


# example model
class Model(torch.nn.Module):
    def forward(self, a):
        return torch.ops.my_ops.mul3.default(a)


def main():
    m = Model()
    input = torch.randn(2, 3)
    # capture and lower
    model_name = "custom_ops_1"
    prog = export_to_exec_prog(
        m,
        (input,),
        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    save_pte_program(prog, model_name)


if __name__ == "__main__":
    main()
