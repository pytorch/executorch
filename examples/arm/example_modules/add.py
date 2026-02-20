# All rights reserved.
# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# Example of an external model for the Arm AOT Compiler
#
# Example of an external Python file to be used as a module by the `run.sh`
# (and the `aot_arm_compiler.py`) scripts in `examples/arm` directory.
#
# Just pass the path of the `add.py` file as `--model_name`
#
# These two variables are picked up by the `aot_arm_compiler.py` and used:
# `ModelUnderTest` should be a `torch.nn.module` instance.
# `ModelInputs` should be a tuple of inputs to the forward function.
#

import torch

b = 2


class myModelAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x + b


ModelUnderTest = myModelAdd()
ModelInputs = (torch.ones(5),)
