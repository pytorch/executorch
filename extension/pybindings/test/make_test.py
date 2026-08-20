# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, Optional, Tuple

import torch
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager, to_edge
from torch.export import export


class ModuleAdd(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleAdd, self).__init__()

    def forward(self, x, y):
        return x + y

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(2, 2), torch.ones(2, 2))


class ModuleChannelsLast(torch.nn.Module):
    """The module to serialize and execute."""

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        )

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(1, 2, 3, 4).to(memory_format=torch.channels_last),)


class ModuleChannelsLastInDefaultOut(torch.nn.Module):
    """The module to serialize and execute."""

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            scale_factor=2,
            mode="nearest",
        ).to(memory_format=torch.contiguous_format)

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(1, 2, 3, 4).to(memory_format=torch.channels_last),)


class ModuleMulti(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleMulti, self).__init__()

    def forward(self, x, y):
        return x + y

    def forward2(self, x, y):
        return x + y + 1

    def get_methods_to_export(self):
        return ("forward", "forward2")

    def get_inputs(self):
        return (torch.ones(2, 2), torch.ones(2, 2))


class ModuleAddSingleInput(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleAddSingleInput, self).__init__()

    def forward(self, x):
        return x + x

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(2, 2),)


class ModuleAddConstReturn(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleAddConstReturn, self).__init__()
        self.state = torch.ones(2, 2)

    def forward(self, x):
        return x + self.state, self.state

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(2, 2),)


class ModuleAddWithAttributes(torch.nn.Module):
    """The module to serialize and execute."""

    def __init__(self):
        super(ModuleAddWithAttributes, self).__init__()
        self.register_buffer("state", torch.zeros(2, 2))

    def forward(self, x, y):
        self.state.add_(1)
        return x + y + self.state

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.ones(2, 2), torch.ones(2, 2))


class ModuleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor):
        return self.linear(x)

    def get_methods_to_export(self):
        return ("forward",)

    def get_inputs(self):
        return (torch.randn(3),)


def create_program(
    eager_module: torch.nn.Module,
    et_config: Optional[ExecutorchBackendConfig] = None,
) -> Tuple[ExecutorchProgramManager, Tuple[Any, ...]]:
    """Returns an executorch program based on ModuleAdd, along with inputs."""

    # Trace the test module and create a serialized ExecuTorch program.
    # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
    #  is not a function.
    inputs = eager_module.get_inputs()
    input_map = {}
    # pyre-fixme[29]: `Union[torch._tensor.Tensor, torch.nn.modules.module.Module]`
    #  is not a function.
    for method in eager_module.get_methods_to_export():
        input_map[method] = inputs

    class WrapperModule(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    exported_methods = {}
    # These cleanup passes are required to convert the `add` op to its out
    # variant, along with some other transformations.
    for method_name, method_input in input_map.items():
        wrapped_mod = WrapperModule(getattr(eager_module, method_name))
        exported_methods[method_name] = export(wrapped_mod, method_input, strict=True)

    exec_prog = to_edge(exported_methods).to_executorch(config=et_config)

    # Create the ExecuTorch program from the graph.
    exec_prog.dump_executorch_program(verbose=True)
    return (exec_prog, inputs)
