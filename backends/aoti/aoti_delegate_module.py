# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from torch.export import ExportedProgram
from torch.utils import _pytree as pytree

# TODO: These should probably be in pytorch


class AOTInductorRunnerWrapper(torch.nn.Module):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, aoti_runner) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self.aoti_runner = aoti_runner

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, *flat_inputs):
        return self.aoti_runner.run(flat_inputs)


class AOTIDelegateModule(torch.nn.Module):
    """
    This module is the primary artifact produced by AOTInductor lowering.
    It is eagerly runnable in Python and traceable by torch.export.
    It also contains all necessary information and metadata to be pacakged and consumed
    by the delegate executor in runtime later.

    """

    def __init__(self, exported_program: ExportedProgram, so_path: str) -> None:
        super().__init__()
        self.so_path = so_path
        self.exported_program = exported_program
        self.exported_program.graph_module.recompile()

        # register parameters
        for name, parameter in self.exported_program.named_parameters():
            normalized_name = name.replace(".", "_")
            self.register_parameter(normalized_name, parameter)

        # register buffers
        non_persistent_buffer_names = (
            exported_program.graph_signature.non_persistent_buffers
        )
        for name, buffer in self.exported_program.named_buffers():
            normalized_name = name.replace(".", "_")
            if name in non_persistent_buffer_names:
                self.register_buffer(normalized_name, buffer, persistent=False)
            else:
                self.register_buffer(normalized_name, buffer, persistent=True)

        # handle tensor constants
        self.constant_names: list[str] = []
        for name, constant in self.exported_program.tensor_constants.items():
            # skip non-persistent buffers
            if name in non_persistent_buffer_names:
                continue
            normalized_name = name.replace(".", "_")
            setattr(self, normalized_name, constant)
            self.constant_names.append(normalized_name)

        # pyre-ignore[4]: Missing attribute annotation
        # pyre-ignore[16]: Undefined attribute
        # TODO: CPU only for now. Add GPU
        self.engine = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
        self.aoti_runner_wrapper = AOTInductorRunnerWrapper(self.engine)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, *inputs):
        weights_args = [
            *self.parameters(),
            *self.buffers(),
        ] + [getattr(self, const_name) for const_name in self.constant_names]

        flat_inputs = pytree.tree_flatten((inputs, {}))[0]
        flat_outputs = torch._higher_order_ops.aoti_call_delegate(
            self.aoti_runner_wrapper,
            self.exported_program.graph_module,
            weights_args,
            flat_inputs,
        )
        return pytree.tree_unflatten(
            flat_outputs, self.exported_program.call_spec.out_spec
        )
