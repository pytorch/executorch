# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, Dict, List, Sequence

from executorch.exir._warnings import experimental

from executorch.extension.pybindings.portable_lib import (
    _load_for_executorch,
    _load_for_executorch_from_buffer,
    ExecuTorchModule,
)
from torch import Tensor


@experimental("This API is experimental and subject to change without notice.")
class TrainingModule:
    def __init__(self, module: ExecuTorchModule):
        self.model = module

        self.gradients_method_prefix = "__et_training_gradients_index_"
        self.parameters_method_prefix = "__et_training_parameters_index_"
        self.fqn_method_prefix = "__et_training_fqn_"

        self.named_grads = None
        self.named_params = None

    def forward_backward(self, method_name: str, inputs: Sequence[Any]) -> List[Any]:
        # The default ET model returns a large list of outputs that can logically be
        # separated into [user outputs, gradients, parameters]. Can use these metadata
        # methods to slice the list into the correct parts.
        grad_start_idx = self.model.run_method(
            self.gradients_method_prefix + method_name, ()
        )[0]
        params_start_idx = self.model.run_method(
            self.parameters_method_prefix + method_name, ()
        )[0]

        full_outputs = self.model.run_method(method_name, inputs)

        user_outs = full_outputs[:grad_start_idx]
        grads = full_outputs[grad_start_idx:params_start_idx]
        params = full_outputs[params_start_idx:]

        # Important that the outputs are not cloned because we need the optimizer to
        # be able to mutate the actual weights and not clones of them.
        fqn = self.model.run_method(
            self.fqn_method_prefix + method_name, (), clone_outputs=False
        )

        self.named_grads = dict(zip(fqn, grads))
        if self.named_params is None:
            self.named_params = dict(zip(fqn, params))

        return user_outs

    def named_gradients(self) -> Dict[str, Tensor]:
        if self.named_grads is None:
            raise RuntimeError("Must call forward_backward before named_grads")
        return self.named_grads

    def named_parameters(self) -> Dict[str, Tensor]:
        if self.named_grads is None:
            raise RuntimeError(
                "Must call forward_backward before named_params. This will be fixed in a later version"
            )
        return self.named_params


@experimental("This API is experimental and subject to change without notice.")
def _load_for_executorch_for_training(path: str) -> TrainingModule:
    et_module = _load_for_executorch(path)
    return TrainingModule(et_module)


@experimental("This API is experimental and subject to change without notice.")
def _load_for_executorch_for_training_from_buffer(
    buffer: bytes,
) -> TrainingModule:
    et_module = _load_for_executorch_from_buffer(buffer)
    return TrainingModule(et_module)
