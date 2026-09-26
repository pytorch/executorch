# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Configuration for PropagateDevicePass.

This is intentionally kept in a lightweight module (no heavy imports such as
the et_copy op registry) so that ``ExecutorchBackendConfig`` -- which is
imported throughout the codebase -- can reference ``PropagateDeviceConfig``
without pulling in the device-copy op registration as an import-time side
effect.
"""

from dataclasses import dataclass
from typing import Dict, Union

from torch.fx._compatibility import compatibility


@compatibility(is_backward_compatible=False)
@dataclass
class PropagateDeviceConfig:
    """Controls whether the runtime copies method inputs and outputs across the device boundary.

    Skipping a copy also means the runtime must not reserve its own buffer for the tensor, or it
    would fill that buffer from the caller's memory and the copy would come back. Memory planning
    allocates graph inputs and outputs by default, so a program using either skip below needs::

        ExecutorchBackendConfig(
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True,
                skip_d2h_for_method_outputs=True,
            ),
            enable_non_cpu_memory_planning=True,
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=False, alloc_graph_output=False
            ),
        )

    Without ``alloc_graph_input=False`` the runtime reserves device memory for an input the caller
    already owns, then copies into it with a host memcpy, which is undefined for device memory.
    """

    # When True, method-level input tensors that feed directly into a device
    # delegate are NOT wrapped with _h2d_copy. The user must provide tensors
    # already on the target device. Useful for pipelines where inputs are
    # pre-staged on GPU.
    #
    # Pair with MemoryPlanningPass(alloc_graph_input=False), or the runtime reserves its own buffer
    # for the input and copies the caller's memory into it, which is what this exists to avoid.
    # A dict can be used to set per-method values, keyed by method name.
    skip_h2d_for_method_inputs: Union[bool, Dict[str, bool]] = False

    # When True, device delegate outputs that are directly method outputs
    # are NOT wrapped with _d2h_copy. The method outputs stay on device.
    # Useful for cross-method GPU pipelines where the next method consumes
    # GPU tensors directly.
    #
    # Pair with MemoryPlanningPass(alloc_graph_output=False) for the same reason as the input flag.
    # A dict can be used to set per-method values, keyed by method name.
    skip_d2h_for_method_outputs: Union[bool, Dict[str, bool]] = False

    def __hash__(self) -> int:
        return hash(
            (
                str(self.skip_h2d_for_method_inputs),
                str(self.skip_d2h_for_method_outputs),
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropagateDeviceConfig):
            return False
        return (
            self.skip_h2d_for_method_inputs == other.skip_h2d_for_method_inputs
            and self.skip_d2h_for_method_outputs == other.skip_d2h_for_method_outputs
        )
