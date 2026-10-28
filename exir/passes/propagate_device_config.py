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

    Memory planning reserves buffers for graph inputs and outputs by default. A program that skips
    the input copy must use ``alloc_graph_input=False``. Unplanning the outputs as well, as below,
    copies the least::

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
    already owns and refuses to copy into it, so every call fails with ``NotSupported``. Planned
    outputs work, but they live in the method's memory, which the next run overwrites, and the
    Python ``execute()`` copies them unless it gets ``clone_outputs=False``.
    """

    # When True, method-level input tensors that feed directly into a device
    # delegate are NOT wrapped with _h2d_copy. The user must provide tensors
    # already on the target device. Useful for pipelines where inputs are
    # pre-staged on GPU.
    #
    # Pair with MemoryPlanningPass(alloc_graph_input=False), or the runtime reserves its own buffer
    # for the input and refuses the caller's tensor.
    # For per-method values, pass a dict of PropagateDeviceConfig, keyed by method name, as the
    # propagate_device_config of ExecutorchBackendConfig.
    skip_h2d_for_method_inputs: Union[bool, Dict[str, bool]] = False

    # When True, device delegate outputs that are directly method outputs
    # are NOT wrapped with _d2h_copy. The method outputs stay on device.
    # Useful for cross-method GPU pipelines where the next method consumes
    # GPU tensors directly.
    #
    # Planned outputs work, but they live in the method's memory, which the next run overwrites,
    # and the Python runtime copies them. MemoryPlanningPass(alloc_graph_output=False) avoids
    # that copy, and then a C++ caller must set each output with Module::set_output.
    # For per-method values, pass a dict of PropagateDeviceConfig, keyed by method name, as the
    # propagate_device_config of ExecutorchBackendConfig.
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
