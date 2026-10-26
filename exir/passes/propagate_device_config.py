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
    # already on the target device: the device of the first delegate that
    # reads the input, in graph order. Any other reader of such an input, such
    # as a host op, a delegate on another device, or the method output without
    # skip_d2h_for_method_outputs, reads a copy of it in host memory, made on
    # every call, and the pass logs a warning. A delegate on another device
    # then gets a second copy, from the host. An input that an op writes in
    # place on the host makes to_executorch raise. Useful for pipelines where
    # inputs are pre-staged on GPU. Constants are not passed
    # by the caller, so they are still copied: their data stays on the host,
    # in the program or in its external data file. That copy runs on every
    # call, once for each device that reads the constant, and stays in device
    # memory until the last delegate there has read it.
    #
    # Pair with MemoryPlanningPass(alloc_graph_input=False), or the runtime reserves its own buffer
    # for the input and copies the caller's memory into it, which is what this exists to avoid.
    # A dict can be used to set per-method values, keyed by method name.
    skip_h2d_for_method_inputs: Union[bool, Dict[str, bool]] = False

    # When True, device delegate outputs that the method returns stay on
    # device, with one exception: when an op outside the delegate, such as a
    # KV cache update, writes the host copy of such an output in place, the
    # method returns that host copy, since the device tensor never sees the
    # write. The copy to the host runs on every call, and the pass logs a
    # warning about it.
    # Useful for cross-method GPU pipelines where the next method consumes GPU
    # tensors directly.
    #
    # Pair with MemoryPlanningPass(alloc_graph_output=False) for the same reason as the input flag.
    # A dict can be used to set per-method values, keyed by method name.
    skip_d2h_for_method_outputs: Union[bool, Dict[str, bool]] = False

    # When True, the copies that the skip flags above cannot remove make
    # to_executorch raise instead of logging a warning: with
    # skip_h2d_for_method_inputs, a constant copied to the device and a method
    # input copied to the host for a reader off its device, and with
    # skip_d2h_for_method_outputs, a returned output copied to the host for a
    # host op that writes it. Each runs on every call. Use it to make sure a
    # program that skips its boundary copies has none left.
    error_on_per_call_copies: bool = False

    def __hash__(self) -> int:
        return hash(
            (
                str(self.skip_h2d_for_method_inputs),
                str(self.skip_d2h_for_method_outputs),
                self.error_on_per_call_copies,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropagateDeviceConfig):
            return False
        return (
            self.skip_h2d_for_method_inputs == other.skip_h2d_for_method_inputs
            and self.skip_d2h_for_method_outputs == other.skip_d2h_for_method_outputs
            and self.error_on_per_call_copies == other.error_on_per_call_copies
        )
