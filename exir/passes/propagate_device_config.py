# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Configuration for PropagateDevicePass, and the target_device compile spec it reads.

This is intentionally kept in a lightweight module (no heavy imports such as
the et_copy op registry) so that ``ExecutorchBackendConfig`` -- which is
imported throughout the codebase -- can reference ``PropagateDeviceConfig``
without pulling in the device-copy op registration as an import-time side
effect.
"""

from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING, Union

import executorch.exir.schema as schema
import torch
from torch.fx._compatibility import compatibility

if TYPE_CHECKING:
    from executorch.exir.lowered_backend_module import LoweredBackendModule

# CompileSpec key convention for specifying the target device.
# Partitioners that target a specific device should include a CompileSpec entry
# with this key and a value encoding the device string (e.g., b"cuda:0").
TARGET_DEVICE_COMPILE_SPEC_KEY = "target_device"

# Ops that read only the sizes, strides or offset of a tensor. Those stay in
# host memory when the data is on a device, so these ops need no copy of it.
_METADATA_READERS = (
    torch.ops.aten.sym_size.default,
    torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_stride.int,
    torch.ops.aten.sym_numel.default,
    torch.ops.aten.sym_storage_offset.default,
)


def _parse_device_spec_value(value: bytes) -> tuple[schema.DeviceType, int]:
    """
    Parse a target_device CompileSpec value (e.g., b"cuda:0") into
    (DeviceType, device_index).

    The type portion is matched case-insensitively against schema.DeviceType
    member names (e.g., "cpu", "cuda").  Raises ValueError for unknown types.
    """
    device_str = value.decode("utf-8").strip().lower()
    if ":" in device_str:
        type_str, index_str = device_str.split(":", 1)
        device_index = int(index_str)
    else:
        type_str = device_str
        device_index = 0
    device_type = next(
        (dt for dt in schema.DeviceType if dt.name.lower() == type_str),
        None,
    )
    if device_type is None:
        valid = ", ".join(dt.name for dt in schema.DeviceType)
        raise ValueError(f"Unknown device type '{type_str}'. Valid types: {valid}")
    return device_type, device_index


def _get_target_device_from_compile_specs(
    lowered_module: "LoweredBackendModule",
) -> Optional[tuple[schema.DeviceType, int]]:
    """
    Look for a CompileSpec with key TARGET_DEVICE_COMPILE_SPEC_KEY and return
    the corresponding (DeviceType, device_index), or None if not found.
    """
    for spec in lowered_module.compile_specs:
        if spec.key == TARGET_DEVICE_COMPILE_SPEC_KEY:
            return _parse_device_spec_value(spec.value)
    return None


@compatibility(is_backward_compatible=False)
@dataclass
class PropagateDeviceConfig:
    """Controls whether the runtime copies method inputs, outputs and constants across the device boundary.

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

    ``skip_h2d_for_constants`` does not need ``alloc_graph_input=False``: it works with planned inputs.
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
    # memory until the last delegate there has read it. Set
    # skip_h2d_for_constants to keep constants on the device instead.
    #
    # Pair with MemoryPlanningPass(alloc_graph_input=False), or the runtime reserves its own buffer
    # for the input and refuses the caller's tensor.
    # For per-method values, pass a dict of PropagateDeviceConfig, keyed by method name, as the
    # propagate_device_config of ExecutorchBackendConfig.
    skip_h2d_for_method_inputs: Union[bool, Dict[str, bool]] = False

    # When True, device delegate outputs that the method returns stay on
    # device, with one exception: when an op outside the delegate, such as a
    # mutable custom op left on the host, writes the host copy of such an
    # output in place, the method returns that host copy, since the device
    # tensor never sees the write. The copy to the host runs on every call, and the pass logs a
    # warning about it.
    # Useful for cross-method GPU pipelines where the next method consumes GPU
    # tensors directly.
    #
    # Planned outputs work, but they live in the method's memory, which the next run overwrites,
    # and the Python runtime copies them. MemoryPlanningPass(alloc_graph_output=False) avoids
    # that copy, and then a C++ caller must set each output with Module::set_output.
    # For per-method values, pass a dict of PropagateDeviceConfig, keyed by method name, as the
    # propagate_device_config of ExecutorchBackendConfig.
    skip_d2h_for_method_outputs: Union[bool, Dict[str, bool]] = False

    # When True, the copies that the skip flags cannot remove make
    # to_executorch raise instead of logging a warning: with
    # skip_h2d_for_method_inputs or skip_h2d_for_constants, a constant copied
    # to the device; with skip_h2d_for_method_inputs, a method input copied to
    # the host for a reader off its device; and with
    # skip_d2h_for_method_outputs, a returned output copied to the host for a
    # host op that writes it. Each runs on every call. Use it to make sure a
    # program that skips its boundary copies has none left.
    error_on_per_call_copies: bool = False

    # When True, a constant that only delegates on one device read is kept in
    # that device's memory instead of being copied there on every call. The
    # runtime loads its data there once, when it loads the method, through
    # host memory. The constant then uses device memory for as long as the
    # method is loaded, and each method that reads it keeps its own copy. A
    # constant that anything else reads is still copied on every call. Only a
    # constant that a delegate reads directly is kept: one changed on the host
    # first, such as by a view, a transpose or a clone, is copied on every
    # call, and error_on_per_call_copies does not catch that copy. To keep
    # it, have the partitioner take that change into the delegate too.
    # MemoryPlanningPass(share_mutable_buffers=True) refuses a registered
    # buffer kept on the device this way only if some method changes it.
    skip_h2d_for_constants: bool = False

    def __hash__(self) -> int:
        return hash(
            (
                str(self.skip_h2d_for_method_inputs),
                str(self.skip_d2h_for_method_outputs),
                self.error_on_per_call_copies,
                self.skip_h2d_for_constants,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PropagateDeviceConfig):
            return False
        return (
            self.skip_h2d_for_method_inputs == other.skip_h2d_for_method_inputs
            and self.skip_d2h_for_method_outputs == other.skip_d2h_for_method_outputs
            and self.error_on_per_call_copies == other.error_on_per_call_copies
            and self.skip_h2d_for_constants == other.skip_h2d_for_constants
        )
