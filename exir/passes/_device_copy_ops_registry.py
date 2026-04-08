# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Registry for device copy ops used to insert explicit H2D (host-to-device)
and D2H (device-to-host) data transfer operations at delegate boundaries.

These ops are inserted by PropagateDevicePass when enable_non_cpu_memory_planning
is True, making the graph functional by explicitly transferring data between
CPU and device memory.

Follows the same registration pattern as dim_order_ops_registry.py.
"""

import torch
from torch.library import impl, Library

lib = Library("et_copy", "DEF")

# _h2d_copy: copies a CPU tensor to device memory.
# At tracing time, this is a clone (both on CPU). At runtime, the out tensor
# is memory-planned on device, and the kernel calls
# DeviceAllocator::copy_host_to_device.
lib.define("_h2d_copy(Tensor self) -> Tensor")
lib.define("_h2d_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

# _d2h_copy: copies a device tensor to CPU memory.
# At tracing time, this is a clone (both on CPU). At runtime, the self tensor
# has device memory, and the kernel calls DeviceAllocator::copy_device_to_host.
lib.define("_d2h_copy(Tensor self) -> Tensor")
lib.define("_d2h_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")


@impl(lib, "_h2d_copy", "CompositeImplicitAutograd")
def _h2d_copy_impl(self: torch.Tensor) -> torch.Tensor:
    # During tracing, both tensors are on CPU. Just clone to represent the transfer.
    return self.clone()


@impl(lib, "_h2d_copy.out", "CompositeImplicitAutograd")
def _h2d_copy_out_impl(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(self)
    return out


@impl(lib, "_d2h_copy", "CompositeImplicitAutograd")
def _d2h_copy_impl(self: torch.Tensor) -> torch.Tensor:
    # During tracing, both tensors are on CPU. Just clone to represent the transfer.
    return self.clone()


@impl(lib, "_d2h_copy.out", "CompositeImplicitAutograd")
def _d2h_copy_out_impl(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(self)
    return out
