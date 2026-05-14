/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Runtime kernels for et_copy._h2d_copy and et_copy._d2h_copy ops.
 *
 * These ops transfer tensor data between CPU and device memory using
 * the DeviceAllocator interface. The device type is inferred from the
 * tensor metadata (out.device_type() for H2D, self.device_type() for D2H),
 * which was set during AOT serialization by PropagateDevicePass.
 */

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace executorch::runtime::native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

/**
 * Copies tensor data from host (CPU) memory to device memory.
 *
 * self: source tensor on CPU
 * out:  destination tensor on device (memory-planned by runtime)
 *
 * The device type and index are inferred from out's TensorImpl metadata.
 */
Tensor&
_h2d_copy_out(KernelRuntimeContext& ctx, const Tensor& self, Tensor& out) {
  auto device_type = out.unsafeGetTensorImpl()->device_type();
  auto device_index = out.unsafeGetTensorImpl()->device_index();

  ET_KERNEL_CHECK_MSG(
      ctx,
      self.unsafeGetTensorImpl()->device_type() == etensor::DeviceType::CPU,
      InvalidArgument,
      out,
      "_h2d_copy: source tensor must be on CPU, got device_type=%d",
      static_cast<int>(self.unsafeGetTensorImpl()->device_type()));

  ET_KERNEL_CHECK_MSG(
      ctx,
      device_type != etensor::DeviceType::CPU,
      InvalidArgument,
      out,
      "_h2d_copy: destination tensor must be on a non-CPU device");

  auto nbytes = self.nbytes();
  ET_KERNEL_CHECK_MSG(
      ctx,
      nbytes == out.nbytes(),
      InvalidArgument,
      out,
      "_h2d_copy: size mismatch: self.nbytes()=%zu, out.nbytes()=%zu",
      nbytes,
      out.nbytes());

  DeviceAllocator* allocator = get_device_allocator(device_type);
  ET_KERNEL_CHECK_MSG(
      ctx,
      allocator != nullptr,
      NotFound,
      out,
      "_h2d_copy: no device allocator registered for device_type=%d",
      static_cast<int>(device_type));

  Error err = allocator->copy_host_to_device(
      out.mutable_data_ptr(), self.const_data_ptr(), nbytes, device_index);
  ET_KERNEL_CHECK_MSG(
      ctx,
      err == Error::Ok,
      Internal,
      out,
      "_h2d_copy: copy_host_to_device failed");

  return out;
}

/**
 * Copies tensor data from device memory to host (CPU) memory.
 *
 * self: source tensor on device
 * out:  destination tensor on CPU (memory-planned by runtime)
 *
 * The device type and index are inferred from self's TensorImpl metadata.
 */
Tensor&
_d2h_copy_out(KernelRuntimeContext& ctx, const Tensor& self, Tensor& out) {
  auto device_type = self.unsafeGetTensorImpl()->device_type();
  auto device_index = self.unsafeGetTensorImpl()->device_index();

  ET_KERNEL_CHECK_MSG(
      ctx,
      device_type != etensor::DeviceType::CPU,
      InvalidArgument,
      out,
      "_d2h_copy: source tensor must be on a non-CPU device");

  ET_KERNEL_CHECK_MSG(
      ctx,
      out.unsafeGetTensorImpl()->device_type() == etensor::DeviceType::CPU,
      InvalidArgument,
      out,
      "_d2h_copy: destination tensor must be on CPU, got device_type=%d",
      static_cast<int>(out.unsafeGetTensorImpl()->device_type()));

  auto nbytes = self.nbytes();
  ET_KERNEL_CHECK_MSG(
      ctx,
      nbytes == out.nbytes(),
      InvalidArgument,
      out,
      "_d2h_copy: size mismatch: self.nbytes()=%zu, out.nbytes()=%zu",
      nbytes,
      out.nbytes());

  DeviceAllocator* allocator = get_device_allocator(device_type);
  ET_KERNEL_CHECK_MSG(
      ctx,
      allocator != nullptr,
      NotFound,
      out,
      "_d2h_copy: no device allocator registered for device_type=%d",
      static_cast<int>(device_type));

  Error err = allocator->copy_device_to_host(
      out.mutable_data_ptr(), self.const_data_ptr(), nbytes, device_index);
  ET_KERNEL_CHECK_MSG(
      ctx,
      err == Error::Ok,
      Internal,
      out,
      "_d2h_copy: copy_device_to_host failed");

  return out;
}

} // namespace executorch::runtime::native

EXECUTORCH_LIBRARY(
    et_copy,
    "_h2d_copy.out",
    executorch::runtime::native::_h2d_copy_out);
EXECUTORCH_LIBRARY(
    et_copy,
    "_d2h_copy.out",
    executorch::runtime::native::_d2h_copy_out);
