/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/cuda_allocator.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace executorch::backends::cuda {

using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::etensor::DeviceType;

Tensor& _h2d_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    Tensor& out) {
  const auto* self_impl = self.unsafeGetTensorImpl();
  const auto* out_impl = out.unsafeGetTensorImpl();
  const auto device_index = out_impl->device_index();

  ET_KERNEL_CHECK_MSG(
      ctx,
      self_impl->device_type() == DeviceType::CPU,
      InvalidArgument,
      out,
      "_h2d_copy: source tensor must be on CPU, got device_type=%d",
      static_cast<int>(self_impl->device_type()));

  ET_KERNEL_CHECK_MSG(
      ctx,
      out_impl->device_type() == DeviceType::CUDA,
      InvalidArgument,
      out,
      "_h2d_copy: destination tensor must be on CUDA, got device_type=%d",
      static_cast<int>(out_impl->device_type()));

  const size_t nbytes = self.nbytes();
  ET_KERNEL_CHECK_MSG(
      ctx,
      nbytes == out.nbytes(),
      InvalidArgument,
      out,
      "_h2d_copy: size mismatch: self.nbytes()=%zu, out.nbytes()=%zu",
      nbytes,
      out.nbytes());

  const Error err = CudaAllocator::instance().copy_host_to_device(
      out.mutable_data_ptr(), self.const_data_ptr(), nbytes, device_index);
  ET_KERNEL_CHECK_MSG(
      ctx,
      err == Error::Ok,
      Internal,
      out,
      "_h2d_copy: copy_host_to_device failed");

  return out;
}

Tensor& _d2h_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& self,
    Tensor& out) {
  const auto* self_impl = self.unsafeGetTensorImpl();
  const auto* out_impl = out.unsafeGetTensorImpl();
  const auto device_index = self_impl->device_index();

  ET_KERNEL_CHECK_MSG(
      ctx,
      self_impl->device_type() == DeviceType::CUDA,
      InvalidArgument,
      out,
      "_d2h_copy: source tensor must be on CUDA, got device_type=%d",
      static_cast<int>(self_impl->device_type()));

  ET_KERNEL_CHECK_MSG(
      ctx,
      out_impl->device_type() == DeviceType::CPU,
      InvalidArgument,
      out,
      "_d2h_copy: destination tensor must be on CPU, got device_type=%d",
      static_cast<int>(out_impl->device_type()));

  const size_t nbytes = self.nbytes();
  ET_KERNEL_CHECK_MSG(
      ctx,
      nbytes == out.nbytes(),
      InvalidArgument,
      out,
      "_d2h_copy: size mismatch: self.nbytes()=%zu, out.nbytes()=%zu",
      nbytes,
      out.nbytes());

  const Error err = CudaAllocator::instance().copy_device_to_host(
      out.mutable_data_ptr(), self.const_data_ptr(), nbytes, device_index);
  ET_KERNEL_CHECK_MSG(
      ctx,
      err == Error::Ok,
      Internal,
      out,
      "_d2h_copy: copy_device_to_host failed");

  return out;
}

} // namespace executorch::backends::cuda

EXECUTORCH_LIBRARY(
    et_copy,
    "_h2d_copy.out",
    executorch::backends::cuda::_h2d_copy_out);
EXECUTORCH_LIBRARY(
    et_copy,
    "_d2h_copy.out",
    executorch::backends::cuda::_d2h_copy_out);
