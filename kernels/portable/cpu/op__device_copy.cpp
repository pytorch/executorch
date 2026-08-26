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
 * These ops transfer tensor data between CPU and device memory using the
 * DeviceAllocator interface.
 *
 * In portable (non-ATen) mode the device type/index are read from the tensor
 * metadata (out for H2D, self for D2H), which PropagateDevicePass set during
 * AOT serialization. In ATen mode the planned at::Tensor does not carry that
 * metadata, so the direction comes from the op identity (_h2d vs _d2h) and the
 * copy targets the registered CUDA DeviceAllocator on the current device.
 */

#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#ifdef USE_ATEN_LIB
#include <c10/util/Exception.h> // TORCH_CHECK (catchable, throws std::exception)
#endif

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using DeviceAllocator = executorch::runtime::DeviceAllocator;
using Error = executorch::runtime::Error;
using RuntimeDeviceIndex = executorch::runtime::etensor::DeviceIndex;
using RuntimeDeviceType = executorch::runtime::etensor::DeviceType;

namespace {

#ifdef USE_ATEN_LIB
// In ATen mode the memory-planned `at::Tensor` does not carry the runtime's
// planned device metadata (it is constructed CPU-side by the ATen tensor
// parser), so the copy direction is taken from the op identity (_h2d vs _d2h)
// rather than from tensor device metadata.
//
// The non-CPU device is CUDA: the DeviceAllocator registry is keyed by
// DeviceType with no "the one non-CPU allocator" lookup, and CUDA is the only
// non-CPU accelerator wired up for ATen-mode device copies today. If another
// ATen-mode accelerator is added, this (and the registry lookup) must learn to
// resolve its type.
//
// `index == -1` requests the allocator's current device. This is a convention
// of the CUDA allocator implementation (see CudaAllocator::copy_* /
// CudaAllocator::allocate in backends/cuda/runtime/cuda_allocator.cpp), not a
// documented guarantee on the DeviceAllocator interface. It assumes the planned
// buffer lives on the current device, i.e. a single-GPU ATen-mode setup;
// multi-GPU ATen mode (planned buffer not on the current device) is not handled
// because the planned device index is not recoverable from the at::Tensor here.
// TODO: support multi-GPU / non-CUDA ATen-mode device copy by recovering the
// planned device type/index (e.g. via the tensor_parser_aten path) instead of
// assuming the current CUDA device.
//
// Unlike the non-ATen branch below, these kernels cannot re-check tensor device
// placement (h2d source-must-be-CPU / dest-must-be-non-CPU, etc.): that
// metadata is not on the at::Tensor. They trust the graph wiring, which the
// device-placement pass controls when it inserts these ops.
constexpr RuntimeDeviceType kAtenDeviceType = RuntimeDeviceType::CUDA;
constexpr RuntimeDeviceIndex kCurrentDeviceIndex = -1;

DeviceAllocator* require_device_allocator(
    RuntimeDeviceType device_type,
    const char* op_name) {
  DeviceAllocator* allocator =
      executorch::runtime::get_device_allocator(device_type);
  TORCH_CHECK(
      allocator != nullptr,
      op_name,
      ": no device allocator registered for device_type=",
      static_cast<int>(device_type));
  return allocator;
}
#else
RuntimeDeviceType runtime_device_type(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->device_type();
}

RuntimeDeviceIndex runtime_device_index(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->device_index();
}
#endif // USE_ATEN_LIB

} // namespace

#ifdef USE_ATEN_LIB
namespace {

// Bytes writable in `t` starting at its data_ptr(), accounting for a possible
// non-zero storage offset (a view may start partway into its storage).
size_t writable_nbytes(const Tensor& t) {
  auto* impl = t.unsafeGetTensorImpl();
  const size_t storage_bytes = impl->storage().nbytes();
  const size_t offset_bytes =
      static_cast<size_t>(impl->storage_offset()) * impl->itemsize();
  return offset_bytes >= storage_bytes ? 0 : storage_bytes - offset_bytes;
}

// Shared body for both directions. In ATen mode the tensors carry no planned
// device metadata, so the direction comes from the caller (`to_device`) and the
// device is the registered CUDA allocator's current device. Validates every
// assumption a raw byte copy makes before touching the allocator:
//  - `out` resizes to `self`'s shape (ATen resize_tensor is metadata-only and
//    does not grow storage, so a too-small `out` would otherwise overrun),
//  - `self` and `out` share dtype and are contiguous,
//  - `out`'s offset-aware writable bytes can hold `self.nbytes()`.
// A zero-byte copy returns early (the allocator rejects null pointers, which
// empty tensors can have). Violations raise via TORCH_CHECK (a catchable
// exception derived from std::exception) rather than aborting: these kernels
// run inside a libtorch process in ATen mode, so a bad boundary tensor or a
// transient copy fault should be recoverable by the host rather than crash the
// process. The ATen custom-op ABI binds the contextless overload, so there is
// no KernelRuntimeContext to return a portable-style Error through.
Tensor& device_copy(const Tensor& self, Tensor& out, bool to_device) {
  const char* op_name = to_device ? "_h2d_copy" : "_d2h_copy";

  // Check preconditions (dtype / contiguity / capacity) BEFORE resizing `out`,
  // so a bad-argument failure leaves `out` untouched. Note the allocator copy
  // below runs after resize_tensor(), so if the copy itself fails `out` may
  // already be resized when we throw; that is a device/transport fault, not a
  // caller-argument error, and the host is expected to treat the copy as failed
  // rather than reuse `out`.
  TORCH_CHECK(
      self.scalar_type() == out.scalar_type(),
      op_name,
      ": self/out dtype mismatch (",
      self.scalar_type(),
      " vs ",
      out.scalar_type(),
      ")");
  TORCH_CHECK(
      self.is_contiguous() && out.is_contiguous(),
      op_name,
      ": self and out must be contiguous");
  const size_t nbytes = self.nbytes();
  const size_t out_writable = writable_nbytes(out);
  TORCH_CHECK(
      nbytes <= out_writable,
      op_name,
      ": out too small (self.nbytes()=",
      nbytes,
      ", out writable=",
      out_writable,
      ")");
  TORCH_CHECK(
      resize_tensor(out, self.sizes()) == Error::Ok,
      op_name,
      ": cannot resize out to self sizes (self.nbytes()=",
      self.nbytes(),
      ")");

  if (nbytes == 0) {
    return out;
  }

  DeviceAllocator* allocator =
      require_device_allocator(kAtenDeviceType, op_name);
  const Error err = to_device ? allocator->copy_host_to_device(
                                    out.mutable_data_ptr(),
                                    self.const_data_ptr(),
                                    nbytes,
                                    kCurrentDeviceIndex)
                              : allocator->copy_device_to_host(
                                    out.mutable_data_ptr(),
                                    self.const_data_ptr(),
                                    nbytes,
                                    kCurrentDeviceIndex);
  TORCH_CHECK(
      err == Error::Ok,
      op_name,
      ": device copy failed (",
      nbytes,
      " bytes, ",
      to_device ? "host->device" : "device->host",
      ")");
  return out;
}

} // namespace

Tensor& _h2d_copy_out(const Tensor& self, Tensor& out) {
  return device_copy(self, out, /*to_device=*/true);
}

Tensor& _d2h_copy_out(const Tensor& self, Tensor& out) {
  return device_copy(self, out, /*to_device=*/false);
}

#else

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
  auto device_type = runtime_device_type(out);
  auto device_index = runtime_device_index(out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      runtime_device_type(self) == RuntimeDeviceType::CPU,
      InvalidArgument,
      out,
      "_h2d_copy: source tensor must be on CPU, got device_type=%d",
      static_cast<int>(runtime_device_type(self)));

  ET_KERNEL_CHECK_MSG(
      ctx,
      device_type != RuntimeDeviceType::CPU,
      InvalidArgument,
      out,
      "_h2d_copy: destination tensor must be on a non-CPU device");

  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, self.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "_h2d_copy: cannot resize out to self sizes (self.nbytes()=%zu exceeds out planned capacity %zu?)",
      self.nbytes(),
      out.nbytes());
  auto nbytes = self.nbytes();

  DeviceAllocator* allocator =
      executorch::runtime::get_device_allocator(device_type);
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
  auto device_type = runtime_device_type(self);
  auto device_index = runtime_device_index(self);

  ET_KERNEL_CHECK_MSG(
      ctx,
      device_type != RuntimeDeviceType::CPU,
      InvalidArgument,
      out,
      "_d2h_copy: source tensor must be on a non-CPU device");

  ET_KERNEL_CHECK_MSG(
      ctx,
      runtime_device_type(out) == RuntimeDeviceType::CPU,
      InvalidArgument,
      out,
      "_d2h_copy: destination tensor must be on CPU, got device_type=%d",
      static_cast<int>(runtime_device_type(out)));

  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, self.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "_d2h_copy: cannot resize out to self sizes (self.nbytes()=%zu exceeds out planned capacity %zu?)",
      self.nbytes(),
      out.nbytes());
  auto nbytes = self.nbytes();

  DeviceAllocator* allocator =
      executorch::runtime::get_device_allocator(device_type);
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

#endif // USE_ATEN_LIB

} // namespace native
} // namespace executor
} // namespace torch
