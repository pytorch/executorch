/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/shims/cuda_guard.h>

namespace executorch {
namespace backends {
namespace cuda {

extern "C" {

AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,
    CUDAGuardHandle* ret_guard) {
  ET_CHECK_OR_RETURN_ERROR(
      ret_guard != nullptr,
      InvalidArgument,
      "aoti_torch_create_cuda_guard failed: ret_guard is null");

  auto result = CUDAGuard::create(device_index);
  if (!result.ok()) {
    return result.error();
  }
  *ret_guard = new CUDAGuard(std::move(result.get()));
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_cuda_guard(CUDAGuardHandle guard) {
  ET_CHECK_OR_RETURN_ERROR(
      guard != nullptr,
      InvalidArgument,
      "aoti_torch_delete_cuda_guard failed: guard is null");

  delete guard;
  return Error::Ok;
}

AOTITorchError aoti_torch_cuda_guard_set_index(
    CUDAGuardHandle guard,
    int32_t device_index) {
  ET_CHECK_OR_RETURN_ERROR(
      guard != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_guard_set_index failed: guard is null");

  ET_CHECK_OK_OR_RETURN_ERROR(guard->set_index(device_index));
  return Error::Ok;
}

AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  ET_CHECK_OR_RETURN_ERROR(
      ret_guard != nullptr,
      InvalidArgument,
      "aoti_torch_create_cuda_stream_guard failed: ret_guard is null");

  ET_CHECK_OR_RETURN_ERROR(
      stream != nullptr,
      InvalidArgument,
      "aoti_torch_create_cuda_stream_guard failed: stream is null");

  auto result =
      CUDAStreamGuard::create(static_cast<cudaStream_t>(stream), device_index);
  if (!result.ok()) {
    return result.error();
  }
  *ret_guard = new CUDAStreamGuard(std::move(result.get()));
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  ET_CHECK_OR_RETURN_ERROR(
      guard != nullptr,
      InvalidArgument,
      "aoti_torch_delete_cuda_stream_guard failed: guard is null");

  delete guard;
  return Error::Ok;
}

AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index,
    void** ret_stream) {
  ET_CHECK_OR_RETURN_ERROR(
      ret_stream != nullptr,
      InvalidArgument,
      "aoti_torch_get_current_cuda_stream failed: ret_stream is null");

  auto result = getCurrentCUDAStream(device_index);
  if (!result.ok()) {
    return result.error();
  }
  *ret_stream = static_cast<void*>(result.get());
  return Error::Ok;
}

} // extern "C"

} // namespace cuda
} // namespace backends
} // namespace executorch
