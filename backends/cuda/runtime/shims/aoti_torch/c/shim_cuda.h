#pragma once

#ifdef USE_CUDA

#include <executorch/backends/cuda/runtime/slim/cuda/Guard.h>
#include <executorch/backends/cuda/runtime/shims/aoti_torch/c/macros.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA utilities
// Keep them in one shim.h similar to libtorch
using CUDAGuardOpaque = slim::cuda::CUDAGuard;
using CUDAGuardHandle = slim::cuda::CUDAGuard *;
using CUDAStreamGuardOpaque = slim::cuda::CUDAStreamGuard;
using CUDAStreamGuardHandle = slim::cuda::CUDAStreamGuard *;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,
    CUDAGuardHandle *ret_guard // returns new reference
) {
  slim::cuda::CUDAGuard *guard = new slim::cuda::CUDAGuard(device_index);
  *ret_guard = reinterpret_cast<CUDAGuardHandle>(guard);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_cuda_guard(CUDAGuardHandle guard) {
  delete reinterpret_cast<slim::cuda::CUDAGuard *>(guard);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_cuda_guard_set_index(CUDAGuardHandle guard, int32_t device_index) {
  reinterpret_cast<slim::cuda::CUDAGuard *>(guard)->set_index(device_index);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_cuda_stream_guard(
    void *stream, int32_t device_index, CUDAStreamGuardHandle *ret_guard) {
  slim::cuda::CUDAStreamGuard *guard = new slim::cuda::CUDAStreamGuard(
      static_cast<cudaStream_t>(stream), device_index);
  *ret_guard = reinterpret_cast<CUDAStreamGuardHandle>(guard);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_cuda_stream_guard(CUDAStreamGuardHandle guard) {
  delete reinterpret_cast<slim::cuda::CUDAStreamGuard *>(guard);
  return AOTI_TORCH_SUCCESS;
}

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_cuda_stream(int32_t device_index, void **ret_stream) {
  *(cudaStream_t *)(ret_stream) =
      slim::cuda::getCurrentCUDAStream(device_index);
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // USE_CUDA
