/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/mm.h>

#include <mutex>
#include <vector>

namespace executorch::backends::cuda {

namespace c10_slim = executorch::backends::aoti::slim::c10;

namespace {

constexpr int kMaxDevices = 16;

struct CuBLASHandles {
  std::mutex mutex;
  cublasHandle_t handles[kMaxDevices] = {};
  bool initialized[kMaxDevices] = {};

  cublasHandle_t get(int device) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!initialized[device]) {
      cudaSetDevice(device);
      cublasCreate(&handles[device]);
      cublasSetMathMode(handles[device], CUBLAS_DEFAULT_MATH);
      initialized[device] = true;
    }
    return handles[device];
  }
};

CuBLASHandles& cublas_handles() {
  static CuBLASHandles instance;
  return instance;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError
aoti_torch_cuda_mm_out(Tensor* out, Tensor* self, Tensor* mat2) {
  ET_CHECK_OR_RETURN_ERROR(
      out != nullptr, InvalidArgument, "mm_out: out is null");
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr, InvalidArgument, "mm_out: self is null");
  ET_CHECK_OR_RETURN_ERROR(
      mat2 != nullptr, InvalidArgument, "mm_out: mat2 is null");
  ET_CHECK_OR_RETURN_ERROR(
      self->dim() == 2 && mat2->dim() == 2 && out->dim() == 2,
      InvalidArgument,
      "mm_out: all tensors must be 2D");
  ET_CHECK_OR_RETURN_ERROR(
      self->is_contiguous() && mat2->is_contiguous() && out->is_contiguous(),
      InvalidArgument,
      "mm_out: all tensors must be contiguous");

  int64_t M = self->size(0);
  int64_t K = self->size(1);
  int64_t N = mat2->size(1);

  ET_CHECK_OR_RETURN_ERROR(
      mat2->size(0) == K,
      InvalidArgument,
      "mm_out: self [%ld,%ld] x mat2 [%ld,%ld] inner dims mismatch",
      M,
      K,
      mat2->size(0),
      N);
  ET_CHECK_OR_RETURN_ERROR(
      out->size(0) == M && out->size(1) == N,
      InvalidArgument,
      "mm_out: out shape mismatch");

  auto dtype = self->dtype();
  ET_CHECK_OR_RETURN_ERROR(
      mat2->dtype() == dtype && out->dtype() == dtype,
      InvalidArgument,
      "mm_out: dtype mismatch");

  cudaDataType_t cuda_dtype;
  cublasComputeType_t compute_type;
  if (dtype == c10_slim::ScalarType::BFloat16) {
    cuda_dtype = CUDA_R_16BF;
    compute_type = CUBLAS_COMPUTE_32F;
  } else if (dtype == c10_slim::ScalarType::Half) {
    cuda_dtype = CUDA_R_16F;
    compute_type = CUBLAS_COMPUTE_32F;
  } else if (dtype == c10_slim::ScalarType::Float) {
    cuda_dtype = CUDA_R_32F;
    compute_type = CUBLAS_COMPUTE_32F;
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        false, InvalidArgument, "mm_out: unsupported dtype");
  }

  int device = self->device_index();
  ET_CHECK_OR_RETURN_ERROR(
      device >= 0 && device < kMaxDevices,
      InvalidArgument,
      "mm_out: device index %d out of range",
      device);

  auto stream_result = getCurrentCUDAStream(device);
  ET_CHECK_OR_RETURN_ERROR(
      stream_result.ok(), Internal, "mm_out: failed to get CUDA stream");

  // Per-device handle; mutex in get() ensures thread-safe initialization.
  // cublasSetStream + cublasGemmEx are serialized under the same mutex to
  // prevent races when multiple threads share a device.
  auto& handles = cublas_handles();
  std::lock_guard<std::mutex> lock(handles.mutex);
  cublasHandle_t handle = handles.get(device);
  cublasSetStream(handle, stream_result.get());

  // cuBLAS is column-major. For row-major C = A @ B:
  //   C^T = B^T @ A^T
  // With column-major interpretation of row-major data:
  //   A_row[M,K] looks like A^T_col[K,M] with lda=K
  //   B_row[K,N] looks like B^T_col[N,K] with ldb=N
  //   C_row[M,N] looks like C^T_col[N,M] with ldc=N
  // So: C^T = B^T @ A^T → gemm(N, N, N, M, K, B, N, A, K, C, N)
  float alpha = 1.0f;
  float beta = 0.0f;

  auto status = cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      N, // m (columns of C^T)
      M, // n (rows of C^T)
      K, // k
      &alpha,
      mat2->data_ptr(), // B^T in col-major = B in row-major
      cuda_dtype,
      N, // ldb (row-major stride of mat2)
      self->data_ptr(), // A^T in col-major = A in row-major
      cuda_dtype,
      K, // lda (row-major stride of self)
      &beta,
      out->data_ptr(),
      cuda_dtype,
      N, // ldc (row-major stride of out)
      compute_type,
      CUBLAS_GEMM_DEFAULT);

  ET_CHECK_OR_RETURN_ERROR(
      status == CUBLAS_STATUS_SUCCESS,
      Internal,
      "mm_out: cublasGemmEx failed with status %d",
      (int)status);

  return Error::Ok;
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
