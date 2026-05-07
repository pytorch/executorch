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

namespace executorch::backends::cuda {

namespace c10_slim = executorch::backends::aoti::slim::c10;

namespace {

constexpr int kMaxDevices = 16;

struct CuBLASHandles {
  std::mutex mutex;
  cublasHandle_t handles[kMaxDevices] = {};
  bool initialized[kMaxDevices] = {};

  // Caller must hold mutex.
  cublasHandle_t get(int device) {
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

bool is_row_major(Tensor* t) {
  return t->stride(1) == 1 && t->stride(0) == t->size(1);
}

bool is_col_major(Tensor* t) {
  return t->stride(0) == 1 && t->stride(1) == t->size(0);
}

size_t dtype_size(c10_slim::ScalarType dtype) {
  switch (dtype) {
    case c10_slim::ScalarType::BFloat16:
    case c10_slim::ScalarType::Half:
      return 2;
    case c10_slim::ScalarType::Float:
      return 4;
    default:
      return 0;
  }
}

// Make a contiguous row-major copy of a 2D tensor if it has non-standard
// strides (broadcast, non-contiguous views). Returns the data pointer to use
// and a device pointer to free (nullptr if no copy was made).
void* ensure_contiguous(
    Tensor* t,
    int64_t rows,
    int64_t cols,
    size_t elem_size,
    cudaStream_t stream,
    void** to_free) {
  *to_free = nullptr;
  if (is_row_major(t) || is_col_major(t)) {
    return t->data_ptr();
  }
  // Non-standard strides — allocate and copy row by row
  size_t row_bytes = cols * elem_size;
  void* dst = nullptr;
  cudaMallocAsync(&dst, rows * row_bytes, stream);
  int64_t s0 = t->stride(0) * elem_size;
  int64_t s1 = t->stride(1) * elem_size;
  char* src = static_cast<char*>(t->data_ptr());
  if (s1 == static_cast<int64_t>(elem_size)) {
    // Stride(1)==1 element but stride(0) is irregular (e.g., broadcast s0=0)
    cudaMemcpy2DAsync(
        dst, row_bytes, src, s0, row_bytes, rows, cudaMemcpyDeviceToDevice, stream);
  } else {
    // Fully general: copy element by element (slow, but correct)
    // This shouldn't happen in practice from Inductor output.
    for (int64_t i = 0; i < rows; i++) {
      for (int64_t j = 0; j < cols; j++) {
        cudaMemcpyAsync(
            static_cast<char*>(dst) + (i * cols + j) * elem_size,
            src + i * s0 + j * s1,
            elem_size,
            cudaMemcpyDeviceToDevice,
            stream);
      }
    }
  }
  *to_free = dst;
  return dst;
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

  int64_t M = self->size(0);
  int64_t K = self->size(1);
  int64_t N = mat2->size(1);

  ET_CHECK_OR_RETURN_ERROR(
      mat2->size(0) == K,
      InvalidArgument,
      "mm_out: self [%ld,%ld] x mat2 [%ld,%ld] inner dims mismatch",
      M, K, mat2->size(0), N);
  ET_CHECK_OR_RETURN_ERROR(
      out->size(0) == M && out->size(1) == N,
      InvalidArgument,
      "mm_out: out shape mismatch");
  ET_CHECK_OR_RETURN_ERROR(
      is_row_major(out), InvalidArgument, "mm_out: out must be contiguous");

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
      "mm_out: device index %d out of range", device);

  auto stream_result = getCurrentCUDAStream(device);
  ET_CHECK_OR_RETURN_ERROR(
      stream_result.ok(), Internal, "mm_out: failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  size_t elem_size = dtype_size(dtype);

  // Ensure inputs are row-major or col-major for cuBLAS. Non-standard strides
  // (broadcast, irregular views) get copied to a contiguous scratch buffer.
  void* a_free = nullptr;
  void* b_free = nullptr;
  void* a_ptr = ensure_contiguous(self, M, K, elem_size, stream, &a_free);
  void* b_ptr = ensure_contiguous(mat2, K, N, elem_size, stream, &b_free);

  // After ensure_contiguous, non-standard inputs are now row-major.
  bool a_row = a_free ? true : is_row_major(self);
  bool b_row = b_free ? true : is_row_major(mat2);

  auto& handles = cublas_handles();
  std::lock_guard<std::mutex> lock(handles.mutex);
  cublasHandle_t handle = handles.get(device);
  cublasSetStream(handle, stream);

  // cuBLAS uses column-major. A row-major [R,C] tensor is seen by cuBLAS
  // as column-major [C,R] — already transposed. So:
  //   row-major → CUBLAS_OP_N (already transposed in cuBLAS's view), ld=stride(0)
  //   col-major → CUBLAS_OP_T (need cuBLAS to transpose), ld=stride(1)
  //
  // We call: C^T[N,M] = B^T[N,K] @ A^T[K,M]
  //   → cublasGemmEx(op_b, op_a, N, M, K, B, ldb, A, lda, C, ldc)

  cublasOperation_t op_a = a_row ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t op_b = b_row ? CUBLAS_OP_N : CUBLAS_OP_T;
  int64_t lda = a_row ? (a_free ? K : self->stride(0))
                      : self->stride(1);
  int64_t ldb = b_row ? (b_free ? N : mat2->stride(0))
                      : mat2->stride(1);
  int64_t ldc = out->stride(0);

  float alpha = 1.0f;
  float beta = 0.0f;

  auto status = cublasGemmEx(
      handle,
      op_b, op_a,
      N, M, K,
      &alpha,
      b_ptr, cuda_dtype, ldb,
      a_ptr, cuda_dtype, lda,
      &beta,
      out->data_ptr(), cuda_dtype, ldc,
      compute_type,
      CUBLAS_GEMM_DEFAULT);

  // Free scratch buffers
  if (a_free) cudaFreeAsync(a_free, stream);
  if (b_free) cudaFreeAsync(b_free, stream);

  ET_CHECK_OR_RETURN_ERROR(
      status == CUBLAS_STATUS_SUCCESS,
      Internal,
      "mm_out: cublasGemmEx failed with status %d", (int)status);

  return Error::Ok;
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
