

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/sdpa.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/backends/cuda/runtime/shims/sdpa.cuh>

namespace executorch::backends::cuda {

// ============================================================================
// CUDA Error Tracking Infrastructure
// ============================================================================

// Global counter for tracking CUDA operations
static std::atomic<int> g_cuda_op_counter{0};

/**
 * Check for CUDA errors at a specific point in code
 * This helper function:
 * 1. Calls cudaGetLastError() to retrieve and CLEAR any pending error
 * 2. Prints detailed information about where the error occurred
 * 3. Returns true if an error was found
 *
 * IMPORTANT: cudaGetLastError() both retrieves AND clears the error,
 * so calling it multiple times will only report the error once.
 */
inline bool
check_cuda_error_at(const char* file, int line, const char* operation) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    int op_id = g_cuda_op_counter.fetch_add(1);
    printf("\n🔴 [CUDA ERROR #%d] at %s:%d\n", op_id, file, line);
    printf("   Operation: %s\n", operation);
    printf("   Error code: %d\n", static_cast<int>(err));
    printf("   Error name: %s\n", cudaGetErrorName(err));
    printf("   Error string: %s\n", cudaGetErrorString(err));
    printf("\n");
    fflush(stdout);
    return true;
  }
  return false;
}

/**
 * Synchronize and check for asynchronous errors
 * This is more thorough - it waits for all GPU operations to complete
 * and then checks for errors that may have occurred during execution.
 */
inline bool sync_and_check_cuda_error_at(
    const char* file,
    int line,
    const char* operation) {
  // First, synchronize to ensure all async operations complete
  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    int op_id = g_cuda_op_counter.fetch_add(1);
    printf("\n🔴 [CUDA SYNC ERROR #%d] at %s:%d\n", op_id, file, line);
    printf("   Operation: %s\n", operation);
    printf("   Sync error code: %d\n", static_cast<int>(sync_err));
    printf("   Sync error name: %s\n", cudaGetErrorName(sync_err));
    printf("   Sync error string: %s\n", cudaGetErrorString(sync_err));
    printf("\n");
    fflush(stdout);
    return true;
  }

  // Then check for any pending errors
  return check_cuda_error_at(file, line, operation);
}

// Macros for easy use
#define CHECK_CUDA_ERROR_WITH_MSG(msg)                             \
  {                                                                \
    cudaError_t pre_err = cudaGetLastError();                      \
    if (pre_err != cudaSuccess) {                                  \
      printf(                                                      \
          "⚠️  WARNING: Pre-existing CUDA error: %s with msg %s\n", \
          cudaGetErrorString(pre_err),                             \
          msg);                                                    \
      fflush(stdout);                                              \
      return nullptr;                                              \
    }                                                              \
  }

#define SYNC_CHECK_CUDA_ERROR(operation) \
  sync_and_check_cuda_error_at(__FILE__, __LINE__, operation)

/**
 * Clear any pending CUDA errors and report what was cleared
 * Use this at the start of a function to start with a clean slate
 */
inline void clear_cuda_errors(const char* context) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("\n⚠️  [CLEARED CUDA ERROR] in %s\n", context);
    printf(
        "   Cleared error: %s (%s)\n",
        cudaGetErrorName(err),
        cudaGetErrorString(err));
    printf("   This error was inherited from a previous operation.\n");
    printf("   Stack trace would be needed to find the original source.\n\n");
    fflush(stdout);
  }
}

// ============================================================================
// Tensor Logging Infrastructure for Debugging
// ============================================================================

// Global counter for tracking kernel invocations
static std::atomic<int> g_kernel_call_counter{0};

/**
 * Helper function to save tensor data to disk for debugging
 *
 * Saves tensor in a format that can be easily read from Python:
 * - metadata.txt: Contains shape, strides, dtype info
 * - data.bin: Raw binary data
 *
 * IMPORTANT: This function is for debugging only. It may introduce
 * CUDA errors if the tensor pointer is invalid or if there are
 * pre-existing CUDA errors. Consider disabling logging in production.
 */
template <typename scalar_t>
bool save_tensor_to_file(
    const std::string& base_path,
    const std::string& tensor_name,
    const scalar_t* data_ptr,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    cudaStream_t stream) {
  // Validate input pointer
  if (data_ptr == nullptr) {
    printf("  [SKIP] %s: data_ptr is nullptr\n", tensor_name.c_str());
    fflush(stdout);
    return false;
  }

  // Calculate total number of elements
  int64_t total_elements = 1;
  for (auto s : shape) {
    total_elements *= s;
  }

  if (total_elements <= 0) {
    printf(
        "  [SKIP] %s: total_elements=%ld <= 0\n",
        tensor_name.c_str(),
        total_elements);
    fflush(stdout);
    return false;
  }

  // For very large tensors, skip to avoid memory issues
  const int64_t MAX_ELEMENTS = 100 * 1024 * 1024; // 100M elements max
  if (total_elements > MAX_ELEMENTS) {
    printf(
        "  [SKIP] %s: too large (%ld elements > %ld max)\n",
        tensor_name.c_str(),
        total_elements,
        MAX_ELEMENTS);
    fflush(stdout);
    // Still save metadata
    std::ofstream meta_file(base_path + "/" + tensor_name + "_meta.txt");
    meta_file << "shape: ";
    for (size_t i = 0; i < shape.size(); ++i) {
      meta_file << shape[i];
      if (i < shape.size() - 1)
        meta_file << ",";
    }
    meta_file << "\n";
    meta_file << "strides: ";
    for (size_t i = 0; i < strides.size(); ++i) {
      meta_file << strides[i];
      if (i < strides.size() - 1)
        meta_file << ",";
    }
    meta_file << "\n";
    meta_file << "dtype: " << typeid(scalar_t).name() << "\n";
    meta_file << "dtype_size: " << sizeof(scalar_t) << "\n";
    meta_file << "skipped: true (too large)\n";
    meta_file.close();
    return true;
  }

  // Allocate host memory
  std::vector<scalar_t> host_data;
  try {
    host_data.resize(total_elements);
  } catch (const std::bad_alloc& e) {
    printf(
        "  [ERROR] %s: failed to allocate host memory for %ld elements\n",
        tensor_name.c_str(),
        total_elements);
    fflush(stdout);
    return false;
  }

  // Use synchronous copy to avoid async error propagation issues
  // This is safer for debugging, though slower
  cudaError_t err = cudaMemcpy(
      host_data.data(),
      data_ptr,
      total_elements * sizeof(scalar_t),
      cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    printf(
        "  [ERROR] %s: cudaMemcpy failed: %s (code %d)\n",
        tensor_name.c_str(),
        cudaGetErrorString(err),
        static_cast<int>(err));
    printf(
        "         data_ptr=%p, size=%ld bytes\n",
        (void*)data_ptr,
        total_elements * sizeof(scalar_t));
    fflush(stdout);
    // Clear the error so it doesn't propagate
    cudaGetLastError();
    return false;
  }

  // Save metadata
  std::ofstream meta_file(base_path + "/" + tensor_name + "_meta.txt");
  meta_file << "shape: ";
  for (size_t i = 0; i < shape.size(); ++i) {
    meta_file << shape[i];
    if (i < shape.size() - 1)
      meta_file << ",";
  }
  meta_file << "\n";

  meta_file << "strides: ";
  for (size_t i = 0; i < strides.size(); ++i) {
    meta_file << strides[i];
    if (i < strides.size() - 1)
      meta_file << ",";
  }
  meta_file << "\n";

  meta_file << "dtype: " << typeid(scalar_t).name() << "\n";
  meta_file << "dtype_size: " << sizeof(scalar_t) << "\n";
  meta_file.close();

  // Save binary data
  std::ofstream data_file(
      base_path + "/" + tensor_name + "_data.bin", std::ios::binary);
  data_file.write(
      reinterpret_cast<const char*>(host_data.data()),
      total_elements * sizeof(scalar_t));
  data_file.close();

  printf("  [OK] Saved %s (shape: ", tensor_name.c_str());
  for (size_t i = 0; i < shape.size(); ++i) {
    printf("%ld", shape[i]);
    if (i < shape.size() - 1)
      printf("x");
  }
  printf(", %ld elements)\n", total_elements);
  fflush(stdout);

  return true;
}

/**
 * Helper to save Tensor object to file
 */
template <typename scalar_t>
bool save_tensor_object_to_file(
    const std::string& base_path,
    const std::string& tensor_name,
    const Tensor* tensor,
    cudaStream_t stream) {
  // Validate tensor pointer
  if (tensor == nullptr) {
    printf("  [SKIP] %s: tensor is nullptr\n", tensor_name.c_str());
    fflush(stdout);
    return false;
  }

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  int dim = tensor->dim();
  if (dim <= 0 || dim > 10) {
    printf("  [SKIP] %s: invalid dim=%d\n", tensor_name.c_str(), dim);
    fflush(stdout);
    return false;
  }

  for (int i = 0; i < dim; ++i) {
    shape.push_back(tensor->size(i));
    strides.push_back(tensor->strides()[i]);
  }

  const scalar_t* data_ptr =
      reinterpret_cast<const scalar_t*>(tensor->data_ptr());
  return save_tensor_to_file(
      base_path, tensor_name, data_ptr, shape, strides, stream);
}

/**
 * Log kernel invocation inputs (before kernel call)
 * Returns the call_id and base_path for later output logging
 *
 * NOTE: Set SDPA_DISABLE_LOGGING=1 environment variable to disable logging
 */
template <typename scalar_t>
std::pair<int, std::string> log_kernel_call_inputs(
    const std::string& kernel_name,
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_bias,
    float scale_factor,
    bool is_causal,
    cudaStream_t stream) {
  int call_id = g_kernel_call_counter.fetch_add(1);
  std::string base_path = "";

  // Check if logging is disabled
  static bool logging_disabled = (getenv("SDPA_DISABLE_LOGGING") != nullptr);
  if (logging_disabled) {
    return std::make_pair(call_id, base_path);
  }

  // Clear any pre-existing CUDA errors before logging
  // This prevents error propagation into the logging code
  cudaError_t pre_err = cudaGetLastError();
  if (pre_err != cudaSuccess) {
    printf(
        "\n⚠️  [LOG WARNING] Pre-existing CUDA error before logging: %s\n",
        cudaGetErrorString(pre_err));
    printf("   Logging may fail or produce incorrect results.\n");
    fflush(stdout);
  }

  // Create directory for this call
  std::stringstream ss;
  ss << "/tmp/sdpa_debug_" << std::setfill('0') << std::setw(4) << call_id;
  base_path = ss.str();

  // Create directory
  std::string mkdir_cmd = "mkdir -p " + base_path;
  int ret = system(mkdir_cmd.c_str());
  if (ret != 0) {
    printf("\n[LOG ERROR] Failed to create directory %s\n", base_path.c_str());
    fflush(stdout);
    return std::make_pair(call_id, "");
  }

  printf(
      "\n=== Logging %s call #%d INPUTS to %s ===\n",
      kernel_name.c_str(),
      call_id,
      base_path.c_str());
  fflush(stdout);

  // Save metadata
  std::ofstream meta_file(base_path + "/call_info.txt");
  if (meta_file.is_open()) {
    meta_file << "kernel_name: " << kernel_name << "\n";
    meta_file << "call_id: " << call_id << "\n";
    meta_file << "scale_factor: " << std::setprecision(10) << scale_factor
              << "\n";
    meta_file << "is_causal: " << (is_causal ? "true" : "false") << "\n";
    if (query) {
      meta_file << "dtype: " << static_cast<int>(query->dtype()) << "\n";
    }
    meta_file.close();
  }

  // Save input tensors (with error checking)
  save_tensor_object_to_file<scalar_t>(base_path, "query", query, stream);
  save_tensor_object_to_file<scalar_t>(base_path, "key", key, stream);
  save_tensor_object_to_file<scalar_t>(base_path, "value", value, stream);

  if (attn_bias != nullptr) {
    save_tensor_object_to_file<scalar_t>(
        base_path, "attn_bias", attn_bias, stream);
  }

  // Check if logging introduced any CUDA errors
  cudaError_t post_err = cudaGetLastError();
  if (post_err != cudaSuccess) {
    printf(
        "\n⚠️  [LOG WARNING] CUDA error occurred during logging: %s\n",
        cudaGetErrorString(post_err));
    printf(
        "   This error has been cleared and will not affect kernel execution.\n");
    fflush(stdout);
  }

  printf("=== Finished logging INPUTS for call #%d ===\n\n", call_id);
  fflush(stdout);

  return std::make_pair(call_id, base_path);
}

/**
 * Log kernel invocation output (after kernel call)
 */
template <typename scalar_t>
void log_kernel_call_output(
    int call_id,
    const std::string& base_path,
    const Tensor* output,
    cudaStream_t stream) {
  // Check if logging is disabled or base_path is empty
  static bool logging_disabled = (getenv("SDPA_DISABLE_LOGGING") != nullptr);
  if (logging_disabled || base_path.empty()) {
    return;
  }

  printf(
      "\n=== Logging call #%d OUTPUT to %s ===\n", call_id, base_path.c_str());
  fflush(stdout);

  // Save output tensor
  save_tensor_object_to_file<scalar_t>(base_path, "output", output, stream);

  // Clear any CUDA errors from logging
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "  [LOG WARNING] CUDA error during output logging: %s (cleared)\n",
        cudaGetErrorString(err));
    fflush(stdout);
  }

  printf("=== Finished logging OUTPUT for call #%d ===\n\n", call_id);
  fflush(stdout);
}

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;
using executorch::runtime::Error;

// ============================================================================
// Input Validation Kernels - Check for Inf/NaN
// ============================================================================

/**
 * Kernel to check if a tensor contains any inf or NaN values
 * Sets flag to 1 if any invalid value is found
 */
template <typename scalar_t>
__global__ void check_for_inf_nan_kernel(
    const scalar_t* data,
    const int64_t size,
    int* has_invalid) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    float val = static_cast<float>(data[idx]);
    if (isinf(val) || isnan(val)) {
      atomicExch(has_invalid, 1);
    }
  }
}

/**
 * Host function to check if tensor contains inf or NaN values
 * Returns true if tensor contains invalid values
 */
template <typename scalar_t>
bool check_tensor_for_inf_nan(
    const Tensor* tensor,
    const char* tensor_name,
    cudaStream_t stream) {
  // Calculate total size
  int64_t total_size = 1;
  for (int i = 0; i < tensor->dim(); ++i) {
    total_size *= tensor->size(i);
  }

  // Allocate flag on device
  int* d_has_invalid = nullptr;
  cudaMalloc(&d_has_invalid, sizeof(int));
  cudaMemset(d_has_invalid, 0, sizeof(int));

  // Launch kernel to check for inf/nan
  const int threads_per_block = 256;
  const int num_blocks =
      (total_size + threads_per_block - 1) / threads_per_block;

  const scalar_t* data_ptr =
      reinterpret_cast<const scalar_t*>(tensor->data_ptr());

  check_for_inf_nan_kernel<scalar_t>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          data_ptr, total_size, d_has_invalid);

  // Copy result back to host
  int h_has_invalid = 0;
  cudaMemcpy(
      &h_has_invalid, d_has_invalid, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_has_invalid);

  if (h_has_invalid) {
    printf(
        "\n⚠️  [SDPA ERROR] Tensor '%s' contains Inf or NaN values!\n",
        tensor_name);
    printf("  Shape: [");
    for (int i = 0; i < tensor->dim(); ++i) {
      printf("%ld", tensor->size(i));
      if (i < tensor->dim() - 1)
        printf(", ");
    }
    printf("]\n");
    printf("  Total elements: %ld\n", total_size);
    printf("  This will cause kernel launch failure!\n\n");
    fflush(stdout);
    return true;
  }

  return false;
}

// ============================================================================
// CUDA Kernels for Softmax and Masking
// ============================================================================

// Helper function for max with different types
__device__ __forceinline__ float device_max(float a, float b) {
  return fmaxf(a, b);
}

__device__ __forceinline__ __half device_max(__half a, __half b) {
  return __hgt(a, b) ? a : b;
}

__device__ __forceinline__ __nv_bfloat16
device_max(__nv_bfloat16 a, __nv_bfloat16 b) {
#if __CUDA_ARCH__ >= 800
  return __hgt(a, b) ? a : b;
#else
  return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
#endif
}

/**
 * Softmax kernel with optional causal masking and attention bias
 *
 * Computes softmax along the last dimension (seq_len_k) of a 4D tensor.
 * Supports:
 * - Causal masking where positions j > i are masked out
 * - Explicit attention bias (additive mask, 0.0 = allow, -inf = mask)
 *
 * Input: [batch, num_heads, seq_len_q, seq_len_k]
 * Output: [batch, num_heads, seq_len_q, seq_len_k]
 * Bias (optional): [batch, num_heads, seq_len_q, seq_len_k] or broadcastable
 *
 * Each thread processes one row (seq_len_q position).
 *
 * Note: Supports in-place operation (input == output).
 */
template <typename scalar_t>
__global__ void softmax_with_mask_kernel(
    const scalar_t* input,
    scalar_t* output,
    const scalar_t* attn_bias, // Optional attention bias (can be nullptr)
    const int64_t batch,
    const int64_t num_heads,
    const int64_t seq_len_q,
    const int64_t seq_len_k,
    const bool is_causal,
    const float scale) {
  // Each block processes one row of the attention matrix
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total_rows = batch * num_heads * seq_len_q;

  if (idx >= total_rows) {
    return;
  }

  // Decode position - we only need i for causal masking
  const int64_t i = idx % seq_len_q;

  // Pointer to the start of this row
  const int64_t row_offset = idx * seq_len_k;
  const scalar_t* input_row = input + row_offset;
  scalar_t* output_row = output + row_offset;

  // Attention bias for this row (if provided)
  const scalar_t* bias_row = attn_bias ? (attn_bias + row_offset) : nullptr;

  // Find max for numerical stability (two-pass algorithm)
  float max_val = -FLT_MAX;
  for (int64_t j = 0; j < seq_len_k; ++j) {
    // Apply scaling and bias
    float val = static_cast<float>(input_row[j]) * scale;
    if (bias_row) {
      float bias = static_cast<float>(bias_row[j]);
      val += bias; // Additive bias (0.0 = allow, -inf = mask)
    }

    // Apply causal mask if needed
    if (!is_causal || j <= i) {
      max_val = fmaxf(max_val, val);
    }
  }

  // Compute exp and sum
  float sum_exp = 0.0f;
  for (int64_t j = 0; j < seq_len_k; ++j) {
    float val = static_cast<float>(input_row[j]) * scale;
    if (bias_row) {
      float bias = static_cast<float>(bias_row[j]);
      val += bias;
    }

    float exp_val;
    if (!is_causal || j <= i) {
      exp_val = expf(val - max_val);
    } else {
      exp_val = 0.0f;
    }
    output_row[j] = static_cast<scalar_t>(exp_val);
    sum_exp += exp_val;
  }

  // Normalize
  const float inv_sum = 1.0f / (sum_exp + 1e-12f); // Add epsilon for stability
  for (int64_t j = 0; j < seq_len_k; ++j) {
    output_row[j] =
        static_cast<scalar_t>(static_cast<float>(output_row[j]) * inv_sum);
  }
}

/**
 * Scale kernel - multiply all elements by a scalar
 */
template <typename scalar_t>
__global__ void scale_kernel(
    scalar_t* __restrict__ data,
    const int64_t size,
    const float scale) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = static_cast<scalar_t>(static_cast<float>(data[idx]) * scale);
  }
}

// ============================================================================
// cuBLAS Helper Functions
// ============================================================================

/**
 * Get or create a cuBLAS handle for the current stream
 *
 * Note: In production, this should use a handle pool or be managed
 * by the backend infrastructure. This is a simplified version.
 */
cublasHandle_t get_cublas_handle(cudaStream_t stream) {
  static cublasHandle_t handle = nullptr;
  static bool init_attempted = false;
  static bool init_success = false;

  if (!init_attempted) {
    init_attempted = true;
    cublasStatus_t status = cublasCreate_v2(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      ET_LOG(
          Error,
          "Failed to create cuBLAS handle: %d",
          static_cast<int>(status));
      handle = nullptr;
      init_success = false;
    } else {
      ET_LOG(Info, "cuBLAS handle created successfully");
      init_success = true;
    }
  }

  if (!init_success || handle == nullptr) {
    return nullptr;
  }

  cublasStatus_t status = cublasSetStream_v2(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    ET_LOG(Error, "Failed to set cuBLAS stream: %d", static_cast<int>(status));
    return nullptr;
  }

  return handle;
}

/**
 * Batched matrix multiplication wrapper for cuBLAS
 *
 * Computes: C = alpha * op(A) @ op(B) + beta * C
 * for a batch of matrices
 */
template <typename scalar_t>
cublasStatus_t batched_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const scalar_t* alpha,
    const scalar_t* A,
    int lda,
    int64_t strideA,
    const scalar_t* B,
    int ldb,
    int64_t strideB,
    const scalar_t* beta,
    scalar_t* C,
    int ldc,
    int64_t strideC,
    int batchCount);

// Specializations for different types
template <>
cublasStatus_t batched_gemm<float>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    int64_t strideA,
    const float* B,
    int ldb,
    int64_t strideB,
    const float* beta,
    float* C,
    int ldc,
    int64_t strideC,
    int batchCount) {
  return cublasSgemmStridedBatched(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      lda,
      strideA,
      B,
      ldb,
      strideB,
      beta,
      C,
      ldc,
      strideC,
      batchCount);
}

template <>
cublasStatus_t batched_gemm<__half>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const __half* alpha,
    const __half* A,
    int lda,
    int64_t strideA,
    const __half* B,
    int ldb,
    int64_t strideB,
    const __half* beta,
    __half* C,
    int ldc,
    int64_t strideC,
    int batchCount) {
  return cublasHgemmStridedBatched(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      lda,
      strideA,
      B,
      ldb,
      strideB,
      beta,
      C,
      ldc,
      strideC,
      batchCount);
}

// Note: BFloat16 uses compute type float internally
template <>
cublasStatus_t batched_gemm<__nv_bfloat16>(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const __nv_bfloat16* alpha,
    const __nv_bfloat16* A,
    int lda,
    int64_t strideA,
    const __nv_bfloat16* B,
    int ldb,
    int64_t strideB,
    const __nv_bfloat16* beta,
    __nv_bfloat16* C,
    int ldc,
    int64_t strideC,
    int batchCount) {
// cuBLAS BFloat16 GEMM - introduced in CUDA 11+
#if CUDA_VERSION >= 11000
  // For BFloat16, we need to use cublasGemmStridedBatchedEx
  // with compute type CUBLAS_COMPUTE_32F
  float alpha_f = 1.0f;
  float beta_f = 0.0f;

  return cublasGemmStridedBatchedEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      &alpha_f,
      A,
      CUDA_R_16BF,
      lda,
      strideA,
      B,
      CUDA_R_16BF,
      ldb,
      strideB,
      &beta_f,
      C,
      CUDA_R_16BF,
      ldc,
      strideC,
      batchCount,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
#else
  ET_LOG(Error, "BFloat16 GEMM requires CUDA 11.0 or later");
  return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
}

// ============================================================================
// Flash Attention Implementation
// ============================================================================

/**
 * Flash Attention kernel - memory-efficient attention computation
 *
 * This kernel implements the Flash Attention algorithm which computes
 * attention in blocks to reduce memory usage from O(N^2) to O(N).
 */
template <typename scalar_t, int BLOCK_SIZE = 64>
__global__ void flash_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ output,
    const int64_t seq_len_q,
    const int64_t seq_len_k,
    const int64_t head_dim,
    const int64_t head_dim_v,
    const float scale,
    const bool is_causal) {
  const int64_t batch_head_idx = blockIdx.z;
  const int64_t q_block_idx = blockIdx.y;
  const int64_t q_start = q_block_idx * BLOCK_SIZE;
  const int64_t q_end = min(q_start + BLOCK_SIZE, seq_len_q);
  const int64_t q_block_size = q_end - q_start;
  const int tid = threadIdx.x;

  extern __shared__ char shared_mem[];
  scalar_t* shared_q = reinterpret_cast<scalar_t*>(shared_mem);
  scalar_t* shared_k = shared_q + BLOCK_SIZE * head_dim;

  const scalar_t* q_base = query + batch_head_idx * seq_len_q * head_dim;
  const scalar_t* k_base = key + batch_head_idx * seq_len_k * head_dim;
  const scalar_t* v_base = value + batch_head_idx * seq_len_k * head_dim_v;
  scalar_t* out_base = output + batch_head_idx * seq_len_q * head_dim_v;

  for (int i = tid; i < q_block_size * head_dim; i += blockDim.x) {
    int q_idx = i / head_dim;
    int d_idx = i % head_dim;
    if (q_start + q_idx < seq_len_q) {
      shared_q[i] = q_base[(q_start + q_idx) * head_dim + d_idx];
    }
  }
  __syncthreads();

  for (int64_t q_local = 0; q_local < q_block_size; ++q_local) {
    if (tid != 0)
      continue;

    const int64_t q_idx = q_start + q_local;
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float output_acc[64];
    for (int d = 0; d < head_dim_v; ++d) {
      output_acc[d] = 0.0f;
    }

    const int64_t k_blocks = (seq_len_k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int64_t k_block_idx = 0; k_block_idx < k_blocks; ++k_block_idx) {
      const int64_t k_start = k_block_idx * BLOCK_SIZE;
      const int64_t k_end = min(k_start + BLOCK_SIZE, seq_len_k);
      const int64_t k_block_size = k_end - k_start;

      float block_scores[64];
      float block_max = -FLT_MAX;

      for (int64_t k_local = 0; k_local < k_block_size; ++k_local) {
        const int64_t k_idx = k_start + k_local;
        if (is_causal && k_idx > q_idx) {
          block_scores[k_local] = -FLT_MAX;
          continue;
        }

        float score = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
          float q_val = static_cast<float>(shared_q[q_local * head_dim + d]);
          float k_val = static_cast<float>(k_base[k_idx * head_dim + d]);
          score += q_val * k_val;
        }
        score *= scale;
        block_scores[k_local] = score;
        block_max = fmaxf(block_max, score);
      }

      float new_max = fmaxf(max_score, block_max);
      float exp_correction = expf(max_score - new_max);

      for (int d = 0; d < head_dim_v; ++d) {
        output_acc[d] *= exp_correction;
      }
      sum_exp *= exp_correction;

      for (int64_t k_local = 0; k_local < k_block_size; ++k_local) {
        const int64_t k_idx = k_start + k_local;
        if (is_causal && k_idx > q_idx)
          continue;

        float exp_score = expf(block_scores[k_local] - new_max);
        sum_exp += exp_score;

        for (int64_t d = 0; d < head_dim_v; ++d) {
          float v_val = static_cast<float>(v_base[k_idx * head_dim_v + d]);
          output_acc[d] += exp_score * v_val;
        }
      }
      max_score = new_max;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int64_t d = 0; d < head_dim_v; ++d) {
      out_base[q_idx * head_dim_v + d] =
          static_cast<scalar_t>(output_acc[d] * inv_sum);
    }
  }
}

/**
 * Flash Attention implementation dispatcher
 */
template <typename scalar_t>
Tensor* sdpa_flash_attention_impl(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    float scale_factor,
    cudaStream_t stream) {
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim = query->size(3);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_v = value->size(3);

  Tensor* output = nullptr;
  std::array<int64_t, 4> output_shape = {
      batch, num_heads, seq_len_q, head_dim_v};
  std::array<int64_t, 4> output_stride = {
      num_heads * seq_len_q * head_dim_v,
      seq_len_q * head_dim_v,
      head_dim_v,
      1};

  auto dtype_int = static_cast<int32_t>(query->dtype());
  aoti_torch_empty_strided(
      4,
      output_shape.data(),
      output_stride.data(),
      dtype_int,
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &output);

  if (output == nullptr) {
    printf("[FLASH ATTN ERROR] Failed to allocate output tensor\n");
    fflush(stdout);
    ET_LOG(Error, "sdpa_flash_attention: Failed to allocate output tensor");
    return nullptr;
  }

  constexpr int BLOCK_SIZE = 64;
  const int64_t q_blocks = (seq_len_q + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int64_t batch_head_count = batch * num_heads;
  const size_t shared_mem_size = BLOCK_SIZE * head_dim * sizeof(scalar_t) * 2 +
      BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

  dim3 grid(1, q_blocks, batch_head_count);
  dim3 block(256);

  const scalar_t* q_ptr = reinterpret_cast<const scalar_t*>(query->data_ptr());
  const scalar_t* k_ptr = reinterpret_cast<const scalar_t*>(key->data_ptr());
  const scalar_t* v_ptr = reinterpret_cast<const scalar_t*>(value->data_ptr());
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(output->data_ptr());

  // //
  // ============================================================================
  // // DEBUG: Print all kernel launch parameters
  // //
  // ============================================================================
  // printf("\n=== FLASH ATTENTION KERNEL LAUNCH DEBUG ===\n");

  // // Print tensor metadata
  // printf("\nQuery Tensor:\n");
  // printf("  ptr: %p\n", (void*)q_ptr);
  // printf(
  //     "  shape: [%ld, %ld, %ld, %ld]\n", batch, num_heads, seq_len_q,
  //     head_dim);
  // printf(
  //     "  dtype: %d (size=%zu bytes)\n",
  //     static_cast<int>(query->dtype()),
  //     sizeof(scalar_t));
  // printf(
  //     "  strides: [%d, %d, %d, %d]\n",
  //     query->strides()[0],
  //     query->strides()[1],
  //     query->strides()[2],
  //     query->strides()[3]);
  // printf("  total_elements: %ld\n", batch * num_heads * seq_len_q *
  // head_dim);

  // printf("\nKey Tensor:\n");
  // printf("  ptr: %p\n", (void*)k_ptr);
  // printf(
  //     "  shape: [%ld, %ld, %ld, %ld]\n", batch, num_heads, seq_len_k,
  //     head_dim);
  // printf(
  //     "  dtype: %d (size=%zu bytes)\n",
  //     static_cast<int>(key->dtype()),
  //     sizeof(scalar_t));
  // printf(
  //     "  strides: [%d, %d, %d, %d]\n",
  //     key->strides()[0],
  //     key->strides()[1],
  //     key->strides()[2],
  //     key->strides()[3]);
  // printf("  total_elements: %ld\n", batch * num_heads * seq_len_k *
  // head_dim);

  // printf("\nValue Tensor:\n");
  // printf("  ptr: %p\n", (void*)v_ptr);
  // printf(
  //     "  shape: [%ld, %ld, %ld, %ld]\n",
  //     batch,
  //     num_heads,
  //     seq_len_k,
  //     head_dim_v);
  // printf(
  //     "  dtype: %d (size=%zu bytes)\n",
  //     static_cast<int>(value->dtype()),
  //     sizeof(scalar_t));
  // printf(
  //     "  strides: [%d, %d, %d, %d]\n",
  //     value->strides()[0],
  //     value->strides()[1],
  //     value->strides()[2],
  //     value->strides()[3]);
  // printf("  total_elements: %ld\n", batch * num_heads * seq_len_k *
  // head_dim_v);

  // printf("\nOutput Tensor:\n");
  // printf("  ptr: %p\n", (void*)out_ptr);
  // printf(
  //     "  shape: [%ld, %ld, %ld, %ld]\n",
  //     batch,
  //     num_heads,
  //     seq_len_q,
  //     head_dim_v);
  // printf(
  //     "  dtype: %d (size=%zu bytes)\n",
  //     static_cast<int>(output->dtype()),
  //     sizeof(scalar_t));
  // printf(
  //     "  expected_strides: [%ld, %ld, %ld, %ld]\n",
  //     output_stride[0],
  //     output_stride[1],
  //     output_stride[2],
  //     output_stride[3]);
  // printf("  total_elements: %ld\n", batch * num_heads * seq_len_q *
  // head_dim_v);

  // // Print scalar parameters
  // printf("\nScalar Parameters:\n");
  // printf("  seq_len_q: %ld\n", seq_len_q);
  // printf("  seq_len_k: %ld\n", seq_len_k);
  // printf("  head_dim: %ld\n", head_dim);
  // printf("  head_dim_v: %ld\n", head_dim_v);
  // printf("  scale_factor: %.10f\n", scale_factor);
  // printf("  is_causal: %d\n", is_causal);

  // // Print grid/block configuration
  // printf("\nKernel Configuration:\n");
  // printf("  BLOCK_SIZE: %d\n", BLOCK_SIZE);
  // printf("  grid: (%d, %ld, %ld)\n", 1, q_blocks, batch_head_count);
  // printf("  block: (%d, 1, 1)\n", 256);
  // printf("  shared_mem_size: %zu bytes\n", shared_mem_size);
  // printf("  stream: %p\n", (void*)stream);

  // // Print derived values
  // printf("\nDerived Values:\n");
  // printf("  batch: %ld\n", batch);
  // printf("  num_heads: %ld\n", num_heads);
  // printf("  q_blocks: %ld\n", q_blocks);
  // printf("  batch_head_count: %ld\n", batch_head_count);

  // // Check for potential issues
  // printf("\nValidation Checks:\n");
  // bool has_issues = false;

  // if (q_ptr == nullptr) {
  //   printf("  ✗ ERROR: query pointer is NULL!\n");
  //   has_issues = true;
  // } else {
  //   printf("  ✓ query pointer is valid\n");
  // }

  // if (k_ptr == nullptr) {
  //   printf("  ✗ ERROR: key pointer is NULL!\n");
  //   has_issues = true;
  // } else {
  //   printf("  ✓ key pointer is valid\n");
  // }

  // if (v_ptr == nullptr) {
  //   printf("  ✗ ERROR: value pointer is NULL!\n");
  //   has_issues = true;
  // } else {
  //   printf("  ✓ value pointer is valid\n");
  // }

  // if (out_ptr == nullptr) {
  //   printf("  ✗ ERROR: output pointer is NULL!\n");
  //   has_issues = true;
  // } else {
  //   printf("  ✓ output pointer is valid\n");
  // }

  // if (head_dim_v > 64) {
  //   printf(
  //       "  ✗ WARNING: head_dim_v=%ld > 64 (max supported by kernel)\n",
  //       head_dim_v);
  //   has_issues = true;
  // } else {
  //   printf("  ✓ head_dim_v=%ld is within supported range\n", head_dim_v);
  // }

  // if (head_dim > 128) {
  //   printf("  ✗ WARNING: head_dim=%ld > 128 (recommended max)\n", head_dim);
  // } else {
  //   printf("  ✓ head_dim=%ld is within recommended range\n", head_dim);
  // }

  // if (shared_mem_size > 48 * 1024) {
  //   printf(
  //       "  ✗ WARNING: shared_mem_size=%zu > 48KB (may fail on some GPUs)\n",
  //       shared_mem_size);
  //   has_issues = true;
  // } else {
  //   printf("  ✓ shared_mem_size=%zu is reasonable\n", shared_mem_size);
  // }

  // if (has_issues) {
  //   printf("\n⚠️  POTENTIAL ISSUES DETECTED - see above\n");
  // } else {
  //   printf("\n✓ All validation checks passed\n");
  // }

  // printf("\n===========================================\n\n");
  // fflush(stdout);

  // //
  // ============================================================================
  // // Check for Inf/NaN in input tensors
  // //
  // ============================================================================
  // printf("Checking input tensors for Inf/NaN values...\n");
  // fflush(stdout);

  // bool has_inf_nan = false;

  // if (query->dtype() == executorch::aten::ScalarType::Float) {
  //   has_inf_nan |= check_tensor_for_inf_nan<float>(query, "Query", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<float>(key, "Key", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<float>(value, "Value", stream);
  // } else if (query->dtype() == executorch::aten::ScalarType::Half) {
  //   has_inf_nan |= check_tensor_for_inf_nan<__half>(query, "Query", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<__half>(key, "Key", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<__half>(value, "Value", stream);
  // } else if (query->dtype() == executorch::aten::ScalarType::BFloat16) {
  //   has_inf_nan |=
  //       check_tensor_for_inf_nan<__nv_bfloat16>(query, "Query", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<__nv_bfloat16>(key, "Key",
  //   stream); has_inf_nan |=
  //       check_tensor_for_inf_nan<__nv_bfloat16>(value, "Value", stream);
  // }

  // if (has_inf_nan) {
  //   printf("\n❌ [FATAL] Input tensors contain Inf or NaN values!\n");
  //   printf(
  //       "   This WILL cause kernel launch failure with 'invalid argument'
  //       error.\n");
  //   printf("   Aborting Flash Attention kernel launch.\n\n");
  //   fflush(stdout);

  //   aoti_torch_delete_tensor_object(output);
  //   ET_LOG(
  //       Error, "sdpa_flash_attention: Input tensors contain Inf or NaN
  //       values");
  //   return nullptr;
  // }

  // printf("✓ All input tensors are valid (no Inf/NaN detected)\n\n");
  // fflush(stdout);

  // ============================================================================
  // DEBUG LOGGING: Log inputs before kernel call
  // ============================================================================
  // auto log_info = log_kernel_call_inputs<scalar_t>(
  //     "flash_attention",
  //     query,
  //     key,
  //     value,
  //     attn_mask, // Will be nullptr for flash attention
  //     scale_factor,
  //     is_causal,
  //     stream);
  // int call_id = log_info.first;
  // std::string base_path = log_info.second;

  // ============================================================================
  // Launch kernel
  // ============================================================================
  flash_attention_kernel<scalar_t, BLOCK_SIZE>
      <<<grid, block, shared_mem_size, stream>>>(
          q_ptr,
          k_ptr,
          v_ptr,
          out_ptr,
          seq_len_q,
          seq_len_k,
          head_dim,
          head_dim_v,
          scale_factor,
          is_causal);

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf(
        "[FLASH ATTN ERROR] Kernel launch failed: %s\n",
        cudaGetErrorString(cuda_err));
    fflush(stdout);
    ET_LOG(
        Error,
        "sdpa_flash_attention: Kernel launch failed: %s",
        cudaGetErrorString(cuda_err));
    aoti_torch_delete_tensor_object(output);
    return nullptr;
  }

  // //
  // ============================================================================
  // // DEBUG LOGGING: Log output after kernel call
  // //
  // ============================================================================
  // log_kernel_call_output<scalar_t>(call_id, base_path, output, stream);

  return output;
}

/**
 * Flash Attention entry point with dtype dispatch
 */
Tensor* sdpa_flash_attention(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    double scale_factor,
    cudaStream_t stream) {
  auto dtype = query->dtype();
  if (dtype == executorch::aten::ScalarType::Float) {
    return sdpa_flash_attention_impl<float>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::Half) {
    return sdpa_flash_attention_impl<__half>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::BFloat16) {
    return sdpa_flash_attention_impl<__nv_bfloat16>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else {
    ET_LOG(Error, "sdpa_flash_attention: Unsupported dtype");
    return nullptr;
  }
}

// ============================================================================
// Memory-Efficient Attention Implementation (with attn_bias support)
// ============================================================================

/**
 * Memory-Efficient Attention kernel with attention bias support
 *
 * This kernel computes scaled dot-product attention with full support for
 * attention bias (additive mask). It uses online softmax for numerical
 * stability.
 *
 * Each thread processes one query position independently.
 *
 * Input shapes:
 * - query: [batch, num_heads, seq_len_q, head_dim]
 * - key: [batch, num_heads, seq_len_k, head_dim]
 * - value: [batch, num_heads, seq_len_k, head_dim_v]
 * - attn_bias: [batch, num_heads, seq_len_q, seq_len_k] (optional)
 * - output: [batch, num_heads, seq_len_q, head_dim_v]
 */
template <typename scalar_t, int MAX_HEAD_DIM_V = 128>
__global__ void efficient_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const scalar_t* __restrict__ attn_bias,
    scalar_t* __restrict__ output,
    const int64_t num_heads,
    const int64_t seq_len_q,
    const int64_t seq_len_k,
    const int64_t head_dim,
    const int64_t head_dim_v,
    const float scale,
    const int is_causal,
    // Query strides [batch, head, seq, dim]
    const int64_t q_stride_batch,
    const int64_t q_stride_head,
    const int64_t q_stride_seq,
    const int64_t q_stride_dim,
    // Key strides [batch, head, seq, dim]
    const int64_t k_stride_batch,
    const int64_t k_stride_head,
    const int64_t k_stride_seq,
    const int64_t k_stride_dim,
    // Value strides [batch, head, seq, dim]
    const int64_t v_stride_batch,
    const int64_t v_stride_head,
    const int64_t v_stride_seq,
    const int64_t v_stride_dim,
    // Output strides [batch, head, seq, dim] - always contiguous
    const int64_t o_stride_batch,
    const int64_t o_stride_head,
    const int64_t o_stride_seq,
    const int64_t o_stride_dim,
    // Bias strides [batch, head, seq_q, seq_k]
    const int64_t bias_stride_batch,
    const int64_t bias_stride_head,
    const int64_t bias_stride_q,
    const int64_t bias_stride_k) {
  const int64_t batch_head_idx = blockIdx.x;
  const int64_t q_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (q_idx >= seq_len_q) {
    return;
  }

  // Decompose batch_head_idx into batch and head indices
  const int64_t batch_idx = batch_head_idx / num_heads;
  const int64_t head_idx = batch_head_idx % num_heads;

  // Compute base pointers using proper 4D strides
  const scalar_t* q_base =
      query + batch_idx * q_stride_batch + head_idx * q_stride_head;
  const scalar_t* k_base =
      key + batch_idx * k_stride_batch + head_idx * k_stride_head;
  const scalar_t* v_base =
      value + batch_idx * v_stride_batch + head_idx * v_stride_head;
  scalar_t* out_base =
      output + batch_idx * o_stride_batch + head_idx * o_stride_head;

  // Compute bias base pointer using proper 4D indexing with broadcasting
  // support
  const scalar_t* bias_base = nullptr;
  if (attn_bias != nullptr) {
    // Only add stride contribution if the dimension size > 1 (not broadcasting)
    int64_t bias_offset = 0;

    // Note: bias_stride will be 0 for broadcasting dimensions (size=1)
    // This is correct - we want all positions to point to the same element
    bias_offset += batch_idx * bias_stride_batch;
    bias_offset += head_idx * bias_stride_head;
    bias_offset += q_idx * bias_stride_q;

    bias_base = attn_bias + bias_offset;
  }

  float output_acc[MAX_HEAD_DIM_V];
  for (int64_t d = 0; d < head_dim_v && d < MAX_HEAD_DIM_V; ++d) {
    output_acc[d] = 0.0f;
  }

  float max_score = -FLT_MAX;
  float sum_exp = 0.0f;

  // Online softmax algorithm for numerical stability
  // Reference: "Online normalizer calculation for softmax" (Algorithm 3)
  // from "Self-attention Does Not Need O(n²) Memory" paper
  //
  // The algorithm maintains:
  // - max_score: running maximum of all scores seen so far
  // - sum_exp: sum of exp(score_i - max_score) for all i seen so far
  // - output_acc: weighted sum of values, properly rescaled
  //
  // When we see a new score that changes the max, we need to rescale
  // all previous contributions by exp(old_max - new_max)
  for (int64_t k_idx = 0; k_idx < seq_len_k; ++k_idx) {
    if (is_causal && k_idx > q_idx) {
      continue;
    }

    // Compute attention score: Q·K^T
    float score = 0.0f;
    for (int64_t d = 0; d < head_dim; ++d) {
      float q_val =
          static_cast<float>(q_base[q_idx * q_stride_seq + d * q_stride_dim]);
      float k_val =
          static_cast<float>(k_base[k_idx * k_stride_seq + d * k_stride_dim]);
      score += q_val * k_val;
    }
    score *= scale;

    // Add attention bias if provided (additive mask)
    // Note: bias_stride_k should be 1 for contiguous last dimension
    if (bias_base != nullptr) {
      float bias_val = static_cast<float>(bias_base[k_idx * bias_stride_k]);
      score += bias_val;
    }

    // Online softmax update - critical section for numerical stability
    float new_max = fmaxf(max_score, score);
    float exp_correction = expf(max_score - new_max);

    // IMPORTANT: Compute exp_score only once to avoid numerical differences
    // Previous bug: expf(score - new_max) was computed twice, which could
    // lead to different rounding in FP arithmetic, causing incorrect results
    float exp_score = expf(score - new_max);

    // Step 1: Rescale all previous accumulators by exp(old_max - new_max)
    // This maintains the invariant that all values are normalized relative
    // to the current maximum
    for (int64_t d = 0; d < head_dim_v && d < MAX_HEAD_DIM_V; ++d) {
      output_acc[d] *= exp_correction;
    }
    sum_exp *= exp_correction;

    // Step 2: Add current contribution using the already-computed exp_score
    sum_exp += exp_score;
    for (int64_t d = 0; d < head_dim_v && d < MAX_HEAD_DIM_V; ++d) {
      float v_val =
          static_cast<float>(v_base[k_idx * v_stride_seq + d * v_stride_dim]);
      output_acc[d] += exp_score * v_val;
    }

    // Update max for next iteration
    max_score = new_max;
  }

  float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
  for (int64_t d = 0; d < head_dim_v && d < MAX_HEAD_DIM_V; ++d) {
    out_base[q_idx * o_stride_seq + d * o_stride_dim] =
        static_cast<scalar_t>(output_acc[d] * inv_sum);
  }
}

/**
 * Memory-Efficient Attention implementation dispatcher
 */
template <typename scalar_t>
Tensor* sdpa_efficient_attention_impl(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_bias,
    bool is_causal,
    float scale_factor,
    cudaStream_t stream) {
  // printf("Running sdpa_efficient_attention_impl...\n");
  // // Clear any previous CUDA errors before launch
  // CHECK_CUDA_ERROR_WITH_MSG("sdpa_efficient_attention_impl");
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim = query->size(3);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_v = value->size(3);

  Tensor* output = nullptr;
  std::array<int64_t, 4> output_shape = {
      batch, num_heads, seq_len_q, head_dim_v};
  std::array<int64_t, 4> output_stride = {
      num_heads * seq_len_q * head_dim_v,
      seq_len_q * head_dim_v,
      head_dim_v,
      1};

  auto dtype_int = static_cast<int32_t>(query->dtype());
  aoti_torch_empty_strided(
      4,
      output_shape.data(),
      output_stride.data(),
      dtype_int,
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &output);

  if (output == nullptr) {
    printf("sdpa_efficient_attention: Failed to allocate output tensor\n");
    fflush(stdout);
    ET_LOG(Error, "sdpa_efficient_attention: Failed to allocate output tensor");
    return nullptr;
  }

  // CHECK_CUDA_ERROR_WITH_MSG("Line 1420");

  const int64_t batch_head_count = batch * num_heads;
  const int threads_per_block = 128;
  const int64_t q_blocks =
      (seq_len_q + threads_per_block - 1) / threads_per_block;

  dim3 grid(batch_head_count, q_blocks);
  dim3 block(threads_per_block);

  const scalar_t* q_ptr = reinterpret_cast<const scalar_t*>(query->data_ptr());
  const scalar_t* k_ptr = reinterpret_cast<const scalar_t*>(key->data_ptr());
  const scalar_t* v_ptr = reinterpret_cast<const scalar_t*>(value->data_ptr());
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(output->data_ptr());

  const scalar_t* bias_ptr = nullptr;
  int64_t bias_stride_batch = 0;
  int64_t bias_stride_head = 0;
  int64_t bias_stride_q = 0;
  int64_t bias_stride_k = 1;

  if (attn_bias != nullptr) {
    bias_ptr = reinterpret_cast<const scalar_t*>(attn_bias->data_ptr());

    int64_t bias_dim = attn_bias->dim();
    // printf("  attn_bias: ptr=%p, dim=%ld\n", (void*)bias_ptr, bias_dim);
    // fflush(stdout);
    if (bias_dim != 4) {
      printf(
          "sdpa_efficient_attention: Attention bias larger than 0 is not supported\n");
      fflush(stdout);
      ET_LOG(
          Error,
          "sdpa_efficient_attention: Attention bias larger than 0 is not supported\n");
      return nullptr;
    }

    // Handle attention bias with shape [batch, num_heads, seq_len_q,
    // seq_len_k] or broadcastable variants
    auto bias_strides = attn_bias->strides();
    auto bias_sizes = attn_bias->sizes();

    // Extract sizes safely
    int64_t bias_size_0 = bias_sizes[0];
    int64_t bias_size_1 = bias_sizes[1];
    int64_t bias_size_2 = bias_sizes[2];
    int64_t bias_size_3 = bias_sizes[3];

    // Extract strides safely
    int64_t bias_stride_0 = bias_strides[0];
    int64_t bias_stride_1 = bias_strides[1];
    int64_t bias_stride_2 = bias_strides[2];
    int64_t bias_stride_3 = bias_strides[3];

    // printf(
    //     "  bias dim=4, sizes=[%ld, %ld, %ld, %ld], strides=[%ld, %ld, %ld,
    //     %ld]\n", bias_size_0, bias_size_1, bias_size_2, bias_size_3,
    //     bias_stride_0,
    //     bias_stride_1,
    //     bias_stride_2,
    //     bias_stride_3);
    // fflush(stdout);

    bias_stride_batch = bias_stride_0;
    bias_stride_head = bias_stride_1;
    bias_stride_q = bias_stride_2;
    bias_stride_k = bias_stride_3;
  }

  // CHECK_CUDA_ERROR_WITH_MSG("Line 1507");

  // Debug: Print query tensor info
  auto query_strides_raw = query->strides();
  auto key_strides_raw = key->strides();
  auto value_strides_raw = value->strides();

  // Convert strides from int32_t to int64_t for kernel
  int64_t query_strides[4] = {
      static_cast<int64_t>(query_strides_raw[0]),
      static_cast<int64_t>(query_strides_raw[1]),
      static_cast<int64_t>(query_strides_raw[2]),
      static_cast<int64_t>(query_strides_raw[3])};
  int64_t key_strides[4] = {
      static_cast<int64_t>(key_strides_raw[0]),
      static_cast<int64_t>(key_strides_raw[1]),
      static_cast<int64_t>(key_strides_raw[2]),
      static_cast<int64_t>(key_strides_raw[3])};
  int64_t value_strides[4] = {
      static_cast<int64_t>(value_strides_raw[0]),
      static_cast<int64_t>(value_strides_raw[1]),
      static_cast<int64_t>(value_strides_raw[2]),
      static_cast<int64_t>(value_strides_raw[3])};

  // printf("Launching efficient_attention_kernel:\n");
  // printf(
  //     "  batch=%ld, num_heads=%ld, seq_len_q=%ld, seq_len_k=%ld,
  //     head_dim=%ld, head_dim_v=%ld\n", batch, num_heads, seq_len_q,
  //     seq_len_k,
  //     head_dim,
  //     head_dim_v);
  // printf(
  //     "  query_strides=[%ld, %ld, %ld, %ld]\n",
  //     query_strides[0],
  //     query_strides[1],
  //     query_strides[2],
  //     query_strides[3]);
  // printf(
  //     "  key_strides=[%ld, %ld, %ld, %ld]\n",
  //     key_strides[0],
  //     key_strides[1],
  //     key_strides[2],
  //     key_strides[3]);
  // printf(
  //     "  value_strides=[%ld, %ld, %ld, %ld]\n",
  //     value_strides[0],
  //     value_strides[1],
  // Debug: Print query tensor info
  // auto query_strides = query->strides();
  // auto key_strides = key->strides();
  // auto value_strides = value->strides();

  // printf("\n=== Efficient Attention Kernel Launch Details ===\n");
  // printf("Tensor Dimensions:\n");
  // printf(
  //     "  batch=%ld, num_heads=%ld, seq_len_q=%ld, seq_len_k=%ld\n",
  //     batch,
  //     num_heads,
  //     seq_len_q,
  //     seq_len_k);
  // printf("  head_dim=%ld, head_dim_v=%ld\n", head_dim, head_dim_v);

  // printf(
  //     "\nQuery Tensor (shape: [%ld, %ld, %ld, %ld]):\n",
  //     batch,
  //     num_heads,
  //     seq_len_q,
  //     head_dim);
  // printf(
  //     "  strides=[%ld, %ld, %ld, %ld] (converted from int32_t to int64_t)\n",
  //     query_strides[0],
  //     query_strides[1],
  //     query_strides[2],
  //     query_strides[3]);

  // printf(
  //     "\nKey Tensor (shape: [%ld, %ld, %ld, %ld]):\n",
  //     batch,
  //     num_heads,
  //     seq_len_k,
  //     head_dim);
  // printf(
  //     "  strides=[%ld, %ld, %ld, %ld] (converted from int32_t to int64_t)\n",
  //     key_strides[0],
  //     key_strides[1],
  //     key_strides[2],
  //     key_strides[3]);

  // printf(
  //     "\nValue Tensor (shape: [%ld, %ld, %ld, %ld]):\n",
  //     batch,
  //     num_heads,
  //     seq_len_k,
  //     head_dim_v);
  // printf(
  //     "  strides=[%ld, %ld, %ld, %ld] (converted from int32_t to int64_t)\n",
  //     value_strides[0],
  //     value_strides[1],
  //     value_strides[2],
  //     value_strides[3]);

  // if (attn_bias != nullptr) {
  //   auto bias_sizes = attn_bias->sizes();
  //   printf("\nAttention Bias Tensor (shape: [");
  //   for (int64_t i = 0; i < attn_bias->dim(); ++i) {
  //     printf("%d", bias_sizes[i]);
  //     if (i < attn_bias->dim() - 1)
  //       printf(", ");
  //   }
  //   printf("]):\n");
  //   printf("  bias_ptr=%p\n", (void*)bias_ptr);
  //   printf(
  //       "  strides=[batch:%ld, head:%ld, q:%ld, k:%ld]\n",
  //       bias_stride_batch,
  //       bias_stride_head,
  //       bias_stride_q,
  //       bias_stride_k);
  //   printf("  dtype: %d\n", static_cast<int>(attn_bias->dtype()));
  // } else {
  //   printf("\nAttention Bias: None (nullptr)\n");
  // }

  // printf("\nKernel Configuration:\n");
  // printf("  scale_factor=%.6f\n", scale_factor);
  // printf("  is_causal=%d\n", is_causal);
  // printf(
  //     "  grid=(%ld, %ld, 1) [batch_head_count=%ld, q_blocks=%ld]\n",
  //     batch_head_count,
  //     q_blocks,
  //     batch_head_count,
  //     q_blocks);
  // printf(
  //     "  block=(%d, 1, 1) [threads_per_block=%d]\n",
  //     threads_per_block,
  //     threads_per_block);

  // // Verify that Q/K/V are contiguous in the batch_head*seq*dim layout
  // bool q_is_contiguous =
  //     (query_strides[0] == num_heads * seq_len_q * head_dim) &&
  //     (query_strides[1] == seq_len_q * head_dim) &&
  //     (query_strides[2] == head_dim) && (query_strides[3] == 1);
  // bool k_is_contiguous = (key_strides[0] == num_heads * seq_len_k * head_dim)
  // &&
  //     (key_strides[1] == seq_len_k * head_dim) &&
  //     (key_strides[2] == head_dim) && (key_strides[3] == 1);
  // bool v_is_contiguous =
  //     (value_strides[0] == num_heads * seq_len_k * head_dim_v) &&
  //     (value_strides[1] == seq_len_k * head_dim_v) &&
  //     (value_strides[2] == head_dim_v) && (value_strides[3] == 1);

  // printf("\nMemory Layout Check:\n");
  // printf("  Q is contiguous: %s\n", q_is_contiguous ? "YES" : "NO");
  // printf("  K is contiguous: %s\n", k_is_contiguous ? "YES" : "NO");
  // printf("  V is contiguous: %s\n", v_is_contiguous ? "YES" : "NO");

  // if (!q_is_contiguous || !k_is_contiguous || !v_is_contiguous) {
  //   printf(
  //       "  WARNING: Non-contiguous tensor detected! Kernel will use
  //       stride-based indexing.\n");
  // }
  // printf("==============================================\n\n");
  // fflush(stdout);

  // //
  // ============================================================================
  // // Check for Inf/NaN in input tensors
  // //
  // ============================================================================
  // printf("Checking input tensors for Inf/NaN values...\n");
  // fflush(stdout);

  // CHECK_CUDA_ERROR_WITH_MSG("Line 1676");

  // bool has_inf_nan = false;

  // if (query->dtype() == executorch::aten::ScalarType::Float) {
  //   has_inf_nan |= check_tensor_for_inf_nan<float>(query, "Query", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<float>(key, "Key", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<float>(value, "Value", stream);
  //   // if (attn_bias != nullptr) {
  //   //   has_inf_nan |=
  //   //       check_tensor_for_inf_nan<float>(attn_bias, "AttnBias", stream);
  //   // }
  // } else if (query->dtype() == executorch::aten::ScalarType::Half) {
  //   has_inf_nan |= check_tensor_for_inf_nan<__half>(query, "Query", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<__half>(key, "Key", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<__half>(value, "Value", stream);
  //   // if (attn_bias != nullptr) {
  //   //   has_inf_nan |=
  //   //       check_tensor_for_inf_nan<__half>(attn_bias, "AttnBias", stream);
  //   // }
  // } else if (query->dtype() == executorch::aten::ScalarType::BFloat16) {
  //   has_inf_nan |=
  //       check_tensor_for_inf_nan<__nv_bfloat16>(query, "Query", stream);
  //   has_inf_nan |= check_tensor_for_inf_nan<__nv_bfloat16>(key, "Key",
  //   stream); has_inf_nan |=
  //       check_tensor_for_inf_nan<__nv_bfloat16>(value, "Value", stream);
  //   // if (attn_bias != nullptr) {
  //   //   has_inf_nan |= check_tensor_for_inf_nan<__nv_bfloat16>(
  //   //       attn_bias, "AttnBias", stream);
  //   // }
  // }

  // if (has_inf_nan) {
  //   printf("\n❌ [FATAL] Input tensors contain Inf or NaN values!\n");
  //   printf(
  //       "   This WILL cause kernel launch failure with 'invalid argument'
  //       error.\n");
  //   printf("   Aborting Efficient Attention kernel launch.\n\n");
  //   fflush(stdout);

  //   aoti_torch_delete_tensor_object(output);
  //   ET_LOG(
  //       Error,
  //       "sdpa_efficient_attention: Input tensors contain Inf or NaN values");
  //   return nullptr;
  // }

  // printf("✓ All input tensors are valid (no Inf/NaN detected)\n\n");
  // fflush(stdout);

  // CHECK_CUDA_ERROR_WITH_MSG("Line 1725");

  // //
  // ============================================================================
  // // DEBUG LOGGING: Log inputs before kernel call
  // //
  // ============================================================================
  // auto log_info = log_kernel_call_inputs<scalar_t>(
  //     "efficient_attention",
  //     query,
  //     key,
  //     value,
  //     attn_bias,
  //     scale_factor,
  //     is_causal,
  //     stream);

  // CHECK_CUDA_ERROR_WITH_MSG("Line 1740");
  // int call_id = log_info.first;
  // std::string base_path = log_info.second;

  // Output strides (always contiguous)
  int64_t o_stride_batch = num_heads * seq_len_q * head_dim_v;
  int64_t o_stride_head = seq_len_q * head_dim_v;
  int64_t o_stride_seq = head_dim_v;
  int64_t o_stride_dim = 1;

  // //
  // ============================================================================
  // // DETAILED PARAMETER DUMP - Print all parameters that will be passed to
  // // kernel
  // //
  // ============================================================================
  // printf("\n=== KERNEL PARAMETER DUMP ===\n");
  // printf("Pointer arguments:\n");
  // printf("  q_ptr=%p\n", (void*)q_ptr);
  // printf("  k_ptr=%p\n", (void*)k_ptr);
  // printf("  v_ptr=%p\n", (void*)v_ptr);
  // printf("  bias_ptr=%p\n", (void*)bias_ptr);
  // printf("  out_ptr=%p\n", (void*)out_ptr);

  // printf("\nDimension arguments (int64_t):\n");
  // printf("  num_heads=%ld (0x%lx)\n", num_heads, num_heads);
  // printf("  seq_len_q=%ld (0x%lx)\n", seq_len_q, seq_len_q);
  // printf("  seq_len_k=%ld (0x%lx)\n", seq_len_k, seq_len_k);
  // printf("  head_dim=%ld (0x%lx)\n", head_dim, head_dim);
  // printf("  head_dim_v=%ld (0x%lx)\n", head_dim_v, head_dim_v);

  // printf("\nScale and flags:\n");
  // printf("  scale_factor=%.10f\n", scale_factor);
  // printf("  is_causal=%d\n", is_causal);

  // printf("\nQuery strides (int64_t):\n");
  // printf("  q_stride_batch=%ld (0x%lx)\n", query_strides[0],
  // query_strides[0]); printf("  q_stride_head=%ld (0x%lx)\n",
  // query_strides[1], query_strides[1]); printf("  q_stride_seq=%ld (0x%lx)\n",
  // query_strides[2], query_strides[2]); printf("  q_stride_dim=%ld (0x%lx)\n",
  // query_strides[3], query_strides[3]);

  // printf("\nKey strides (int64_t):\n");
  // printf("  k_stride_batch=%ld (0x%lx)\n", key_strides[0], key_strides[0]);
  // printf("  k_stride_head=%ld (0x%lx)\n", key_strides[1], key_strides[1]);
  // printf("  k_stride_seq=%ld (0x%lx)\n", key_strides[2], key_strides[2]);
  // printf("  k_stride_dim=%ld (0x%lx)\n", key_strides[3], key_strides[3]);

  // printf("\nValue strides (int64_t):\n");
  // printf("  v_stride_batch=%ld (0x%lx)\n", value_strides[0],
  // value_strides[0]); printf("  v_stride_head=%ld (0x%lx)\n",
  // value_strides[1], value_strides[1]); printf("  v_stride_seq=%ld (0x%lx)\n",
  // value_strides[2], value_strides[2]); printf("  v_stride_dim=%ld (0x%lx)\n",
  // value_strides[3], value_strides[3]);

  // printf("\nOutput strides (int64_t):\n");
  // printf("  o_stride_batch=%ld (0x%lx)\n", o_stride_batch, o_stride_batch);
  // printf("  o_stride_head=%ld (0x%lx)\n", o_stride_head, o_stride_head);
  // printf("  o_stride_seq=%ld (0x%lx)\n", o_stride_seq, o_stride_seq);
  // printf("  o_stride_dim=%ld (0x%lx)\n", o_stride_dim, o_stride_dim);

  // printf("\nBias strides (int64_t):\n");
  // printf(
  //     "  bias_stride_batch=%ld (0x%lx)\n",
  //     bias_stride_batch,
  //     bias_stride_batch);
  // printf(
  //     "  bias_stride_head=%ld (0x%lx)\n", bias_stride_head,
  //     bias_stride_head);
  // printf("  bias_stride_q=%ld (0x%lx)\n", bias_stride_q, bias_stride_q);
  // printf("  bias_stride_k=%ld (0x%lx)\n", bias_stride_k, bias_stride_k);

  // printf("\nParameter count: %d arguments total\n", 29);
  // printf("===========================\n\n");
  // fflush(stdout);

  // CHECK_CUDA_ERROR_WITH_MSG("Line 1809");

  // //
  // ============================================================================
  // // DTYPE VALIDATION: Check that attn_bias has the same dtype as Q/K/V
  // //
  // ============================================================================
  // if (attn_bias != nullptr) {
  //   auto query_dtype = query->dtype();
  //   auto bias_dtype = attn_bias->dtype();
  //   if (query_dtype != bias_dtype) {
  //     printf("\n❌ [CRITICAL ERROR] dtype mismatch detected!\n");
  //     printf("   Query dtype: %d\n", static_cast<int>(query_dtype));
  //     printf("   AttnBias dtype: %d\n", static_cast<int>(bias_dtype));
  //     printf("   (Float=6, Half=5, BFloat16=15)\n");
  //     printf("   This WILL cause 'invalid argument' error!\n");
  //     printf(
  //         "   The kernel assumes attn_bias has the same dtype as
  //         Q/K/V.\n\n");
  //     fflush(stdout);

  //     ET_LOG(
  //         Error,
  //         "sdpa_efficient_attention: attn_bias dtype (%d) does not match
  //         query dtype (%d)", static_cast<int>(bias_dtype),
  //         static_cast<int>(query_dtype));
  //     aoti_torch_delete_tensor_object(output);
  //     return nullptr;
  //   }
  // }

  // //
  // ============================================================================
  // // PRE-LAUNCH VALIDATION: Check CUDA device properties and kernel
  // requirements
  // //
  // ============================================================================
  // {
  //   int device_id = 0;
  //   cudaGetDevice(&device_id);
  //   cudaDeviceProp prop;
  //   cudaGetDeviceProperties(&prop, device_id);

  //   printf("\n=== CUDA Device Validation ===\n");
  //   printf("Device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
  //   printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  //   printf(
  //       "Max grid dimensions: [%d, %d, %d]\n",
  //       prop.maxGridSize[0],
  //       prop.maxGridSize[1],
  //       prop.maxGridSize[2]);
  //   printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);

  //   // Validate grid dimensions
  //   if (batch_head_count > prop.maxGridSize[0]) {
  //     printf(
  //         "❌ ERROR: grid.x (%ld) exceeds max (%d)\n",
  //         batch_head_count,
  //         prop.maxGridSize[0]);
  //     fflush(stdout);
  //     return nullptr;
  //   }
  //   if (q_blocks > prop.maxGridSize[1]) {
  //     printf(
  //         "❌ ERROR: grid.y (%ld) exceeds max (%d)\n",
  //         q_blocks,
  //         prop.maxGridSize[1]);
  //     fflush(stdout);
  //     return nullptr;
  //   }

  //   // Validate block dimensions
  //   if (threads_per_block > prop.maxThreadsPerBlock) {
  //     printf(
  //         "❌ ERROR: block size (%d) exceeds max (%d)\n",
  //         threads_per_block,
  //         prop.maxThreadsPerBlock);
  //     fflush(stdout);
  //     return nullptr;
  //   }

  //   printf("================================\n\n");
  //   fflush(stdout);
  // }

  // // Clear any previous CUDA errors before launch
  // cudaError_t pre_err = cudaGetLastError();
  // if (pre_err != cudaSuccess) {
  //   printf(
  //       "⚠️  WARNING: Pre-existing CUDA error before kernel launch: %s\n",
  //       cudaGetErrorString(pre_err));
  //   fflush(stdout);
  //   return nullptr;
  // }

  // printf(
  //     "About to launch kernel with MAX_HEAD_DIM_V=%d...\n",
  //     head_dim_v <= 64 ? 64 : 128);
  // printf("  scalar_t size: %zu bytes\n", sizeof(scalar_t));
  // printf(
  //     "  Template instantiation: efficient_attention_kernel<%s, %d>\n",
  //     sizeof(scalar_t) == 4
  //         ? "float"
  //         : (sizeof(scalar_t) == 2 ? "half/bfloat16" : "unknown"),
  //     head_dim_v <= 64 ? 64 : 128);
  // fflush(stdout);

  // CHECK_CUDA_ERROR_WITH_MSG("Line 1910");

  if (head_dim_v <= 64) {
    efficient_attention_kernel<scalar_t, 64><<<grid, block, 0, stream>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        head_dim_v,
        scale_factor,
        is_causal,
        // Query strides
        query_strides[0],
        query_strides[1],
        query_strides[2],
        query_strides[3],
        // Key strides
        key_strides[0],
        key_strides[1],
        key_strides[2],
        key_strides[3],
        // Value strides
        value_strides[0],
        value_strides[1],
        value_strides[2],
        value_strides[3],
        // Output strides
        o_stride_batch,
        o_stride_head,
        o_stride_seq,
        o_stride_dim,
        // Bias strides
        bias_stride_batch,
        bias_stride_head,
        bias_stride_q,
        bias_stride_k);
  } else {
    efficient_attention_kernel<scalar_t, 128><<<grid, block, 0, stream>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        num_heads,
        seq_len_q,
        seq_len_k,
        head_dim,
        head_dim_v,
        scale_factor,
        is_causal,
        // Query strides
        query_strides[0],
        query_strides[1],
        query_strides[2],
        query_strides[3],
        // Key strides
        key_strides[0],
        key_strides[1],
        key_strides[2],
        key_strides[3],
        // Value strides
        value_strides[0],
        value_strides[1],
        value_strides[2],
        value_strides[3],
        // Output strides
        o_stride_batch,
        o_stride_head,
        o_stride_seq,
        o_stride_dim,
        // Bias strides
        bias_stride_batch,
        bias_stride_head,
        bias_stride_q,
        bias_stride_k);
  }

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf(
        "efficient_attention_kernel failed to launch: %s\n",
        cudaGetErrorString(cuda_err));
    fflush(stdout);
    ET_LOG(
        Error,
        "sdpa_efficient_attention: Kernel launch failed: %s",
        cudaGetErrorString(cuda_err));
    aoti_torch_delete_tensor_object(output);
    return nullptr;
  }

  // Synchronize to check for kernel execution errors
  cuda_err = cudaStreamSynchronize(stream);
  // if (cuda_err != cudaSuccess) {
  //   printf("cudaStreamSynchronize failed to execute\n");
  //   fflush(stdout);
  //   ET_LOG(
  //       Error,
  //       "sdpa_efficient_attention: Kernel execution failed: %s",
  //       cudaGetErrorString(cuda_err));
  //   aoti_torch_delete_tensor_object(output);
  //   return nullptr;
  // }

  // printf("efficient_attention_kernel completed successfully\n");
  // fflush(stdout);

  // //
  // ============================================================================
  // // DEBUG LOGGING: Log output after kernel call
  // //
  // ============================================================================
  // log_kernel_call_output<scalar_t>(call_id, base_path, output, stream);

  return output;
}

/**
 * Memory-Efficient Attention entry point with dtype dispatch
 */
Tensor* sdpa_efficient_attention(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_bias,
    bool is_causal,
    double scale_factor,
    cudaStream_t stream) {
  auto dtype = query->dtype();

  // if (dtype == executorch::aten::ScalarType::Float) {
  //   return sdpa_efficient_attention_impl<float>(
  //       query,
  //       key,
  //       value,
  //       attn_bias,
  //       is_causal,
  //       static_cast<float>(scale_factor),
  //       stream);
  // } else if (dtype == executorch::aten::ScalarType::Half) {
  //   return sdpa_efficient_attention_impl<__half>(
  //       query,
  //       key,
  //       value,
  //       attn_bias,
  //       is_causal,
  //       static_cast<float>(scale_factor),
  //       stream);
  // } else
  if (dtype == executorch::aten::ScalarType::BFloat16) {
    return sdpa_efficient_attention_impl<__nv_bfloat16>(
        query,
        key,
        value,
        attn_bias,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else {
    printf(
        "sdpa_efficient_attention: Unsupported dtype %d\n",
        static_cast<int>(dtype));
    ET_LOG(Error, "sdpa_efficient_attention: Unsupported dtype");
    return nullptr;
  }
}

// ============================================================================
// Math Fallback Implementation
// ============================================================================

/**
 * Math fallback implementation for SDPA
 *
 * This implementation uses cuBLAS for matrix multiplications and custom
 * kernels for softmax. It provides maximum compatibility across all CUDA
 * devices but may not be as optimized as Flash Attention or Memory Efficient
 * Attention.
 *
 * Algorithm:
 * 1. Compute attention scores: S = (Q @ K^T)
 * 2. Apply scaling and compute softmax with optional causal mask
 * 3. Compute output: O = attention_weights @ V
 */
template <typename scalar_t>
Tensor* sdpa_math_fallback_impl(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    float scale_factor,
    cudaStream_t stream) {
  printf("Inside sdpa_math_fallback_impl\n");
  fflush(stdout);

  // Get tensor dimensions
  printf("Getting tensor dimensions\n");
  fflush(stdout);
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim = query->size(3);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_v = value->size(3);

  printf(
      "Dimensions: batch=%ld, num_heads=%ld, seq_len_q=%ld, head_dim=%ld\n",
      batch,
      num_heads,
      seq_len_q,
      head_dim);
  fflush(stdout);

  // Get cuBLAS handle
  printf("About to get cuBLAS handle\n");
  fflush(stdout);
  cublasHandle_t handle = get_cublas_handle(stream);
  printf("Got cuBLAS handle: %p\n", (void*)handle);
  fflush(stdout);

  // Step 1: Allocate temporary buffer for attention scores
  // Shape: [batch, num_heads, seq_len_q, seq_len_k]
  const int64_t scores_size = batch * num_heads * seq_len_q * seq_len_k;
  scalar_t* scores_ptr = nullptr;
  cudaMalloc(&scores_ptr, scores_size * sizeof(scalar_t));
  if (scores_ptr == nullptr) {
    ET_LOG(Error, "sdpa_math_fallback: Failed to allocate scores buffer");
    return nullptr;
  }

  // Step 2: Compute Q @ K^T using cuBLAS
  // Q: [batch * num_heads, seq_len_q, head_dim]
  // K^T: [batch * num_heads, head_dim, seq_len_k]
  // Output: [batch * num_heads, seq_len_q, seq_len_k]

  const int m = seq_len_q;
  const int n = seq_len_k;
  const int k = head_dim;
  const int batch_count = batch * num_heads;

  const scalar_t alpha = static_cast<scalar_t>(1.0f);
  const scalar_t beta = static_cast<scalar_t>(0.0f);

  const scalar_t* q_ptr = reinterpret_cast<const scalar_t*>(query->data_ptr());
  const scalar_t* k_ptr = reinterpret_cast<const scalar_t*>(key->data_ptr());

  // Strides for batched GEMM
  const int64_t stride_q = seq_len_q * head_dim;
  const int64_t stride_k = seq_len_k * head_dim;
  const int64_t stride_scores = seq_len_q * seq_len_k;

  // Q @ K^T
  cublasStatus_t status = batched_gemm<scalar_t>(
      handle,
      CUBLAS_OP_T, // Transpose K
      CUBLAS_OP_N, // No transpose Q
      n, // seq_len_k
      m, // seq_len_q
      k, // head_dim
      &alpha,
      k_ptr,
      k, // K matrix
      stride_k,
      q_ptr,
      k, // Q matrix
      stride_q,
      &beta,
      scores_ptr,
      n, // Output scores
      stride_scores,
      batch_count);

  if (status != CUBLAS_STATUS_SUCCESS) {
    ET_LOG(Error, "sdpa_math_fallback: cuBLAS GEMM failed for Q @ K^T");
    cudaFree(scores_ptr);
    return nullptr;
  }

  // Step 3: Apply softmax with scaling, optional causal mask, and attention
  // bias
  const int threads_per_block = 256;
  const int64_t total_rows = batch * num_heads * seq_len_q;
  const int num_blocks =
      (total_rows + threads_per_block - 1) / threads_per_block;

  // Get attn_bias pointer if provided
  const scalar_t* bias_ptr = attn_mask
      ? reinterpret_cast<const scalar_t*>(attn_mask->data_ptr())
      : nullptr;

  printf("About to launch softmax kernel with attn_bias=%p\n", (void*)bias_ptr);
  fflush(stdout);

  softmax_with_mask_kernel<scalar_t>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          scores_ptr,
          scores_ptr, // in-place
          bias_ptr, // attention bias (additive)
          batch,
          num_heads,
          seq_len_q,
          seq_len_k,
          is_causal,
          scale_factor);

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess) {
    printf(
        "sdpa_math_fallback: Softmax kernel launch failed: %s",
        cudaGetErrorString(cuda_err));
    fflush(stdout);
    ET_LOG(
        Error,
        "sdpa_math_fallback: Softmax kernel launch failed: %s",
        cudaGetErrorString(cuda_err));
    cudaFree(scores_ptr);
    return nullptr;
  }

  // Step 4: Allocate output tensor [batch, num_heads, seq_len_q, head_dim_v]
  Tensor* output = nullptr;
  std::array<int64_t, 4> output_shape = {
      batch, num_heads, seq_len_q, head_dim_v};
  std::array<int64_t, 4> output_stride = {
      num_heads * seq_len_q * head_dim_v,
      seq_len_q * head_dim_v,
      head_dim_v,
      1};

  auto dtype_int = static_cast<int32_t>(query->dtype());
  aoti_torch_empty_strided(
      4,
      output_shape.data(),
      output_stride.data(),
      dtype_int,
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &output);

  if (output == nullptr) {
    printf("sdpa_math_fallback: Failed to allocate output tensor");
    fflush(stdout);
    ET_LOG(Error, "sdpa_math_fallback: Failed to allocate output tensor");
    cudaFree(scores_ptr);
    return nullptr;
  }

  // Step 5: Compute attention_weights @ V
  // attention_weights: [batch * num_heads, seq_len_q, seq_len_k]
  // V: [batch * num_heads, seq_len_k, head_dim_v]
  // Output: [batch * num_heads, seq_len_q, head_dim_v]

  const int m_v = seq_len_q;
  const int n_v = head_dim_v;
  const int k_v = seq_len_k;

  const scalar_t* v_ptr = reinterpret_cast<const scalar_t*>(value->data_ptr());
  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(output->data_ptr());

  const int64_t stride_v = seq_len_k * head_dim_v;
  const int64_t stride_out = seq_len_q * head_dim_v;

  status = batched_gemm<scalar_t>(
      handle,
      CUBLAS_OP_N, // No transpose V
      CUBLAS_OP_N, // No transpose attention_weights
      n_v, // head_dim_v
      m_v, // seq_len_q
      k_v, // seq_len_k
      &alpha,
      v_ptr,
      n_v, // V matrix
      stride_v,
      scores_ptr,
      k_v, // attention_weights
      stride_scores,
      &beta,
      out_ptr,
      n_v, // Output
      stride_out,
      batch_count);

  // Cleanup temporary buffers
  cudaFree(scores_ptr);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("sdpa_math_fallback: cuBLAS GEMM failed for attention_weights @ V");
    fflush(stdout);
    ET_LOG(
        Error,
        "sdpa_math_fallback: cuBLAS GEMM failed for attention_weights @ V");
    aoti_torch_delete_tensor_object(output);
    return nullptr;
  }

  return output;
}

Tensor* sdpa_math_fallback(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    bool is_causal,
    double scale_factor,
    cudaStream_t stream) {
  printf("Inside sdpa_math_fallback\n");
  fflush(stdout);

  // Dispatch based on dtype
  auto dtype = query->dtype();
  printf("Query dtype: %d\n", static_cast<int>(dtype));
  fflush(stdout);

  if (dtype == executorch::aten::ScalarType::Float) {
    printf("Calling sdpa_math_fallback_impl<float>\n");
    fflush(stdout);
    return sdpa_math_fallback_impl<float>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::Half) {
    printf("Calling sdpa_math_fallback_impl<half>\n");
    fflush(stdout);
    return sdpa_math_fallback_impl<__half>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else if (dtype == executorch::aten::ScalarType::BFloat16) {
    printf("Calling sdpa_math_fallback_impl<bfloat16>\n");
    fflush(stdout);
    return sdpa_math_fallback_impl<__nv_bfloat16>(
        query,
        key,
        value,
        attn_mask,
        is_causal,
        static_cast<float>(scale_factor),
        stream);
  } else {
    ET_LOG(Error, "sdpa_math_fallback: Unsupported dtype");
    return nullptr;
  }
}

/**
 * Main entry point for SDPA computation
 */
Tensor* scaled_dot_product_attention_cuda(
    const Tensor* query,
    const Tensor* key,
    const Tensor* value,
    const Tensor* attn_mask,
    double dropout_p,
    bool is_causal,
    const double* scale,
    bool enable_gqa,
    cudaStream_t stream) {
  // Select backend
  SDPBackend backend =
      select_sdp_backend(query, key, value, attn_mask, dropout_p, is_causal);

  if (backend == SDPBackend::Error) {
    ET_LOG(
        Error, "scaled_dot_product_attention_cuda: No valid backend selected");
    return nullptr;
  }

  printf("selected backend: %d\n", static_cast<int>(backend));
  fflush(stdout);

  // Calculate scale factor
  printf("About to calculate scale factor\n");
  fflush(stdout);
  double scale_factor = calculate_scale(query, scale);
  printf("Calculated scale factor: %f\n", scale_factor);
  fflush(stdout);

  printf("enable_gqa: %d\n", enable_gqa);
  fflush(stdout);

  // Handle GQA if needed
  printf("About to check GQA configuration\n");
  fflush(stdout);

  if (enable_gqa && is_gqa_configuration(query, key, value)) {
    printf("GQA configuration detected\n");
    fflush(stdout);

    if (!validate_gqa(query, key, value)) {
      ET_LOG(
          Error,
          "scaled_dot_product_attention_cuda: Invalid GQA configuration");
      return nullptr;
    }
    ET_LOG(
        Error,
        "scaled_dot_product_attention_cuda: GQA support not yet implemented. "
        "Need to repeat K/V heads to match Q heads.");
    return nullptr;
  }

  printf("Passed GQA check\n");
  fflush(stdout);

  printf(
      "About to enter switch statement, backend = %d\n",
      static_cast<int>(backend));
  fflush(stdout);

  // Dispatch to appropriate backend
  switch (backend) {
    case SDPBackend::Math:
      printf("In Math case, about to call sdpa_math_fallback\n");
      fflush(stdout);
      return sdpa_math_fallback(
          query, key, value, attn_mask, is_causal, scale_factor, stream);

    case SDPBackend::FlashAttention:
      printf("In FlashAttention case\n");
      fflush(stdout);
      return sdpa_flash_attention(
          query, key, value, attn_mask, is_causal, scale_factor, stream);

    case SDPBackend::MemoryEfficientAttention:
      printf("Memory Efficient Attention backend\n");
      fflush(stdout);
      return sdpa_efficient_attention(
          query, key, value, attn_mask, is_causal, scale_factor, stream);

    case SDPBackend::CuDNN:
      printf("cuDNN backend not yet implemented\n");
      fflush(stdout);
      return nullptr;

    default:
      printf("Unknown SDPA backend\n");
      fflush(stdout);
      return nullptr;
  }
}

// ============================================================================
// C API Implementation
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda__scaled_dot_product_flash_attention(
    Tensor* query,
    Tensor* key,
    Tensor* value,
    double dropout_p,
    int32_t is_causal,
    int32_t return_debug_mask,
    double* scale,
    Tensor** ret0, // output
    Tensor** ret1, // logsumexp (nullptr for inference)
    Tensor** ret2, // cum_seq_q (nullptr for inference)
    Tensor** ret3, // cum_seq_k (nullptr for inference)
    int64_t* max_seqlen_q,
    int64_t* max_seqlen_k,
    Tensor** ret4, // philox_seed (nullptr for inference)
    Tensor** ret5, // philox_offset (nullptr for inference)
    Tensor** ret6) { // debug_attn_mask (nullptr for inference)
  // Input validation
  if (!query || !key || !value || !ret0) {
    printf("[FLASH ATTN ERROR] C API: Null pointer input\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Null pointer input");
    return Error::InvalidArgument;
  }

  // Currently only support dropout_p = 0.0 for inference
  if (dropout_p != 0.0) {
    printf("[FLASH ATTN ERROR] C API: dropout_p != 0.0 is not supported\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: dropout_p != 0.0 is not supported");
    return Error::InvalidArgument;
  }

  // Check tensor dimensions
  if (query->dim() != 4 || key->dim() != 4 || value->dim() != 4) {
    printf("[FLASH ATTN ERROR] C API: Query, Key, Value must be 4D tensors\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Query, Key, Value must be 4D tensors");
    return Error::InvalidArgument;
  }

  // Check that Q, K, V have the same dtype
  if (query->dtype() != key->dtype() || query->dtype() != value->dtype()) {
    printf(
        "[FLASH ATTN ERROR] C API: Query, Key, Value must have the same dtype\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Query, Key, Value must have the same dtype");
    return Error::InvalidArgument;
  }

  // Check dtype support
  if (!is_supported_dtype(query) || !is_supported_dtype(key) ||
      !is_supported_dtype(value)) {
    printf(
        "[FLASH ATTN ERROR] C API: Unsupported dtype, only Float32/Float16/BFloat16 supported\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Unsupported dtype, only Float32/Float16/BFloat16 supported");
    return Error::InvalidArgument;
  }

  // Check tensor shapes
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim_q = query->size(3);

  const int64_t num_heads_kv = key->size(1);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_k = key->size(3);

  const int64_t seq_len_v = value->size(2);
  const int64_t head_dim_v = value->size(3);

  // Validate shapes
  if (key->size(0) != batch || value->size(0) != batch) {
    printf("[FLASH ATTN ERROR] C API: Batch size mismatch\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Batch size mismatch");
    return Error::InvalidArgument;
  }

  if (seq_len_k != seq_len_v) {
    printf(
        "[FLASH ATTN ERROR] C API: Key and Value sequence length mismatch\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Key and Value sequence length mismatch");
    return Error::InvalidArgument;
  }

  if (head_dim_q != head_dim_k) {
    printf("[FLASH ATTN ERROR] C API: Query and Key head dimension mismatch\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Query and Key head dimension mismatch");
    return Error::InvalidArgument;
  }

  if (value->size(1) != num_heads_kv) {
    printf("[FLASH ATTN ERROR] C API: Key and Value num_heads mismatch\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Key and Value num_heads mismatch");
    return Error::InvalidArgument;
  }

  // Determine if GQA is being used
  bool enable_gqa = (num_heads != num_heads_kv);

  // GQA validation and check
  if (enable_gqa) {
    if (num_heads % num_heads_kv != 0) {
      printf(
          "[FLASH ATTN ERROR] C API: For GQA, num_heads must be divisible by num_heads_kv\n");
      fflush(stdout);
      ET_LOG(
          Error,
          "aoti_torch_cuda__scaled_dot_product_flash_attention: For GQA, num_heads must be divisible by num_heads_kv");
      return Error::InvalidArgument;
    }
    printf("[FLASH ATTN ERROR] C API: GQA support not yet implemented\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: GQA support not yet implemented");
    return Error::InvalidArgument;
  }

  // Check if flash attention can be used
  if (!supports_flash_attention()) {
    printf(
        "[FLASH ATTN ERROR] C API: Flash Attention not supported on this GPU\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Flash Attention not supported on this GPU");
    return Error::InvalidArgument;
  }

  if (!can_use_flash_attention(query, key, value, nullptr, is_causal != 0)) {
    printf(
        "[FLASH ATTN ERROR] C API: Input conditions not suitable for Flash Attention\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Input conditions not suitable for Flash Attention");
    return Error::InvalidArgument;
  }

  // Calculate scale factor
  double scale_factor = calculate_scale(query, scale);

  // Get CUDA stream
  auto stream_result = getCurrentCUDAStream(0);
  if (!stream_result.ok()) {
    printf("[FLASH ATTN ERROR] C API: Failed to get CUDA stream\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Failed to get CUDA stream");
    return Error::Internal;
  }
  cudaStream_t stream = stream_result.get();

  // Call flash attention directly
  Tensor* output = sdpa_flash_attention(
      query,
      key,
      value,
      nullptr, // attn_mask - Flash Attention doesn't support it
      is_causal != 0,
      scale_factor,
      stream);

  if (output == nullptr) {
    printf("[FLASH ATTN ERROR] C API: Flash Attention computation failed\n");
    fflush(stdout);
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_flash_attention: Flash Attention computation failed");
    return Error::Internal;
  }

  // Set the main output
  *ret0 = output;

  // Set all training-related outputs to nullptr (for inference)
  if (ret1)
    *ret1 = nullptr; // logsumexp
  if (ret2)
    *ret2 = nullptr; // cum_seq_q
  if (ret3)
    *ret3 = nullptr; // cum_seq_k
  if (ret4)
    *ret4 = nullptr; // philox_seed
  if (ret5)
    *ret5 = nullptr; // philox_offset
  if (ret6)
    *ret6 = nullptr; // debug_attn_mask

  return Error::Ok;
}

AOTITorchError aoti_torch_cuda__scaled_dot_product_efficient_attention(
    Tensor* query,
    Tensor* key,
    Tensor* value,
    Tensor** attn_bias, // Optional attention bias (can be nullptr)
    int32_t compute_log_sumexp,
    double dropout_p,
    int32_t is_causal,
    double* scale,
    Tensor** ret0, // output
    Tensor** ret1, // logsumexp (nullptr for inference)
    Tensor** ret2, // philox_seed (nullptr for inference)
    Tensor** ret3) { // philox_offset (nullptr for inference)

  // printf(
  //     "\n[SDPA DEBUG] C API:
  //     aoti_torch_cuda__scaled_dot_product_efficient_attention called\n");
  // printf(
  //     "  query=%p, key=%p, value=%p, attn_bias=%p\n",
  //     (void*)query,
  //     (void*)key,
  //     (void*)value,
  //     (void*)attn_bias);
  // if (attn_bias && *attn_bias) {
  //   printf("  attn_bias tensor=%p\n", (void*)(*attn_bias));
  // }
  // fflush(stdout);

  // Input validation
  if (!query || !key || !value || !ret0) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Null pointer input");
    return Error::InvalidArgument;
  }

  // Currently only support dropout_p = 0.0 for inference
  if (dropout_p != 0.0) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: dropout_p != 0.0 is not supported");
    return Error::InvalidArgument;
  }

  // Check tensor dimensions
  if (query->dim() != 4 || key->dim() != 4 || value->dim() != 4) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Query, Key, Value must be 4D tensors");
    return Error::InvalidArgument;
  }

  // Check that Q, K, V have the same dtype
  if (query->dtype() != key->dtype() || query->dtype() != value->dtype()) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Query, Key, Value must have the same dtype");
    return Error::InvalidArgument;
  }

  // Check dtype support
  if (!is_supported_dtype(query) || !is_supported_dtype(key) ||
      !is_supported_dtype(value)) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Unsupported dtype, only Float32/Float16/BFloat16 supported");
    return Error::InvalidArgument;
  }

  // Check tensor shapes
  const int64_t batch = query->size(0);
  const int64_t num_heads = query->size(1);
  const int64_t seq_len_q = query->size(2);
  const int64_t head_dim_q = query->size(3);

  const int64_t num_heads_kv = key->size(1);
  const int64_t seq_len_k = key->size(2);
  const int64_t head_dim_k = key->size(3);

  const int64_t seq_len_v = value->size(2);
  const int64_t head_dim_v = value->size(3);

  // Validate shapes
  if (key->size(0) != batch || value->size(0) != batch) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Batch size mismatch");
    return Error::InvalidArgument;
  }

  if (seq_len_k != seq_len_v) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Key and Value sequence length mismatch");
    return Error::InvalidArgument;
  }

  if (head_dim_q != head_dim_k) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Query and Key head dimension mismatch");
    return Error::InvalidArgument;
  }

  if (value->size(1) != num_heads_kv) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Key and Value num_heads mismatch");
    return Error::InvalidArgument;
  }

  // Determine if GQA is being used
  bool enable_gqa = (num_heads != num_heads_kv);

  // GQA validation and check
  if (enable_gqa) {
    if (num_heads % num_heads_kv != 0) {
      ET_LOG(
          Error,
          "aoti_torch_cuda__scaled_dot_product_efficient_attention: For GQA, num_heads must be divisible by num_heads_kv");
      return Error::InvalidArgument;
    }
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: GQA support not yet implemented");
    return Error::InvalidArgument;
  }

  // Extract attn_bias tensor if provided
  Tensor* attn_bias_tensor = (attn_bias && *attn_bias) ? *attn_bias : nullptr;

  // Check if efficient attention can be used
  if (!can_use_efficient_attention(
          query, key, value, attn_bias_tensor, is_causal != 0)) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Input conditions not suitable for Efficient Attention");
    return Error::InvalidArgument;
  }

  // Calculate scale factor
  double scale_factor = calculate_scale(query, scale);

  // Get CUDA stream
  auto stream_result = getCurrentCUDAStream(0);
  if (!stream_result.ok()) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Failed to get CUDA stream");
    return Error::Internal;
  }
  cudaStream_t stream = stream_result.get();

  // Call efficient attention directly
  Tensor* output = sdpa_efficient_attention(
      query,
      key,
      value,
      attn_bias_tensor, // Pass attn_bias (can be nullptr)
      is_causal != 0,
      scale_factor,
      stream);

  if (output == nullptr) {
    ET_LOG(
        Error,
        "aoti_torch_cuda__scaled_dot_product_efficient_attention: Efficient Attention computation failed");
    return Error::Internal;
  }

  // Set the main output
  *ret0 = output;

  // Set all training-related outputs to nullptr (for inference)
  if (ret1)
    *ret1 = nullptr; // logsumexp
  if (ret2)
    *ret2 = nullptr; // philox_seed
  if (ret3)
    *ret3 = nullptr; // philox_offset

  return Error::Ok;
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
