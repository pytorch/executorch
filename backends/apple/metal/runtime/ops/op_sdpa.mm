/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/metal/runtime/ops/common.h>

namespace executorch {
namespace backends {
namespace metal {
namespace {

// Helper function to get the Metal shader source for SDPA
static std::string get_sdpa_metal_source() {
  return R"(
// Ported from PyTorch's Attention.metal
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Attention.metal
// Largely influenced by
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
// Modified to support floating point masks and transposed middle dimensions (dims 1 & 2)

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_math>

using namespace metal;

// PyTorch's sdpa_vector kernel (one-pass variant)
template <typename T, int D, int V = D>
[[kernel]] void sdpa_vector(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    const device T* values [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant uint& gqa_factor [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint3& qkv_head_strides [[buffer(6)]],
    constant uint3& qkv_seq_strides [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    const device T* mask [[buffer(9)]],  // Changed from bool* to T* for floating point masks
    constant uint3& mask_strides [[buffer(10)]],
    constant bool& has_mask [[buffer(11)]],
    constant uint3& qkv_batch_strides [[buffer(12)]],  // NEW: batch strides for Q, K, V
    constant uint& num_q_heads [[buffer(13)]],          // NEW: number of query heads
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr uint BN = 32;
  constexpr uint BD = 32;
  constexpr uint qk_per_thread = D / BD;
  constexpr uint v_per_thread = V / BD;
  const uint q_head_stride = qkv_head_strides.x;
  const uint q_seq_stride = qkv_seq_strides.x;
  const uint q_batch_stride = qkv_batch_strides.x;
  const uint k_head_stride = qkv_head_strides.y;
  const uint k_seq_stride = qkv_seq_strides.y;
  const uint k_batch_stride = qkv_batch_strides.y;
  const uint v_head_stride = qkv_head_strides.z;
  const uint v_seq_stride = qkv_seq_strides.z;
  const uint v_batch_stride = qkv_batch_strides.z;
  const uint mask_head_stride = mask_strides.x;
  const uint mask_kv_seq_stride = mask_strides.y;
  const uint mask_q_seq_stride = mask_strides.z;
  uint inner_k_stride = BN * int(k_seq_stride);
  uint inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int head_idx = tid.x;  // Flattened batch*heads index
  const int q_seq_idx = tid.y;

  // Decompose flattened head_idx into batch and head indices
  const int batch_idx = head_idx / num_q_heads;
  const int head_in_batch = head_idx % num_q_heads;
  const int kv_head_idx = head_in_batch / gqa_factor;

  const int Q = tpg.y;
  const int group_offset = head_idx * Q + q_seq_idx;
  const int o_offset = group_offset;

  // Use decomposed indices with separate batch and head strides
  queries += batch_idx * q_batch_stride + head_in_batch * q_head_stride + q_seq_idx * q_seq_stride +
      simd_lid * qk_per_thread;
  keys += batch_idx * k_batch_stride + kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += batch_idx * v_batch_stride + kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (has_mask) {
    mask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
        q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (uint i = 0; i < qk_per_thread; i++) {
    q[i] = scale * static_cast<U>(queries[i]);
  }
  for (uint i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;

  // For each key
  for (uint i = simd_gid; i < N; i += BN) {
    // Check mask: for floating point masks, values > -1e9 are considered valid (not masked)
    // Masked positions typically have -inf or very negative values
    const bool is_valid = !has_mask || (static_cast<U>(mask[0]) > -1e9f);

    if (is_valid) {
      // Read the key
      for (uint j = 0; j < qk_per_thread; j++) {
        k[j] = static_cast<U>(keys[j]);
      }

      // Compute the i-th score
      U score = 0;
      for (uint j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Add mask value to score if mask is present
      if (has_mask) {
        score += static_cast<U>(mask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = metal::fast::exp(max_score - new_max);
      U exp_score = metal::fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (uint j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * static_cast<U>(values[j]);
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (has_mask) {
      mask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = metal::fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (uint i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const U safe_sum = (sum_exp_score == 0 ? 1e-6f : sum_exp_score);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / safe_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (uint i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

#define INSTANTIATE_SDPA_VECTOR(DTYPE, QK_DIM, VALUE_DIM)   \
  template [[host_name("sdpa_vector_" #DTYPE "_" #QK_DIM    \
                       "_" #VALUE_DIM)]] kernel void        \
  sdpa_vector<DTYPE, QK_DIM, VALUE_DIM>(                    \
      const device DTYPE* queries [[buffer(0)]],            \
      const device DTYPE* keys [[buffer(1)]],               \
      const device DTYPE* values [[buffer(2)]],             \
      device DTYPE* out [[buffer(3)]],                      \
      constant uint& gqa_factor [[buffer(4)]],        \
      constant uint& N [[buffer(5)]],                 \
      constant uint3& qkv_head_strides [[buffer(6)]], \
      constant uint3& qkv_seq_strides [[buffer(7)]],  \
      constant float& scale [[buffer(8)]],            \
      const device DTYPE* mask [[buffer(9)]],         \
      constant uint3& mask_strides [[buffer(10)]],    \
      constant bool& has_mask [[buffer(11)]],         \
      constant uint3& qkv_batch_strides [[buffer(12)]], \
      constant uint& num_q_heads [[buffer(13)]],       \
      uint3 tid [[threadgroup_position_in_grid]],           \
      uint3 tpg [[threadgroups_per_grid]],                  \
      uint simd_gid [[simdgroup_index_in_threadgroup]],     \
      uint simd_lid [[thread_index_in_simdgroup]]);

#define INSTANTIATE_SDPA_VECTOR_HEADS(DTYPE)        \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 64, 64);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 96, 96);           \
  INSTANTIATE_SDPA_VECTOR(DTYPE, 128, 128);

INSTANTIATE_SDPA_VECTOR_HEADS(float);
INSTANTIATE_SDPA_VECTOR_HEADS(bfloat);
)";
}

std::unique_ptr<ETMetalShaderLibrary> sdpa_shader_library = nullptr;

std::once_flag sdpa_shader_library_once_flag;

ETMetalShaderLibrary* get_sdpa_shader_library() {
  std::call_once(sdpa_shader_library_once_flag, []() {
    std::string source = get_sdpa_metal_source();
    sdpa_shader_library = std::make_unique<ETMetalShaderLibrary>(source);
  });
  return sdpa_shader_library.get();
}

} // namespace


extern "C" {

AOTITorchError aoti_torch_mps__scaled_dot_product_attention_math_for_mps(
    AOTITensorHandle query,
    AOTITensorHandle key,
    AOTITensorHandle value,
    AOTITensorHandle* attn_mask,
    double dropout_p,
    int32_t is_causal,
    AOTITensorHandle* dropout_mask,
    double* scale,
    AOTITensorHandle* ret0,
    AOTITensorHandle* ret1) {

  ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Starting with Metal kernel implementation");

  if (!query || !key || !value || !ret0 || !ret1) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: null required tensor handles");
    return Error::InvalidArgument;
  }

  if (is_causal) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: is_causal=True not implemented");
    return Error::NotImplemented;
  }
  if (dropout_p != 0.0) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: dropout_p != 0 not implemented (dropout_p=%f)", dropout_p);
    return Error::NotImplemented;
  }
  if (dropout_mask && *dropout_mask) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: dropout_mask provided not implemented");
    return Error::NotImplemented;
  }

  // Use the same dispatch pattern as other MPS operations for consistent synchronization
  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to get current Metal stream");
    return Error::Internal;
  }

  try {
    @autoreleasepool {
      // Convert AOTITensorHandle to ExecuTorch tensors
      auto* query_tensor = reinterpret_cast<Tensor*>(query);
      auto* key_tensor = reinterpret_cast<Tensor*>(key);
      auto* value_tensor = reinterpret_cast<Tensor*>(value);

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Converted tensor handles to ET tensors");

      // Log query tensor shape and strides
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Query tensor - dim=%d, shape=[%d, %d, %d, %d], strides=[%d, %d, %d, %d]",
              (int)query_tensor->dim(),
              query_tensor->dim() > 0 ? query_tensor->sizes()[0] : 0,
              query_tensor->dim() > 1 ? query_tensor->sizes()[1] : 0,
              query_tensor->dim() > 2 ? query_tensor->sizes()[2] : 0,
              query_tensor->dim() > 3 ? query_tensor->sizes()[3] : 0,
              query_tensor->dim() > 0 ? query_tensor->strides()[0] : 0,
              query_tensor->dim() > 1 ? query_tensor->strides()[1] : 0,
              query_tensor->dim() > 2 ? query_tensor->strides()[2] : 0,
              query_tensor->dim() > 3 ? query_tensor->strides()[3] : 0);

      // Log key tensor shape and strides
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Key tensor - dim=%d, shape=[%d, %d, %d, %d], strides=[%d, %d, %d, %d]",
              (int)key_tensor->dim(),
              key_tensor->dim() > 0 ? key_tensor->sizes()[0] : 0,
              key_tensor->dim() > 1 ? key_tensor->sizes()[1] : 0,
              key_tensor->dim() > 2 ? key_tensor->sizes()[2] : 0,
              key_tensor->dim() > 3 ? key_tensor->sizes()[3] : 0,
              key_tensor->dim() > 0 ? key_tensor->strides()[0] : 0,
              key_tensor->dim() > 1 ? key_tensor->strides()[1] : 0,
              key_tensor->dim() > 2 ? key_tensor->strides()[2] : 0,
              key_tensor->dim() > 3 ? key_tensor->strides()[3] : 0);

      // Log value tensor shape and strides
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Value tensor - dim=%d, shape=[%d, %d, %d, %d], strides=[%d, %d, %d, %d]",
              (int)value_tensor->dim(),
              value_tensor->dim() > 0 ? value_tensor->sizes()[0] : 0,
              value_tensor->dim() > 1 ? value_tensor->sizes()[1] : 0,
              value_tensor->dim() > 2 ? value_tensor->sizes()[2] : 0,
              value_tensor->dim() > 3 ? value_tensor->sizes()[3] : 0,
              value_tensor->dim() > 0 ? value_tensor->strides()[0] : 0,
              value_tensor->dim() > 1 ? value_tensor->strides()[1] : 0,
              value_tensor->dim() > 2 ? value_tensor->strides()[2] : 0,
              value_tensor->dim() > 3 ? value_tensor->strides()[3] : 0);

      // Validate tensor dimensions
      if (query_tensor->dim() < 3 || key_tensor->dim() < 3 || value_tensor->dim() < 3) {
        std::string error_msg = "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: tensors must be at least 3-D, got " +
                                std::to_string(query_tensor->dim()) + ", " +
                                std::to_string(key_tensor->dim()) + ", " +
                                std::to_string(value_tensor->dim());
        ET_LOG(Error, "%s", error_msg.c_str());
        throw std::runtime_error(error_msg);
      }

      // Get tensor dimensions (assuming [batch, num_heads, seq_len, head_dim] format)
      int64_t batchSize = query_tensor->sizes()[0];
      int64_t num_heads = query_tensor->sizes()[1];
      int64_t qSize = query_tensor->sizes()[2];
      int64_t headSize = query_tensor->sizes()[3];
      int64_t kvSeqLength = key_tensor->sizes()[2];

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: batchSize=%lld, num_heads=%lld, qSize=%lld, headSize=%lld, kvSeqLength=%lld",
              batchSize, num_heads, qSize, headSize, kvSeqLength);

      // Determine data type and element size
      int32_t dtype = static_cast<int32_t>(query_tensor->scalar_type());
      size_t element_size;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        element_size = sizeof(float);
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        element_size = sizeof(uint16_t);  // bfloat16 is 16 bits
      } else {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Unsupported data type: %d", dtype);
        throw std::runtime_error("Unsupported data type for scaled dot product attention");
      }

      // Check that headSize is not zero to avoid division by zero
      if (headSize == 0) {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: headSize is zero");
        throw std::runtime_error("headSize must be non-zero for scaled dot product attention");
      }

      // Validate key tensor head dimension to avoid division by zero in gqa_factor calculation
      int64_t key_num_heads = key_tensor->sizes()[1];
      if (key_num_heads == 0) {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: key tensor head dimension (sizes()[1]) is zero");
        throw std::runtime_error("key tensor must have non-zero head dimension for scaled dot product attention");
      }

      // Calculate scale factor
      double scale_factor = scale ? *scale : (1.0 / sqrt(static_cast<double>(headSize)));
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: scale_factor=%f", scale_factor);

      // Calculate output tensor dimensions
      std::vector<int64_t> output_sizes = {batchSize, num_heads, qSize, headSize};
      std::vector<int64_t> attn_sizes = {batchSize, num_heads, qSize, kvSeqLength};

      // Calculate strides for contiguous tensors
      std::vector<int64_t> out_strides = {
          num_heads * qSize * headSize,
          qSize * headSize,
          headSize,
          1
      };

      std::vector<int64_t> attn_strides = {
          num_heads * qSize * kvSeqLength,
          qSize * kvSeqLength,
          kvSeqLength,
          1
      };

      // Allocate output Metal buffers via AOTI API to keep GPU residency and reuse
      size_t out_size_bytes = batchSize * num_heads * qSize * headSize * element_size;
      size_t attn_size_bytes = batchSize * num_heads * qSize * kvSeqLength * element_size;

      void* out_contents_ptr = nullptr;
      allocate_mtl_buffer(&out_contents_ptr, out_size_bytes);

      void* attn_contents_ptr = nullptr;
      allocate_mtl_buffer(&attn_contents_ptr, attn_size_bytes);

      // Use MLX-style Metal kernels instead of MPSGraph
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Implementing using MLX Metal kernels");

      // Get shader library
      ETMetalShaderLibrary* library = get_sdpa_shader_library();
      if (!library) {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to get shader library");
        throw std::runtime_error("Failed to get SDPA shader library");
      }

      // Determine kernel name based on dtype and head_dim (PyTorch format)
      std::string type_name;
      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        type_name = "float";
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        type_name = "bfloat";
      } else {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Unsupported dtype for Metal kernel");
        throw std::runtime_error("Unsupported dtype for Metal SDPA kernel");
      }

      // Select head_dim - must match exactly one of the supported sizes (64, 96, 128)
      int64_t head_dim = headSize;
      if (head_dim != 64 && head_dim != 96 && head_dim != 128) {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Unsupported head_dim %lld (must be 64, 96, or 128)", head_dim);
        throw std::runtime_error("Unsupported head_dim for Metal SDPA kernel - must be exactly 64, 96, or 128");
      }

      std::string kernel_name = "sdpa_vector_" + type_name + "_" + std::to_string(head_dim) + "_" + std::to_string(head_dim);
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Using kernel: %s", kernel_name.c_str());

      // Get kernel function
      auto kernel_func = library->getKernelFunction(kernel_name);
      if (!kernel_func) {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to get kernel function: %s", kernel_name.c_str());
        throw std::runtime_error("Failed to get SDPA kernel function");
      }

      // Create output tensor handle first so we can use it in the kernel
      AOTITensorHandle out_tensor_handle = nullptr;
      AOTITorchError create_out_result = aoti_torch_create_tensor_from_blob_v2(
          out_contents_ptr,
          4,  // ndim
          output_sizes.data(),
          out_strides.data(),
          0,  // storage_offset
          dtype,
          13,  // device_type (MPS)
          0,  // device_index
          &out_tensor_handle,
          0,  // layout (strided)
          nullptr,  // opaque_metadata
          0   // opaque_metadata_size
      );

      if (create_out_result != Error::Ok || !out_tensor_handle) {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create output tensor");
        aoti_torch_mps_free(out_contents_ptr);
        aoti_torch_mps_free(attn_contents_ptr);
        throw std::runtime_error("Failed to create output tensor");
      }

      // Mark that we own the memory
      extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
      memory_to_n_tensor[out_contents_ptr] = 1;

      auto* out_tensor = reinterpret_cast<Tensor*>(out_tensor_handle);

      // Prepare kernel arguments (PyTorch format)
      uint gqa_factor = static_cast<uint>(num_heads / key_tensor->sizes()[1]);
      uint N = static_cast<uint>(kvSeqLength);

      // Get strides for Q, K, V (all 3 stride levels: batch, head, seq)
      uint q_batch_stride = static_cast<uint>(query_tensor->strides()[0]);
      uint q_head_stride = static_cast<uint>(query_tensor->strides()[1]);
      uint q_seq_stride = static_cast<uint>(query_tensor->strides()[2]);
      uint q_dim_stride = static_cast<uint>(query_tensor->strides()[3]);

      uint k_batch_stride = static_cast<uint>(key_tensor->strides()[0]);
      uint k_head_stride = static_cast<uint>(key_tensor->sizes()[1] == 1 ? key_tensor->strides()[0] : key_tensor->strides()[1]);
      uint k_seq_stride = static_cast<uint>(key_tensor->strides()[2]);
      uint k_dim_stride = static_cast<uint>(key_tensor->strides()[3]);

      uint v_batch_stride = static_cast<uint>(value_tensor->strides()[0]);
      uint v_head_stride = static_cast<uint>(value_tensor->sizes()[1] == 1 ? value_tensor->strides()[0] : value_tensor->strides()[1]);
      uint v_seq_stride = static_cast<uint>(value_tensor->strides()[2]);
      uint v_dim_stride = static_cast<uint>(value_tensor->strides()[3]);

      // Log strides for debugging
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Q strides - batch:%u, head:%u, seq:%u, dim:%u",
              q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride);
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: K strides - batch:%u, head:%u, seq:%u, dim:%u",
              k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride);
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: V strides - batch:%u, head:%u, seq:%u, dim:%u",
              v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride);

      // Check if middle dimensions (1 and 2) are transposed
      // For contiguous [batch, num_heads, seq, dim]: stride[1] > stride[2] (head_stride > seq_stride)
      // For transposed [batch, seq, num_heads, dim] in memory: stride[1] < stride[2] (head_stride < seq_stride)
      bool q_transposed = (q_head_stride < q_seq_stride);
      bool k_transposed = (k_head_stride < k_seq_stride);
      bool v_transposed = (v_head_stride < v_seq_stride);

      if (q_transposed || k_transposed || v_transposed) {
        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Transposed middle dimensions detected (dims 1&2 swapped)! Q:%d, K:%d, V:%d",  q_transposed, k_transposed, v_transposed);
        ET_LOG(Debug, "  For transposed layout: head_stride < seq_stride");
        ET_LOG(Debug, "  Q: head_stride=%u, seq_stride=%u (transposed=%d)", q_head_stride, q_seq_stride, q_transposed);
        ET_LOG(Debug, "  K: head_stride=%u, seq_stride=%u (transposed=%d)", k_head_stride, k_seq_stride, k_transposed);
        ET_LOG(Debug, "  V: head_stride=%u, seq_stride=%u (transposed=%d)", v_head_stride, v_seq_stride, v_transposed);
        ET_LOG(Debug, "  The updated kernel will handle this by decomposing batch and head indices.");
      }

      // Verify innermost dimension has stride=1 (required by current kernel implementation)
      if (q_dim_stride != 1 || k_dim_stride != 1 || v_dim_stride != 1) {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Non-unit dim stride detected!");
        ET_LOG(Error, "  Q dim_stride=%u, K dim_stride=%u, V dim_stride=%u", q_dim_stride, k_dim_stride, v_dim_stride);
        ET_LOG(Error, "  Current kernel implementation requires innermost dimension to be contiguous (stride=1)");
        throw std::runtime_error("SDPA Metal kernel requires innermost dimension to be contiguous (dim_stride must be 1)");
      }

      bool has_mask_val = (attn_mask && *attn_mask);

      // Calculate mask strides if mask is present
      uint mask_head_stride = 0;
      uint mask_kv_seq_stride = 0;
      uint mask_q_seq_stride = 0;
      if (has_mask_val) {
        auto* mask_tensor = reinterpret_cast<Tensor*>(*attn_mask);
        int nd = mask_tensor->dim();
        mask_kv_seq_stride = (nd >= 1 && mask_tensor->sizes()[nd - 1] > 1) ? static_cast<uint>(mask_tensor->strides()[nd - 1]) : 0;
        mask_q_seq_stride = (nd >= 2 && mask_tensor->sizes()[nd - 2] > 1) ? static_cast<uint>(mask_tensor->strides()[nd - 2]) : 0;
        mask_head_stride = (nd >= 3 && mask_tensor->sizes()[nd - 3] > 1) ? static_cast<uint>(mask_tensor->strides()[nd - 3]) : 0;
      }

      // Execute kernel
      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Preparing to execute kernel with grid [%llu, %llu, %llu], group [1024, 1, 1]",
              (unsigned long long)(batchSize * num_heads), (unsigned long long)qSize, 1ULL);

      kernel_func->runCommandBlock([&]() {
        kernel_func->startEncoding();

        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Encoder started, setting arguments");

        // Set buffer arguments (0-3: Q, K, V, out)
        kernel_func->setArg(0, *query_tensor);
        kernel_func->setArg(1, *key_tensor);
        kernel_func->setArg(2, *value_tensor);
        kernel_func->setArg(3, *out_tensor);

        // Set scalar arguments (uint values)
        kernel_func->setArg(4, gqa_factor);
        kernel_func->setArg(5, N);

        // Set uint3 for qkv_head_strides (buffer 6)
        kernel_func->setArgUint3(6, q_head_stride, k_head_stride, v_head_stride);

        // Set uint3 for qkv_seq_strides (buffer 7)
        kernel_func->setArgUint3(7, q_seq_stride, k_seq_stride, v_seq_stride);

        // Set scale as float (buffer 8)
        kernel_func->setArg(8, static_cast<float>(scale_factor));

        // Set mask buffer (buffer 9)
        if (has_mask_val) {
          auto* mask_tensor = reinterpret_cast<Tensor*>(*attn_mask);
          kernel_func->setArg(9, *mask_tensor);
        } else {
          // Dummy buffer if no mask (won't be accessed)
          kernel_func->setArg(9, *query_tensor);
        }

        // Set uint3 for mask_strides (buffer 10)
        kernel_func->setArgUint3(10, mask_head_stride, mask_kv_seq_stride, mask_q_seq_stride);

        // Set has_mask as bool (buffer 11)
        kernel_func->setArg(11, has_mask_val);

        // Set uint3 for qkv_batch_strides (buffer 12) - NEW
        kernel_func->setArgUint3(12, q_batch_stride, k_batch_stride, v_batch_stride);

        // Set num_q_heads (buffer 13) - NEW
        kernel_func->setArg(13, static_cast<uint>(num_heads));

        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: All arguments set, dispatching");

        // Dispatch using threadgroups (PyTorch uses grid: [batch*heads, qSize, 1], group: [1024, 1, 1])
        // Note: We need to use dispatchThreadgroups, not dispatchThreads
        // Each threadgroup processes one query token across all key-value tokens
        kernel_func->dispatchThreadgroups(
            batchSize * num_heads,  // gridX
            qSize,                   // gridY
            1,                       // gridZ
            1024,                    // threadsX
            1,                       // threadsY
            1);                      // threadsZ
      });

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Command block completed");

      AOTITensorHandle attn_tensor_handle = nullptr;
      AOTITorchError create_attn_result = aoti_torch_create_tensor_from_blob_v2(
          attn_contents_ptr,
          4,  // ndim
          attn_sizes.data(),
          attn_strides.data(),
          0,  // storage_offset
          dtype,
          13,  // device_type (MPS)
          0,  // device_index
          &attn_tensor_handle,
          0,  // layout (strided)
          nullptr,  // opaque_metadata
          0   // opaque_metadata_size
      );

      if (create_attn_result != Error::Ok || !attn_tensor_handle) {
        ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create attention weights tensor");
        aoti_torch_mps_free(attn_contents_ptr);
        throw std::runtime_error("Failed to create attention weights tensor");
      }

      memory_to_n_tensor[attn_contents_ptr] = 1;

      // Set output tensor handles
      *ret0 = out_tensor_handle;
      *ret1 = attn_tensor_handle;

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Metal kernel implementation completed successfully");

    }  // @autoreleasepool

    ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Executed successfully");
    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: unknown exception");
    return Error::Internal;
  }
}


} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
