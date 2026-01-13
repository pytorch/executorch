/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal_ops.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <executorch/backends/apple/metal/runtime/shims/shim_mps.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <functional>
#include <unordered_map>
#include <memory>

namespace executorch {
namespace backends {
namespace metal {

using executorch::runtime::etensor::Tensor;

// Forward declaration of dispatch_sync_with_rethrow from et_metal.mm
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());

// Declare the global mapping from et_metal.mm
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

// =======================
// MPSGraph Caching Infrastructure
// =======================

namespace {

// Cache key structure for different operations
struct GraphCacheKey {
  std::string op_name;
  std::vector<int64_t> shape_params;
  int32_t dtype;
  bool transpose_flag;

  bool operator==(const GraphCacheKey& other) const {
    return op_name == other.op_name &&
           shape_params == other.shape_params &&
           dtype == other.dtype &&
           transpose_flag == other.transpose_flag;
  }
};

// Hash function for GraphCacheKey
struct GraphCacheKeyHash {
  std::size_t operator()(const GraphCacheKey& key) const {
    std::size_t hash = std::hash<std::string>{}(key.op_name);
    for (auto val : key.shape_params) {
      hash ^= std::hash<int64_t>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    hash ^= std::hash<int32_t>{}(key.dtype) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<bool>{}(key.transpose_flag) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

// Struct to store both the compiled graph and its tensors for reuse
struct CachedGraph {
  MPSGraph* graph;
  MPSGraphTensor* input1;
  MPSGraphTensor* input2;
  MPSGraphTensor* input3;  // Optional (e.g., bias, mask)
  MPSGraphTensor* output;
};

// Global cache for compiled MPSGraphs
// These graphs are never released - they're reused across calls
static std::unordered_map<GraphCacheKey, CachedGraph, GraphCacheKeyHash> graph_cache;

// Statistics for monitoring cache effectiveness
struct CacheStats {
  size_t hits = 0;
  size_t misses = 0;

  void logStats() {
    if ((hits + misses) % 100 == 0 && (hits + misses) > 0) {
      double hit_rate = 100.0 * hits / (hits + misses);
      ET_LOG(Debug, "MPSGraph cache stats: %zu hits, %zu misses (%.1f%% hit rate)",
             hits, misses, hit_rate);
    }
  }
};

static CacheStats cache_stats;

// Helper function to get Metal buffer from the global mapping
static id<MTLBuffer> get_mtl_buffer(Tensor* tensor, const char* op_name, const char* tensor_name) {
  void* data_ptr = tensor->mutable_data_ptr();
  auto it = ptr_to_mtl_buffer.find(data_ptr);
  if (it == ptr_to_mtl_buffer.end()) {
    ET_LOG(Error, "%s: %s tensor not found in Metal buffer mapping", op_name, tensor_name);
    throw std::runtime_error(std::string(tensor_name) + " tensor not found in Metal buffer mapping");
  }
  return it->second;
}

// Helper function to allocate a Metal buffer and register it in the global mapping.
static id<MTLBuffer> allocate_mtl_buffer(void** data_ptr, size_t size_bytes) {
  AOTITorchError malloc_err = aoti_torch_mps_malloc(data_ptr, size_bytes);
  if (malloc_err != Error::Ok) {
    ET_LOG(Error, "allocate_and_register_mtl_buffer: Failed to allocate Metal buffer via aoti_torch_mps_malloc");
    throw std::runtime_error("Failed to allocate output Metal buffer");
  }

  auto it = ptr_to_mtl_buffer.find(*data_ptr);
  if (it == ptr_to_mtl_buffer.end()) {
    ET_LOG(Error, "allocate_and_register_mtl_buffer: aoti_torch_mps_malloc did not register buffer in map");
    throw std::runtime_error("Failed to look up allocated Metal buffer");
  }
  return it->second;
}

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

// Global shader library cache for SDPA
static std::unique_ptr<ETMetalShaderLibrary> sdpa_shader_library = nullptr;

static std::once_flag sdpa_shader_library_once_flag;

static ETMetalShaderLibrary* get_sdpa_shader_library() {
  std::call_once(sdpa_shader_library_once_flag, []() {
    std::string source = get_sdpa_metal_source();
    sdpa_shader_library = std::make_unique<ETMetalShaderLibrary>(source);
  });
  return sdpa_shader_library.get();
}

}  // anonymous namespace

extern "C" {

AOTITorchError aoti_torch_mps_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  ET_LOG(Debug, "aoti_torch_mps_mm_out: Starting with out=%p, self=%p, mat2=%p",
         out, self, mat2);

  if (!out || !self || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto out_tensor = reinterpret_cast<Tensor*>(out);
      auto self_tensor = reinterpret_cast<Tensor*>(self);
      auto mat2_tensor = reinterpret_cast<Tensor*>(mat2);

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Converted tensor handles to ET tensors");

      // Validate tensor dimensions
      if (self_tensor->dim() != 2 || mat2_tensor->dim() != 2) {
        std::string error_msg = "aoti_torch_mps_mm_out: tensors must be 2-D, got " +
                               std::to_string(self_tensor->dim()) + " and " +
                               std::to_string(mat2_tensor->dim());
        ET_LOG(Error, "%s", error_msg.c_str());
        throw std::runtime_error(error_msg);
      }

      int64_t M = self_tensor->sizes()[0];  // rows of self
      int64_t K = self_tensor->sizes()[1];  // cols of self / rows of mat2
      int64_t N = mat2_tensor->sizes()[1];  // cols of mat2

      // Check matrix multiplication compatibility
      if (self_tensor->sizes()[1] != mat2_tensor->sizes()[0]) {
        std::string error_msg = "aoti_torch_mps_mm_out: incompatible matrix sizes for mm (" +
                               std::to_string(M) + "x" + std::to_string(K) + " and " +
                               std::to_string(mat2_tensor->sizes()[0]) + "x" + std::to_string(N) + ")";
        ET_LOG(Error, "%s", error_msg.c_str());
        throw std::runtime_error(error_msg);
      }

      // Log tensor shapes for debugging
      ET_LOG(Debug, "aoti_torch_mps_mm_out: self shape: [%d, %d], mat2 shape: [%d, %d], out shape: [%d, %d]",
             (int)M, (int)K, (int)mat2_tensor->sizes()[0], (int)N,
             out_tensor->dim() > 0 ? (int)out_tensor->sizes()[0] : 0,
             out_tensor->dim() > 1 ? (int)out_tensor->sizes()[1] : 0);

      // Check if mat2 is transposed (non-contiguous due to transpose)
      // A transposed matrix will have stride(-2) == 1 (column-major instead of row-major)
      // For a 2D tensor with shape [K, N]:
      //   - Contiguous (row-major): strides = [N, 1]
      //   - Transposed (column-major): strides = [1, K]
      bool mat2_is_transposed = false;
      int64_t mat2_stride_0 = mat2_tensor->strides()[0];  // stride for dimension 0
      int64_t mat2_stride_1 = mat2_tensor->strides()[1];  // stride for dimension 1

      // Detect transposed layout: stride(-2) == 1 indicates column-major layout
      if (mat2_stride_0 == 1 && mat2_stride_1 != 1) {
        mat2_is_transposed = true;
        ET_LOG(Debug, "aoti_torch_mps_mm_out: mat2 is transposed (strides=[%lld, %lld])",
               mat2_stride_0, mat2_stride_1);
      } else {
        ET_LOG(Debug, "aoti_torch_mps_mm_out: mat2 is contiguous (strides=[%lld, %lld])",
               mat2_stride_0, mat2_stride_1);
      }

      // Use the same dispatch pattern as other MPS operations for consistent synchronization
      ETMetalStream* stream = getCurrentMetalStream();
      if (!stream) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get current Metal stream");
        return Error::Internal;
      }

      // Get Metal device
      id<MTLDevice> device = get_metal_device();
      if (!device) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get Metal device");
        throw std::runtime_error("Failed to get Metal device");
      }

      // Get Metal buffers for input and output tensors
      id<MTLBuffer> self_buffer = get_mtl_buffer(self_tensor, "aoti_torch_mps_mm_out", "self");
      id<MTLBuffer> mat2_buffer = get_mtl_buffer(mat2_tensor, "aoti_torch_mps_mm_out", "mat2");
      id<MTLBuffer> out_buffer = get_mtl_buffer(out_tensor, "aoti_torch_mps_mm_out", "out");

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Using existing Metal buffers - self=%p, mat2=%p, out=%p",
             self_buffer, mat2_buffer, out_buffer);

      // End any existing kernel coalescing to ensure a clean state for MPS
      stream->endKernelCoalescing();

      // Determine data type and element size
      int32_t dtype = static_cast<int32_t>(self_tensor->scalar_type());
      MPSDataType mps_dtype;
      size_t element_size;

      ET_LOG(Debug, "aoti_torch_mps_mm_out: self_tensor scalar_type=%d, SupportedDTypes::FLOAT32=%d, SupportedDTypes::BFLOAT16=%d",
             dtype, static_cast<int32_t>(SupportedDTypes::FLOAT32), static_cast<int32_t>(SupportedDTypes::BFLOAT16));

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        mps_dtype = MPSDataTypeFloat32;
        element_size = sizeof(float);
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        mps_dtype = MPSDataTypeBFloat16;
        element_size = sizeof(uint16_t);  // bfloat16 is 16 bits
      } else {
        ET_LOG(Error, "aoti_torch_mps_mm_out: Unsupported data type: %d", dtype);
        throw std::runtime_error("Unsupported data type for matrix multiplication");
      }

      ET_LOG(Debug, "aoti_torch_mps_mm_out: dtype=%d, element_size=%zu", dtype, element_size);
      ET_LOG(Debug, "aoti_torch_mps_mm_out: M=%lld, K=%lld, N=%lld", M, K, N);

      // Define tensor shapes for placeholders (needed for both cache hit and miss)
      NSArray<NSNumber*>* selfShape = @[@(M), @(K)];

      // For mat2, we need to handle both contiguous and transposed cases
      // If mat2 is transposed, its physical layout in memory is [N, K] (column-major)
      // but logically we need [K, N] for the matrix multiplication
      NSArray<NSNumber*>* mat2PhysicalShape;
      if (mat2_is_transposed) {
        // Physical shape reflects the actual memory layout (transposed)
        mat2PhysicalShape = @[@(N), @(K)];
        ET_LOG(Debug, "aoti_torch_mps_mm_out: mat2 physical shape (transposed): [%d,%d]", (int)N, (int)K);
      } else {
        // Physical shape is the logical shape (contiguous)
        mat2PhysicalShape = @[@(K), @(N)];
        ET_LOG(Debug, "aoti_torch_mps_mm_out: mat2 physical shape (contiguous): [%d,%d]", (int)K, (int)N);
      }

      // Create cache key for this matrix multiplication
      GraphCacheKey cache_key;
      cache_key.op_name = "mm";
      cache_key.shape_params = {M, K, N};
      cache_key.dtype = dtype;
      cache_key.transpose_flag = mat2_is_transposed;

      // Check if we have a cached graph
      MPSGraph* mpsGraph = nullptr;
      MPSGraphTensor* mmOutput = nil;
      MPSGraphTensor* selfPlaceholder = nil;
      MPSGraphTensor* mat2Placeholder = nil;

      auto cache_it = graph_cache.find(cache_key);
      if (cache_it != graph_cache.end()) {
        // Cache hit - reuse compiled graph and tensor references
        CachedGraph& cached = cache_it->second;
        mpsGraph = cached.graph;
        selfPlaceholder = cached.input1;
        mat2Placeholder = cached.input2;
        mmOutput = cached.output;

        cache_stats.hits++;
        cache_stats.logStats();
        ET_LOG(Debug, "aoti_torch_mps_mm_out: Using cached MPSGraph (cache hit, %zu total hits)", cache_stats.hits);

      } else {
        // Cache miss - create and compile new graph
        mpsGraph = [MPSGraph new];
        cache_stats.misses++;
        cache_stats.logStats();
        ET_LOG(Debug, "aoti_torch_mps_mm_out: Created new MPSGraph instance (cache miss, %zu total misses)", cache_stats.misses);

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Creating placeholders with shapes self:[%d,%d] mat2:[%d,%d]",
                (int)M, (int)K,
                mat2_is_transposed ? (int)N : (int)K,
                mat2_is_transposed ? (int)K : (int)N);

        // Create placeholders for input tensors
        selfPlaceholder = [mpsGraph placeholderWithShape:selfShape
                                                dataType:mps_dtype
                                                    name:@"self"];
        mat2Placeholder = [mpsGraph placeholderWithShape:mat2PhysicalShape
                                                dataType:mps_dtype
                                                    name:@"mat2_physical"];

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Created input placeholders");

        // If mat2 is transposed, apply transpose operation in the graph to get the logical shape
        MPSGraphTensor* mat2Logical;
        if (mat2_is_transposed) {
          // Transpose from physical [N, K] to logical [K, N]
          // MPSGraph transposeTensor swaps the last two dimensions for 2D tensors
          mat2Logical = [mpsGraph transposeTensor:mat2Placeholder
                                        dimension:-2
                                    withDimension:-1
                                              name:@"mat2_transposed"];
          ET_LOG(Debug, "aoti_torch_mps_mm_out: Applied transpose operation to mat2 in graph");
        } else {
          // No transpose needed, use placeholder directly
          mat2Logical = mat2Placeholder;
          ET_LOG(Debug, "aoti_torch_mps_mm_out: Using mat2 placeholder directly (no transpose needed)");
        }

        // Perform matrix multiplication using MPSGraph with the logical mat2 tensor
        mmOutput = [mpsGraph matrixMultiplicationWithPrimaryTensor:selfPlaceholder
                                                                    secondaryTensor:mat2Logical
                                                                              name:@"matrix_multiplication"];

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Successfully created matrix multiplication tensor");

        // Cache the compiled graph and tensor references for reuse
        CachedGraph cached_graph;
        cached_graph.graph = mpsGraph;
        cached_graph.input1 = selfPlaceholder;
        cached_graph.input2 = mat2Placeholder;
        cached_graph.input3 = nil;
        cached_graph.output = mmOutput;
        graph_cache[cache_key] = cached_graph;

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Cached compiled MPSGraph for future reuse");
      }  // End of cache miss/hit block

      // Define output shape
      NSArray<NSNumber*>* outShape = @[@(M), @(N)];

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Create MPSGraphTensorData objects for input tensors
      // Use physical shapes to match how data is actually laid out in memory
      MPSGraphTensorData* selfData = [[MPSGraphTensorData alloc] initWithMTLBuffer:self_buffer
                                                                              shape:selfShape
                                                                           dataType:mps_dtype];
      MPSGraphTensorData* mat2Data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mat2_buffer
                                                                              shape:mat2PhysicalShape
                                                                           dataType:mps_dtype];

      feeds[selfPlaceholder] = selfData;
      feeds[mat2Placeholder] = mat2Data;

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Created feeds dictionary with physical shapes");

      // Create results dictionary
      MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                                               shape:outShape
                                                                            dataType:mps_dtype];

      NSDictionary* results = @{mmOutput: outputData};
      ET_LOG(Debug, "aoti_torch_mps_mm_out: Created results dictionary");

      // Execute the MPSGraph
      ET_LOG(Debug, "aoti_torch_mps_mm_out: Executing MPSGraph");

      @try {
        // Use stream helper to encode and synchronize correctly
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        throw std::runtime_error("MPSGraph execution failed with NSException");
      }

      ET_LOG(Debug, "aoti_torch_mps_mm_out: MPSGraph execution completed successfully");

      [selfData release];
      [mat2Data release];
      [outputData release];

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Executed successfully");
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_mm_out exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_mm_out: unknown exception");
      return Error::Internal;
    }
  }
}

AOTITorchError aoti_torch_mps_bmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {

  // Validate non-null handles
  if (!out || !self || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_bmm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto out_tensor = reinterpret_cast<Tensor*>(out);
      auto self_tensor = reinterpret_cast<Tensor*>(self);
      auto mat2_tensor = reinterpret_cast<Tensor*>(mat2);

      // Validate tensor dimensions - bmm requires 3-D tensors
      if (self_tensor->dim() != 3 || mat2_tensor->dim() != 3 || out_tensor->dim() != 3) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: tensors must be 3-D. "
               "Got self.dim=%zd (shape=[%d,%d,%d]), "
               "mat2.dim=%zd (shape=[%d,%d,%d]), "
               "out.dim=%zd (shape=[%d,%d,%d])",
               self_tensor->dim(),
               self_tensor->dim() > 0 ? (int)self_tensor->sizes()[0] : 0,
               self_tensor->dim() > 1 ? (int)self_tensor->sizes()[1] : 0,
               self_tensor->dim() > 2 ? (int)self_tensor->sizes()[2] : 0,
               mat2_tensor->dim(),
               mat2_tensor->dim() > 0 ? (int)mat2_tensor->sizes()[0] : 0,
               mat2_tensor->dim() > 1 ? (int)mat2_tensor->sizes()[1] : 0,
               mat2_tensor->dim() > 2 ? (int)mat2_tensor->sizes()[2] : 0,
               out_tensor->dim(),
               out_tensor->dim() > 0 ? (int)out_tensor->sizes()[0] : 0,
               out_tensor->dim() > 1 ? (int)out_tensor->sizes()[1] : 0,
               out_tensor->dim() > 2 ? (int)out_tensor->sizes()[2] : 0);
        return Error::InvalidArgument;
      }

      int64_t B = self_tensor->sizes()[0];  // batch size
      int64_t M = self_tensor->sizes()[1];  // rows of self
      int64_t K = self_tensor->sizes()[2];  // cols of self / rows of mat2
      int64_t N = mat2_tensor->sizes()[2];  // cols of mat2

      // Validate shape constraints
      // self: [B, M, K], mat2: [B, K, N], out: [B, M, N]
      if (mat2_tensor->sizes()[0] != B) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: batch size mismatch. "
               "Expected mat2[0]=%d to match self[0]=%lld. "
               "self.shape=[%lld,%lld,%lld], mat2.shape=[%d,%d,%d]",
               (int)mat2_tensor->sizes()[0], (long long)B,
               (long long)B, (long long)M, (long long)K,
               (int)mat2_tensor->sizes()[0], (int)mat2_tensor->sizes()[1], (int)mat2_tensor->sizes()[2]);
        return Error::InvalidArgument;
      }

      if (mat2_tensor->sizes()[1] != K) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: incompatible matrix dimensions for bmm. "
               "Expected mat2[1]=%d to match self[2]=%lld. "
               "Cannot multiply [%lld,%lld,%lld] @ [%d,%d,%d]",
               (int)mat2_tensor->sizes()[1], (long long)K,
               (long long)B, (long long)M, (long long)K,
               (int)mat2_tensor->sizes()[0], (int)mat2_tensor->sizes()[1], (int)mat2_tensor->sizes()[2]);
        return Error::InvalidArgument;
      }

      if (out_tensor->sizes()[0] != B || out_tensor->sizes()[1] != M || out_tensor->sizes()[2] != N) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: output shape mismatch. "
               "Expected out.shape=[%lld,%lld,%lld], got [%d,%d,%d]",
               (long long)B, (long long)M, (long long)N,
               (int)out_tensor->sizes()[0], (int)out_tensor->sizes()[1], (int)out_tensor->sizes()[2]);
        return Error::InvalidArgument;
      }

      // Validate dtype consistency
      int32_t self_dtype = static_cast<int32_t>(self_tensor->scalar_type());
      int32_t mat2_dtype = static_cast<int32_t>(mat2_tensor->scalar_type());
      int32_t out_dtype = static_cast<int32_t>(out_tensor->scalar_type());

      if (self_dtype != mat2_dtype || self_dtype != out_dtype) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: dtype mismatch. "
               "All tensors must have same dtype. Got self.dtype=%d, mat2.dtype=%d, out.dtype=%d",
               self_dtype, mat2_dtype, out_dtype);
        return Error::InvalidArgument;
      }

      int32_t dtype = self_dtype;

      // Validate layout: BMM requires strictly contiguous 3D tensors
      // For shape [B, M, K], contiguous strides MUST be [M*K, K, 1]
      //
      // Why strict contiguity is required:
      // - MPSGraphTensorData initWithMTLBuffer:shape:dataType: interprets the MTLBuffer
      //   as containing dense row-major data for the given shape
      // - Non-contiguous layouts (transposed, views with strides, etc.) have different
      //   memory layouts that don't match what MPS expects
      // - This would result in SILENT WRONG RESULTS
      // - This is an _out op: we must NOT create implicit copies
      // - Policy: Reject non-contiguous inputs explicitly (transposed/view tensors unsupported)
      //
      // Limitation: This implementation does not explicitly check storage offset (no API available).
      // Tensors with non-zero storage offsets are not explicitly rejected but may work if they
      // happen to have contiguous strides. Users should ensure tensors are base tensors without offsets.
      auto self_strides = self_tensor->strides();
      auto mat2_strides = mat2_tensor->strides();
      auto out_strides = out_tensor->strides();

      // Check self tensor is contiguous [B, M, K] with strides [M*K, K, 1]
      if (self_strides[2] != 1 || self_strides[1] != K || self_strides[0] != M * K) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: self tensor must be contiguous. "
               "Only dense row-major layout supported; transposed/view tensors are unsupported. "
               "Expected strides=[%lld,%lld,1] for shape=[%lld,%lld,%lld], got strides=[%d,%d,%d].",
               (long long)(M * K), (long long)K, (long long)B, (long long)M, (long long)K,
               self_strides[0], self_strides[1], self_strides[2]);
        return Error::InvalidArgument;
      }

      // Check mat2 tensor is contiguous [B, K, N] with strides [K*N, N, 1]
      if (mat2_strides[2] != 1 || mat2_strides[1] != N || mat2_strides[0] != K * N) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: mat2 tensor must be contiguous. "
               "Only dense row-major layout supported; transposed/view tensors are unsupported. "
               "Expected strides=[%lld,%lld,1] for shape=[%lld,%lld,%lld], got strides=[%d,%d,%d].",
               (long long)(K * N), (long long)N, (long long)B, (long long)K, (long long)N,
               mat2_strides[0], mat2_strides[1], mat2_strides[2]);
        return Error::InvalidArgument;
      }

      // Check out tensor is contiguous [B, M, N] with strides [M*N, N, 1]
      if (out_strides[2] != 1 || out_strides[1] != N || out_strides[0] != M * N) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: out tensor must be contiguous. "
               "Only dense row-major layout supported; transposed/view tensors are unsupported. "
               "Expected strides=[%lld,%lld,1] for shape=[%lld,%lld,%lld], got strides=[%d,%d,%d].",
               (long long)(M * N), (long long)N, (long long)B, (long long)M, (long long)N,
               out_strides[0], out_strides[1], out_strides[2]);
        return Error::InvalidArgument;
      }

      // Get Metal stream and device
      ETMetalStream* stream = getCurrentMetalStream();
      if (!stream) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: Failed to get current Metal stream");
        return Error::Internal;
      }

      id<MTLDevice> device = get_metal_device();
      if (!device) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: Failed to get Metal device");
        return Error::Internal;
      }
      (void)device;  // Used for validation, consistent with other ops

      // Get Metal buffers for input and output tensors
      id<MTLBuffer> self_buffer = get_mtl_buffer(self_tensor, "aoti_torch_mps_bmm_out", "self");
      id<MTLBuffer> mat2_buffer = get_mtl_buffer(mat2_tensor, "aoti_torch_mps_bmm_out", "mat2");
      id<MTLBuffer> out_buffer = get_mtl_buffer(out_tensor, "aoti_torch_mps_bmm_out", "out");

      // Validate buffers are non-null
      if (!self_buffer || !mat2_buffer || !out_buffer) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: Failed to get Metal buffers. "
               "self_buffer=%p, mat2_buffer=%p, out_buffer=%p",
               self_buffer, mat2_buffer, out_buffer);
        return Error::Internal;
      }

      // End any existing kernel coalescing to ensure clean state
      // (consistent with mm_out and conv pattern)
      stream->endKernelCoalescing();

      // Map dtype to MPS type and validate support
      // Note: Only FLOAT32 and BFLOAT16 are supported in Metal backend (see utils.h)
      // FLOAT16 is not in SupportedDTypes enum and is not supported
      MPSDataType mps_dtype;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        mps_dtype = MPSDataTypeFloat32;
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        mps_dtype = MPSDataTypeBFloat16;
      } else {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: Unsupported data type: %d. "
               "Supported types: FLOAT32 (%d), BFLOAT16 (%d)",
               dtype,
               static_cast<int32_t>(SupportedDTypes::FLOAT32),
               static_cast<int32_t>(SupportedDTypes::BFLOAT16));
        return Error::InvalidArgument;
      }

      // Define shapes for graph placeholders and tensor data
      NSArray<NSNumber*>* selfShape = @[@(B), @(M), @(K)];
      NSArray<NSNumber*>* mat2Shape = @[@(B), @(K), @(N)];
      NSArray<NSNumber*>* outShape = @[@(B), @(M), @(N)];

      // Create cache key for this batched matrix multiplication
      // Cache key includes: op_name, shape params {B, M, K, N}, dtype, transpose_flag
      // This allows reuse when same BMM shape/dtype is called repeatedly
      GraphCacheKey cache_key;
      cache_key.op_name = "bmm";
      cache_key.shape_params = {B, M, K, N};
      cache_key.dtype = dtype;
      cache_key.transpose_flag = false;  // BMM has no transpose handling

      // Check if we have a cached graph
      MPSGraph* mpsGraph = nullptr;
      MPSGraphTensor* outputTensor = nil;
      MPSGraphTensor* selfPlaceholder = nil;
      MPSGraphTensor* mat2Placeholder = nil;

      auto cache_it = graph_cache.find(cache_key);
      if (cache_it != graph_cache.end()) {
        // Cache hit - reuse compiled graph and tensor references
        CachedGraph& cached = cache_it->second;
        mpsGraph = cached.graph;
        selfPlaceholder = cached.input1;
        mat2Placeholder = cached.input2;
        outputTensor = cached.output;

        cache_stats.hits++;
        cache_stats.logStats();

      } else {
        // Cache miss - create and compile new graph
        mpsGraph = [MPSGraph new];
        cache_stats.misses++;
        cache_stats.logStats();

        // Create 3D placeholders for batched matrices
        // These represent the logical shapes for the batched matrix multiplication
        selfPlaceholder = [mpsGraph placeholderWithShape:selfShape
                                                dataType:mps_dtype
                                                    name:@"self"];
        mat2Placeholder = [mpsGraph placeholderWithShape:mat2Shape
                                                dataType:mps_dtype
                                                    name:@"mat2"];

        // MPSGraph matrixMultiplication handles batched case natively when given 3D tensors
        // For 3D inputs [B,M,K] @ [B,K,N] -> [B,M,N]
        outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:selfPlaceholder
                                                       secondaryTensor:mat2Placeholder
                                                                  name:@"bmm_result"];

        // Cache the compiled graph and tensor references for reuse
        CachedGraph cached_graph;
        cached_graph.graph = mpsGraph;
        cached_graph.input1 = selfPlaceholder;
        cached_graph.input2 = mat2Placeholder;
        cached_graph.input3 = nil;  // No third input for BMM
        cached_graph.output = outputTensor;
        graph_cache[cache_key] = cached_graph;

      }  // End of cache miss/hit block

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Create MPSGraphTensorData objects for input tensors
      // These wrap the MTLBuffers with the shape information
      // Initialize to nil for safe cleanup in exception path
      MPSGraphTensorData* selfData = nil;
      MPSGraphTensorData* mat2Data = nil;
      MPSGraphTensorData* outputData = nil;

      selfData = [[MPSGraphTensorData alloc] initWithMTLBuffer:self_buffer
                                                          shape:selfShape
                                                       dataType:mps_dtype];
      mat2Data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mat2_buffer
                                                          shape:mat2Shape
                                                       dataType:mps_dtype];

      feeds[selfPlaceholder] = selfData;
      feeds[mat2Placeholder] = mat2Data;

      // Create output tensor data
      outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                           shape:outShape
                                                        dataType:mps_dtype];

      // Build results dictionary
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        outputTensor: outputData
      };

      // Execute the batched matrix multiplication
      @try {
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        // Guard releases against nil
        if (selfData) [selfData release];
        if (mat2Data) [mat2Data release];
        if (outputData) [outputData release];
        return Error::Internal;
      }

      // Release MPSGraphTensorData objects
      [selfData release];
      [mat2Data release];
      [outputData release];

      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_bmm_out exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_bmm_out: unknown exception");
      return Error::Internal;
    }
  }
}

AOTITorchError aoti_torch_mps_convolution(
    AOTITensorHandle input,
    AOTITensorHandle weight,
    AOTITensorHandle* bias,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_len_,
    int64_t groups,
    AOTITensorHandle* ret0) {
  ET_LOG(Debug, "aoti_torch_mps_convolution: Starting with input=%p, weight=%p, bias=%p, groups=%lld, transposed=%d",
         input, weight, bias, groups, transposed);

  if (!input || !weight || !ret0) {
    ET_LOG(Error, "aoti_torch_mps_convolution: null required handles (input, weight, or ret0)");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto input_tensor = reinterpret_cast<Tensor*>(input);
      auto weight_tensor = reinterpret_cast<Tensor*>(weight);

      // bias can be null for convolutions without bias
      Tensor* bias_tensor = nullptr;
      if (bias && *bias) {
        bias_tensor = reinterpret_cast<Tensor*>(*bias);
        ET_LOG(Debug, "aoti_torch_mps_convolution: Has bias tensor");
      } else {
        ET_LOG(Debug, "aoti_torch_mps_convolution: No bias tensor");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Converted tensor handles to ET tensors");

      // Log tensor shapes for debugging
      ET_LOG(Debug, "aoti_torch_mps_convolution: input shape: [%d, %d, %d, %d]",
             input_tensor->dim() > 0 ? (int)input_tensor->sizes()[0] : 0,
             input_tensor->dim() > 1 ? (int)input_tensor->sizes()[1] : 0,
             input_tensor->dim() > 2 ? (int)input_tensor->sizes()[2] : 0,
             input_tensor->dim() > 3 ? (int)input_tensor->sizes()[3] : 0);

      ET_LOG(Debug, "aoti_torch_mps_convolution: weight shape: [%d, %d, %d, %d]",
             weight_tensor->dim() > 0 ? (int)weight_tensor->sizes()[0] : 0,
             weight_tensor->dim() > 1 ? (int)weight_tensor->sizes()[1] : 0,
             weight_tensor->dim() > 2 ? (int)weight_tensor->sizes()[2] : 0,
             weight_tensor->dim() > 3 ? (int)weight_tensor->sizes()[3] : 0);

      // Log convolution parameters
      if (stride && stride_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: stride: [%lld, %lld]", stride[0], stride[1]);
      }
      if (padding && padding_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: padding: [%lld, %lld]", padding[0], padding[1]);
      }
      if (dilation && dilation_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: dilation: [%lld, %lld]", dilation[0], dilation[1]);
      }
      if (output_padding && output_padding_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: output_padding: [%lld, %lld]", output_padding[0], output_padding[1]);
      }

      // Support conv1d and conv2d by inspecting weight rank.
      // conv1d: weight dims = [C_out, C_in, K]
      // conv2d: weight dims = [C_out, C_in, Kh, Kw]
      bool is_conv1d = (weight_tensor->dim() == 3);

      // Accept input ranks:
      // conv1d: 2D (C,W) or 3D (N,C,W)
      // conv2d: 3D (C,H,W) or 4D (N,C,H,W)
      bool has_batch_dim = false;
      bool is_input_4d = false;
      int64_t N = 1, C_in = 0, H_in = 1, W_in = 0;
      if (is_conv1d) {
        if (input_tensor->dim() == 2) {
          // (C, W)
          has_batch_dim = false;
          C_in = input_tensor->sizes()[0];
          W_in = input_tensor->sizes()[1];
          H_in = 1;
        } else if (input_tensor->dim() == 3) {
          // (N, C, W)
          has_batch_dim = true;
          N = input_tensor->sizes()[0];
          C_in = input_tensor->sizes()[1];
          W_in = input_tensor->sizes()[2];
          H_in = 1;
        } else {
          ET_LOG(Error, "aoti_torch_mps_convolution: conv1d expects 2D or 3D input, got %d", (int)input_tensor->dim());
          return Error::InvalidArgument;
        }
      } else {
        is_input_4d = (input_tensor->dim() == 4);
        if (is_input_4d) {
          // (N, C, H, W)
          has_batch_dim = true;
          N = input_tensor->sizes()[0];
          C_in = input_tensor->sizes()[1];
          H_in = input_tensor->sizes()[2];
          W_in = input_tensor->sizes()[3];
        } else if (input_tensor->dim() == 3) {
          // (C, H, W)
          has_batch_dim = false;
          N = 1;
          C_in = input_tensor->sizes()[0];
          H_in = input_tensor->sizes()[1];
          W_in = input_tensor->sizes()[2];
        } else {
          ET_LOG(Error, "aoti_torch_mps_convolution: conv2d expects 3D or 4D input, got %d", (int)input_tensor->dim());
          return Error::InvalidArgument;
        }
      }

      // Get weight dimensions
      int64_t C_out = weight_tensor->sizes()[0];  // output channels
      int64_t kernel_h = is_conv1d ? 1 : weight_tensor->sizes()[2];  // kernel height
      int64_t kernel_w = is_conv1d ? weight_tensor->sizes()[2] : weight_tensor->sizes()[3];  // kernel width

      // Calculate output spatial dimensions
      int64_t stride_h = is_conv1d ? 1 : (stride && stride_len_ > 0 ? stride[0] : 1);
      int64_t stride_w = is_conv1d ? (stride && stride_len_ > 0 ? stride[0] : 1)
                                   : (stride && stride_len_ > 1 ? stride[1] : 1);
      int64_t pad_h = is_conv1d ? 0 : (padding && padding_len_ > 0 ? padding[0] : 0);
      int64_t pad_w = is_conv1d ? (padding && padding_len_ > 0 ? padding[0] : 0)
                                : (padding && padding_len_ > 1 ? padding[1] : 0);
      int64_t dil_h = is_conv1d ? 1 : (dilation && dilation_len_ > 0 ? dilation[0] : 1);
      int64_t dil_w = is_conv1d ? (dilation && dilation_len_ > 0 ? dilation[0] : 1)
                                : (dilation && dilation_len_ > 1 ? dilation[1] : 1);

      int64_t H_out, W_out;
      if (transposed) {
        // For transposed convolution, output size calculation is different
        int64_t output_pad_h = is_conv1d ? 0 : (output_padding && output_padding_len_ > 0 ? output_padding[0] : 0);
        int64_t output_pad_w = is_conv1d ? (output_padding && output_padding_len_ > 0 ? output_padding[0] : 0)
                                         : (output_padding && output_padding_len_ > 1 ? output_padding[1] : 0);
        H_out = is_conv1d ? 1 : ((H_in - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + output_pad_h + 1);
        W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kernel_w - 1) + output_pad_w + 1;
      } else {
        // Regular convolution output size calculation
        H_out = is_conv1d ? 1 : ((H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1);
        W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;
      }

      if (!is_conv1d && is_input_4d) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 4D output shape: [%lld, %lld, %lld, %lld]", N, C_out, H_out, W_out);
      } else if (!is_conv1d) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 3D output shape: [%lld, %lld, %lld]", C_out, H_out, W_out);
      } else if (is_conv1d && has_batch_dim) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 3D (1D conv) output shape: [%lld, %lld, %lld]", N, C_out, W_out);
      } else {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 2D (1D conv) output shape: [%lld, %lld]", C_out, W_out);
      }

      // Validate output dimensions are positive
      if (N <= 0 || C_out <= 0 || H_out <= 0 || W_out <= 0) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Invalid output dimensions N=%lld, C_out=%lld, H_out=%lld, W_out=%lld",
               N, C_out, H_out, W_out);
        return Error::InvalidArgument;
      }

      // Use the same dispatch pattern as other MPS operations for consistent synchronization
      ETMetalStream* stream = getCurrentMetalStream();
      if (!stream) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to get current Metal stream");
        return Error::Internal;
      }

      // Get Metal device
      id<MTLDevice> device = get_metal_device();
      if (!device) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to get Metal device");
        throw std::runtime_error("Failed to get Metal device");
      }

      // End any existing kernel coalescing to ensure a clean state for MPS
      stream->endKernelCoalescing();

      // Ensure stream is ready; command buffer handled internally by stream helpers

      // Determine data type and element size
      int32_t dtype = static_cast<int32_t>(input_tensor->scalar_type());
      MPSDataType mps_dtype;
      size_t element_size;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        mps_dtype = MPSDataTypeFloat32;
        element_size = sizeof(float);
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        mps_dtype = MPSDataTypeBFloat16;
        element_size = sizeof(uint16_t);  // bfloat16 is 16 bits
      } else {
        ET_LOG(Error, "aoti_torch_mps_convolution: Unsupported data type: %d", dtype);
        throw std::runtime_error("Unsupported data type for convolution");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: mps_dtype=%d, element_size=%zu", mps_dtype, element_size);
      // Get weight's input channel dimension from the weight tensor (not from input)
      // For grouped convolutions, weight shape is [C_out, C_in/groups, kH, kW]
      int64_t weight_C_in = weight_tensor->sizes()[1];  // This handles grouped convs correctly

      // Define tensor shapes for placeholders (needed for both cache hit and miss)
      NSArray<NSNumber*>* inputShape = @[@(N), @(C_in), @(H_in), @(W_in)];
      NSArray<NSNumber*>* weightShape = @[@(C_out), @(weight_C_in), @(kernel_h), @(kernel_w)];

      // Create cache key for this convolution
      GraphCacheKey cache_key;
      cache_key.op_name = "conv";
      cache_key.shape_params = {N, C_in, H_in, W_in, C_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups};
      cache_key.dtype = dtype;
      cache_key.transpose_flag = (transposed != 0);

      // Check if we have a cached graph
      MPSGraph* mpsGraph = nullptr;
      MPSGraphTensor* convOutput = nil;
      MPSGraphTensor* finalOutput = nil;
      MPSGraphTensor* inputPlaceholder = nil;
      MPSGraphTensor* weightPlaceholder = nil;
      MPSGraphTensor* biasPlaceholder = nil;
      bool has_bias = (bias_tensor != nullptr);

      auto cache_it = graph_cache.find(cache_key);
      if (cache_it != graph_cache.end()) {
        // Cache hit - reuse compiled graph and tensor references
        CachedGraph& cached = cache_it->second;
        mpsGraph = cached.graph;
        inputPlaceholder = cached.input1;
        weightPlaceholder = cached.input2;
        biasPlaceholder = cached.input3;  // May be nil if no bias
        finalOutput = cached.output;

        cache_stats.hits++;
        cache_stats.logStats();
        ET_LOG(Debug, "aoti_torch_mps_convolution: Using cached MPSGraph (cache hit, %zu total hits)", cache_stats.hits);

      } else {
        // Cache miss - create and compile new graph
        mpsGraph = [MPSGraph new];
        cache_stats.misses++;
        cache_stats.logStats();
        ET_LOG(Debug, "aoti_torch_mps_convolution: Created new MPSGraph instance (cache miss, %zu total misses)", cache_stats.misses);

        ET_LOG(Debug, "aoti_torch_mps_convolution: Creating placeholders with shapes input:[%d,%d,%d,%d] weight:[%d,%d,%d,%d]",
                (int)N, (int)C_in, (int)H_in, (int)W_in,
                (int)C_out, (int)C_in, (int)kernel_h, (int)kernel_w);

        // Create placeholders for input tensors
        inputPlaceholder = [mpsGraph placeholderWithShape:inputShape
                                                  dataType:mps_dtype
                                                      name:@"input"];
        weightPlaceholder = [mpsGraph placeholderWithShape:weightShape
                                                    dataType:mps_dtype
                                                        name:@"weight"];

        ET_LOG(Debug, "aoti_torch_mps_convolution: Created input and weight placeholders");

        // Create convolution descriptor
        MPSGraphConvolution2DOpDescriptor* convDesc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride_w
                                                                                                        strideInY:stride_h
                                                                                                      dilationRateInX:dil_w
                                                                                                      dilationRateInY:dil_h
                                                                                                          groups:groups
                                                                                                          paddingLeft:pad_w
                                                                                                        paddingRight:pad_w
                                                                                                          paddingTop:pad_h
                                                                                                        paddingBottom:pad_h
                                                                                                          paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                          dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        ET_LOG(Debug, "aoti_torch_mps_convolution: Created convolution descriptor with stride=[%lld,%lld], padding=[%lld,%lld], dilation=[%lld,%lld], groups=%lld",
                stride_w, stride_h, pad_w, pad_h, dil_w, dil_h, groups);

        // Perform convolution using MPSGraph
        if (transposed) {
          ET_LOG(Debug, "aoti_torch_mps_convolution: Using transposed convolution");
          // For transposed convolution, we need to handle output padding
          int64_t output_pad_h = output_padding && output_padding_len_ > 0 ? output_padding[0] : 0;
          int64_t output_pad_w = output_padding && output_padding_len_ > 1 ? output_padding[1] : 0;

          // For transposed convolution, we need to adjust the padding calculation
          // In transposed convolution, the effective padding is typically negative
          // and we use output_padding to control the final output size
          int64_t transposed_pad_h = pad_h - output_pad_h;
          int64_t transposed_pad_w = pad_w - output_pad_w;

          // Create transposed convolution descriptor with adjusted padding
          MPSGraphConvolution2DOpDescriptor* transposedConvDesc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride_w
                                                                                                                    strideInY:stride_h
                                                                                                              dilationRateInX:dil_w
                                                                                                              dilationRateInY:dil_h
                                                                                                                      groups:groups
                                                                                                                paddingLeft:transposed_pad_w
                                                                                                                paddingRight:transposed_pad_w
                                                                                                                  paddingTop:transposed_pad_h
                                                                                                              paddingBottom:transposed_pad_h
                                                                                                                paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                                dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                                            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

          convOutput = [mpsGraph convolution2DWithSourceTensor:inputPlaceholder
                                                    weightsTensor:weightPlaceholder
                                                        descriptor:transposedConvDesc
                                                              name:@"transposed_convolution"];
        } else {
          ET_LOG(Debug, "aoti_torch_mps_convolution: Using regular convolution");
          convOutput = [mpsGraph convolution2DWithSourceTensor:inputPlaceholder
                                                    weightsTensor:weightPlaceholder
                                                        descriptor:convDesc
                                                              name:@"convolution"];
        }

        ET_LOG(Debug, "aoti_torch_mps_convolution: Successfully created convolution tensor");

        // Handle bias if provided
        if (bias_tensor) {
          ET_LOG(Debug, "aoti_torch_mps_convolution: Adding bias to convolution output");

          // Create bias placeholder
          NSArray<NSNumber*>* biasShape = @[@(C_out)];
          biasPlaceholder = [mpsGraph placeholderWithShape:biasShape
                                                    dataType:mps_dtype
                                                        name:@"bias"];

          // Add bias to convolution output
          finalOutput = [mpsGraph additionWithPrimaryTensor:convOutput
                                            secondaryTensor:biasPlaceholder
                                                        name:@"add_bias"];

          ET_LOG(Debug, "aoti_torch_mps_convolution: Added bias placeholder to graph");
        } else {
          finalOutput = convOutput;
        }

        // Cache the compiled graph and tensor references for reuse
        CachedGraph cached_graph;
        cached_graph.graph = mpsGraph;
        cached_graph.input1 = inputPlaceholder;
        cached_graph.input2 = weightPlaceholder;
        cached_graph.input3 = biasPlaceholder;  // May be nil if no bias
        cached_graph.output = finalOutput;
        graph_cache[cache_key] = cached_graph;

        ET_LOG(Debug, "aoti_torch_mps_convolution: Cached compiled MPSGraph for future reuse");
      }  // End of cache miss block

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Get Metal buffers from tensors
      id<MTLBuffer> input_buffer = get_mtl_buffer(input_tensor, "aoti_torch_mps_convolution", "input");
      id<MTLBuffer> weight_buffer = get_mtl_buffer(weight_tensor, "aoti_torch_mps_convolution", "weight");

      ET_LOG(Debug, "aoti_torch_mps_convolution: Using existing Metal buffers - input=%p, weight=%p",
              input_buffer, weight_buffer);

      // Create MPSGraphTensorData objects for input tensors
      MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                                shape:inputShape
                                                                            dataType:mps_dtype];
      MPSGraphTensorData* weightData = [[MPSGraphTensorData alloc] initWithMTLBuffer:weight_buffer
                                                                                shape:weightShape
                                                                            dataType:mps_dtype];

      feeds[inputPlaceholder] = inputData;
      feeds[weightPlaceholder] = weightData;

      MPSGraphTensorData* biasData = nil;

      // Add bias data to feeds if provided
      if (bias_tensor && biasPlaceholder) {
        id<MTLBuffer> bias_buffer = get_mtl_buffer(bias_tensor, "aoti_torch_mps_convolution", "bias");

        NSArray<NSNumber*>* biasShape = @[@(C_out)];
        biasData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bias_buffer
                                                           shape:biasShape
                                                        dataType:mps_dtype];

        feeds[biasPlaceholder] = biasData;
        ET_LOG(Debug, "aoti_torch_mps_convolution: Added bias tensor to feeds");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created feeds dictionary");

      // Create Metal buffer for output tensor
      size_t output_size_bytes = N * C_out * H_out * W_out * element_size;
      void* output_contents_ptr = nullptr;
      id<MTLBuffer> output_buffer = allocate_mtl_buffer(&output_contents_ptr, output_size_bytes);

      // Create results dictionary (MPSGraph output is 4D)
      NSArray<NSNumber*>* outputShape = @[@(N), @(C_out), @(H_out), @(W_out)];
      MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer
                                                                                shape:outputShape
                                                                              dataType:mps_dtype];

      NSDictionary* results = @{finalOutput: outputData};
      ET_LOG(Debug, "aoti_torch_mps_convolution: Created results dictionary");

      // Execute the MPSGraph
      ET_LOG(Debug, "aoti_torch_mps_convolution: Executing MPSGraph");

      @try {
        // Use stream helper to encode and synchronize correctly
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_convolution: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        throw std::runtime_error("MPSGraph execution failed with NSException");
      } @catch (...) {
        ET_LOG(Error, "aoti_torch_mps_convolution: MPSGraph execution failed");
        throw std::runtime_error("MPSGraph execution failed");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: MPSGraph execution completed successfully");

      // Create output tensor handle on device (MPS) that points to GPU buffer
      std::vector<int64_t> output_sizes_int64;
      std::vector<int64_t> output_strides;
      if (!is_conv1d && is_input_4d) {
        output_sizes_int64 = {N, C_out, H_out, W_out};
        // Contiguous NCHW strides
        output_strides = {
            C_out * H_out * W_out,
            H_out * W_out,
            W_out,
            1
        };
      } else if (!is_conv1d) {
        output_sizes_int64 = {C_out, H_out, W_out};
        // Contiguous CHW strides
        output_strides = {
            H_out * W_out,
            W_out,
            1
        };
      } else if (is_conv1d && has_batch_dim) {
        output_sizes_int64 = {N, C_out, W_out};
        // Contiguous NCW strides
        output_strides = {
            C_out * W_out,
            W_out,
            1
        };
      } else {
        output_sizes_int64 = {C_out, W_out};
        // Contiguous CW strides
        output_strides = {
            W_out,
            1
        };
      }

      // Use the GPU buffer contents pointer directly for the tensor storage
      void* tensor_data = output_contents_ptr;

      AOTITensorHandle output_tensor_handle = nullptr;

      AOTITorchError create_result = aoti_torch_create_tensor_from_blob_v2(
          tensor_data,
          static_cast<int64_t>(output_sizes_int64.size()),  // ndim
          output_sizes_int64.data(),
          output_strides.data(),
          0,  // storage_offset
          dtype,  // dtype
          13,  // device_type (MPS)
          0,  // device_index
          &output_tensor_handle,
          0,  // layout (strided)
          nullptr,  // opaque_metadata
          0   // opaque_metadata_size
      );

      if (create_result != Error::Ok || !output_tensor_handle) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to create output tensor, error code: %d", static_cast<int>(create_result));
        aoti_torch_mps_free(tensor_data);  // Free the allocated GPU memory on failure
        throw std::runtime_error("Failed to create output tensor");
      }

      // Verify the tensor was created with the correct size
      auto* et_tensor = reinterpret_cast<Tensor*>(output_tensor_handle);
      size_t actual_numel = et_tensor->numel();
      size_t expected_numel = static_cast<size_t>(N * C_out * H_out * W_out);

      if (actual_numel != expected_numel) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Tensor size mismatch. Expected %zu, got %zu", expected_numel, actual_numel);
        aoti_torch_mps_free(tensor_data);  // Free the allocated GPU memory on failure
        throw std::runtime_error("Tensor size mismatch");
      }

      // Store the tensor handle - mark that we own the memory since we manually allocated it
      *ret0 = output_tensor_handle;
      // Mark that we own the memory for these tensors
      // Note: memory_to_n_tensor is managed automatically in aoti_torch_create_tensor_from_blob_v2
      // The function sets it to NOT_OWN, but we need to change it to 1 since we allocated it
      extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
      memory_to_n_tensor[tensor_data] = 1;

      [inputData release];
      [weightData release];
      if (biasData) [biasData release];
      [outputData release];

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created output tensor with %zu elements using MPSGraph", actual_numel);

      ET_LOG(Debug, "aoti_torch_mps_convolution: Executed successfully");
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_convolution exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_convolution: unknown exception");
      return Error::Internal;
    }
  }
}

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
      // Convert AOTITensorHandle to ExecutorTorch tensors
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
