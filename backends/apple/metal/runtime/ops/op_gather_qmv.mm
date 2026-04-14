/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gather-indexed quantized matmul for Mixture-of-Experts.
//
// gather_qmv_fast: y[i] = W[expert_idx[i]] @ x[i]  (M=1 per pair, GEMV)
//
// Extends the qmv_fast kernel (ported from MLX in op_linear_4bit.mm) with
// expert index-based pointer offsets — the same pattern as MLX's
// affine_gather_qmv_fast.
//
// The quantization format matches op_linear_4bit.mm (MLX affine):
//   dequant(w, scale, bias) = scale * w_accum + activation_sum * bias

#include <executorch/backends/apple/metal/runtime/ops/common.h>

namespace executorch {
namespace backends {
namespace metal {
namespace {

static std::string get_gather_qmv_metal_source() {
  return R"(
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    static constant constexpr const int SIMD_SIZE = 32;

    // 4-bit load_vector: pre-divides activations for the qdot bitmask trick.
    // Identical to op_linear_4bit.mm (from MLX, Copyright 2023-2024 Apple Inc., MIT License).
    template <typename T, typename U, int values_per_thread>
    inline U load_vector_4bit(constant T* x, thread U* x_thread) {
      U sum = 0;
      for (int i = 0; i < values_per_thread; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    // 4-bit qdot: quantized dot product using bitmask trick.
    template <typename U, int values_per_thread>
    inline U qdot_4bit(
        constant uint8_t* w,
        const thread U* x_thread,
        U scale,
        U bias,
        U sum) {
      U accum = 0;
      constant uint16_t* ws = (constant uint16_t*)w;
      for (int i = 0; i < (values_per_thread / 4); i++) {
        accum +=
            (x_thread[4 * i] * (ws[i] & 0x000f) +
            x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
            x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
            x_thread[4 * i + 3] * (ws[i] & 0xf000));
      }
      return scale * accum + sum * bias;
    }

    // 4-bit load_vector_safe: same as load_vector_4bit but handles partial reads.
    template <typename T, typename U, int values_per_thread>
    inline U load_vector_safe_4bit(constant T* x, thread U* x_thread, int N) {
      U sum = 0;
      for (int i = 0; i < N; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      for (int i = N; i < values_per_thread; i++) {
        x_thread[i] = 0;
      }
      return sum;
    }

    // 4-bit qdot_safe: handles partial K dimension.
    template <typename U, int values_per_thread>
    inline U qdot_safe_4bit(
        constant uint8_t* w,
        const thread U* x_thread,
        U scale,
        U bias,
        U sum,
        int N) {
      U accum = 0;
      constant uint16_t* ws = (constant uint16_t*)w;
      for (int i = 0; i < (N / 4); i++) {
        accum +=
            (x_thread[4 * i] * (ws[i] & 0x000f) +
            x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
            x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
            x_thread[4 * i + 3] * (ws[i] & 0xf000));
      }
      return scale * accum + sum * bias;
    }

    // gather_qmv_fast: per-expert quantized GEMV for MoE.
    //
    // Same as qmv_fast but offsets w/scales/biases by expert_indices[tid.x]
    // before the matmul loop. This is the M=1 (decode) path.
    //
    // Buffers:
    //   0: x [P, K]              — activations (P = num token-expert pairs)
    //   1: w [E, N, K/2]         — packed 4-bit expert weights
    //   2: scales [E, N, K/gs]   — per-group scales
    //   3: biases [E, N, K/gs]   — per-group biases (zero points)
    //   4: y [P, N]              — output
    //   5: sizes (P, K, N)
    //   6: expert_indices [P]    — expert index per pair
    //   7: expert_strides (w_stride, s_stride, b_stride) per expert
    template <typename T, int group_size>
    [[kernel]] void gather_qmv_fast(
        constant T* x [[buffer(0)]],
        constant uchar* w [[buffer(1)]],
        constant T* scales [[buffer(2)]],
        constant T* biases [[buffer(3)]],
        device T* y [[buffer(4)]],
        constant uint3 &sizes [[buffer(5)]],
        constant uint32_t* expert_indices [[buffer(6)]],
        constant uint3 &expert_strides [[buffer(7)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]) {
      const int in_vec_size = static_cast<int>(sizes.y); // K
      const int out_vec_size = static_cast<int>(sizes.z); // N

      constexpr int bits = 4;
      constexpr int packs_per_thread = 2;
      constexpr int num_simdgroups = 2;
      constexpr int results_per_simdgroup = 4;
      constexpr int pack_factor = 32 / bits; // 8
      constexpr int bytes_per_pack = 4;
      constexpr int values_per_thread = pack_factor * packs_per_thread; // 16
      constexpr int block_size = values_per_thread * SIMD_SIZE;
      constexpr int scale_step_per_thread = group_size / values_per_thread;

      // Offset to this expert's weights
      uint expert_idx = expert_indices[tid.x];
      constant uint8_t* ws = (constant uint8_t*)w + expert_idx * expert_strides.x;
      constant T* sc = scales + expert_idx * expert_strides.y;
      constant T* bi = biases + expert_idx * expert_strides.z;

      typedef float U;

      thread U x_thread[values_per_thread];
      thread U result[results_per_simdgroup] = {0};

      // Adjust positions within this expert's weight matrix
      const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
      const int in_vec_size_g = in_vec_size / group_size;
      const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
          simd_gid * results_per_simdgroup;

      ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
      sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
      bi += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
      x += tid.x * in_vec_size + simd_lid * values_per_thread;
      y += tid.x * out_vec_size + out_row;

      for (int k = 0; k < in_vec_size; k += block_size) {
        U sum = load_vector_4bit<T, U, values_per_thread>(x, x_thread);

        for (int row = 0; row < results_per_simdgroup; row++) {
          auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
          constant T* sl = sc + row * in_vec_size_g;
          constant T* bl = bi + row * in_vec_size_g;

          U s = sl[0];
          U b = bl[0];
          result[row] += qdot_4bit<U, values_per_thread>(wl, x_thread, s, b, sum);
        }

        ws += block_size * bytes_per_pack / pack_factor;
        sc += block_size / group_size;
        bi += block_size / group_size;
        x += block_size;
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) {
          y[row] = static_cast<T>(result[row]);
        }
      }
    }

    #define INSTANTIATE_GATHER_QMV_FAST(DTYPE, GSIZE)                                       \
      template [[host_name("gather_qmv_fast_4bit_" #GSIZE "_" #DTYPE)]] kernel void         \
      gather_qmv_fast<DTYPE, GSIZE>(                                                         \
          constant DTYPE * x [[buffer(0)]],                                                  \
          constant uchar * w [[buffer(1)]],                                                  \
          constant DTYPE * scales [[buffer(2)]],                                             \
          constant DTYPE * biases [[buffer(3)]],                                             \
          device DTYPE * y [[buffer(4)]],                                                    \
          constant uint3 & sizes [[buffer(5)]],                                              \
          constant uint32_t * expert_indices [[buffer(6)]],                                  \
          constant uint3 & expert_strides [[buffer(7)]],                                     \
          uint3 tid [[threadgroup_position_in_grid]],                                        \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                                  \
          uint simd_lid [[thread_index_in_simdgroup]])

    INSTANTIATE_GATHER_QMV_FAST(float, 32);
    INSTANTIATE_GATHER_QMV_FAST(float, 64);
    INSTANTIATE_GATHER_QMV_FAST(float, 128);
    INSTANTIATE_GATHER_QMV_FAST(bfloat, 32);
    INSTANTIATE_GATHER_QMV_FAST(bfloat, 64);
    INSTANTIATE_GATHER_QMV_FAST(bfloat, 128);

    // gather_qmv_impl: generic-K fallback (handles any K, any N).
    // Same as qmv_impl in op_linear_4bit.mm but with expert index offset.
    template <typename T, int group_size>
    [[kernel]] void gather_qmv_impl(
        constant T* x [[buffer(0)]],
        constant uchar* w [[buffer(1)]],
        constant T* scales [[buffer(2)]],
        constant T* biases [[buffer(3)]],
        device T* y [[buffer(4)]],
        constant uint3 &sizes [[buffer(5)]],
        constant uint32_t* expert_indices [[buffer(6)]],
        constant uint3 &expert_strides [[buffer(7)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]) {
      const int in_vec_size = static_cast<int>(sizes.y); // K
      const int out_vec_size = static_cast<int>(sizes.z); // N

      constexpr int bits = 4;
      constexpr int packs_per_thread = 2;
      constexpr int num_simdgroups = 2;
      constexpr int results_per_simdgroup = 4;
      constexpr int pack_factor = 32 / bits; // 8
      constexpr int bytes_per_pack = 4;
      constexpr int values_per_thread = pack_factor * packs_per_thread; // 16
      constexpr int block_size = values_per_thread * SIMD_SIZE;
      constexpr int scale_step_per_thread = group_size / values_per_thread;

      // Offset to this expert's weights
      uint expert_idx = expert_indices[tid.x];
      constant uint8_t* ws = (constant uint8_t*)w + expert_idx * expert_strides.x;
      constant T* sc = scales + expert_idx * expert_strides.y;
      constant T* bi = biases + expert_idx * expert_strides.z;

      typedef float U;

      thread U x_thread[values_per_thread];
      thread U result[results_per_simdgroup] = {0};

      const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
      const int in_vec_size_g = (in_vec_size + group_size - 1) / group_size;
      const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
          simd_gid * results_per_simdgroup;
      const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

      if (out_row >= out_vec_size) {
        return;
      }

      // Small N path: fewer than 1 tile of output rows
      if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
        ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        sc += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        bi += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += tid.x * in_vec_size + simd_lid * values_per_thread;
        y += tid.x * out_vec_size + out_row;

        int k = 0;
        for (; k < in_vec_size - block_size; k += block_size) {
          U sum = load_vector_4bit<T, U, values_per_thread>(x, x_thread);
          for (int row = 0; out_row + row < out_vec_size; row++) {
            auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
            constant T* sl = sc + row * in_vec_size_g;
            constant T* bl = bi + row * in_vec_size_g;
            result[row] += qdot_4bit<U, values_per_thread>(wl, x_thread, sl[0], bl[0], sum);
          }
          ws += block_size * bytes_per_pack / pack_factor;
          sc += block_size / group_size;
          bi += block_size / group_size;
          x += block_size;
        }
        const int remaining = clamp(
            static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
        if (remaining > 0) {
          U sum = load_vector_safe_4bit<T, U, values_per_thread>(x, x_thread, remaining);
          for (int row = 0; out_row + row < out_vec_size; row++) {
            auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
            constant T* sl = sc + row * in_vec_size_g;
            constant T* bl = bi + row * in_vec_size_g;
            result[row] += qdot_safe_4bit<U, values_per_thread>(wl, x_thread, sl[0], bl[0], sum, remaining);
          }
        }
        for (int row = 0; out_row + row < out_vec_size; row++) {
          result[row] = simd_sum(result[row]);
          if (simd_lid == 0) { y[row] = static_cast<T>(result[row]); }
        }
      }
      // Normal path: last tile may overlap with previous
      else {
        ws += used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        sc += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        bi += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += tid.x * in_vec_size + simd_lid * values_per_thread;
        y += tid.x * out_vec_size + used_out_row;

        int k = 0;
        for (; k < in_vec_size - block_size; k += block_size) {
          U sum = load_vector_4bit<T, U, values_per_thread>(x, x_thread);
          for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
            constant T* sl = sc + row * in_vec_size_g;
            constant T* bl = bi + row * in_vec_size_g;
            result[row] += qdot_4bit<U, values_per_thread>(wl, x_thread, sl[0], bl[0], sum);
          }
          ws += block_size * bytes_per_pack / pack_factor;
          sc += block_size / group_size;
          bi += block_size / group_size;
          x += block_size;
        }
        const int remaining = clamp(
            static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
        if (remaining > 0) {
          U sum = load_vector_safe_4bit<T, U, values_per_thread>(x, x_thread, remaining);
          for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
            constant T* sl = sc + row * in_vec_size_g;
            constant T* bl = bi + row * in_vec_size_g;
            result[row] += qdot_safe_4bit<U, values_per_thread>(wl, x_thread, sl[0], bl[0], sum, remaining);
          }
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
          result[row] = simd_sum(result[row]);
          if (simd_lid == 0) { y[row] = static_cast<T>(result[row]); }
        }
      }
    }

    #define INSTANTIATE_GATHER_QMV_IMPL(DTYPE, GSIZE)                                       \
      template [[host_name("gather_qmv_impl_4bit_" #GSIZE "_" #DTYPE)]] kernel void         \
      gather_qmv_impl<DTYPE, GSIZE>(                                                         \
          constant DTYPE * x [[buffer(0)]],                                                  \
          constant uchar * w [[buffer(1)]],                                                  \
          constant DTYPE * scales [[buffer(2)]],                                             \
          constant DTYPE * biases [[buffer(3)]],                                             \
          device DTYPE * y [[buffer(4)]],                                                    \
          constant uint3 & sizes [[buffer(5)]],                                              \
          constant uint32_t * expert_indices [[buffer(6)]],                                  \
          constant uint3 & expert_strides [[buffer(7)]],                                     \
          uint3 tid [[threadgroup_position_in_grid]],                                        \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                                  \
          uint simd_lid [[thread_index_in_simdgroup]])

    INSTANTIATE_GATHER_QMV_IMPL(float, 32);
    INSTANTIATE_GATHER_QMV_IMPL(float, 64);
    INSTANTIATE_GATHER_QMV_IMPL(float, 128);
    INSTANTIATE_GATHER_QMV_IMPL(bfloat, 32);
    INSTANTIATE_GATHER_QMV_IMPL(bfloat, 64);
    INSTANTIATE_GATHER_QMV_IMPL(bfloat, 128);

  )";
}

std::unique_ptr<ETMetalShaderLibrary> gather_qmv_shader_library = nullptr;
std::once_flag gather_qmv_shader_library_once_flag;

ETMetalShaderLibrary* get_gather_qmv_shader_library() {
  std::call_once(gather_qmv_shader_library_once_flag, []() {
    std::string source = get_gather_qmv_metal_source();
    gather_qmv_shader_library = std::make_unique<ETMetalShaderLibrary>(source);
  });
  return gather_qmv_shader_library.get();
}

} // namespace


extern "C" {

AOTITorchError aoti_torch_mps_gather_qmv(
    AOTITensorHandle X,
    AOTITensorHandle W,
    AOTITensorHandle S,
    AOTITensorHandle Z,
    AOTITensorHandle ExpertIndices,
    int64_t group_size,
    AOTITensorHandle* ret) {

  ET_LOG(Debug, "aoti_torch_mps_gather_qmv: Starting, group_size=%lld", group_size);

  if (!X || !W || !S || !Z || !ExpertIndices || !ret) {
    ET_LOG(Error, "aoti_torch_mps_gather_qmv: null required tensor handles");
    return Error::InvalidArgument;
  }

  if (group_size != 32 && group_size != 64 && group_size != 128) {
    ET_LOG(Error, "aoti_torch_mps_gather_qmv: Invalid group_size %lld (must be 32, 64, or 128)", group_size);
    return Error::InvalidArgument;
  }

  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps_gather_qmv: Failed to get current Metal stream");
    return Error::Internal;
  }

  try {
    @autoreleasepool {
      auto* x_tensor = reinterpret_cast<Tensor*>(X);        // [P, K]
      auto* w_tensor = reinterpret_cast<Tensor*>(W);        // [E, N, K/2]
      auto* s_tensor = reinterpret_cast<Tensor*>(S);        // [E, N, K/gs]
      auto* z_tensor = reinterpret_cast<Tensor*>(Z);        // [E, N, K/gs]
      auto* idx_tensor = reinterpret_cast<Tensor*>(ExpertIndices); // [P]

      // Validate dimensions
      if (x_tensor->dim() != 2) {
        ET_LOG(Error, "aoti_torch_mps_gather_qmv: x must be 2D, got %d", (int)x_tensor->dim());
        return Error::InvalidArgument;
      }
      if (w_tensor->dim() != 3) {
        ET_LOG(Error, "aoti_torch_mps_gather_qmv: w must be 3D [E, N, K_packed], got %d", (int)w_tensor->dim());
        return Error::InvalidArgument;
      }

      int32_t P = static_cast<int32_t>(x_tensor->sizes()[0]);
      int32_t K = static_cast<int32_t>(x_tensor->sizes()[1]);
      int32_t E = static_cast<int32_t>(w_tensor->sizes()[0]);
      int32_t N = static_cast<int32_t>(w_tensor->sizes()[1]);
      int32_t K_packed = static_cast<int32_t>(w_tensor->sizes()[2]);

      ET_LOG(Debug, "aoti_torch_mps_gather_qmv: P=%d, K=%d, N=%d, E=%d, gs=%lld", P, K, N, E, group_size);

      // Validate K packing: K_packed should be K/2 for 4-bit
      if (K_packed != K / 2) {
        ET_LOG(Error, "aoti_torch_mps_gather_qmv: K_packed=%d != K/2=%d", K_packed, K / 2);
        return Error::InvalidArgument;
      }

      // Determine dtype
      int32_t dtype = static_cast<int32_t>(x_tensor->scalar_type());
      size_t element_size;
      std::string type_str;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        element_size = sizeof(float);
        type_str = "float";
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        element_size = sizeof(uint16_t);
        type_str = "bfloat";
      } else {
        ET_LOG(Error, "aoti_torch_mps_gather_qmv: Unsupported dtype %d", dtype);
        return Error::InvalidArgument;
      }

      // Get shader library
      ETMetalShaderLibrary* library = get_gather_qmv_shader_library();
      if (!library) {
        ET_LOG(Error, "aoti_torch_mps_gather_qmv: Failed to get shader library");
        return Error::Internal;
      }

      // Select kernel: fast path for aligned K, impl path for generic K
      bool use_fast = (N % 8 == 0 && K % 512 == 0);
      std::string kernel_name = use_fast
          ? "gather_qmv_fast_4bit_" + std::to_string(group_size) + "_" + type_str
          : "gather_qmv_impl_4bit_" + std::to_string(group_size) + "_" + type_str;
      ET_LOG(Debug, "aoti_torch_mps_gather_qmv: Using kernel: %s", kernel_name.c_str());

      auto kernel_func = library->getKernelFunction(kernel_name);
      if (!kernel_func) {
        ET_LOG(Error, "aoti_torch_mps_gather_qmv: Failed to get kernel function: %s", kernel_name.c_str());
        return Error::Internal;
      }

      // Allocate output [P, N]
      size_t out_size_bytes = P * N * element_size;
      void* out_contents_ptr = nullptr;
      allocate_mtl_buffer(&out_contents_ptr, out_size_bytes);

      std::vector<int64_t> output_sizes = {P, N};
      std::vector<int64_t> output_strides = {N, 1};

      AOTITensorHandle out_tensor_handle = nullptr;
      AOTITorchError create_result = aoti_torch_create_tensor_from_blob_v2(
          out_contents_ptr, 2, output_sizes.data(), output_strides.data(),
          0, dtype, 13, 0, &out_tensor_handle, 0, nullptr, 0);

      if (create_result != Error::Ok || !out_tensor_handle) {
        ET_LOG(Error, "aoti_torch_mps_gather_qmv: Failed to create output tensor");
        aoti_torch_mps_free(out_contents_ptr);
        return Error::Internal;
      }

      extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
      memory_to_n_tensor[out_contents_ptr] = 1;

      auto* out_tensor = reinterpret_cast<Tensor*>(out_tensor_handle);

      // Prepare kernel arguments
      std::array<uint32_t, 4> sizes = {
          static_cast<uint32_t>(P),
          static_cast<uint32_t>(K),
          static_cast<uint32_t>(N),
          0
      };

      // Expert strides: bytes offset per expert for w, scales, biases
      int32_t K_g = K / static_cast<int32_t>(group_size);
      std::array<uint32_t, 4> expert_strides = {
          static_cast<uint32_t>(N * K_packed),      // w stride: N * K/2 bytes
          static_cast<uint32_t>(N * K_g),           // scales stride: N * K/gs elements
          static_cast<uint32_t>(N * K_g),           // biases stride: N * K/gs elements
          0
      };

      // Execute kernel
      kernel_func->runCommandBlock([&]() {
        kernel_func->startEncoding();

        kernel_func->setArg(0, *x_tensor);
        kernel_func->setArg(1, *w_tensor);
        kernel_func->setArg(2, *s_tensor);
        kernel_func->setArg(3, *z_tensor);
        kernel_func->setArg(4, *out_tensor);
        kernel_func->setArg(5, sizes.data(), sizeof(uint32_t) * sizes.size());
        kernel_func->setArg(6, *idx_tensor);
        kernel_func->setArg(7, expert_strides.data(), sizeof(uint32_t) * expert_strides.size());

        // dispatch_qmv: grid (P, (N+7)/8, 1), group (32, 2, 1)
        kernel_func->dispatchThreadgroups(
            P,                       // gridX: one per token-expert pair
            (N + 7) / 8,             // gridY: output rows
            1,                       // gridZ
            32,                      // threadsX (SIMD_SIZE)
            2,                       // threadsY (num_simdgroups)
            1);                      // threadsZ
      });

      *ret = out_tensor_handle;

      ET_LOG(Debug, "aoti_torch_mps_gather_qmv: Completed successfully");

    } // @autoreleasepool

    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps_gather_qmv exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps_gather_qmv: unknown exception");
    return Error::Internal;
  }
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
