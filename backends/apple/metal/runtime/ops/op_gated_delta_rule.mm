/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gated delta rule recurrence kernel for linear attention (Qwen 3.5 MoE).
//
// Ported from the MLX delegate PR (#18785) Metal shader. The kernel processes
// the full sequence sequentially within a single GPU dispatch, keeping
// recurrent state in per-thread registers.
//
// Recurrence per time step:
//   state *= exp(g_t)                          -- decay
//   kv_mem = sum(state * k_t, dim=-1)          -- project state by key
//   delta = beta_t * (v_t - kv_mem)            -- delta rule update
//   state += outer(k_t, delta)                 -- rank-1 state update
//   output_t = sum(state * q_t, dim=-1)        -- read from state
//
// Grid: [32, Dv, B*Hv]  Threadgroup: [32, 4, 1]
// Each simdgroup of 32 threads handles Dk/32 elements of the key dimension.

#include <executorch/backends/apple/metal/runtime/ops/common.h>

namespace executorch {
namespace backends {
namespace metal {
namespace {

static std::string get_gated_delta_rule_metal_source() {
  return R"(
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    // Gated delta rule recurrence kernel.
    // Template args: InT=data type, Dk/Dv/Hk/Hv=static dimensions.
    // From MLX delegate PR #18785 (Copyright Meta Platforms, Inc.).
    template <typename InT, int Dk, int Dv, int Hk, int Hv>
    [[kernel]] void gated_delta_step(
        const device InT* q [[buffer(0)]],          // [B, T, Hk, Dk]
        const device InT* k [[buffer(1)]],          // [B, T, Hk, Dk]
        const device InT* v [[buffer(2)]],          // [B, T, Hv, Dv]
        const device InT* g [[buffer(3)]],          // [B, T, Hv]
        const device InT* beta [[buffer(4)]],       // [B, T, Hv]
        const device InT* state_in [[buffer(5)]],   // [B, Hv, Dv, Dk]
        device InT* y [[buffer(6)]],                // [B, T, Hv, Dv]
        device InT* state_out [[buffer(7)]],        // [B, Hv, Dv, Dk]
        constant uint& T_val [[buffer(8)]],         // sequence length
        uint3 thread_position_in_grid [[thread_position_in_grid]],
        uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
        uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {

      auto n = thread_position_in_grid.z;
      auto b_idx = n / Hv;
      auto hv_idx = n % Hv;
      auto hk_idx = hv_idx / (Hv / Hk);
      constexpr int n_per_t = Dk / 32;

      int T = static_cast<int>(T_val);

      // q, k: [B, T, Hk, Dk]
      auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
      auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

      // v, y: [B, T, Hv, Dv]
      auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
      y += b_idx * T * Hv * Dv + hv_idx * Dv;

      auto dk_idx = thread_position_in_threadgroup.x;
      auto dv_idx = thread_position_in_grid.y;

      // state_in, state_out: [B, Hv, Dv, Dk]
      auto i_state = state_in + (n * Dv + dv_idx) * Dk;
      auto o_state = state_out + (n * Dv + dv_idx) * Dk;

      float state[n_per_t];
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = static_cast<float>(i_state[s_idx]);
      }

      // g, beta: [B, T, Hv]
      auto g_ = g + b_idx * T * Hv;
      auto beta_ = beta + b_idx * T * Hv;

      for (int t = 0; t < T; ++t) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * static_cast<float>(g_[hv_idx]);
          kv_mem += state[i] * static_cast<float>(k_[s_idx]);
        }
        kv_mem = simd_sum(kv_mem);

        auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * static_cast<float>(beta_[hv_idx]);

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
          out += state[i] * static_cast<float>(q_[s_idx]);
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
          y[dv_idx] = static_cast<InT>(out);
        }
        // Advance to next time step
        q_ += Hk * Dk;
        k_ += Hk * Dk;
        v_ += Hv * Dv;
        y += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
      }
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        o_state[s_idx] = static_cast<InT>(state[i]);
      }
    }

    // Instantiate for Qwen 3.5 MoE dimensions: Dk=128, Dv=128, Hk=16, Hv=32
    #define INSTANTIATE_GDR(DTYPE, Dk, Dv, Hk, Hv)                            \
      template [[host_name("gated_delta_step_" #DTYPE                          \
                           "_dk" #Dk "_dv" #Dv "_hk" #Hk "_hv" #Hv)]]         \
      [[kernel]] void gated_delta_step<DTYPE, Dk, Dv, Hk, Hv>(                \
          const device DTYPE* q [[buffer(0)]],                                 \
          const device DTYPE* k [[buffer(1)]],                                 \
          const device DTYPE* v [[buffer(2)]],                                 \
          const device DTYPE* g [[buffer(3)]],                                 \
          const device DTYPE* beta [[buffer(4)]],                              \
          const device DTYPE* state_in [[buffer(5)]],                          \
          device DTYPE* y [[buffer(6)]],                                       \
          device DTYPE* state_out [[buffer(7)]],                               \
          constant uint& T_val [[buffer(8)]],                                  \
          uint3 thread_position_in_grid [[thread_position_in_grid]],           \
          uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
          uint thread_index_in_simdgroup [[thread_index_in_simdgroup]])

    // Qwen 3.5 MoE real model dimensions (Hk=16 after repeat_interleave → 32)
    INSTANTIATE_GDR(float, 128, 128, 32, 32);
    INSTANTIATE_GDR(bfloat, 128, 128, 32, 32);
    // Tiny test model dimensions (Hk=2 after repeat_interleave → 4)
    INSTANTIATE_GDR(float, 64, 64, 4, 4);
    INSTANTIATE_GDR(bfloat, 64, 64, 4, 4);

  )";
}

std::unique_ptr<ETMetalShaderLibrary> gdr_shader_library = nullptr;
std::once_flag gdr_shader_library_once_flag;

ETMetalShaderLibrary* get_gdr_shader_library() {
  std::call_once(gdr_shader_library_once_flag, []() {
    std::string source = get_gated_delta_rule_metal_source();
    gdr_shader_library = std::make_unique<ETMetalShaderLibrary>(source);
  });
  return gdr_shader_library.get();
}

} // namespace


extern "C" {

AOTITorchError aoti_torch_mps_gated_delta_rule(
    AOTITensorHandle Q,
    AOTITensorHandle K,
    AOTITensorHandle V,
    AOTITensorHandle G,
    AOTITensorHandle Beta,
    AOTITensorHandle StateIn,
    AOTITensorHandle* retY) {

  ET_LOG(Debug, "aoti_torch_mps_gated_delta_rule: Starting");

  if (!Q || !K || !V || !G || !Beta || !StateIn || !retY) {
    ET_LOG(Error, "aoti_torch_mps_gated_delta_rule: null required tensor handles");
    return Error::InvalidArgument;
  }

  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps_gated_delta_rule: Failed to get Metal stream");
    return Error::Internal;
  }

  try {
    @autoreleasepool {
      auto* q_tensor = reinterpret_cast<Tensor*>(Q);          // [B, T, Hk, Dk]
      auto* k_tensor = reinterpret_cast<Tensor*>(K);          // [B, T, Hk, Dk]
      auto* v_tensor = reinterpret_cast<Tensor*>(V);          // [B, T, Hv, Dv]
      auto* g_tensor = reinterpret_cast<Tensor*>(G);          // [B, T, Hv]
      auto* beta_tensor = reinterpret_cast<Tensor*>(Beta);    // [B, T, Hv]
      auto* state_tensor = reinterpret_cast<Tensor*>(StateIn); // [B, Hv, Dv, Dk]

      if (q_tensor->dim() != 4 || v_tensor->dim() != 4 || state_tensor->dim() != 4) {
        ET_LOG(Error, "aoti_torch_mps_gated_delta_rule: q/v must be 4D, state must be 4D");
        return Error::InvalidArgument;
      }

      int32_t B = static_cast<int32_t>(q_tensor->sizes()[0]);
      int32_t T = static_cast<int32_t>(q_tensor->sizes()[1]);
      int32_t Hk = static_cast<int32_t>(q_tensor->sizes()[2]);
      int32_t Dk = static_cast<int32_t>(q_tensor->sizes()[3]);
      int32_t Hv = static_cast<int32_t>(v_tensor->sizes()[2]);
      int32_t Dv = static_cast<int32_t>(v_tensor->sizes()[3]);

      ET_LOG(Debug, "aoti_torch_mps_gated_delta_rule: B=%d, T=%d, Hk=%d, Dk=%d, Hv=%d, Dv=%d",
             B, T, Hk, Dk, Hv, Dv);

      if (Dk % 32 != 0) {
        ET_LOG(Error, "aoti_torch_mps_gated_delta_rule: Dk=%d must be multiple of 32", Dk);
        return Error::InvalidArgument;
      }

      // Determine dtype
      int32_t dtype = static_cast<int32_t>(q_tensor->scalar_type());
      size_t element_size;
      std::string type_str;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        element_size = sizeof(float);
        type_str = "float";
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        element_size = sizeof(uint16_t);
        type_str = "bfloat";
      } else {
        ET_LOG(Error, "aoti_torch_mps_gated_delta_rule: Unsupported dtype %d", dtype);
        return Error::InvalidArgument;
      }

      ETMetalShaderLibrary* library = get_gdr_shader_library();
      if (!library) {
        ET_LOG(Error, "aoti_torch_mps_gated_delta_rule: Failed to get shader library");
        return Error::Internal;
      }

      std::string kernel_name = "gated_delta_step_" + type_str +
          "_dk" + std::to_string(Dk) + "_dv" + std::to_string(Dv) +
          "_hk" + std::to_string(Hk) + "_hv" + std::to_string(Hv);
      ET_LOG(Debug, "aoti_torch_mps_gated_delta_rule: Using kernel: %s", kernel_name.c_str());

      auto kernel_func = library->getKernelFunction(kernel_name);
      if (!kernel_func) {
        ET_LOG(Error, "aoti_torch_mps_gated_delta_rule: Kernel not found: %s", kernel_name.c_str());
        return Error::Internal;
      }

      // Allocate output y [B, T, Hv, Dv]
      size_t y_bytes = B * T * Hv * Dv * element_size;
      void* y_ptr = nullptr;
      allocate_mtl_buffer(&y_ptr, y_bytes);

      std::vector<int64_t> y_sizes = {B, T, Hv, Dv};
      std::vector<int64_t> y_strides = {T * Hv * Dv, Hv * Dv, Dv, 1};

      AOTITensorHandle y_handle = nullptr;
      aoti_torch_create_tensor_from_blob_v2(
          y_ptr, 4, y_sizes.data(), y_strides.data(),
          0, dtype, 13, 0, &y_handle, 0, nullptr, 0);

      if (!y_handle) {
        aoti_torch_mps_free(y_ptr);
        return Error::Internal;
      }
      extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
      memory_to_n_tensor[y_ptr] = 1;

      auto* y_tensor = reinterpret_cast<Tensor*>(y_handle);

      // State is mutated in-place: kernel writes to state_tensor directly
      // (state_out = state_in in the kernel args)
      uint T_uint = static_cast<uint>(T);

      // Execute kernel
      kernel_func->runCommandBlock([&]() {
        kernel_func->startEncoding();

        kernel_func->setArg(0, *q_tensor);
        kernel_func->setArg(1, *k_tensor);
        kernel_func->setArg(2, *v_tensor);
        kernel_func->setArg(3, *g_tensor);
        kernel_func->setArg(4, *beta_tensor);
        kernel_func->setArg(5, *state_tensor);    // state_in
        kernel_func->setArg(6, *y_tensor);
        kernel_func->setArg(7, *state_tensor);    // state_out = state_in (in-place)
        kernel_func->setArg(8, T_uint);

        // Grid: [32, Dv, B*Hv]  Threadgroup: [32, 4, 1]
        // Grid: [32, Dv, B*Hv] total threads, threadgroup: [32, 4, 1]
        // dispatchThreadgroups takes threadgroup counts, not thread counts
        kernel_func->dispatchThreadgroups(
            1,                       // gridX: 1 group × 32 threads = 32 threads
            (Dv + 3) / 4,           // gridY: ceil(Dv/4) groups × 4 threads = Dv threads
            B * Hv,                  // gridZ: B*Hv groups × 1 thread = B*Hv threads
            32,                      // threadsPerGroupX
            4,                       // threadsPerGroupY
            1);                      // threadsPerGroupZ
      });

      *retY = y_handle;

      ET_LOG(Debug, "aoti_torch_mps_gated_delta_rule: Completed successfully");

    } // @autoreleasepool

    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps_gated_delta_rule exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps_gated_delta_rule: unknown exception");
    return Error::Internal;
  }
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
