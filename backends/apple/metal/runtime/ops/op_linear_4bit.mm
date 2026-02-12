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

// Helper function to get the Metal shader source for Int4MM
static std::string get_int4_metal_source() {
  return R"(
  /**
  * common.metal
  */

    // Copyright (c) Meta Platforms, Inc. and affiliates.
    // All rights reserved.
    //
    // This source code is licensed under the BSD 3-Clause license found in the
    // LICENSE file in the root directory of this source tree.

    template <typename T> struct Vec4Type {};

    template <> struct Vec4Type<float> {
      using type = float4;
    };

    template <> struct Vec4Type<half> {
      using type = half4;
    };

    #if __METAL_VERSION__ >= 310
    template <> struct Vec4Type<bfloat> {
      using type = bfloat4;
    };
    #endif

  /**
  * int4mm_opt.metal
  */

    // Copyright (c) Meta Platforms, Inc. and affiliates.
    // All rights reserved.
    //
    // This source code is licensed under the BSD 3-Clause license found in the
    // LICENSE file in the root directory of this source tree.
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    /*
      This code takes heavy inspiration from MLX:
      https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h
      Specifically:
        - Multiplying activation by inverse scaling factor to reduce compute
      boundedness
        - Handling zero point by accumulating act in separate sum term. Needed with
      optimization done above. MLX MIT License:
      https://github.com/ml-explore/mlx/blob/main/LICENSE
    */

    /*
      A matrix is [M x K] (right now this kernel does not support M > 1 but this is
      a very easy fix that will follow right after) B matrix is [N x K]. For 4 bit
      2 of the k values are packed in one byte so you can think of B as [N x K/2]
      matrix from layout perspective.

      Since this kernel is optimizing for gemv case, we split work, along reduction
      dim k, among the threads of same simdgroup. Ex: if K = 4096 and simdgroup
      size is 32 (current algorithm should work as long as simdgroup size is > 32).
      Then each thread will accumulate 4096/32 = 128 k values. However these 128
      values, handled by each thread are not laid out contiguously. Each thread
      handles 4 contiguous k values and then jumps 128 elements, k_jump =
      thread_per_channel (32) * ks_per_thread (4). Take a simpler example where
      simdgroup is of size 4. In this case threads_per_channel = 4. Assume K = 32
          k                thread
      [0, 1, 2, 3,          0
        4, 5, 6, 7,          1
        8, 9, 10, 11,        2
        12, 13, 14, 15,      3
        16, 17, 18, 19,      0
        20, 21, 22, 23,      1
        24, 25, 26, 27,      2
        28, 29, 30, 31]      3
      thread id in simd group that handle corresponding
      ks
      Thread 0 here is handling (0, 1, 2, 3) and then (16, 17, 18, 19). They are
      apart by k_jump = 4 * 4 = 16 This is done to improve memory access locality
      amonng threads that are working co-operatively. Once each thread has their
      partial sums accumulated, we use tree reduction (Metal offers simd_sum but
      not used so that we support simdgroup size = 64). In the
      example above we will have 4 partial sums.

      Each thread also handles 4 different output rows. Thus each simdgroup will be
      responsible for (1x4) tile of the output. We haven't evaluated whether a
      different tile size is better or not. We probably will do some auto-tuning
      once initial work is done.
    */

    /*
      @brief This shader implements 4-bit matrix-vector multiplication where A
      matrix is fp16, bfloat or float and B matrix is a 4-bit groupwise-quantized weight
      matrix.
      @param [in] A is activation matrix of size M x K.
      @param [in] B is weight matrix of size M x K. Each byte contains 2 4-bit
      values, along K dim, packed together.
      @param [in] scales_ptr is scales ptr corresponding each
      output channel x groups. These are packed as [N, num_groups = ceil(K / group_size)]. N = output
      channels.
      @param [in] zeros_ptr is zero points corresponding each
      output channel x groups. These are packed as [N, num_groups = ceil(K / group_size)]. N = output
      channels.
      @param [out] output_data is output matrix of size M x N.
      @param [in] sizes array contains values of M, K and N.
      @param [in] thread_index is global thread id.
      @param [in] tid_in_simdgruop is thread id in simdgroup. e.g. in simdgroup of size 32 it can be in [0-31].
    */
    template <typename T, unsigned group_size>
    kernel void int4pack_mm(constant T *A [[buffer(0)]],
                            constant uchar *B [[buffer(1)]],
                            constant T *scales_ptr [[buffer(2)]],
                            constant T *zeros_ptr [[buffer(3)]],
                            device T *output_data [[buffer(4)]],
                            constant uint3 &sizes [[buffer(5)]], // M, K, N
                            uint3 thread_index [[thread_position_in_grid]],
                            uint tid_in_simdgroup [[thread_index_in_simdgroup]]) {
      constexpr uint threads_per_channel = 32;
      constexpr uint ks_per_thread = 4;
      constexpr uint k_pack_factor = 2;
      const uint K = sizes.y;
      const uint N = sizes.z;
      const uint num_groups = (K + group_size - 1) / group_size;
      uint n = thread_index.x; // 0..N/4-1
      uint m = thread_index.z; // 0..M
      n = n / threads_per_channel;
      n = n * 4;
      // This is starting k for each thread. In the example above, for thread 1 this
      // value will be 4.
      uint k = (tid_in_simdgroup % threads_per_channel) * ks_per_thread;
      constexpr int k_jump = threads_per_channel * ks_per_thread;

      using vecT = typename Vec4Type<T>::type;
      constant vecT *A_ptr = reinterpret_cast<constant vecT *>(A + m * K);
      constant uchar *B_ptr = B + ((n * K) / k_pack_factor);

      thread float4 result = float4(0.0);
      // We multipy group of 4 channels with these scales.
      // Because corresponding values from weight matrix are effectively left
      // shifted. This is to avoid doing right shift on those values which ends up
      // affecting performance. This is the trick applied in MLX kernels.
      float4 act_div_scales = {1.f, 1 / 16.f, 1 / 256.f, 1 / 4096.f};

      for (; k < K; k += k_jump) {
        // Find specific group to which channels handled by this thread
        // belong.
        uint k_block_index = k / group_size;
        uint scales_group_offset = (n * num_groups + k_block_index);

        vecT scales =
            vecT(scales_ptr[scales_group_offset],
                scales_ptr[scales_group_offset + num_groups],
                scales_ptr[scales_group_offset + 2 * num_groups],
                scales_ptr[scales_group_offset + 3 * num_groups]);
        // Adding zero point results in 10% perf penalty.
        vecT zeros =
            vecT(zeros_ptr[scales_group_offset],
                zeros_ptr[scales_group_offset + num_groups],
                zeros_ptr[scales_group_offset + 2 * num_groups],
                zeros_ptr[scales_group_offset + 3 * num_groups]);
        float4 zeros_float = float4(zeros);

        float4 a_val = float4(A_ptr[k / 4]);
        // We are gonna skip right-shifts of the weights and hence divide by corresponding factor.
        float4 a_vec = a_val * act_div_scales;
        float a_val_sum = a_val[0] + a_val[1] + a_val[2] + a_val[3];

        float4x4 b_mat;
        ushort b_val0 = (reinterpret_cast<constant ushort *>(
            B_ptr + (k + 0 * K) / k_pack_factor))[0];
        ushort b_val1 = (reinterpret_cast<constant ushort *>(
            B_ptr + (k + 1 * K) / k_pack_factor))[0];
        ushort b_val2 = (reinterpret_cast<constant ushort *>(
            B_ptr + (k + 2 * K) / k_pack_factor))[0];
        ushort b_val3 = (reinterpret_cast<constant ushort *>(
            B_ptr + (k + 3 * K) / k_pack_factor))[0];
        b_mat[0] = scales[0] * float4(float(b_val0 & 0x000f), float(b_val0 & 0x00f0),
                                  float(b_val0 & 0x0f00), float(b_val0 & 0xf000));
        b_mat[1] = scales[1] * float4(float(b_val1 & 0x000f), float(b_val1 & 0x00f0),
                                  float(b_val1 & 0x0f00), float(b_val1 & 0xf000));
        b_mat[2] = scales[2] * float4(float(b_val2 & 0x000f), float(b_val2 & 0x00f0),
                                  float(b_val2 & 0x0f00), float(b_val2 & 0xf000));
        b_mat[3] = scales[3] * float4(float(b_val3 & 0x000f), float(b_val3 & 0x00f0),
                                  float(b_val3 & 0x0f00), float(b_val3 & 0xf000));

        result += a_vec * b_mat;
        result += a_val_sum * zeros_float;
      }
      result += simd_shuffle_down(result, 1);
      result += simd_shuffle_down(result, 2);
      result += simd_shuffle_down(result, 4);
      result += simd_shuffle_down(result, 8);
      result += simd_shuffle_down(result, 16);
      if (tid_in_simdgroup % threads_per_channel == 0) {
        reinterpret_cast<device vecT *>(output_data + m * N)[n / 4] = vecT(result);
      }
    }

    #define INSTANTIATE_INT4MM(DTYPE, GSIZE)                                       \
      template [[host_name("int4pack_mm_" #GSIZE "_" #DTYPE)]] kernel void         \
      int4pack_mm<DTYPE, GSIZE>(                                                   \
          constant DTYPE * A [[buffer(0)]], constant uchar * B [[buffer(1)]],      \
          constant DTYPE * scales_ptr [[buffer(2)]],                               \
          constant DTYPE * zeros_ptr [[buffer(3)]],                                \
          device DTYPE * output_data [[buffer(4)]],                                \
          constant uint3 & sizes [[buffer(5)]],                                    \
          uint3 thread_index [[thread_position_in_grid]],                          \
          uint tid_in_simdgroup [[thread_index_in_simdgroup]])

    INSTANTIATE_INT4MM(float, 32);
    INSTANTIATE_INT4MM(half, 32);
    INSTANTIATE_INT4MM(float, 64);
    INSTANTIATE_INT4MM(half, 64);
    INSTANTIATE_INT4MM(float, 128);
    INSTANTIATE_INT4MM(half, 128);
    INSTANTIATE_INT4MM(float, 256);
    INSTANTIATE_INT4MM(half, 256);
    #if __METAL_VERSION__ >= 310
    INSTANTIATE_INT4MM(bfloat, 32);
    INSTANTIATE_INT4MM(bfloat, 64);
    INSTANTIATE_INT4MM(bfloat, 128);
    INSTANTIATE_INT4MM(bfloat, 256);
    #endif

  /**
  * qmv_fast.metal
  */

    // Copyright (c) Meta Platforms, Inc. and affiliates.
    // All rights reserved.
    //
    // This source code is licensed under the BSD 3-Clause license found in the
    // LICENSE file in the root directory of this source tree.

    /*
      This code was taken from MLX, and modified to add support for 1, 5 & 7 bit packing.
      The original code is Copyright © 2023-2024 Apple Inc.
      https://github.com/ml-explore/mlx/blob/481349495b8c3d094eb699e678077bbe1406392d/mlx/backend/metal/kernels/quantized.h#L1
      MLX MIT License: https://github.com/ml-explore/mlx/blob/main/LICENSE
    */

    #include <metal_simdgroup>
    #include <metal_stdlib>

    static constant constexpr const int SIMD_SIZE = 32;

    template <typename T, typename U, int values_per_thread, int bits>
    inline U load_vector(constant T* x, thread U* x_thread) {
      static_assert(
          1 <= bits && bits <= 7,
          "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 7}");

      U sum = 0;

      if (bits == 1) {
        for (int i = 0; i < values_per_thread; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 2.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 8.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 32.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 128.0f;
        }
      }

      else if (bits == 2) {
        for (int i = 0; i < values_per_thread; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 4.0f;
          x_thread[i + 2] = x[i + 2] / 16.0f;
          x_thread[i + 3] = x[i + 3] / 64.0f;
        }
      }

      else if (bits == 3) {
        for (int i = 0; i < values_per_thread; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 8.0f;
          x_thread[i + 2] = x[i + 2] / 64.0f;
          x_thread[i + 3] = x[i + 3] / 2.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 128.0f;
          x_thread[i + 6] = x[i + 6] / 4.0f;
          x_thread[i + 7] = x[i + 7] / 32.0f;
        }
      }

      else if (bits == 4) {
        for (int i = 0; i < values_per_thread; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 16.0f;
          x_thread[i + 2] = x[i + 2] / 256.0f;
          x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
      }

      else if (bits == 5) {
        for (int i = 0; i < values_per_thread; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 32.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 128.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 2.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 8.0f;
        }
      }

      else if (bits == 6) {
        for (int i = 0; i < values_per_thread; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 64.0f;
          x_thread[i + 2] = x[i + 2] / 16.0f;
          x_thread[i + 3] = x[i + 3] / 4.0f;
        }
      }

      else if (bits == 7) {
        for (int i = 0; i < values_per_thread; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 128.0f;
          x_thread[i + 2] = x[i + 2] / 64.0f;
          x_thread[i + 3] = x[i + 3] / 32.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 8.0f;
          x_thread[i + 6] = x[i + 6] / 4.0f;
          x_thread[i + 7] = x[i + 7] / 2.0f;
        }
      }

      return sum;
    }

    template <typename T, typename U, int values_per_thread, int bits>
    inline U load_vector_safe(constant T* x, thread U* x_thread, int N) {
      static_assert(
          1 <= bits && bits <= 7,
          "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 7}");

      U sum = 0;

      if (bits == 1) {
        for (int i = 0; i < N; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 2.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 8.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 32.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 128.0f;
        }
      }

      else if (bits == 2) {
        for (int i = 0; i < N; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 4.0f;
          x_thread[i + 2] = x[i + 2] / 16.0f;
          x_thread[i + 3] = x[i + 3] / 64.0f;
        }
      }

      else if (bits == 3) {
        for (int i = 0; i < N; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];

          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 8.0f;
          x_thread[i + 2] = x[i + 2] / 64.0f;
          x_thread[i + 3] = x[i + 3] / 2.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 128.0f;
          x_thread[i + 6] = x[i + 6] / 4.0f;
          x_thread[i + 7] = x[i + 7] / 32.0f;
        }
      }

      else if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 16.0f;
          x_thread[i + 2] = x[i + 2] / 256.0f;
          x_thread[i + 3] = x[i + 3] / 4096.0f;
        }
      }

      else if (bits == 5) {
        for (int i = 0; i < N; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 32.0f;
          x_thread[i + 2] = x[i + 2] / 4.0f;
          x_thread[i + 3] = x[i + 3] / 128.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 2.0f;
          x_thread[i + 6] = x[i + 6] / 64.0f;
          x_thread[i + 7] = x[i + 7] / 8.0f;
        }
      }

      else if (bits == 6) {
        for (int i = 0; i < N; i += 4) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 64.0f;
          x_thread[i + 2] = x[i + 2] / 16.0f;
          x_thread[i + 3] = x[i + 3] / 4.0f;
        }
      }

      else if (bits == 7) {
        for (int i = 0; i < N; i += 8) {
          sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
              x[i + 6] + x[i + 7];
          x_thread[i] = x[i];
          x_thread[i + 1] = x[i + 1] / 128.0f;
          x_thread[i + 2] = x[i + 2] / 64.0f;
          x_thread[i + 3] = x[i + 3] / 32.0f;
          x_thread[i + 4] = x[i + 4] / 16.0f;
          x_thread[i + 5] = x[i + 5] / 8.0f;
          x_thread[i + 6] = x[i + 6] / 4.0f;
          x_thread[i + 7] = x[i + 7] / 2.0f;
        }
      }

      for (int i = N; i < values_per_thread; i++) {
        x_thread[i] = 0;
      }

      return sum;
    }

    template <typename U, int values_per_thread, int bits>
    inline U qdot(
        constant uint8_t* w,
        const thread U* x_thread,
        U scale,
        U bias,
        U sum) {
      static_assert(
          1 <= bits && bits <= 7,
          "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 7}");

      U accum = 0;

      if (bits == 1) {
        for (int i = 0; i < (values_per_thread / 8); i++) {
          x_thread += 8 * i;

          accum +=
              (x_thread[0] * (w[i] & 0x01) +
              x_thread[1] * (w[i] & 0x02) +
              x_thread[2] * (w[i] & 0x04) +
              x_thread[3] * (w[i] & 0x08) +
              x_thread[4] * (w[i] & 0x10) +
              x_thread[5] * (w[i] & 0x20) +
              x_thread[6] * (w[i] & 0x40) +
              x_thread[7] * (w[i] & 0x80));
        }
      }

      else if (bits == 2) {
        for (int i = 0; i < (values_per_thread / 4); i++) {
          accum +=
              (x_thread[4 * i] * (w[i] & 0x03) +
              x_thread[4 * i + 1] * (w[i] & 0x0c) +
              x_thread[4 * i + 2] * (w[i] & 0x30) +
              x_thread[4 * i + 3] * (w[i] & 0xc0));
        }
      }

      else if (bits == 3) {
        for (int i = 0; i < (values_per_thread / 8); i++) {
          x_thread += 8 * i;
          w += 3 * i;

          accum += (w[0] & 0x07) * x_thread[0];
          accum += (w[0] & 0x38) * x_thread[1];
          accum += (w[0] & 0xc0) * x_thread[2];
          accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

          accum += (w[1] & 0x0e) * x_thread[3];
          accum += (w[1] & 0x70) * x_thread[4];
          accum += (w[1] & 0x80) * x_thread[5];
          accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

          accum += (w[2] & 0x1c) * x_thread[6];
          accum += (w[2] & 0xe0) * x_thread[7];
        }
      }

      else if (bits == 4) {
        constant uint16_t* ws = (constant uint16_t*)w;
        for (int i = 0; i < (values_per_thread / 4); i++) {
          accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
              x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
              x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
              x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }
      }

      else if (bits == 5) {
        for (int i = 0; i < (values_per_thread / 8); i++) {
          x_thread += 8 * i;
          w += 5 * i;

          accum += (w[0] & 0x1f) * x_thread[0];
          accum += (w[0] & 0xe0) * x_thread[1];

          accum += (w[1] & 0x03) * (x_thread[1] * 256.0f);
          accum += (w[1] & 0x7c) * x_thread[2];
          accum += (w[1] & 0x80) * x_thread[3];

          accum += (w[2] & 0x0f) * (x_thread[3] * 256.0f);
          accum += (w[2] & 0xf0) * x_thread[4];

          accum += (w[3] & 0x01) * (x_thread[4] * 256.0f);
          accum += (w[3] & 0x3e) * x_thread[5];
          accum += (w[3] & 0xc0) * x_thread[6];

          accum += (w[4] & 0x07) * (x_thread[6] * 256.0f);
          accum += (w[4] & 0xf8) * x_thread[7];
        }
      }

      else if (bits == 6) {
        for (int i = 0; i < (values_per_thread / 4); i++) {
          x_thread += 4 * i;
          w += 3 * i;

          accum += (w[0] & 0x3f) * x_thread[0];

          accum += (w[0] & 0xc0) * x_thread[1];
          accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

          accum += (w[1] & 0xf0) * x_thread[2];
          accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

          accum += (w[2] & 0xfc) * x_thread[3];
        }
      }

      else if (bits == 7) {
        for (int i = 0; i < (values_per_thread / 8); i++) {
          x_thread += 8 * i;
          w += 7 * i;

          accum += (w[0] & 0x7f) * x_thread[0];
          accum += (w[0] & 0x80) * x_thread[1];

          accum += (w[1] & 0x3f) * (x_thread[1] * 256.0f);
          accum += (w[1] & 0xc0) * x_thread[2];

          accum += (w[2] & 0x1f) * (x_thread[2] * 256.0f);
          accum += (w[2] & 0xe0) * x_thread[3];

          accum += (w[3] & 0x0f) * (x_thread[3] * 256.0f);
          accum += (w[3] & 0xf0) * x_thread[4];

          accum += (w[4] & 0x07) * (x_thread[4] * 256.0f);
          accum += (w[4] & 0xf8) * x_thread[5];

          accum += (w[5] & 0x03) * (x_thread[5] * 256.0f);
          accum += (w[5] & 0xfc) * x_thread[6];

          accum += (w[6] & 0x01) * (x_thread[6] * 256.0f);
          accum += (w[6] & 0xfe) * x_thread[7];
        }
      }

      return scale * accum + sum * bias;
    }

    template <typename U, int values_per_thread, int bits>
    inline U qdot_safe(
        constant uint8_t* w,
        const thread U* x_thread,
        U scale,
        U bias,
        U sum,
        int N) {
      static_assert(
          1 <= bits && bits <= 7,
          "Template undefined for bits not in {1, 2, 3, 4, 5, 6, 7}");

      U accum = 0;

      if (bits == 1) {
        for (int i = 0; i < (N / 8); i++) {
          x_thread += 8 * i;

          accum +=
              (x_thread[0] * (w[i] & 0x01) +
              x_thread[1] * (w[i] & 0x02) +
              x_thread[2] * (w[i] & 0x04) +
              x_thread[3] * (w[i] & 0x08) +
              x_thread[4] * (w[i] & 0x10) +
              x_thread[5] * (w[i] & 0x20) +
              x_thread[6] * (w[i] & 0x40) +
              x_thread[7] * (w[i] & 0x80));
        }
      }

      else if (bits == 2) {
        for (int i = 0; i < (N / 4); i++) {
          accum +=
              (x_thread[4 * i] * (w[i] & 0x03) +
              x_thread[4 * i + 1] * (w[i] & 0x0c) +
              x_thread[4 * i + 2] * (w[i] & 0x30) +
              x_thread[4 * i + 3] * (w[i] & 0xc0));
        }
      }

      else if (bits == 3) {
        for (int i = 0; i < (N / 8); i++) {
          x_thread += 8 * i;
          w += 3 * i;

          accum += (w[0] & 0x07) * x_thread[0];
          accum += (w[0] & 0x38) * x_thread[1];
          accum += (w[0] & 0xc0) * x_thread[2];
          accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

          accum += (w[1] & 0x0e) * x_thread[3];
          accum += (w[1] & 0x70) * x_thread[4];
          accum += (w[1] & 0x80) * x_thread[5];
          accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

          accum += (w[2] & 0x1c) * x_thread[6];
          accum += (w[2] & 0xe0) * x_thread[7];
        }
      }

      else if (bits == 4) {
        constant uint16_t* ws = (constant uint16_t*)w;
        for (int i = 0; i < (N / 4); i++) {
          accum +=
              (x_thread[4 * i] * (ws[i] & 0x000f) +
              x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
              x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
              x_thread[4 * i + 3] * (ws[i] & 0xf000));
        }
      }

      else if (bits == 5) {
        for (int i = 0; i < (N / 8); i++) {
          x_thread += 8 * i;
          w += 5 * i;

          accum += (w[0] & 0x1f) * x_thread[0];
          accum += (w[0] & 0xe0) * x_thread[1];

          accum += (w[1] & 0x03) * (x_thread[1] * 256.0f);
          accum += (w[1] & 0x7c) * x_thread[2];
          accum += (w[1] & 0x80) * x_thread[3];

          accum += (w[2] & 0x0f) * (x_thread[3] * 256.0f);
          accum += (w[2] & 0xf0) * x_thread[4];

          accum += (w[3] & 0x01) * (x_thread[4] * 256.0f);
          accum += (w[3] & 0x3e) * x_thread[5];
          accum += (w[3] & 0xc0) * x_thread[6];

          accum += (w[4] & 0x07) * (x_thread[6] * 256.0f);
          accum += (w[4] & 0xf8) * x_thread[7];
        }
      }

      else if (bits == 6) {
        for (int i = 0; i < (N / 4); i++) {
          x_thread += 4 * i;
          w += 3 * i;

          accum += (w[0] & 0x3f) * x_thread[0];

          accum += (w[0] & 0xc0) * x_thread[1];
          accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

          accum += (w[1] & 0xf0) * x_thread[2];
          accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

          accum += (w[2] & 0xfc) * x_thread[3];
        }
      }

      else if (bits == 7) {
        for (int i = 0; i < (N / 8); i++) {
          x_thread += 8 * i;
          w += 7 * i;

          accum += (w[0] & 0x7f) * x_thread[0];
          accum += (w[0] & 0x80) * x_thread[1];

          accum += (w[1] & 0x3f) * (x_thread[1] * 256.0f);
          accum += (w[1] & 0xc0) * x_thread[2];

          accum += (w[2] & 0x1f) * (x_thread[2] * 256.0f);
          accum += (w[2] & 0xe0) * x_thread[3];

          accum += (w[3] & 0x0f) * (x_thread[3] * 256.0f);
          accum += (w[3] & 0xf0) * x_thread[4];

          accum += (w[4] & 0x07) * (x_thread[4] * 256.0f);
          accum += (w[4] & 0xf8) * x_thread[5];

          accum += (w[5] & 0x03) * (x_thread[5] * 256.0f);
          accum += (w[5] & 0xfc) * x_thread[6];

          accum += (w[6] & 0x01) * (x_thread[6] * 256.0f);
          accum += (w[6] & 0xfe) * x_thread[7];
        }
      }

      return scale * accum + sum * bias;
    }

    template <typename T, int group_size, int bits>
    [[kernel]] void qmv_fast(
        constant T* x [[buffer(0)]],
        constant uchar* w [[buffer(1)]],
        constant T* scales [[buffer(2)]],
        constant T* biases [[buffer(3)]],
        device T* y [[buffer(4)]],
        constant uint3 &sizes [[buffer(5)]], // M, K, N
        uint3 tid [[threadgroup_position_in_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]) {
      const int in_vec_size = static_cast<int>(sizes.y); // K
      const int out_vec_size = static_cast<int>(sizes.z); // N

      constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
      constexpr int packs_per_thread = (bits == 1 || bits == 2) ? 1 : 2;
      constexpr int num_simdgroups = 2;
      constexpr int results_per_simdgroup = 4;
      constexpr int pack_factor = bits == 1 ? 16 : power_of_2_bits ? 32 / bits : bits == 6 ? 4 : 8;
      constexpr int bytes_per_pack = bits == 1 ? 2 : power_of_2_bits ? 4 : bits == 6 ? 3 : bits;
      constexpr int values_per_thread = pack_factor * packs_per_thread;
      constexpr int block_size = values_per_thread * SIMD_SIZE;
      constexpr int scale_step_per_thread = group_size / values_per_thread;

      constant uint8_t* ws = (constant uint8_t*)w;

      typedef float U;

      thread U x_thread[values_per_thread];
      thread U result[results_per_simdgroup] = {0};

      // Adjust positions
      const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
      const int in_vec_size_g = in_vec_size / group_size;
      const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
          simd_gid * results_per_simdgroup;

      ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
      scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
      biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
      x += tid.x * in_vec_size + simd_lid * values_per_thread;
      y += tid.x * out_vec_size + out_row;

      for (int k = 0; k < in_vec_size; k += block_size) {
        U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

        for (int row = 0; row < results_per_simdgroup; row++) {
          auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
          constant T* sl = scales + row * in_vec_size_g;
          constant T* bl = biases + row * in_vec_size_g;

          U s = sl[0];
          U b = bl[0];
          result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
        }

        ws += block_size * bytes_per_pack / pack_factor;
        scales += block_size / group_size;
        biases += block_size / group_size;
        x += block_size;
      }

      for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) {
          y[row] = static_cast<T>(result[row]);
        }
      }
    }

    #define INSTANTIATE_QMV_FAST(DTYPE, GSIZE, NBIT)                                 \
      template [[host_name("qmv_fast_" #NBIT "bit_" #GSIZE "_" #DTYPE)]] kernel void \
      qmv_fast<DTYPE, GSIZE, NBIT>(                                                  \
          constant DTYPE * A [[buffer(0)]],                                          \
          constant uchar * B [[buffer(1)]],                                          \
          constant DTYPE * scales_ptr [[buffer(2)]],                                 \
          constant DTYPE * zeros_ptr [[buffer(3)]],                                  \
          device DTYPE * output_data [[buffer(4)]],                                  \
          constant uint3 & sizes [[buffer(5)]],                                      \
          uint3 thread_index [[thread_position_in_grid]],                            \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                          \
          uint tid_in_simdgroup [[thread_index_in_simdgroup]])

    #define INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, GSIZE) \
      INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 1);               \
      INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 2);               \
      INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 3);               \
      INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 4);               \
      INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 5);               \
      INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 6);               \
      INSTANTIATE_QMV_FAST(DTYPE, GSIZE, 7);

    #define INSTANTIATE_QMV_FAST_DTYPE(DTYPE)       \
      INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, 32);  \
      INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, 64);  \
      INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, 128); \
      INSTANTIATE_QMV_FAST_DTYPE_GSIZE(DTYPE, 256);

    INSTANTIATE_QMV_FAST_DTYPE(float);
    INSTANTIATE_QMV_FAST_DTYPE(half);
    #if __METAL_VERSION__ >= 310
    INSTANTIATE_QMV_FAST_DTYPE(bfloat);
    #endif

  /**
  * qmv_impl.metal - handles generic N (any even N, not just N % 8 == 0)
  */

    template <typename T, int group_size, int bits>
    [[kernel]] void qmv_impl(
        constant T* x [[buffer(0)]],
        constant uchar* w [[buffer(1)]],
        constant T* scales [[buffer(2)]],
        constant T* biases [[buffer(3)]],
        device T* y [[buffer(4)]],
        constant uint3 &sizes [[buffer(5)]], // M, K, N
        uint3 tid [[threadgroup_position_in_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]) {
      const int in_vec_size = static_cast<int>(sizes.y); // K
      const int out_vec_size = static_cast<int>(sizes.z); // N

      constexpr int num_simdgroups = 2;
      constexpr int results_per_simdgroup = 4;
      constexpr int packs_per_thread = (bits == 1 || bits == 2) ? 1 : 2;
      constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
      constexpr int pack_factor = bits == 1 ? 16 : power_of_2_bits ? 32 / bits : bits == 6 ? 4 : 8;
      constexpr int bytes_per_pack = bits == 1 ? 2 : power_of_2_bits ? 4 : bits == 6 ? 3 : bits;

      constexpr int values_per_thread = pack_factor * packs_per_thread;
      constexpr int block_size = values_per_thread * SIMD_SIZE;
      constexpr int scale_step_per_thread = group_size / values_per_thread;

      constant uint8_t* ws = (constant uint8_t*)w;

      typedef float U;

      thread U x_thread[values_per_thread];
      thread U result[results_per_simdgroup] = {0};

      // Adjust positions
      const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
      const int in_vec_size_g = (in_vec_size + group_size - 1) / group_size;
      const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
          simd_gid * results_per_simdgroup;
      const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

      if (out_row >= out_vec_size) {
        return;
      }

      // In this case we need to properly guard all our reads because there isn't
      // even 1 tile in the matrix
      if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
        ws +=
            out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
        scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += tid.x * in_vec_size + simd_lid * values_per_thread;
        y += tid.x * out_vec_size + out_row;

        int k = 0;
        for (; k < in_vec_size - block_size; k += block_size) {
          U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

          for (int row = 0; out_row + row < out_vec_size; row++) {
            auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
            constant T* sl = scales + row * in_vec_size_g;
            constant T* bl = biases + row * in_vec_size_g;

            U s = sl[0];
            U b = bl[0];
            result[row] +=
                qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
          }

          ws += block_size * bytes_per_pack / pack_factor;
          scales += block_size / group_size;
          biases += block_size / group_size;
          x += block_size;
        }
        const int remaining = clamp(
            static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
            0,
            values_per_thread);
        if (remaining > 0) {
          U sum = load_vector_safe<T, U, values_per_thread, bits>(
              x, x_thread, remaining);

          for (int row = 0; out_row + row < out_vec_size; row++) {
            auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
            constant T* sl = scales + row * in_vec_size_g;
            constant T* bl = biases + row * in_vec_size_g;

            U s = sl[0];
            U b = bl[0];
            result[row] += qdot_safe<U, values_per_thread, bits>(
                wl, x_thread, s, b, sum, remaining);
          }
        }

        for (int row = 0; out_row + row < out_vec_size; row++) {
          result[row] = simd_sum(result[row]);
          if (simd_lid == 0) {
            y[row] = static_cast<T>(result[row]);
          }
        }
      }

      // In this case the last tile is moved back to redo some output values
      else {
        ws += used_out_row * in_vec_size_w +
            simd_lid * packs_per_thread * bytes_per_pack;
        scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
        x += tid.x * in_vec_size + simd_lid * values_per_thread;
        y += tid.x * out_vec_size + used_out_row;

        int k = 0;
        for (; k < in_vec_size - block_size; k += block_size) {
          U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);

          for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
            constant T* sl = scales + row * in_vec_size_g;
            constant T* bl = biases + row * in_vec_size_g;

            U s = sl[0];
            U b = bl[0];
            result[row] +=
                qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
          }

          ws += block_size * bytes_per_pack / pack_factor;
          scales += block_size / group_size;
          biases += block_size / group_size;
          x += block_size;
        }
        const int remaining = clamp(
            static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
            0,
            values_per_thread);
        if (remaining > 0) {
          U sum = load_vector_safe<T, U, values_per_thread, bits>(
              x, x_thread, remaining);

          for (int row = 0; row < results_per_simdgroup; row++) {
            auto wl = (constant uint8_t*)(ws + row * in_vec_size_w);
            constant T* sl = scales + row * in_vec_size_g;
            constant T* bl = biases + row * in_vec_size_g;

            U s = sl[0];
            U b = bl[0];
            result[row] += qdot_safe<U, values_per_thread, bits>(
                wl, x_thread, s, b, sum, remaining);
          }
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
          result[row] = simd_sum(result[row]);
          if (simd_lid == 0) {
            y[row] = static_cast<T>(result[row]);
          }
        }
      }
    }

    #define INSTANTIATE_QMV_IMPL(DTYPE, GSIZE, NBIT)                                 \
      template [[host_name("qmv_impl_" #NBIT "bit_" #GSIZE "_" #DTYPE)]] kernel void \
      qmv_impl<DTYPE, GSIZE, NBIT>(                                                  \
          constant DTYPE * A [[buffer(0)]],                                          \
          constant uchar * B [[buffer(1)]],                                          \
          constant DTYPE * scales_ptr [[buffer(2)]],                                 \
          constant DTYPE * zeros_ptr [[buffer(3)]],                                  \
          device DTYPE * output_data [[buffer(4)]],                                  \
          constant uint3 & sizes [[buffer(5)]],                                      \
          uint3 thread_index [[thread_position_in_grid]],                            \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                          \
          uint tid_in_simdgroup [[thread_index_in_simdgroup]])

    #define INSTANTIATE_QMV_IMPL_DTYPE_GSIZE(DTYPE, GSIZE) \
      INSTANTIATE_QMV_IMPL(DTYPE, GSIZE, 1);               \
      INSTANTIATE_QMV_IMPL(DTYPE, GSIZE, 2);               \
      INSTANTIATE_QMV_IMPL(DTYPE, GSIZE, 3);               \
      INSTANTIATE_QMV_IMPL(DTYPE, GSIZE, 4);               \
      INSTANTIATE_QMV_IMPL(DTYPE, GSIZE, 5);               \
      INSTANTIATE_QMV_IMPL(DTYPE, GSIZE, 6);               \
      INSTANTIATE_QMV_IMPL(DTYPE, GSIZE, 7);

    #define INSTANTIATE_QMV_IMPL_DTYPE(DTYPE)       \
      INSTANTIATE_QMV_IMPL_DTYPE_GSIZE(DTYPE, 32);  \
      INSTANTIATE_QMV_IMPL_DTYPE_GSIZE(DTYPE, 64);  \
      INSTANTIATE_QMV_IMPL_DTYPE_GSIZE(DTYPE, 128); \
      INSTANTIATE_QMV_IMPL_DTYPE_GSIZE(DTYPE, 256);

    INSTANTIATE_QMV_IMPL_DTYPE(float);
    INSTANTIATE_QMV_IMPL_DTYPE(half);
    #if __METAL_VERSION__ >= 310
    INSTANTIATE_QMV_IMPL_DTYPE(bfloat);
    #endif

  )";
}

// Global shader library cache for Int4MM
static std::unique_ptr<ETMetalShaderLibrary> int4_shader_library = nullptr;

static std::once_flag int4_shader_library_once_flag;

static ETMetalShaderLibrary* get_int4_shader_library() {
  std::call_once(int4_shader_library_once_flag, []() {
    std::string source = get_int4_metal_source();
    int4_shader_library = std::make_unique<ETMetalShaderLibrary>(source);
  });
  return int4_shader_library.get();
}

} // namespace

extern "C" {

AOTITorchError aoti_torch_mps__linear_fp_act_4bit_weight(
    AOTITensorHandle A,
    AOTITensorHandle B,
    int64_t group_size,
    AOTITensorHandle S,
    AOTITensorHandle Z,
    AOTITensorHandle* ret) {

  ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Starting with Metal kernel implementation");

  if (!A || !B || !S || !Z || !ret) {
    ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: null required tensor handles");
    return Error::InvalidArgument;
  }

  // Validate group_size
  if (group_size != 32 && group_size != 64 && group_size != 128 && group_size != 256) {
    ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: Invalid group_size %lld (must be 32, 64, 128, or 256)", group_size);
    return Error::InvalidArgument;
  }

  // Use the same dispatch pattern as other MPS operations for consistent synchronization
  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: Failed to get current Metal stream");
    return Error::Internal;
  }

  try {
    @autoreleasepool {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto* a_tensor = reinterpret_cast<Tensor*>(A);  // Activation: [M, K]
      auto* b_tensor = reinterpret_cast<Tensor*>(B);  // Weight (packed): [N, K/2] (4-bit packed)
      auto* s_tensor = reinterpret_cast<Tensor*>(S);  // Scales: [N, num_groups]
      auto* z_tensor = reinterpret_cast<Tensor*>(Z);  // Zero points: [N, num_groups]

      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Converted tensor handles to ET tensors");

      // Validate A tensor: ndim, dtype, contiguity
      if (a_tensor->dim() != 2) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect A to be 2D tensor, got %d", (int)a_tensor->dim());
        return Error::InvalidArgument;
      }
      auto a_dtype = a_tensor->scalar_type();
      if (a_dtype != exec_aten::ScalarType::Float &&
          a_dtype != exec_aten::ScalarType::BFloat16) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect A to be 32-bit or 16-bit float tensor, got dtype %d", (int)a_dtype);
        return Error::InvalidArgument;
      }
      // Check A is contiguous (stride[1] == 1 and stride[0] == size[1])
      if (a_tensor->strides()[1] != 1 || a_tensor->strides()[0] != a_tensor->sizes()[1]) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect A to be contiguous, strides=[%lld, %lld]",
               (long long)a_tensor->strides()[0], (long long)a_tensor->strides()[1]);
        return Error::InvalidArgument;
      }


      // Validate B tensor: ndim, dtype (uint8), contiguity
      if (b_tensor->dim() != 2) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect B to be 2D tensor, got %d", (int)b_tensor->dim());
        return Error::InvalidArgument;
      }
      if (b_tensor->scalar_type() != exec_aten::ScalarType::Byte) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect B to be uint8 tensor, got dtype %d", (int)b_tensor->scalar_type());
        return Error::InvalidArgument;
      }
      // Check B is contiguous
      if (b_tensor->strides()[1] != 1 || b_tensor->strides()[0] != b_tensor->sizes()[1]) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect B to be contiguous, strides=[%lld, %lld]",
               (long long)b_tensor->strides()[0], (long long)b_tensor->strides()[1]);
        return Error::InvalidArgument;
      }

      // Get dimensions: A is [M, K], B is [N, K/2] (4-bit packed, 2 values per byte)
      int32_t M = static_cast<int32_t>(a_tensor->sizes()[0]);
      int32_t K = static_cast<int32_t>(a_tensor->sizes()[1]);
      int32_t N = static_cast<int32_t>(b_tensor->sizes()[0]);
      constexpr int nbit = 4;

      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: M=%d, K=%d, N=%d, group_size=%lld", M, K, N, group_size);

      // B.size(1) should be (K / 8) * nbit for 4-bit packing
      int64_t expected_b_size1 = (K / 8) * nbit;
      if (b_tensor->sizes()[1] != expected_b_size1) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect B.size(1) == %lld, got %lld",
               (long long)expected_b_size1, (long long)b_tensor->sizes()[1]);
        return Error::InvalidArgument;
      }

      // Validate K alignment
      if (K % 8 != 0) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect K to be multiple of 8, got %d", K);
        return Error::InvalidArgument;
      }

      // Validate N alignment
      if (N % 4 != 0 && M != 1) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect N to be multiple of 4 when M != 1, got M=%d, N=%d", M, N);
        return Error::InvalidArgument;
      }

      // Validate S tensor: 2D with S.size(0) == N, contiguous, dtype matches A
      if (s_tensor->dim() != 2 || s_tensor->sizes()[0] != N) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect S to be 2D tensor with shape [%d, :], got dim=%d, size[0]=%lld",
               N, (int)s_tensor->dim(), (long long)s_tensor->sizes()[0]);
        return Error::InvalidArgument;
      }
      if (s_tensor->scalar_type() != a_dtype) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect S dtype to match A dtype %d, got %d",
               (int)a_dtype, (int)s_tensor->scalar_type());
        return Error::InvalidArgument;
      }
      if (s_tensor->strides()[1] != 1 || s_tensor->strides()[0] != s_tensor->sizes()[1]) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect S to be contiguous, strides=[%lld, %lld]",
               (long long)s_tensor->strides()[0], (long long)s_tensor->strides()[1]);
        return Error::InvalidArgument;
      }

      // Validate Z tensor: 2D with Z.size(0) == N, contiguous, dtype matches A
      if (z_tensor->dim() != 2 || z_tensor->sizes()[0] != N) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect Z to be 2D tensor with shape [%d, :], got dim=%d, size[0]=%lld",
               N, (int)z_tensor->dim(), (long long)z_tensor->sizes()[0]);
        return Error::InvalidArgument;
      }
      if (z_tensor->scalar_type() != a_dtype) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect Z dtype to match A dtype %d, got %d",
               (int)a_dtype, (int)z_tensor->scalar_type());
        return Error::InvalidArgument;
      }
      if (z_tensor->strides()[1] != 1 || z_tensor->strides()[0] != z_tensor->sizes()[1]) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: expect Z to be contiguous, strides=[%lld, %lld]",
               (long long)z_tensor->strides()[0], (long long)z_tensor->strides()[1]);
        return Error::InvalidArgument;
      }

      // Log shapes and strides for all tensors
      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: A tensor shape=[%lld, %lld], strides=[%lld, %lld]",
             (long long)a_tensor->sizes()[0], (long long)a_tensor->sizes()[1],
             (long long)a_tensor->strides()[0], (long long)a_tensor->strides()[1]);
      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: B tensor shape=[%lld, %lld], strides=[%lld, %lld]",
             (long long)b_tensor->sizes()[0], (long long)b_tensor->sizes()[1],
             (long long)b_tensor->strides()[0], (long long)b_tensor->strides()[1]);
      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: S tensor shape=[%lld, %lld], strides=[%lld, %lld]",
             (long long)s_tensor->sizes()[0], (long long)s_tensor->sizes()[1],
             (long long)s_tensor->strides()[0], (long long)s_tensor->strides()[1]);
      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Z tensor shape=[%lld, %lld], strides=[%lld, %lld]",
             (long long)z_tensor->sizes()[0], (long long)z_tensor->sizes()[1],
             (long long)z_tensor->strides()[0], (long long)z_tensor->strides()[1]);

      // Determine data type
      int32_t dtype = static_cast<int32_t>(a_tensor->scalar_type());
      size_t element_size;
      std::string type_str;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        element_size = sizeof(float);
        type_str = "float";
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        element_size = sizeof(uint16_t);
        type_str = "bfloat";
      } else {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: Unsupported data type: %d", dtype);
        return Error::InvalidArgument;
      }

      // Get shader library
      ETMetalShaderLibrary* library = get_int4_shader_library();
      if (!library) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: Failed to get shader library");
        return Error::Internal;
      }

      // Select kernel based on dimensions (matching torchao's get_shader_func_and_dispatch)
      std::string kernel_name;
      bool use_qmv_fast = (M == 1 && N % 8 == 0 && K % 512 == 0);
      bool use_qmv_impl = (M == 1 && !use_qmv_fast);

      if (use_qmv_fast) {
        // Use optimized qmv_fast kernel for M=1 case with aligned dimensions
        kernel_name = "qmv_fast_4bit_" + std::to_string(group_size) + "_" + type_str;
        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Using qmv_fast kernel: %s", kernel_name.c_str());
      } else if (use_qmv_impl) {
        // Use qmv_impl kernel for M=1 case with generic N (handles any even N)
        kernel_name = "qmv_impl_4bit_" + std::to_string(group_size) + "_" + type_str;
        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Using qmv_impl kernel: %s", kernel_name.c_str());
      } else {
        // Use general int4pack_mm kernel
        kernel_name = "int4pack_mm_" + std::to_string(group_size) + "_" + type_str;
        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Using int4pack_mm kernel: %s", kernel_name.c_str());
      }

      // Get kernel function
      auto kernel_func = library->getKernelFunction(kernel_name);
      if (!kernel_func) {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: Failed to get kernel function: %s", kernel_name.c_str());
        return Error::Internal;
      }

      // Allocate output tensor [M, N]
      size_t out_size_bytes = M * N * element_size;
      void* out_contents_ptr = nullptr;
      id<MTLBuffer> out_buffer = allocate_mtl_buffer(&out_contents_ptr, out_size_bytes);

      // Create output tensor handle
      std::vector<int64_t> output_sizes = {M, N};
      std::vector<int64_t> output_strides = {N, 1};

      AOTITensorHandle out_tensor_handle = nullptr;
      AOTITorchError create_out_result = aoti_torch_create_tensor_from_blob_v2(
          out_contents_ptr,
          2,  // ndim
          output_sizes.data(),
          output_strides.data(),
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
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: Failed to create output tensor");
        aoti_torch_mps_free(out_contents_ptr);
        return Error::Internal;
      }

      // Mark that we own the memory
      extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
      memory_to_n_tensor[out_contents_ptr] = 1;

      auto* out_tensor = reinterpret_cast<Tensor*>(out_tensor_handle);

      // Prepare sizes array for the kernel (M, K, N, 0)
      std::array<uint32_t, 4> sizes = {
          static_cast<uint32_t>(M),
          static_cast<uint32_t>(K),
          static_cast<uint32_t>(N),
          0
      };

      // Execute kernel
      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Preparing to execute kernel");

      kernel_func->runCommandBlock([&]() {
        kernel_func->startEncoding();

        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Encoder started, setting arguments");

        // Set buffer arguments
        // Buffer 0: A (activation) [M, K]
        kernel_func->setArg(0, *a_tensor);
        // Buffer 1: B (weight, packed) [N, K/2]
        kernel_func->setArg(1, *b_tensor);
        // Buffer 2: scales [N, num_groups]
        kernel_func->setArg(2, *s_tensor);
        // Buffer 3: zeros [N, num_groups]
        kernel_func->setArg(3, *z_tensor);
        // Buffer 4: output [M, N]
        kernel_func->setArg(4, *out_tensor);
        // Buffer 5: sizes (M, K, N, 0)
        kernel_func->setArg(5, sizes.data(), sizeof(uint32_t) * sizes.size());

        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: All arguments set, dispatching");

        // Dispatch based on kernel type (matching torchao dispatch patterns)
        if (use_qmv_fast || use_qmv_impl) {
          // dispatch_qmv: dispatchThreadgroups with grid (M, (N+7)/8, 1), group (32, 2, 1)
          ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Dispatching kernel: %s", kernel_name.c_str());
          kernel_func->dispatchThreadgroups(
              M,                       // gridX
              (N + 7) / 8,             // gridY
              1,                       // gridZ
              32,                      // threadsX
              2,                       // threadsY
              1);                      // threadsZ
        } else {
          // dispatch_mm_Mr1xNr4_per_TG: dispatchThreads with grid (N/4 * 32, 1, M), group (32, 1, 1)
          ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Dispatching kernel: %s", kernel_name.c_str());
          uint64_t grid_dims[3] = {static_cast<uint64_t>(N / 4 * 32), 1, static_cast<uint64_t>(M)};
          uint64_t group_dims[3] = {32, 1, 1};
          kernel_func->dispatchArrayWithGroupSize(grid_dims, 3, group_dims, 3);
        }
      });

      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Command block completed");

      // Set output tensor handle
      *ret = out_tensor_handle;

      ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Metal kernel implementation completed successfully");

    }  // @autoreleasepool

    ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Executed successfully");
    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: unknown exception");
    return Error::Internal;
  }
}


} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
