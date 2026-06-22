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
    #include <metal_simdgroup>
    #include <metal_stdlib>
    using namespace metal;

    static constant constexpr const int SIMD_SIZE = 32;

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


  /**
   * ============================================================================
   * Steel Library + Quantized GEMM Kernels (M > 1)
   * Ported from MLX: https://github.com/ml-explore/mlx
   * Copyright © 2024 Apple Inc. (MIT License)
   * ============================================================================
   */

  /**
   * steel/defines.h
   */
    #define STEEL_CONST static constant constexpr const

    // Pragma macros - defined as empty to avoid raw string literal issues
    // These are optimization hints that Metal compiler may apply automatically
    #define STEEL_PRAGMA_UNROLL
    #define STEEL_PRAGMA_NO_UNROLL

  /**
   * steel/utils/integral_constant.h
   */

    template <typename T, T v>
    struct IntegralConstant {
      STEEL_CONST T value = v;
      using value_type = T;
      using type = IntegralConstant;
      constexpr operator value_type() const noexcept { return value; }
      constexpr value_type operator()() const noexcept { return value; }
    };

    template <bool B>
    using BoolConstant = IntegralConstant<bool, B>;
    using TrueType = BoolConstant<true>;
    using FalseType = BoolConstant<false>;

    template <int v>
    using Int = IntegralConstant<int, v>;
    template <short v>
    using Short = IntegralConstant<short, v>;

  /**
   * steel/gemm/transforms.h
   */

    template <typename OutT, typename InT>
    struct TransformNone {
      static METAL_FUNC OutT apply(InT x) { return static_cast<OutT>(x); }
      static METAL_FUNC OutT apply(InT x, OutT) { return static_cast<OutT>(x); }
    };

    template <typename T>
    struct AccumHelper { typedef float accum_type; };

  /**
   * steel/gemm/loader.h
   */

    template <
        typename T,
        short BROWS,
        short BCOLS,
        short dst_ld,
        short reduction_dim,
        short tgp_size,
        short alignment = 1,
        short n_reads = (BCOLS * BROWS) / (tgp_size),
        short TCOLS = BCOLS / n_reads,
        short TROWS = tgp_size / TCOLS>
    struct BlockLoader {
      STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
      STEEL_CONST short vec_size = n_reads;
      const int src_ld;
      const int tile_stride;
      const short thread_idx;
      const short bi;
      const short bj;
      threadgroup T* dst;
      const device T* src;

      struct alignas(alignment * sizeof(T)) ReadVector {
        uint8_t v[sizeof(T) * vec_size];
      };

      METAL_FUNC BlockLoader(
          const device T* src_,
          const int src_ld_,
          threadgroup T* dst_,
          ushort simd_group_id [[simdgroup_index_in_threadgroup]],
          ushort simd_lane_id [[thread_index_in_simdgroup]])
          : src_ld(src_ld_),
            tile_stride(reduction_dim ? BCOLS : BROWS * src_ld),
            thread_idx(simd_group_id * 32 + simd_lane_id),
            bi(thread_idx / TCOLS),
            bj(vec_size * (thread_idx % TCOLS)),
            dst(dst_ + bi * dst_ld + bj),
            src(src_ + bi * src_ld + bj) {}

      METAL_FUNC void load_unsafe() const {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < BROWS; i += TROWS) {
          *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
              *((const device ReadVector*)(&src[i * src_ld]));
        }
      }

      METAL_FUNC void load_safe(short2 src_tile_dim) const {
        src_tile_dim = src_tile_dim - short2(bj, bi);
        if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < BROWS; i += TROWS) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < vec_size; j++) {
              dst[i * dst_ld + j] = T(0);
            }
          }
          return;
        }
        bool tmp_idx[vec_size];
        T tmp_val[vec_size];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < BROWS; i += TROWS) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            tmp_idx[j] = (i < src_tile_dim.y) && (j < src_tile_dim.x);
          }
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            tmp_val[j] = src[(tmp_idx[j] ? i * src_ld + j : 0)];
          }
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
          }
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < vec_size; j++) {
            dst[i * dst_ld + j] = tmp_val[j];
          }
        }
      }

      METAL_FUNC void next() { src += tile_stride; }
    };

  /**
   * Quantized utilities
   */

    template <int bits, int wsize = 8>
    inline constexpr short get_pack_factor() {
      return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
    }

    template <int bits, int wsize = 8>
    inline constexpr short get_bytes_per_pack() {
      constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
      return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
    }

    template <typename U, int N, int bits>
    inline void dequantize(
        const device uint8_t* w, U scale, U bias, threadgroup U* w_local) {
      static_assert(bits == 4, "Only 4-bit quantization supported");

      U s[2] = {scale, scale / static_cast<U>(16.0f)};
      for (int i = 0; i < (N / 2); i++) {
        w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
        w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
      }
    }

  /**
   * QuantizedBlockLoader
   */

    template <
        typename T,
        short BROWS,
        short BCOLS,
        short dst_ld,
        short reduction_dim,
        short tgp_size,
        short group_size,
        short bits>
    struct QuantizedBlockLoader {
      static_assert(BCOLS <= group_size, "group_size should be >= BCOLS");
      static_assert(group_size % BCOLS == 0, "group_size should be divisible by BCOLS");
      static_assert(bits == 4, "Only 4-bit quantization supported");

      STEEL_CONST short pack_factor = ::get_pack_factor<bits, 8>();
      STEEL_CONST short bytes_per_pack = ::get_bytes_per_pack<bits>();
      STEEL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
      STEEL_CONST short n_reads =
          (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
      STEEL_CONST short group_steps = group_size / BCOLS;

      const int src_ld;
      const int tile_stride;
      short group_step_cnt;
      const int group_stride;
      const short thread_idx;
      const short bi;
      const short bj;
      threadgroup T* dst;
      const device uint8_t* src;
      const device T* scales;
      const device T* biases;

      QuantizedBlockLoader(
          const device uint8_t* src_,
          const device T* scales_,
          const device T* biases_,
          const int src_ld_,
          threadgroup T* dst_,
          ushort simd_group_id [[simdgroup_index_in_threadgroup]],
          ushort simd_lane_id [[thread_index_in_simdgroup]])
          : src_ld(src_ld_),
            tile_stride(
                reduction_dim ? BCOLS_PACKED * bytes_per_pack
                              : BROWS * src_ld * bytes_per_pack / pack_factor),
            group_step_cnt(0),
            group_stride(BROWS * src_ld / group_size),
            thread_idx(simd_group_id * 32 + simd_lane_id),
            bi(n_reads * thread_idx / BCOLS_PACKED),
            bj((n_reads * thread_idx) % BCOLS_PACKED),
            dst(dst_ + bi * dst_ld + bj * pack_factor),
            src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
                bj * bytes_per_pack),
            scales(scales_ + bi * src_ld / group_size),
            biases(biases_ + bi * src_ld / group_size) {}

      void load_unsafe() const {
        if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
          return;
        }
        T scale = *scales;
        T bias = *biases;
        for (int i = 0; i < n_reads; i++) {
          dequantize<T, pack_factor, bits>(
              src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
        }
      }

      void load_safe(short2 src_tile_dim) const {
        if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
          return;
        }
        if (reduction_dim == 1 && bi >= src_tile_dim.x) {
          for (int i = 0; i < n_reads * pack_factor; i++) {
            dst[i] = T(0);
          }
          return;
        }
        if (reduction_dim == 0 && bi >= src_tile_dim.y) {
          for (int i = 0; i < n_reads * pack_factor; i++) {
            dst[i] = T(0);
          }
          return;
        }
        T scale = *scales;
        T bias = *biases;
        for (int i = 0; i < n_reads; i++) {
          dequantize<T, pack_factor, bits>(
              src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
        }
      }

      void next() {
        src += tile_stride;
        if (reduction_dim == 1) {
          if (group_steps > 1) {
            group_step_cnt++;
            if (group_step_cnt == group_steps) {
              group_step_cnt = 0;
              scales++;
              biases++;
            }
          } else {
            scales++;
            biases++;
          }
        } else {
          scales += group_stride;
          biases += group_stride;
        }
      }
    };

  /**
   * steel/gemm/mma.h - Matrix multiply-accumulate
   */

    template <typename T, int kFragRows_, int kFragCols_>
    struct BaseMMAFrag {
      static_assert(kFragRows_ == 8, "Only 8x8 fragments supported");
      static_assert(kFragCols_ == 8, "Only 8x8 fragments supported");
    };

    template <typename T>
    struct BaseMMAFrag<T, 8, 8> {
      STEEL_CONST int kFragRows = 8;
      STEEL_CONST int kFragCols = 8;
      STEEL_CONST int kElemsPerFrag = 2;
      typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
      typedef metal::vec<T, kElemsPerFrag> frag_type;

      METAL_FUNC static constexpr short2 get_coord(ushort simd_lane_id) {
        const short qid = simd_lane_id / 4;
        const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
        const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
        return short2{fn, fm};
      }

      template <typename SrcPtrType, typename StrX, typename StrY>
      METAL_FUNC static constexpr void
      load(thread frag_type& dst, SrcPtrType src, StrX str_x, StrY str_y) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 1; i++) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 2; j++) {
            dst[i * 2 + j] = static_cast<T>(src[i * str_x + j * str_y]);
          }
        }
      }

      template <typename DstPtrType, typename StrX, typename StrY>
      METAL_FUNC static constexpr void
      store(const thread frag_type& src, DstPtrType dst, StrX str_x, StrY str_y) {
        using U = typename metal::remove_pointer<DstPtrType>::type;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 1; i++) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 2; j++) {
            dst[i * str_x + j * str_y] = static_cast<U>(src[i * 2 + j]);
          }
        }
      }

      template <
          typename DstPtrType,
          typename StrX,
          typename StrY,
          typename LimX,
          typename LimY,
          typename OffX,
          typename OffY>
      METAL_FUNC static constexpr void store_safe(
          const thread frag_type& src,
          DstPtrType dst,
          StrX str_x,
          StrY str_y,
          LimX lim_x,
          LimY lim_y,
          OffX off_x = Int<0>{},
          OffY off_y = Int<0>{}) {
        using U = typename metal::remove_pointer<DstPtrType>::type;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 1; i++) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 2; j++) {
            if ((off_x + i) < lim_x && (off_y + j) < lim_y) {
              dst[(off_x + i) * str_x + (off_y + j) * str_y] =
                  static_cast<U>(src[i * 2 + j]);
            }
          }
        }
      }

      METAL_FUNC static constexpr void mma(
          thread frag_type& D,
          thread frag_type& A,
          thread frag_type& B,
          thread frag_type& C) {
        mat_type D_mat, A_mat, B_mat, C_mat;
        reinterpret_cast<thread frag_type&>(A_mat.thread_elements()) = A;
        reinterpret_cast<thread frag_type&>(B_mat.thread_elements()) = B;
        reinterpret_cast<thread frag_type&>(C_mat.thread_elements()) = C;
        simdgroup_multiply_accumulate(D_mat, A_mat, B_mat, C_mat);
        D = reinterpret_cast<thread frag_type&>(D_mat.thread_elements());
      }
    };

    template <
        typename T,
        int kTileRows_,
        int kTileCols_,
        class MMAFrag_ = BaseMMAFrag<T, 8, 8>>
    struct MMATile {
      using MMAFrag_t = MMAFrag_;
      using elem_type = T;
      STEEL_CONST int kFragRows = MMAFrag_t::kFragRows;
      STEEL_CONST int kFragCols = MMAFrag_t::kFragCols;
      STEEL_CONST int kElemsPerFrag = MMAFrag_t::kElemsPerFrag;
      STEEL_CONST int kTileRows = kTileRows_;
      STEEL_CONST int kTileCols = kTileCols_;
      STEEL_CONST int kRows = kTileRows * kFragRows;
      STEEL_CONST int kCols = kTileCols * kFragCols;
      STEEL_CONST int kNumFrags = kTileRows * kTileCols;
      STEEL_CONST int kElemsPerTile = kNumFrags * kElemsPerFrag;

      typedef typename MMAFrag_t::frag_type frag_type;
      frag_type val_frags[kNumFrags] = {frag_type(0)};

      METAL_FUNC MMATile() thread {}

      METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) {
        return val_frags[i * kTileCols + j];
      }

      METAL_FUNC constexpr const thread frag_type& frag_at(
          const short i, const short j) const {
        return val_frags[i * kTileCols + j];
      }

      METAL_FUNC thread elem_type* elems() {
        return reinterpret_cast<thread elem_type*>(val_frags);
      }

      template <typename U, int w_x, int w_y, int str_x, int str_y>
      METAL_FUNC void load(const threadgroup U* src) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kTileRows; ++i) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kTileCols; ++j) {
            MMAFrag_t::load(
                frag_at(i, j),
                &(src[(i * kFragRows) * w_x * str_x +
                      (j * kFragCols) * w_y * str_y]),
                Int<str_x>{},
                Int<str_y>{});
          }
        }
      }

      template <typename U, int w_x, int w_y>
      METAL_FUNC void store(device U* dst, const int ld) const {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kTileRows; ++i) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kTileCols; ++j) {
            MMAFrag_t::store(
                frag_at(i, j),
                &(dst[(i * kFragRows) * w_x * ld + (j * kFragCols) * w_y]),
                ld,
                Int<1>{});
          }
        }
      }

      template <typename U, int w_x, int w_y>
      METAL_FUNC void
      store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const {
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < kTileRows; ++i) {
          STEEL_PRAGMA_UNROLL
          for (int j = 0; j < kTileCols; ++j) {
            MMAFrag_t::store_safe(
                frag_at(i, j),
                dst,
                ld,
                Int<1>{},
                dst_tile_dims.y,
                dst_tile_dims.x,
                (i * kFragRows) * w_x,
                (j * kFragCols) * w_y);
          }
        }
      }
    };

    template <typename T, typename U, int M, int N, int K>
    METAL_FUNC void tile_matmad(
        thread MMATile<T, M, N>& D,
        thread MMATile<U, M, K>& A,
        thread MMATile<U, K, N>& B,
        thread MMATile<T, M, N>& C) {
      STEEL_PRAGMA_UNROLL
      for (short m = 0; m < M; ++m) {
        STEEL_PRAGMA_UNROLL
        for (short n = 0; n < N; ++n) {
          short n_serp = (m % 2) ? (N - 1 - n) : n;
          STEEL_PRAGMA_UNROLL
          for (short k = 0; k < K; ++k) {
            MMATile<T, M, N>::MMAFrag_t::mma(
                D.frag_at(m, n_serp),
                A.frag_at(m, k),
                B.frag_at(k, n_serp),
                C.frag_at(m, n_serp));
          }
        }
      }
    }

    template <
        typename T,
        typename U,
        int BM,
        int BN,
        int BK,
        int WM,
        int WN,
        bool transpose_a,
        bool transpose_b,
        short lda_tgp,
        short ldb_tgp,
        typename AccumType = float,
        typename Epilogue = TransformNone<U, AccumType>>
    struct BlockMMA {
      STEEL_CONST short kFragSize = 8;
      using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

      STEEL_CONST short TM_stride = kFragSize * WM;
      STEEL_CONST short TN_stride = kFragSize * WN;
      STEEL_CONST short TM = BM / (kFragSize * WM);
      STEEL_CONST short TN = BN / (kFragSize * WN);

      STEEL_CONST short A_str_m = transpose_a ? 1 : lda_tgp;
      STEEL_CONST short A_str_k = transpose_a ? lda_tgp : 1;
      STEEL_CONST short B_str_k = transpose_b ? 1 : ldb_tgp;
      STEEL_CONST short B_str_n = transpose_b ? ldb_tgp : 1;

      STEEL_CONST short tile_stride_a = kFragSize * A_str_k;
      STEEL_CONST short tile_stride_b = kFragSize * B_str_k;

      MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
      MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
      MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;

      short sm, sn;
      short As_offset, Bs_offset;

      METAL_FUNC BlockMMA(
          ushort simd_group_id [[simdgroup_index_in_threadgroup]],
          ushort simd_lane_id [[thread_index_in_simdgroup]]) {
        short tm = kFragSize * (simd_group_id / WN);
        short tn = kFragSize * (simd_group_id % WN);
        short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
        sm = simd_coord.y;
        sn = simd_coord.x;
        As_offset = (tm + sm) * A_str_m + (sn) * A_str_k;
        Bs_offset = (sm) * B_str_k + (tn + sn) * B_str_n;
        sm += tm;
        sn += tn;
      }

      METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) {
        As += As_offset;
        Bs += Bs_offset;
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < BK; kk += kFragSize) {
          simdgroup_barrier(mem_flags::mem_none);
          Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);
          simdgroup_barrier(mem_flags::mem_none);
          Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);
          simdgroup_barrier(mem_flags::mem_none);
          tile_matmad(Ctile, Atile, Btile, Ctile);
          As += tile_stride_a;
          Bs += tile_stride_b;
        }
      }

      METAL_FUNC void store_result(device U* D, const int ldd) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
          Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
        }
        D += sm * ldd + sn;
        Ctile.template store<U, WM, WN>(D, ldd);
      }

      METAL_FUNC void
      store_result_safe(device U* D, const int ldd, short2 dst_tile_dims) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < decltype(Ctile)::kElemsPerTile; i++) {
          Ctile.elems()[i] = Epilogue::apply(Ctile.elems()[i]);
        }
        D += sm * ldd + sn;
        dst_tile_dims -= short2(sn, sm);
        if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
          return;
        Ctile.template store_safe<U, WM, WN>(D, ldd, dst_tile_dims);
      }
    };

  /**
   * Quantized GEMM implementation (transposed weights)
   */
    template <
        typename T,
        const int group_size,
        const int BM = 32,
        const int BK = 32,
        const int BN = 32>
    METAL_FUNC void qmm_t_impl(
        const device uint8_t* w,
        const device T* scales,
        const device T* biases,
        const device T* x,
        device T* y,
        threadgroup T* Xs,
        threadgroup T* Ws,
        const int K,
        const int N,
        const int M,
        uint3 tid [[threadgroup_position_in_grid]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]) {

      constexpr int WM = 2;
      constexpr int WN = 2;
      constexpr int bits = 4;
      constexpr int BK_padded = (BK + 16 / sizeof(T));

      using mma_t = BlockMMA<T, T, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
      using loader_x_t = BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
      using loader_w_t = QuantizedBlockLoader<
          T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, bits>;

      const int K_g = K / group_size;
      const int y_row = tid.y * BM;
      const int y_col = tid.x * BN;

      x += y_row * static_cast<int64_t>(K);
      w += y_col * (K / 2);  // 4-bit packing: K/2 bytes per row
      scales += y_col * K_g;
      biases += y_col * K_g;
      y += y_row * static_cast<int64_t>(N) + y_col;

      const short num_els = min(BM, M - y_row);
      const short num_outs = min(BN, N - y_col);
      loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
      loader_w_t loader_w(w, scales, biases, K, Ws, simd_gid, simd_lid);
      mma_t mma_op(simd_gid, simd_lid);

      if (num_els < BM) {
        if (num_outs < BN) {
          for (int k = 0; k < K; k += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader_x.load_safe(short2(BK, num_els));
            loader_w.load_safe(short2(BK, num_outs));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(Xs, Ws);
            loader_x.next();
            loader_w.next();
          }
        } else {
          for (int k = 0; k < K; k += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader_x.load_safe(short2(BK, num_els));
            loader_w.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(Xs, Ws);
            loader_x.next();
            loader_w.next();
          }
        }
      } else {
        if (num_outs < BN) {
          for (int k = 0; k < K; k += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader_x.load_unsafe();
            loader_w.load_safe(short2(BK, num_outs));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(Xs, Ws);
            loader_x.next();
            loader_w.next();
          }
        } else {
          for (int k = 0; k < K; k += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader_x.load_unsafe();
            loader_w.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(Xs, Ws);
            loader_x.next();
            loader_w.next();
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (num_els < BM || num_outs < BN) {
        mma_op.store_result_safe(y, N, short2(num_outs, num_els));
      } else {
        mma_op.store_result(y, N);
      }
    }

  /**
   * Quantized GEMM kernel wrappers
   */
    template <typename T, const int group_size, const int BM = 32, const int BK = 32, const int BN = 32>
    [[kernel]] void affine_qmm_t(
        const device uint8_t* w [[buffer(0)]],
        const device T* scales [[buffer(1)]],
        const device T* biases [[buffer(2)]],
        const device T* x [[buffer(3)]],
        device T* y [[buffer(4)]],
        const constant int& K [[buffer(5)]],
        const constant int& N [[buffer(6)]],
        const constant int& M [[buffer(7)]],
        uint3 tid [[threadgroup_position_in_grid]],
        uint lid [[thread_index_in_threadgroup]],
        uint simd_gid [[simdgroup_index_in_threadgroup]],
        uint simd_lid [[thread_index_in_simdgroup]]) {
      (void)lid;
      constexpr int BK_padded = (BK + 16 / sizeof(T));
      threadgroup T Xs[BM * BK_padded];
      threadgroup T Ws[BN * BK_padded];
      qmm_t_impl<T, group_size, BM, BK, BN>(
          w, scales, biases, x, y, Xs, Ws, K, N, M, tid, simd_gid, simd_lid);
    }

    // Instantiate GEMM kernels
    #define INSTANTIATE_QMM_T(DTYPE, GSIZE)                                    \
      template [[host_name("affine_qmm_t_4bit_gs" #GSIZE "_" #DTYPE)]]         \
      [[kernel]] void affine_qmm_t<DTYPE, GSIZE>(                             \
          const device uint8_t* w [[buffer(0)]],                              \
          const device DTYPE* scales [[buffer(1)]],                           \
          const device DTYPE* biases [[buffer(2)]],                           \
          const device DTYPE* x [[buffer(3)]],                                \
          device DTYPE* y [[buffer(4)]],                                      \
          const constant int& K [[buffer(5)]],                                \
          const constant int& N [[buffer(6)]],                                \
          const constant int& M [[buffer(7)]],                                \
          uint3 tid [[threadgroup_position_in_grid]],                         \
          uint lid [[thread_index_in_threadgroup]],                           \
          uint simd_gid [[simdgroup_index_in_threadgroup]],                   \
          uint simd_lid [[thread_index_in_simdgroup]])

    INSTANTIATE_QMM_T(float, 32);
    INSTANTIATE_QMM_T(float, 64);
    INSTANTIATE_QMM_T(float, 128);
    INSTANTIATE_QMM_T(float, 256);
    #if __METAL_VERSION__ >= 310
    INSTANTIATE_QMM_T(bfloat, 32);
    INSTANTIATE_QMM_T(bfloat, 64);
    INSTANTIATE_QMM_T(bfloat, 128);
    INSTANTIATE_QMM_T(bfloat, 256);
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

      // Select kernel based on dimensions (following MLX dispatch strategy)
      std::string kernel_name;
      bool use_qmv_fast = (M == 1 && N % 8 == 0 && K % 512 == 0);
      bool use_qmv_impl = (M == 1 && !use_qmv_fast);
      bool use_qmm = (M > 1);  // Use GEMM for M > 1

      if (use_qmv_fast) {
        // Use optimized qmv_fast kernel for M=1 case with aligned dimensions
        kernel_name = "qmv_fast_4bit_" + std::to_string(group_size) + "_" + type_str;
        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Using qmv_fast kernel: %s", kernel_name.c_str());
      } else if (use_qmv_impl) {
        // Use qmv_impl kernel for M=1 case with generic N (handles any even N)
        kernel_name = "qmv_impl_4bit_" + std::to_string(group_size) + "_" + type_str;
        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Using qmv_impl kernel: %s", kernel_name.c_str());
      } else if (use_qmm) {
        // Use steel-based GEMM kernel for M > 1 (affine_qmm_t with transposed weights)
        kernel_name = "affine_qmm_t_4bit_gs" + std::to_string(group_size) + "_" + type_str;
        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Using affine_qmm_t kernel: %s (M=%d, N=%d, K=%d)",
               kernel_name.c_str(), M, N, K);
      } else {
        ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: No suitable kernel found for M=%d, N=%d, K=%d", M, N, K);
        return Error::Internal;
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

        // Set buffer arguments (layout differs between GEMV and GEMM)
        if (use_qmm) {
          // GEMM kernel (affine_qmm_t) buffer layout:
          // Buffer 0: w (uint8_t*, quantized weights) [N, K/2]
          // Buffer 1: scales (T*) [N, num_groups]
          // Buffer 2: biases (T*) [N, num_groups]
          // Buffer 3: x (T*, activations) [M, K]
          // Buffer 4: y (T*, output) [M, N]
          // Buffer 5: K (constant int&)
          // Buffer 6: N (constant int&)
          // Buffer 7: M (constant int&)
          kernel_func->setArg(0, *b_tensor);     // w
          kernel_func->setArg(1, *s_tensor);     // scales
          kernel_func->setArg(2, *z_tensor);     // biases
          kernel_func->setArg(3, *a_tensor);     // x
          kernel_func->setArg(4, *out_tensor);   // y
          int32_t K_val = K, N_val = N, M_val = M;
          kernel_func->setArg(5, &K_val, sizeof(int32_t));
          kernel_func->setArg(6, &N_val, sizeof(int32_t));
          kernel_func->setArg(7, &M_val, sizeof(int32_t));
        } else {
          // GEMV kernel (qmv_*) buffer layout:
          // Buffer 0: A (activation) [M, K]
          // Buffer 1: B (weight, packed) [N, K/2]
          // Buffer 2: scales [N, num_groups]
          // Buffer 3: zeros [N, num_groups]
          // Buffer 4: output [M, N]
          // Buffer 5: sizes (M, K, N, 0)
          kernel_func->setArg(0, *a_tensor);
          kernel_func->setArg(1, *b_tensor);
          kernel_func->setArg(2, *s_tensor);
          kernel_func->setArg(3, *z_tensor);
          kernel_func->setArg(4, *out_tensor);
          kernel_func->setArg(5, sizes.data(), sizeof(uint32_t) * sizes.size());
        }

        ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: All arguments set, dispatching");

        // Dispatch based on kernel type
        if (use_qmv_fast || use_qmv_impl) {
          // dispatch_qmv: dispatchThreadgroups with grid (M, (N+7)/8, 1), group (32, 2, 1)
          ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Dispatching GEMV kernel: %s", kernel_name.c_str());
          kernel_func->dispatchThreadgroups(
              M,                       // gridX
              (N + 7) / 8,             // gridY
              1,                       // gridZ
              32,                      // threadsX
              2,                       // threadsY
              1);                      // threadsZ
        } else if (use_qmm) {
          // dispatch GEMM (affine_qmm_t):
          // Block size: BM=32, BN=32
          // Threadgroup size: 128 threads (4 simdgroups)
          // Grid: ((N+31)/32, (M+31)/32, 1)
          constexpr int BM = 32;
          constexpr int BN = 32;
          uint32_t grid_n = (N + BN - 1) / BN;
          uint32_t grid_m = (M + BM - 1) / BM;
          ET_LOG(Debug, "aoti_torch_mps__linear_fp_act_4bit_weight: Dispatching GEMM kernel: %s (grid: %u x %u)",
                 kernel_name.c_str(), grid_n, grid_m);
          kernel_func->dispatchThreadgroups(
              grid_n,                  // gridX: number of N blocks
              grid_m,                  // gridY: number of M blocks
              1,                       // gridZ
              128,                     // threadsX: 4 simdgroups × 32 threads
              1,                       // threadsY
              1);                      // threadsZ
        } else {
          ET_LOG(Error, "aoti_torch_mps__linear_fp_act_4bit_weight: Unknown kernel type");
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
