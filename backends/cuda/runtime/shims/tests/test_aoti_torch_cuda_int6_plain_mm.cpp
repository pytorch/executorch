/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/int6_plain_mm.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>

using executorch::backends::cuda::aoti_torch_cuda_int6_plain_mm;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::backends::cuda::AOTITorchError;
using executorch::runtime::Error;
namespace slim_c10 = executorch::backends::aoti::slim::c10;

using Tensor = executorch::backends::aoti::slim::SlimTensor;

// W6A8 dp4a matvec shim for packed-INT6 decode (CudaDp4aPlanarInt6Tensor
// layout, GGUF Q6_K). The 6-bit weight is split into two planes plus a
// per-group scale; there is NO zero tensor (Q6_K is symmetric, the -32 offset
// is applied in the kernel):
//   ql    : [N, K/2] uint8 — low-nibble plane, nibble-packed even/odd
//   qh    : [N, K/4] uint8 — high-2-bit plane, 4 values/byte (per 32-weight
//           chunk: hi_even_packed[4] then hi_odd_packed[4])
//   scale : [N, K/gs] int8 — per-group scale *codes* (row-major)
//   steps : [N, 1] bf16 — per-row super-scale; the real per-group scale is
//           scale_code * step.
// Test vectors are generated from the production pack path (pack_int6 +
// _encode_int8_per_row, dp4a_planar_int6_tensor.py) and the expected[] outputs
// from the export-path reference _dequant_matmul_int6 (int6_dispatch.py):
//   w[n, k] = q[n, k] * (scale_code[n, k//gs] * step[n]); out = A @ w^T
// (q symmetric, in [-32, 31]). The kernel runs W6A8 (it also quantizes
// activations to int8), so a 0.5 atol absorbs the activation-quant noise.
class AOTITorchInt6PlainMMTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  Tensor* create_tensor(
      const std::vector<int64_t>& sizes,
      slim_c10::ScalarType dtype) {
    Tensor* tensor;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        nullptr,
        static_cast<int32_t>(dtype),
        static_cast<int32_t>(slim_c10::DeviceType::CUDA),
        0,
        &tensor);
    return (error == Error::Ok) ? tensor : nullptr;
  }

  Tensor* create_bf16(const std::vector<int64_t>& sizes) {
    return create_tensor(sizes, slim_c10::ScalarType::BFloat16);
  }

  // ql / qh are uint8 (ScalarType::Byte) packed planes.
  Tensor* create_uint8(const std::vector<int64_t>& sizes) {
    return create_tensor(sizes, slim_c10::ScalarType::Byte);
  }

  // scale codes are signed int8 (ScalarType::Char).
  Tensor* create_int8(const std::vector<int64_t>& sizes) {
    return create_tensor(sizes, slim_c10::ScalarType::Char);
  }

  // Upload raw bytes to a CUDA tensor.
  void upload(Tensor* t, const void* host_data, size_t bytes) {
    cudaMemcpy(t->data_ptr(), host_data, bytes, cudaMemcpyHostToDevice);
  }

  // Download CUDA tensor to host buffer.
  void download(const Tensor* t, void* host_data, size_t bytes) {
    cudaMemcpy(host_data, t->data_ptr(), bytes, cudaMemcpyDeviceToHost);
  }

  // Run the shim and return the output tensor (asserts success).
  Tensor* run(
      Tensor* A,
      Tensor* ql,
      Tensor* qh,
      Tensor* scale,
      Tensor* steps,
      int64_t group_size) {
    Tensor* output = nullptr;
    AOTITorchError error = aoti_torch_cuda_int6_plain_mm(
        A, ql, qh, scale, steps, group_size, &output);
    EXPECT_EQ(error, Error::Ok);
    EXPECT_NE(output, nullptr);
    return output;
  }

  // Check output bf16 values against expected, with absolute tolerance.
  void check_bf16_output(
      Tensor* output,
      const uint16_t* expected_data,
      int64_t count,
      float atol = 0.5f) {
    std::vector<uint16_t> actual(count);
    download(output, actual.data(), count * sizeof(uint16_t));
    cudaDeviceSynchronize();

    for (int64_t i = 0; i < count; i++) {
      // Convert bf16 raw bits to float: bf16 is the upper 16 bits of float32.
      uint32_t actual_bits = static_cast<uint32_t>(actual[i]) << 16;
      uint32_t expected_bits = static_cast<uint32_t>(expected_data[i]) << 16;
      float actual_f, expected_f;
      memcpy(&actual_f, &actual_bits, sizeof(float));
      memcpy(&expected_f, &expected_bits, sizeof(float));

      EXPECT_NEAR(actual_f, expected_f, atol)
          << "Mismatch at index " << i << ": actual=" << actual_f
          << " expected=" << expected_f;
    }
  }

  // Upload data and run the shim. ql/qh are uint8; scale is int8 codes; steps
  // and A are bf16.
  Tensor* setup_and_run(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t gs,
      const uint8_t* ql_host,
      const uint8_t* qh_host,
      const int8_t* scale_codes,
      const uint16_t* steps_host,
      const uint16_t* A_host) {
    int64_t ng = K / gs;
    Tensor* A = create_bf16({M, K});
    Tensor* ql = create_uint8({N, K / 2});
    Tensor* qh = create_uint8({N, K / 4});
    Tensor* scale = create_int8({N, ng});
    Tensor* steps = create_bf16({N, 1});
    EXPECT_NE(A, nullptr);
    EXPECT_NE(ql, nullptr);
    EXPECT_NE(qh, nullptr);
    EXPECT_NE(scale, nullptr);
    EXPECT_NE(steps, nullptr);

    upload(A, A_host, static_cast<size_t>(M) * K * sizeof(uint16_t));
    upload(ql, ql_host, static_cast<size_t>(N) * (K / 2) * sizeof(uint8_t));
    upload(qh, qh_host, static_cast<size_t>(N) * (K / 4) * sizeof(uint8_t));
    upload(scale, scale_codes, static_cast<size_t>(N) * ng * sizeof(int8_t));
    upload(steps, steps_host, static_cast<size_t>(N) * sizeof(uint16_t));

    return run(A, ql, qh, scale, steps, gs);
  }
};

// Q6KGroupSize16: M=2, N=4, K=64, gs=16, symmetric (no zero), q in [-32,31].
// The canonical GGUF Q6_K shape (group_size=16).
TEST_F(AOTITorchInt6PlainMMTest, Q6KGroupSize16) {
  int64_t M = 2, N = 4, K = 64, gs = 16;
  int64_t ng = K / gs; // 4

  // clang-format off
  uint8_t ql_host[] = {
      196,  68,  54,  41,  38, 120, 121, 172,   1, 200,  46, 101,
       88, 108, 202,  93, 254, 104,  45, 107,  52, 125, 169,  19,
      182,  88,  20, 186,  27, 135, 187, 151, 206, 230, 189, 222,
      143,  11, 145,  60, 212,  85,   5, 193, 112,  98, 235, 121,
      226, 139,  42, 171, 138, 156, 177,  71, 109, 195, 153,   5,
      221,  12,  82, 237,   9, 114,  91, 234, 116,  63, 139, 235,
        1,  18,  44, 216, 103, 200,   6, 196,  86, 178, 201, 186,
      127, 244,  36, 238, 165, 139, 223,   5, 159, 167, 246,  42,
       25,  15, 108, 171,  32,  79, 194,  36, 224,  43,  52, 181,
       10, 143,  44, 130, 161,  97, 131, 221,  73, 111, 205, 154,
      175, 236, 159,  46, 254,  16,   0, 129
  };
  uint8_t qh_host[] = {
      149,  59,  68, 176,  22,   1, 231,  10, 132,  68,   6, 184,
      113, 170, 155, 183,  96,   7, 138, 170, 117,   0, 162,  12,
       99, 250,  44, 231,  59,  30, 158, 110, 166, 221,  26, 115,
      130, 160,  31, 148, 121,  25, 222,  24, 247,  71, 234,  77,
      129, 104,  39, 103,  36, 249,  85, 179, 216,  54, 100,  18,
       91, 227,  57, 169
  };
  int8_t scale_codes[] = {
        68, -127,  105,  -15,  127,   39,   81,  103,   76,  -35, -127,  111,
        80,   35,  -50, -127
  };
  uint16_t steps_host[] = {
      0x39C4, 0x39DB, 0x3A55, 0x3A5C
  };
  uint16_t A_host[] = {
      0x3F81, 0xC01B, 0x3F2B, 0xBE50, 0x3EE2, 0x3EEF, 0x3FB7, 0xBDF3,
      0x3F78, 0xBF99, 0xBEEF, 0xBF9F, 0x3F8D, 0x3E41, 0xBD6A, 0xBFBE,
      0xBF14, 0x3E3C, 0xBDC1, 0x3F14, 0xBE8A, 0xC018, 0x3E98, 0xBF33,
      0xBEAA, 0xBDA1, 0xBF0D, 0x3F7A, 0xBF16, 0xBF88, 0x3F3A, 0x3DF0,
      0xC009, 0xBD1C, 0x3EAE, 0xBFBD, 0x400F, 0xBE44, 0xBFA4, 0x3F09,
      0x3EAA, 0xBD3F, 0x3F91, 0xBE37, 0xBDC8, 0xBF4C, 0xBFC8, 0xBF24,
      0xBE15, 0xBEC4, 0x3F8A, 0x3E9A, 0x3E25, 0x3E0A, 0x3DB2, 0x3DC0,
      0x3FD9, 0x3FAB, 0xBF4D, 0xBF22, 0xBD08, 0x3E80, 0xBF56, 0xBF79,
      0x3F0F, 0xBFAA, 0xBF3B, 0x3E69, 0x3F50, 0x3F9E, 0xBD8E, 0x3D04,
      0xBE86, 0xBECF, 0xBFDE, 0x3F7D, 0x3D27, 0x3F60, 0xBFC0, 0x3F40,
      0x3FCB, 0xBD60, 0x3CA4, 0x3F99, 0xBFDB, 0xBF1E, 0x3EEC, 0x3F55,
      0x3F95, 0x3FB4, 0x3DDB, 0x3E40, 0xBF15, 0xBE70, 0xBF9D, 0xBEBE,
      0xBF2C, 0xBFA1, 0x3DCA, 0x3EDB, 0xBF2C, 0x3F5B, 0xBE0C, 0xBF69,
      0xBF9D, 0x3EE8, 0x3F4C, 0x3F82, 0xBF13, 0x3F08, 0x3ED6, 0xBFB3,
      0xBEBD, 0x3F15, 0x3EBE, 0xBF3E, 0xBF0B, 0xBF94, 0x3EC0, 0xBF14,
      0x3FCD, 0x3F37, 0xBFF9, 0x3F4B, 0x3F93, 0x3F9B, 0x3FEA, 0xBF04
  };
  uint16_t expected[] = {
      0x3F66, 0x40D6, 0xC184, 0xC0FA, 0x4057, 0x4103, 0xC138, 0x4156
  };
  // clang-format on

  Tensor* output = setup_and_run(
      M, N, K, gs, ql_host, qh_host, scale_codes, steps_host, A_host);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(0), M);
  EXPECT_EQ(output->size(1), N);
  check_bf16_output(output, expected, M * N, 0.5f);
}

// LargeKGroupSize16: M=1, N=2, K=256, gs=16, symmetric — a larger-K decode case
// (16 groups) exercising the multi-iteration warp loop on the gs=16 path.
TEST_F(AOTITorchInt6PlainMMTest, LargeKGroupSize16) {
  int64_t M = 1, N = 2, K = 256, gs = 16;
  int64_t ng = K / gs; // 16

  // clang-format off
  uint8_t ql_host[] = {
      214, 227, 226,  92, 214, 223, 203, 136, 116, 138,   8, 212,
       63, 195, 178, 145, 137, 223, 224, 221, 115, 119, 220,  73,
       93, 185, 148, 181, 245, 132, 132, 208,  77,   0, 177, 115,
       21,  42, 237,  70, 130,  36,  92,  71,  59,  38, 124, 112,
      243,   2, 238,  62, 246,  29,  92, 225,  98, 246, 104,  63,
      159,   8,  69, 121, 122, 102, 197,  54, 248, 232,  96,  69,
      214, 125, 104,  83,  17, 161, 205, 133, 218, 150, 220, 143,
       11, 234,   4, 244,  82, 166,  10, 188, 255, 183, 197, 248,
      176, 192, 180, 202, 174, 124, 132, 238, 130, 236, 178, 183,
      146, 213,  64, 190,  75, 211, 232, 226, 103,  38,  55, 228,
      137, 211, 248, 116, 139,  86, 139,  38,  37,  33,  15, 167,
      182,  17, 246,  46,  43, 251, 227,  79,  13, 201, 199,  66,
       75,  14, 192,  81,  46,   9, 164, 187, 177, 203,  98, 103,
      226, 111, 232, 105, 144,  48, 101, 146, 207, 253,  11, 209,
      119,  74, 148, 132, 222,  99,  96, 227, 131,  30,  17,  91,
      112,  45,  20, 238, 239,  50, 201, 213, 169, 185,  64, 150,
      170, 213, 252, 183,  54,  19, 123,  58, 151, 172,  46, 139,
      196, 207,  41, 237, 140, 209, 172,  70, 227,  38, 207,  36,
      130,  64,  98, 254,  44, 106, 224,  29, 101,   6, 126, 158,
      116, 253, 247,  60,  57, 131,  88,  16, 185, 172, 249, 161,
      187, 206, 171, 113,  44,  38, 228, 158, 254,  47, 209, 170,
      153, 252, 205, 145
  };
  uint8_t qh_host[] = {
      140,  45, 119, 122, 178, 254,  20, 229, 112, 130,  14, 199,
       76,  71, 132,  92, 121,  73, 218, 251,   4,  27, 217,  59,
      143, 112,  77,  43,  32,  40,  73,  87, 157, 240, 206, 245,
      237, 110,   2,   2, 144, 175, 198, 234, 149, 229,  35,  47,
      215, 128, 227, 118, 149,  51, 248, 129, 174,  69, 188,  53,
      113, 117,   1,  98, 124,  21, 176,  12,  46, 238, 157,  84,
       27, 209,  35, 212, 175,  11,  63,  93,  54, 134, 251,  12,
      226, 243, 222, 157, 183,  58,  55, 163, 146,  15, 146,  92,
      213, 133, 146, 101,  56, 150,  73, 191,  40,  99, 216, 254,
      191, 116, 119,  55,  42, 172, 112, 199, 209, 170,  68, 161,
      146,  45, 112, 159, 204,  40, 125, 176
  };
  int8_t scale_codes[] = {
       -54,    1,  -89,  -67,  -73,   47,    8,   66,  -14,  -30,  -15,  -13,
      -118,  -67,  106,  127,   -2,   55,   48, -120,  -31,  111,  -26,   26,
        30,   35,  -76,    0,  -70,  -47,  127,  -33
  };
  uint16_t steps_host[] = {
      0x3A60, 0x3A7E
  };
  uint16_t A_host[] = {
      0x3FAB, 0x3F78, 0xBF81, 0xBFA2, 0x3EFC, 0xBFCF, 0xBF87, 0x3F2D,
      0x3B0C, 0x3D2F, 0xBF28, 0x3F26, 0xBF51, 0x3F9E, 0x3E5C, 0x3F7C,
      0x3EBB, 0x3FF8, 0xBD43, 0x3F75, 0xBF50, 0xBCDD, 0xBE29, 0xBF39,
      0x3F05, 0x3FC5, 0xBF9E, 0x3E5B, 0xBF2F, 0xBF4D, 0xBE4A, 0xBEB9,
      0x3D74, 0xBEA5, 0x3F7E, 0xBE1D, 0xBF8B, 0xBF6E, 0xBF37, 0xC016,
      0xBEC0, 0xBF3F, 0x3F8B, 0x402A, 0xBD27, 0x3F9C, 0x3F11, 0xBFBB,
      0xBEC0, 0xBCA1, 0xBEAD, 0x3F7B, 0xBF62, 0xBF85, 0x3F87, 0xBBEC,
      0x3EFC, 0xBE27, 0xBF6F, 0x4001, 0x3F82, 0xBFC7, 0x3FBF, 0x3E0C,
      0x3DEE, 0x3EB9, 0xC024, 0x3E74, 0xBE43, 0xBEBB, 0xBE0A, 0x3EE9,
      0xBE81, 0xBF16, 0xBEA2, 0xBF6C, 0x3DD0, 0xBEB2, 0xBFC2, 0xBF92,
      0xBF06, 0x3F4D, 0x400A, 0x3F8B, 0xBF38, 0xC021, 0xBFD4, 0x3F5E,
      0xBE8C, 0xBF95, 0xBE28, 0xBF46, 0xBF19, 0xBF3E, 0xBFB2, 0xBDD0,
      0xBF79, 0xBF4A, 0x3FD0, 0xBF9A, 0xBCB6, 0xBF8F, 0xBEFA, 0x3FC2,
      0xBF3C, 0x3F5C, 0x3EB1, 0xBEE0, 0x3F7A, 0xBF08, 0x3E0B, 0x3F9C,
      0xBE57, 0xBF08, 0x3FD5, 0x3EC9, 0x3F40, 0xBF6E, 0xBF10, 0xBF44,
      0xBE02, 0x3EDD, 0x3FC3, 0xBF90, 0x3F91, 0x3F26, 0x3F6B, 0xBF88,
      0xBFE4, 0x3F1D, 0x402A, 0xBE87, 0xBE6A, 0xBF53, 0xBE86, 0xBF65,
      0xBDA2, 0x3BF1, 0xBECD, 0xBF5B, 0x3E66, 0xBFF9, 0xBE7D, 0xBF09,
      0x3EB3, 0x3F8C, 0xBEA4, 0x3FBF, 0xBDB3, 0x3E5C, 0xBE97, 0xBFEC,
      0xBE19, 0xBF52, 0x3D22, 0x3EB5, 0xBF6A, 0x3F66, 0xBF83, 0xBF51,
      0x3EAD, 0x4005, 0x3FA2, 0xBF32, 0x3F02, 0x3F2F, 0xBFC1, 0xC007,
      0xBEAE, 0xBEC2, 0x3F22, 0xBE39, 0xBF75, 0x3F00, 0x3DC5, 0xBF70,
      0x3F63, 0xC024, 0x3F6B, 0xBF4B, 0x3F5F, 0xBF1D, 0x3F94, 0x3D1D,
      0x3FB4, 0x3FD0, 0x3E64, 0x3F53, 0xBF5E, 0xBF85, 0x3F89, 0x3DE3,
      0xBF1A, 0x3EE0, 0x3FE1, 0x3F02, 0xBEC5, 0xBF01, 0x3F3E, 0xBF5F,
      0xBEBF, 0x3FF1, 0xBF0A, 0x3F6B, 0x3F28, 0xBE5D, 0xBDF4, 0x3F27,
      0xBF4D, 0x3E61, 0xBF64, 0xBFF1, 0xBCC4, 0xBDF5, 0xBE1D, 0x3E3D,
      0x3F1F, 0xBF8F, 0xBEE6, 0x3F20, 0x3E82, 0x3F51, 0x3F49, 0x3EDE,
      0xBDDF, 0xBF82, 0x3FE1, 0xBF02, 0xBF42, 0x3F91, 0xBF96, 0xBEBE,
      0xBE0C, 0xBF6C, 0xBF2D, 0xBFFA, 0x3F73, 0xBFE5, 0x3F50, 0xBE8A,
      0x3EC4, 0xBD65, 0xBF2A, 0xBF8F, 0x3F97, 0xBFB4, 0x3FD8, 0x3D74,
      0x3F0A, 0xBE81, 0xBF11, 0xBDC5, 0xBF5C, 0xBDEB, 0x3F78, 0xBE60
  };
  uint16_t expected[] = {
      0x418B, 0xC1A1
  };
  // clang-format on

  Tensor* output = setup_and_run(
      M, N, K, gs, ql_host, qh_host, scale_codes, steps_host, A_host);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(0), M);
  EXPECT_EQ(output->size(1), N);
  check_bf16_output(output, expected, M * N, 0.5f);
}

TEST_F(AOTITorchInt6PlainMMTest, NullInputHandling) {
  int64_t M = 2, K = 128, N = 64, gs = 16;
  int64_t ng = K / gs;

  Tensor* A = create_bf16({M, K});
  Tensor* ql = create_uint8({N, K / 2});
  Tensor* qh = create_uint8({N, K / 4});
  Tensor* scale = create_int8({N, ng});
  Tensor* steps = create_bf16({N, 1});
  Tensor* output = nullptr;

  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(nullptr, ql, qh, scale, steps, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, nullptr, qh, scale, steps, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, ql, nullptr, scale, steps, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, ql, qh, nullptr, steps, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, ql, qh, scale, nullptr, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, ql, qh, scale, steps, gs, nullptr),
      Error::InvalidArgument);
}
