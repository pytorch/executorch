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

// W6A8 dp4a matvec shim for packed-INT6 decode (CudaPackedInt6Tensor layout,
// GGUF Q6_K). The 6-bit weight is split into two planes plus a per-group scale;
// there is NO zero tensor (Q6_K is symmetric, the -32 offset is applied in the
// kernel):
//   ql    : [N, K/2] uint8 — low-nibble plane, nibble-packed even/odd
//   qh    : [N, K/4] uint8 — high-2-bit plane, 4 values/byte (per 32-weight
//           chunk: hi_even_packed[4] then hi_odd_packed[4])
//   scale : [N, K//gs] bf16 — per-group scales (row-major)
//
// Expected outputs are generated from the export-path reference
// _dequant_matmul_int6 (backends/cuda/quantize_op_dispatch/int6_dispatch.py):
//   w[n, k] = q[n, k] * scale[n, k//gs]; out = A @ w^T (q symmetric, in
//   [-32,31]).
// The kernel runs W6A8 (it also quantizes activations to int8), so a 0.5 atol
// absorbs the activation-quant noise (same tolerance as the int4/int8 tests).
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

  // Upload raw bytes to a CUDA tensor.
  void upload(Tensor* t, const void* host_data, size_t bytes) {
    cudaMemcpy(t->data_ptr(), host_data, bytes, cudaMemcpyHostToDevice);
  }

  // Download CUDA tensor to host buffer.
  void download(const Tensor* t, void* host_data, size_t bytes) {
    cudaMemcpy(host_data, t->data_ptr(), bytes, cudaMemcpyDeviceToHost);
  }

  // Run the shim and return the output tensor (asserts success).
  Tensor*
  run(Tensor* A, Tensor* ql, Tensor* qh, Tensor* scale, int64_t group_size) {
    Tensor* output = nullptr;
    AOTITorchError error =
        aoti_torch_cuda_int6_plain_mm(A, ql, qh, scale, group_size, &output);
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

  // Upload data and run the shim. ql/qh are uint8; scale/A are bf16.
  Tensor* setup_and_run(
      int64_t M,
      int64_t N,
      int64_t K,
      int64_t gs,
      const uint8_t* ql_host,
      const uint8_t* qh_host,
      const uint16_t* scale_host,
      const uint16_t* A_host) {
    int64_t ng = K / gs;
    Tensor* A = create_bf16({M, K});
    Tensor* ql = create_uint8({N, K / 2});
    Tensor* qh = create_uint8({N, K / 4});
    Tensor* scale = create_bf16({N, ng});
    EXPECT_NE(A, nullptr);
    EXPECT_NE(ql, nullptr);
    EXPECT_NE(qh, nullptr);
    EXPECT_NE(scale, nullptr);

    upload(A, A_host, static_cast<size_t>(M) * K * sizeof(uint16_t));
    upload(ql, ql_host, static_cast<size_t>(N) * (K / 2) * sizeof(uint8_t));
    upload(qh, qh_host, static_cast<size_t>(N) * (K / 4) * sizeof(uint8_t));
    upload(scale, scale_host, static_cast<size_t>(N) * ng * sizeof(uint16_t));

    return run(A, ql, qh, scale, gs);
  }
};

// Q6KGroupSize16: M=2, N=4, K=64, gs=16, symmetric (no zero), q in [-32,31].
// The canonical GGUF Q6_K shape (group_size=16).
TEST_F(AOTITorchInt6PlainMMTest, Q6KGroupSize16) {
  int64_t M = 2, N = 4, K = 64, gs = 16;

  // clang-format off
  uint8_t ql_host[] = {
    249, 176, 113, 205, 113, 130, 205, 208, 208, 220,  36,  28,  90, 117,  20, 139,
     24,  99,  43,   2, 253, 112, 107, 185, 154, 203, 229, 119,  15,   8, 139,  95,
    117,  50,  27,  48, 120,  65,  40, 224, 147, 165, 182, 177, 210, 160, 239, 192,
    136,  20, 241, 201,  43,  56,  64,  34, 219, 104,  39, 103,  79,  70, 196, 157,
    193,  90,  70,  26,  31,  78, 234,  55,  53,  19, 198,  24,  26,  71,  88, 181,
    205, 210,  95, 167,  16,  80, 183,  76, 106,  66,  44, 124,  17, 197,  49, 227,
     46,  51,   2, 185,  46, 243, 128,  59,  39, 121,  45, 252, 221,  98, 155, 170,
     27,  31, 108,  91, 235, 129, 177, 104,  44,  22, 110, 142, 169, 226, 255, 217
  };
  uint8_t qh_host[] = {
     21, 230,  10,  92,  55, 212,  46,  90, 227,  91,  52,  88,  49, 132, 203,  60,
    255, 132, 109, 173,   8,  49, 181, 163, 130, 224, 227,  13, 216,  86, 234, 219,
    180, 142, 137, 139,  87, 161, 244,  72, 109,  20, 107, 165,  31,  47,  99,  59,
    215, 173,   1, 159, 180,  83, 227, 190,  15, 222,  95, 108, 117, 157, 225, 105
  };
  uint16_t scale_host[] = {
    0x3C6E, 0xBCF6, 0x3CC3, 0xBB88, 0xBD0C, 0x3D5A, 0x3B40, 0x3D43, 0xBB71, 0xBD6A, 0x3D16, 0xBCC3,
    0xBC1E, 0x3D2A, 0xBCC3, 0xBD37
  };
  uint16_t A_host[] = {
    0x3F5C, 0xBF3E, 0x0000, 0xBC33, 0xBE9A, 0x3CAA, 0x3F7A, 0xBF94, 0xC016, 0xBFF6, 0x0000, 0x3E71,
    0xBFD3, 0x3F5E, 0xBF96, 0x3E2A, 0x4023, 0x3EC0, 0x3E90, 0xC00C, 0x3F84, 0xBEEA, 0xBE32, 0x3F71,
    0x0000, 0x3EC9, 0xBEE2, 0x3EE8, 0x3F30, 0xBECB, 0x3F1F, 0xBF2F, 0xBF2A, 0x3F01, 0x3F11, 0x3F88,
    0xBF6A, 0x3FD4, 0xBDD5, 0x3F8F, 0xBF5F, 0xBEBA, 0xBF24, 0xBF45, 0xBF3F, 0x3E51, 0xBE7D, 0xBF35,
    0x3E73, 0x3F1B, 0x3F34, 0x3EA2, 0xBF13, 0xBF4F, 0xBEE2, 0x4006, 0x3F37, 0x3EC5, 0x3F9F, 0xBD79,
    0x3F21, 0xBF0C, 0xBEA9, 0x3FF2, 0x3F55, 0x3FD6, 0x3FAB, 0x3F89, 0xBDA1, 0x3EDD, 0xBF8D, 0xBE4F,
    0xC005, 0xBFBD, 0xBF59, 0x3CD7, 0x3E07, 0xBEEA, 0x3EAC, 0x4038, 0x3F7E, 0xBE4B, 0xBE3A, 0xBF99,
    0xBFCC, 0x3EF0, 0xBF84, 0xBEE8, 0xBF6E, 0xBC97, 0xBF57, 0xBF3F, 0x3FD7, 0xBFB5, 0x3F0C, 0x3E3F,
    0x3F77, 0xBE45, 0x3FAA, 0x3FE1, 0x3D9C, 0x3F8F, 0xBF38, 0xBF1F, 0xBF07, 0xBE94, 0xBF58, 0xBF85,
    0x3FCE, 0x3F2A, 0x3EAC, 0xBF45, 0x3DC4, 0x3E9E, 0xBF9C, 0x3F0A, 0x3E8F, 0x3EA7, 0xBEFB, 0xBE65,
    0xBFB1, 0xBF58, 0xBF88, 0x3EC2, 0xC008, 0x3F7C, 0xBFFC, 0xBF66
  };
  uint16_t expected[] = {
    0x3F46, 0xC02B, 0x40C5, 0xBED9, 0xBECA, 0x4098, 0x3F96, 0x3F19
  };
  // clang-format on

  Tensor* output =
      setup_and_run(M, N, K, gs, ql_host, qh_host, scale_host, A_host);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(0), M);
  EXPECT_EQ(output->size(1), N);
  check_bf16_output(output, expected, M * N, 0.5f);
}

// LargeKGroupSize16: M=1, N=2, K=256, gs=16, symmetric — a larger-K decode case
// (16 groups) exercising the multi-iteration warp loop on the gs=16 path.
TEST_F(AOTITorchInt6PlainMMTest, LargeKGroupSize16) {
  int64_t M = 1, N = 2, K = 256, gs = 16;

  // clang-format off
  uint8_t ql_host[] = {
     69,  12, 100, 182, 132,  79,  45, 206, 141, 218,  39, 249, 136, 245,  75, 210,
     18, 150,  51, 178, 183, 119, 174, 151, 235,  77,  75, 247,  29, 241,  55, 154,
     12, 189,  29,  93,  92, 153,  20,  52,  67, 219,  12, 178,  99, 207,  12, 151,
      5, 133,  30, 141,  56, 234,  26, 101,  93, 150,  46, 101,  80,  30,  33, 153,
    240,  83, 103, 193,  72, 152, 248,  85,  69,  52, 240, 168,   4,  81, 134,  98,
    101, 106, 122, 199, 212, 244, 190, 139,  33,  62,   6, 147, 243, 106, 105, 196,
    120,  49, 123,  17,  38, 205, 200,  90,  10, 248, 177, 182,   9, 195,  90,   9,
    127, 194, 250, 109, 105, 141, 182,  53,  35, 162, 151, 192, 134, 134, 246, 198,
    202, 191,  86,  93, 221, 185,  60, 230, 242, 167, 247, 189,  35, 210, 188, 146,
      8, 218,  95, 120, 119,  39, 177, 110, 158, 144,   0,  36,  69, 219, 134,  94,
     29,  25,  81, 213, 207, 185, 206,  89, 113,   1,  50,  59, 238,  29,  69, 128,
     97,  97, 229, 181, 211, 253, 157, 118,  71, 232,  63,  21, 171,  62, 115,  78,
      3, 109, 188, 187, 172,   5, 144, 190,  60, 214, 171, 194, 232,   6, 192, 189,
    136,  45, 201,  26, 110, 239,  63, 229, 197,  85,  25, 121, 147,  63, 227,  20,
     30,  66, 228, 231, 197,  90,  65, 116, 255,  50,  51,  88, 142,  60, 112,  10,
     18, 192,  52, 144, 148,  19, 197,  32,   3, 157, 152,  52, 176,  31,  38, 242
  };
  uint8_t qh_host[] = {
    235,  21, 174, 144, 160, 216, 229,  90,  25, 104, 128, 211,  93, 165, 189, 219,
     87, 210, 115, 144,  79,  31, 166, 108, 199,  41,  50,  92,  21,  45, 124, 158,
    142, 126,   0, 139,  23,  77, 180, 181, 218, 246,  98, 252,  50, 141,  10,  82,
     82,  31, 128, 233, 230, 216, 156, 120, 193, 161,  94, 122,  62,  85, 233,   8,
    199, 237, 102, 124, 105, 252,  43,  58,  34, 218,  77, 242, 219,  85,  16, 221,
    102,  49,  77, 226,  23,  30, 142,  36, 110,  63,  97,  59, 164, 214, 221, 103,
    253,  67, 106, 140,  18,  75, 207, 144,  21,  18, 108,  84, 110, 217,  45, 114,
    180, 170,   6, 111, 131, 171, 200, 246,  55, 206,  40, 185,  16, 114,  54,  62
  };
  uint16_t scale_host[] = {
    0x3CF1, 0x3C5B, 0x3B89, 0x3B53, 0x3865, 0xBD3E, 0x3D2F, 0x3AD1, 0x3CC6, 0x3D06, 0xBCFE, 0x3BDD,
    0x3D60, 0x3BD0, 0xBD1A, 0x3D1F, 0xBBBA, 0x3D58, 0x3CD5, 0xBCD3, 0x3BB7, 0x3CF3, 0x3D05, 0x3D0B,
    0x3D42, 0xBBF0, 0x3CC5, 0xBC17, 0xBD73, 0xBC09, 0xBC01, 0xBD24
  };
  uint16_t A_host[] = {
    0xBF33, 0xBF48, 0xBE27, 0x3F25, 0xBFF5, 0x3F5C, 0xBFCE, 0xBF36, 0x3DFA, 0x3EE3, 0xBF64, 0x3E14,
    0xBF41, 0x3E5C, 0x3ED3, 0xBF93, 0xBF45, 0x3BC7, 0xBEF0, 0x3D95, 0xBF20, 0x3E4D, 0xBEA8, 0xBF49,
    0x3F65, 0xBF75, 0xBEA2, 0x3F35, 0x3DE0, 0xBDB1, 0xBEA7, 0xBF5B, 0x3F7F, 0x3F47, 0x3FA4, 0x3FB6,
    0xBE20, 0xBFDE, 0xBD38, 0xBFC6, 0x3F22, 0xBF91, 0xBEA8, 0xBFEA, 0x3FA0, 0xBFAB, 0x3F78, 0xBFAC,
    0x3EA4, 0x3FB3, 0xBF88, 0xBF3B, 0xBEA4, 0x3EDF, 0x3F01, 0x3E7A, 0xBF5F, 0xBD3E, 0x3FA3, 0xBF68,
    0xBF32, 0x3EC0, 0xBF59, 0x3EE9, 0xBEB9, 0xBEC4, 0x3F1E, 0xBE8A, 0x3FBE, 0x3F19, 0x3FC2, 0xC00B,
    0xBEF4, 0xBF45, 0xBEC8, 0x3FC7, 0x3F09, 0x3F97, 0x3F43, 0xBF47, 0x3FCF, 0x3E26, 0x3E10, 0xBEA9,
    0x3EA2, 0x3FAE, 0x3F3F, 0x3E93, 0xBFB6, 0x3FCA, 0x3F70, 0x3FD6, 0x3E58, 0xBF17, 0x3FB2, 0xBE16,
    0x4006, 0x3FC1, 0x3F7D, 0x3F3E, 0xBE03, 0x3ED5, 0x3F0A, 0xBE95, 0xBE89, 0x3F8E, 0x3EF0, 0x3FBB,
    0x3F83, 0xBFCB, 0x3E18, 0x3FA8, 0x3F60, 0x3F1D, 0xBFB4, 0x3FB8, 0xBDB3, 0xBF77, 0xBEBC, 0x3E68,
    0x3EAC, 0x3F54, 0x3F72, 0xC01B, 0x3E4C, 0x3FA9, 0xBDCC, 0xBE59, 0xBF8D, 0xBE29, 0x3E80, 0x3FB9,
    0xBFD0, 0x3E11, 0xBF42, 0xBECE, 0xBE42, 0x4016, 0x3C98, 0x3E5B, 0x3F43, 0x3FB1, 0x3F30, 0xBE69,
    0x3F2C, 0x3F4A, 0x3F43, 0x3FAB, 0x3E4C, 0xBF9C, 0xBEF7, 0xBF87, 0x3DA9, 0x3F2E, 0xBEA8, 0xBF4A,
    0x3F80, 0xBF1E, 0xBE81, 0x3EA5, 0x3F0E, 0xBF50, 0x3EA4, 0x3FD3, 0xBE3C, 0x3F8D, 0xBF38, 0xBEB3,
    0x3E86, 0x3F79, 0xBF77, 0x3E26, 0x3F6E, 0x3DDF, 0xBCB2, 0x3F92, 0xBE11, 0xBF0E, 0xBFFE, 0xBF6A,
    0x3FA0, 0x0000, 0xBF84, 0x3FA7, 0x3F23, 0x3F8F, 0xBF90, 0xBF2F, 0x3F8A, 0x0000, 0xBDA4, 0x3F6A,
    0x3E9D, 0x3FAB, 0xBEDB, 0x3F06, 0x3EFB, 0xBF86, 0x3DAD, 0xBE1C, 0xBF85, 0x3F65, 0xBF5C, 0xBE89,
    0x3EC4, 0x3F85, 0x3EF7, 0x3C47, 0x3E98, 0x3EFB, 0x3DC9, 0x3D1B, 0xBECD, 0x4007, 0x3ED0, 0xBF28,
    0x3F99, 0x3E9F, 0xBF7A, 0x3EBD, 0xBEEE, 0xBF1C, 0xBED0, 0xBF01, 0x3F76, 0xBE8A, 0xBF8C, 0x3EDD,
    0x3FE6, 0x3ECA, 0x3F45, 0xBF64, 0xBE8F, 0x3FC7, 0x3FD4, 0xBF2D, 0x3F0C, 0x3F58, 0x3F45, 0x3E8B,
    0x3A08, 0x3F9E, 0x4004, 0x3F9D, 0xBFDE, 0xBF69, 0xBF8E, 0xBF0B, 0x3F89, 0x3DFA, 0xBF91, 0xC019,
    0x3DAA, 0x3F09, 0x3F69, 0x3F3E
  };
  uint16_t expected[] = {
    0xC196, 0x40F7
  };
  // clang-format on

  Tensor* output =
      setup_and_run(M, N, K, gs, ql_host, qh_host, scale_host, A_host);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->size(0), M);
  EXPECT_EQ(output->size(1), N);
  check_bf16_output(output, expected, M * N, 0.5f);
}

TEST_F(AOTITorchInt6PlainMMTest, NullInputHandling) {
  int64_t M = 2, K = 128, N = 64, gs = 16;

  Tensor* A = create_bf16({M, K});
  Tensor* ql = create_uint8({N, K / 2});
  Tensor* qh = create_uint8({N, K / 4});
  Tensor* scale = create_bf16({N, K / gs});
  Tensor* output = nullptr;

  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(nullptr, ql, qh, scale, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, nullptr, qh, scale, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, ql, nullptr, scale, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, ql, qh, nullptr, gs, &output),
      Error::InvalidArgument);
  EXPECT_EQ(
      aoti_torch_cuda_int6_plain_mm(A, ql, qh, scale, gs, nullptr),
      Error::InvalidArgument);
}
