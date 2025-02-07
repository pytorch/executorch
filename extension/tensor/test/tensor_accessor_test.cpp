/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_accessor.h>

#include <gtest/gtest.h>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::make_tensor_accessor;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorAccessor;

class TensorAccessorTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    executorch::runtime::runtime_init();
  }
};

TEST_F(TensorAccessorTest, From1DTensor) {
  constexpr int32_t kN = 16;
  std::vector<uint8_t> data(kN, 0);
  for (int32_t i = 0; i < kN; i++) {
    data[i] = i;
  }

  auto tensor =
      make_tensor_ptr({kN}, data.data(), executorch::aten::ScalarType::Byte);
  auto tensor_accessor = make_tensor_accessor<uint8_t, 1>(*tensor.get());
  EXPECT_TRUE(tensor_accessor.ok());
  for (int32_t i = 0; i < kN; i++) {
    EXPECT_EQ(tensor_accessor.get()[i], i);
  }
}

int32_t
value_at_pos_in_4d_int_tensor(int32_t n, int32_t c, int32_t h, int32_t w) {
  // just encode the position into the value, assuming dimensions fit in 8 bits
  return (n << 24) | (c << 16) | (h << 8) | w;
}

void check_4d_int_tensor_accessor(
    TensorAccessor<int32_t, 4> accessor,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W) {
  for (int32_t n = 0; n < N; n++) {
    for (int32_t c = 0; c < C; c++) {
      for (int32_t h = 0; h < H; h++) {
        for (int32_t w = 0; w < W; w++) {
          EXPECT_EQ(
              accessor[n][c][h][w], value_at_pos_in_4d_int_tensor(n, c, h, w));
        }
      }
    }
  }
}

TEST_F(TensorAccessorTest, From4DTensor) {
  constexpr int32_t kN = 2;
  constexpr int32_t kC = 8;
  constexpr int32_t kH = 4;
  constexpr int32_t kW = 6;
  std::vector<int32_t> data(kN * kC * kH * kW, 0);
  size_t idx = 0;
  for (int32_t n = 0; n < kN; n++) {
    for (int32_t c = 0; c < kC; c++) {
      for (int32_t h = 0; h < kH; h++) {
        for (int32_t w = 0; w < kW; w++) {
          data[idx++] = value_at_pos_in_4d_int_tensor(n, c, h, w);
        }
      }
    }
  }

  auto tensor = make_tensor_ptr(
      {kN, kC, kH, kW}, data.data(), executorch::aten::ScalarType::Int);
  auto accessor = make_tensor_accessor<int32_t, 4>(*tensor.get());
  EXPECT_TRUE(accessor.ok());
  check_4d_int_tensor_accessor(accessor.get(), kN, kC, kH, kW);
}

#ifdef USE_ATEN_LIB // Non-contiguous tensor is only allowed in ATen mode.
TEST_F(TensorAccessorTest, FromNonContiguousTensor) {
  constexpr int32_t kN = 2;
  constexpr int32_t kC = 8;
  constexpr int32_t kH = 4;
  constexpr int32_t kW = 6;
  constexpr int32_t kW_padded = 8;
  std::vector<int32_t> data(kN * kC * kH * kW_padded, 0);
  std::array<executorch::aten::SizesType, 4> sizes = {kN, kC, kH, kW};
  std::array<executorch::aten::StridesType, 4> strides = {
      kC * kH * kW_padded,
      1, // channel last
      kC * kW_padded, // width is padded
      kC};

  size_t idx = 0;
  for (int32_t n = 0; n < kN; n++) {
    for (int32_t h = 0; h < kH; h++) {
      for (int32_t w = 0; w < kW_padded; w++) {
        for (int32_t c = 0; c < kC; c++) {
          data[idx++] = value_at_pos_in_4d_int_tensor(n, c, h, w);
        }
      }
    }
  }

  auto tensor = at::from_blob(
      data.data(), sizes, strides, at::TensorOptions().dtype(at::kInt));
  auto accessor = make_tensor_accessor<int32_t, 4>(tensor);
  EXPECT_TRUE(accessor.ok());
  check_4d_int_tensor_accessor(accessor.get(), kN, kC, kH, kW);
}
#endif // ifdef USE_ATEN_LIB

TEST_F(TensorAccessorTest, FailOnIncorrectDtypeOrRank) {
  constexpr int32_t kN = 16;
  std::vector<float> data(kN, 0);
  auto tensor = make_tensor_ptr({kN}, data.data());

  // Tensor has rank 1 but creating accessor with rank 2.
  auto fail1 = make_tensor_accessor<float, 2>(*tensor.get());
  EXPECT_FALSE(fail1.ok());

  // Tensor has dtype float but creating accoessor with dtype uint8_t.
  auto fail2 = make_tensor_accessor<uint8_t, 1>(*tensor.get());
  EXPECT_FALSE(fail2.ok());
}

#ifndef USE_ATEN_LIB // Dim order is only defined for portable Tensor
TEST_F(TensorAccessorTest, FailOnNonTrivialDimOrder) {
  constexpr int32_t kN = 8;
  constexpr int32_t kM = 16;
  std::vector<float> data(kN * kM, 0);
  auto tensor = make_tensor_ptr(
      {kN, kM},
      data.data(),
      /*dim_order=*/{1, 0},
      /*strides=*/{1, kN});

  // Non trivial dim order is not supported.
  auto fail = make_tensor_accessor<float, 2>(*tensor.get());
  EXPECT_FALSE(fail.ok());
}
#endif // ifndef USE_ATEN_LIB
