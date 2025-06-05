/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>

#include <numeric>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <gtest/gtest.h>

using executorch::runtime::dim_order_to_stride;
using executorch::runtime::Error;
using executorch::runtime::is_channels_last_dim_order;
using executorch::runtime::is_contiguous_dim_order;
using executorch::runtime::stride_to_dim_order;

namespace {
void check_strides_eq(
    executorch::aten::ArrayRef<executorch::aten::StridesType> strides_a,
    executorch::aten::ArrayRef<executorch::aten::StridesType> strides_b) {
  for (const auto i : c10::irange(strides_a.size())) {
    EXPECT_EQ(strides_a[i], strides_b[i]);
  }
}

void check_dim_order_eq(
    executorch::aten::ArrayRef<executorch::aten::DimOrderType> dim_order_a,
    executorch::aten::ArrayRef<executorch::aten::DimOrderType> dim_order_b) {
  for (const auto i : c10::irange(dim_order_a.size())) {
    EXPECT_EQ(dim_order_a[i], dim_order_b[i]);
  }
}
} // namespace

TEST(DimOrderUtilTest, DimOrderToStride) {
  executorch::aten::SizesType sizes_1[1] = {5};
  executorch::aten::SizesType dim_order_1[1] = {0};
  executorch::aten::SizesType strides_1[1] = {0};
  executorch::aten::SizesType expected_strides_1[1] = {1};
  auto error = dim_order_to_stride(sizes_1, dim_order_1, 1, strides_1);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_1, 1}, {expected_strides_1, 1});

  executorch::aten::SizesType sizes_2[2] = {2, 5};
  executorch::aten::SizesType dim_order_2[2] = {0, 1};
  executorch::aten::SizesType strides_2[2] = {0, 0};
  executorch::aten::SizesType expected_strides_2[2] = {5, 1};
  error = dim_order_to_stride(sizes_2, dim_order_2, 2, strides_2);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_2, 2}, {expected_strides_2, 2});

  dim_order_2[0] = 1;
  dim_order_2[1] = 0;
  expected_strides_2[0] = 1;
  expected_strides_2[1] = 2;
  error = dim_order_to_stride(sizes_2, dim_order_2, 2, strides_2);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_2, 2}, {expected_strides_2, 2});

  executorch::aten::SizesType sizes_3[3] = {2, 5, 7};
  executorch::aten::SizesType dim_order_3[3] = {0, 1, 2};
  executorch::aten::SizesType strides_3[3] = {0, 0, 0};
  executorch::aten::SizesType expected_strides_3[3] = {35, 7, 1};
  error = dim_order_to_stride(sizes_3, dim_order_3, 3, strides_3);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_3, 3}, {expected_strides_3, 3});

  // {0, 2, 1}
  dim_order_3[0] = 0, dim_order_3[1] = 2, dim_order_3[2] = 1;
  // Expected stride {35, 1, 5}
  expected_strides_3[0] = 35, expected_strides_3[1] = 1,
  expected_strides_3[2] = 5;
  error = dim_order_to_stride(sizes_3, dim_order_3, 3, strides_3);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_3, 3}, {expected_strides_3, 3});

  // {2, 5, 7}
  // {1, 2, 0}
  dim_order_3[0] = 1, dim_order_3[1] = 2, dim_order_3[2] = 0;
  // Expected stride {35, 1, 5}
  expected_strides_3[0] = 1, expected_strides_3[1] = 14,
  expected_strides_3[2] = 2;
  error = dim_order_to_stride(sizes_3, dim_order_3, 3, strides_3);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_3, 3}, {expected_strides_3, 3});

  executorch::aten::SizesType sizes_4[4] = {2, 5, 7, 8};
  executorch::aten::SizesType dim_order_4[4] = {0, 1, 2, 3};
  executorch::aten::SizesType strides_4[4] = {0, 0, 0, 0};
  executorch::aten::SizesType expected_strides_4[4] = {280, 56, 8, 1};
  error = dim_order_to_stride(sizes_4, dim_order_4, 4, strides_4);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_4, 4}, {expected_strides_4, 4});

  // {2, 5, 7, 8}
  // {0, 2, 3, 1}
  dim_order_4[0] = 0;
  dim_order_4[1] = 2;
  dim_order_4[2] = 3;
  dim_order_4[3] = 1;
  // Expected stride {280, 1, 40, 5}
  expected_strides_4[0] = 280;
  expected_strides_4[1] = 1;
  expected_strides_4[2] = 40;
  expected_strides_4[3] = 5;
  error = dim_order_to_stride(sizes_4, dim_order_4, 4, strides_4);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_4, 4}, {expected_strides_4, 4});

  // {2, 5, 7, 8}
  // {3, 1, 2, 0}
  dim_order_4[0] = 3;
  dim_order_4[1] = 1;
  dim_order_4[2] = 2;
  dim_order_4[3] = 0;
  // Expected stride {1, 14, 2, 70}
  expected_strides_4[0] = 1;
  expected_strides_4[1] = 14;
  expected_strides_4[2] = 2;
  expected_strides_4[3] = 70;
  error = dim_order_to_stride(sizes_4, dim_order_4, 4, strides_4);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_4, 4}, {expected_strides_4, 4});

  executorch::aten::SizesType sizes_5[5] = {2, 5, 7, 8, 9};
  executorch::aten::SizesType dim_order_5[5] = {0, 1, 2, 3, 4};
  executorch::aten::SizesType strides_5[5] = {0, 0, 0, 0, 0};
  executorch::aten::SizesType expected_strides_5[5] = {2520, 504, 72, 9, 1};
  error = dim_order_to_stride(sizes_5, dim_order_5, 5, strides_5);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_5, 5}, {expected_strides_5, 5});

  // {2, 5, 7, 8, 9}
  // {0, 2, 3, 4, 1}
  dim_order_5[0] = 0;
  dim_order_5[1] = 2;
  dim_order_5[2] = 3;
  dim_order_5[3] = 4;
  dim_order_5[4] = 1;
  // Expected stride {2520, 1, 360, 45, 5}
  expected_strides_5[0] = 2520;
  expected_strides_5[1] = 1;
  expected_strides_5[2] = 360;
  expected_strides_5[3] = 45;
  expected_strides_5[4] = 5;
  error = dim_order_to_stride(sizes_5, dim_order_5, 5, strides_5);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_5, 5}, {expected_strides_5, 5});

  // {2, 5, 7, 8, 9}
  // {4, 2, 0, 3, 1}
  dim_order_5[0] = 4;
  dim_order_5[1] = 2;
  dim_order_5[2] = 0;
  dim_order_5[3] = 3;
  dim_order_5[4] = 1;
  // Expected stride {40, 1, 80, 5, 560}
  expected_strides_5[0] = 40;
  expected_strides_5[1] = 1;
  expected_strides_5[2] = 80;
  expected_strides_5[3] = 5;
  expected_strides_5[4] = 560;
  error = dim_order_to_stride(sizes_5, dim_order_5, 5, strides_5);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_5, 5}, {expected_strides_5, 5});

  // Check 0 sized dims
  executorch::aten::SizesType sizes_3_zero[3] = {2, 5, 0};
  executorch::aten::SizesType dim_order_3_zero[3] = {0, 1, 2};
  executorch::aten::SizesType strides_3_zero[3] = {0, 0, 0};
  executorch::aten::SizesType expected_strides_3_zero[3] = {5, 1, 1};
  error =
      dim_order_to_stride(sizes_3_zero, dim_order_3_zero, 3, strides_3_zero);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_3_zero, 3}, {expected_strides_3_zero, 3});

  // {0, 2, 1}
  // {2, 0, 5}
  dim_order_3_zero[0] = 0, dim_order_3_zero[1] = 2, dim_order_3_zero[2] = 1;
  // Expected stride {5, 5, 1}
  expected_strides_3_zero[0] = 5, expected_strides_3_zero[1] = 1,
  expected_strides_3_zero[2] = 5;
  error =
      dim_order_to_stride(sizes_3_zero, dim_order_3_zero, 3, strides_3_zero);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_3_zero, 3}, {expected_strides_3_zero, 3});

  // {2, 0, 1}
  // {0, 2, 5}
  dim_order_3_zero[0] = 2, dim_order_3_zero[1] = 0, dim_order_3_zero[2] = 1;
  // Expected stride {10, 5, 1}
  expected_strides_3_zero[0] = 5, expected_strides_3_zero[1] = 1,
  expected_strides_3_zero[2] = 10;
  error =
      dim_order_to_stride(sizes_3_zero, dim_order_3_zero, 3, strides_3_zero);
  EXPECT_EQ(error, Error::Ok);
  check_strides_eq({strides_3_zero, 3}, {expected_strides_3_zero, 3});
}

TEST(DimOrderUtilTest, StrideToDimOrder) {
  executorch::aten::SizesType strides[3] = {5, 1, 15};
  executorch::aten::DimOrderType dim_order[3] = {0, 0, 0};

  auto error = stride_to_dim_order(strides, 3, dim_order);

  EXPECT_EQ(error, Error::Ok);

  executorch::aten::DimOrderType expected_dim_order[3] = {2, 0, 1};
  check_dim_order_eq(dim_order, expected_dim_order);
}

TEST(DimOrderUtilTest, StrideToDimOrderSameStrides) {
  executorch::aten::SizesType strides[4] = {4, 3, 1, 1};
  executorch::aten::DimOrderType dim_order[4] = {0, 0, 0, 0};

  auto error = stride_to_dim_order(strides, 4, dim_order);
  EXPECT_EQ(error, Error::Ok);

  executorch::aten::DimOrderType expected_dim_order[4] = {0, 1, 2, 3};
  check_dim_order_eq(dim_order, expected_dim_order);
}

TEST(DimOrderUtilTest, IsDefaultDimOrderTest) {
  for (const auto i : c10::irange(1, 7)) {
    std::vector<executorch::aten::DimOrderType> dim_order(i);
    std::iota(dim_order.begin(), dim_order.end(), 0);

    EXPECT_TRUE(is_contiguous_dim_order(dim_order.data(), dim_order.size()));

    // As a bonus, check that is_channels_last returns false
    EXPECT_FALSE(
        is_channels_last_dim_order(dim_order.data(), dim_order.size()));
  }
}

TEST(DimOrderUtilTest, IsDefaultDimOrderFailCasesTest) {
  // Dims is default order but have two elements swapped
  for (const auto i : c10::irange(3, 8)) {
    std::vector<executorch::aten::DimOrderType> dim_order(i);
    std::iota(dim_order.begin(), dim_order.end(), 0);
    std::swap(dim_order[0], dim_order[1]);

    EXPECT_FALSE(is_contiguous_dim_order(dim_order.data(), dim_order.size()));
  }

  // Dims is default order but shifted by 1
  for (const auto i : c10::irange(3, 8)) {
    std::vector<executorch::aten::DimOrderType> dim_order(i);
    for (const auto d : c10::irange(i)) {
      dim_order[d] = (d + 1) % i;
    }

    EXPECT_FALSE(is_contiguous_dim_order(dim_order.data(), dim_order.size()));
  }
}

TEST(DimOrderUtilTest, IsChannelsLastDimOrderTest) {
  executorch::aten::DimOrderType dim_order_4d[4] = {0, 2, 3, 1};
  executorch::aten::DimOrderType dim_order_5d[5] = {0, 2, 3, 4, 1};

  EXPECT_TRUE(is_channels_last_dim_order(dim_order_4d, 4));
  EXPECT_TRUE(is_channels_last_dim_order(dim_order_5d, 5));

  // As a bonus, check that is_default returns false
  EXPECT_FALSE(is_contiguous_dim_order(dim_order_4d, 4));
  EXPECT_FALSE(is_contiguous_dim_order(dim_order_5d, 5));
}

TEST(DimOrderUtilTest, IsChannelsLastDimOrderFailCasesTest) {
  // Non 4D and 5D dim order returns false
  executorch::aten::DimOrderType dim_order_3d[4] = {1, 2, 0};
  executorch::aten::DimOrderType dim_order_6d[6] = {0, 2, 3, 4, 5, 1};

  EXPECT_FALSE(is_channels_last_dim_order(dim_order_3d, 3));
  EXPECT_FALSE(is_channels_last_dim_order(dim_order_6d, 6));

  executorch::aten::DimOrderType dim_order_4d[4] = {0, 3, 2, 1};
  executorch::aten::DimOrderType dim_order_5d[5] = {4, 3, 2, 0, 1};

  EXPECT_FALSE(is_channels_last_dim_order(dim_order_4d, 4));
  EXPECT_FALSE(is_channels_last_dim_order(dim_order_5d, 5));
}
