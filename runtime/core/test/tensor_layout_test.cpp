/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/tensor_layout.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorLayout;

TEST(TestTensorLayout, Ctor) {
  std::array<int32_t, 2> sizes = {1, 2};
  std::array<uint8_t, 2> dim_order = {0, 1};
  Span<const int32_t> sizes_span = {sizes.data(), sizes.size()};
  Span<const uint8_t> dim_order_span = {dim_order.data(), dim_order.size()};

  Result<const TensorLayout> layout_res =
      TensorLayout::create(sizes_span, dim_order_span, ScalarType::Float);
  EXPECT_TRUE(layout_res.ok());

  TensorLayout layout = layout_res.get();
  EXPECT_EQ(layout.scalar_type(), ScalarType::Float);

  EXPECT_EQ(layout.sizes().size(), sizes_span.size());
  EXPECT_EQ(layout.sizes()[0], sizes_span[0]);
  EXPECT_EQ(layout.sizes()[1], sizes_span[1]);

  EXPECT_EQ(layout.dim_order().size(), dim_order_span.size());
  EXPECT_EQ(layout.dim_order()[0], dim_order_span[0]);
  EXPECT_EQ(layout.dim_order()[1], dim_order_span[1]);

  EXPECT_EQ(layout.nbytes(), 8);
}

TEST(TestTensorLayout, Ctor_InvalidDimOrder) {
  std::array<int32_t, 1> sizes = {2};
  std::array<uint8_t, 1> dim_order = {1};
  Span<const int32_t> sizes_span = {sizes.data(), sizes.size()};
  Span<const uint8_t> dim_order_span = {dim_order.data(), dim_order.size()};

  Result<const TensorLayout> layout_res =
      TensorLayout::create(sizes_span, dim_order_span, ScalarType::Float);
  EXPECT_EQ(layout_res.error(), Error::InvalidArgument);
}

TEST(TestTensorLayout, Ctor_InvalidSizes) {
  std::array<int32_t, 1> sizes = {-1};
  std::array<uint8_t, 1> dim_order = {0};
  Span<const int32_t> sizes_span = {sizes.data(), sizes.size()};
  Span<const uint8_t> dim_order_span = {dim_order.data(), dim_order.size()};

  Result<const TensorLayout> layout_res =
      TensorLayout::create(sizes_span, dim_order_span, ScalarType::Float);
  EXPECT_EQ(layout_res.error(), Error::InvalidArgument);
}

TEST(TestTensorLayout, Ctor_SizesDimOrderMismatch) {
  std::array<int32_t, 1> sizes = {2};
  std::array<uint8_t, 2> dim_order = {0, 1};
  Span<const int32_t> sizes_span = {sizes.data(), sizes.size()};
  Span<const uint8_t> dim_order_span = {dim_order.data(), dim_order.size()};

  Result<const TensorLayout> layout_res =
      TensorLayout::create(sizes_span, dim_order_span, ScalarType::Float);
  EXPECT_EQ(layout_res.error(), Error::InvalidArgument);
}
