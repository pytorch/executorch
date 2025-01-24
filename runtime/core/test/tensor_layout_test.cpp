/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/tensor_layout.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::runtime::Span;
using executorch::runtime::TensorLayout;

TEST(TestTensorLayout, Ctor) {
  int32_t sizes[2] = {1, 2};
  uint8_t dim_order[2] = {0, 1};

  Span<const int32_t> sizes_span = {sizes, sizes + 2};
  Span<const uint8_t> dim_order_span = {dim_order, dim_order + 2};

  TensorLayout layout =
      TensorLayout(ScalarType::Float, sizes_span, dim_order_span);

  EXPECT_EQ(layout.scalar_type(), ScalarType::Float);

  EXPECT_EQ(layout.sizes().size(), sizes_span.size());
  EXPECT_EQ(layout.sizes()[0], sizes_span[0]);
  EXPECT_EQ(layout.sizes()[1], sizes_span[1]);

  EXPECT_EQ(layout.dim_order().size(), dim_order_span.size());
  EXPECT_EQ(layout.dim_order()[0], dim_order_span[0]);
  EXPECT_EQ(layout.dim_order()[1], dim_order_span[1]);

  EXPECT_EQ(layout.nbytes(), 8);
}
