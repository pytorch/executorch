/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/flamingo/cross_attention/cross_attention_mask.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorImpl;

TEST(CrossAttentxnMaskTest, TestCrossAttentionMask) {
  std::vector<int> tokens = {
      1, 1, 9673, 527, 1403, 12875, 13, 1, 1115, 374, 264, 8415};

  // Initialize image tensors.
  TensorImpl::SizesType sizes[2] = {2, 2};
  TensorImpl::DimOrderType dim_order[2] = {0, 1};
  TensorImpl::StridesType strides[2] = {2, 1};

  int32_t a_data[4] = {1, 2, 3, 4};
  auto a_impl =
      TensorImpl(ScalarType::Int, 2, sizes, a_data, dim_order, strides);
  Tensor a(&a_impl);

  int32_t b_data[4] = {5, 6, 7, 8};
  auto b_impl =
      TensorImpl(ScalarType::Int, 2, sizes, b_data, dim_order, strides);
  Tensor b(&b_impl);

  int32_t c_data[4] = {9, 10, 11, 12};
  auto c_impl =
      TensorImpl(ScalarType::Int, 2, sizes, c_data, dim_order, strides);
  Tensor c(&c_impl);

  std::vector<Tensor> images = {a, b, c};
  std::vector<std::vector<int>> mask_data;
  auto output_masks = example::cross_attention_mask(
      tokens,
      images,
      /*tile_size=*/1,
      /*patch_size=*/1,
      /*image_token_id=*/1,
      /*out=*/mask_data);

  // Check contents of the mask.
  std::vector<std::vector<size_t>> expected_intervals = {
      {0, 7}, {1, 7}, {7, 12}};
  for (size_t mask_idx = 0; mask_idx < output_masks.size(); ++mask_idx) {
    auto& output_tensor = output_masks[mask_idx];
    for (size_t i = 0; i < output_tensor->size(0); ++i) {
      for (size_t j = 0; j < output_tensor->strides()[0]; ++j) {
        size_t unrolled_index = i * output_tensor->strides()[0] + j;
        if (i >= expected_intervals[mask_idx][0] &&
            i < expected_intervals[mask_idx][1]) {
          EXPECT_EQ(output_tensor->const_data_ptr<int>()[unrolled_index], 1);
        } else {
          EXPECT_EQ(output_tensor->const_data_ptr<int>()[unrolled_index], 0);
        }
      }
    }
  }
}
