// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "utils.h"

namespace executorch::vulkan::prototyping {
namespace {

TEST(ValueSpecTest, SetConstant_DoesNotMaterializeTensorData) {
  ValueSpec value(
      {16},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ONES);

  value.set_constant(true);

  EXPECT_FALSE(value.is_data_generated());
}

TEST(ValueSpecTest, ShareDataFrom_SharesImmutableTensorData) {
  ValueSpec source(
      {16},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ONES);
  source.ensure_data_generated();

  ValueSpec copy(
      {16},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kChannelsPacked,
      DataGenType::ONES);
  copy.share_data_from(source);

  const ValueSpec& const_source = source;
  const ValueSpec& const_copy = copy;
  EXPECT_EQ(
      const_source.get_float_data().data(), const_copy.get_float_data().data());
}

TEST(ValueSpecTest, MutableDataAccess_DetachesSharedTensorData) {
  ValueSpec source(
      {4},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ONES);
  source.ensure_data_generated();

  ValueSpec copy = source;
  copy.get_float_data()[0] = 7.0f;

  const ValueSpec& const_source = source;
  const ValueSpec& const_copy = copy;
  EXPECT_FLOAT_EQ(const_source.get_float_data()[0], 1.0f);
  EXPECT_FLOAT_EQ(const_copy.get_float_data()[0], 7.0f);
  EXPECT_NE(
      const_source.get_float_data().data(), const_copy.get_float_data().data());
}

TEST(ValueSpecTest, CopyConstruction_SharesImmutableReferenceData) {
  ValueSpec source(
      {4},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ZEROS);
  source.get_ref_float_data() = {1.0f, 2.0f, 3.0f, 4.0f};

  ValueSpec copy = source;

  const ValueSpec& const_source = source;
  const ValueSpec& const_copy = copy;
  EXPECT_EQ(
      const_source.get_ref_float_data().data(),
      const_copy.get_ref_float_data().data());
}

TEST(ValueSpecTest, MutableReferenceAccess_DetachesSharedReferenceData) {
  ValueSpec source(
      {2},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ZEROS);
  source.get_ref_float_data() = {1.0f, 2.0f};

  ValueSpec copy = source;
  copy.get_ref_float_data()[0] = 9.0f;

  const ValueSpec& const_source = source;
  const ValueSpec& const_copy = copy;
  EXPECT_FLOAT_EQ(const_source.get_ref_float_data()[0], 1.0f);
  EXPECT_FLOAT_EQ(const_copy.get_ref_float_data()[0], 9.0f);
  EXPECT_NE(
      const_source.get_ref_float_data().data(),
      const_copy.get_ref_float_data().data());
}

TEST(ValueSpecTest, ConstGetter_MaterializesDeferredData) {
  ValueSpec value(
      {4},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ONES);
  const ValueSpec& const_value = value;
  EXPECT_FALSE(const_value.is_data_generated());
  EXPECT_FLOAT_EQ(const_value.get_float_data()[0], 1.0f);
  EXPECT_FLOAT_EQ(const_value.get_float_value(), 1.0f);
  EXPECT_TRUE(const_value.is_data_generated());
}

TEST(ValueSpecTest, ResizeData_PreservesGeneratedPattern) {
  ValueSpec value(
      {4},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ONES);
  value.resize_data(8);
  const ValueSpec& const_value = value;
  ASSERT_EQ(const_value.get_float_data().size(), 8u);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(const_value.get_float_data()[i], 1.0f);
  }
}

TEST(ValueSpecTest, MutableDataPtr_DetachesSharedTensorData) {
  ValueSpec source(
      {4},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ONES);
  source.ensure_data_generated();

  ValueSpec copy = source;
  auto* mutable_ptr = static_cast<float*>(copy.get_mutable_data_ptr());
  ASSERT_NE(mutable_ptr, nullptr);
  mutable_ptr[0] = 7.0f;

  const ValueSpec& const_source = source;
  const ValueSpec& const_copy = copy;
  EXPECT_FLOAT_EQ(const_source.get_float_data()[0], 1.0f);
  EXPECT_FLOAT_EQ(const_copy.get_float_data()[0], 7.0f);
}

TEST(ValueSpecTest, ShareReferenceFrom_IgnoresNonTensorSpecs) {
  ValueSpec scalar(3);
  ValueSpec tensor(
      {4},
      vkcompute::vkapi::kFloat,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kWidthPacked,
      DataGenType::ZEROS);
  const void* before = tensor.get_ref_float_data().data();
  tensor.share_reference_from(scalar);
  EXPECT_EQ(tensor.get_ref_float_data().data(), before);
}

} // namespace
} // namespace executorch::vulkan::prototyping
