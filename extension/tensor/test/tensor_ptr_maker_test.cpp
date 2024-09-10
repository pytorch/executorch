/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <gtest/gtest.h>

#include <executorch/runtime/platform/runtime.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

class TensorPtrMakerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
  }
};

TEST_F(TensorPtrMakerTest, CreateTensorUsingTensorMaker) {
  float data[20] = {2};
  auto tensor = for_blob(data, {4, 5})
                    .dim_order({0, 1})
                    .strides({5, 1})
                    .dynamism(exec_aten::TensorShapeDynamism::DYNAMIC_BOUND)
                    .make_tensor_ptr();

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->strides()[0], 5);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->const_data_ptr<float>(), data);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 2);
}

TEST_F(TensorPtrMakerTest, PerfectForwardingLValue) {
  float data[20] = {2};
  std::vector<exec_aten::SizesType> sizes = {4, 5};
  std::vector<exec_aten::DimOrderType> dim_order = {0, 1};
  std::vector<exec_aten::StridesType> strides = {5, 1};

  auto tensor = for_blob(data, sizes)
                    .dim_order(dim_order)
                    .strides(strides)
                    .make_tensor_ptr();

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->strides()[0], 5);
  EXPECT_EQ(tensor->strides()[1], 1);

  EXPECT_EQ(sizes.size(), 2);
  EXPECT_EQ(dim_order.size(), 2);
  EXPECT_EQ(strides.size(), 2);
}

TEST_F(TensorPtrMakerTest, PerfectForwardingRValue) {
  float data[20] = {2};
  std::vector<exec_aten::SizesType> sizes = {4, 5};
  std::vector<exec_aten::DimOrderType> dim_order = {0, 1};
  std::vector<exec_aten::StridesType> strides = {5, 1};

  auto tensor = for_blob(data, std::move(sizes))
                    .dim_order(std::move(dim_order))
                    .strides(std::move(strides))
                    .make_tensor_ptr();

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->strides()[0], 5);
  EXPECT_EQ(tensor->strides()[1], 1);
  // for_blob() moved the contents of the vectors, leaving these empty.
  EXPECT_EQ(sizes.size(), 0); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(dim_order.size(), 0); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(strides.size(), 0); // NOLINT(bugprone-use-after-move)
}

TEST_F(TensorPtrMakerTest, CreateTensorFromBlob) {
  float data[20] = {2};
  auto tensor = from_blob(data, {4, 5});

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->strides()[0], 5);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->const_data_ptr<float>(), data);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 2);
  EXPECT_EQ(tensor->const_data_ptr<float>()[19], 0);
}

TEST_F(TensorPtrMakerTest, CreateTensorUsingFromBlobWithStrides) {
  float data[20] = {3};
  auto tensor = from_blob(data, {2, 2, 2}, {4, 2, 1});

  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 2);
  EXPECT_EQ(tensor->size(2), 2);
  EXPECT_EQ(tensor->strides()[0], 4);
  EXPECT_EQ(tensor->strides()[1], 2);
  EXPECT_EQ(tensor->strides()[2], 1);
  EXPECT_EQ(tensor->const_data_ptr<float>(), data);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 3);
}

TEST_F(TensorPtrMakerTest, TensorMakerConversionOperator) {
  float data[20] = {2};
  TensorPtr tensor =
      for_blob(data, {4, 5})
          .dynamism(exec_aten::TensorShapeDynamism::DYNAMIC_BOUND);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
}

TEST_F(TensorPtrMakerTest, CreateTensorWithZeroDimensions) {
  float data[1] = {2};
  auto tensor = from_blob(data, {});

  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->numel(), 1);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 2);
}

TEST_F(TensorPtrMakerTest, TensorWithCustomDataDeleter) {
  auto deleter_called = false;
  float* data = new float[20]();
  auto tensor = for_blob(data, {4, 5})
                    .deleter([&deleter_called](void* ptr) {
                      deleter_called = true;
                      delete[] static_cast<float*>(ptr);
                    })
                    .make_tensor_ptr();

  tensor.reset();
  EXPECT_TRUE(deleter_called);
}

TEST_F(TensorPtrMakerTest, TensorManagesMovedVector) {
  auto deleter_called = false;
  std::vector<float> data(20, 3.0f);
  auto* data_ptr = data.data();
  auto tensor = for_blob(data_ptr, {4, 5})
                    .deleter([moved_data = std::move(data), &deleter_called](
                                 void*) mutable { deleter_called = true; })
                    .make_tensor_ptr();

  EXPECT_TRUE(data.empty()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(tensor->data_ptr<float>(), data_ptr);

  tensor.reset();
  EXPECT_TRUE(deleter_called);
}

TEST_F(TensorPtrMakerTest, TensorDeleterReleasesCapturedSharedPtr) {
  auto deleter_called = false;
  std::shared_ptr<float[]> data_ptr(
      new float[10], [](float* ptr) { delete[] ptr; });
  auto tensor = from_blob(
      data_ptr.get(),
      {4, 5},
      exec_aten::ScalarType::Float,
      [data_ptr, &deleter_called](void*) mutable { deleter_called = true; });

  EXPECT_EQ(data_ptr.use_count(), 2);

  tensor.reset();
  EXPECT_TRUE(deleter_called);
  EXPECT_EQ(data_ptr.use_count(), 1);
}
