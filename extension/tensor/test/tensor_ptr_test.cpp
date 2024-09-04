/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_ptr.h>

#include <gtest/gtest.h>

#include <executorch/runtime/platform/runtime.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

class TensorPtrTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
  }
};

TEST_F(TensorPtrTest, CreateTensorWithStridesAndDimOrder) {
  float data[20] = {2};
  auto tensor = make_tensor_ptr(
      exec_aten::ScalarType::Float, {4, 5}, data, {0, 1}, {5, 1});
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->strides()[0], 5);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->const_data_ptr<float>(), data);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 2);
}

TEST_F(TensorPtrTest, TensorSharingImpl) {
  float data[20] = {2};
  auto tensor1 = make_tensor_ptr(exec_aten::ScalarType::Float, {4, 5}, data);
  auto tensor2 = make_tensor_ptr(tensor1);
  EXPECT_EQ(tensor1->unsafeGetTensorImpl(), tensor2->unsafeGetTensorImpl());
}

TEST_F(TensorPtrTest, TensorImplLifetime) {
  TensorPtr tensor;
  EXPECT_EQ(tensor, nullptr);
  {
    float data[20] = {2};
    auto tensor_impl =
        make_tensor_impl_ptr(exec_aten::ScalarType::Float, {4, 5}, data);
    tensor = make_tensor_ptr(tensor_impl);
  }
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
}

TEST_F(TensorPtrTest, TensorWithZeroDimensionAndElements) {
  float data[20] = {2};
  auto tensor = make_tensor_ptr(exec_aten::ScalarType::Float, {}, data);
  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->numel(), 1);
  tensor = make_tensor_ptr(exec_aten::ScalarType::Float, {0, 5}, data);
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->numel(), 0);
}

TEST_F(TensorPtrTest, TensorResize) {
  float data[20] = {2};
  auto tensor = make_tensor_ptr(
      exec_aten::ScalarType::Float,
      {4, 5},
      data,
      {},
      {},
      exec_aten::TensorShapeDynamism::DYNAMIC_UNBOUND);
  EXPECT_EQ(resize_tensor_ptr(tensor, {5, 4}), Error::Ok);
  EXPECT_EQ(tensor->size(0), 5);
  EXPECT_EQ(tensor->size(1), 4);
}

TEST_F(TensorPtrTest, TensorDataAccess) {
  float data[6] = {1, 2, 3, 4, 5, 6};
  auto tensor = make_tensor_ptr(exec_aten::ScalarType::Float, {2, 3}, data);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 1);
  EXPECT_EQ(tensor->const_data_ptr<float>()[5], 6);
  tensor->mutable_data_ptr<float>()[0] = 10;
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 10);
}

TEST_F(TensorPtrTest, TensorWithCustomDataDeleter) {
  auto deleter_called = false;
  float* data = new float[20]();
  auto tensor = make_tensor_ptr(
      exec_aten::ScalarType::Float,
      {4, 5},
      data,
      {},
      {},
      exec_aten::TensorShapeDynamism::STATIC,
      [&deleter_called](void* ptr) {
        deleter_called = true;
        delete[] static_cast<float*>(ptr);
      });

  tensor.reset();
  EXPECT_TRUE(deleter_called);
}

TEST_F(TensorPtrTest, TensorManagesMovedVector) {
  auto deleter_called = false;
  std::vector<float> data(20, 3.0f);
  auto* data_ptr = data.data();
  auto tensor = make_tensor_ptr(
      exec_aten::ScalarType::Float,
      {4, 5},
      data_ptr,
      {},
      {},
      exec_aten::TensorShapeDynamism::STATIC,
      [moved_data = std::move(data), &deleter_called](void*) mutable {
        deleter_called = true;
      });

  EXPECT_TRUE(data.empty()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(tensor->data_ptr<float>(), data_ptr);

  tensor.reset();
  EXPECT_TRUE(deleter_called);
}

TEST_F(TensorPtrTest, TensorDeleterReleasesCapturedSharedPtr) {
  auto deleter_called = false;
  std::shared_ptr<float[]> data_ptr(
      new float[10], [](float* ptr) { delete[] ptr; });
  auto tensor = make_tensor_ptr(
      exec_aten::ScalarType::Float,
      {4, 5},
      data_ptr.get(),
      {},
      {},
      exec_aten::TensorShapeDynamism::STATIC,
      [data_ptr, &deleter_called](void*) mutable { deleter_called = true; });

  EXPECT_EQ(data_ptr.use_count(), 2);

  tensor.reset();
  EXPECT_TRUE(deleter_called);
  EXPECT_EQ(data_ptr.use_count(), 1);
}

TEST_F(TensorPtrTest, TensorOwningData) {
  auto tensor = make_tensor_ptr(
      {2, 5},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
      {1, 0},
      {1, 2});

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->strides()[1], 2);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 1.0f);
  EXPECT_EQ(tensor->const_data_ptr<float>()[9], 10.0f);
}

TEST_F(TensorPtrTest, TensorOwningEmptyData) {
  auto tensor = make_tensor_ptr({0, 5}, {});

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 0);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->strides()[0], 5);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->data_ptr<float>(), nullptr);
}
