/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/tensor/tensor_impl_ptr.h>

#include <gtest/gtest.h>

#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace executorch::extension;
using namespace executorch::runtime;

class TensorImplPtrTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
  }
};

TEST_F(TensorImplPtrTest, TensorImplCreation) {
  float data[20] = {2};
  auto tensor_impl = make_tensor_impl_ptr(
      exec_aten::ScalarType::Float, {4, 5}, data, {0, 1}, {5, 1});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 4);
  EXPECT_EQ(tensor_impl->size(1), 5);
  EXPECT_EQ(tensor_impl->strides()[0], 5);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
  EXPECT_EQ(tensor_impl->data(), data);
  EXPECT_EQ(tensor_impl->mutable_data(), data);
  EXPECT_EQ(((float*)tensor_impl->mutable_data())[0], 2);
}

TEST_F(TensorImplPtrTest, TensorImplSharedOwnership) {
  float data[20] = {2};
  auto tensor_impl1 =
      make_tensor_impl_ptr(exec_aten::ScalarType::Float, {4, 5}, data);
  auto tensor_impl2 = tensor_impl1;

  EXPECT_EQ(tensor_impl1.get(), tensor_impl2.get());
  EXPECT_EQ(tensor_impl1.use_count(), tensor_impl2.use_count());

  tensor_impl1.reset();
  EXPECT_EQ(tensor_impl2.use_count(), 1);
  EXPECT_NE(tensor_impl2.get(), nullptr);
}

TEST_F(TensorImplPtrTest, TensorImplInferredDimOrderAndStrides) {
  float data[12] = {0};
  auto tensor_impl = make_tensor_impl_ptr(
      exec_aten::ScalarType::Float, {3, 4}, data, {}, {4, 1});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->size(1), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 4);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
  EXPECT_EQ(tensor_impl->data(), data);
}

TEST_F(TensorImplPtrTest, TensorImplInferredDimOrderCustomStrides) {
  float data[12] = {0};
  auto tensor_impl = make_tensor_impl_ptr(
      exec_aten::ScalarType::Float, {3, 4}, data, {}, {1, 3});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->size(1), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 1);
  EXPECT_EQ(tensor_impl->strides()[1], 3);
}

TEST_F(TensorImplPtrTest, TensorImplDefaultDimOrderAndStrides) {
  float data[24] = {0};
  auto tensor_impl =
      make_tensor_impl_ptr(exec_aten::ScalarType::Float, {2, 3, 4}, data);

  EXPECT_EQ(tensor_impl->dim(), 3);
  EXPECT_EQ(tensor_impl->size(0), 2);
  EXPECT_EQ(tensor_impl->size(1), 3);
  EXPECT_EQ(tensor_impl->size(2), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 12);
  EXPECT_EQ(tensor_impl->strides()[1], 4);
  EXPECT_EQ(tensor_impl->strides()[2], 1);
}

TEST_F(TensorImplPtrTest, TensorImplMismatchStridesAndDimOrder) {
  float data[12] = {0};
  ET_EXPECT_DEATH(
      {
        auto _ = make_tensor_impl_ptr(
            exec_aten::ScalarType::Float, {3, 4}, data, {1, 0}, {1, 4});
      },
      "");
}

TEST_F(TensorImplPtrTest, TensorImplCustomDimOrderAndStrides) {
  float data[12] = {0};
  auto tensor_impl = make_tensor_impl_ptr(
      exec_aten::ScalarType::Float, {3, 4}, data, {1, 0}, {1, 3});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->size(1), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 1);
  EXPECT_EQ(tensor_impl->strides()[1], 3);
}

TEST_F(TensorImplPtrTest, TensorImplInvalidDimOrder) {
  ET_EXPECT_DEATH(
      {
        float data[20] = {2};
        auto _ = make_tensor_impl_ptr(
            exec_aten::ScalarType::Float, {4, 5}, data, {2, 1});
      },
      "");
}

TEST_F(TensorImplPtrTest, TensorImplCustomDeleter) {
  float data[20] = {4};
  auto tensor_impl =
      make_tensor_impl_ptr(exec_aten::ScalarType::Float, {4, 5}, data);

  TensorImplPtr copied_tensor_impl = tensor_impl;
  EXPECT_EQ(tensor_impl.use_count(), copied_tensor_impl.use_count());

  tensor_impl.reset();
  EXPECT_EQ(copied_tensor_impl.use_count(), 1);
}

TEST_F(TensorImplPtrTest, TensorImplDataDeleterReleasesCapturedSharedPtr) {
  auto deleter_called = false;
  std::shared_ptr<float[]> data_ptr(
      new float[10], [](float* ptr) { delete[] ptr; });
  auto tensor_impl = make_tensor_impl_ptr(
      exec_aten::ScalarType::Float,
      {4, 5},
      data_ptr.get(),
      {},
      {},
      exec_aten::TensorShapeDynamism::STATIC,
      [data_ptr, &deleter_called](void*) mutable { deleter_called = true; });

  EXPECT_EQ(data_ptr.use_count(), 2);

  tensor_impl.reset();
  EXPECT_TRUE(deleter_called);
  EXPECT_EQ(data_ptr.use_count(), 1);
}

TEST_F(TensorImplPtrTest, TensorImplOwningData) {
  auto tensor_impl = make_tensor_impl_ptr(
      {2, 5},
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
      {1, 0},
      {1, 2});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 2);
  EXPECT_EQ(tensor_impl->size(1), 5);
  EXPECT_EQ(tensor_impl->strides()[0], 1);
  EXPECT_EQ(tensor_impl->strides()[1], 2);
  EXPECT_EQ(((float*)tensor_impl->data())[0], 1.0f);
  EXPECT_EQ(((float*)tensor_impl->data())[9], 10.0f);
}

TEST_F(TensorImplPtrTest, TensorImplOwningEmptyData) {
  auto tensor_impl = make_tensor_impl_ptr({0, 5}, {});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 0);
  EXPECT_EQ(tensor_impl->size(1), 5);
  EXPECT_EQ(tensor_impl->strides()[0], 5);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
  EXPECT_EQ(tensor_impl->data(), nullptr);
}
