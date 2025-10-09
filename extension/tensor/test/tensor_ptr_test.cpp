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
#include <executorch/test/utils/DeathTest.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

class TensorPtrTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
  }
};

TEST_F(TensorPtrTest, ScalarTensorCreation) {
  float scalar_data = 3.14f;
  auto tensor = make_tensor_ptr({}, &scalar_data);

  EXPECT_EQ(tensor->numel(), 1);
  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->sizes().size(), 0);
  EXPECT_EQ(tensor->strides().size(), 0);
  EXPECT_EQ(tensor->const_data_ptr<float>(), &scalar_data);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 3.14f);
}

TEST_F(TensorPtrTest, ScalarTensorOwningData) {
  auto tensor = make_tensor_ptr({}, {3.14f});

  EXPECT_EQ(tensor->numel(), 1);
  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->sizes().size(), 0);
  EXPECT_EQ(tensor->strides().size(), 0);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 3.14f);
}

TEST_F(TensorPtrTest, ScalarTensorSingleValueCreation) {
  auto tensor_float = make_tensor_ptr(3.14f);
  EXPECT_EQ(tensor_float->dim(), 0);
  EXPECT_EQ(tensor_float->numel(), 1);
  EXPECT_EQ(tensor_float->sizes().size(), 0);
  EXPECT_EQ(tensor_float->strides().size(), 0);
  EXPECT_EQ(tensor_float->const_data_ptr<float>()[0], 3.14f);
  EXPECT_EQ(tensor_float->scalar_type(), executorch::aten::ScalarType::Float);

  auto tensor_int32 = make_tensor_ptr(42);
  EXPECT_EQ(tensor_int32->dim(), 0);
  EXPECT_EQ(tensor_int32->numel(), 1);
  EXPECT_EQ(tensor_int32->sizes().size(), 0);
  EXPECT_EQ(tensor_int32->strides().size(), 0);
  EXPECT_EQ(tensor_int32->const_data_ptr<int32_t>()[0], 42);
  EXPECT_EQ(tensor_int32->scalar_type(), executorch::aten::ScalarType::Int);

  auto tensor_double = make_tensor_ptr(2.718);
  EXPECT_EQ(tensor_double->dim(), 0);
  EXPECT_EQ(tensor_double->numel(), 1);
  EXPECT_EQ(tensor_double->sizes().size(), 0);
  EXPECT_EQ(tensor_double->strides().size(), 0);
  EXPECT_EQ(tensor_double->const_data_ptr<double>()[0], 2.718);
  EXPECT_EQ(tensor_double->scalar_type(), executorch::aten::ScalarType::Double);

  auto tensor_int64 = make_tensor_ptr(static_cast<int64_t>(10000000000));
  EXPECT_EQ(tensor_int64->dim(), 0);
  EXPECT_EQ(tensor_int64->numel(), 1);
  EXPECT_EQ(tensor_int64->sizes().size(), 0);
  EXPECT_EQ(tensor_int64->strides().size(), 0);
  EXPECT_EQ(tensor_int64->const_data_ptr<int64_t>()[0], 10000000000);
  EXPECT_EQ(tensor_int64->scalar_type(), executorch::aten::ScalarType::Long);
}

TEST_F(TensorPtrTest, CreateTensorWithStridesAndDimOrder) {
  float data[20] = {2};
  auto tensor = make_tensor_ptr({4, 5}, data, {0, 1}, {5, 1});
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
  auto tensor1 = make_tensor_ptr({4, 5}, data);
  auto tensor2 = tensor1;
  EXPECT_EQ(tensor1.get(), tensor2.get());
  EXPECT_EQ(tensor1->unsafeGetTensorImpl(), tensor2->unsafeGetTensorImpl());
}

TEST_F(TensorPtrTest, TensorLifetime) {
  TensorPtr tensor;
  EXPECT_EQ(tensor, nullptr);
  {
    float data[20] = {2};
    tensor = make_tensor_ptr({4, 5}, data);
  }
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
}

TEST_F(TensorPtrTest, TensorWithZeroDimensionAndElements) {
  float data[20] = {2};
  auto tensor = make_tensor_ptr({}, data);
  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->numel(), 1);
  tensor = make_tensor_ptr({0, 5}, data);
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->numel(), 0);
}

TEST_F(TensorPtrTest, TensorResize) {
  float data[20] = {2};
  auto tensor = make_tensor_ptr(
      {4, 5},
      data,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_UNBOUND);
  EXPECT_EQ(resize_tensor_ptr(tensor, {5, 4}), Error::Ok);
  EXPECT_EQ(tensor->size(0), 5);
  EXPECT_EQ(tensor->size(1), 4);
}

TEST_F(TensorPtrTest, TensorDataAccess) {
  float data[6] = {1, 2, 3, 4, 5, 6};
  auto tensor = make_tensor_ptr({2, 3}, data);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 1);
  EXPECT_EQ(tensor->const_data_ptr<float>()[5], 6);
  tensor->mutable_data_ptr<float>()[0] = 10;
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 10);
}

TEST_F(TensorPtrTest, TensorWithCustomDataDeleter) {
  auto deleter_called = false;
  float* data = new float[20]();
  auto tensor = make_tensor_ptr(
      {4, 5},
      data,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
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
      {4, 5},
      data_ptr,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
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
      {4, 5},
      data_ptr.get(),
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
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
  auto tensor = make_tensor_ptr({0, 5}, std::vector<float>());

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 0);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->strides()[0], 5);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->data_ptr<float>(), nullptr);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Float);
}

TEST_F(TensorPtrTest, TensorDataOnly) {
  auto tensor = make_tensor_ptr({1.0f, 2.0f, 3.0f, 4.0f});

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 1.0);
  EXPECT_EQ(tensor->const_data_ptr<float>()[3], 4.0);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Float);
}

TEST_F(TensorPtrTest, TensorDataOnlyDoubleType) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  auto tensor = make_tensor_ptr(std::move(data));

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->const_data_ptr<double>()[0], 1.0);
  EXPECT_EQ(tensor->const_data_ptr<double>()[3], 4.0);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Double);
}

TEST_F(TensorPtrTest, TensorDataOnlyInt32Type) {
  std::vector<int32_t> data = {10, 20, 30, 40};
  auto tensor = make_tensor_ptr(std::move(data));

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->const_data_ptr<int32_t>()[0], 10);
  EXPECT_EQ(tensor->const_data_ptr<int32_t>()[3], 40);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Int);
}

TEST_F(TensorPtrTest, TensorDataOnlyInt64Type) {
  std::vector<int64_t> data = {100, 200, 300, 400};
  auto tensor = make_tensor_ptr(std::move(data));

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->const_data_ptr<int64_t>()[0], 100);
  EXPECT_EQ(tensor->const_data_ptr<int64_t>()[3], 400);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Long);
}

TEST_F(TensorPtrTest, TensorDataOnlyUint8Type) {
  std::vector<uint8_t> data = {10, 20, 30, 40};
  auto tensor = make_tensor_ptr(std::move(data));

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->const_data_ptr<uint8_t>()[0], 10);
  EXPECT_EQ(tensor->const_data_ptr<uint8_t>()[3], 40);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Byte);
}

TEST_F(TensorPtrTest, TensorAmbiguityWithMixedVectors) {
  std::vector<executorch::aten::SizesType> sizes = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor = make_tensor_ptr(std::move(sizes), std::move(data));

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 2);
  EXPECT_EQ(tensor->strides()[0], 2);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 1.0f);
  EXPECT_EQ(tensor->const_data_ptr<float>()[3], 4.0f);

  auto tensor2 = make_tensor_ptr({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  EXPECT_EQ(tensor2->dim(), 2);
  EXPECT_EQ(tensor2->size(0), 2);
  EXPECT_EQ(tensor2->size(1), 2);
  EXPECT_EQ(tensor2->strides()[0], 2);
  EXPECT_EQ(tensor2->strides()[1], 1);
  EXPECT_EQ(tensor2->const_data_ptr<float>()[0], 1.0f);
  EXPECT_EQ(tensor2->const_data_ptr<float>()[3], 4.0f);
}

TEST_F(TensorPtrTest, TensorSharingImplModifiesSharedDataVector) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};

  auto tensor1 = make_tensor_ptr({2, 3}, std::move(data));
  auto tensor2 = tensor1;

  tensor1->mutable_data_ptr<float>()[0] = 10;
  EXPECT_EQ(tensor2->const_data_ptr<float>()[0], 10);

  tensor2->mutable_data_ptr<float>()[5] = 20;
  EXPECT_EQ(tensor1->const_data_ptr<float>()[5], 20);
}

TEST_F(TensorPtrTest, TensorSharingImplResizingAffectsBothVector) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  auto tensor1 = make_tensor_ptr({3, 4}, std::move(data));
  auto tensor2 = tensor1;

  EXPECT_EQ(resize_tensor_ptr(tensor1, {2, 6}), Error::Ok);
  EXPECT_EQ(tensor2->size(0), 2);
  EXPECT_EQ(tensor2->size(1), 6);

  EXPECT_EQ(resize_tensor_ptr(tensor2, {4, 3}), Error::Ok);
  EXPECT_EQ(tensor1->size(0), 4);
  EXPECT_EQ(tensor1->size(1), 3);
}

TEST_F(TensorPtrTest, MakeTensorPtrFromExistingTensorInt32) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  auto tensor = make_tensor_ptr({2, 2}, data);
  auto new_tensor = make_tensor_ptr(tensor);

  EXPECT_EQ(new_tensor->dim(), tensor->dim());
  EXPECT_EQ(new_tensor->size(0), tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), tensor->size(1));
  EXPECT_EQ(
      new_tensor->const_data_ptr<int32_t>(), tensor->const_data_ptr<int32_t>());
  EXPECT_EQ(new_tensor->scalar_type(), executorch::aten::ScalarType::Int);
}

TEST_F(TensorPtrTest, CloneTensorPtrFromExistingTensorInt32) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));
  auto cloned_tensor = clone_tensor_ptr(tensor);

  EXPECT_EQ(cloned_tensor->dim(), tensor->dim());
  EXPECT_EQ(cloned_tensor->size(0), tensor->size(0));
  EXPECT_EQ(cloned_tensor->size(1), tensor->size(1));
  EXPECT_NE(
      cloned_tensor->const_data_ptr<int32_t>(),
      tensor->const_data_ptr<int32_t>());
  EXPECT_EQ(cloned_tensor->const_data_ptr<int32_t>()[0], 1);
  EXPECT_EQ(cloned_tensor->const_data_ptr<int32_t>()[3], 4);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Int);
}

TEST_F(TensorPtrTest, MakeTensorPtrFromTensorPtrInt32) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  auto tensor = make_tensor_ptr({2, 2}, data);
  auto new_tensor = make_tensor_ptr(tensor);

  EXPECT_EQ(new_tensor->dim(), tensor->dim());
  EXPECT_EQ(new_tensor->size(0), tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), tensor->size(1));
  EXPECT_EQ(
      new_tensor->const_data_ptr<int32_t>(), tensor->const_data_ptr<int32_t>());
  EXPECT_EQ(new_tensor->scalar_type(), executorch::aten::ScalarType::Int);
}

TEST_F(TensorPtrTest, MakeTensorPtrFromTensorPtrDouble) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  auto tensor = make_tensor_ptr({2, 2}, data);
  auto new_tensor = make_tensor_ptr(tensor);

  EXPECT_EQ(new_tensor->dim(), tensor->dim());
  EXPECT_EQ(new_tensor->size(0), tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), tensor->size(1));
  EXPECT_EQ(
      new_tensor->const_data_ptr<double>(), tensor->const_data_ptr<double>());
  EXPECT_EQ(new_tensor->scalar_type(), executorch::aten::ScalarType::Double);
}

TEST_F(TensorPtrTest, MakeTensorPtrFromTensorPtrInt64) {
  std::vector<int64_t> data = {100, 200, 300, 400};
  auto tensor = make_tensor_ptr({2, 2}, data);
  auto new_tensor = make_tensor_ptr(tensor);

  EXPECT_EQ(new_tensor->dim(), tensor->dim());
  EXPECT_EQ(new_tensor->size(0), tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), tensor->size(1));
  EXPECT_EQ(
      new_tensor->const_data_ptr<int64_t>(), tensor->const_data_ptr<int64_t>());
  EXPECT_EQ(new_tensor->scalar_type(), executorch::aten::ScalarType::Long);
}

TEST_F(TensorPtrTest, MakeTensorPtrFromTensorPtrNull) {
  auto tensor = make_tensor_ptr({2, 2}, nullptr);
  auto new_tensor = make_tensor_ptr(tensor);

  EXPECT_EQ(new_tensor->dim(), tensor->dim());
  EXPECT_EQ(new_tensor->size(0), tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), tensor->size(1));
  EXPECT_EQ(new_tensor->const_data_ptr(), tensor->const_data_ptr());
  EXPECT_EQ(new_tensor->const_data_ptr(), nullptr);
}

TEST_F(TensorPtrTest, CloneTensorPtrFromTensorPtrInt32) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));
  auto cloned_tensor = clone_tensor_ptr(tensor);

  EXPECT_EQ(cloned_tensor->dim(), tensor->dim());
  EXPECT_EQ(cloned_tensor->size(0), tensor->size(0));
  EXPECT_EQ(cloned_tensor->size(1), tensor->size(1));
  EXPECT_NE(
      cloned_tensor->const_data_ptr<int32_t>(),
      tensor->const_data_ptr<int32_t>());
  EXPECT_EQ(cloned_tensor->const_data_ptr<int32_t>()[0], 1);
  EXPECT_EQ(cloned_tensor->const_data_ptr<int32_t>()[3], 4);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Int);
}

TEST_F(TensorPtrTest, MakeTensorPtrFromExistingTensorDouble) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  auto tensor = make_tensor_ptr({2, 2}, data);
  auto new_tensor = make_tensor_ptr(tensor);

  EXPECT_EQ(new_tensor->dim(), tensor->dim());
  EXPECT_EQ(new_tensor->size(0), tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), tensor->size(1));
  EXPECT_EQ(
      new_tensor->const_data_ptr<double>(), tensor->const_data_ptr<double>());
  EXPECT_EQ(new_tensor->scalar_type(), executorch::aten::ScalarType::Double);
}

TEST_F(TensorPtrTest, CloneTensorPtrFromExistingTensorDouble) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));
  auto cloned_tensor = clone_tensor_ptr(tensor);

  EXPECT_EQ(cloned_tensor->dim(), tensor->dim());
  EXPECT_EQ(cloned_tensor->size(0), tensor->size(0));
  EXPECT_EQ(cloned_tensor->size(1), tensor->size(1));
  EXPECT_NE(
      cloned_tensor->const_data_ptr<double>(),
      tensor->const_data_ptr<double>());
  EXPECT_EQ(cloned_tensor->const_data_ptr<double>()[0], 1.0);
  EXPECT_EQ(cloned_tensor->const_data_ptr<double>()[3], 4.0);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Double);
}

TEST_F(TensorPtrTest, CloneTensorPtrFromTensorPtrDouble) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));
  auto cloned_tensor = clone_tensor_ptr(tensor);

  EXPECT_EQ(cloned_tensor->dim(), tensor->dim());
  EXPECT_EQ(cloned_tensor->size(0), tensor->size(0));
  EXPECT_EQ(cloned_tensor->size(1), tensor->size(1));
  EXPECT_NE(
      cloned_tensor->const_data_ptr<double>(),
      tensor->const_data_ptr<double>());
  EXPECT_EQ(cloned_tensor->const_data_ptr<double>()[0], 1.0);
  EXPECT_EQ(cloned_tensor->const_data_ptr<double>()[3], 4.0);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Double);
}

TEST_F(TensorPtrTest, MakeTensorPtrFromExistingTensorInt64) {
  std::vector<int64_t> data = {100, 200, 300, 400};
  auto tensor = make_tensor_ptr({2, 2}, data);
  auto new_tensor = make_tensor_ptr(tensor);

  EXPECT_EQ(new_tensor->dim(), tensor->dim());
  EXPECT_EQ(new_tensor->size(0), tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), tensor->size(1));
  EXPECT_EQ(
      new_tensor->const_data_ptr<int64_t>(), tensor->const_data_ptr<int64_t>());
  EXPECT_EQ(new_tensor->scalar_type(), executorch::aten::ScalarType::Long);
}

TEST_F(TensorPtrTest, CloneTensorPtrFromExistingTensorInt64) {
  std::vector<int64_t> data = {100, 200, 300, 400};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));
  auto cloned_tensor = clone_tensor_ptr(tensor);

  EXPECT_EQ(cloned_tensor->dim(), tensor->dim());
  EXPECT_EQ(cloned_tensor->size(0), tensor->size(0));
  EXPECT_EQ(cloned_tensor->size(1), tensor->size(1));
  EXPECT_NE(
      cloned_tensor->const_data_ptr<int64_t>(),
      tensor->const_data_ptr<int64_t>());
  EXPECT_EQ(cloned_tensor->const_data_ptr<int64_t>()[0], 100);
  EXPECT_EQ(cloned_tensor->const_data_ptr<int64_t>()[3], 400);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Long);
}

TEST_F(TensorPtrTest, CloneTensorPtrFromTensorPtrInt64) {
  std::vector<int64_t> data = {100, 200, 300, 400};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));
  auto cloned_tensor = clone_tensor_ptr(tensor);

  EXPECT_EQ(cloned_tensor->dim(), tensor->dim());
  EXPECT_EQ(cloned_tensor->size(0), tensor->size(0));
  EXPECT_EQ(cloned_tensor->size(1), tensor->size(1));
  EXPECT_NE(
      cloned_tensor->const_data_ptr<int64_t>(),
      tensor->const_data_ptr<int64_t>());
  EXPECT_EQ(cloned_tensor->const_data_ptr<int64_t>()[0], 100);
  EXPECT_EQ(cloned_tensor->const_data_ptr<int64_t>()[3], 400);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Long);
}

TEST_F(TensorPtrTest, CloneTensorPtrFromTensorPtrNull) {
  auto tensor = make_tensor_ptr({2, 2}, nullptr);
  auto cloned_tensor = clone_tensor_ptr(tensor);

  EXPECT_EQ(cloned_tensor->dim(), tensor->dim());
  EXPECT_EQ(cloned_tensor->size(0), tensor->size(0));
  EXPECT_EQ(cloned_tensor->size(1), tensor->size(1));
  EXPECT_EQ(cloned_tensor->const_data_ptr(), tensor->const_data_ptr());
  EXPECT_EQ(cloned_tensor->const_data_ptr(), nullptr);
}

TEST_F(TensorPtrTest, TensorDataCastingFromIntToFloat) {
  std::vector<int32_t> int_data = {1, 2, 3, 4, 5, 6};
  auto tensor = make_tensor_ptr(
      {2, 3}, std::move(int_data), {}, {}, executorch::aten::ScalarType::Float);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Float);

  auto data_ptr = tensor->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data_ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(data_ptr[5], 6.0f);
}

TEST_F(TensorPtrTest, TensorDataCastingFromIntToDouble) {
  std::vector<int32_t> int_data = {1, 2, 3};
  auto tensor = make_tensor_ptr(
      std::move(int_data), executorch::aten::ScalarType::Double);

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Double);

  auto data_ptr = tensor->const_data_ptr<double>();
  EXPECT_DOUBLE_EQ(data_ptr[0], 1.0);
  EXPECT_DOUBLE_EQ(data_ptr[1], 2.0);
  EXPECT_DOUBLE_EQ(data_ptr[2], 3.0);
}

TEST_F(TensorPtrTest, TensorDataCastingFromFloatToHalf) {
  std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
  auto tensor = make_tensor_ptr(
      std::move(float_data), executorch::aten::ScalarType::Half);

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Half);

  auto data_ptr = tensor->const_data_ptr<executorch::aten::Half>();
  EXPECT_EQ(static_cast<float>(data_ptr[0]), 1.0f);
  EXPECT_EQ(static_cast<float>(data_ptr[1]), 2.0f);
  EXPECT_EQ(static_cast<float>(data_ptr[2]), 3.0f);
}

TEST_F(TensorPtrTest, TensorDataCastingFromDoubleToFloat) {
  std::vector<double> double_data = {1.1, 2.2, 3.3};
  auto tensor = make_tensor_ptr(
      std::move(double_data), executorch::aten::ScalarType::Float);

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Float);

  auto data_ptr = tensor->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(data_ptr[0], 1.1f);
  EXPECT_FLOAT_EQ(data_ptr[1], 2.2f);
  EXPECT_FLOAT_EQ(data_ptr[2], 3.3f);
}

TEST_F(TensorPtrTest, TensorDataCastingFromInt64ToInt32) {
  std::vector<int64_t> int64_data = {10000000000, 20000000000, 30000000000};
  auto tensor =
      make_tensor_ptr(std::move(int64_data), executorch::aten::ScalarType::Int);

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Int);

  auto data_ptr = tensor->const_data_ptr<int32_t>();
  EXPECT_NE(data_ptr[0], 10000000000); // Expected overflow
}

TEST_F(TensorPtrTest, TensorDataCastingFromFloatToBFloat16) {
  std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
  auto tensor = make_tensor_ptr(
      std::move(float_data), executorch::aten::ScalarType::BFloat16);

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::BFloat16);

  auto data_ptr = tensor->const_data_ptr<executorch::aten::BFloat16>();
  EXPECT_EQ(static_cast<float>(data_ptr[0]), 1.0f);
  EXPECT_EQ(static_cast<float>(data_ptr[1]), 2.0f);
  EXPECT_EQ(static_cast<float>(data_ptr[2]), 3.0f);
}

TEST_F(TensorPtrTest, InitializerListDoubleToHalf) {
  auto tensor = make_tensor_ptr<double>(
      {1.5, 2.7, 3.14}, executorch::aten::ScalarType::Half);
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Half);

  auto data_ptr = tensor->const_data_ptr<executorch::aten::Half>();
  EXPECT_NEAR(static_cast<float>(data_ptr[0]), 1.5f, 0.01);
  EXPECT_NEAR(static_cast<float>(data_ptr[1]), 2.7f, 0.01);
  EXPECT_NEAR(static_cast<float>(data_ptr[2]), 3.14f, 0.01);
}

TEST_F(TensorPtrTest, InitializerListInt8ToInt64) {
  auto tensor = make_tensor_ptr<int8_t>(
      {1, -2, 3, -4}, executorch::aten::ScalarType::Long);
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Long);

  auto data_ptr = tensor->const_data_ptr<int64_t>();
  EXPECT_EQ(data_ptr[0], 1);
  EXPECT_EQ(data_ptr[1], -2);
  EXPECT_EQ(data_ptr[2], 3);
  EXPECT_EQ(data_ptr[3], -4);
}

TEST_F(TensorPtrTest, TensorInferredDimOrderAndStrides) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data, {}, {4, 1});

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->size(1), 4);
  EXPECT_EQ(tensor->strides()[0], 4);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->const_data_ptr(), data);
}

TEST_F(TensorPtrTest, TensorInferredDimOrderCustomStrides) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data, {}, {1, 3});

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->size(1), 4);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->strides()[1], 3);
}

TEST_F(TensorPtrTest, TensorDefaultDimOrderAndStrides) {
  float data[24] = {0};
  auto tensor = make_tensor_ptr({2, 3, 4}, data);

  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
  EXPECT_EQ(tensor->size(2), 4);
  EXPECT_EQ(tensor->strides()[0], 12);
  EXPECT_EQ(tensor->strides()[1], 4);
  EXPECT_EQ(tensor->strides()[2], 1);
}

TEST_F(TensorPtrTest, TensorMismatchStridesAndDimOrder) {
  float data[12] = {0};
  ET_EXPECT_DEATH(
      { auto _ = make_tensor_ptr({3, 4}, data, {1, 0}, {1, 4}); }, "");
}

TEST_F(TensorPtrTest, TensorCustomDimOrderAndStrides) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data, {1, 0}, {1, 3});

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->size(1), 4);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->strides()[1], 3);
}

TEST_F(TensorPtrTest, TensorInvalidDimOrder) {
  ET_EXPECT_DEATH(
      {
        float data[20] = {2};
        auto _ = make_tensor_ptr({4, 5}, data, {2, 1}, {1, 4});
      },
      "");
}

TEST_F(TensorPtrTest, TensorCustomDeleter) {
  float data[20] = {4};
  auto tensor = make_tensor_ptr({4, 5}, data);

  TensorPtr copied_tensor = tensor;
  EXPECT_EQ(tensor.use_count(), copied_tensor.use_count());

  tensor.reset();
  EXPECT_EQ(copied_tensor.use_count(), 1);
}

TEST_F(TensorPtrTest, TensorDataDeleterReleasesCapturedSharedPtr) {
  auto deleter_called = false;
  std::shared_ptr<float[]> data_ptr(
      new float[10], [](float* ptr) { delete[] ptr; });
  auto tensor = make_tensor_ptr(
      {4, 5},
      data_ptr.get(),
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      [data_ptr, &deleter_called](void*) mutable { deleter_called = true; });

  EXPECT_EQ(data_ptr.use_count(), 2);

  tensor.reset();
  EXPECT_TRUE(deleter_called);
  EXPECT_EQ(data_ptr.use_count(), 1);
}

TEST_F(TensorPtrTest, SharedDataManagement) {
  auto data = std::make_shared<std::vector<float>>(100, 1.0f);
  auto tensor1 = make_tensor_ptr({10, 10}, data->data());
  auto tensor2 = tensor1;

  EXPECT_EQ(tensor1.get(), tensor2.get());
  EXPECT_EQ(tensor1.use_count(), 2);
  EXPECT_EQ(tensor1->const_data_ptr<float>()[0], 1.0f);

  tensor1->mutable_data_ptr<float>()[0] = 2.0f;
  EXPECT_EQ(tensor1->const_data_ptr<float>()[0], 2.0f);

  tensor1.reset();
  EXPECT_NE(tensor2.get(), nullptr);
  EXPECT_EQ(tensor2.use_count(), 1);

  EXPECT_EQ(tensor2->const_data_ptr<float>()[0], 2.0f);
}

TEST_F(TensorPtrTest, CustomDeleterWithSharedData) {
  auto data = std::make_shared<std::vector<float>>(100, 1.0f);
  bool deleter_called = false;
  {
    auto tensor = make_tensor_ptr(
        {10, 10},
        data->data(),
        {},
        {},
        executorch::aten::ScalarType::Float,
        executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
        [data, &deleter_called](void*) mutable {
          deleter_called = true;
          data.reset();
        });

    EXPECT_EQ(data.use_count(), 2);
    EXPECT_FALSE(deleter_called);
  }
  EXPECT_TRUE(deleter_called);
  EXPECT_EQ(data.use_count(), 1);
}

TEST_F(TensorPtrTest, TensorDeducedScalarType) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 2);
  EXPECT_EQ(tensor->strides()[0], 2);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->const_data_ptr<double>()[0], 1.0);
  EXPECT_EQ(tensor->const_data_ptr<double>()[3], 4.0);
}

TEST_F(TensorPtrTest, TensorUint8BufferWithFloatScalarType) {
  std::vector<uint8_t> data(
      4 * executorch::aten::elementSize(executorch::aten::ScalarType::Float));

  float* float_data = reinterpret_cast<float*>(data.data());
  float_data[0] = 1.0f;
  float_data[1] = 2.0f;
  float_data[2] = 3.0f;
  float_data[3] = 4.0f;

  auto tensor = make_tensor_ptr({2, 2}, std::move(data));

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 2);
  EXPECT_EQ(tensor->strides()[0], 2);
  EXPECT_EQ(tensor->strides()[1], 1);

  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 1.0f);
  EXPECT_EQ(tensor->const_data_ptr<float>()[1], 2.0f);
  EXPECT_EQ(tensor->const_data_ptr<float>()[2], 3.0f);
  EXPECT_EQ(tensor->const_data_ptr<float>()[3], 4.0f);
}

TEST_F(TensorPtrTest, TensorUint8BufferTooSmallExpectDeath) {
  std::vector<uint8_t> data(
      2 * executorch::aten::elementSize(executorch::aten::ScalarType::Float));
  ET_EXPECT_DEATH(
      { auto tensor = make_tensor_ptr({2, 2}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, TensorUint8BufferTooLargeExpectDeath) {
  std::vector<uint8_t> data(
      5 * executorch::aten::elementSize(executorch::aten::ScalarType::Float));
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({2, 2}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, VectorFloatTooSmallExpectDeath) {
  std::vector<float> data(9, 1.f);
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({2, 5}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, VectorFloatTooLargeExpectDeath) {
  std::vector<float> data(11, 1.f);
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({2, 5}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, VectorIntToFloatCastTooSmallExpectDeath) {
  std::vector<int32_t> data(9, 1);
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({2, 5}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, VectorIntToFloatCastTooLargeExpectDeath) {
  std::vector<int32_t> data(11, 1);
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({2, 5}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, StridesAndDimOrderMustMatchSizes) {
  float data[12] = {0};
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({3, 4}, data, {}, {1}); }, "");
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({3, 4}, data, {0}, {4, 1}); }, "");
}

TEST_F(TensorPtrTest, TensorDataCastingInvalidCast) {
  std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
  ET_EXPECT_DEATH(
      {
        auto _ = make_tensor_ptr(
            std::move(float_data), executorch::aten::ScalarType::Int);
      },
      "");
}
