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

TEST_F(TensorPtrTest, MakeViewOverrideSizesRankIncrease) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  auto tensor = make_tensor_ptr({2, 3}, std::move(data));
  auto view = make_tensor_ptr(tensor, {1, 2, 3});

  EXPECT_EQ(view->dim(), 3);
  EXPECT_EQ(view->size(0), 1);
  EXPECT_EQ(view->size(1), 2);
  EXPECT_EQ(view->size(2), 3);
  EXPECT_EQ(view->const_data_ptr<float>(), tensor->const_data_ptr<float>());
  EXPECT_EQ(view->strides()[0], 6);
  EXPECT_EQ(view->strides()[1], 3);
  EXPECT_EQ(view->strides()[2], 1);
}

TEST_F(TensorPtrTest, MakeViewOverrideSizesSameRankRecomputesStrides) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data);
  auto view = make_tensor_ptr(tensor, {4, 3});

  EXPECT_EQ(view->dim(), 2);
  EXPECT_EQ(view->size(0), 4);
  EXPECT_EQ(view->size(1), 3);
  EXPECT_EQ(view->strides()[0], 3);
  EXPECT_EQ(view->strides()[1], 1);
}

TEST_F(TensorPtrTest, MakeViewOverrideDimOrderOnly) {
  float data[6] = {0};
  auto tensor = make_tensor_ptr({2, 3}, data);
  auto view = make_tensor_ptr(tensor, {}, {1, 0}, {});

  EXPECT_EQ(view->dim(), 2);
  EXPECT_EQ(view->size(0), 2);
  EXPECT_EQ(view->size(1), 3);
  EXPECT_EQ(view->strides()[0], 1);
  EXPECT_EQ(view->strides()[1], 2);
}

TEST_F(TensorPtrTest, MakeViewOverrideStridesOnlyInfersDimOrder) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data);
  auto view = make_tensor_ptr(tensor, {}, {}, {1, 3});

  EXPECT_EQ(view->dim(), 2);
  EXPECT_EQ(view->size(0), 3);
  EXPECT_EQ(view->size(1), 4);
  EXPECT_EQ(view->strides()[0], 1);
  EXPECT_EQ(view->strides()[1], 3);
}

TEST_F(TensorPtrTest, MakeViewReuseMetadataWhenShapeSame) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data, {1, 0}, {1, 3});
  auto view = make_tensor_ptr(tensor, {3, 4});

  EXPECT_EQ(view->dim(), 2);
  EXPECT_EQ(view->size(0), 3);
  EXPECT_EQ(view->size(1), 4);
  EXPECT_EQ(view->strides()[0], 1);
  EXPECT_EQ(view->strides()[1], 3);
}

TEST_F(TensorPtrTest, MakeViewShapeChangeWithExplicitOldStridesExpectDeath) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data);
  std::vector<executorch::aten::StridesType> old_strides(
      tensor->strides().begin(), tensor->strides().end());

  ET_EXPECT_DEATH(
      { auto _ = make_tensor_ptr(tensor, {2, 6}, {}, old_strides); }, "");
}

TEST_F(TensorPtrTest, MakeViewInvalidDimOrderExpectDeath) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data);

  ET_EXPECT_DEATH(
      { auto _ = make_tensor_ptr(tensor, {3, 4}, {2, 1}, {1, 4}); }, "");
}

TEST_F(TensorPtrTest, MakeViewFromTensorPtrConvenienceOverload) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr({3, 4}, data);
  auto view = make_tensor_ptr(tensor, {}, {1, 0}, {});

  EXPECT_EQ(view->dim(), 2);
  EXPECT_EQ(view->size(0), 3);
  EXPECT_EQ(view->size(1), 4);
  EXPECT_EQ(view->strides()[0], 1);
  EXPECT_EQ(view->strides()[1], 3);
}

TEST_F(TensorPtrTest, MakeViewRankDecreaseFlatten) {
  float data[6] = {1, 2, 3, 4, 5, 6};
  auto tensor = make_tensor_ptr(
      {2, 3},
      data,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_UNBOUND);
  auto view = make_tensor_ptr(tensor, {6});
  EXPECT_EQ(view->dim(), 1);
  EXPECT_EQ(view->size(0), 6);
  EXPECT_EQ(view->strides()[0], 1);
  EXPECT_NE(tensor->unsafeGetTensorImpl(), view->unsafeGetTensorImpl());
  EXPECT_EQ(resize_tensor_ptr(view, {3, 2}), Error::NotSupported);
  EXPECT_EQ(view->dim(), 1);
  EXPECT_EQ(view->size(0), 6);
}

TEST_F(TensorPtrTest, MakeViewFromScalarAliasAnd1D) {
  float scalar_value = 7.f;
  auto tensor = make_tensor_ptr({}, &scalar_value);
  auto alias = make_tensor_ptr(tensor);
  EXPECT_EQ(alias->dim(), 0);
  EXPECT_EQ(alias->numel(), 1);
  auto reshaped = make_tensor_ptr(tensor, {1});
  EXPECT_EQ(reshaped->dim(), 1);
  EXPECT_EQ(reshaped->size(0), 1);
  EXPECT_EQ(reshaped->strides()[0], 1);
  ET_EXPECT_DEATH({ auto unused = make_tensor_ptr(tensor, {}, {0}, {}); }, "");
  ET_EXPECT_DEATH({ auto unused = make_tensor_ptr(tensor, {}, {}, {1}); }, "");
}

TEST_F(TensorPtrTest, MakeViewExplicitDimOrderAndStridesShapeChange) {
  float data[6] = {0};
  auto tensor = make_tensor_ptr({2, 3}, data);
  auto view = make_tensor_ptr(tensor, {3, 2}, {1, 0}, {1, 3});
  EXPECT_EQ(view->dim(), 2);
  EXPECT_EQ(view->size(0), 3);
  EXPECT_EQ(view->size(1), 2);
  EXPECT_EQ(view->strides()[0], 1);
  EXPECT_EQ(view->strides()[1], 3);
}

TEST_F(TensorPtrTest, TensorUint8dataInt16Type) {
  std::vector<int16_t> int16_values = {-1, 2, -3, 4};
  auto byte_pointer = reinterpret_cast<const uint8_t*>(int16_values.data());
  std::vector<uint8_t> byte_data(
      byte_pointer, byte_pointer + int16_values.size() * sizeof(int16_t));
  auto tensor = make_tensor_ptr(
      {4}, std::move(byte_data), executorch::aten::ScalarType::Short);
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  auto int16_data = tensor->const_data_ptr<int16_t>();
  EXPECT_EQ(int16_data[0], -1);
  EXPECT_EQ(int16_data[1], 2);
  EXPECT_EQ(int16_data[2], -3);
  EXPECT_EQ(int16_data[3], 4);
}

TEST_F(TensorPtrTest, MakeView3DDimOrderOnly) {
  float data[24] = {0};
  auto tensor = make_tensor_ptr({2, 3, 4}, data);
  auto view = make_tensor_ptr(tensor, {}, {2, 0, 1}, {});
  EXPECT_EQ(view->dim(), 3);
  EXPECT_EQ(view->size(0), 2);
  EXPECT_EQ(view->size(1), 3);
  EXPECT_EQ(view->size(2), 4);
  EXPECT_EQ(view->strides()[0], 3);
  EXPECT_EQ(view->strides()[1], 1);
  EXPECT_EQ(view->strides()[2], 6);
}

#ifndef USE_ATEN_LIB
TEST_F(TensorPtrTest, MakeViewDynamismPropagationResizeAlias) {
  float data[12] = {0};
  auto tensor = make_tensor_ptr(
      {3, 4},
      data,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_UNBOUND);
  auto alias = make_tensor_ptr(tensor);
  EXPECT_EQ(resize_tensor_ptr(alias, {2, 6}), Error::Ok);
  EXPECT_EQ(alias->size(0), 2);
  EXPECT_EQ(alias->size(1), 6);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->size(1), 4);
}

TEST_F(TensorPtrTest, MakeViewSameRankShapeChangeCopiesDimOrder) {
  float data[24] = {0};
  auto tensor = make_tensor_ptr({2, 3, 4}, data, {2, 0, 1}, {3, 1, 6});
  auto view = make_tensor_ptr(tensor, {4, 2, 3});
  EXPECT_EQ(view->dim(), 3);
  EXPECT_EQ(view->size(0), 4);
  EXPECT_EQ(view->size(1), 2);
  EXPECT_EQ(view->size(2), 3);
  EXPECT_EQ(view->strides()[0], 2);
  EXPECT_EQ(view->strides()[1], 1);
  EXPECT_EQ(view->strides()[2], 8);
}
#endif

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

TEST_F(TensorPtrTest, CloneTensorPtrCastInt32ToFloat) {
  std::vector<int32_t> data = {1, 2, 3, 4};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));
  auto cloned_tensor =
      clone_tensor_ptr(*tensor, executorch::aten::ScalarType::Float);

  EXPECT_EQ(cloned_tensor->dim(), 2);
  EXPECT_EQ(cloned_tensor->size(0), 2);
  EXPECT_EQ(cloned_tensor->size(1), 2);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Float);
  auto ptr = cloned_tensor->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(ptr[1], 2.0f);
  EXPECT_FLOAT_EQ(ptr[2], 3.0f);
  EXPECT_FLOAT_EQ(ptr[3], 4.0f);
}

TEST_F(TensorPtrTest, CloneTensorPtrCastFloatToBFloat16) {
  std::vector<float> data = {1.0f, 2.0f, 3.5f};
  auto tensor = make_tensor_ptr({3}, std::move(data));
  auto cloned_tensor =
      clone_tensor_ptr(*tensor, executorch::aten::ScalarType::BFloat16);

  EXPECT_EQ(cloned_tensor->dim(), 1);
  EXPECT_EQ(cloned_tensor->size(0), 3);
  EXPECT_EQ(
      cloned_tensor->scalar_type(), executorch::aten::ScalarType::BFloat16);
  auto ptr = cloned_tensor->const_data_ptr<executorch::aten::BFloat16>();
  EXPECT_NEAR(static_cast<float>(ptr[0]), 1.0f, 0.01f);
  EXPECT_NEAR(static_cast<float>(ptr[1]), 2.0f, 0.01f);
  EXPECT_NEAR(static_cast<float>(ptr[2]), 3.5f, 0.01f);
}

TEST_F(TensorPtrTest, CloneTensorPtrCastKeepsMetadata) {
  std::vector<uint8_t> data(
      6 * executorch::aten::elementSize(executorch::aten::ScalarType::Float));
  auto tensor = make_tensor_ptr({2, 3}, std::move(data));
  auto cloned_tensor =
      clone_tensor_ptr(*tensor, executorch::aten::ScalarType::Float);

  EXPECT_EQ(cloned_tensor->dim(), 2);
  EXPECT_EQ(cloned_tensor->size(0), 2);
  EXPECT_EQ(cloned_tensor->size(1), 3);
  EXPECT_EQ(cloned_tensor->strides()[0], 3);
  EXPECT_EQ(cloned_tensor->strides()[1], 1);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Float);
}

TEST_F(TensorPtrTest, CloneTensorPtrCastNullData) {
  auto tensor = make_tensor_ptr(
      {2, 2},
      nullptr,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND);
  auto cloned_tensor =
      clone_tensor_ptr(*tensor, executorch::aten::ScalarType::Int);

  EXPECT_EQ(cloned_tensor->dim(), 2);
  EXPECT_EQ(cloned_tensor->size(0), 2);
  EXPECT_EQ(cloned_tensor->size(1), 2);
  EXPECT_EQ(cloned_tensor->const_data_ptr(), nullptr);
  EXPECT_EQ(cloned_tensor->scalar_type(), executorch::aten::ScalarType::Int);
}

TEST_F(TensorPtrTest, CloneTensorPtrCastInvalidExpectDeath) {
  std::vector<float> data = {1.0f, 2.0f};
  auto tensor = make_tensor_ptr({2}, std::move(data));
  ET_EXPECT_DEATH(
      {
        auto _ = clone_tensor_ptr(*tensor, executorch::aten::ScalarType::Int);
      },
      "");
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

TEST_F(TensorPtrTest, TensorUint8dataWithFloatScalarType) {
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

TEST_F(TensorPtrTest, TensorUint8dataTooSmallExpectDeath) {
  std::vector<uint8_t> data(
      2 * executorch::aten::elementSize(executorch::aten::ScalarType::Float));
  ET_EXPECT_DEATH(
      { auto tensor = make_tensor_ptr({2, 2}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, TensorUint8dataTooLargeExpectDeath) {
  std::vector<uint8_t> data(
      5 * executorch::aten::elementSize(executorch::aten::ScalarType::Float));
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({2, 2}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, MakeViewFromTensorPtrKeepsSourceAlive) {
  bool freed = false;
  auto* data = new float[6]{1, 2, 3, 4, 5, 6};
  auto tensor = make_tensor_ptr(
      {2, 3},
      data,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      [&freed](void* p) {
        freed = true;
        delete[] static_cast<float*>(p);
      });
  auto view = make_tensor_ptr(tensor);
  tensor.reset();
  EXPECT_FALSE(freed);
  EXPECT_EQ(view->const_data_ptr<float>()[0], 1.0f);
  view->mutable_data_ptr<float>()[0] = 42.0f;
  EXPECT_EQ(view->const_data_ptr<float>()[0], 42.0f);
  view.reset();
  EXPECT_TRUE(freed);
}

TEST_F(TensorPtrTest, MakeViewFromTensorDoesNotKeepAliveByDefault) {
  bool freed = false;
  auto* data = new float[2]{7.0f, 8.0f};
  auto tensor = make_tensor_ptr(
      {2, 1},
      data,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      [&freed](void* p) {
        freed = true;
        delete[] static_cast<float*>(p);
      });
  auto view = make_tensor_ptr(*tensor);
  auto raw = view->const_data_ptr<float>();
  EXPECT_EQ(raw, data);
  tensor.reset();
  EXPECT_TRUE(freed);
  view.reset();
}

TEST_F(TensorPtrTest, MakeViewFromTensorWithDeleterKeepsAlive) {
  bool freed = false;
  auto* data = new float[3]{1.0f, 2.0f, 3.0f};
  auto tensor = make_tensor_ptr(
      {3},
      data,
      {},
      {},
      executorch::aten::ScalarType::Float,
      executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND,
      [&freed](void* p) {
        freed = true;
        delete[] static_cast<float*>(p);
      });
  auto view = make_tensor_ptr(*tensor, {}, {}, {}, [tensor](void*) {});
  tensor.reset();
  EXPECT_FALSE(freed);
  EXPECT_EQ(view->const_data_ptr<float>()[2], 3.0f);
  view.reset();
  EXPECT_TRUE(freed);
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

TEST_F(TensorPtrTest, TensorDataOnlyUInt16Type) {
  std::vector<uint16_t> data = {1u, 65535u, 42u, 0u};
  auto tensor = make_tensor_ptr(std::move(data));
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::UInt16);
  auto ptr = tensor->const_data_ptr<uint16_t>();
  EXPECT_EQ(ptr[0], 1u);
  EXPECT_EQ(ptr[1], 65535u);
  EXPECT_EQ(ptr[2], 42u);
  EXPECT_EQ(ptr[3], 0u);
}

TEST_F(TensorPtrTest, TensorDataOnlyUInt32Type) {
  std::vector<uint32_t> data = {0u, 123u, 4000000000u};
  auto tensor = make_tensor_ptr(std::move(data));
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::UInt32);
  auto ptr = tensor->const_data_ptr<uint32_t>();
  EXPECT_EQ(ptr[0], 0u);
  EXPECT_EQ(ptr[1], 123u);
  EXPECT_EQ(ptr[2], 4000000000u);
}

TEST_F(TensorPtrTest, TensorDataOnlyUInt64Type) {
  std::vector<uint64_t> data = {0ull, 1ull, 9000000000000000000ull};
  auto tensor = make_tensor_ptr(std::move(data));
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->strides()[0], 1);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::UInt64);
  auto ptr = tensor->const_data_ptr<uint64_t>();
  EXPECT_EQ(ptr[0], 0ull);
  EXPECT_EQ(ptr[1], 1ull);
  EXPECT_EQ(ptr[2], 9000000000000000000ull);
}

TEST_F(TensorPtrTest, TensorUint8dataUInt32Type) {
  std::vector<uint32_t> values = {1u, 4000000000u, 123u};
  const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
  std::vector<uint8_t> raw(bytes, bytes + values.size() * sizeof(uint32_t));
  auto tensor = make_tensor_ptr(
      {3}, std::move(raw), executorch::aten::ScalarType::UInt32);
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::UInt32);
  auto ptr = tensor->const_data_ptr<uint32_t>();
  EXPECT_EQ(ptr[0], 1u);
  EXPECT_EQ(ptr[1], 4000000000u);
  EXPECT_EQ(ptr[2], 123u);
}

TEST_F(TensorPtrTest, TensorUint8dataUInt64Type) {
  std::vector<uint64_t> values = {0ull, 42ull, 9000000000000000000ull};
  const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
  std::vector<uint8_t> raw(bytes, bytes + values.size() * sizeof(uint64_t));
  auto tensor = make_tensor_ptr(
      {3}, std::move(raw), executorch::aten::ScalarType::UInt64);
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::UInt64);
  auto ptr = tensor->const_data_ptr<uint64_t>();
  EXPECT_EQ(ptr[0], 0ull);
  EXPECT_EQ(ptr[1], 42ull);
  EXPECT_EQ(ptr[2], 9000000000000000000ull);
}

TEST_F(TensorPtrTest, TensorUint8dataSizeMismatchUInt32ExpectDeath) {
  std::vector<uint8_t> data(
      3 * executorch::aten::elementSize(executorch::aten::ScalarType::UInt32) -
      1);
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({3}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, TensorUint8dataSizeMismatchUInt64ExpectDeath) {
  std::vector<uint8_t> data(
      2 * executorch::aten::elementSize(executorch::aten::ScalarType::UInt64) +
      1);
  ET_EXPECT_DEATH({ auto _ = make_tensor_ptr({2}, std::move(data)); }, "");
}

TEST_F(TensorPtrTest, TensorDataCastingFromInt32ToUInt16) {
  std::vector<int32_t> data = {-1, 65535, 65536, -65536};
  auto tensor =
      make_tensor_ptr(std::move(data), executorch::aten::ScalarType::UInt16);
  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::UInt16);
  auto ptr = tensor->const_data_ptr<uint16_t>();
  EXPECT_EQ(ptr[0], static_cast<uint16_t>(-1));
  EXPECT_EQ(ptr[1], static_cast<uint16_t>(65535));
  EXPECT_EQ(ptr[2], static_cast<uint16_t>(65536));
  EXPECT_EQ(ptr[3], static_cast<uint16_t>(-65536));
}

TEST_F(TensorPtrTest, TensorDataCastingFromUInt32ToFloat) {
  std::vector<uint32_t> data = {0u, 123u, 4000000000u};
  auto tensor =
      make_tensor_ptr(std::move(data), executorch::aten::ScalarType::Float);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::Float);
  auto ptr = tensor->const_data_ptr<float>();
  EXPECT_FLOAT_EQ(ptr[0], 0.0f);
  EXPECT_FLOAT_EQ(ptr[1], 123.0f);
  EXPECT_FLOAT_EQ(ptr[2], 4000000000.0f);
}

TEST_F(TensorPtrTest, TensorDataCastingFromFloatToUInt32) {
  std::vector<float> data = {1.0f, 2.0f};
  auto tensor =
      make_tensor_ptr(std::move(data), executorch::aten::ScalarType::UInt32);

  EXPECT_EQ(tensor->dim(), 1);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::UInt32);

  auto ptr = tensor->const_data_ptr<uint32_t>();
  EXPECT_EQ(ptr[0], 1u);
  EXPECT_EQ(ptr[1], 2u);
}

TEST_F(TensorPtrTest, MakeTensorPtrFromExistingTensorUInt32) {
  std::vector<uint32_t> data = {10u, 20u, 30u, 40u};
  auto tensor = make_tensor_ptr({2, 2}, data);
  auto alias = make_tensor_ptr(tensor);
  EXPECT_EQ(alias->dim(), 2);
  EXPECT_EQ(alias->size(0), 2);
  EXPECT_EQ(alias->size(1), 2);
  EXPECT_EQ(alias->scalar_type(), executorch::aten::ScalarType::UInt32);
  EXPECT_EQ(
      alias->const_data_ptr<uint32_t>(), tensor->const_data_ptr<uint32_t>());
}

TEST_F(TensorPtrTest, CloneTensorPtrFromExistingTensorUInt32) {
  std::vector<uint32_t> data = {10u, 20u, 30u, 40u};
  auto tensor = make_tensor_ptr({2, 2}, std::move(data));
  auto cloned = clone_tensor_ptr(tensor);
  EXPECT_EQ(cloned->dim(), 2);
  EXPECT_EQ(cloned->size(0), 2);
  EXPECT_EQ(cloned->size(1), 2);
  EXPECT_EQ(cloned->scalar_type(), executorch::aten::ScalarType::UInt32);
  EXPECT_NE(
      cloned->const_data_ptr<uint32_t>(), tensor->const_data_ptr<uint32_t>());
  auto ptr = cloned->const_data_ptr<uint32_t>();
  EXPECT_EQ(ptr[0], 10u);
  EXPECT_EQ(ptr[3], 40u);
}

TEST_F(TensorPtrTest, Tensor2DUInt16OwningData) {
  std::vector<uint16_t> data = {1u, 2u, 3u, 4u, 5u, 6u};
  auto tensor = make_tensor_ptr({2, 3}, std::move(data));
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
  EXPECT_EQ(tensor->strides()[0], 3);
  EXPECT_EQ(tensor->strides()[1], 1);
  EXPECT_EQ(tensor->scalar_type(), executorch::aten::ScalarType::UInt16);
  auto ptr = tensor->const_data_ptr<uint16_t>();
  EXPECT_EQ(ptr[0], 1u);
  EXPECT_EQ(ptr[5], 6u);
}
