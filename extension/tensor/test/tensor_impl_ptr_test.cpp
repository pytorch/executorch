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

TEST_F(TensorImplPtrTest, ScalarTensorCreation) {
  float scalar_data = 3.14f;
  auto tensor_impl = make_tensor_impl_ptr({}, &scalar_data);

  EXPECT_EQ(tensor_impl->numel(), 1);
  EXPECT_EQ(tensor_impl->dim(), 0);
  EXPECT_EQ(tensor_impl->sizes().size(), 0);
  EXPECT_EQ(tensor_impl->strides().size(), 0);
  EXPECT_EQ((float*)tensor_impl->data(), &scalar_data);
  EXPECT_EQ(((float*)tensor_impl->data())[0], 3.14f);
}

TEST_F(TensorImplPtrTest, ScalarTensorOwningData) {
  auto tensor_impl = make_tensor_impl_ptr({}, {3.14f});

  EXPECT_EQ(tensor_impl->numel(), 1);
  EXPECT_EQ(tensor_impl->dim(), 0);
  EXPECT_EQ(tensor_impl->sizes().size(), 0);
  EXPECT_EQ(tensor_impl->strides().size(), 0);
  EXPECT_EQ(((float*)tensor_impl->data())[0], 3.14f);
}

TEST_F(TensorImplPtrTest, TensorImplCreation) {
  float data[20] = {2};
  auto tensor_impl = make_tensor_impl_ptr({4, 5}, data, {0, 1}, {5, 1});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 4);
  EXPECT_EQ(tensor_impl->size(1), 5);
  EXPECT_EQ(tensor_impl->strides()[0], 5);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
  EXPECT_EQ(tensor_impl->data(), data);
  EXPECT_EQ(tensor_impl->data(), data);
  EXPECT_EQ(((float*)tensor_impl->data())[0], 2);
}

TEST_F(TensorImplPtrTest, TensorImplSharedOwnership) {
  float data[20] = {2};
  auto tensor_impl1 = make_tensor_impl_ptr({4, 5}, data);
  auto tensor_impl2 = tensor_impl1;

  EXPECT_EQ(tensor_impl1.get(), tensor_impl2.get());
  EXPECT_EQ(tensor_impl1.use_count(), tensor_impl2.use_count());

  tensor_impl1.reset();
  EXPECT_EQ(tensor_impl2.use_count(), 1);
  EXPECT_NE(tensor_impl2.get(), nullptr);
}

TEST_F(TensorImplPtrTest, TensorImplInferredDimOrderAndStrides) {
  float data[12] = {0};
  auto tensor_impl = make_tensor_impl_ptr({3, 4}, data, {}, {4, 1});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->size(1), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 4);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
  EXPECT_EQ(tensor_impl->data(), data);
}

TEST_F(TensorImplPtrTest, TensorImplInferredDimOrderCustomStrides) {
  float data[12] = {0};
  auto tensor_impl = make_tensor_impl_ptr({3, 4}, data, {}, {1, 3});

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->size(1), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 1);
  EXPECT_EQ(tensor_impl->strides()[1], 3);
}

TEST_F(TensorImplPtrTest, TensorImplDefaultDimOrderAndStrides) {
  float data[24] = {0};
  auto tensor_impl = make_tensor_impl_ptr({2, 3, 4}, data);

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
      { auto _ = make_tensor_impl_ptr({3, 4}, data, {1, 0}, {1, 4}); }, "");
}

TEST_F(TensorImplPtrTest, TensorImplCustomDimOrderAndStrides) {
  float data[12] = {0};
  auto tensor_impl = make_tensor_impl_ptr({3, 4}, data, {1, 0}, {1, 3});

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
        auto _ = make_tensor_impl_ptr({4, 5}, data, {2, 1}, {1, 4});
      },
      "");
}

TEST_F(TensorImplPtrTest, TensorImplCustomDeleter) {
  float data[20] = {4};
  auto tensor_impl = make_tensor_impl_ptr({4, 5}, data);

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
      {4, 5},
      data_ptr.get(),
      {},
      {},
      exec_aten::ScalarType::Float,
      exec_aten::TensorShapeDynamism::DYNAMIC_BOUND,
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
  auto tensor_impl = make_tensor_impl_ptr({0, 5}, std::vector<float>());

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 0);
  EXPECT_EQ(tensor_impl->size(1), 5);
  EXPECT_EQ(tensor_impl->strides()[0], 5);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
  EXPECT_EQ(tensor_impl->data(), nullptr);
}

TEST_F(TensorImplPtrTest, TensorImplDataOnlyDoubleType) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  auto tensor_impl = make_tensor_impl_ptr(std::move(data));

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 1);
  EXPECT_EQ(((double*)tensor_impl->data())[0], 1.0);
  EXPECT_EQ(((double*)tensor_impl->data())[3], 4.0);
}

TEST_F(TensorImplPtrTest, TensorImplDataOnlyInt32Type) {
  std::vector<int32_t> data = {10, 20, 30, 40};
  auto tensor_impl = make_tensor_impl_ptr(std::move(data));

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 1);
  EXPECT_EQ(((int32_t*)tensor_impl->data())[0], 10);
  EXPECT_EQ(((int32_t*)tensor_impl->data())[3], 40);
}

TEST_F(TensorImplPtrTest, TensorImplDataOnlyInt64Type) {
  std::vector<int64_t> data = {100, 200, 300, 400};
  auto tensor_impl = make_tensor_impl_ptr(std::move(data));

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 1);
  EXPECT_EQ(((int64_t*)tensor_impl->data())[0], 100);
  EXPECT_EQ(((int64_t*)tensor_impl->data())[3], 400);
}

TEST_F(TensorImplPtrTest, TensorImplDataOnlyUint8Type) {
  std::vector<uint8_t> data = {10, 20, 30, 40};
  auto tensor_impl = make_tensor_impl_ptr(std::move(data));

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 4);
  EXPECT_EQ(tensor_impl->strides()[0], 1);
  EXPECT_EQ(((uint8_t*)tensor_impl->data())[0], 10);
  EXPECT_EQ(((uint8_t*)tensor_impl->data())[3], 40);
}

TEST_F(TensorImplPtrTest, TensorImplAmbiguityWithMixedVectors) {
  std::vector<exec_aten::SizesType> sizes = {2, 2};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor_impl = make_tensor_impl_ptr(std::move(sizes), std::move(data));

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 2);
  EXPECT_EQ(tensor_impl->size(1), 2);
  EXPECT_EQ(tensor_impl->strides()[0], 2);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
  EXPECT_EQ(((float*)tensor_impl->data())[0], 1.0f);
  EXPECT_EQ(((float*)tensor_impl->data())[3], 4.0f);

  auto tensor_impl2 = make_tensor_impl_ptr({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  EXPECT_EQ(tensor_impl2->dim(), 2);
  EXPECT_EQ(tensor_impl2->size(0), 2);
  EXPECT_EQ(tensor_impl2->size(1), 2);
  EXPECT_EQ(tensor_impl2->strides()[0], 2);
  EXPECT_EQ(tensor_impl2->strides()[1], 1);
  EXPECT_EQ(((float*)tensor_impl2->data())[0], 1.0f);
  EXPECT_EQ(((float*)tensor_impl2->data())[3], 4.0f);
}

TEST_F(TensorImplPtrTest, SharedDataManagement) {
  auto data = std::make_shared<std::vector<float>>(100, 1.0f);
  auto tensor_impl1 = make_tensor_impl_ptr({10, 10}, data->data());
  auto tensor_impl2 = tensor_impl1;

  EXPECT_EQ(tensor_impl1.get(), tensor_impl2.get());
  EXPECT_EQ(tensor_impl1.use_count(), 2);
  EXPECT_EQ(((float*)tensor_impl1->data())[0], 1.0f);

  ((float*)tensor_impl1->mutable_data())[0] = 2.0f;
  EXPECT_EQ(((float*)tensor_impl2->data())[0], 2.0f);

  tensor_impl1.reset();
  EXPECT_NE(tensor_impl2.get(), nullptr);
  EXPECT_EQ(tensor_impl2.use_count(), 1);

  EXPECT_EQ(((float*)tensor_impl2->data())[0], 2.0f);
}

TEST_F(TensorImplPtrTest, CustomDeleterWithSharedData) {
  auto data = std::make_shared<std::vector<float>>(100, 1.0f);
  bool deleter_called = false;
  {
    auto tensor_impl = make_tensor_impl_ptr(
        {10, 10},
        data->data(),
        {},
        {},
        exec_aten::ScalarType::Float,
        exec_aten::TensorShapeDynamism::DYNAMIC_BOUND,
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

TEST_F(TensorImplPtrTest, TensorImplDeducedScalarType) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
  auto tensor_impl = make_tensor_impl_ptr({2, 2}, std::move(data));

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 2);
  EXPECT_EQ(tensor_impl->size(1), 2);
  EXPECT_EQ(tensor_impl->strides()[0], 2);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
  EXPECT_EQ(((double*)tensor_impl->data())[0], 1.0);
  EXPECT_EQ(((double*)tensor_impl->data())[3], 4.0);
}

TEST_F(TensorImplPtrTest, TensorImplUint8BufferWithFloatScalarType) {
  std::vector<uint8_t> data(
      4 * exec_aten::elementSize(exec_aten::ScalarType::Float));

  float* float_data = reinterpret_cast<float*>(data.data());
  float_data[0] = 1.0f;
  float_data[1] = 2.0f;
  float_data[2] = 3.0f;
  float_data[3] = 4.0f;

  auto tensor_impl = make_tensor_impl_ptr({2, 2}, std::move(data));

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 2);
  EXPECT_EQ(tensor_impl->size(1), 2);
  EXPECT_EQ(tensor_impl->strides()[0], 2);
  EXPECT_EQ(tensor_impl->strides()[1], 1);

  EXPECT_EQ(((float*)tensor_impl->data())[0], 1.0f);
  EXPECT_EQ(((float*)tensor_impl->data())[1], 2.0f);
  EXPECT_EQ(((float*)tensor_impl->data())[2], 3.0f);
  EXPECT_EQ(((float*)tensor_impl->data())[3], 4.0f);
}

TEST_F(TensorImplPtrTest, TensorImplUint8BufferTooSmallExpectDeath) {
  std::vector<uint8_t> data(
      2 * exec_aten::elementSize(exec_aten::ScalarType::Float));
  ET_EXPECT_DEATH(
      { auto tensor_impl = make_tensor_impl_ptr({2, 2}, std::move(data)); },
      "");
}

TEST_F(TensorImplPtrTest, TensorImplUint8BufferTooLarge) {
  std::vector<uint8_t> data(
      4 * exec_aten::elementSize(exec_aten::ScalarType::Float));
  auto tensor_impl = make_tensor_impl_ptr({2, 2}, std::move(data));

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 2);
  EXPECT_EQ(tensor_impl->size(1), 2);
  EXPECT_EQ(tensor_impl->strides()[0], 2);
  EXPECT_EQ(tensor_impl->strides()[1], 1);
}

TEST_F(TensorImplPtrTest, StridesAndDimOrderMustMatchSizes) {
  float data[12] = {0};
  ET_EXPECT_DEATH(
      { auto _ = make_tensor_impl_ptr({3, 4}, data, {}, {1}); }, "");
  ET_EXPECT_DEATH(
      { auto _ = make_tensor_impl_ptr({3, 4}, data, {0}, {4, 1}); }, "");
}

TEST_F(TensorImplPtrTest, TensorDataCastingFromIntToFloat) {
  std::vector<int32_t> int_data = {1, 2, 3, 4, 5, 6};
  auto tensor_impl = make_tensor_impl_ptr(
      {2, 3}, std::move(int_data), {}, {}, exec_aten::ScalarType::Float);

  EXPECT_EQ(tensor_impl->dim(), 2);
  EXPECT_EQ(tensor_impl->size(0), 2);
  EXPECT_EQ(tensor_impl->size(1), 3);
  EXPECT_EQ(tensor_impl->dtype(), exec_aten::ScalarType::Float);

  auto data_ptr = static_cast<const float*>(tensor_impl->data());
  EXPECT_FLOAT_EQ(data_ptr[0], 1.0f);
  EXPECT_FLOAT_EQ(data_ptr[5], 6.0f);
}

TEST_F(TensorImplPtrTest, TensorDataCastingFromIntToDouble) {
  std::vector<int32_t> int_data = {1, 2, 3};
  auto tensor_impl =
      make_tensor_impl_ptr(std::move(int_data), exec_aten::ScalarType::Double);

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->dtype(), exec_aten::ScalarType::Double);

  auto data_ptr = static_cast<const double*>(tensor_impl->data());
  EXPECT_DOUBLE_EQ(data_ptr[0], 1.0);
  EXPECT_DOUBLE_EQ(data_ptr[1], 2.0);
  EXPECT_DOUBLE_EQ(data_ptr[2], 3.0);
}

TEST_F(TensorImplPtrTest, TensorDataCastingInvalidCast) {
  std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
  ET_EXPECT_DEATH(
      {
        auto _ = make_tensor_impl_ptr(
            std::move(float_data), exec_aten::ScalarType::Int);
      },
      "");
}

TEST_F(TensorImplPtrTest, TensorDataCastingFromFloatToHalf) {
  std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
  auto tensor_impl =
      make_tensor_impl_ptr(std::move(float_data), exec_aten::ScalarType::Half);

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->dtype(), exec_aten::ScalarType::Half);

  auto data_ptr = static_cast<const exec_aten::Half*>(tensor_impl->data());
  EXPECT_EQ(static_cast<float>(data_ptr[0]), 1.0f);
  EXPECT_EQ(static_cast<float>(data_ptr[1]), 2.0f);
  EXPECT_EQ(static_cast<float>(data_ptr[2]), 3.0f);
}

TEST_F(TensorImplPtrTest, TensorDataCastingFromDoubleToFloat) {
  std::vector<double> double_data = {1.1, 2.2, 3.3};
  auto tensor_impl = make_tensor_impl_ptr(
      std::move(double_data), exec_aten::ScalarType::Float);

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->dtype(), exec_aten::ScalarType::Float);

  auto data_ptr = static_cast<const float*>(tensor_impl->data());
  EXPECT_FLOAT_EQ(data_ptr[0], 1.1f);
  EXPECT_FLOAT_EQ(data_ptr[1], 2.2f);
  EXPECT_FLOAT_EQ(data_ptr[2], 3.3f);
}

TEST_F(TensorImplPtrTest, TensorDataCastingFromInt64ToInt32) {
  std::vector<int64_t> int64_data = {10000000000, 20000000000, 30000000000};
  auto tensor_impl =
      make_tensor_impl_ptr(std::move(int64_data), exec_aten::ScalarType::Int);

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->dtype(), exec_aten::ScalarType::Int);

  auto data_ptr = static_cast<const int32_t*>(tensor_impl->data());
  // Since the values exceed int32_t range, they may overflow
  // Here we just check that the cast was performed
  EXPECT_NE(data_ptr[0], 10000000000); // Expected overflow
}

TEST_F(TensorImplPtrTest, TensorDataCastingFromFloatToBFloat16) {
  std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
  auto tensor_impl = make_tensor_impl_ptr(
      std::move(float_data), exec_aten::ScalarType::BFloat16);

  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->dtype(), exec_aten::ScalarType::BFloat16);

  auto data_ptr = static_cast<const exec_aten::BFloat16*>(tensor_impl->data());
  EXPECT_EQ(static_cast<float>(data_ptr[0]), 1.0f);
  EXPECT_EQ(static_cast<float>(data_ptr[1]), 2.0f);
  EXPECT_EQ(static_cast<float>(data_ptr[2]), 3.0f);
}

TEST_F(TensorImplPtrTest, InitializerListDoubleToHalf) {
  auto tensor_impl = make_tensor_impl_ptr<double>(
      {1.5, 2.7, 3.14}, exec_aten::ScalarType::Half);
  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 3);
  EXPECT_EQ(tensor_impl->dtype(), exec_aten::ScalarType::Half);
  auto data_ptr = static_cast<const exec_aten::Half*>(tensor_impl->data());
  EXPECT_NEAR(static_cast<float>(data_ptr[0]), 1.5f, 0.01);
  EXPECT_NEAR(static_cast<float>(data_ptr[1]), 2.7f, 0.01);
  EXPECT_NEAR(static_cast<float>(data_ptr[2]), 3.14f, 0.01);
}

TEST_F(TensorImplPtrTest, InitializerListInt8ToInt64) {
  auto tensor_impl =
      make_tensor_impl_ptr<int8_t>({1, -2, 3, -4}, exec_aten::ScalarType::Long);
  EXPECT_EQ(tensor_impl->dim(), 1);
  EXPECT_EQ(tensor_impl->size(0), 4);
  EXPECT_EQ(tensor_impl->dtype(), exec_aten::ScalarType::Long);
  auto data_ptr = static_cast<const int64_t*>(tensor_impl->data());
  EXPECT_EQ(data_ptr[0], 1);
  EXPECT_EQ(data_ptr[1], -2);
  EXPECT_EQ(data_ptr[2], 3);
  EXPECT_EQ(data_ptr[3], -4);
}
