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

TEST_F(TensorPtrMakerTest, CreateEmpty) {
  auto tensor = empty({4, 5});
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Float);

  auto tensor2 = empty({4, 5}, exec_aten::ScalarType::Int);
  EXPECT_EQ(tensor2->dim(), 2);
  EXPECT_EQ(tensor2->size(0), 4);
  EXPECT_EQ(tensor2->size(1), 5);
  EXPECT_EQ(tensor2->scalar_type(), exec_aten::ScalarType::Int);

  auto tensor3 = empty({4, 5}, exec_aten::ScalarType::Long);
  EXPECT_EQ(tensor3->dim(), 2);
  EXPECT_EQ(tensor3->size(0), 4);
  EXPECT_EQ(tensor3->size(1), 5);
  EXPECT_EQ(tensor3->scalar_type(), exec_aten::ScalarType::Long);

  auto tensor4 = empty({4, 5}, exec_aten::ScalarType::Double);
  EXPECT_EQ(tensor4->dim(), 2);
  EXPECT_EQ(tensor4->size(0), 4);
  EXPECT_EQ(tensor4->size(1), 5);
  EXPECT_EQ(tensor4->scalar_type(), exec_aten::ScalarType::Double);
}

TEST_F(TensorPtrMakerTest, CreateFull) {
  auto tensor = full({4, 5}, 7);
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Float);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 7);

  auto tensor2 = full({4, 5}, 3, exec_aten::ScalarType::Int);
  EXPECT_EQ(tensor2->dim(), 2);
  EXPECT_EQ(tensor2->size(0), 4);
  EXPECT_EQ(tensor2->size(1), 5);
  EXPECT_EQ(tensor2->scalar_type(), exec_aten::ScalarType::Int);
  EXPECT_EQ(tensor2->const_data_ptr<int32_t>()[0], 3);

  auto tensor3 = full({4, 5}, 9, exec_aten::ScalarType::Long);
  EXPECT_EQ(tensor3->dim(), 2);
  EXPECT_EQ(tensor3->size(0), 4);
  EXPECT_EQ(tensor3->size(1), 5);
  EXPECT_EQ(tensor3->scalar_type(), exec_aten::ScalarType::Long);
  EXPECT_EQ(tensor3->const_data_ptr<int64_t>()[0], 9);

  auto tensor4 = full({4, 5}, 11, exec_aten::ScalarType::Double);
  EXPECT_EQ(tensor4->dim(), 2);
  EXPECT_EQ(tensor4->size(0), 4);
  EXPECT_EQ(tensor4->size(1), 5);
  EXPECT_EQ(tensor4->scalar_type(), exec_aten::ScalarType::Double);
  EXPECT_EQ(tensor4->const_data_ptr<double>()[0], 11);
}

TEST_F(TensorPtrMakerTest, CreateScalar) {
  auto tensor = scalar_tensor(3.14f);

  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->numel(), 1);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Float);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 3.14f);

  auto tensor2 = scalar_tensor(5, exec_aten::ScalarType::Int);

  EXPECT_EQ(tensor2->dim(), 0);
  EXPECT_EQ(tensor2->numel(), 1);
  EXPECT_EQ(tensor2->scalar_type(), exec_aten::ScalarType::Int);
  EXPECT_EQ(tensor2->const_data_ptr<int32_t>()[0], 5);

  auto tensor3 = scalar_tensor(7.0, exec_aten::ScalarType::Double);

  EXPECT_EQ(tensor3->dim(), 0);
  EXPECT_EQ(tensor3->numel(), 1);
  EXPECT_EQ(tensor3->scalar_type(), exec_aten::ScalarType::Double);
  EXPECT_EQ(tensor3->const_data_ptr<double>()[0], 7.0);
}

TEST_F(TensorPtrMakerTest, CreateOnes) {
  auto tensor = ones({4, 5});
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Float);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 1);

  auto tensor2 = ones({4, 5}, exec_aten::ScalarType::Int);
  EXPECT_EQ(tensor2->dim(), 2);
  EXPECT_EQ(tensor2->size(0), 4);
  EXPECT_EQ(tensor2->size(1), 5);
  EXPECT_EQ(tensor2->scalar_type(), exec_aten::ScalarType::Int);
  EXPECT_EQ(tensor2->const_data_ptr<int32_t>()[0], 1);

  auto tensor3 = ones({4, 5}, exec_aten::ScalarType::Long);
  EXPECT_EQ(tensor3->dim(), 2);
  EXPECT_EQ(tensor3->size(0), 4);
  EXPECT_EQ(tensor3->size(1), 5);
  EXPECT_EQ(tensor3->scalar_type(), exec_aten::ScalarType::Long);
  EXPECT_EQ(tensor3->const_data_ptr<int64_t>()[0], 1);

  auto tensor4 = ones({4, 5}, exec_aten::ScalarType::Double);
  EXPECT_EQ(tensor4->dim(), 2);
  EXPECT_EQ(tensor4->size(0), 4);
  EXPECT_EQ(tensor4->size(1), 5);
  EXPECT_EQ(tensor4->scalar_type(), exec_aten::ScalarType::Double);
  EXPECT_EQ(tensor4->const_data_ptr<double>()[0], 1);
}

TEST_F(TensorPtrMakerTest, CreateZeros) {
  auto tensor = zeros({4, 5});
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Float);
  EXPECT_EQ(tensor->const_data_ptr<float>()[0], 0);

  auto tensor2 = zeros({4, 5}, exec_aten::ScalarType::Int);
  EXPECT_EQ(tensor2->dim(), 2);
  EXPECT_EQ(tensor2->size(0), 4);
  EXPECT_EQ(tensor2->size(1), 5);
  EXPECT_EQ(tensor2->scalar_type(), exec_aten::ScalarType::Int);
  EXPECT_EQ(tensor2->const_data_ptr<int32_t>()[0], 0);

  auto tensor3 = zeros({4, 5}, exec_aten::ScalarType::Long);
  EXPECT_EQ(tensor3->dim(), 2);
  EXPECT_EQ(tensor3->size(0), 4);
  EXPECT_EQ(tensor3->size(1), 5);
  EXPECT_EQ(tensor3->scalar_type(), exec_aten::ScalarType::Long);
  EXPECT_EQ(tensor3->const_data_ptr<int64_t>()[0], 0);

  auto tensor4 = zeros({4, 5}, exec_aten::ScalarType::Double);
  EXPECT_EQ(tensor4->dim(), 2);
  EXPECT_EQ(tensor4->size(0), 4);
  EXPECT_EQ(tensor4->size(1), 5);
  EXPECT_EQ(tensor4->scalar_type(), exec_aten::ScalarType::Double);
  EXPECT_EQ(tensor4->const_data_ptr<double>()[0], 0);
}

TEST_F(TensorPtrMakerTest, CreateRandTensor) {
  auto tensor = rand({4, 5});

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Float);

  for (auto i = 0; i < tensor->numel(); ++i) {
    auto val = tensor->const_data_ptr<float>()[i];
    EXPECT_GE(val, 0.0f);
    EXPECT_LT(val, 1.0f);
  }
}

TEST_F(TensorPtrMakerTest, CreateRandTensorWithIntType) {
  auto tensor = rand({4, 5}, exec_aten::ScalarType::Int);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Int);

  for (auto i = 0; i < tensor->numel(); ++i) {
    auto val = tensor->const_data_ptr<int32_t>()[i];
    EXPECT_EQ(val, 0);
  }
}

TEST_F(TensorPtrMakerTest, CreateRandTensorWithDoubleType) {
  auto tensor = rand({4, 5}, exec_aten::ScalarType::Double);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Double);

  for (auto i = 0; i < tensor->numel(); ++i) {
    auto val = tensor->const_data_ptr<double>()[i];
    EXPECT_GE(val, 0.0);
    EXPECT_LT(val, 1.0);
  }
}

TEST_F(TensorPtrMakerTest, CreateRandnTensor) {
  auto tensor = randn({100, 100});

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 100);
  EXPECT_EQ(tensor->size(1), 100);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Float);

  auto sum = 0.0f;
  for (auto i = 0; i < tensor->numel(); ++i) {
    sum += tensor->const_data_ptr<float>()[i];
  }
  const auto average = sum / tensor->numel();
  EXPECT_NEAR(average, 0.0f, 1.0f);
}

TEST_F(TensorPtrMakerTest, CreateRandnTensorWithDoubleType) {
  auto tensor = randn({100, 100}, exec_aten::ScalarType::Double);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 100);
  EXPECT_EQ(tensor->size(1), 100);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Double);

  auto sum = 0.0;
  for (auto i = 0; i < tensor->numel(); ++i) {
    sum += tensor->const_data_ptr<double>()[i];
  }
  const auto average = sum / tensor->numel();
  EXPECT_NEAR(average, 0.0, 1.0);
}

TEST_F(TensorPtrMakerTest, CreateRandIntTensorWithIntType) {
  auto tensor = randint(10, 20, {4, 5}, exec_aten::ScalarType::Int);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Int);

  for (auto i = 0; i < tensor->numel(); ++i) {
    auto val = tensor->const_data_ptr<int32_t>()[i];
    EXPECT_GE(val, 10);
    EXPECT_LT(val, 20);
  }
}

TEST_F(TensorPtrMakerTest, CreateRandIntTensorWithLongType) {
  auto tensor = randint(10, 20, {4, 5}, exec_aten::ScalarType::Long);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Long);

  for (auto i = 0; i < tensor->numel(); ++i) {
    auto val = tensor->const_data_ptr<int64_t>()[i];
    EXPECT_GE(val, 10);
    EXPECT_LT(val, 20);
  }
}

TEST_F(TensorPtrMakerTest, CreateRandnTensorWithIntType) {
  auto tensor = rand({4, 5}, exec_aten::ScalarType::Int);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 4);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->scalar_type(), exec_aten::ScalarType::Int);

  for (auto i = 0; i < tensor->numel(); ++i) {
    auto val = tensor->const_data_ptr<int32_t>()[i];
    EXPECT_EQ(val, 0);
  }
}
