/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

#ifdef USE_ATEN_LIB
#include <ATen/ATen.h>
#endif // USE_ATEN_LIB

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::SizesType;
using exec_aten::StridesType;
using exec_aten::Tensor;
using exec_aten::TensorList;
using executorch::runtime::Error;
using executorch::runtime::resize_tensor;
using executorch::runtime::TensorShapeDynamism;
using executorch::runtime::testing::TensorFactory;
using executorch::runtime::testing::TensorListFactory;

// The tensor under test will be modified so pass an rvalue ref
void resize_tensor_to_assert_static(Tensor&& t) {
  ASSERT_GT(t.numel(), 1)
      << "Need to resize to an 1x1 tensor, so the input size should be > 1";
  // Resizing to an 1x1 tensor doesn't work means it is static
#ifdef USE_ATEN_LIB
  EXPECT_EQ(resize_tensor(t, ArrayRef<SizesType>({1, 1})), Error::Ok);
#else
  EXPECT_NE(resize_tensor(t, ArrayRef<SizesType>({1, 1})), Error::Ok);
#endif
}

// The tensor under test will be modified so pass an rvalue ref
void resize_tensor_to_assert_dynamic_bound(Tensor&& t) {
  ASSERT_GT(t.numel(), 1)
      << "Need to resize to an 1x1 tensor, so the input size should be > 1";
  ASSERT_LT(t.numel(), 100 * 100)
      << "Need to resize to an 100x100 tensor, so the input size should be < 10000";
#ifdef USE_ATEN_LIB
  EXPECT_EQ(resize_tensor(t, ArrayRef<SizesType>({1, 1})), Error::Ok);
  EXPECT_EQ(resize_tensor(t, ArrayRef<SizesType>({100, 100})), Error::Ok);
#else
  EXPECT_EQ(resize_tensor(t, ArrayRef<SizesType>({1, 1})), Error::Ok);
  EXPECT_NE(resize_tensor(t, ArrayRef<SizesType>({100, 100})), Error::Ok);
#endif
}

// The tensor under test will be modified so pass an rvalue ref
void resize_tensor_to_assert_dynamic_unbound(Tensor&& t) {
  ASSERT_GT(t.numel(), 1)
      << "Need to resize to an 1x1 tensor, so the input size should be > 1";
  ASSERT_LT(t.numel(), 100 * 100)
      << "Need to resize to an 100x100 tensor, so the input size should be < 10000";
  EXPECT_EQ(resize_tensor(t, ArrayRef<SizesType>({1, 1})), Error::Ok);

#ifdef USE_ATEN_LIB
  EXPECT_EQ(resize_tensor(t, ArrayRef<SizesType>({100, 100})), Error::Ok);
#else
  // TODO(T175194371): For now, we can't resize past the original capacity.
  EXPECT_NE(resize_tensor(t, ArrayRef<SizesType>({100, 100})), Error::Ok);
#endif
}

#ifndef USE_ATEN_LIB
using exec_aten::DimOrderType;
using torch::executor::TensorImpl;
#endif // !USE_ATEN_LIB

#define CHECK_ARRAY_REF_EQUAL(a1, a2)                                \
  ET_CHECK_MSG(                                                      \
      a1.size() == a2.size(),                                        \
      "Arrays are not equal size." #a1 " size:%zu," #a2 " size:%zu", \
      a1.size(),                                                     \
      a2.size());                                                    \
  for (size_t i = 0; i < a1.size(); ++i) {                           \
    ET_CHECK_MSG(                                                    \
        a1[i] == a2[i],                                              \
        "Value mismatch at index %zu, " #a1                          \
        "[%zu] = %hd"                                                \
        ", " #a2 "[%zu] = %hd",                                      \
        i,                                                           \
        i,                                                           \
        a1[i],                                                       \
        i,                                                           \
        a2[i]);                                                      \
  }

//
// Tests for TensorFactory
//

class TensorFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

TEST_F(TensorFactoryTest, MakeIntTensor) {
  TensorFactory<ScalarType::Int> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});

  // A Tensor created manually, that should be identical to `actual`.
  int32_t data[] = {1, 2, 3, 4};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  Tensor expected =
      at::zeros(at::IntArrayRef(sizes), at::dtype(ScalarType::Int));
  memcpy(expected.mutable_data_ptr<int32_t>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {0, 1};
  int32_t strides[dim] = {2, 1}; // Contiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Int, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

// Test Float as a proxy for non-Int dtypes. Don't test as thoroughly as Int,
// which we use for the bulk of the test coverage.
TEST_F(TensorFactoryTest, MakeFloatTensor) {
  TensorFactory<ScalarType::Float> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.make(/*sizes=*/{2, 2}, /*data=*/{1.1, 2.2, 3.3, 4.4});

  // A Tensor created manually, that should be identical to `actual`.
  float data[] = {1.1, 2.2, 3.3, 4.4};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  Tensor expected =
      at::zeros(at::IntArrayRef(sizes), at::dtype(ScalarType::Float));
  memcpy(expected.mutable_data_ptr<float>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {0, 1};
  int32_t strides[dim] = {2, 1}; // Contiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Float, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

// Also test Bool, which is a bit of a special case because the factory uses
// `uint8_t` instead of `bool` as the underlying C type.
TEST_F(TensorFactoryTest, MakeBoolTensor) {
  TensorFactory<ScalarType::Bool> tf;

  // A Tensor created by the factory.
  Tensor actual =
      tf.make(/*sizes=*/{2, 2}, /*data=*/{true, false, true, false});

  // A Tensor created manually, that should be identical to `actual`.
  bool data[] = {true, false, true, false};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  Tensor expected =
      at::zeros(at::IntArrayRef(sizes), at::dtype(ScalarType::Bool));
  memcpy(expected.mutable_data_ptr<bool>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {0, 1};
  int32_t strides[dim] = {2, 1}; // Contiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Bool, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, DataIsCopied) {
  TensorFactory<ScalarType::Int> tf;

  // Create two tensors using the same input data vector.
  std::vector<int32_t> data = {1, 2, 3, 4};
  Tensor t1 = tf.make(/*sizes=*/{2, 2}, data);
  Tensor t2 = tf.make(/*sizes=*/{2, 2}, data);

  // Demonstrate that they start out identical.
  EXPECT_TENSOR_EQ(t1, t2);

  // Modify data in one of them.
  t1.mutable_data_ptr<int32_t>()[0] = 99;

  // If they shared the same underlying data, they would still be identical,
  // but since they use copies they should now be different.
  EXPECT_TENSOR_NE(t1, t2);
}

TEST_F(TensorFactoryTest, DefaultStridesAreContiguous) {
  TensorFactory<ScalarType::Int> tf;

  // Create a Tensor with prime number dimensions to more clearly show how the
  // strides will be constructed. 30 is 2 * 3 * 5.
  Tensor t1 =
      tf.make(/*sizes=*/{2, 3, 5}, /*data=*/std::vector<int32_t>(30, 99));

  // Get the strides into a vector for easy comparison.
  std::vector<int32_t> actual_strides(t1.strides().begin(), t1.strides().end());

  // TensorFactory should have generated a strides list for contiguous data.
  // sizes: {D, H, W}
  // strides: {H*W, W, 1}
  // expected: {3*5, 5, 1}
  std::vector<int32_t> expected_strides = {15, 5, 1};

  EXPECT_EQ(expected_strides, actual_strides);
}

TEST_F(TensorFactoryTest, StridesForEmptyTensor) {
  TensorFactory<ScalarType::Int> tf;

  // Create a Tensor with some dim = 0 to clearly demonstrate how the tf.make
  // handle empty tensors.
  Tensor t1 =
      tf.make(/*sizes=*/{2, 0, 3, 0, 5}, /*data=*/std::vector<int32_t>());

  // Get the strides into a vector for easy comparison.
  std::vector<int32_t> actual_strides(t1.strides().begin(), t1.strides().end());

  // When calculating stride for empty data, TensorFactory should take dim whose
  // length equals to zero as equals to one. So the t1's stride should be same
  // as tensor(size=(2, 1, 3, 1, 5))
  // expected: {5*1*3*1, 5*1*3, 5*1, 5, 1} -> {15, 15, 5, 5, 1}
  std::vector<int32_t> expected_strides = {15, 15, 5, 5, 1};

  EXPECT_EQ(expected_strides, actual_strides);
}

TEST_F(TensorFactoryTest, StridesForZeroDimTensor) {
  TensorFactory<ScalarType::Int> tf;

  // Create a Tensor with zero dimension to clearly demonstrate how the tf.make
  // handle zero dim tensors.
  Tensor t1 = tf.make(
      /*sizes=*/{},
      /*data=*/{1});

  // Get the strides into a vector for easy comparison.
  std::vector<int32_t> actual_strides(t1.strides().begin(), t1.strides().end());

  // Zero dimension tensor should have zero dimension stride
  // expected: {}
  std::vector<int32_t> expected_strides = {};

  EXPECT_EQ(expected_strides, actual_strides);
}

TEST_F(TensorFactoryTest, NotEnoughDataDies) {
  TensorFactory<ScalarType::Int> tf;

  // Provide three data elements when the tensor needs four.
  ET_EXPECT_DEATH(tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3}), "");
}

TEST_F(TensorFactoryTest, TooMuchDataDies) {
  TensorFactory<ScalarType::Int> tf;

  // Provide five data elements when the tensor needs four.
  ET_EXPECT_DEATH(tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4, 5}), "");
}

//
// Tests for TensorFactory::make()
//

// Test if factory can produce strided incontiguous int tensors.
TEST_F(TensorFactoryTest, MakeStridedIntTensor) {
  TensorFactory<ScalarType::Int> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.make(
      /*sizes=*/{2, 2},
      /*data=*/{1, 2, 3, 4},
      /*strides=*/{1, 2} /*incontiguous tensor*/);

  // A Tensor created manually, that should be identical to `actual`.
  int32_t data[] = {1, 2, 3, 4};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  std::vector<int64_t> strides = {1, 2};
  Tensor expected = at::empty_strided(
      sizes,
      strides,
      ScalarType::Int,
      /*layout_opt=*/at::Layout::Strided,
      /*device_opt=*/at::Device(at::DeviceType::CPU),
      /*pin_memory_opt=*/false);
  memcpy(expected.mutable_data_ptr<int>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {1, 0};
  int32_t strides[dim] = {1, 2}; // Incontiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Int, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

// Test if factory can produce strided incontiguous float tensors.
TEST_F(TensorFactoryTest, MakeStridedFloatTensor) {
  TensorFactory<ScalarType::Float> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.make(
      /*sizes=*/{2, 2},
      /*data=*/{1.1, 2.2, 3.3, 4.4},
      /*strides=*/{1, 2} /*incontiguous tensor*/);

  // A Tensor created manually, that should be identical to `actual`.
  float data[] = {1.1, 2.2, 3.3, 4.4};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  std::vector<int64_t> strides = {1, 2};
  Tensor expected = at::empty_strided(
      sizes,
      strides,
      ScalarType::Float,
      /*layout_opt=*/at::Layout::Strided,
      /*device_opt=*/at::Device(at::DeviceType::CPU),
      /*pin_memory_opt=*/false);
  memcpy(expected.mutable_data_ptr<float>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {1, 0};
  int32_t strides[dim] = {1, 2}; // Incontiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Float, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

// Test if factory can produce incontiguous strided bool tensors.
TEST_F(TensorFactoryTest, MakeStridedBoolTensor) {
  TensorFactory<ScalarType::Bool> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.make(
      /*sizes=*/{2, 2},
      /*data=*/{true, false, true, false},
      /*strides=*/{1, 2} /*incontiguous tensor*/);

  // A Tensor created manually, that should be identical to `actual`.
  bool data[] = {true, false, true, false};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  std::vector<int64_t> strides = {1, 2};
  Tensor expected = at::empty_strided(
      sizes,
      strides,
      ScalarType::Bool,
      /*layout_opt=*/at::Layout::Strided,
      /*device_opt=*/at::Device(at::DeviceType::CPU),
      /*pin_memory_opt=*/false);
  memcpy(expected.mutable_data_ptr<bool>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {1, 0};
  int32_t strides[dim] = {1, 2}; // Incontiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Bool, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

// Test if factory can test the legitimacy of given `strides`, when `strides`
// contains multiple zeros and ones.
TEST_F(TensorFactoryTest, MakeStridedSameStrideTensorSupported) {
  TensorFactory<ScalarType::Bool> tf;

  // A Tensor created by the factory.
  // actual = tensor(size=(2, 0 ,3, 0, 0, 0, 1, 5, 0, 1, 2)).permute(0, 6, 8, 2,
  // 7, 10, 9, 3, 5, 4, 1)
  Tensor actual = tf.make(
      /*sizes=*/{2, 1, 0, 3, 5, 2, 1, 0, 0, 0, 0},
      /*data=*/{},
      /*strides=*/
      {30, 10, 2, 10, 2, 1, 2, 10, 10, 10, 30} /*incontiguous tensor*/);

#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 1, 0, 3, 5, 2, 1, 0, 0, 0, 0};
  std::vector<int64_t> strides = {30, 10, 2, 10, 2, 1, 2, 10, 10, 10, 30};
  Tensor expected = at::empty_strided(
      sizes,
      strides,
      ScalarType::Bool,
      /*layout_opt=*/at::Layout::Strided,
      /*device_opt=*/at::Device(at::DeviceType::CPU),
      /*pin_memory_opt=*/false);
#else // !USE_ATEN_LIB
  constexpr size_t dim = 11;
  int32_t sizes[dim] = {2, 1, 0, 3, 5, 2, 1, 0, 0, 0, 0};
  uint8_t dim_order[dim] = {0, 10, 1, 3, 7, 8, 9, 2, 4, 6, 5};
  int32_t strides[dim] = {
      30, 10, 2, 10, 2, 1, 2, 10, 10, 10, 30}; // Incontiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Bool, dim, sizes, {}, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, MakeStridedDataIsCopied) {
  TensorFactory<ScalarType::Int> tf;

  // Create two tensors using the same input data and strided vector.
  std::vector<int32_t> data = {1, 2, 3, 4};
  std::vector<exec_aten::StridesType> strides = {1, 2};
  Tensor t1 = tf.make(/*sizes=*/{2, 2}, data, strides);
  Tensor t2 = tf.make(/*sizes=*/{2, 2}, data, strides);

  // Demonstrate that they start out identical.
  EXPECT_TENSOR_EQ(t1, t2);

  // Modify data in one of them.
  t1.mutable_data_ptr<int32_t>()[0] = 99;

  // If they shared the same underlying data, they would still be identical,
  // but since they use copies they should now be different.
  EXPECT_TENSOR_NE(t1, t2);
}

TEST_F(TensorFactoryTest, MakeStridedEmptyDataSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor actual = tf.make(
      /*sizes=*/{2, 0, 3, 0, 5}, /*data=*/{}, /*strides=*/{15, 15, 5, 5, 1});
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 0, 3, 0, 5};
  std::vector<int64_t> strides = {15, 15, 5, 5, 1};
  Tensor expected = at::empty_strided(
      sizes,
      strides,
      ScalarType::Int,
      /*layout_opt=*/at::Layout::Strided,
      /*device_opt=*/at::Device(at::DeviceType::CPU),
      /*pin_memory_opt=*/false);
#else // !USE_ATEN_LIB
  constexpr size_t dim = 5;
  int32_t sizes[dim] = {2, 0, 3, 0, 5};
  uint8_t dim_order[dim] = {0, 1, 2, 3, 4};
  int32_t strides[dim] = {15, 15, 5, 5, 1}; // Incontiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Int, dim, sizes, /*data=*/{}, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, MakeStridedZeroDimSupported) {
  TensorFactory<ScalarType::Int> tf;

  // Provide three data elements when the tensor needs four.
  Tensor actual = tf.make(/*sizes=*/{}, /*data=*/{1}, /*strides=*/{});
  int32_t data[] = {1};

#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {};
  std::vector<int64_t> strides = {};
  Tensor expected = at::empty_strided(
      sizes,
      strides,
      ScalarType::Int,
      /*layout_opt=*/at::Layout::Strided,
      /*device_opt=*/at::Device(at::DeviceType::CPU),
      /*pin_memory_opt=*/false);
  memcpy(expected.mutable_data_ptr<int>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 0;
  int32_t sizes[dim] = {};
  uint8_t dim_order[dim] = {};
  int32_t strides[dim] = {};
  TensorImpl impl =
      TensorImpl(ScalarType::Int, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, MakeStridedNotEnoughDataDie) {
  TensorFactory<ScalarType::Int> tf;

  // Provide three data elements when the tensor needs four.
  ET_EXPECT_DEATH(
      tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3}, /*strides=*/{1, 2}), "");
}

TEST_F(TensorFactoryTest, MakeStridedTooMuchDataDie) {
  TensorFactory<ScalarType::Int> tf;

  // Provide five data elements when the tensor needs four.
  ET_EXPECT_DEATH(
      tf.make(
          /*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4, 5}, /*strides=*/{1, 2}),
      "");
}

TEST_F(TensorFactoryTest, MakeStridedNotEnoughStrideDie) {
  TensorFactory<ScalarType::Int> tf;

  // Provide one stride when the tensor needs two.
  ET_EXPECT_DEATH(
      tf.make(
          /*sizes=*/{2, 2},
          /*data=*/{1, 2, 3, 4},
          /*strides=*/{1}),
      "");
}

TEST_F(TensorFactoryTest, MakeStridedTooMuchStrideDie) {
  TensorFactory<ScalarType::Int> tf;

  // Provide three strides when the tensor needs two.
  ET_EXPECT_DEATH(
      tf.make(
          /*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4}, /*strides=*/{1, 2, 1}),
      "");
}

TEST_F(TensorFactoryTest, MakeStridedTooLargeStrideDie) {
  TensorFactory<ScalarType::Int> tf;

  // Stride is too large. e.g. [0, 1] is the 4th element of data, which jump out
  // the bounds [0, 3]
  ET_EXPECT_DEATH(
      tf.make(
          /*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4}, /*strides=*/{1, 4}),
      "");
}

TEST_F(TensorFactoryTest, MakeStridedTooSmalleStrideDie) {
  TensorFactory<ScalarType::Int> tf;

  // Stride is too small (only 0th element can be accessed).
  ET_EXPECT_DEATH(
      tf.make(
          /*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4}, /*strides=*/{1, 1}),
      "");
}

TEST_F(TensorFactoryTest, MakeStridedNonPositiveStrideDie) {
  TensorFactory<ScalarType::Int> tf;

  // Stride shall be positive.
  ET_EXPECT_DEATH(
      tf.make(
          /*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4}, /*strides=*/{2, -1}),
      "");
}

TEST_F(TensorFactoryTest, MakeStridedWrongStrideForEmptyDataDie) {
  TensorFactory<ScalarType::Int> tf;

  // When calculating strides based on sizes, we need to treat the size of
  // dimension equals to zero as one.
  ET_EXPECT_DEATH(
      tf.make(/*sizes=*/{0, 2, 2}, /*data=*/{}, /*strides=*/{0, 2, 1}), "");
}

TEST_F(TensorFactoryTest, MakeStridedWrongStrideForZeroDimDataDie) {
  TensorFactory<ScalarType::Int> tf;

  // Stride should be empty
  ET_EXPECT_DEATH(tf.make(/*sizes=*/{}, /*data=*/{1}, /*strides=*/{0}), "");
}

TEST_F(TensorFactoryTest, Full) {
  TensorFactory<ScalarType::Int> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.full(/*sizes=*/{2, 2}, 5);

  // A Tensor created manually, that should be identical to `actual`.
  int32_t data[] = {5, 5, 5, 5};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  Tensor expected =
      at::zeros(at::IntArrayRef(sizes), at::dtype(ScalarType::Int));
  memcpy(expected.mutable_data_ptr<int32_t>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {0, 1};
  int32_t strides[dim] = {2, 1}; // Contiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Int, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

// Use Float as a proxy for demonstrating that full/zeros/ones works for non-Int
// dtypes.
TEST_F(TensorFactoryTest, FullFloat) {
  TensorFactory<ScalarType::Float> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.full(/*sizes=*/{2, 2}, 5.5);

  // A Tensor created manually, that should be identical to `actual`.
  float data[] = {5.5f, 5.5f, 5.5f, 5.5f};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  Tensor expected =
      at::zeros(at::IntArrayRef(sizes), at::dtype(ScalarType::Float));
  memcpy(expected.mutable_data_ptr<float>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {0, 1};
  int32_t strides[dim] = {2, 1}; // Contiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Float, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, Zeros) {
  TensorFactory<ScalarType::Int> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.zeros(/*sizes=*/{2, 2});

  // A Tensor created manually, that should be identical to `actual`.
  int32_t data[] = {0, 0, 0, 0};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  Tensor expected =
      at::zeros(at::IntArrayRef(sizes), at::dtype(ScalarType::Int));
  memcpy(expected.mutable_data_ptr<int32_t>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {0, 1};
  int32_t strides[dim] = {2, 1}; // Contiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Int, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, Ones) {
  TensorFactory<ScalarType::Int> tf;

  // A Tensor created by the factory.
  Tensor actual = tf.ones(/*sizes=*/{2, 2});

  // A Tensor created manually, that should be identical to `actual`.
  int32_t data[] = {1, 1, 1, 1};
#ifdef USE_ATEN_LIB
  std::vector<int64_t> sizes = {2, 2};
  Tensor expected =
      at::zeros(at::IntArrayRef(sizes), at::dtype(ScalarType::Int));
  memcpy(expected.mutable_data_ptr<int32_t>(), data, sizeof(data));
#else // !USE_ATEN_LIB
  constexpr size_t dim = 2;
  int32_t sizes[dim] = {2, 2};
  uint8_t dim_order[dim] = {0, 1};
  int32_t strides[dim] = {2, 1}; // Contiguous
  TensorImpl impl =
      TensorImpl(ScalarType::Int, dim, sizes, data, dim_order, strides);
  Tensor expected(&impl);
#endif // !USE_ATEN_LIB

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, ZeroDimensionalTensor) {
  TensorFactory<ScalarType::Int> tf;

  // Demonstrate that we can create a zero-dimensional tensor in various ways.
  {
    Tensor t = tf.make({}, {7});
    EXPECT_EQ(t.dim(), 0);
    EXPECT_EQ(t.nbytes(), sizeof(int32_t));
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.const_data_ptr<int32_t>()[0], 7);
  }
  {
    Tensor t = tf.zeros({});
    EXPECT_EQ(t.dim(), 0);
    EXPECT_EQ(t.nbytes(), sizeof(int32_t));
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.const_data_ptr<int32_t>()[0], 0);
  }
  {
    Tensor t = tf.ones({});
    EXPECT_EQ(t.dim(), 0);
    EXPECT_EQ(t.nbytes(), sizeof(int32_t));
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.const_data_ptr<int32_t>()[0], 1);
  }
}

TEST_F(TensorFactoryTest, EmptyTensor) {
  TensorFactory<ScalarType::Int> tf;

  // Demonstrate that we can create a completely empty tensor by providing
  // a zero-width dimension.
  {
    Tensor t = tf.make({0}, {});
    EXPECT_EQ(t.dim(), 1);
    EXPECT_EQ(t.nbytes(), 0);
    EXPECT_EQ(t.numel(), 0);
  }
  {
    Tensor t = tf.zeros({0});
    EXPECT_EQ(t.dim(), 1);
    EXPECT_EQ(t.nbytes(), 0);
    EXPECT_EQ(t.numel(), 0);
  }
  {
    Tensor t = tf.ones({0});
    EXPECT_EQ(t.dim(), 1);
    EXPECT_EQ(t.nbytes(), 0);
    EXPECT_EQ(t.numel(), 0);
  }
}

void run_zeros_like_test(Tensor input) {
  // Demonstrate that we can create a new tensor filled by 0 with same size as
  // the given tensor input.
  TensorFactory<ScalarType::Int> tf;
  std::vector<int64_t> input_sizes_64 = {
      input.sizes().begin(), input.sizes().end()};
  Tensor actual = tf.zeros_like(input);

  // A Tensor created manually, that should be identical to `actual`.
  std::vector<int32_t> expected_data;
  for (int i = 0; i < input.numel(); i++) {
    expected_data.push_back(0);
  }
#ifdef USE_ATEN_LIB
  Tensor expected =
      at::zeros(at::IntArrayRef(input_sizes_64), at::dtype(ScalarType::Int));
  if (input.numel()) {
    // memcpy shouldn't be done if input tensor is empty tensor. Memcpy doesn't
    // allow copy null ptr, even if size is 0.
    memcpy(
        expected.mutable_data_ptr<int32_t>(),
        expected_data.data(),
        sizeof(int32_t) * input.numel());
  }
#else // !USE_ATEN_LIB
  TensorImpl impl = TensorImpl(
      ScalarType::Int,
      input.dim(),
      // static shape tensor so const_cast is fine.
      /*sizes=*/const_cast<SizesType*>(input.sizes().data()),
      /*data=*/expected_data.data(),
      /*dim_order=*/const_cast<DimOrderType*>(input.dim_order().data()),
      /*strides=*/const_cast<StridesType*>(input.strides().data()));
  Tensor expected(&impl);
#endif

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, ZerosLike) {
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*size=*/{3, 2, 1}, /*data=*/{1, 2, 3, 4, 5, 6});
  run_zeros_like_test(input);
}

TEST_F(TensorFactoryTest, ZerosLikeZeroDimensionalTensorSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*size=*/{}, /*data=*/{1});
  run_zeros_like_test(input);
}

TEST_F(TensorFactoryTest, ZerosLikeEmptyTensorSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*size=*/{0}, /*data=*/{});
  run_zeros_like_test(input);
}

void run_ones_like_test(Tensor input) {
  // Demonstrate that we can create a new tensor filled by 1 with same size as
  // the given tensor input.
  TensorFactory<ScalarType::Int> tf;
  std::vector<int64_t> input_sizes_64 = {
      input.sizes().begin(), input.sizes().end()};
  Tensor actual = tf.ones_like(input);

  // A Tensor created manually, that should be identical to `actual`.
  std::vector<int32_t> expected_data;
  for (int i = 0; i < input.numel(); i++) {
    expected_data.push_back(1);
  }
#ifdef USE_ATEN_LIB
  Tensor expected =
      at::zeros(at::IntArrayRef(input_sizes_64), at::dtype(ScalarType::Int));
  if (input.numel()) {
    // memcpy shouldn't be done if input tensor is empty tensor. Memcpy doesn't
    // allow copy null ptr, even if size is 0.
    memcpy(
        expected.mutable_data_ptr<int32_t>(),
        expected_data.data(),
        sizeof(int32_t) * input.numel());
  }
#else // !USE_ATEN_LIB
  TensorImpl impl = TensorImpl(
      ScalarType::Int,
      input.dim(),
      // static shape tensor so const_cast is fine.
      /*sizes=*/const_cast<SizesType*>(input.sizes().data()),
      /*data=*/expected_data.data(),
      /*dim_order=*/const_cast<DimOrderType*>(input.dim_order().data()),
      /*strides=*/const_cast<StridesType*>(input.strides().data()));
  Tensor expected(&impl);
#endif

  // Ensure that both tensors are identical.
  EXPECT_TENSOR_EQ(expected, actual);
}

TEST_F(TensorFactoryTest, OnesLike) {
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*size=*/{3, 2, 1}, /*data=*/{1, 2, 3, 4, 5, 6});
  run_ones_like_test(input);
}

TEST_F(TensorFactoryTest, OnesLikeZeroDimensionalTensorSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*size=*/{}, /*data=*/{2});
  run_ones_like_test(input);
}

TEST_F(TensorFactoryTest, OnesLikeEmptyTensorSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*size=*/{0}, /*data=*/{});
  run_ones_like_test(input);
}

//
// Tests for TensorListFactory
//

TEST(TensorListFactoryTest, ZerosLike) {
  TensorFactory<ScalarType::Int> tf;
  TensorListFactory<ScalarType::Int> tlf;

  // Some templates with different shapes and non-zero element values.
  std::vector<Tensor> templates = {
      tf.make(/*sizes=*/{1}, /*data=*/{1}),
      tf.make(/*sizes=*/{1, 2}, /*data=*/{2, 3}),
      tf.make(/*sizes=*/{2, 2, 2}, /*data=*/{4, 5, 6, 7, 8, 9, 10, 11}),
  };

  TensorList actual = tlf.zeros_like(templates);

  // Should have the same shapes as the templates, but all elements should be
  // zero.
  std::vector<Tensor> expected = {
      tf.zeros(/*sizes=*/{1}),
      tf.zeros(/*sizes=*/{1, 2}),
      tf.zeros(/*sizes=*/{2, 2, 2}),
  };

  EXPECT_TENSOR_LISTS_EQ(actual, expected);
}

TEST(TensorListFactoryTest, ZerosLikeMixedDtypes) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Short> tf_short;
  TensorFactory<ScalarType::Float> tf_float;

  TensorFactory<ScalarType::Int> tf_int;
  TensorListFactory<ScalarType::Int> tlf;

  // Some templates with different shapes and non-zero element values, and
  // different dtypes than the TensorListFactory. Demonstrates that the template
  // dtypes don't matter, only the shapes.
  std::vector<Tensor> templates = {
      tf_byte.make(/*sizes=*/{1}, /*data=*/{1}),
      tf_short.make(/*sizes=*/{1, 2}, /*data=*/{2, 3}),
      tf_float.make(/*sizes=*/{2, 2, 2}, /*data=*/{4, 5, 6, 7, 8, 9, 10, 11}),
  };

  TensorList actual = tlf.zeros_like(templates);

  // Should have the same shapes as the templates, but all elements should be
  // zero, and the dtypes should all be Int.
  std::vector<Tensor> expected = {
      tf_int.zeros(/*sizes=*/{1}),
      tf_int.zeros(/*sizes=*/{1, 2}),
      tf_int.zeros(/*sizes=*/{2, 2, 2}),
  };

  EXPECT_TENSOR_LISTS_EQ(actual, expected);
}

TEST(TensorListFactoryTest, ZerosLikeEmpty) {
  TensorListFactory<ScalarType::Int> tlf;

  // Empty templates list.
  std::vector<Tensor> templates = {};

  TensorList actual = tlf.zeros_like(templates);

  // Should produce an empty TensorList.
  std::vector<Tensor> expected = {};

  EXPECT_TENSOR_LISTS_EQ(actual, expected);
}

TEST_F(TensorFactoryTest, ZerosDynamismParameter) {
  TensorFactory<ScalarType::Int> tf;
  resize_tensor_to_assert_static(tf.zeros({2, 2}, TensorShapeDynamism::STATIC));
  resize_tensor_to_assert_dynamic_bound(
      tf.zeros({2, 2}, TensorShapeDynamism::DYNAMIC_BOUND));
  resize_tensor_to_assert_dynamic_unbound(
      tf.zeros({2, 2}, TensorShapeDynamism::DYNAMIC_UNBOUND));

  // The tensor itself should be equal
  EXPECT_TENSOR_EQ(
      tf.zeros({2, 2}, TensorShapeDynamism::STATIC),
      tf.zeros({2, 2}, TensorShapeDynamism::DYNAMIC_BOUND));
  EXPECT_TENSOR_EQ(
      tf.zeros({2, 2}, TensorShapeDynamism::STATIC),
      tf.zeros({2, 2}, TensorShapeDynamism::DYNAMIC_UNBOUND));
}

TEST_F(TensorFactoryTest, ZerosLikeDynamismParameter) {
  TensorFactory<ScalarType::Int> tf;
  Tensor zeros = tf.zeros({2, 2});
  resize_tensor_to_assert_static(
      tf.zeros_like(zeros, TensorShapeDynamism::STATIC));
  resize_tensor_to_assert_dynamic_bound(
      tf.zeros_like(zeros, TensorShapeDynamism::DYNAMIC_BOUND));
  resize_tensor_to_assert_dynamic_unbound(
      tf.zeros_like(zeros, TensorShapeDynamism::DYNAMIC_UNBOUND));

  // The tensor itself should be equal
  EXPECT_TENSOR_EQ(
      tf.zeros_like(zeros, TensorShapeDynamism::STATIC),
      tf.zeros_like(zeros, TensorShapeDynamism::DYNAMIC_BOUND));
  EXPECT_TENSOR_EQ(
      tf.zeros_like(zeros, TensorShapeDynamism::STATIC),
      tf.zeros_like(zeros, TensorShapeDynamism::DYNAMIC_UNBOUND));
}

TEST_F(TensorFactoryTest, OnesDynamismParameter) {
  TensorFactory<ScalarType::Int> tf;
  resize_tensor_to_assert_static(tf.ones({2, 2}, TensorShapeDynamism::STATIC));
  resize_tensor_to_assert_dynamic_bound(
      tf.ones({2, 2}, TensorShapeDynamism::DYNAMIC_BOUND));
  resize_tensor_to_assert_dynamic_unbound(
      tf.ones({2, 2}, TensorShapeDynamism::DYNAMIC_UNBOUND));

  // The tensor itself should be equal
  EXPECT_TENSOR_EQ(
      tf.ones({2, 2}, TensorShapeDynamism::STATIC),
      tf.ones({2, 2}, TensorShapeDynamism::DYNAMIC_BOUND));
  EXPECT_TENSOR_EQ(
      tf.ones({2, 2}, TensorShapeDynamism::STATIC),
      tf.ones({2, 2}, TensorShapeDynamism::DYNAMIC_UNBOUND));
}

TEST_F(TensorFactoryTest, OnesLikeDynamismParameter) {
  TensorFactory<ScalarType::Int> tf;
  Tensor ones = tf.ones({2, 2});
  resize_tensor_to_assert_static(
      tf.ones_like(ones, TensorShapeDynamism::STATIC));
  resize_tensor_to_assert_dynamic_bound(
      tf.ones_like(ones, TensorShapeDynamism::DYNAMIC_BOUND));
  resize_tensor_to_assert_dynamic_unbound(
      tf.ones_like(ones, TensorShapeDynamism::DYNAMIC_UNBOUND));

  // The tensor itself should be equal
  EXPECT_TENSOR_EQ(
      tf.ones_like(ones, TensorShapeDynamism::STATIC),
      tf.ones_like(ones, TensorShapeDynamism::DYNAMIC_BOUND));
  EXPECT_TENSOR_EQ(
      tf.ones_like(ones, TensorShapeDynamism::STATIC),
      tf.ones_like(ones, TensorShapeDynamism::DYNAMIC_UNBOUND));
}

TEST_F(TensorFactoryTest, FullDynamismParameter) {
  TensorFactory<ScalarType::Int> tf;
  resize_tensor_to_assert_static(
      tf.full({2, 2}, 1, TensorShapeDynamism::STATIC));
  resize_tensor_to_assert_dynamic_bound(
      tf.full({2, 2}, 1, TensorShapeDynamism::DYNAMIC_BOUND));
  resize_tensor_to_assert_dynamic_unbound(
      tf.full({2, 2}, 1, TensorShapeDynamism::DYNAMIC_UNBOUND));

  // The tensor itself should be equal
  EXPECT_TENSOR_EQ(
      tf.full({2, 2}, 1, TensorShapeDynamism::STATIC),
      tf.full({2, 2}, 1, TensorShapeDynamism::DYNAMIC_BOUND));
  EXPECT_TENSOR_EQ(
      tf.full({2, 2}, 1, TensorShapeDynamism::STATIC),
      tf.full({2, 2}, 1, TensorShapeDynamism::DYNAMIC_UNBOUND));
}

TEST_F(TensorFactoryTest, MakeDynamismParameter) {
  TensorFactory<ScalarType::Int> tf;
  resize_tensor_to_assert_static(
      tf.make({2, 2}, {1, 2, 3, 4}, {}, TensorShapeDynamism::STATIC));
  resize_tensor_to_assert_dynamic_bound(
      tf.make({2, 2}, {1, 2, 3, 4}, {}, TensorShapeDynamism::DYNAMIC_BOUND));
  resize_tensor_to_assert_dynamic_unbound(
      tf.make({2, 2}, {1, 2, 3, 4}, {}, TensorShapeDynamism::DYNAMIC_UNBOUND));

  // The tensor itself should be equal
  EXPECT_TENSOR_EQ(
      tf.make({2, 2}, {1, 2, 3, 4}, {}, TensorShapeDynamism::STATIC),
      tf.make({2, 2}, {1, 2, 3, 4}, {}, TensorShapeDynamism::DYNAMIC_BOUND));
  EXPECT_TENSOR_EQ(
      tf.make({2, 2}, {1, 2, 3, 4}, {}, TensorShapeDynamism::STATIC),
      tf.make({2, 2}, {1, 2, 3, 4}, {}, TensorShapeDynamism::DYNAMIC_UNBOUND));
}

#if !defined(USE_ATEN_LIB)
TEST_F(TensorFactoryTest, FullDynamic) {
  TensorFactory<ScalarType::Int> tf;

  // A Tensor created by the factory.
  Tensor out = tf.full(/*sizes=*/{2, 2}, 5);
  int32_t new_sizes[2] = {1, 1};
  ET_EXPECT_DEATH(torch::executor::resize(out, new_sizes), "");

  out = tf.full(
      /*sizes=*/{2, 2}, 5, TensorShapeDynamism::DYNAMIC_BOUND);
  new_sizes[1] = 2;
  EXPECT_EQ(
      torch::executor::resize_tensor(out, new_sizes),
      torch::executor::Error::Ok);
  new_sizes[0] = 3;
  new_sizes[1] = 3;
  ET_EXPECT_DEATH(torch::executor::resize(out, new_sizes), "");
}

TEST_F(TensorFactoryTest, MakeIntTensorDynamic) {
  TensorFactory<ScalarType::Int> tf;

  // A Tensor created by the factory.
  Tensor out = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});

  int32_t new_sizes[2] = {1, 1};
  ET_EXPECT_DEATH(torch::executor::resize(out, new_sizes), "");

  std::vector<int32_t> data = {1, 2, 3, 4};
  out = tf.make(
      /*sizes=*/{2, 2}, data, {}, TensorShapeDynamism::DYNAMIC_BOUND);
  new_sizes[1] = 2;
  EXPECT_EQ(
      torch::executor::resize_tensor(out, new_sizes),
      torch::executor::Error::Ok);
  new_sizes[0] = 3;
  new_sizes[1] = 3;
  ET_EXPECT_DEATH(torch::executor::resize(out, new_sizes), "");
}

TEST_F(TensorFactoryTest, MakeZerosDynamic) {
  TensorFactory<ScalarType::Int> tf;

  // A Tensor created by the factory.
  Tensor out = tf.zeros(/*sizes=*/{2, 2});

  int32_t new_sizes[2] = {1, 1};
  ET_EXPECT_DEATH(torch::executor::resize(out, new_sizes), "");

  out = tf.zeros(
      /*sizes=*/{2, 2}, TensorShapeDynamism::DYNAMIC_BOUND);
  new_sizes[1] = 2;
  EXPECT_EQ(
      torch::executor::resize_tensor(out, new_sizes),
      torch::executor::Error::Ok);
  new_sizes[0] = 3;
  new_sizes[1] = 3;
  ET_EXPECT_DEATH(torch::executor::resize(out, new_sizes), "");

  Tensor out_like = tf.zeros_like(out);
  new_sizes[0] = 1;
  new_sizes[1] = 1;
  ET_EXPECT_DEATH(torch::executor::resize(out_like, new_sizes), "");

  out = tf.zeros_like(out, TensorShapeDynamism::DYNAMIC_BOUND);
  new_sizes[1] = 2;
  EXPECT_EQ(
      torch::executor::resize_tensor(out, new_sizes),
      torch::executor::Error::Ok);
  new_sizes[0] = 3;
  new_sizes[1] = 3;
  ET_EXPECT_DEATH(torch::executor::resize(out, new_sizes), "");
}

TEST_F(TensorFactoryTest, DimOrderToStrideTest) {
  TensorFactory<ScalarType::Int> tf;
  // A Tensor created by the factory.
  Tensor out = tf.zeros(/*sizes=*/{2, 2});
  std::vector<DimOrderType> dim_order;

  dim_order.resize(2);
  dim_order[0] = 0;
  dim_order[1] = 1;
  exec_aten::ArrayRef<DimOrderType> dim_order_ref(
      dim_order.data(), dim_order.size());

  CHECK_ARRAY_REF_EQUAL(dim_order_ref, out.dim_order());

  out = tf.zeros(/*sizes=*/{1, 2, 5});
  dim_order.resize(out.sizes().size());
  dim_order[0] = 0;
  dim_order[1] = 1;
  dim_order[2] = 2;
  dim_order_ref =
      exec_aten::ArrayRef<DimOrderType>(dim_order.data(), dim_order.size());

  CHECK_ARRAY_REF_EQUAL(dim_order_ref, out.dim_order());

  std::vector<TensorFactory<ScalarType::Int>::ctype> data(10);
  Tensor strided_out = tf.make({1, 2, 5}, data, {10, 1, 2});
  dim_order.resize(out.sizes().size());
  dim_order[0] = 0;
  dim_order[1] = 2;
  dim_order[2] = 1;
  dim_order_ref =
      exec_aten::ArrayRef<DimOrderType>(dim_order.data(), dim_order.size());

  CHECK_ARRAY_REF_EQUAL(dim_order_ref, strided_out.dim_order());

  data.resize(12);
  strided_out = tf.make({3, 2, 2}, data, {1, 6, 3});
  dim_order.resize(out.sizes().size());
  dim_order[0] = 1;
  dim_order[1] = 2;
  dim_order[2] = 0;
  dim_order_ref =
      exec_aten::ArrayRef<DimOrderType>(dim_order.data(), dim_order.size());

  CHECK_ARRAY_REF_EQUAL(dim_order_ref, strided_out.dim_order());
}

TEST_F(TensorFactoryTest, AmbgiuousDimOrderToStrideTest) {
  TensorFactory<ScalarType::Int> tf;
  std::vector<TensorFactory<ScalarType::Int>::ctype> data(10);
  Tensor strided_out = tf.make({1, 2, 5}, data, {1, 1, 2});
  std::vector<DimOrderType> dim_order(strided_out.sizes().size());
  dim_order[0] = 2;
  dim_order[1] = 0;
  dim_order[2] = 1;
  // Note that above strides of {1, 1, 2} can also be
  // interpreter as dim_order = {2, 1, 0}, however when converting
  // strides to dim order we preseve dimension order,
  // using stable_sort when converting strides to dim_order,
  // see dim_order_from_stride in TensorFactorh.h,
  // and hence valid dim_order = {2, 0, 1}
  // Since strides can give ambiguous dimension order when crossing
  // boundary from strides land to dim order land, we have to resolve
  // such ambiguity in a deterministic way.
  // In dim order land, it is less ambiguous
  auto dim_order_ref =
      exec_aten::ArrayRef<DimOrderType>(dim_order.data(), dim_order.size());

  CHECK_ARRAY_REF_EQUAL(dim_order_ref, strided_out.dim_order());

  strided_out = tf.make({1, 2, 5}, data, {1, 1, 2});
  dim_order.resize(strided_out.sizes().size());
  dim_order[0] = 2;
  dim_order[1] = 0;
  dim_order[2] = 1;
  dim_order_ref =
      exec_aten::ArrayRef<DimOrderType>(dim_order.data(), dim_order.size());

  CHECK_ARRAY_REF_EQUAL(dim_order_ref, strided_out.dim_order());
}
#endif // !USE_ATEN_LIB
