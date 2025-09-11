/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/kernel/test/test_util.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::testing;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::get_op_function_from_registry;
using executorch::runtime::Kernel;
using executorch::runtime::KernelKey;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::OpFunction;
using executorch::runtime::register_kernels;
using executorch::runtime::registry_has_op_function;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorMeta;
using executorch::runtime::internal::kKernelKeyBufSize;
using executorch::runtime::testing::make_kernel_key;

//
// Tests for make_kernel_key_string
//

// Helper for testing make_kernel_key_string.
void test_make_kernel_key_string(
    const std::vector<std::pair<
        executorch::aten::ScalarType,
        std::vector<executorch::aten::DimOrderType>>>& tensors,
    const char* expected_key) {
  const size_t min_buf_size = strlen(expected_key) + 1;

  // Sweep across too-small buffer sizes, exercising all possible failure
  // checks. Rely on ASAN to detect buffer overflows.
  for (size_t buf_size = 0; buf_size < min_buf_size; buf_size++) {
    std::vector<char> actual_key(buf_size, 0x55);
    Error err = make_kernel_key(
        tensors,
        // nullptr should be valid for buf_size == 0 because it won't be written
        // to.
        buf_size == 0 ? nullptr : actual_key.data(),
        actual_key.size());
    EXPECT_NE(err, Error::Ok);
  }

  // Demonstrate that it succeeds for buffers of exactly the right size or
  // larger.
  for (size_t buf_size = min_buf_size; buf_size < min_buf_size + 1;
       buf_size++) {
    std::vector<char> actual_key(buf_size, 0x55);
    Error err = make_kernel_key(tensors, actual_key.data(), actual_key.size());
    ASSERT_EQ(err, Error::Ok);
    EXPECT_STREQ(actual_key.data(), expected_key);
  }
}

TEST(MakeKernelKeyStringTest, ZeroTensorSuccessWithNullBuffer) {
  Error err = make_kernel_key({}, nullptr, 0);
  EXPECT_EQ(err, Error::Ok);
}

TEST(MakeKernelKeyStringTest, ZeroTensorSuccessMakesEmptyString) {
  char buf = 0x55;
  Error err = make_kernel_key({}, &buf, 1);
  EXPECT_EQ(err, Error::Ok);
  EXPECT_EQ(buf, '\0');
}

TEST(MakeKernelKeyStringTest, OneTensorSuccess) {
  test_make_kernel_key_string(
      {{ScalarType::Long, {0, 1, 2, 3}}}, "v1/4;0,1,2,3");
}

TEST(MakeKernelKeyStringTest, TwoTensorSuccess) {
  test_make_kernel_key_string(
      {{ScalarType::Long, {0, 1, 2, 3}}, {ScalarType::Double, {3, 2, 1, 0}}},
      "v1/4;0,1,2,3|7;3,2,1,0");
}

TEST(MakeKernelKeyStringTest, ThreeTensorSuccess) {
  test_make_kernel_key_string(
      {{ScalarType::Long, {0, 1, 2, 3}},
       {ScalarType::Double, {3, 2, 1, 0}},
       {ScalarType::Byte, {2, 1, 3, 0}}},
      "v1/4;0,1,2,3|7;3,2,1,0|0;2,1,3,0");
}

TEST(MakeKernelKeyStringTest, TwoDigitDimOrderSuccess) {
  test_make_kernel_key_string(
      {{ScalarType::Long, {0, 10, 2, 99}}}, "v1/4;0,10,2,99");
}

TEST(MakeKernelKeyStringTest, ThreeDigitDimOrderFailure) {
  std::vector<char> actual_key(1024, 0x55); // Large enough for any key.
  Error err = make_kernel_key(
      // Cannot represent a dim order entry with more than two digits.
      {{ScalarType::Long, {0, 100, 2, 255}}},
      actual_key.data(),
      actual_key.size());
  EXPECT_NE(err, Error::Ok);
}

TEST(MakeKernelKeyStringTest, NegativeScalarTypeFailure) {
  std::vector<char> actual_key(1024, 0x55); // Large enough for any key.
  Error err = make_kernel_key(
      // Cannot represent a ScalarType (aka int8_t) with a negative value.
      {{(ScalarType)-1, {0, 1, 2, 3}}},
      actual_key.data(),
      actual_key.size());
  EXPECT_NE(err, Error::Ok);
}

TEST(MakeKernelKeyStringTest, KeyBufSizeMeetsAssumptions) {
  // Create the longest key that fits in the assupmtions of kKernelKeyBufSize:
  // 16 tensors, 16 dims, with two-digit ScalarTypes.
  std::vector<std::pair<
      executorch::aten::ScalarType,
      std::vector<executorch::aten::DimOrderType>>>
      tensors;
  tensors.reserve(16);
  for (int i = 0; i < 16; i++) {
    std::vector<executorch::aten::DimOrderType> dims;
    dims.reserve(16);
    for (int j = 0; j < 16; j++) {
      dims.emplace_back(j);
    }
    tensors.emplace_back((ScalarType)10, dims);
  }

  std::vector<char> actual_key(kKernelKeyBufSize, 0x55);
  Error err = make_kernel_key(tensors, actual_key.data(), actual_key.size());
  ASSERT_EQ(err, Error::Ok);
  EXPECT_STREQ(
      actual_key.data(),
      "v1/"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15|"
      "10;0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15");
  EXPECT_LE(strlen(actual_key.data()) + 1, kKernelKeyBufSize);
}

//
// Tests for public operator registry APIs
//

class OperatorRegistryTest : public ::testing::Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(OperatorRegistryTest, Basic) {
  Kernel kernels[] = {
      Kernel("foo", [](KernelRuntimeContext&, Span<EValue*>) {})};
  Span<const Kernel> kernels_span(kernels);
  Error err = register_kernels(kernels_span);
  ASSERT_EQ(err, Error::Ok);
  EXPECT_FALSE(registry_has_op_function("fpp"));
  EXPECT_TRUE(registry_has_op_function("foo"));
}

TEST_F(OperatorRegistryTest, RegisterOpsMoreThanOnceDie) {
  Kernel kernels[] = {
      Kernel("foo", [](KernelRuntimeContext&, Span<EValue*>) {}),
      Kernel("foo", [](KernelRuntimeContext&, Span<EValue*>) {})};
  Span<const Kernel> kernels_span = Span<const Kernel>(kernels);
  ET_EXPECT_DEATH({ (void)register_kernels(kernels_span); }, "");
}

TEST_F(OperatorRegistryTest, KernelKeyEquals) {
  std::array<char, kKernelKeyBufSize> buf_long_contiguous;
  Error err = make_kernel_key(
      {{ScalarType::Long, {0, 1, 2, 3}}},
      buf_long_contiguous.data(),
      buf_long_contiguous.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey long_contiguous = KernelKey(buf_long_contiguous.data());

  KernelKey long_key_1 = KernelKey(long_contiguous);

  KernelKey long_key_2 = KernelKey(long_contiguous);

  EXPECT_EQ(long_key_1, long_key_2);

  std::array<char, kKernelKeyBufSize> buf_float_contiguous;
  err = make_kernel_key(
      {{ScalarType::Float, {0, 1, 2, 3}}},
      buf_float_contiguous.data(),
      buf_float_contiguous.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey float_key = KernelKey(buf_float_contiguous.data());

  EXPECT_NE(long_key_1, float_key);

  std::array<char, kKernelKeyBufSize> buf_channel_first;
  err = make_kernel_key(
      {{ScalarType::Long, {0, 3, 1, 2}}},
      buf_channel_first.data(),
      buf_channel_first.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey long_key_3 = KernelKey(buf_channel_first.data());

  EXPECT_NE(long_key_1, long_key_3);
}

TEST_F(OperatorRegistryTest, GetOpFailsForLongKernelKey) {
  // Looking up a way-too-long kernel key should fail with an error.
  std::vector<std::pair<
      executorch::aten::ScalarType,
      std::vector<executorch::aten::DimOrderType>>>
      tensors;
  // 1000 is a lot of tensors.
  tensors.reserve(1000);
  for (int i = 0; i < 1000; i++) {
    std::vector<executorch::aten::DimOrderType> dims;
    dims.reserve(16);
    for (int j = 0; j < 16; j++) {
      dims.emplace_back(j);
    }
    tensors.emplace_back((ScalarType)10, dims);
  }
  std::vector<TensorMeta> meta;
  for (auto& t : tensors) {
    Span<executorch::aten::DimOrderType> dim_order(
        t.second.data(), t.second.size());
    meta.emplace_back(t.first, dim_order);
  }
  Span<const TensorMeta> metadata(meta.data(), meta.size());

  auto op = get_op_function_from_registry("test::not-real", metadata);
  EXPECT_NE(op.error(), Error::Ok);
  EXPECT_NE(op.error(), Error::OperatorMissing);
  // The lookup failed, but not because the operator is missing.
}

TEST_F(OperatorRegistryTest, RegisterKernels) {
  std::array<char, kKernelKeyBufSize> buf_long_contiguous;
  Error err = make_kernel_key(
      {{ScalarType::Long, {0, 1, 2, 3}}},
      buf_long_contiguous.data(),
      buf_long_contiguous.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey key = KernelKey(buf_long_contiguous.data());

  Kernel kernel_1 = Kernel(
      "test::boo", key, [](KernelRuntimeContext& context, Span<EValue*> stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  err = register_kernels({&kernel_1, 1});
  ASSERT_EQ(err, Error::Ok);

  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = Span<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  Span<const TensorMeta> user_kernel_key(meta);

  // no fallback kernel is registered
  EXPECT_FALSE(registry_has_op_function("test::boo", {}));
  Result<OpFunction> fallback_func =
      get_op_function_from_registry("test::boo", {});
  EXPECT_NE(fallback_func.error(), Error::Ok);

  EXPECT_TRUE(registry_has_op_function("test::boo", user_kernel_key));
  Result<OpFunction> func =
      get_op_function_from_registry("test::boo", user_kernel_key);
  EXPECT_EQ(func.error(), Error::Ok);

  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  KernelRuntimeContext context{};
  (*func)(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}

TEST_F(OperatorRegistryTest, RegisterTwoKernels) {
  std::array<char, kKernelKeyBufSize> buf_long_contiguous;
  Error err = make_kernel_key(
      {{ScalarType::Long, {0, 1, 2, 3}}},
      buf_long_contiguous.data(),
      buf_long_contiguous.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey key_1 = KernelKey(buf_long_contiguous.data());

  std::array<char, kKernelKeyBufSize> buf_float_contiguous;
  err = make_kernel_key(
      {{ScalarType::Float, {0, 1, 2, 3}}},
      buf_float_contiguous.data(),
      buf_float_contiguous.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey key_2 = KernelKey(buf_float_contiguous.data());
  Kernel kernel_1 = Kernel(
      "test::bar",
      key_1,
      [](KernelRuntimeContext& context, Span<EValue*> stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  Kernel kernel_2 = Kernel(
      "test::bar",
      key_2,
      [](KernelRuntimeContext& context, Span<EValue*> stack) {
        (void)context;
        *(stack[0]) = Scalar(50);
      });
  Kernel kernels[] = {kernel_1, kernel_2};
  err = register_kernels(kernels);
  ASSERT_EQ(err, Error::Ok);

  // has both kernels
  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = Span<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  Span<const TensorMeta> user_kernel_key_1(meta);

  TensorMeta meta_2[] = {TensorMeta(ScalarType::Float, dim_order_type)};
  Span<const TensorMeta> user_kernel_key_2(meta_2);

  // no fallback kernel is registered
  EXPECT_FALSE(registry_has_op_function("test::bar", {}));
  Result<OpFunction> fallback_func =
      get_op_function_from_registry("test::bar", {});
  EXPECT_NE(fallback_func.error(), Error::Ok);

  EValue values[1];
  values[0] = Scalar(0);
  EValue* evalues[1];
  evalues[0] = &values[0];
  KernelRuntimeContext context{};

  // test kernel_1
  EXPECT_TRUE(registry_has_op_function("test::bar", user_kernel_key_1));
  Result<OpFunction> func_1 =
      get_op_function_from_registry("test::bar", user_kernel_key_1);
  EXPECT_EQ(func_1.error(), Error::Ok);
  (*func_1)(context, evalues);

  auto val_1 = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val_1, 100);

  // test kernel_2
  EXPECT_TRUE(registry_has_op_function("test::bar", user_kernel_key_2));
  Result<OpFunction> func_2 =
      get_op_function_from_registry("test::bar", user_kernel_key_2);
  EXPECT_EQ(func_2.error(), Error::Ok);
  values[0] = Scalar(0);
  (*func_2)(context, evalues);

  auto val_2 = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val_2, 50);
}

TEST_F(OperatorRegistryTest, DoubleRegisterKernelsDies) {
  std::array<char, kKernelKeyBufSize> buf_long_contiguous;
  Error err = make_kernel_key(
      {{ScalarType::Long, {0, 1, 2, 3}}},
      buf_long_contiguous.data(),
      buf_long_contiguous.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey key = KernelKey(buf_long_contiguous.data());

  Kernel kernel_1 = Kernel(
      "test::baz", key, [](KernelRuntimeContext& context, Span<EValue*> stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  Kernel kernel_2 = Kernel(
      "test::baz", key, [](KernelRuntimeContext& context, Span<EValue*> stack) {
        (void)context;
        *(stack[0]) = Scalar(50);
      });
  Kernel kernels[] = {kernel_1, kernel_2};
  // clang-tidy off
  ET_EXPECT_DEATH({ (void)register_kernels(kernels); }, "");
  // clang-tidy on
}

TEST_F(OperatorRegistryTest, ExecutorChecksKernel) {
  std::array<char, kKernelKeyBufSize> buf_long_contiguous;
  Error err = make_kernel_key(
      {{ScalarType::Long, {0, 1, 2, 3}}},
      buf_long_contiguous.data(),
      buf_long_contiguous.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey key = KernelKey(buf_long_contiguous.data());

  Kernel kernel_1 = Kernel(
      "test::qux", key, [](KernelRuntimeContext& context, Span<EValue*> stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  err = register_kernels({&kernel_1, 1});
  ASSERT_EQ(err, Error::Ok);

  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = Span<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  Span<const TensorMeta> user_kernel_key_1(meta);
  EXPECT_TRUE(registry_has_op_function("test::qux", user_kernel_key_1));

  Tensor::DimOrderType dims_channel_first[] = {0, 3, 1, 2};
  auto dim_order_type_channel_first =
      Span<Tensor::DimOrderType>(dims_channel_first, 4);
  TensorMeta meta_channel_first[] = {
      TensorMeta(ScalarType::Long, dim_order_type_channel_first)};
  Span<const TensorMeta> user_kernel_key_2(meta_channel_first);
  EXPECT_FALSE(registry_has_op_function("test::qux", user_kernel_key_2));

  TensorMeta meta_float[] = {TensorMeta(ScalarType::Float, dim_order_type)};
  Span<const TensorMeta> user_kernel_key_3(meta_float);
  EXPECT_FALSE(registry_has_op_function("test::qux", user_kernel_key_3));
}

TEST_F(OperatorRegistryTest, ExecutorUsesKernel) {
  std::array<char, kKernelKeyBufSize> buf_long_contiguous;
  Error err = make_kernel_key(
      {{ScalarType::Long, {0, 1, 2, 3}}},
      buf_long_contiguous.data(),
      buf_long_contiguous.size());
  ASSERT_EQ(err, Error::Ok);
  KernelKey key = KernelKey(buf_long_contiguous.data());

  Kernel kernel_1 = Kernel(
      "test::quux",
      key,
      [](KernelRuntimeContext& context, Span<EValue*> stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  err = register_kernels({&kernel_1, 1});
  ASSERT_EQ(err, Error::Ok);

  Tensor::DimOrderType dims[] = {0, 1, 2, 3};
  auto dim_order_type = Span<Tensor::DimOrderType>(dims, 4);
  TensorMeta meta[] = {TensorMeta(ScalarType::Long, dim_order_type)};
  Span<const TensorMeta> user_kernel_key_1(meta);

  EXPECT_TRUE(registry_has_op_function("test::quux", user_kernel_key_1));
  Result<OpFunction> func =
      get_op_function_from_registry("test::quux", user_kernel_key_1);
  EXPECT_EQ(func.error(), Error::Ok);

  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  KernelRuntimeContext context{};
  (*func)(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}

TEST_F(OperatorRegistryTest, ExecutorUsesFallbackKernel) {
  Kernel kernel_1 = Kernel(
      "test::corge",
      KernelKey{},
      [](KernelRuntimeContext& context, Span<EValue*> stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  Error err = register_kernels({&kernel_1, 1});
  EXPECT_EQ(err, Error::Ok);

  EXPECT_TRUE(registry_has_op_function("test::corge"));
  EXPECT_TRUE(registry_has_op_function("test::corge", {}));

  Result<OpFunction> func = get_op_function_from_registry("test::corge", {});
  EXPECT_EQ(func.error(), Error::Ok);

  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  KernelRuntimeContext context{};
  (*func)(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}
