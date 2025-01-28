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
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
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
using executorch::runtime::testing::make_kernel_key;

class OperatorRegistryTest : public ::testing::Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(OperatorRegistryTest, Basic) {
  Kernel kernels[] = {Kernel("foo", [](KernelRuntimeContext&, EValue**) {})};
  Span<const Kernel> kernels_span(kernels);
  (void)register_kernels(kernels_span);
  EXPECT_FALSE(registry_has_op_function("fpp"));
  EXPECT_TRUE(registry_has_op_function("foo"));
}

TEST_F(OperatorRegistryTest, RegisterOpsMoreThanOnceDie) {
  Kernel kernels[] = {
      Kernel("foo", [](KernelRuntimeContext&, EValue**) {}),
      Kernel("foo", [](KernelRuntimeContext&, EValue**) {})};
  Span<const Kernel> kernels_span = Span<const Kernel>(kernels);
  ET_EXPECT_DEATH({ (void)register_kernels(kernels_span); }, "");
}

constexpr int BUF_SIZE = KernelKey::MAX_SIZE;

TEST_F(OperatorRegistryTest, KernelKeyEquals) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey long_contiguous = KernelKey(buf_long_contiguous);

  KernelKey long_key_1 = KernelKey(long_contiguous);

  KernelKey long_key_2 = KernelKey(long_contiguous);

  EXPECT_EQ(long_key_1, long_key_2);

  char buf_float_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Float, {0, 1, 2, 3}}}, buf_float_contiguous);
  KernelKey float_key = KernelKey(buf_float_contiguous);

  EXPECT_NE(long_key_1, float_key);

  char buf_channel_first[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 3, 1, 2}}}, buf_channel_first);
  KernelKey long_key_3 = KernelKey(buf_channel_first);

  EXPECT_NE(long_key_1, long_key_3);
}

TEST_F(OperatorRegistryTest, RegisterKernels) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key = KernelKey(buf_long_contiguous);

  Kernel kernel_1 = Kernel(
      "test::boo", key, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernels({&kernel_1, 1});
  EXPECT_EQ(s1, Error::Ok);

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
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key_1 = KernelKey(buf_long_contiguous);

  char buf_float_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Float, {0, 1, 2, 3}}}, buf_float_contiguous);
  KernelKey key_2 = KernelKey(buf_float_contiguous);
  Kernel kernel_1 = Kernel(
      "test::bar", key_1, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  Kernel kernel_2 = Kernel(
      "test::bar", key_2, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(50);
      });
  Kernel kernels[] = {kernel_1, kernel_2};
  auto s1 = register_kernels(kernels);
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
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key = KernelKey(buf_long_contiguous);

  Kernel kernel_1 = Kernel(
      "test::baz", key, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  Kernel kernel_2 = Kernel(
      "test::baz", key, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(50);
      });
  Kernel kernels[] = {kernel_1, kernel_2};
  // clang-tidy off
  ET_EXPECT_DEATH({ auto s1 = register_kernels(kernels); }, "");
  // clang-tidy on
}

TEST_F(OperatorRegistryTest, ExecutorChecksKernel) {
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key = KernelKey(buf_long_contiguous);

  Kernel kernel_1 = Kernel(
      "test::qux", key, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernels({&kernel_1, 1});
  EXPECT_EQ(s1, Error::Ok);

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
  char buf_long_contiguous[BUF_SIZE];
  make_kernel_key({{ScalarType::Long, {0, 1, 2, 3}}}, buf_long_contiguous);
  KernelKey key = KernelKey(buf_long_contiguous);

  Kernel kernel_1 = Kernel(
      "test::quux", key, [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernels({&kernel_1, 1});
  EXPECT_EQ(s1, Error::Ok);

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
      [](KernelRuntimeContext& context, EValue** stack) {
        (void)context;
        *(stack[0]) = Scalar(100);
      });
  auto s1 = register_kernels({&kernel_1, 1});
  EXPECT_EQ(s1, Error::Ok);

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
