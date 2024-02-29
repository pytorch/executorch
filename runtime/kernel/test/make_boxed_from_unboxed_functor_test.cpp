/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using namespace ::testing;
using RuntimeContext = torch::executor::KernelRuntimeContext;
using namespace torch::executor;

Tensor& my_op_out(RuntimeContext& ctx, const Tensor& a, Tensor& out) {
  (void)ctx;
  (void)a;
  return out;
}

Tensor& set_1_out(RuntimeContext& ctx, Tensor& out) {
  (void)ctx;
  out.mutable_data_ptr<int32_t>()[0] = 1;
  return out;
}

class MakeBoxedFromUnboxedFunctorTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(MakeBoxedFromUnboxedFunctorTest, Basic) {
  Kernel my_kernel =
      Kernel::make_boxed_kernel("my_ns::my_op.out", EXECUTORCH_FN(my_op_out));
  ArrayRef<Kernel> kernels_array = ArrayRef<Kernel>(my_kernel);
  // @lint-ignore CLANGTIDY
  auto s1 = register_kernels(kernels_array);
  EXPECT_TRUE(hasOpsFn("my_ns::my_op.out"));
}

TEST_F(MakeBoxedFromUnboxedFunctorTest, UnboxLogicWorks) {
  Kernel my_kernel =
      Kernel::make_boxed_kernel("my_ns::set_1.out", EXECUTORCH_FN(set_1_out));
  ArrayRef<Kernel> kernels_array = ArrayRef<Kernel>(my_kernel);
  // @lint-ignore CLANGTIDY
  auto s1 = register_kernels(kernels_array);
  EXPECT_TRUE(hasOpsFn("my_ns::set_1.out"));

  // prepare out tensor
  TensorImpl::SizesType sizes[1] = {5};
  TensorImpl::DimOrderType dim_order[1] = {0};
  int32_t data[5] = {0, 0, 0, 0, 0};
  auto a_impl = TensorImpl(ScalarType::Int, 1, sizes, data, dim_order, nullptr);
  auto a = Tensor(&a_impl);

  // get boxed callable
  auto fn = getOpsFn("my_ns::set_1.out");

  // run it
  RuntimeContext context;
  EValue values[1];
  values[0] = a;
  EValue* stack[1];
  stack[0] = &values[0];

  fn(context, stack);

  // check result
  EXPECT_EQ(a.const_data_ptr<int32_t>()[0], 1);
}
