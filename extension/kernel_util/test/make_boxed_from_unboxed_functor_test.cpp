/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using exec_aten::TensorImpl;
using executorch::runtime::BoxedEvalueList;
using executorch::runtime::EValue;
using executorch::runtime::getOpsFn;
using executorch::runtime::hasOpsFn;
using executorch::runtime::KernelRuntimeContext;

Tensor& my_op_out(KernelRuntimeContext& ctx, const Tensor& a, Tensor& out) {
  (void)ctx;
  (void)a;
  return out;
}

Tensor& set_1_out(KernelRuntimeContext& ctx, Tensor& out) {
  (void)ctx;
  out.mutable_data_ptr<int32_t>()[0] = 1;
  return out;
}

Tensor&
add_tensor_out(KernelRuntimeContext& ctx, ArrayRef<Tensor> a, Tensor& out) {
  (void)ctx;
  for (int i = 0; i < out.numel(); i++) {
    int sum = 0;
    for (int j = 0; j < a.size(); j++) {
      sum += a[j].const_data_ptr<int32_t>()[i];
    }
    out.mutable_data_ptr<int32_t>()[i] = sum;
  }
  return out;
}

Tensor& add_optional_scalar_out(
    KernelRuntimeContext& ctx,
    optional<int64_t> s1,
    optional<int64_t> s2,
    Tensor& out) {
  (void)ctx;
  if (s1.has_value()) {
    out.mutable_data_ptr<int32_t>()[0] += s1.value();
  }
  if (s2.has_value()) {
    out.mutable_data_ptr<int32_t>()[0] += s2.value();
  }
  return out;
}

Tensor& add_optional_tensor_out(
    KernelRuntimeContext& ctx,
    ArrayRef<optional<Tensor>> a,
    Tensor& out) {
  (void)ctx;
  for (int i = 0; i < a.size(); i++) {
    if (a[i].has_value()) {
      for (int j = 0; j < a[i].value().numel(); j++) {
        out.mutable_data_ptr<int32_t>()[j] +=
            a[i].value().const_data_ptr<int32_t>()[j];
      }
    }
  }
  return out;
}

class MakeBoxedFromUnboxedFunctorTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(MakeBoxedFromUnboxedFunctorTest, Basic) {
  EXECUTORCH_LIBRARY(my_ns, "my_op.out", my_op_out);
  EXPECT_TRUE(hasOpsFn("my_ns::my_op.out"));
}

TEST_F(MakeBoxedFromUnboxedFunctorTest, UnboxLogicWorks) {
  EXECUTORCH_LIBRARY(my_ns, "set_1.out", set_1_out);
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
  KernelRuntimeContext context;
  EValue values[1];
  values[0] = a;
  EValue* stack[1];
  stack[0] = &values[0];

  fn(context, stack);

  // check result
  EXPECT_EQ(a.const_data_ptr<int32_t>()[0], 1);
}

TEST_F(MakeBoxedFromUnboxedFunctorTest, UnboxArrayRef) {
  EXECUTORCH_LIBRARY(my_ns, "add_tensor.out", add_tensor_out);
  EXPECT_TRUE(hasOpsFn("my_ns::add_tensor.out"));

  // prepare ArrayRef input.
  torch::executor::testing::TensorFactory<ScalarType::Int> tf;
  Tensor storage[2] = {tf.ones({5}), tf.ones({5})};
  EValue evalues[2] = {storage[0], storage[1]};
  EValue* values_p[2] = {&evalues[0], &evalues[1]};
  BoxedEvalueList<Tensor> a_box(values_p, storage, 2);
  EValue boxed_array_ref(a_box);
  // prepare out tensor.
  EValue out(tf.zeros({5}));

  auto fn = getOpsFn("my_ns::add_tensor.out");

  // run it.
  KernelRuntimeContext context;
  EValue values[2] = {boxed_array_ref, out};
  EValue* stack[2] = {&values[0], &values[1]};
  fn(context, stack);

  // check result.
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(stack[1]->toTensor().const_data_ptr<int32_t>()[i], 2);
  }
}

TEST_F(MakeBoxedFromUnboxedFunctorTest, UnboxOptional) {
  EXECUTORCH_LIBRARY(my_ns, "add_optional_scalar.out", add_optional_scalar_out);
  EXPECT_TRUE(hasOpsFn("my_ns::add_optional_scalar.out"));

  // prepare optional input.
  EValue scalar((int64_t)3);
  EValue scalar_none;

  // prepare out tensor.
  torch::executor::testing::TensorFactory<ScalarType::Int> tf;
  EValue out(tf.ones({1}));
  auto fn = getOpsFn("my_ns::add_optional_scalar.out");

  // run it.
  KernelRuntimeContext context;
  EValue values[3] = {scalar, scalar_none, out};
  EValue* stack[3] = {&values[0], &values[1], &values[2]};
  fn(context, stack);

  // check result.
  EXPECT_EQ(stack[2]->toTensor().const_data_ptr<int32_t>()[0], 4);
}

TEST_F(MakeBoxedFromUnboxedFunctorTest, UnboxOptionalArrayRef) {
  EXECUTORCH_LIBRARY(my_ns, "add_optional_tensor.out", add_optional_tensor_out);
  EXPECT_TRUE(hasOpsFn("my_ns::add_optional_tensor.out"));

  // prepare optional tensors.
  torch::executor::testing::TensorFactory<ScalarType::Int> tf;
  optional<Tensor> storage[2];
  EValue evalues[2] = {EValue(tf.ones({5})), EValue()};
  EValue* values_p[2] = {&evalues[0], &evalues[1]};
  BoxedEvalueList<optional<Tensor>> a_box(values_p, storage, 2);
  EValue boxed_array_ref(a_box);

  // prepare out tensor.
  EValue out(tf.zeros({5}));
  auto fn = getOpsFn("my_ns::add_optional_tensor.out");

  // run it.
  KernelRuntimeContext context;
  EValue values[2] = {boxed_array_ref, out};
  EValue* stack[2] = {&values[0], &values[1]};
  fn(context, stack);

  // check result.
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(stack[1]->toTensor().const_data_ptr<int32_t>()[i], 1);
  }
}
