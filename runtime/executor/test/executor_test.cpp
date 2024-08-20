/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/extension/pytree/pytree.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using exec_aten::IntArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::SizesType;
using exec_aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::getOpsFn;
using executorch::runtime::hasOpsFn;
using executorch::runtime::Kernel;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::register_kernels;
using executorch::runtime::testing::TensorFactory;

class ExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(ExecutorTest, Tensor) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make({2, 2}, {1, 2, 3, 4});

  auto data_p = a.const_data_ptr<int32_t>();
  ASSERT_EQ(data_p[0], 1);
  ASSERT_EQ(data_p[1], 2);
  ASSERT_EQ(data_p[2], 3);
  ASSERT_EQ(data_p[3], 4);
}

TEST_F(ExecutorTest, EValue) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make({2, 2}, {1, 2, 3, 4});

  EValue v(a);
  ASSERT_TRUE(v.isTensor());
  ASSERT_EQ(v.toTensor().nbytes(), 16);
}

/**
 * According to the precision limitations listed here:
 * https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations
 * The max precision error for a half in the range [2^n, 2^(n+1)] is 2^(n-10)
 */
float toleranceFloat16(float f) {
  return pow(2, static_cast<int>(log2(fabs(f))) - 10);
}

TEST_F(ExecutorTest, TensorHalf) {
  TensorFactory<ScalarType::Half> tf;
  Tensor a = tf.make({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

  ASSERT_EQ(a.nbytes(), 8);
  ASSERT_EQ(a.element_size(), 2);
  ASSERT_EQ(a.numel(), 4);
  ASSERT_EQ(a.scalar_type(), ScalarType::Half);

  auto data_p = a.const_data_ptr<exec_aten::Half>();
  ASSERT_NEAR(
      data_p[0], 1.0f, toleranceFloat16(fmax(fabs(1.0f), fabs(data_p[0]))));
  ASSERT_NEAR(
      data_p[1], 2.0f, toleranceFloat16(fmax(fabs(2.0f), fabs(data_p[1]))));
}

TEST_F(ExecutorTest, RegistryLookupAndCall) {
  const char* op_name = "aten::add.out";
  ASSERT_TRUE(hasOpsFn(op_name));
  auto func = getOpsFn(op_name);
  ASSERT_TRUE(func);

  TensorFactory<ScalarType::Int> tf;
  constexpr size_t num_evalues = 4;
  EValue evalues[num_evalues] = {
      tf.make({2, 2}, {1, 2, 3, 4}),
      tf.make({2, 2}, {5, 6, 7, 8}),
      Scalar(1),
      tf.make({2, 2}, {0, 0, 0, 0}),
  };

  EValue* kernel_args[5];
  for (size_t i = 0; i < num_evalues; i++) {
    kernel_args[i] = &evalues[i];
  }
  // x and x_out args are same evalue for out variant kernels
  kernel_args[4] = &evalues[3];

  KernelRuntimeContext context{};
  func(context, kernel_args);
  auto c_ptr = evalues[3].toTensor().const_data_ptr<int32_t>();
  ASSERT_EQ(c_ptr[3], 12);
}

TEST_F(ExecutorTest, IntArrayRefSingleElement) {
  // Create an IntArrayRef with a single element. `ref` will contain a pointer
  // to `one`, which must outlive the array ref.
  const IntArrayRef::value_type one = 1;
  IntArrayRef ref(one);
  EXPECT_EQ(ref[0], 1);
}

TEST_F(ExecutorTest, IntArrayRefDataAndLength) {
  // Create an IntArrayRef from an array. `ref` will contain a pointer to
  // `array`, which must outlive the array ref.
  const IntArrayRef::value_type array[4] = {5, 6, 7, 8};
  const IntArrayRef::size_type length = 4;
  IntArrayRef ref(array, length);

  EXPECT_EQ(ref.size(), length);
  EXPECT_EQ(ref.front(), 5);
  EXPECT_EQ(ref.back(), 8);
}

TEST_F(ExecutorTest, EValueFromScalar) {
  Scalar b((bool)true);
  Scalar i((int64_t)2);
  Scalar d((double)3.0);

  EValue evalue_b(b);
  ASSERT_TRUE(evalue_b.isScalar());
  ASSERT_TRUE(evalue_b.isBool());
  ASSERT_EQ(evalue_b.toBool(), true);

  EValue evalue_i(i);
  ASSERT_TRUE(evalue_i.isScalar());
  ASSERT_TRUE(evalue_i.isInt());
  ASSERT_EQ(evalue_i.toInt(), 2);

  EValue evalue_d(d);
  ASSERT_TRUE(evalue_d.isScalar());
  ASSERT_TRUE(evalue_d.isDouble());
  ASSERT_NEAR(evalue_d.toDouble(), 3.0, 0.01);
}

TEST_F(ExecutorTest, EValueToScalar) {
  EValue v((int64_t)2);
  ASSERT_TRUE(v.isScalar());

  Scalar s = v.toScalar();
  ASSERT_TRUE(s.isIntegral(false));
  ASSERT_EQ(s.to<int64_t>(), 2);
}

void test_op(KernelRuntimeContext& /*unused*/, EValue** /*unused*/) {}

TEST_F(ExecutorTest, OpRegistration) {
  auto s1 = register_kernels({Kernel("test", test_op)});
  auto s2 = register_kernels({Kernel("test_2", test_op)});
  ASSERT_EQ(Error::Ok, s1);
  ASSERT_EQ(Error::Ok, s2);
  ET_EXPECT_DEATH(
      []() { (void)register_kernels({Kernel("test", test_op)}); }(), "");

  ASSERT_TRUE(hasOpsFn("test"));
  ASSERT_TRUE(hasOpsFn("test_2"));
}

TEST_F(ExecutorTest, OpRegistrationWithContext) {
  auto op = Kernel(
      "test_op_with_context",
      [](KernelRuntimeContext& context, EValue** values) {
        (void)context;
        *(values[0]) = Scalar(100);
      });
  auto s1 = register_kernels({op});
  ASSERT_EQ(Error::Ok, s1);
  ASSERT_TRUE(hasOpsFn("test_op_with_context"));

  auto func = getOpsFn("test_op_with_context");
  EValue values[1];
  values[0] = Scalar(0);
  EValue* kernels[1];
  kernels[0] = &values[0];
  KernelRuntimeContext context{};
  func(context, kernels);

  auto val = values[0].toScalar().to<int64_t>();
  ASSERT_EQ(val, 100);
}

TEST_F(ExecutorTest, AddMulAlreadyRegistered) {
  ASSERT_TRUE(hasOpsFn("aten::add.out"));
  ASSERT_TRUE(hasOpsFn("aten::mul.out"));
}

TEST(PyTreeEValue, List) {
  std::string spec = "L2#1#1($,$)";

  Scalar i((int64_t)2);
  Scalar d((double)3.0);
  EValue items[2] = {i, d};

  auto c = torch::executor::pytree::unflatten(spec, items);
  ASSERT_TRUE(c.isList());
  ASSERT_EQ(c.size(), 2);

  const auto& child0 = c[0];
  const auto& child1 = c[1];

  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());

  EValue ev_child0 = child0;
  ASSERT_TRUE(ev_child0.isScalar());
  ASSERT_TRUE(ev_child0.isInt());
  ASSERT_EQ(ev_child0.toInt(), 2);

  ASSERT_TRUE(child1.leaf().isScalar());
  ASSERT_TRUE(child1.leaf().isDouble());
  ASSERT_NEAR(child1.leaf().toDouble(), 3.0, 0.01);
}

auto unflatten(EValue* items) {
  std::string spec = "D4#1#1#1#1('key0':$,1:$,23:$,123:$)";
  return torch::executor::pytree::unflatten(spec, items);
}

TEST(PyTreeEValue, DestructedSpec) {
  Scalar i0((int64_t)2);
  Scalar d1((double)3.0);
  Scalar i2((int64_t)4);
  Scalar d3((double)5.0);
  EValue items[4] = {i0, d1, i2, d3};
  auto c = unflatten(items);

  ASSERT_TRUE(c.isDict());
  ASSERT_EQ(c.size(), 4);

  auto& key0 = c.key(0);
  auto& key1 = c.key(1);

  ASSERT_TRUE(key0 == torch::executor::pytree::Key("key0"));
  ASSERT_TRUE(key1 == torch::executor::pytree::Key(1));

  const auto& child0 = c[0];
  const auto& child1 = c[1];
  ASSERT_TRUE(child0.isLeaf());
  ASSERT_TRUE(child1.isLeaf());

  EValue ev_child0 = child0;
  ASSERT_TRUE(ev_child0.isScalar());
  ASSERT_TRUE(ev_child0.isInt());
  ASSERT_EQ(ev_child0.toInt(), 2);

  ASSERT_TRUE(child1.leaf().isScalar());
  ASSERT_TRUE(child1.leaf().isDouble());
  ASSERT_NEAR(child1.leaf().toDouble(), 3.0, 0.01);
}
