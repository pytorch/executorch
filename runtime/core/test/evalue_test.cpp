/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using executorch::runtime::BoxedEvalueList;
using executorch::runtime::EValue;
using executorch::runtime::Tag;
using executorch::runtime::testing::TensorFactory;

TEST(TestEValue, CopyTrivialType) {
  EValue a;
  EValue b(true);
  EXPECT_TRUE(a.isNone());
  a = b;
  EXPECT_TRUE(a.isBool());
  EXPECT_EQ(a.to<bool>(), true);
  EXPECT_EQ(b.to<bool>(), true);
}

TEST(TestEValue, CopyTensor) {
  TensorFactory<ScalarType::Float> tf;
  EValue a(tf.ones({3, 2}));
  EValue b(tf.ones({1}));
  EXPECT_EQ(a.toTensor().dim(), 2);
  a = b;
  EXPECT_EQ(a.toTensor().dim(), 1);
}

TEST(TestEValue, TypeMismatchFatals) {
  ET_EXPECT_DEATH(
      {
        auto e = EValue(true);
        e.toInt();
      },
      "");
}

TEST(TestEValue, NoneByDefault) {
  EValue e;
  EXPECT_TRUE(e.isNone());
}

TEST(TestEValue, ToOptionalInt) {
  EValue e((int64_t)5);
  EXPECT_TRUE(e.isInt());
  EXPECT_FALSE(e.isNone());

  exec_aten::optional<int64_t> o = e.toOptional<int64_t>();
  EXPECT_TRUE(o.has_value());
  EXPECT_EQ(o.value(), 5);
}

TEST(TestEValue, NoneToOptionalInt) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  exec_aten::optional<int64_t> o = e.toOptional<int64_t>();
  EXPECT_FALSE(o.has_value());
}

TEST(TestEValue, ToOptionalScalar) {
  exec_aten::Scalar s((double)3.141);
  EValue e(s);
  EXPECT_TRUE(e.isScalar());
  EXPECT_FALSE(e.isNone());

  exec_aten::optional<exec_aten::Scalar> o = e.toOptional<exec_aten::Scalar>();
  EXPECT_TRUE(o.has_value());
  EXPECT_TRUE(o.value().isFloatingPoint());
  EXPECT_EQ(o.value().to<double>(), 3.141);
}

TEST(TESTEValue, ScalarToType) {
  exec_aten::Scalar s_d((double)3.141);
  EXPECT_EQ(s_d.to<double>(), 3.141);
  exec_aten::Scalar s_i((int64_t)3);
  EXPECT_EQ(s_i.to<int64_t>(), 3);
  exec_aten::Scalar s_b(true);
  EXPECT_EQ(s_b.to<bool>(), true);
}

TEST(TestEValue, NoneToOptionalScalar) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  exec_aten::optional<exec_aten::Scalar> o = e.toOptional<exec_aten::Scalar>();
  EXPECT_FALSE(o.has_value());
}

TEST(TestEValue, NoneToOptionalTensor) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  exec_aten::optional<exec_aten::Tensor> o = e.toOptional<exec_aten::Tensor>();
  EXPECT_FALSE(o.has_value());
}

TEST(TestEValue, ToScalarType) {
  EValue e((int64_t)4);
  auto o = e.toScalarType();
  EXPECT_EQ(o, exec_aten::ScalarType::Long);
  EValue f((int64_t)4);
  auto o2 = e.toOptional<exec_aten::ScalarType>();
  EXPECT_TRUE(o2.has_value());
  EXPECT_EQ(o2.value(), exec_aten::ScalarType::Long);
}

TEST(TestEValue, toString) {
  const EValue e("foo", 3);
  EXPECT_TRUE(e.isString());
  EXPECT_FALSE(e.isNone());

  exec_aten::string_view x = e.toString();
  EXPECT_EQ(x, "foo");
}

TEST(TestEValue, MemoryFormat) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const exec_aten::MemoryFormat m = e.to<exec_aten::MemoryFormat>();
  EXPECT_EQ(m, exec_aten::MemoryFormat::Contiguous);
}

TEST(TestEValue, Layout) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const exec_aten::Layout l = e.to<exec_aten::Layout>();
  EXPECT_EQ(l, exec_aten::Layout::Strided);
}

TEST(TestEValue, Device) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const exec_aten::Device d = e.to<exec_aten::Device>();
  EXPECT_TRUE(d.is_cpu());
}

TEST(TestEValue, BoxedEvalueList) {
  // create fake values table to point to
  EValue values[3] = {
      EValue((int64_t)1), EValue((int64_t)2), EValue((int64_t)3)};
  // create wrapped and unwrapped lists
  EValue* values_p[3] = {&values[0], &values[1], &values[2]};
  int64_t storage[3] = {0, 0, 0};
  // Create Object List and test
  BoxedEvalueList<int64_t> x{values_p, storage, 3};
  auto unwrapped = x.get();
  EXPECT_EQ(unwrapped.size(), 3);
  EXPECT_EQ(unwrapped[0], 1);
  EXPECT_EQ(unwrapped[1], 2);
  EXPECT_EQ(unwrapped[2], 3);
}

TEST(TestEValue, toOptionalTensorList) {
  // create list, empty evalue ctor gets tag::None
  EValue values[2] = {EValue(), EValue()};
  EValue* values_p[2] = {&values[0], &values[1]};
  exec_aten::optional<exec_aten::Tensor> storage[2];
  // wrap in array ref
  BoxedEvalueList<exec_aten::optional<exec_aten::Tensor>> a(
      values_p, storage, 2);

  // create Evalue
  EValue e(a);
  e.tag = Tag::ListOptionalTensor;
  EXPECT_TRUE(e.isListOptionalTensor());

  // Convert back to list
  exec_aten::ArrayRef<exec_aten::optional<exec_aten::Tensor>> x =
      e.toListOptionalTensor();
  EXPECT_EQ(x.size(), 2);
  EXPECT_FALSE(x[0].has_value());
  EXPECT_FALSE(x[1].has_value());
}
