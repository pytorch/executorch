/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/evalue.h>

#include <gtest/gtest.h>

#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::testing;

using exec_aten::ScalarType;
using executorch::runtime::BoxedEvalueList;
using executorch::runtime::EValue;
using executorch::runtime::Tag;
using executorch::runtime::testing::TensorFactory;

class EValueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    executorch::runtime::runtime_init();
  }
};

// An utility class used in tests to simulate objects that manage Tensors.
// The overloaded operator*() is used to return the underlying Tensor, mimicking
// behavior of smart pointers.
class TensorWrapper {
 public:
  explicit TensorWrapper(exec_aten::Tensor tensor)
      : tensor_(std::make_unique<exec_aten::Tensor>(std::move(tensor))) {}

  exec_aten::Tensor& operator*() {
    return *tensor_;
  }

  const exec_aten::Tensor& operator*() const {
    return *tensor_;
  }

  operator bool() const {
    return static_cast<bool>(tensor_);
  }

  bool operator==(std::nullptr_t) const {
    return tensor_ == nullptr;
  }

  bool operator!=(std::nullptr_t) const {
    return tensor_ != nullptr;
  }

 private:
  std::unique_ptr<exec_aten::Tensor> tensor_;
};

TEST_F(EValueTest, CopyTrivialType) {
  EValue a;
  EValue b(true);
  EXPECT_TRUE(a.isNone());
  a = b;
  EXPECT_TRUE(a.isBool());
  EXPECT_EQ(a.to<bool>(), true);
  EXPECT_EQ(b.to<bool>(), true);
}

TEST_F(EValueTest, CopyTensor) {
  TensorFactory<ScalarType::Float> tf;
  EValue a(tf.ones({3, 2}));
  EValue b(tf.ones({1}));
  EXPECT_EQ(a.toTensor().dim(), 2);
  a = b;
  EXPECT_EQ(a.toTensor().dim(), 1);
}

TEST_F(EValueTest, TypeMismatchFatals) {
  ET_EXPECT_DEATH(
      {
        auto e = EValue(true);
        e.toInt();
      },
      "");
}

TEST_F(EValueTest, NoneByDefault) {
  EValue e;
  EXPECT_TRUE(e.isNone());
}

TEST_F(EValueTest, ToOptionalInt) {
  EValue e((int64_t)5);
  EXPECT_TRUE(e.isInt());
  EXPECT_FALSE(e.isNone());

  exec_aten::optional<int64_t> o = e.toOptional<int64_t>();
  EXPECT_TRUE(o.has_value());
  EXPECT_EQ(o.value(), 5);
}

TEST_F(EValueTest, NoneToOptionalInt) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  exec_aten::optional<int64_t> o = e.toOptional<int64_t>();
  EXPECT_FALSE(o.has_value());
}

TEST_F(EValueTest, ToOptionalScalar) {
  exec_aten::Scalar s((double)3.141);
  EValue e(s);
  EXPECT_TRUE(e.isScalar());
  EXPECT_FALSE(e.isNone());

  exec_aten::optional<exec_aten::Scalar> o = e.toOptional<exec_aten::Scalar>();
  EXPECT_TRUE(o.has_value());
  EXPECT_TRUE(o.value().isFloatingPoint());
  EXPECT_EQ(o.value().to<double>(), 3.141);
}

TEST_F(EValueTest, ScalarToType) {
  exec_aten::Scalar s_d((double)3.141);
  EXPECT_EQ(s_d.to<double>(), 3.141);
  exec_aten::Scalar s_i((int64_t)3);
  EXPECT_EQ(s_i.to<int64_t>(), 3);
  exec_aten::Scalar s_b(true);
  EXPECT_EQ(s_b.to<bool>(), true);
}

TEST_F(EValueTest, NoneToOptionalScalar) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  exec_aten::optional<exec_aten::Scalar> o = e.toOptional<exec_aten::Scalar>();
  EXPECT_FALSE(o.has_value());
}

TEST_F(EValueTest, NoneToOptionalTensor) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  exec_aten::optional<exec_aten::Tensor> o = e.toOptional<exec_aten::Tensor>();
  EXPECT_FALSE(o.has_value());
}

TEST_F(EValueTest, ToScalarType) {
  EValue e((int64_t)4);
  auto o = e.toScalarType();
  EXPECT_EQ(o, exec_aten::ScalarType::Long);
  EValue f((int64_t)4);
  auto o2 = e.toOptional<exec_aten::ScalarType>();
  EXPECT_TRUE(o2.has_value());
  EXPECT_EQ(o2.value(), exec_aten::ScalarType::Long);
}

TEST_F(EValueTest, toString) {
  const EValue e("foo", 3);
  EXPECT_TRUE(e.isString());
  EXPECT_FALSE(e.isNone());

  exec_aten::string_view x = e.toString();
  EXPECT_EQ(x, "foo");
}

TEST_F(EValueTest, MemoryFormat) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const exec_aten::MemoryFormat m = e.to<exec_aten::MemoryFormat>();
  EXPECT_EQ(m, exec_aten::MemoryFormat::Contiguous);
}

TEST_F(EValueTest, Layout) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const exec_aten::Layout l = e.to<exec_aten::Layout>();
  EXPECT_EQ(l, exec_aten::Layout::Strided);
}

TEST_F(EValueTest, Device) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const exec_aten::Device d = e.to<exec_aten::Device>();
  EXPECT_TRUE(d.is_cpu());
}

TEST_F(EValueTest, BoxedEvalueList) {
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

TEST_F(EValueTest, toOptionalTensorList) {
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

TEST_F(EValueTest, ConstructFromUniquePtr) {
  TensorFactory<ScalarType::Float> tf;
  auto tensor_ptr = std::make_unique<exec_aten::Tensor>(tf.ones({2, 3}));

  EValue evalue(std::move(tensor_ptr));

  EXPECT_TRUE(evalue.isTensor());
  EXPECT_EQ(evalue.toTensor().dim(), 2);
  EXPECT_EQ(evalue.toTensor().numel(), 6);

  EValue evalue2(std::make_unique<exec_aten::Tensor>(tf.ones({4, 5})));

  EXPECT_TRUE(evalue2.isTensor());
  EXPECT_EQ(evalue2.toTensor().dim(), 2);
  EXPECT_EQ(evalue2.toTensor().numel(), 20);
}

TEST_F(EValueTest, ConstructFromSharedPtr) {
  TensorFactory<ScalarType::Float> tf;
  auto tensor_ptr = std::make_shared<exec_aten::Tensor>(tf.ones({4, 5}));

  EValue evalue(tensor_ptr);

  EXPECT_TRUE(evalue.isTensor());
  EXPECT_EQ(evalue.toTensor().dim(), 2);
  EXPECT_EQ(evalue.toTensor().numel(), 20);
}

TEST_F(EValueTest, ConstructFromTensorWrapper) {
  TensorFactory<ScalarType::Float> tf;
  TensorWrapper tensor_wrapper(tf.ones({4, 5}));

  EValue evalue(tensor_wrapper);

  EXPECT_TRUE(evalue.isTensor());
  EXPECT_EQ(evalue.toTensor().dim(), 2);
  EXPECT_EQ(evalue.toTensor().numel(), 20);
}

TEST_F(EValueTest, ConstructFromNullPtrAborts) {
  std::unique_ptr<exec_aten::Tensor> null_ptr;

  ET_EXPECT_DEATH({ EValue evalue(null_ptr); }, "");
}
