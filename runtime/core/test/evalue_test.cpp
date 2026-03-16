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

using executorch::aten::ScalarType;
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
  explicit TensorWrapper(executorch::aten::Tensor tensor)
      : tensor_(std::make_unique<executorch::aten::Tensor>(std::move(tensor))) {
  }

  executorch::aten::Tensor& operator*() {
    return *tensor_;
  }

  const executorch::aten::Tensor& operator*() const {
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
  std::unique_ptr<executorch::aten::Tensor> tensor_;
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
      "EValue is not an int");
}

TEST_F(EValueTest, NoneByDefault) {
  EValue e;
  EXPECT_TRUE(e.isNone());
}

TEST_F(EValueTest, ToOptionalInt) {
  EValue e((int64_t)5);
  EXPECT_TRUE(e.isInt());
  EXPECT_FALSE(e.isNone());

  std::optional<int64_t> o = e.toOptional<int64_t>();
  EXPECT_TRUE(o.has_value());
  EXPECT_EQ(o.value(), 5);
}

TEST_F(EValueTest, NoneToOptionalInt) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  std::optional<int64_t> o = e.toOptional<int64_t>();
  EXPECT_FALSE(o.has_value());
}

TEST_F(EValueTest, ToOptionalScalar) {
  executorch::aten::Scalar s((double)3.141);
  EValue e(s);
  EXPECT_TRUE(e.isScalar());
  EXPECT_FALSE(e.isNone());

  std::optional<executorch::aten::Scalar> o =
      e.toOptional<executorch::aten::Scalar>();
  EXPECT_TRUE(o.has_value());
  EXPECT_TRUE(o.value().isFloatingPoint());
  EXPECT_EQ(o.value().to<double>(), 3.141);
}

TEST_F(EValueTest, ScalarToType) {
  executorch::aten::Scalar s_d((double)3.141);
  EXPECT_EQ(s_d.to<double>(), 3.141);
  executorch::aten::Scalar s_i((int64_t)3);
  EXPECT_EQ(s_i.to<int64_t>(), 3);
  executorch::aten::Scalar s_b(true);
  EXPECT_EQ(s_b.to<bool>(), true);
}

TEST_F(EValueTest, NoneToOptionalScalar) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  std::optional<executorch::aten::Scalar> o =
      e.toOptional<executorch::aten::Scalar>();
  EXPECT_FALSE(o.has_value());
}

TEST_F(EValueTest, NoneToOptionalTensor) {
  EValue e;
  EXPECT_TRUE(e.isNone());

  std::optional<executorch::aten::Tensor> o =
      e.toOptional<executorch::aten::Tensor>();
  EXPECT_FALSE(o.has_value());
}

TEST_F(EValueTest, ToScalarType) {
  EValue e((int64_t)4);
  auto o = e.toScalarType();
  EXPECT_EQ(o, executorch::aten::ScalarType::Long);
  EValue f((int64_t)4);
  auto o2 = e.toOptional<executorch::aten::ScalarType>();
  EXPECT_TRUE(o2.has_value());
  EXPECT_EQ(o2.value(), executorch::aten::ScalarType::Long);
}

TEST_F(EValueTest, toString) {
  auto string_ref =
      std::make_unique<executorch::aten::ArrayRef<char>>("foo", 3);
  const EValue e(string_ref.get());
  EXPECT_TRUE(e.isString());
  EXPECT_FALSE(e.isNone());

  std::string_view x = e.toString();
  EXPECT_EQ(x, "foo");
}

TEST_F(EValueTest, MemoryFormat) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const executorch::aten::MemoryFormat m =
      e.to<executorch::aten::MemoryFormat>();
  EXPECT_EQ(m, executorch::aten::MemoryFormat::Contiguous);
}

TEST_F(EValueTest, Layout) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const executorch::aten::Layout l = e.to<executorch::aten::Layout>();
  EXPECT_EQ(l, executorch::aten::Layout::Strided);
}

TEST_F(EValueTest, Device) {
  const EValue e((int64_t)0);
  EXPECT_TRUE(e.isInt());
  const executorch::aten::Device d = e.to<executorch::aten::Device>();
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
  std::optional<executorch::aten::Tensor> storage[2];
  // wrap in array ref
  auto boxed_list = std::make_unique<
      BoxedEvalueList<std::optional<executorch::aten::Tensor>>>(
      values_p, storage, 2);

  // create Evalue
  EValue e(boxed_list.get());
  e.tag = Tag::ListOptionalTensor;
  EXPECT_TRUE(e.isListOptionalTensor());

  // Convert back to list
  executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>> x =
      e.toListOptionalTensor();
  EXPECT_EQ(x.size(), 2);
  EXPECT_FALSE(x[0].has_value());
  EXPECT_FALSE(x[1].has_value());
}

TEST_F(EValueTest, ConstructFromUniquePtr) {
  TensorFactory<ScalarType::Float> tf;
  auto tensor_ptr = std::make_unique<executorch::aten::Tensor>(tf.ones({2, 3}));

  EValue evalue(std::move(tensor_ptr));

  EXPECT_TRUE(evalue.isTensor());
  EXPECT_EQ(evalue.toTensor().dim(), 2);
  EXPECT_EQ(evalue.toTensor().numel(), 6);

  EValue evalue2(std::make_unique<executorch::aten::Tensor>(tf.ones({4, 5})));

  EXPECT_TRUE(evalue2.isTensor());
  EXPECT_EQ(evalue2.toTensor().dim(), 2);
  EXPECT_EQ(evalue2.toTensor().numel(), 20);
}

TEST_F(EValueTest, ConstructFromSharedPtr) {
  TensorFactory<ScalarType::Float> tf;
  auto tensor_ptr = std::make_shared<executorch::aten::Tensor>(tf.ones({4, 5}));

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
  std::unique_ptr<executorch::aten::Tensor> null_ptr;

  ET_EXPECT_DEATH({ EValue evalue(null_ptr); }, "Pointer is null");
}

TEST_F(EValueTest, StringConstructorNullCheck) {
  executorch::aten::ArrayRef<char>* null_string_ptr = nullptr;
  ET_EXPECT_DEATH(
      { EValue evalue(null_string_ptr); }, "pointer cannot be null");
}

TEST_F(EValueTest, BoolListConstructorNullCheck) {
  executorch::aten::ArrayRef<bool>* null_bool_list_ptr = nullptr;
  ET_EXPECT_DEATH(
      { EValue evalue(null_bool_list_ptr); }, "pointer cannot be null");
}

TEST_F(EValueTest, DoubleListConstructorNullCheck) {
  executorch::aten::ArrayRef<double>* null_double_list_ptr = nullptr;
  ET_EXPECT_DEATH(
      { EValue evalue(null_double_list_ptr); }, "pointer cannot be null");
}

TEST_F(EValueTest, IntListConstructorNullCheck) {
  BoxedEvalueList<int64_t>* null_int_list_ptr = nullptr;
  ET_EXPECT_DEATH(
      { EValue evalue(null_int_list_ptr); }, "pointer cannot be null");
}

TEST_F(EValueTest, TensorListConstructorNullCheck) {
  BoxedEvalueList<executorch::aten::Tensor>* null_tensor_list_ptr = nullptr;
  ET_EXPECT_DEATH(
      { EValue evalue(null_tensor_list_ptr); }, "pointer cannot be null");
}

TEST_F(EValueTest, OptionalTensorListConstructorNullCheck) {
  BoxedEvalueList<std::optional<executorch::aten::Tensor>>*
      null_optional_tensor_list_ptr = nullptr;
  ET_EXPECT_DEATH(
      { EValue evalue(null_optional_tensor_list_ptr); },
      "pointer cannot be null");
}

TEST_F(EValueTest, BoxedEvalueListConstructorNullChecks) {
  std::array<int64_t, 3> storage = {0, 0, 0};
  std::array<EValue, 3> values = {
      EValue((int64_t)1), EValue((int64_t)2), EValue((int64_t)3)};
  std::array<EValue*, 3> values_p = {&values[0], &values[1], &values[2]};

  // Test null wrapped_vals
  ET_EXPECT_DEATH(
      { BoxedEvalueList<int64_t> list(nullptr, storage.data(), 3); },
      "wrapped_vals cannot be null");

  // Test null unwrapped_vals
  ET_EXPECT_DEATH(
      { BoxedEvalueList<int64_t> list(values_p.data(), nullptr, 3); },
      "unwrapped_vals cannot be null");

  // Test negative size
  ET_EXPECT_DEATH(
      { BoxedEvalueList<int64_t> list(values_p.data(), storage.data(), -1); },
      "size cannot be negative");
}

TEST_F(EValueTest, toListOptionalTensorTypeCheck) {
  // Create an EValue that's not a ListOptionalTensor
  EValue e((int64_t)42);
  EXPECT_TRUE(e.isInt());
  EXPECT_FALSE(e.isListOptionalTensor());

  // Should fail type check
  ET_EXPECT_DEATH({ e.toListOptionalTensor(); }, "EValue is not a");
}

TEST_F(EValueTest, toStringNullPointerCheck) {
  // Create an EValue with String tag but null pointer
  EValue e;
  e.tag = Tag::String;
  e.payload.copyable_union.as_string_ptr = nullptr;

  // Should pass isString() check but fail null pointer check
  EXPECT_TRUE(e.isString());
  ET_EXPECT_DEATH({ e.toString(); }, "string pointer is null");
}

TEST_F(EValueTest, toIntListNullPointerCheck) {
  // Create an EValue with ListInt tag but null pointer
  EValue e;
  e.tag = Tag::ListInt;
  e.payload.copyable_union.as_int_list_ptr = nullptr;

  // Should pass isIntList() check but fail null pointer check
  EXPECT_TRUE(e.isIntList());
  ET_EXPECT_DEATH({ e.toIntList(); }, "int list pointer is null");
}

TEST_F(EValueTest, toBoolListNullPointerCheck) {
  // Create an EValue with ListBool tag but null pointer
  EValue e;
  e.tag = Tag::ListBool;
  e.payload.copyable_union.as_bool_list_ptr = nullptr;

  // Should pass isBoolList() check but fail null pointer check
  EXPECT_TRUE(e.isBoolList());
  ET_EXPECT_DEATH({ e.toBoolList(); }, "bool list pointer is null");
}

TEST_F(EValueTest, toDoubleListNullPointerCheck) {
  // Create an EValue with ListDouble tag but null pointer
  EValue e;
  e.tag = Tag::ListDouble;
  e.payload.copyable_union.as_double_list_ptr = nullptr;

  // Should pass isDoubleList() check but fail null pointer check
  EXPECT_TRUE(e.isDoubleList());
  ET_EXPECT_DEATH({ e.toDoubleList(); }, "double list pointer is null");
}

TEST_F(EValueTest, toTensorListNullPointerCheck) {
  // Create an EValue with ListTensor tag but null pointer
  EValue e;
  e.tag = Tag::ListTensor;
  e.payload.copyable_union.as_tensor_list_ptr = nullptr;

  // Should pass isTensorList() check but fail null pointer check
  EXPECT_TRUE(e.isTensorList());
  ET_EXPECT_DEATH({ e.toTensorList(); }, "tensor list pointer is null");
}

TEST_F(EValueTest, toListOptionalTensorNullPointerCheck) {
  // Create an EValue with ListOptionalTensor tag but null pointer
  EValue e;
  e.tag = Tag::ListOptionalTensor;
  e.payload.copyable_union.as_list_optional_tensor_ptr = nullptr;

  // Should pass isListOptionalTensor() check but fail null pointer check
  EXPECT_TRUE(e.isListOptionalTensor());
  ET_EXPECT_DEATH({ e.toListOptionalTensor(); }, "pointer is null");
}
