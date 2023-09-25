/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/evalue_util/print_evalue.h>

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <array>
#include <cmath>
#include <memory>
#include <sstream>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using torch::executor::BoxedEvalueList;
using torch::executor::EValue;
using torch::executor::testing::TensorFactory;

void expect_output(const EValue& value, const char* expected) {
  std::ostringstream os;
  os << value;
  EXPECT_STREQ(expected, os.str().c_str());
}

//
// None
//

TEST(PrintEvalueTest, None) {
  EValue value;
  expect_output(value, "None");
}

//
// Bool
//

TEST(PrintEvalueTest, TrueBool) {
  EValue value(exec_aten::Scalar(true));
  expect_output(value, "True");
}

TEST(PrintEvalueTest, FalseBool) {
  EValue value(exec_aten::Scalar(false));
  expect_output(value, "False");
}

//
// Int
//

TEST(PrintEvalueTest, ZeroInt) {
  EValue value(exec_aten::Scalar(0));
  expect_output(value, "0");
}

TEST(PrintEvalueTest, PositiveInt) {
  EValue value(exec_aten::Scalar(10));
  expect_output(value, "10");
}

TEST(PrintEvalueTest, NegativeInt) {
  EValue value(exec_aten::Scalar(-10));
  expect_output(value, "-10");
}

TEST(PrintEvalueTest, LargePositiveInt) {
  // A value that can't fit in 32 bits. Saying Scalar(<literal-long-long>) is
  // ambiguous with c10::Scalar, so use a non-literal value.
  constexpr int64_t i = 1152921504606846976;
  EValue value = {exec_aten::Scalar(i)};
  expect_output(value, "1152921504606846976");
}

TEST(PrintEvalueTest, LargeNegativeInt) {
  // A value that can't fit in 32 bits. Saying Scalar(<literal-long-long>) is
  // ambiguous with c10::Scalar, so use a non-literal value.
  constexpr int64_t i = -1152921504606846976;
  EValue value = {exec_aten::Scalar(i)};
  expect_output(value, "-1152921504606846976");
}

//
// Double
//

TEST(PrintEvalueTest, ZeroDouble) {
  EValue value(exec_aten::Scalar(0.0));
  expect_output(value, "0.");
}

TEST(PrintEvalueTest, PositiveZeroDouble) {
  EValue value(exec_aten::Scalar(+0.0));
  expect_output(value, "0.");
}

TEST(PrintEvalueTest, NegativeZeroDouble) {
  EValue value(exec_aten::Scalar(-0.0));
  expect_output(value, "-0.");
}

TEST(PrintEvalueTest, PositiveIntegralDouble) {
  EValue value(exec_aten::Scalar(10.0));
  expect_output(value, "10.");
}

TEST(PrintEvalueTest, PositiveFractionalDouble) {
  EValue value(exec_aten::Scalar(10.1));
  expect_output(value, "10.1");
}

TEST(PrintEvalueTest, NegativeIntegralDouble) {
  EValue value(exec_aten::Scalar(-10.0));
  expect_output(value, "-10.");
}

TEST(PrintEvalueTest, NegativeFractionalDouble) {
  EValue value(exec_aten::Scalar(-10.1));
  expect_output(value, "-10.1");
}

TEST(PrintEvalueTest, PositiveInfinityDouble) {
  EValue value((exec_aten::Scalar(INFINITY)));
  expect_output(value, "inf");
}

TEST(PrintEvalueTest, NegativeInfinityDouble) {
  EValue value((exec_aten::Scalar(-INFINITY)));
  expect_output(value, "-inf");
}

TEST(PrintEvalueTest, NaNDouble) {
  EValue value((exec_aten::Scalar(NAN)));
  expect_output(value, "nan");
}

// Don't test exponents or values with larger numbers of truncated decimal
// digits since their formatting may be system-dependent.

//
// String
//

TEST(PrintEvalueTest, EmptyString) {
  std::string str = "";
  EValue value(str.c_str(), str.size());
  expect_output(value, "\"\"");
}

TEST(PrintEvalueTest, BasicString) {
  // No escaping required.
  std::string str = "Test Data";
  EValue value(str.c_str(), str.size());
  expect_output(value, "\"Test Data\"");
}

TEST(PrintEvalueTest, EscapedString) {
  // Contains characters that need to be escaped.
  std::string str = "double quote: \" backslash: \\";
  EValue value(str.c_str(), str.size());
  expect_output(value, "\"double quote: \\\" backslash: \\\\\"");
}

//
// Tensor
//

TEST(PrintEvalueTest, BoolTensor) {
  TensorFactory<ScalarType::Bool> tf;
  {
    // Unelided
    EValue value(tf.make({2, 2}, {true, false, true, false}));
    expect_output(value, "tensor(sizes=[2, 2], [True, False, True, False])");
  }
  {
    // Elided
    EValue value(tf.make(
        {5, 2},
        {true, false, true, false, true, false, true, false, true, false}));
    expect_output(
        value,
        "tensor(sizes=[5, 2], [True, False, True, ..., False, True, False])");
  }
}

template <ScalarType DTYPE>
void test_print_integer_tensor() {
  TensorFactory<DTYPE> tf;
  {
    // Unelided
    EValue value(tf.make({2, 2}, {1, 2, 3, 4}));
    expect_output(value, "tensor(sizes=[2, 2], [1, 2, 3, 4])");
  }
  {
    // Elided
    EValue value(tf.make({5, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    expect_output(value, "tensor(sizes=[5, 2], [1, 2, 3, ..., 8, 9, 10])");
  }
}

TEST(PrintEvalueTest, ByteTensor) {
  test_print_integer_tensor<ScalarType::Byte>();
}

TEST(PrintEvalueTest, CharTensor) {
  test_print_integer_tensor<ScalarType::Char>();
}

TEST(PrintEvalueTest, ShortTensor) {
  test_print_integer_tensor<ScalarType::Short>();
}

TEST(PrintEvalueTest, IntTensor) {
  test_print_integer_tensor<ScalarType::Int>();
}

TEST(PrintEvalueTest, LongTensor) {
  test_print_integer_tensor<ScalarType::Long>();
}

template <ScalarType DTYPE>
void test_print_float_tensor() {
  TensorFactory<DTYPE> tf;
  {
    // Unelided
    EValue value(tf.make({2, 2}, {1.0, 2.2, 3.3, 4.0}));
    expect_output(value, "tensor(sizes=[2, 2], [1., 2.2, 3.3, 4.])");
  }
  {
    // Elided
    EValue value(
        tf.make({5, 2}, {1.0, 2.2, 3.3, 4.0, 5.5, 6.6, 7.0, 8.8, 9.0, 10.1}));
    expect_output(
        value, "tensor(sizes=[5, 2], [1., 2.2, 3.3, ..., 8.8, 9., 10.1])");
  }
}

TEST(PrintEvalueTest, FloatTensor) {
  test_print_float_tensor<ScalarType::Float>();
}

TEST(PrintEvalueTest, DoubleTensor) {
  test_print_float_tensor<ScalarType::Double>();
}

//
// BoolList
//

TEST(PrintEvalueTest, UnelidedBoolLists) {
  // Default edge items is 3, so the longest unelided list length is 6.
  std::array<bool, 6> list = {true, false, true, false, true, false};

  // Important to test the cases where the length is less than or equal to the
  // number of edge items (3 by default). Use bool as a proxy for this edge
  // case; the other scalar types use the same underlying code, so they don't
  // need to test this again.
  {
    EValue value(ArrayRef<bool>(list.data(), 0ul));
    expect_output(value, "(len=0)[]");
  }
  {
    EValue value(ArrayRef<bool>(list.data(), 1));
    expect_output(value, "(len=1)[True]");
  }
  {
    EValue value(ArrayRef<bool>(list.data(), 2));
    expect_output(value, "(len=2)[True, False]");
  }
  {
    EValue value(ArrayRef<bool>(list.data(), 3));
    expect_output(value, "(len=3)[True, False, True]");
  }
  {
    EValue value(ArrayRef<bool>(list.data(), 4));
    expect_output(value, "(len=4)[True, False, True, False]");
  }
  {
    EValue value(ArrayRef<bool>(list.data(), 5));
    expect_output(value, "(len=5)[True, False, True, False, True]");
  }
  {
    EValue value(ArrayRef<bool>(list.data(), 6));
    expect_output(value, "(len=6)[True, False, True, False, True, False]");
  }
}

TEST(PrintEvalueTest, ElidedBoolLists) {
  std::array<bool, 10> list = {
      true, false, true, false, true, false, true, false, true, false};

  {
    // Default edge items is 3, so the shortest elided list length is 7.
    EValue value(ArrayRef<bool>(list.data(), 7));
    expect_output(value, "(len=7)[True, False, True, ..., True, False, True]");
  }
  {
    EValue value(ArrayRef<bool>(list.data(), 8));
    expect_output(value, "(len=8)[True, False, True, ..., False, True, False]");
  }
  {
    // Multi-digit length.
    EValue value(ArrayRef<bool>(list.data(), 10));
    expect_output(
        value, "(len=10)[True, False, True, ..., False, True, False]");
  }
}

//
// IntList
//

TEST(PrintEvalueTest, UnelidedIntLists) {
  // Default edge items is 3, so the longest unelided list length is 6. EValue
  // treats int lists specially, and must be constructed from a BoxedEvalueList
  // instead of from an ArrayRef.
  std::array<EValue, 6> values = {
      Scalar(-2), Scalar(-1), Scalar(0), Scalar(1), Scalar(2), Scalar(3)};
  std::array<EValue*, values.size()> wrapped_values = {
      &values[0],
      &values[1],
      &values[2],
      &values[3],
      &values[4],
      &values[5],
  };
  // Memory that BoxedEvalueList will use to unpack wrapped_values into an
  // ArrayRef.
  std::array<int64_t, wrapped_values.size()> unwrapped_values;

  {
    BoxedEvalueList<int64_t> list(
        wrapped_values.data(), unwrapped_values.data(), 0);
    EValue value(list);
    expect_output(value, "(len=0)[]");
  }
  {
    BoxedEvalueList<int64_t> list(
        wrapped_values.data(), unwrapped_values.data(), 3);
    EValue value(list);
    expect_output(value, "(len=3)[-2, -1, 0]");
  }
  {
    BoxedEvalueList<int64_t> list(
        wrapped_values.data(), unwrapped_values.data(), 6);
    EValue value(list);
    expect_output(value, "(len=6)[-2, -1, 0, 1, 2, 3]");
  }
}

TEST(PrintEvalueTest, ElidedIntLists) {
  std::array<EValue, 10> values = {
      Scalar(-4),
      Scalar(-3),
      Scalar(-2),
      Scalar(-1),
      Scalar(0),
      Scalar(1),
      Scalar(2),
      Scalar(3),
      Scalar(4),
      Scalar(5),
  };
  std::array<EValue*, values.size()> wrapped_values = {
      &values[0],
      &values[1],
      &values[2],
      &values[3],
      &values[4],
      &values[5],
      &values[6],
      &values[7],
      &values[8],
      &values[9],
  };
  // Memory that BoxedEvalueList will use to unpack wrapped_values into an
  // ArrayRef.
  std::array<int64_t, wrapped_values.size()> unwrapped_values;

  {
    // Default edge items is 3, so the shortest elided list length is 7.
    BoxedEvalueList<int64_t> list(
        wrapped_values.data(), unwrapped_values.data(), 7);
    EValue value(list);
    expect_output(value, "(len=7)[-4, -3, -2, ..., 0, 1, 2]");
  }
  {
    BoxedEvalueList<int64_t> list(
        wrapped_values.data(), unwrapped_values.data(), 8);
    EValue value(list);
    expect_output(value, "(len=8)[-4, -3, -2, ..., 1, 2, 3]");
  }
  {
    // Multi-digit length.
    BoxedEvalueList<int64_t> list(
        wrapped_values.data(), unwrapped_values.data(), 10);
    EValue value(list);
    expect_output(value, "(len=10)[-4, -3, -2, ..., 3, 4, 5]");
  }
}

//
// DoubleList
//

TEST(PrintEvalueTest, UnelidedDoubleLists) {
  // Default edge items is 3, so the longest unelided list length is 6.
  std::array<double, 6> list = {-2.2, -1, 0, INFINITY, NAN, 3.3};

  {
    EValue value(ArrayRef<double>(list.data(), 0ul));
    expect_output(value, "(len=0)[]");
  }
  {
    EValue value(ArrayRef<double>(list.data(), 3));
    expect_output(value, "(len=3)[-2.2, -1., 0.]");
  }
  {
    EValue value(ArrayRef<double>(list.data(), 6));
    expect_output(value, "(len=6)[-2.2, -1., 0., inf, nan, 3.3]");
  }
}

TEST(PrintEvalueTest, ElidedDoubleLists) {
  std::array<double, 10> list = {
      -4.4, -3.0, -2.2, -1, 0, INFINITY, NAN, 3.3, 4.0, 5.5};

  {
    // Default edge items is 3, so the shortest elided list length is 7.
    EValue value(ArrayRef<double>(list.data(), 7));
    expect_output(value, "(len=7)[-4.4, -3., -2.2, ..., 0., inf, nan]");
  }
  {
    EValue value(ArrayRef<double>(list.data(), 8));
    expect_output(value, "(len=8)[-4.4, -3., -2.2, ..., inf, nan, 3.3]");
  }
  {
    // Multi-digit length.
    EValue value(ArrayRef<double>(list.data(), 10));
    expect_output(value, "(len=10)[-4.4, -3., -2.2, ..., 3.3, 4., 5.5]");
  }
}

//
// TensorList
//

void expect_tensor_list_output(size_t num_tensors, const char* expected) {
  TensorFactory<ScalarType::Float> tf;

  std::array<EValue, 10> values = {
      tf.make({2, 2}, {0, 0, 0, 0}),
      tf.make({2, 2}, {1, 1, 1, 1}),
      tf.make({2, 2}, {2, 2, 2, 2}),
      tf.make({2, 2}, {3, 3, 3, 3}),
      tf.make({2, 2}, {4, 4, 4, 4}),
      tf.make({2, 2}, {5, 5, 5, 5}),
      tf.make({2, 2}, {6, 6, 6, 6}),
      tf.make({2, 2}, {7, 7, 7, 7}),
      tf.make({2, 2}, {8, 8, 8, 8}),
      tf.make({2, 2}, {9, 9, 9, 9}),
  };
  std::array<EValue*, values.size()> wrapped_values = {
      &values[0],
      &values[1],
      &values[2],
      &values[3],
      &values[4],
      &values[5],
      &values[6],
      &values[7],
      &values[8],
      &values[9],
  };
  // Memory that BoxedEvalueList will use to assemble a contiguous array of
  // Tensor entries. It's important not to destroy these entries, because the
  // values list will own the underlying Tensors.
  auto unwrapped_values_memory = std::make_unique<uint8_t[]>(
      sizeof(exec_aten::Tensor) * wrapped_values.size());
  exec_aten::Tensor* unwrapped_values =
      reinterpret_cast<exec_aten::Tensor*>(unwrapped_values_memory.get());
#if USE_ATEN_LIB
  // Must be initialized because BoxedEvalueList will use operator=() on each
  // entry. But we can't do this in non-ATen mode because
  // torch::executor::Tensor doesn't have a default constructor.
  for (int i = 0; i < wrapped_values.size(); ++i) {
    new (&unwrapped_values[i]) at::Tensor();
  }
#endif

  ASSERT_LE(num_tensors, wrapped_values.size());
  BoxedEvalueList<exec_aten::Tensor> list(
      wrapped_values.data(), unwrapped_values, num_tensors);
  EValue value(list);
  expect_output(value, expected);
}

TEST(PrintEvalueTest, EmptyTensorListIsOnOneLine) {
  expect_tensor_list_output(0, "(len=0)[]");
}

TEST(PrintEvalueTest, SingleTensorListIsOnOneLine) {
  expect_tensor_list_output(
      1, "(len=1)[tensor(sizes=[2, 2], [0., 0., 0., 0.])]");
}

TEST(PrintEvalueTest, AllTensorListEntriesArePrinted) {
  expect_tensor_list_output(
      10,
      "(len=10)[\n"
      "  [0]: tensor(sizes=[2, 2], [0., 0., 0., 0.]),\n"
      "  [1]: tensor(sizes=[2, 2], [1., 1., 1., 1.]),\n"
      "  [2]: tensor(sizes=[2, 2], [2., 2., 2., 2.]),\n"
      // Inner entries are never elided.
      "  [3]: tensor(sizes=[2, 2], [3., 3., 3., 3.]),\n"
      "  [4]: tensor(sizes=[2, 2], [4., 4., 4., 4.]),\n"
      "  [5]: tensor(sizes=[2, 2], [5., 5., 5., 5.]),\n"
      "  [6]: tensor(sizes=[2, 2], [6., 6., 6., 6.]),\n"
      "  [7]: tensor(sizes=[2, 2], [7., 7., 7., 7.]),\n"
      "  [8]: tensor(sizes=[2, 2], [8., 8., 8., 8.]),\n"
      "  [9]: tensor(sizes=[2, 2], [9., 9., 9., 9.]),\n"
      "]");
}

//
// ListOptionalTensor
//

void expect_list_optional_tensor_output(
    size_t num_tensors,
    const char* expected) {
  TensorFactory<ScalarType::Float> tf;

  std::array<EValue, 5> values = {
      tf.make({2, 2}, {0, 0, 0, 0}),
      tf.make({2, 2}, {2, 2, 2, 2}),
      tf.make({2, 2}, {4, 4, 4, 4}),
      tf.make({2, 2}, {6, 6, 6, 6}),
      tf.make({2, 2}, {8, 8, 8, 8}),
  };
  std::array<EValue*, 10> wrapped_values = {
      nullptr, // None is represented by a nullptr in the wrapped array.
      &values[0],
      nullptr,
      &values[1],
      nullptr,
      &values[2],
      nullptr,
      &values[3],
      nullptr,
      &values[4],
  };
  // Memory that BoxedEvalueList will use to assemble a contiguous array of
  // optional<Tensor> entries. It's important not to destroy these entries,
  // because the values list will own the underlying Tensors.
  auto unwrapped_values_memory = std::make_unique<uint8_t[]>(
      sizeof(exec_aten::optional<exec_aten::Tensor>) * wrapped_values.size());
  exec_aten::optional<exec_aten::Tensor>* unwrapped_values =
      reinterpret_cast<exec_aten::optional<exec_aten::Tensor>*>(
          unwrapped_values_memory.get());
  // Must be initialized because BoxedEvalueList will use operator=() on each
  // entry.
  for (int i = 0; i < wrapped_values.size(); ++i) {
    new (&unwrapped_values[i]) exec_aten::optional<exec_aten::Tensor>();
  }

  ASSERT_LE(num_tensors, wrapped_values.size());
  BoxedEvalueList<exec_aten::optional<exec_aten::Tensor>> list(
      wrapped_values.data(), unwrapped_values, num_tensors);
  EValue value(list);
  expect_output(value, expected);
}

TEST(PrintEvalueTest, EmptyListOptionalTensorIsOnOneLine) {
  expect_list_optional_tensor_output(0, "(len=0)[]");
}

TEST(PrintEvalueTest, SingleListOptionalTensorIsOnOneLine) {
  expect_list_optional_tensor_output(1, "(len=1)[None]");
}

TEST(PrintEvalueTest, AllListOptionalTensorEntriesArePrinted) {
  expect_list_optional_tensor_output(
      10,
      "(len=10)[\n"
      "  [0]: None,\n"
      "  [1]: tensor(sizes=[2, 2], [0., 0., 0., 0.]),\n"
      "  [2]: None,\n"
      // Inner entries are never elided.
      "  [3]: tensor(sizes=[2, 2], [2., 2., 2., 2.]),\n"
      "  [4]: None,\n"
      "  [5]: tensor(sizes=[2, 2], [4., 4., 4., 4.]),\n"
      "  [6]: None,\n"
      "  [7]: tensor(sizes=[2, 2], [6., 6., 6., 6.]),\n"
      "  [8]: None,\n"
      "  [9]: tensor(sizes=[2, 2], [8., 8., 8., 8.]),\n"
      "]");
}

//
// Unknown tag
//

TEST(PrintEvalueTest, UnknownTag) {
  EValue value;
  value.tag = static_cast<torch::executor::Tag>(5555);
  expect_output(value, "<Unknown EValue tag 5555>");
}

//
// evalue_edge_items
//
// Use double as a proxy for testing the edge_items logic; the other scalar
// types use the same underlying code, so they don't need to test this again.
//

TEST(PrintEvalueTest, EdgeItemsOverride) {
  std::array<double, 7> list = {-3.0, -2.2, -1, 0, 3.3, 4.0, 5.5};
  EValue value(ArrayRef<double>(list.data(), 7));

  {
    // Default edge items is 3, so this should elide.
    std::ostringstream os;
    os << value;
    EXPECT_STREQ(
        os.str().c_str(), "(len=7)[-3., -2.2, -1., ..., 3.3, 4., 5.5]");
  }
  {
    // Override to one edge item.
    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(1) << value;
    EXPECT_STREQ(os.str().c_str(), "(len=7)[-3., ..., 5.5]");
  }
  {
    // Override to more edge items than the list contains, removing the elision.
    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(20) << value;
    EXPECT_STREQ(os.str().c_str(), "(len=7)[-3., -2.2, -1., 0., 3.3, 4., 5.5]");
  }
}

TEST(PrintEvalueTest, EdgeItemsDefaults) {
  std::array<double, 7> list = {-3.0, -2.2, -1, 0, 3.3, 4.0, 5.5};
  EValue value(ArrayRef<double>(list.data(), 7));

  {
    // Default edge items is 3, so this should elide.
    std::ostringstream os;
    os << value;
    EXPECT_STREQ(
        os.str().c_str(), "(len=7)[-3., -2.2, -1., ..., 3.3, 4., 5.5]");
  }
  {
    // A value of zero should be the same as the default.
    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(0) << value;
    EXPECT_STREQ(
        os.str().c_str(), "(len=7)[-3., -2.2, -1., ..., 3.3, 4., 5.5]");
  }
  {
    // A negative should be the same as the default.
    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(-5) << value;
    EXPECT_STREQ(
        os.str().c_str(), "(len=7)[-3., -2.2, -1., ..., 3.3, 4., 5.5]");
  }
}

TEST(PrintEvalueTest, EdgeItemsSingleStream) {
  std::array<double, 7> list = {-3.0, -2.2, -1, 0, 3.3, 4.0, 5.5};
  EValue value(ArrayRef<double>(list.data(), 7));
  std::ostringstream os_before;

  // Print to the same stream multiple times, showing that evalue_edge_items
  // can be changed, and is sticky.
  std::ostringstream os;
  os << "default: " << value << "\n";
  os << "      1: " << torch::executor::util::evalue_edge_items(1) << value
     << "\n";
  os << "still 1: " << value << "\n";
  os << "default: " << torch::executor::util::evalue_edge_items(0) << value
     << "\n";
  os << "     20: " << torch::executor::util::evalue_edge_items(20) << value
     << "\n";
  EXPECT_STREQ(
      os.str().c_str(),
      "default: (len=7)[-3., -2.2, -1., ..., 3.3, 4., 5.5]\n"
      "      1: (len=7)[-3., ..., 5.5]\n"
      "still 1: (len=7)[-3., ..., 5.5]\n"
      "default: (len=7)[-3., -2.2, -1., ..., 3.3, 4., 5.5]\n"
      "     20: (len=7)[-3., -2.2, -1., 0., 3.3, 4., 5.5]\n");

  // The final value of 20 does not affect other streams, whether they were
  // created before or after the modified stream.
  os_before << value;
  EXPECT_STREQ(
      os_before.str().c_str(), "(len=7)[-3., -2.2, -1., ..., 3.3, 4., 5.5]");

  std::ostringstream os_after;
  os_after << value;
  EXPECT_STREQ(
      os_after.str().c_str(), "(len=7)[-3., -2.2, -1., ..., 3.3, 4., 5.5]");
}

// Demonstrate the evalue_edge_items affects the data lists inside tensors.
TEST(PrintEvalueTest, EdgeItemsAffectsTensorData) {
  TensorFactory<ScalarType::Double> tf;
  EValue value(tf.make(
      {5, 1, 1, 2}, {1.0, 2.2, 3.3, 4.0, 5.5, 6.6, 7.0, 8.8, 9.0, 10.1}));

  std::ostringstream os;
  os << value << "\n";
  os << torch::executor::util::evalue_edge_items(1) << value << "\n";
  EXPECT_STREQ(
      os.str().c_str(),
      "tensor(sizes=[5, 1, 1, 2], [1., 2.2, 3.3, ..., 8.8, 9., 10.1])\n"
      // Notice that it doesn't affect the sizes list, which is always printed
      // in full.
      "tensor(sizes=[5, 1, 1, 2], [1., ..., 10.1])\n");
}

//
// Long list wrapping.
//
// Use double as a proxy for testing the wrapping logic; the other scalar
// types use the same underlying code, so they don't need to test this again.
//

// Duplicates the internal value in the cpp file under test.
constexpr size_t kItemsPerLine = 10;

TEST(PrintEvalueTest, ListWrapping) {
  // A large list of scalars.
  std::array<double, 100> list;
  for (int i = 0; i < list.size(); ++i) {
    list[i] = static_cast<double>(i);
  }

  {
    // Should elide by default and print on a single line.
    EValue value(ArrayRef<double>(list.data(), list.size()));

    std::ostringstream os;
    os << value;
    EXPECT_STREQ(os.str().c_str(), "(len=100)[0., 1., 2., ..., 97., 98., 99.]");
  }
  {
    // Exactly the per-line length should not wrap when increasing the number of
    // edge items to disable elision.
    EValue value(ArrayRef<double>(list.data(), kItemsPerLine));

    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(1000) << value;
    EXPECT_STREQ(
        os.str().c_str(), "(len=10)[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]");
  }
  {
    // One more than the per-line length should wrap; no elision.
    EValue value(ArrayRef<double>(list.data(), kItemsPerLine + 1));

    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(1000) << value;
    EXPECT_STREQ(
        os.str().c_str(),
        "(len=11)[\n"
        "  0., 1., 2., 3., 4., 5., 6., 7., 8., 9., \n"
        "  10., \n"
        "]");
  }
  {
    // Exactly twice the per-line length, without elision.
    EValue value(ArrayRef<double>(list.data(), kItemsPerLine * 2));

    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(1000) << value;
    EXPECT_STREQ(
        os.str().c_str(),
        "(len=20)[\n"
        "  0., 1., 2., 3., 4., 5., 6., 7., 8., 9., \n"
        "  10., 11., 12., 13., 14., 15., 16., 17., 18., 19., \n"
        // Make sure there is no extra newline here.
        "]");
  }
  {
    // Exactly one whole line, with elision.
    EValue value(ArrayRef<double>(list.data(), kItemsPerLine * 3));

    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(kItemsPerLine) << value;
    EXPECT_STREQ(
        os.str().c_str(),
        "(len=30)[\n"
        "  0., 1., 2., 3., 4., 5., 6., 7., 8., 9., \n"
        // Elision always on its own line when wrapping.
        "  ...,\n"
        "  20., 21., 22., 23., 24., 25., 26., 27., 28., 29., \n"
        "]");
  }
  {
    // Edge item count slightly larger than per-line length, with elision.
    EValue value(ArrayRef<double>(list.data(), kItemsPerLine * 3));

    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(kItemsPerLine + 1) << value;
    EXPECT_STREQ(
        os.str().c_str(),
        "(len=30)[\n"
        "  0., 1., 2., 3., 4., 5., 6., 7., 8., 9., \n"
        "  10., \n"
        // Elision always on its own line when wrapping.
        "  ...,\n"
        // The ragged line always comes just after the elision so that
        // we will end on a full line.
        "  19., \n"
        "  20., 21., 22., 23., 24., 25., 26., 27., 28., 29., \n"
        "]");
  }
  {
    // Large wrapped, ragged, elided example.
    EValue value(ArrayRef<double>(list.data(), list.size()));

    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(33) << value;
    EXPECT_STREQ(
        os.str().c_str(),
        "(len=100)[\n"
        "  0., 1., 2., 3., 4., 5., 6., 7., 8., 9., \n"
        "  10., 11., 12., 13., 14., 15., 16., 17., 18., 19., \n"
        "  20., 21., 22., 23., 24., 25., 26., 27., 28., 29., \n"
        "  30., 31., 32., \n"
        "  ...,\n"
        "  67., 68., 69., \n"
        "  70., 71., 72., 73., 74., 75., 76., 77., 78., 79., \n"
        "  80., 81., 82., 83., 84., 85., 86., 87., 88., 89., \n"
        "  90., 91., 92., 93., 94., 95., 96., 97., 98., 99., \n"
        "]");
  }
}

TEST(PrintEvalueTest, WrappedTensorData) {
  TensorFactory<ScalarType::Double> tf;
  // A tensor with a large number of elements.
  EValue value(tf.ones({10, 10}));

  std::ostringstream os;
  os << torch::executor::util::evalue_edge_items(33) << value;
  EXPECT_STREQ(
      os.str().c_str(),
      "tensor(sizes=[10, 10], [\n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "  1., 1., 1., \n"
      "  ...,\n"
      "  1., 1., 1., \n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "])");
}

TEST(PrintEvalueTest, WrappedTensorSizes) {
  TensorFactory<ScalarType::Double> tf;

  {
    // A tensor with enough dimensions that the sizes list is wrapped, but
    // the data is not.
    std::vector<int32_t> sizes(kItemsPerLine + 1, 1);
    sizes[0] = 5;
    EValue value(tf.ones(sizes));

    std::ostringstream os;
    os << value;
    EXPECT_STREQ(
        os.str().c_str(),
        "tensor(sizes=[\n"
        "  5, 1, 1, 1, 1, 1, 1, 1, 1, 1, \n"
        "  1, \n"
        "], [1., 1., 1., 1., 1.])");
  }
  {
    // Both sizes and data are wrapped.
    std::vector<int32_t> sizes(kItemsPerLine + 1, 1);
    sizes[0] = 100;
    EValue value(tf.ones(sizes));

    std::ostringstream os;
    os << torch::executor::util::evalue_edge_items(15) << value;
    EXPECT_STREQ(
        os.str().c_str(),
        "tensor(sizes=[\n"
        "  100, 1, 1, 1, 1, 1, 1, 1, 1, 1, \n"
        "  1, \n"
        // TODO(T159700776): Indent this further to look more like python.
        "], [\n"
        "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
        "  1., 1., 1., 1., 1., \n"
        "  ...,\n"
        "  1., 1., 1., 1., 1., \n"
        "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
        "])");
  }
}

TEST(PrintEvalueTest, WrappedTensorLists) {
  TensorFactory<ScalarType::Float> tf;

  std::array<EValue, 2> values = {
      // Tensors that are large enough for their data to wrap.
      tf.ones({10, 10}),
      tf.ones({11, 11}),
  };
  std::array<EValue*, values.size()> wrapped_values = {
      &values[0],
      &values[1],
  };
  // Memory that BoxedEvalueList will use to assemble a contiguous array of
  // Tensor entries. It's important not to destroy these entries, because the
  // values list will own the underlying Tensors.
  auto unwrapped_values_memory = std::make_unique<uint8_t[]>(
      sizeof(exec_aten::Tensor) * wrapped_values.size());
  exec_aten::Tensor* unwrapped_values =
      reinterpret_cast<exec_aten::Tensor*>(unwrapped_values_memory.get());
#if USE_ATEN_LIB
  // Must be initialized because BoxedEvalueList will use operator=() on each
  // entry. But we can't do this in non-ATen mode because
  // torch::executor::Tensor doesn't have a default constructor.
  for (int i = 0; i < wrapped_values.size(); ++i) {
    new (&unwrapped_values[i]) at::Tensor();
  }
#endif

  // Demonstrate the formatting when printing a list with multiple tensors.
  BoxedEvalueList<exec_aten::Tensor> list(
      wrapped_values.data(), unwrapped_values, wrapped_values.size());
  EValue value(list);

  std::ostringstream os;
  os << torch::executor::util::evalue_edge_items(15) << value;
  EXPECT_STREQ(
      os.str().c_str(),
      "(len=2)[\n"
      "  [0]: tensor(sizes=[10, 10], [\n"
      // TODO(T159700776): Indent these entries further to look more like
      // python.
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "  1., 1., 1., 1., 1., \n"
      "  ...,\n"
      "  1., 1., 1., 1., 1., \n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "]),\n"
      "  [1]: tensor(sizes=[11, 11], [\n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "  1., 1., 1., 1., 1., \n"
      "  ...,\n"
      "  1., 1., 1., 1., 1., \n"
      "  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., \n"
      "]),\n"
      "]");
}
