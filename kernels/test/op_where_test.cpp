/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpWhereOutTest : public OperatorTest {
 protected:
  Tensor& op_where_self_out(
      const Tensor& condition,
      const Tensor& self,
      const Tensor& other,
      Tensor& out) {
    return torch::executor::aten::where_outf(
        context_, condition, self, other, out);
  }

  template <ScalarType DTYPE_A, ScalarType DTYPE_B, ScalarType DTYPE_OUT>
  void test_where() {
    if (DTYPE_OUT == ScalarType::Byte || DTYPE_OUT == ScalarType::Char) {
      return;
    }
    TensorFactory<ScalarType::Bool> tf_condition;
    TensorFactory<ScalarType::Byte> tf_condition_byte;
    TensorFactory<DTYPE_A> tf_a;
    TensorFactory<DTYPE_B> tf_b;
    TensorFactory<DTYPE_OUT> tf_out;

    const std::vector<int32_t> condition_sizes = {12};
    const std::vector<int32_t> sizes = {1, 12};

    Tensor out = tf_out.zeros(sizes);

    // clang-format off
    std::vector<uint8_t> condition_data = {
      false, true, false, true, true, false,
      false, true, false, true, true, false
    };
    const auto a_tensor = tf_a.make(sizes, /*data=*/{  1,  2,  3,  4,  5,  6,  6,  5,  4,  3,  2,  1});
    const auto b_tensor = tf_b.make(sizes, /*data=*/{  6,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  6});
    // clang-format on
    op_where_self_out(
        tf_condition.make(condition_sizes, /*data=*/condition_data),
        a_tensor,
        b_tensor,
        out);

    auto expectedOut =
        tf_out.make(sizes, /*data=*/{6, 2, 4, 4, 5, 1, 1, 5, 3, 3, 2, 6});
    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(out, expectedOut);

    op_where_self_out(
        tf_condition_byte.make(condition_sizes, condition_data),
        a_tensor,
        b_tensor,
        out);
    EXPECT_TENSOR_CLOSE(out, expectedOut);
  }

  template <ScalarType DTYPE_A, ScalarType DTYPE_B>
  void test_where_enumerate_out_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_where<DTYPE_A, DTYPE_B, ScalarType::dtype>();

    ET_FORALL_REALHBF16_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }

  template <ScalarType DTYPE_A>
  void test_where_enumerate_b_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_where<DTYPE_A, ScalarType::dtype, DTYPE_A>();

    ET_FORALL_REALHBBF16_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(where_template) */

    TensorFactory<ScalarType::Bool> tfBool;
    TensorFactory<ScalarType::Float> tf;

    Tensor condition = tfBool.make(
        {2, 3, 4}, {true,  false, true, true,  true,  false, false, true,
                    false, true,  true, false, false, false, false, false,
                    false, false, true, true,  false, false, true,  true});
    Tensor input = tf.make(
        {2, 3, 4},
        {0.41940832138061523,  0.5529070496559143,   0.9527381062507629,
         0.036164820194244385, 0.1852310299873352,   0.37341737747192383,
         0.3051000237464905,   0.9320003986358643,   0.17591017484664917,
         0.2698335647583008,   0.15067976713180542,  0.03171950578689575,
         0.20812976360321045,  0.9297990202903748,   0.7231091856956482,
         0.7423362731933594,   0.5262957811355591,   0.24365824460983276,
         0.584592342376709,    0.033152639865875244, 0.13871687650680542,
         0.242235004901886,    0.815468966960907,    0.793160617351532});
    Tensor other = tf.make(
        {2, 3, 4},
        {0.2782524824142456,  0.48195880651474,   0.8197803497314453,
         0.9970665574073792,  0.6984410881996155, 0.5675464272499084,
         0.8352431654930115,  0.2055988311767578, 0.593172013759613,
         0.11234724521636963, 0.1534569263458252, 0.24170821905136108,
         0.7262365221977234,  0.7010802030563354, 0.2038237452507019,
         0.6510535478591919,  0.7744860053062439, 0.4368913173675537,
         0.5190907716751099,  0.6158523559570312, 0.8101882934570312,
         0.9800970554351807,  0.1146882176399231, 0.3167651295661926});
    Tensor expected = tf.make(
        {2, 3, 4},
        {0.41940832138061523,  0.48195880651474,     0.9527381062507629,
         0.036164820194244385, 0.1852310299873352,   0.5675464272499084,
         0.8352431654930115,   0.9320003986358643,   0.593172013759613,
         0.2698335647583008,   0.15067976713180542,  0.24170821905136108,
         0.7262365221977234,   0.7010802030563354,   0.2038237452507019,
         0.6510535478591919,   0.7744860053062439,   0.4368913173675537,
         0.584592342376709,    0.033152639865875244, 0.8101882934570312,
         0.9800970554351807,   0.815468966960907,    0.793160617351532});
    Tensor out = tf.zeros(out_shape, dynamism);

    op_where_self_out(condition, input, other, out);
    EXPECT_TENSOR_EQ(out, expected);
  }

  void test_where_enumerate_a_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_where_enumerate_b_types<ScalarType::dtype>();

    ET_FORALL_REALHBBF16_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }

  void test_where_enumerate_a_types_aten() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_where<ScalarType::dtype, ScalarType::dtype, ScalarType::dtype>();

    ET_FORALL_REALHBF16_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
  }
};

//
// Correctness Test
//

TEST_F(OpWhereOutTest, AllRealDtypesSupported) {
  test_where_enumerate_a_types_aten();
}

// Condition is true, all items will be from x
TEST_F(OpWhereOutTest, AllTrueTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Float> tf_y;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> condition_sizes = {1};
  const std::vector<int32_t> sizes = {1, 12};

  Tensor out = tf_out.zeros(sizes);

  // clang-format off
  op_where_self_out(
      tf_condition.make(condition_sizes, /*data=*/{true}),
      tf_x.make(sizes, /*data=*/{ 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 100.0f}),
      tf_y.make(sizes, /*data=*/{ 0.1f, 1.1f,  2.1f,  3.1f, 4.1f,  5.1f,
                                   6.1f, 7.1f, 8.1f, 9.1f, 10.1f, 100.1f}),
      out);

  // Check that it matches (or close to) the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(
          sizes, /*data=*/{ 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 100.0f}));
  // clang-format on
}

// Condition is false, all items will be from y
TEST_F(OpWhereOutTest, AllFalseTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Float> tf_y;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> condition_sizes = {1};
  const std::vector<int32_t> sizes = {1, 12};

  // Destination for the where operator.
  Tensor out = tf_out.zeros(sizes);

  // clang-format off
  op_where_self_out(
      tf_condition.make(condition_sizes, /*data=*/{false}),
      tf_x.make(sizes, /*data=*/{ 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 100.0f}),
      tf_y.make(sizes, /*data=*/{ 0.1f, 1.1f, 2.1f, 3.1f, 4.1f, 5.1f,
                                  6.1f, 7.1f, 8.1f, 9.1f, 10.1f, 100.1f}),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(
          sizes, /*data=*/{ 0.1f, 1.1f, 2.1f, 3.1f, 4.1f, 5.1f,
                            6.1f, 7.1f, 8.1f, 9.1f, 10.1f, 100.1f}));
  // clang-format on
}

// Choosing based on condition[i] ? x[i] : y[i]
TEST_F(OpWhereOutTest, MixedTrueFalseTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Float> tf_y;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> condition_sizes = {12};
  const std::vector<int32_t> sizes = {1, 12};

  // Destination for the where operator.
  Tensor out = tf_out.zeros(sizes);

  // clang-format off
  op_where_self_out(
      tf_condition.make(condition_sizes, /*data=*/{false, true, false ,true, true, false,
                                                    false, true, false ,true, true, false}),
      tf_x.make(sizes, /*data=*/{ 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 100.0f}),
      tf_y.make(sizes, /*data=*/{ 0.1f, 1.1f,  2.1f,  3.1f, 4.1f,  5.1f,
                                  6.1f, 7.1f, 8.1f, 9.1f, 10.1f, 100.1f}),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(
          sizes, /*data=*/{ 0.1f, 1.0f, 2.1f, 3.0f, 4.0f, 5.1f,
                            6.1f, 7.0f, 8.1f, 9.0f, 10.0f, 100.1f}));
  // clang-format on
}

// Choosing based on condition[i] ? x[i] : y[i]
TEST_F(OpWhereOutTest, BroadcastConditionTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Float> tf_y;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> condition_sizes = {3, 1};
  const std::vector<int32_t> x_sizes = {3, 4};
  const std::vector<int32_t> y_sizes = {3, 4};

  // Destination for the where operator.
  Tensor out = tf_out.zeros(x_sizes);

  // clang-format off
  op_where_self_out(
      tf_condition.make(condition_sizes, /*data=*/{
                                  false,
                                  true,
                                  false}),
      tf_x.make(x_sizes, /*data=*/{
                                  0.0f, 1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f, 7.0f,
                                  8.0f,  9.0f, 10.0f, 100.0f}),
      tf_y.make(y_sizes, /*data=*/
                                  {0.1f, 1.1f, 2.1f, 3.1f,
                                  4.1f,  5.1f, 6.1f, 7.1f,
                                  8.1f,  9.1f, 10.1f, 100.1f}),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(
          x_sizes, /*data=*/{ 0.1f, 1.1f, 2.1f, 3.1f,
                              4.0f, 5.0f, 6.0f, 7.0f,
                              8.1f,  9.1f, 10.1f, 100.1f}));
  // clang-format on
}

// Choosing based on condition[i] ? x[i] : y[i]
TEST_F(OpWhereOutTest, BroadcastConditionAndBroadCastYTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Float> tf_y;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> condition_sizes = {3, 1};
  const std::vector<int32_t> x_sizes = {3, 4};
  const std::vector<int32_t> y_sizes = {3, 1};

  // Destination for the where operator.
  Tensor out = tf_out.zeros(x_sizes);

  // clang-format off
  op_where_self_out(
      tf_condition.make(condition_sizes, /*data=*/{
                                  false,
                                  true,
                                  false}),
      tf_x.make(x_sizes, /*data=*/{
                                  0.0f, 1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f, 7.0f,
                                  8.0f,  9.0f, 10.0f, 100.0f}),
      tf_y.make(y_sizes, /*data=*/{
                                  0.1f,
                                  4.1f,
                                  8.1f}),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(
          x_sizes, /*data=*/{
                          0.1f, 0.1f, 0.1f, 0.1f,
                          4.0f, 5.0f, 6.0f, 7.0f,
                          8.1f, 8.1f, 8.1f, 8.1f}));
  // clang-format on
}

// Choosing based on condition[i] ? x[i] : y[i]
TEST_F(OpWhereOutTest, DoubleTypeTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Double> tf_x;
  TensorFactory<ScalarType::Double> tf_y;
  TensorFactory<ScalarType::Double> tf_out;

  const std::vector<int32_t> condition_sizes = {3, 1};
  const std::vector<int32_t> x_sizes = {3, 4};
  const std::vector<int32_t> y_sizes = {3, 1};

  // Destination for the where operator.
  Tensor out = tf_out.zeros(x_sizes);

  // clang-format off
  op_where_self_out(
      tf_condition.make(condition_sizes, /*data=*/{
                                  false,
                                  true,
                                  false}),
      tf_x.make(x_sizes, /*data=*/{
                                  0.0, 1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0, 7.0,
                                  8.0, 9.0, 10.0, 100.0}),
      tf_y.make(y_sizes, /*data=*/{
                                  0.1,
                                  4.1,
                                  8.1}),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(
          x_sizes, /*data=*/{
                          0.1, 0.1, 0.1, 0.1,
                          4.0, 5.0, 6.0, 7.0,
                          8.1, 8.1, 8.1, 8.1}));
  // clang-format on
}

// Choosing based on condition[i] ? x[i] : y[i]
TEST_F(OpWhereOutTest, MismatchedShapeTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Double> tf_y;
  TensorFactory<ScalarType::Double> tf_out;

  const std::vector<int32_t> condition_sizes = {3, 1};
  const std::vector<int32_t> x_sizes = {3, 4};
  const std::vector<int32_t> y_sizes = {4, 1};

  // Destination for the where operator.
  Tensor out = tf_out.zeros(x_sizes);

  // clang-format off
  ET_EXPECT_KERNEL_FAILURE(context_, op_where_self_out(
      tf_condition.make(condition_sizes, /*data=*/{
                                  false,
                                  true,
                                  false}),
      tf_x.make(x_sizes, /*data=*/{
                                  0.0f, 1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f, 7.0f,
                                  8.0f,  9.0f, 10.0f, 100.0f}),
      tf_y.make(y_sizes, /*data=*/{
                                  0.1,
                                  4.1,
                                  8.1,
                                  11.1}),
      out));
  // clang-format on
}

/* %python
import torch
torch.manual_seed(0)
input_shape = (2, 3, 4)
condition = torch.randint(10, input_shape) < 5
input = torch.rand(input_shape)
other = torch.rand(input_shape)
expected = torch.where(condition, input, other)

where_template = f"""
  {declare_tensor_factory("ScalarType::Bool", "tfBool")}
  {declare_tensor_factory("ScalarType::Float", "tf")}

  {declare_tensor_make_t("condition", "tfBool")}
  {declare_tensor_make_t("input", "tf")}
  {declare_tensor_make_t("other", "tf")}
  {declare_tensor_make_t("expected", "tf")}
  {declare_tensor_zeros("out_shape, dynamism", "tf", "out")}

  op_where_self_out(condition, input, other, out);
  EXPECT_TENSOR_EQ(out, expected);""" */

TEST_F(OpWhereOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpWhereOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpWhereOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}

TEST_F(OpWhereOutTest, HalfSupport) {
  TensorFactory<ScalarType::Bool> tb;
  TensorFactory<ScalarType::Half> tf;
  Tensor cond = tb.make({2, 3}, {true, false, true, false, true, false});
  Tensor a = tf.full({2, 3}, 1.5);
  Tensor b = tf.full({2, 3}, 2.5);
  Tensor out = tf.zeros({2, 3});

  op_where_self_out(cond, a, b, out);
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 3}, {1.5, 2.5, 1.5, 2.5, 1.5, 2.5}));
}
