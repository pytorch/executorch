// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

Tensor& _where_out(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::where_outf(
      context, condition, self, other, out);
}

//
// Correctness Test
//

template <ScalarType DTYPE_A, ScalarType DTYPE_B, ScalarType DTYPE_OUT>
void test_where() {
  if (DTYPE_OUT == ScalarType::Byte || DTYPE_OUT == ScalarType::Char) {
    return;
  }
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<DTYPE_A> tf_a;
  TensorFactory<DTYPE_B> tf_b;
  TensorFactory<DTYPE_OUT> tf_out;

  const std::vector<int32_t> condition_sizes = {12};
  const std::vector<int32_t> sizes = {1, 12};

  Tensor out = tf_out.zeros(sizes);

  // clang-format off
  _where_out(
      tf_condition.make(condition_sizes, /*data=*/{false, true, false, true, true, false,
                                                   false, true, false, true, true, false}),
      tf_a.make(sizes, /*data=*/{  1,  2,  3,  4,  5,  6,  6,  5,  4,  3,  2,  1}),
      tf_b.make(sizes, /*data=*/{  6,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  6}),
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(
      out,
      tf_out.make(
          sizes, /*data=*/{  6,  2,  4,  4,  5,  1,  1,  5,  3,  3,  2,  6}));
  // clang-format on
}

template <ScalarType DTYPE_A, ScalarType DTYPE_B>
void test_where_enumerate_out_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_where<DTYPE_A, DTYPE_B, ScalarType::dtype>();

  ET_FORALL_FLOAT_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

template <ScalarType DTYPE_A>
void test_where_enumerate_b_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_where<DTYPE_A, ScalarType::dtype, DTYPE_A>();

  ET_FORALL_REAL_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

void test_where_enumerate_a_types() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_where_enumerate_b_types<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

void test_where_enumerate_a_types_aten() {
#define ENUMERATE_TEST_ENTRY(ctype, dtype) \
  test_where<ScalarType::dtype, ScalarType::dtype, ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(ENUMERATE_TEST_ENTRY)

#undef ENUMERATE_TEST_ENTRY
}

/**
 * Uses the functions above to test various input dtype combinations. Note that
 * in Executorch, more dtype combinations are allowed as compared to ATen. The
 * ATen kernel supports different input dtypes by casting both inputs to the
 * same type via type promotion rules, and expects the output to be the same
 * type as the casted inputs. Because of this, testing in ATen mode will only
 * test cases where all tensors are the same type.
 */
TEST(OpWhereOutKernelTest, AllRealDtypesSupported) {
  if (SupportedFeatures::get()->is_aten) {
    test_where_enumerate_a_types_aten();
  } else {
    test_where_enumerate_a_types();
  }
}

// Condition is true, all items will be from x
TEST(OpWhereOutKernelTest, AllTrueTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Float> tf_y;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> condition_sizes = {1};
  const std::vector<int32_t> sizes = {1, 12};

  Tensor out = tf_out.zeros(sizes);

  // clang-format off
  _where_out(
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
TEST(OpWhereOutKernelTest, AllFalseTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Float> tf_y;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> condition_sizes = {1};
  const std::vector<int32_t> sizes = {1, 12};

  // Destination for the where operator.
  Tensor out = tf_out.zeros(sizes);

  // clang-format off
  _where_out(
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
TEST(OpWhereOutKernelTest, MixedTrueFalseTest) {
  TensorFactory<ScalarType::Bool> tf_condition;
  TensorFactory<ScalarType::Float> tf_x;
  TensorFactory<ScalarType::Float> tf_y;
  TensorFactory<ScalarType::Float> tf_out;

  const std::vector<int32_t> condition_sizes = {12};
  const std::vector<int32_t> sizes = {1, 12};

  // Destination for the where operator.
  Tensor out = tf_out.zeros(sizes);

  // clang-format off
  _where_out(
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
TEST(OpWhereOutKernelTest, BroadcastConditionTest) {
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
  _where_out(
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
TEST(OpWhereOutKernelTest, BroadcastConditionAndBroadCastYTest) {
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
  _where_out(
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
TEST(OpWhereOutKernelTest, DoubleTypeTest) {
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
  _where_out(
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
TEST(OpWhereOutKernelTest, MismatchedShapeTest) {
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
  ET_EXPECT_KERNEL_FAILURE(_where_out(
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

  _where_out(condition, input, other, out);
  EXPECT_TENSOR_EQ(out, expected);""" */

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
      {2, 3, 4}, {0.2782524824142456,  0.48195880651474,   0.8197803497314453,
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

  _where_out(condition, input, other, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpWhereOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpWhereOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpWhereOutKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
