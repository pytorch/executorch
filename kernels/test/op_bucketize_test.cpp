#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::Scalar;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpBucketizeScalarTest : public OperatorTest {
 protected:
  Tensor& op_bucketize_out(
      const Scalar& self,
      const Tensor& boundaries,
      bool out_int32,
      bool right,
      Tensor& out) {
    return torch::executor::aten::bucketize_outf(
        context_, self, boundaries, out_int32, right, out);
  }

  template <ScalarType BOUND_DTYPE>
  void test_bucketize_types() {
    TensorFactory<ScalarType::Long> tf_out;
    TensorFactory<BOUND_DTYPE> tf_bound;

    Scalar value = 2;
    Tensor boundaries = tf_bound.make({5}, {0, 3, 5, 7, 9});
    Tensor expected = tf_out.make({}, {1});
    Tensor out = tf_out.zeros({});

    Tensor ret = op_bucketize_out(value, boundaries, false, true, out);

    EXPECT_TENSOR_EQ(ret, expected);
    EXPECT_TENSOR_EQ(out, expected);
  }

  void test_bucketize_bound_types() {
#define RUN_TEST(ctype, dtype) test_bucketize_types<ScalarType::dtype>();
    ET_FORALL_REALHBF16_TYPES(RUN_TEST)
#undef RUN_TEST
  }
};

TEST_F(OpBucketizeScalarTest, SanityCheck) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_bound;

  Scalar value = 2.5;
  Tensor boundaries = tf_bound.make({5}, {0, 2, 4, 6, 8});
  Tensor expected = tf_out.make({}, {2});
  Tensor out = tf_out.zeros({});

  Tensor ret = op_bucketize_out(value, boundaries, false, true, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeScalarTest, ScalarEmptyBoundaries) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_bound;

  Scalar value = 2.5;
  Tensor boundaries = tf_bound.make({0}, {});
  Tensor expected = tf_out.make({}, {0});
  Tensor out = tf_out.zeros({});

  Tensor ret = op_bucketize_out(value, boundaries, false, true, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeScalarTest, ScalarBoundaryTypes) {
  test_bucketize_bound_types();
}

TEST_F(OpBucketizeScalarTest, ScalarOut1DFails) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_bound;

  Scalar value = 2;
  Tensor boundaries = tf_bound.make({5}, {0, 3, 5, 7, 9});
  Tensor out = tf_out.zeros({5});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_bucketize_out(value, boundaries, false, true, out));
}

TEST_F(OpBucketizeScalarTest, ScalarOutNDFails) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_bound;

  Scalar value = 2;
  Tensor boundaries = tf_bound.make({5}, {0, 3, 5, 7, 9});
  Tensor out = tf_out.zeros({5, 5});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_bucketize_out(value, boundaries, false, true, out));
}

class OpBucketizeTest : public OperatorTest {
 protected:
  Tensor& op_bucketize_out(
      const Tensor& in,
      const Tensor& boundaries,
      bool out_int32,
      bool right,
      Tensor& out) {
    return torch::executor::aten::bucketize_outf(
        context_, in, boundaries, out_int32, right, out);
  }

  template <ScalarType IN_DTYPE, ScalarType BOUND_DTYPE>
  void test_bucketize_types() {
    TensorFactory<ScalarType::Long> tf_out;
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<BOUND_DTYPE> tf_bound;

    Tensor values = tf_in.make({2, 2}, {1, 4, 6, 8});
    Tensor boundaries = tf_bound.make({5}, {0, 3, 5, 7, 9});
    Tensor expected = tf_out.make({2, 2}, {1, 2, 3, 4});
    Tensor out = tf_out.zeros({2, 2});

    Tensor ret = op_bucketize_out(values, boundaries, false, true, out);

    EXPECT_TENSOR_EQ(ret, expected);
    EXPECT_TENSOR_EQ(out, expected);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_bucketize_complex_boundary() {
    TensorFactory<ScalarType::Long> tf_out;
    TensorFactory<ScalarType::Float> tf_in;
    TensorFactory<DTYPE> tf_bound;

    Tensor values = tf_in.make({2, 2}, {1, 4, 6, 8});
    Tensor boundaries = tf_bound.make({1}, {CTYPE(0, 1)});
    Tensor out = tf_out.zeros({2, 2});

    ET_EXPECT_KERNEL_FAILURE(
        context_, op_bucketize_out(values, boundaries, false, false, out));
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_bucketize_complex_input() {
    TensorFactory<ScalarType::Long> tf_out;
    TensorFactory<DTYPE> tf_in;
    TensorFactory<ScalarType::Float> tf_bound;

    Tensor values = tf_in.make({1}, {CTYPE(0, 1)});
    Tensor boundaries = tf_bound.make({5}, {0, 3, 5, 7, 9});
    Tensor out = tf_out.zeros({2, 2});

    ET_EXPECT_KERNEL_FAILURE(
        context_, op_bucketize_out(values, boundaries, false, false, out));
  }

  template <ScalarType IN_DTYPE>
  void test_bucketize_bound_types() {
#define RUN_TEST(ctype, dtype) \
  test_bucketize_types<IN_DTYPE, ScalarType::dtype>();
    ET_FORALL_REALHBF16_TYPES(RUN_TEST)
#undef RUN_TEST
  }

  void test_bucketize_in_types() {
#define RUN_TEST(ctype, dtype) test_bucketize_bound_types<ScalarType::dtype>();
    ET_FORALL_REALHBF16_TYPES(RUN_TEST)
#undef RUN_TEST
  }
};

TEST_F(OpBucketizeTest, SanityCheck) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_comp;

  Tensor values = tf_comp.make(
      {2, 4, 4}, {0, 4, 6, 8, 1, 4, 5, 8, 1,  5, 6, 8, -1, 4, 6, 9,

                  1, 4, 6, 8, 1, 4, 7, 8, -2, 4, 6, 8, 1,  4, 6, 8});

  Tensor boundaries = tf_comp.make({5}, {0, 3, 5, 7, 9});

  Tensor expected =
      tf_out.make({2, 4, 4}, {1, 2, 3, 4, 1, 2, 3, 4, 1, 3, 3, 4, 0, 2, 3, 5,

                              1, 2, 3, 4, 1, 2, 4, 4, 0, 2, 3, 4, 1, 2, 3, 4});

  Tensor out = tf_out.zeros({2, 4, 4});

  // The execution of the operator
  Tensor ret = op_bucketize_out(values, boundaries, false, true, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, InAndBoundaryTypes) {
  test_bucketize_in_types();
}

TEST_F(OpBucketizeTest, Int64Out) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
  Tensor boundaries = tf_dtype.make({5}, {0, 3, 5, 7, 9});
  Tensor expected = tf_out.make({2, 2}, {1, 2, 3, 4});
  Tensor out = tf_out.zeros({2, 2});

  Tensor ret = op_bucketize_out(values, boundaries, false, true, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, Int32Out) {
  TensorFactory<ScalarType::Int> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
  Tensor boundaries = tf_dtype.make({5}, {0, 3, 5, 7, 9});
  Tensor expected = tf_out.make({2, 2}, {1, 2, 3, 4});
  Tensor out = tf_out.zeros({2, 2});

  Tensor ret = op_bucketize_out(values, boundaries, true, true, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, BoundariesRight) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 2, 3, 4});
  Tensor boundaries = tf_dtype.make({5}, {1, 2, 3, 4, 5});
  Tensor expected = tf_out.make({2, 2}, {1, 2, 3, 4});
  Tensor out = tf_out.zeros({2, 2});

  Tensor ret = op_bucketize_out(values, boundaries, false, true, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, BoundariesLeft) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 2, 3, 4});
  Tensor boundaries = tf_dtype.make({5}, {1, 2, 3, 4, 5});
  Tensor expected = tf_out.make({2, 2}, {0, 1, 2, 3});
  Tensor out = tf_out.zeros({2, 2});

  Tensor ret = op_bucketize_out(values, boundaries, false, false, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, OutOfBoundary) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {-1, -2, 6, 40});
  Tensor boundaries = tf_dtype.make({5}, {1, 2, 3, 4, 5});
  Tensor expected = tf_out.make({2, 2}, {0, 0, 5, 5});
  Tensor out = tf_out.zeros({2, 2});

  Tensor ret = op_bucketize_out(values, boundaries, false, false, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, EmptyBoundaries) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
  Tensor boundaries = tf_dtype.make({0}, {});
  Tensor expected = tf_out.make({2, 2}, {0, 0, 0, 0});
  Tensor out = tf_out.zeros({2, 2});

  Tensor ret = op_bucketize_out(values, boundaries, false, false, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, EmptyInput) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({0}, {});
  Tensor boundaries = tf_dtype.make({5}, {1, 2, 3, 4, 5});
  Tensor expected = tf_out.make({0}, {});
  Tensor out = tf_out.zeros({0});

  Tensor ret = op_bucketize_out(values, boundaries, false, false, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, EmptyAll) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({0}, {});
  Tensor boundaries = tf_dtype.make({0}, {});
  Tensor expected = tf_out.make({0}, {});
  Tensor out = tf_out.zeros({0});

  Tensor ret = op_bucketize_out(values, boundaries, false, false, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, BoundariesNDFails) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
  Tensor boundaries = tf_dtype.make({3, 2}, {0, 3, 5, 7, 9, 11});
  Tensor out = tf_out.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_bucketize_out(values, boundaries, false, false, out));
}

TEST_F(OpBucketizeTest, MismatchingInOutDimsFails) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
  Tensor boundaries = tf_dtype.make({5}, {0, 3, 5, 7, 9});
  Tensor out = tf_out.zeros({2, 3});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_bucketize_out(values, boundaries, false, false, out));
}

TEST_F(OpBucketizeTest, MismatchingIntArg32Fails) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
  Tensor boundaries = tf_dtype.make({5}, {0, 3, 5, 7, 9});
  Tensor out = tf_out.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_bucketize_out(values, boundaries, true, false, out));
}

TEST_F(OpBucketizeTest, MismatchingIntArg64Fails) {
  TensorFactory<ScalarType::Int> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
  Tensor boundaries = tf_dtype.make({5}, {0, 3, 5, 7, 9});
  Tensor out = tf_out.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_bucketize_out(values, boundaries, false, false, out));
}

TEST_F(OpBucketizeTest, ComplexBoundaryTypesFails) {
#define RUN_TEST(ctype, dtype) \
  test_bucketize_complex_boundary<ctype, ScalarType::dtype>();
  ET_FORALL_COMPLEXH_TYPES(RUN_TEST)
#undef RUN_TEST
}

TEST_F(OpBucketizeTest, ComplexInputTypesFails) {
#define RUN_TEST(ctype, dtype) \
  test_bucketize_complex_input<ctype, ScalarType::dtype>();
  ET_FORALL_COMPLEXH_TYPES(RUN_TEST)
#undef RUN_TEST
}