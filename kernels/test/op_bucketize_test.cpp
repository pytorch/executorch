#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

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

  template <ScalarType dtype>
  void run_smoke_test_int64() {
    TensorFactory<ScalarType::Long> tf_out;
    TensorFactory<dtype> tf_dtype;

    Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
    Tensor boundaries = tf_dtype.make({5}, {0, 3, 5, 7, 9});
    Tensor expected = tf_out.make({2, 2}, {1, 2, 3, 4});
    Tensor out = tf_out.zeros({2, 2});

    Tensor ret = op_bucketize_out(values, boundaries, false, true, out);

    EXPECT_TENSOR_EQ(ret, expected);
    EXPECT_TENSOR_EQ(out, expected);
  }

  template <ScalarType dtype>
  void run_smoke_test_int32() {
    TensorFactory<ScalarType::Int> tf_out;
    TensorFactory<dtype> tf_dtype;

    Tensor values = tf_dtype.make({2, 2}, {1, 4, 6, 8});
    Tensor boundaries = tf_dtype.make({5}, {0, 3, 5, 7, 9});
    Tensor expected = tf_out.make({2, 2}, {1, 2, 3, 4});
    Tensor out = tf_out.zeros({2, 2});

    Tensor ret = op_bucketize_out(values, boundaries, true, true, out);

    EXPECT_TENSOR_EQ(ret, expected);
    EXPECT_TENSOR_EQ(out, expected);
  }

  template <ScalarType dtype>
  void run_smoke_test_non_int_out() {
    TensorFactory<dtype> tf_out;
    TensorFactory<ScalarType::Float> tf_dtype;

    Tensor values = tf_dtype.make({2, 2}, {1.5, 2.5, 3.5, 4.5});
    Tensor boundaries = tf_dtype.make({5}, {1, 2, 3, 4, 5});
    Tensor expected = tf_dtype.make({2, 2}, {1, 2, 3, 4});
    Tensor out = tf_out.zeros({2, 2});

    ET_EXPECT_KERNEL_FAILURE(
        context_, op_bucketize_out(values, boundaries, true, false, out));
  }
};

TEST_F(OpBucketizeTest, SmokeTestInt64) {
#define RUN_SMOKE_TEST(ctype, dtype) run_smoke_test_int64<ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(RUN_SMOKE_TEST);
#undef RUN_SMOKE_TEST
}

TEST_F(OpBucketizeTest, SmokeTestInt32) {
#define RUN_SMOKE_TEST(ctype, dtype) run_smoke_test_int32<ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(RUN_SMOKE_TEST);
#undef RUN_SMOKE_TEST
}

TEST_F(OpBucketizeTest, RightTest) {
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

TEST_F(OpBucketizeTest, LeftTest) {
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

TEST_F(OpBucketizeTest, OutOfBoundaryTest) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {-1, -2, 30, 40});
  Tensor boundaries = tf_dtype.make({5}, {1, 2, 3, 4, 5});
  Tensor expected = tf_out.make({2, 2}, {0, 0, 5, 5});
  Tensor out = tf_out.zeros({2, 2});

  Tensor ret = op_bucketize_out(values, boundaries, false, false, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, Boundaries1DTest) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {-1, -2, 30, 40});
  Tensor boundaries = tf_dtype.make({5}, {1, 2, 3, 4, 5});
  Tensor expected = tf_out.make({2, 2}, {0, 0, 5, 5});
  Tensor out = tf_out.zeros({2, 2});

  Tensor ret = op_bucketize_out(values, boundaries, false, false, out);

  EXPECT_TENSOR_EQ(ret, expected);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpBucketizeTest, BoundariesNDimTest) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {-1, -2, 30, 40});
  Tensor boundaries = tf_dtype.make({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor out = tf_out.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_bucketize_out(values, boundaries, false, false, out));
}

TEST_F(OpBucketizeTest, MismatchingInOutTest) {
  TensorFactory<ScalarType::Long> tf_out;
  TensorFactory<ScalarType::Float> tf_dtype;

  Tensor values = tf_dtype.make({2, 2}, {-1, -2, 30, 40});
  Tensor boundaries = tf_dtype.make({5}, {1, 2, 3, 4, 5});
  Tensor out = tf_out.zeros({2, 3});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_bucketize_out(values, boundaries, false, false, out));
}

TEST_F(OpBucketizeTest, NonIntOutTest) {
#define RUN_SMOKE_TEST(ctype, dtype) \
  run_smoke_test_non_int_out<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(RUN_SMOKE_TEST);
#undef RUN_SMOKE_TEST
}