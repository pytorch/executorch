// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

Tensor& rsub_out(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::rsub_outf(context, self, other, alpha, out);
}

// Common testing for substraction of scalar for integer Tensor.
template <ScalarType DTYPE>
void test_integer_rsub_scalar_out() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the rsub.
  Tensor out = tf.zeros(sizes);

  // Performs substraction of tensor from scalar.
  rsub_out(
      tf.make(sizes, /*data=*/{1, 2, 4, 5}),
      10,
      /*alpha=*/2,
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{8, 6, 2, 0}));
}

TEST(OpRSubScalarOutKernelTest, ByteTensors) {
  test_integer_rsub_scalar_out<ScalarType::Byte>();
}

TEST(OpSubOutKernelTest, CharTensors) {
  test_integer_rsub_scalar_out<ScalarType::Char>();
}

TEST(OpSubOutKernelTest, ShortTensors) {
  test_integer_rsub_scalar_out<ScalarType::Short>();
}

TEST(OpSubOutKernelTest, IntTensors) {
  test_integer_rsub_scalar_out<ScalarType::Int>();
}

TEST(OpSubOutKernelTest, LongTensors) {
  test_integer_rsub_scalar_out<ScalarType::Long>();
}

TEST(OpRSubScalarOutKernelTest, IntTensorFloatAlphaDies) {
  // rsub_scalar_out() doesn't handle floating alpha for intergal inputs
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the op.
  Tensor out = tf.zeros(sizes);

  // Subtraction operation on integral tensor with floating alpha
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(rsub_out(tf.ones(sizes), 0, /*alpha=*/.7, out));
}

// Common testing for substraction between floating point tensor and scalar.
template <ScalarType DTYPE>
void test_floating_point_rsub_scalar_out() {
  TensorFactory<DTYPE> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the rsub.
  Tensor out = tf.zeros(sizes);

  // Performs substraction of tensor from scalar.
  rsub_out(
      tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}),
      1.1,
      /*alpha=*/1,
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{0.0, -1.1, -3.3, -7.7}));
}

TEST(OpRSubScalarOutKernelTest, FloatTensors) {
  test_floating_point_rsub_scalar_out<ScalarType::Float>();
}

TEST(OpRSubScalarOutKernelTest, DoubleTensors) {
  test_floating_point_rsub_scalar_out<ScalarType::Double>();
}

TEST(OpRSubScalarOutKernelTest, UnhandledDtypeDies) {
  // rsub_scalar_out() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Subtrahend
  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});

  // Destination for the subtraction.
  Tensor out = tf.zeros(sizes);

  // Subtraction operation on boolean tensor should cause an assertion and
  // kill the test process.
  ET_EXPECT_KERNEL_FAILURE(rsub_out(a, false, /*alpha=*/0, out));
}

// The input arguments may not have different dtypes
TEST(OpRSubScalarOutKernelTest, MismatchedInputDtypesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched input dtypes";
  }

  // Two different dtypes: int8 tensor and Integral scalar
  // that can not be represented as int8.
  TensorFactory<ScalarType::Char> tf_byte;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor t_byte = tf_byte.ones(sizes);

  // Destination for the rsub; matches the type of the input.
  Tensor out = tf_byte.zeros(sizes);

  // Performing reverse substraction over scalar and tensor of different type
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(rsub_out(t_byte, 128, /*alpha=*/0, out));
}

// The output tensor may not have a dtype different from the input even if it
// has the same shape.
TEST(OpRSubOutKernelTest, MismatchedOutputDtypeDies) {
  // Two different dtypes. This test uses two types with the same size to
  // demonstrate that the ScalarType itself matters, not the size of the
  // tensor elements.
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;

  const std::vector<int32_t> sizes = {2, 2};

  // Minuend and subtrahend of the same dtype.
  Tensor a = tf_byte.ones(sizes);

  // Destination with a dtype different from the inputs.
  Tensor out = tf_char.zeros(sizes);

  // Performing substraction of scalar from tesnor and write into a mismatched
  // output should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(rsub_out(a, 1, /*alpha=*/0, out));
}

// Mismatched shape tests.

TEST(OpRSubScalarOutKernelTest, MismatchedOutputShapesDies) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle output shapes";
  }

  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.ones(sizes);

  // Destination with a different shape.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Performing substraction of scalar from tensor into a mismatched output
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(rsub_out(a, 1, /*alpha=*/0, out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(2, 3)
other = 10
alpha = 2
res = other - alpha * x
op = "rsub_out"
opt_setup_params = f"""
  Scalar other = {other};
  Scalar alpha = {alpha};
"""
opt_extra_params = "other, alpha,"
out_args = "out_shape, dynamism"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_CLOSE" */

void test_dynamic_shape(
    const std::vector<int32_t>& out_shape,
    enum torch::executor::TensorShapeDynamism dynamism) {
  /* %python
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 3},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor expected = tf.make(
      {2, 3},
      {9.007486343383789,
       8.463556289672852,
       9.823044776916504,
       9.735939025878906,
       9.385154724121094,
       8.731842994689941});

  Scalar other = 10;
  Scalar alpha = 2;

  Tensor out = tf.zeros(out_shape, dynamism);
  rsub_out(x, other, alpha, out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST(OpRSubScalarOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpRSubScalarOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpRSubScalarOutKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
