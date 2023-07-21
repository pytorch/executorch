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
using torch::executor::testing::TensorFactory;

Tensor& _minimum_out(const Tensor& self, const Tensor& other, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::minimum_outf(context, self, other, out);
}

// Common testing for minimum operator
template <ScalarType DTYPE>
void test_minimum_out_same_size() {
  TensorFactory<DTYPE> tf;
  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the minimum operator.
  Tensor out = tf.zeros(sizes);

  _minimum_out(
      tf.make(sizes, /*data=*/{1, 2, 4, 8}),
      tf.make(sizes, /*data=*/{3, 0, 4, 9}),
      out);

  // Check that it matches to the expected output.
  EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{1, 0, 4, 8}));
}

TEST(OpMinimumOutKernelTest, ByteTensors) {
  test_minimum_out_same_size<ScalarType::Byte>();
}

TEST(OpMinimumOutKernelTest, CharTensors) {
  test_minimum_out_same_size<ScalarType::Char>();
}

TEST(OpMinimumOutKernelTest, ShortTensors) {
  test_minimum_out_same_size<ScalarType::Short>();
}

TEST(OpMinimumOutKernelTest, IntTensors) {
  test_minimum_out_same_size<ScalarType::Int>();
}

TEST(OpMinimumOutKernelTest, LongTensors) {
  test_minimum_out_same_size<ScalarType::Long>();
}

TEST(OpMinimumOutKernelTest, FloatTensors) {
  test_minimum_out_same_size<ScalarType::Float>();
}

TEST(OpMinimumOutKernelTest, DoubleTensors) {
  test_minimum_out_same_size<ScalarType::Double>();
}

TEST(OpMinimumOutKernelTest, BothScalarTensors) {
  // Checks the case when both cases are scalar.
  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> sizes = {1, 1};
  Tensor out = tf.zeros(sizes);
  _minimum_out(tf.make(sizes, {1.2}), tf.make(sizes, {3.5}), out);
  EXPECT_TENSOR_EQ(out, tf.make(sizes, {1.2}));
}

TEST(OpMinimumOutKernelTest, LeftScalarTensor) {
  // Checks the case where one of the tensor is a singleton tensor.

  TensorFactory<ScalarType::Float> tf;
  const std::vector<int32_t> sizes_1 = {1, 1};
  const std::vector<int32_t> sizes_2 = {2, 2};
  Tensor out1 = tf.zeros(sizes_2);
  Tensor out2 = tf.zeros(sizes_2);

  auto a = tf.make(sizes_1, /*data=*/{1.0});
  auto b = tf.make(sizes_2, /*data=*/{3.5, -1.0, 0.0, 5.5});

  // Case 1 : First argument is singleton.
  _minimum_out(a, b, out1);
  EXPECT_TENSOR_EQ(out1, tf.make(sizes_2, {1.0, -1.0, 0.0, 1.0}));

  // Case 2: Second argument is singleton
  _minimum_out(b, a, out2);
  EXPECT_TENSOR_EQ(out2, tf.make(sizes_2, {1.0, -1.0, 0.0, 1.0}));
}

TEST(OpMinimumOutKernelTest, UnhandledDtypeDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle bool dtype";
  }
  TensorFactory<ScalarType::Bool> tfb;

  Tensor outb = tfb.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(_minimum_out(
      tfb.make({2, 2}, /*data=*/{true, true, false, false}),
      tfb.make({2, 2}, /*data=*/{true, true, false, false}),
      outb));
}

TEST(OpMinimumOutKernelTest, MixedDtypesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched input dtypes";
  }
  // Mismatched type between first and second input argument

  TensorFactory<ScalarType::Int> tfi;
  TensorFactory<ScalarType::Float> tff;
  const std::vector<int32_t> sizes = {2, 2};
  Tensor outf = tff.zeros(sizes);
  ET_EXPECT_KERNEL_FAILURE(
      _minimum_out(tfi.ones(sizes), tff.ones(sizes), outf));
}

TEST(OpMinimumOutKernelTest, UnmatchedOutputDtypesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output dtype";
  }
  // Same type first and second input argument but mismatched
  // with the output argument.
  TensorFactory<ScalarType::Int> tfi;
  TensorFactory<ScalarType::Float> tff;

  const std::vector<int32_t> sizes = {2, 2};
  Tensor outf = tff.zeros(sizes);
  ET_EXPECT_KERNEL_FAILURE(
      _minimum_out(tfi.ones(sizes), tfi.ones(sizes), outf));
}

TEST(OpMinimumOutKernelTest, MismatchedInputShapesDies) {
  // First and second argument have different shape
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({2, 2});

  ET_EXPECT_KERNEL_FAILURE(_minimum_out(tf.ones({2, 2}), tf.ones({3, 3}), out));
}

TEST(OpMinimumOutKernelTest, MismatchedOutputShapesDies) {
  // First and second argument have same shape, but output has different shape.
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({3, 3});

  ET_EXPECT_KERNEL_FAILURE(_minimum_out(tf.ones({2, 2}), tf.ones({3, 3}), out));
}

TEST(OpMinimumOutKernelTest, MismatchedOutputShapeWithSingletonDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }
  // First argument is singleton but second and output has different shape.
  TensorFactory<ScalarType::Float> tf;
  Tensor out = tf.zeros({4, 4});

  ET_EXPECT_KERNEL_FAILURE(_minimum_out(tf.ones({1, 1}), tf.ones({3, 3}), out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(3, 2)
y = torch.rand(3, 2)
res = torch.minimum(x, y)
op = "_minimum_out"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST(OpMinimumOutKernelTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  _minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpMinimumOutKernelTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  _minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpMinimumOutKernelTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(binary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor y = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.8964447379112244,
       0.455627977848053,
       0.6323062777519226,
       0.3488934636116028,
       0.40171730518341064});
  Tensor expected = tf.make(
      {3, 2},
      {0.4900934100151062,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.40171730518341064});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  _minimum_out(x, y, out);
  EXPECT_TENSOR_EQ(out, expected);
}
