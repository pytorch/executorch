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
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpMaskedFillTest : public OperatorTest {
 protected:
  Tensor& op_masked_fill_scalar_out(
      const Tensor& self,
      const Tensor& mask,
      const Scalar& value,
      Tensor& out) {
    return torch::executor::aten::masked_fill_outf(
        context_, self, mask, value, out);
  }

  // Common testing for masked fill of integer Tensor.
  template <ScalarType DTYPE>
  void test_integer_masked_fill_scalar_out() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Bool> tf_bool;

    const std::vector<int32_t> sizes = {2, 2};

    // Destination for the masked_fill.
    Tensor out = tf.zeros(sizes);

    // Masked fill half of the tensor.
    op_masked_fill_scalar_out(
        tf.make(sizes, /*data=*/{23, 29, 31, 37}),
        tf_bool.make(sizes, /*data=*/{false, true, true, false}),
        /*value=*/71,
        out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_EQ(out, tf.make(sizes, /*data=*/{23, 71, 71, 37}));
  }

  // Common testing for masked fill of floating point Tensor.
  template <ScalarType DTYPE>
  void test_floating_point_masked_fill_scalar_out() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Bool> tf_bool;

    const std::vector<int32_t> sizes = {2, 2};

    // Destination for the masked_fill.
    Tensor out = tf.zeros(sizes);

    // Masked fill half of the tensor.
    op_masked_fill_scalar_out(
        tf.make(sizes, /*data=*/{1.1, 2.2, 4.4, 8.8}),
        tf_bool.make(sizes, /*data=*/{true, false, false, true}),
        /*value=*/3.3,
        out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(out, tf.make(sizes, /*data=*/{3.3, 2.2, 4.4, 3.3}));
  }
};

TEST_F(OpMaskedFillTest, ByteTensors) {
  test_integer_masked_fill_scalar_out<ScalarType::Byte>();
}

TEST_F(OpMaskedFillTest, CharTensors) {
  test_integer_masked_fill_scalar_out<ScalarType::Char>();
}

TEST_F(OpMaskedFillTest, ShortTensors) {
  test_integer_masked_fill_scalar_out<ScalarType::Short>();
}

TEST_F(OpMaskedFillTest, IntTensors) {
  test_integer_masked_fill_scalar_out<ScalarType::Int>();
}

TEST_F(OpMaskedFillTest, LongTensors) {
  test_integer_masked_fill_scalar_out<ScalarType::Long>();
}

TEST_F(OpMaskedFillTest, IntTensorFloatAlphaDies) {
  // add_out() doesn't handle floating alpha for intergal inputs
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Destination for the op.
  Tensor out = tf.zeros(sizes);

  // Elementwise add operation on two integral tensor with floating alpha
  // should cause an assertion and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_,
      op_masked_fill_scalar_out(
          tf.ones(sizes), tf.ones(sizes), /*alpha=*/.7, out));
}

TEST_F(OpMaskedFillTest, FloatTensors) {
  test_floating_point_masked_fill_scalar_out<ScalarType::Float>();
}

TEST_F(OpMaskedFillTest, DoubleTensors) {
  test_floating_point_masked_fill_scalar_out<ScalarType::Double>();
}

TEST_F(OpMaskedFillTest, BoolTensors) {
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Input and mask
  Tensor self = tf.make(sizes, /*data=*/{false, true, false, true});
  Tensor mask = tf.make(sizes, /*data=*/{true, false, true, false});

  // Destination for the masked_fill.
  Tensor out = tf.zeros(sizes);

  op_masked_fill_scalar_out(self, mask, /*value=*/true, out);
  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.ones(sizes));
}

// The input tensor and value may not have different dtypes.
TEST_F(OpMaskedFillTest, MismatchedInputAndValueDtypesDies) {
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;

  const std::vector<int32_t> sizes = {2, 2};

  // Dummy input and mask value
  Tensor self = tf_byte.ones(sizes);
  Tensor mask = tf_char.ones(sizes);

  // Destination for the fill; matches the type of the input.
  Tensor out = tf_byte.zeros(sizes);

  // Filling tensor with mismatched scalar should cause an assertion and kill
  // the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_masked_fill_scalar_out(self, mask, /*value=*/1.3, out));
}

// The output tensor may not have a dtype different from the inputs even if it
// has the same shape.
TEST_F(OpMaskedFillTest, MismatchedOutputDtypeDies) {
  // Two different dtypes. This test uses two types with the same size to
  // demonstrate that the ScalarType itself matters, not the size of the
  // tensor elements.
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Byte> tf_byte;
  TensorFactory<ScalarType::Char> tf_char;

  const std::vector<int32_t> sizes = {2, 2};

  // Input and mask
  Tensor self = tf_byte.ones(sizes);
  Tensor mask = tf_bool.ones(sizes);

  // Destination with a dtype then input.
  Tensor out = tf_char.zeros(sizes);

  // Filling the tensor into a mismatched output should cause an assertion and
  // kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_masked_fill_scalar_out(self, mask, /*fill=*/0, out));
}
// The mask tensor type must be bool, even if shapes are the same
TEST_F(OpMaskedFillTest, MismatchedMaskDtypeDies) {
  TensorFactory<ScalarType::Int> tf;

  const std::vector<int32_t> sizes = {2, 2};

  // Input and destination
  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes);

  // Mask tensor with non bool dtype
  Tensor mask = tf.ones(sizes);

  // Filling the tensor using non boolean mask should cause an assertion and
  // kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_masked_fill_scalar_out(self, mask, /*fill=*/0, out));
}

// Mismatched shape tests.
TEST_F(OpMaskedFillTest, MismatchedInputShapesDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Input and mask of different shapes that cannot be broadcasted.
  Tensor self = tf.ones(/*sizes=*/{4});
  Tensor mask = tf_bool.ones(/*sizes=*/{2});

  // Destination for the sum; matches the shape of one of the inputs.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Masked fill with mismatch input and mask shapes should cause an assertion
  // and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_masked_fill_scalar_out(self, mask, /*value=*/0, out));
}

TEST_F(OpMaskedFillTest, BroadcastTest) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  // Input and mask of different shapes
  Tensor self = tf.make({2, 2}, /*data=*/{1, 2, 4, 8});
  Tensor mask = tf_bool.make({2}, /*data=*/{true, false});

  // Destination for the masked_fill.
  Tensor out = tf.zeros({2, 2});

  // Masked fill half of the tensor.
  op_masked_fill_scalar_out(
      self,
      mask,
      /*value=*/3,
      out);

  // Check that it matches the expected output.
  EXPECT_TENSOR_CLOSE(out, tf.make({2, 2}, /*data=*/{3, 2, 3, 8}));
}

TEST_F(OpMaskedFillTest, MismatchedOutputShapesDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched output shape";
  }
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  const std::vector<int32_t> sizes = {2, 2};

  // Input and mask of different shapes
  Tensor a = tf.ones(sizes);
  Tensor b = tf_bool.ones(sizes);

  // Destination with a different shape.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Mask filling the tensor into a mismatched output should cause an assertion
  // and kill the test process.
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_masked_fill_scalar_out(a, b, /*value=*/0, out));
}

TEST_F(OpMaskedFillTest, BroadcastDimSizeIsOneAB) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> bool_tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9701170325279236,
       0.4185227155685425,
       0.39851099252700806,
       0.8725584745407104,
       0.714692234992981,
       0.3167606592178345});
  Tensor y = bool_tf.make({1, 2}, {false, false});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.9701170325279236,
       0.4185227155685425,
       0.39851099252700806,
       0.8725584745407104,
       0.714692234992981,
       0.3167606592178345});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_masked_fill_scalar_out(x, y, Scalar(3.0), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMaskedFillTest, BroadcastDimSizeMissingAB) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> bool_tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9701170325279236,
       0.4185227155685425,
       0.39851099252700806,
       0.8725584745407104,
       0.714692234992981,
       0.3167606592178345});
  Tensor y = bool_tf.make({2}, {false, false});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.9701170325279236,
       0.4185227155685425,
       0.39851099252700806,
       0.8725584745407104,
       0.714692234992981,
       0.3167606592178345});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_masked_fill_scalar_out(x, y, Scalar(3.0), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMaskedFillTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> bool_tf;

  Tensor x = tf.make(
      {3, 2},
      {0.974706768989563,
       0.46383917331695557,
       0.050839245319366455,
       0.26296138763427734,
       0.8404526114463806,
       0.49675875902175903});
  Tensor y = bool_tf.make({3, 2}, {false, false, false, false, false, true});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.974706768989563,
       0.46383917331695557,
       0.050839245319366455,
       0.26296138763427734,
       0.8404526114463806,
       3.0});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_masked_fill_scalar_out(x, y, Scalar(3.0), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMaskedFillTest, DynamicShapeUpperBoundLargerThanExpected) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> bool_tf;

  Tensor x = tf.make(
      {3, 2},
      {0.974706768989563,
       0.46383917331695557,
       0.050839245319366455,
       0.26296138763427734,
       0.8404526114463806,
       0.49675875902175903});
  Tensor y = bool_tf.make({3, 2}, {false, false, false, false, false, true});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.974706768989563,
       0.46383917331695557,
       0.050839245319366455,
       0.26296138763427734,
       0.8404526114463806,
       3.0});

  Tensor out =
      tf.zeros({6, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_masked_fill_scalar_out(x, y, Scalar(3.0), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMaskedFillTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape unbound not supported";
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> bool_tf;

  Tensor x = tf.make(
      {3, 2},
      {0.974706768989563,
       0.46383917331695557,
       0.050839245319366455,
       0.26296138763427734,
       0.8404526114463806,
       0.49675875902175903});
  Tensor y = bool_tf.make({3, 2}, {false, false, false, false, false, true});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.974706768989563,
       0.46383917331695557,
       0.050839245319366455,
       0.26296138763427734,
       0.8404526114463806,
       3.0});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_masked_fill_scalar_out(x, y, Scalar(3.0), out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMaskedFillTest, BroadcastDimSizeIsOneBA) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  auto x = tf.make(
      {3, 2},
      {0.38566190004348755,
       0.47776442766189575,
       0.1954779028892517,
       0.6691004633903503,
       0.6580829620361328,
       0.48968571424484253});
  auto y = tf_bool.make({2}, {false, false});
  auto z = Scalar(3.0);
  Tensor expected_result = tf.make(
      {3, 2},
      {0.38566190004348755,
       0.47776442766189575,
       0.1954779028892517,
       0.6691004633903503,
       0.6580829620361328,
       0.48968571424484253});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_masked_fill_scalar_out(x, y, z, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpMaskedFillTest, BroadcastDimSizeMissingBA) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tf_bool;

  auto x = tf.make(
      {3, 2},
      {0.38566190004348755,
       0.47776442766189575,
       0.1954779028892517,
       0.6691004633903503,
       0.6580829620361328,
       0.48968571424484253});
  auto y = tf_bool.make({2}, {false, false});
  auto z = Scalar(3.0);
  Tensor expected_result = tf.make(
      {3, 2},
      {0.38566190004348755,
       0.47776442766189575,
       0.1954779028892517,
       0.6691004633903503,
       0.6580829620361328,
       0.48968571424484253});

  Tensor out = tf.zeros({3, 2});
  Tensor ret = op_masked_fill_scalar_out(x, y, z, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
