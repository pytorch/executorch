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
using exec_aten::string_view;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpGeluTest : public OperatorTest {
 protected:
  Tensor&
  op_gelu_out(const Tensor& self, string_view approximate, Tensor& out) {
    return torch::executor::aten::gelu_outf(context_, self, approximate, out);
  }

  // Common testing for gelu on two floating point Tensors.
  template <ScalarType DTYPE>
  void test_gelu_execution() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {3, 2};

    Tensor in = tf.make(
        sizes, /*data=*/{-0.4775, 0.2948, -0.3984, 1.8690, -0.4048, -0.4848});

    // Destination for the gelu.
    Tensor out = tf.zeros(sizes);

    // Run full gelu.
    op_gelu_out(in, "none", out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(
        out,
        tf.make(
            sizes,
            /*data=*/
            {-0.15113, 0.181575, -0.137515, 1.81141, -0.13877, -0.152183}));

    // Run tanh gelu appx.
    op_gelu_out(in, "tanh", out);

    // Check that it matches the expected output.
    EXPECT_TENSOR_CLOSE(
        out,
        tf.make(
            sizes,
            /*data=*/
            {-0.151145, 0.181573, -0.137522, 1.8114, -0.138778, -0.152199}));
  }
};

TEST_F(OpGeluTest, FloatTensors) {
  test_gelu_execution<ScalarType::Float>();
}

TEST_F(OpGeluTest, DoubleTensors) {
  if (!SupportedFeatures::get()->op_gelu_dtype_double) {
    GTEST_SKIP();
  }

  test_gelu_execution<ScalarType::Double>();
}

TEST_F(OpGeluTest, UnhandledDtypeDies) {
  // gelu() doesn't handle Bool.
  TensorFactory<ScalarType::Bool> tf;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf.make(sizes, /*data=*/{false, true, false, true});

  // Destination for the gelu.
  Tensor out = tf.zeros(sizes);

  ET_EXPECT_KERNEL_FAILURE(context_, op_gelu_out(a, "none", out));
}

// The output tensor may not have a dtype different from the inputs even if it
// has the same shape.
TEST_F(OpGeluTest, MismatchedOutputDtypeDies) {
  // Two different dtypes. This test uses two types with the same size to
  // demonstrate that the ScalarType itself matters, not the size of the
  // tensor elements.
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;

  const std::vector<int32_t> sizes = {2, 2};

  Tensor a = tf_float.ones(sizes);

  // Destination with a dtype different from the input.
  Tensor out = tf_double.zeros(sizes);

  // Running Gelu on an input into an output of a different dtype should kill
  // the program
  ET_EXPECT_KERNEL_FAILURE(context_, op_gelu_out(a, "none", out));
}

TEST_F(OpGeluTest, InvalidAppxStringDies) {
  TensorFactory<ScalarType::Float> tf;

  Tensor a = tf.ones(/*sizes=*/{4});

  // Destination for the gelu; matches the shape of the inputs.
  Tensor out = tf.zeros(/*sizes=*/{4});

  // Running Gelu with an invalid appx method should kill the program.
  ET_EXPECT_KERNEL_FAILURE(context_, op_gelu_out(a, "foo", out));
}

TEST_F(OpGeluTest, SimpleGeneratedCase) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {10, 10},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  Tensor expected_result = tf.make(
      {10, 10}, {0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193, 0.8411920070648193, 0.8411920070648193,
                 0.8411920070648193});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_gelu_out(x, "tanh", out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpGeluTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9769402146339417,
       0.4728269577026367,
       0.04416435956954956,
       0.7145527601242065,
       0.7109619975090027,
       0.36388522386550903});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.8162848949432373,
       0.3223743438720703,
       0.022860059514641762,
       0.5448282957077026,
       0.5413010716438293,
       0.23361928761005402});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_gelu_out(x, "tanh", out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpGeluTest, DynamicShapeUpperBoundLargerThanExpected) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9769402146339417,
       0.4728269577026367,
       0.04416435956954956,
       0.7145527601242065,
       0.7109619975090027,
       0.36388522386550903});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.8162848949432373,
       0.3223743438720703,
       0.022860059514641762,
       0.5448282957077026,
       0.5413010716438293,
       0.23361928761005402});

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_gelu_out(x, "tanh", out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpGeluTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {3, 2},
      {0.9769402146339417,
       0.4728269577026367,
       0.04416435956954956,
       0.7145527601242065,
       0.7109619975090027,
       0.36388522386550903});
  Tensor expected_result = tf.make(
      {3, 2},
      {0.8162848949432373,
       0.3223743438720703,
       0.022860059514641762,
       0.5448282957077026,
       0.5413010716438293,
       0.23361928761005402});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_gelu_out(x, "tanh", out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}
