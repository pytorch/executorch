/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <ostream>

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::IntArrayRef;
using exec_aten::nullopt;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

using OptScalar = exec_aten::optional<Scalar>;

class OpNativeLayerNormTest : public OperatorTest {
 protected:
  ::std::tuple<Tensor&, Tensor&, Tensor&> op_native_layer_norm_out(
      const Tensor& input,
      IntArrayRef normalized_shape,
      const optional<Tensor>& weight,
      const optional<Tensor>& bias,
      double eps,
      Tensor& out0,
      Tensor& out1,
      Tensor& out2) {
    return torch::executor::aten::native_layer_norm_outf(
        context_, input, normalized_shape, weight, bias, eps, out0, out1, out2);
  }

  template <ScalarType DTYPE>
  struct NativeLayerNormTestCase {
    using ctype = typename TensorFactory<DTYPE>::ctype;

    // Human-readable, unique title for the test case. Printed if the test
    // fails.
    const std::string title;
    // Size vector for the input/output
    const std::vector<int32_t> sizes;
    // Data for the input tensor; must agree with `sizes`.
    const std::vector<ctype> input_data;
    // The normalized shape. Only the last dim is accepted.
    const std::vector<int32_t> normalized_shape;
    // Affine transform weight.
    const std::vector<ctype> weight_data;
    // Affine transform bias.
    const std::vector<ctype> bias_data;
    // a value added to the denominator for numerical stability
    const ctype eps;
    // The expected output data.
    const std::vector<ctype> expected_data;
  };

  /// Runs the provided test cases.
  template <ScalarType DTYPE>
  void run_test_cases(std::vector<NativeLayerNormTestCase<DTYPE>> test_cases) {
    TensorFactory<DTYPE> tf;
    for (const auto& test_case : test_cases) {
      SCOPED_TRACE(test_case.title); // Printed if the test fails

      Tensor in = tf.make(test_case.sizes, test_case.input_data);
      Tensor weight =
          tf.make(test_case.normalized_shape, test_case.weight_data);
      Tensor bias = tf.make(test_case.normalized_shape, test_case.bias_data);
      Tensor out0 = tf.zeros(test_case.sizes);
      Tensor out1 = tf.zeros(
          test_case.sizes, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
      Tensor out2 = tf.zeros(
          test_case.sizes, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
      auto normalized_shape_vec = std::vector<int64_t>(
          test_case.normalized_shape.begin(), test_case.normalized_shape.end());
      auto normalized_shape = exec_aten::ArrayRef<int64_t>(
          normalized_shape_vec.data(), normalized_shape_vec.size());
      auto result = op_native_layer_norm_out(
          in, normalized_shape, weight, bias, test_case.eps, out0, out1, out2);
      EXPECT_TENSOR_CLOSE(out0, std::get<0>(result));

      Tensor expected = tf.make(test_case.sizes, test_case.expected_data);
      EXPECT_TENSOR_CLOSE(out0, expected);
    }
  }

  // Test cases that are compatible with float and double.
  template <ScalarType DTYPE>
  void run_floating_point_test_cases() {
    constexpr auto kInfinity =
        std::numeric_limits<typename TensorFactory<DTYPE>::ctype>::infinity();
    // Reference colab note:
    // https://colab.research.google.com/drive/1KZT6sEY-h7lwZlwBanbLl77M5OuzzsZI#scrollTo=18WtUPCXYCPx
    std::vector<NativeLayerNormTestCase<DTYPE>> test_cases = {
        {
            std::string(__func__) + ": Simple negative/positive layer norm",
            {2, 3}, // sizes
            {1.0, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {1.0, 1.0, 1.0}, // weights
            {0.0, 0.0, 0.0}, // bias
            1.0e-5, // eps
            {1.22474,
             0.0000,
             -1.22474,
             -0.925819,
             1.38873,
             -0.46291}, // expected_data
        },
        {
            std::string(__func__) + ": non-default eps",
            {2, 3}, // sizes
            {1.0, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {1.0, 1.0, 1.0}, // weights
            {0.0, 0.0, 0.0}, // bias
            1.0e-3, // eps
            {1.22383,
             0,
             -1.22383,
             -0.925721,
             1.38858,
             -0.46286}, // expected_data
        },
        {
            std::string(__func__) + ": non-default weights",
            {2, 3}, // sizes
            {1.0, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {2.0, 2.0, 2.0}, // weights
            {0.0, 0.0, 0.0}, // bias
            1.0e-5, // eps
            {2.44947,
             0,
             -2.44947,
             -1.85164,
             2.77746,
             -0.925819}, // expected_data
        },
        {
            std::string(__func__) + ": non-default bias",
            {2, 3}, // sizes
            {1.0, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {1.0, 1.0, 1.0}, // weights
            {1.0, 1.0, 1.0}, // bias
            1.0e-5, // eps
            {2.22474,
             1,
             -0.224736,
             0.0741809,
             2.38873,
             0.53709}, // expected_data
        },
        {
            std::string(__func__) + ": infinite input brings NAN results",
            {2, 3}, // sizes
            {kInfinity, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {1.0, 1.0, 1.0}, // weights
            {1.0, 1.0, 1.0}, // bias
            1.0e-5, // eps
            {-NAN, -NAN, -NAN, 0.0741809, 2.38873, 0.53709}, // expected_data
        },
        {
            std::string(__func__) + ": NAN input brings NAN results",
            {2, 3}, // sizes
            {NAN, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {1.0, 1.0, 1.0}, // weights
            {1.0, 1.0, 1.0}, // bias
            1.0e-5, // eps
            {-NAN, -NAN, -NAN, 0.0741809, 2.38873, 0.53709}, // expected_data
        },
        {
            std::string(__func__) + ": NAN weight brings NAN results",
            {2, 3}, // sizes
            {1.0, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {NAN, 1.0, 1.0}, // weights
            {1.0, 1.0, 1.0}, // bias
            1.0e-5, // eps
            {NAN, 1, -0.224736, NAN, 2.38873, 0.53709}, // expected_data
        },
        {
            std::string(__func__) + ": inf weight brings inf results",
            {2, 3}, // sizes
            {1.0, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {kInfinity, 1.0, 1.0}, // weights
            {0.0, 0.0, 0.0}, // bias
            1.0e-5, // eps
            {kInfinity,
             0,
             -1.22474,
             -kInfinity,
             1.38873,
             -0.46291}, // expected_data
        },
        {
            std::string(__func__) + ": inf bias brings inf results",
            {2, 3}, // sizes
            {1.0, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {3}, // normalized shape
            {kInfinity, 1.0, 1.0}, // weights
            {0.0, 0.0, 0.0}, // bias
            1.0e-5, // eps
            {kInfinity,
             0,
             -1.22474,
             -kInfinity,
             1.38873,
             -0.46291}, // expected_data
        },
    };

    run_test_cases(test_cases);
  }

  // Runs death test cases.
  template <ScalarType DTYPE>
  void run_death_test_cases(
      std::vector<NativeLayerNormTestCase<DTYPE>> test_cases) {
    TensorFactory<DTYPE> tf;
    for (const auto& test_case : test_cases) {
      SCOPED_TRACE(test_case.title); // Printed if the test fails

      Tensor in = tf.make(test_case.sizes, test_case.input_data);
      exec_aten::optional<Tensor> weight, bias;
      if (!test_case.weight_data.empty()) {
        weight = tf.make(test_case.normalized_shape, test_case.weight_data);
      }
      if (!test_case.bias_data.empty()) {
        bias = tf.make(test_case.normalized_shape, test_case.bias_data);
      }
      Tensor out0 = tf.zeros(test_case.sizes);
      Tensor out1 = tf.zeros(test_case.sizes);
      Tensor out2 = tf.zeros(test_case.sizes);
      auto normalized_shape_vec = std::vector<int64_t>(
          test_case.normalized_shape.begin(), test_case.normalized_shape.end());
      auto normalized_shape = exec_aten::ArrayRef<int64_t>(
          normalized_shape_vec.data(), normalized_shape_vec.size());
      ET_EXPECT_KERNEL_FAILURE(
          context_,
          op_native_layer_norm_out(
              in,
              normalized_shape,
              weight,
              bias,
              test_case.eps,
              out0,
              out1,
              out2));
    }
  }

  // Test cases with imcompatible types.
  template <ScalarType DTYPE>
  void run_int_test_cases() {
    std::vector<NativeLayerNormTestCase<DTYPE>> test_cases = {
        {
            std::string(__func__) + ": Simple negative/positive layer norm",
            // Cannot be represented by a type other than float.
            {2, 3}, // sizes
            {1, 0, -1, -1, 4, 0}, // input_data
            {3}, // normalized shape
            {1, 1, 1}, // weights
            {0, 0, 0}, // bias
            1, // eps
            {0, 0, 0, 0, 0, 0}, // expected_data
        },
    };
    run_death_test_cases(test_cases);
  }

  // Test cases with wrong normalized shape.
  template <ScalarType DTYPE>
  void run_wrong_shape_test_cases() {
    std::vector<NativeLayerNormTestCase<DTYPE>> test_cases = {
        {
            std::string(__func__) + ": Test with wrong normalized shape",
            {2, 3}, // sizes
            {1.0, 0.0, -1.0, -1.0, 4.0, 0.0}, // input_data
            {1}, // wrong normalized shape
            {1.0}, // weights
            {0.0}, // bias
            1.0e-5, // eps
            {1.22474,
             0.0000,
             -1.22474,
             -0.925819,
             1.38873,
             -0.46291}, // expected_data
        },
    };
    run_death_test_cases(test_cases);
  }

  /* %python
  import torch
  torch.manual_seed(0)

  input = torch.rand(2, 3)
  normalized_shape = [3]
  weight = torch.tensor([1.0, 1.0, 1.0])
  bias = torch.tensor([0.0, 0.0, 0.0])
  eps = 1e-05
  expected = torch.nn.functional.layer_norm(
    input, normalized_shape, weight=weight, bias=bias, eps=eps)

  native_layer_norm_template = f"""
    {declare_tensor_factory("ScalarType::Float", "tf")}

    {declare_tensor_make_t("input", "tf")}
    {declare_optional_tensor_make_t("weight", "tf")}
    {declare_optional_tensor_make_t("bias", "tf")}
    {declare_tensor_make_t("expected", "tf")}
    {declare_tensor_zeros("out_shape, dynamism", "tf", "out0")}
    {declare_tensor_zeros("out_shape, dynamism", "tf", "out1")}
    {declare_tensor_zeros("out_shape, dynamism", "tf", "out2")}

    int64_t normalized_shape[] = $normalized_shape$;

    op_native_layer_norm_out(
      input, normalized_shape, weight, bias, $eps$, out0, out1, out2);
    EXPECT_TENSOR_CLOSE(out0, expected);""" */
  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(native_layer_norm_template) */

    TensorFactory<ScalarType::Float> tf;

    Tensor input = tf.make(
        {2, 3},
        {0.49625658988952637,
         0.7682217955589294,
         0.08847743272781372,
         0.13203048706054688,
         0.30742281675338745,
         0.6340786814689636});
    optional<Tensor> weight(tf.make({3}, {1.0, 1.0, 1.0}));
    optional<Tensor> bias(tf.make({3}, {0.0, 0.0, 0.0}));
    Tensor expected = tf.make(
        {2, 3},
        {0.16205203533172607,
         1.1355723142623901,
         -1.2976245880126953,
         -1.0853172540664673,
         -0.24233698844909668,
         1.3276543617248535});
    Tensor out0 = tf.zeros(out_shape, dynamism);
    Tensor out1 = tf.zeros(out_shape, dynamism);
    Tensor out2 = tf.zeros(out_shape, dynamism);

    int64_t normalized_shape[] = {3};

    op_native_layer_norm_out(
        input, normalized_shape, weight, bias, 1e-05, out0, out1, out2);
    EXPECT_TENSOR_CLOSE(out0, expected);
  }
};

namespace {
std::vector<int64_t> vector_32_to_64(std::vector<int32_t> vector_32) {
  std::vector<int64_t> vector_64(vector_32.size());
  std::transform(
      vector_32.begin(), vector_32.end(), vector_64.begin(), [](int32_t x) {
        return static_cast<int64_t>(x);
      });
  return vector_64;
}

} // namespace

/// Describes a test case, using tensors of the specified DTYPE.
TEST_F(OpNativeLayerNormTest, FloatTensors) {
  run_floating_point_test_cases<ScalarType::Float>();
  run_floating_point_test_cases<ScalarType::Double>();
}

TEST_F(OpNativeLayerNormTest, IntTensorsDies) {
  // Cannot be represented by a type other than float.
  run_int_test_cases<ScalarType::Int>();
}

TEST_F(OpNativeLayerNormTest, WrongNomalizedShape) {
  // Normalized shape does not match last dim of input.
  run_wrong_shape_test_cases<ScalarType::Float>();
}

TEST_F(OpNativeLayerNormTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpNativeLayerNormTest, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpNativeLayerNormTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
