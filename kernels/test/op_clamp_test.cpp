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
using exec_aten::nullopt;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

using OptScalar = exec_aten::optional<Scalar>;

class OpClampOutTest : public OperatorTest {
 protected:
  Tensor& op_clamp_out(
      const Tensor& self,
      const optional<Scalar>& min,
      const optional<Scalar>& max,
      Tensor& out) {
    return torch::executor::aten::clamp_outf(context_, self, min, max, out);
  }

  template <ScalarType DTYPE>
  struct ClampTestCase {
    using ctype = typename TensorFactory<DTYPE>::ctype;

    // Human-readable, unique title for the test case. Printed if the test
    // fails.
    const std::string title;
    // Size vector for the input/output tensors.
    const std::vector<int32_t> sizes;
    // Data for the input tensor; must agree with `sizes`.
    const std::vector<ctype> input_data;
    // The (optional) min value to clamp to. Can be of any Scalar type.
    const OptScalar min;
    // The (optional) max value to clamp to. Can be of any Scalar type.
    const OptScalar max;
    // The expected output data when clamping `input_data` to `min`/`max`.
    const std::vector<ctype> expected_data;
  };

  /// Runs the provided test cases.
  template <ScalarType DTYPE>
  void run_test_cases(std::vector<ClampTestCase<DTYPE>> test_cases) {
    TensorFactory<DTYPE> tf;
    for (const auto& test_case : test_cases) {
      SCOPED_TRACE(test_case.title); // Printed if the test fails

      Tensor in = tf.make(test_case.sizes, test_case.input_data);
      Tensor out = tf.zeros(test_case.sizes);
      Tensor ret = op_clamp_out(in, test_case.min, test_case.max, out);
      EXPECT_TENSOR_EQ(out, ret);

      Tensor expected = tf.make(test_case.sizes, test_case.expected_data);
      ET_CHECK_SAME_SHAPE_AND_DTYPE2(out, expected);
      EXPECT_TENSOR_EQ(out, expected);
    }
  }

  template <ScalarType DTYPE>
  void run_unsigned_integer_test_cases() {
    const std::vector<ClampTestCase<DTYPE>> test_cases = {
        {
            std::string(__func__) + ": Simple clamp",
            {2, 2}, // sizes
            {0, 1, 10, 100}, // input_data
            OptScalar(1), // min
            OptScalar(6), // max
            {1, 1, 6, 6}, // expected_data
        },
        {
            std::string(__func__) + ": No max",
            {2, 2}, // sizes
            {0, 1, 10, 100}, // input_data
            OptScalar(1), // min
            nullopt, // max
            {1, 1, 10, 100}, // expected_data
        },
        {
            std::string(__func__) + ": No min",
            {2, 2}, // sizes
            {0, 1, 10, 100}, // input_data
            nullopt, // min
            OptScalar(6), // max
            {0, 1, 6, 6}, // expected_data
        },
        {
            std::string(__func__) + ": min > max",
            {2, 2}, // sizes
            {0, 1, 10, 100}, // input_data
            OptScalar(10), // min
            OptScalar(6), // max
            // Should set all elements to max.
            {6, 6, 6, 6}, // expected_data
        },
    };

    run_test_cases(test_cases);
  }

  // types.
  template <ScalarType DTYPE>
  void run_signed_integer_test_cases() {
    std::vector<ClampTestCase<DTYPE>> test_cases = {
        {
            std::string(__func__) + ": Simple negative/positive clamp",
            {2, 2}, // sizes
            {-10, -1, 1, 10}, // input_data
            OptScalar(-5), // min
            OptScalar(5), // max
            {-5, -1, 1, 5}, // expected_data
        },
        {
            std::string(__func__) + ": Simple negative-only clamp",
            {2, 2}, // sizes
            {-10, -5, 1, 10}, // input_data
            OptScalar(-6), // min
            OptScalar(-1), // max
            {-6, -5, -1, -1}, // expected_data
        },
    };

    run_test_cases(test_cases);
  }

  // Test cases that are compatible with float and double.
  template <ScalarType DTYPE>
  void run_floating_point_test_cases() {
    using ctype = typename TensorFactory<DTYPE>::ctype;
    using opt_infinity_type = std::conditional_t<
        std::is_same<ctype, exec_aten::Half>::value,
        float,
        ctype>;
    constexpr auto kInfinity = std::numeric_limits<ctype>::infinity();
    const auto kOptInfinity =
        OptScalar(static_cast<opt_infinity_type>(kInfinity));
    const auto kOptMinusInfinity =
        OptScalar(static_cast<opt_infinity_type>(-kInfinity));
    std::vector<ClampTestCase<DTYPE>> test_cases = {
        {
            std::string(__func__) + ": Simple negative/positive clamp",
            {2, 2}, // sizes
            {-10.1, -1.1, 1.1, 10.1}, // input_data
            OptScalar(-5.5), // min
            OptScalar(5.5), // max
            {-5.5, -1.1, 1.1, 5.5}, // expected_data
        },
        {
            std::string(__func__) + ": Simple negative-only clamp",
            {2, 2}, // sizes
            {-10.1, -5.5, 1.1, 10.1}, // input_data
            OptScalar(-6.6), // min
            OptScalar(-1.1), // max
            {-6.6, -5.5, -1.1, -1.1}, // expected_data
        },
        {
            std::string(__func__) + ": Infinities are clamped",
            {2, 2}, // sizes
            {-kInfinity, -1.1, 1.1, kInfinity}, // input_data
            OptScalar(-5.5), // min
            OptScalar(5.5), // max
            {-5.5, -1.1, 1.1, 5.5}, // expected_data
        },
        {
            std::string(__func__) + ": Infinite min",
            {2, 2}, // sizes
            {-10.1, -1.1, 1.1, 10.1}, // input_data
            kOptMinusInfinity, // min
            OptScalar(5.5), // max
            {-10.1, -1.1, 1.1, 5.5}, // expected_data
        },
        {
            std::string(__func__) + ": Infinite max",
            {2, 2}, // sizes
            {-10.1, -1.1, 1.1, 10.1}, // input_data
            OptScalar(-5.5), // min
            kOptInfinity, // max
            {-5.5, -1.1, 1.1, 10.1}, // expected_data
        },
        {
            std::string(__func__) + ": NaN entries preserved",
            {2, 2}, // sizes
            {-10.1, NAN, NAN, 10.1}, // input_data
            OptScalar(0.0), // min
            OptScalar(0.0), // max
            {0.0, NAN, NAN, 0.0}, // expected_data
        },
        {
            std::string(__func__) + ": NaN min produces all NaN output",
            {2, 2}, // sizes
            {-10.1, -1.1, 1.1, 10.1}, // input_data
            OptScalar(NAN), // min
            OptScalar(5.5), // max
            {NAN, NAN, NAN, NAN}, // expected_data
        },
        {
            std::string(__func__) + ": NaN max produces all NaN output",
            {2, 2}, // sizes
            {-10.1, -1.1, 1.1, 10.1}, // input_data
            OptScalar(-5.5), // min
            OptScalar(NAN), // max
            {NAN, NAN, NAN, NAN}, // expected_data
        },
    };

    run_test_cases(test_cases);
  }

  // Tries clamping a DTYPE tensor to the provided value and expects it to die.
  template <ScalarType DTYPE>
  void expect_bad_clamp_value_dies(Scalar bad_value) {
    TensorFactory<DTYPE> tf;
    Tensor in = tf.ones({2, 2});
    Tensor out = tf.zeros({2, 2});

    ET_EXPECT_KERNEL_FAILURE(
        context_, op_clamp_out(in, /*min=*/bad_value, /*max=*/nullopt, out));
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_clamp_out(in, /*min=*/nullopt, /*max=*/bad_value, out));
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_clamp_out(in, /*min=*/bad_value, /*max=*/bad_value, out));
  }

  // One of min and max should be non-null
  void expect_both_min_max_null_die() {
    TensorFactory<ScalarType::Float> tf;
    Tensor in = tf.ones({2, 2});
    Tensor out = tf.zeros({2, 2});

    ET_EXPECT_KERNEL_FAILURE(
        context_, op_clamp_out(in, /*min=*/nullopt, /*max=*/nullopt, out));
  }
};

class OpClampTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_clamp_tensor_out(
      const Tensor& self,
      const optional<Tensor>& min,
      const optional<Tensor>& max,
      Tensor& out) {
    executorch::runtime::KernelRuntimeContext context{};
    return torch::executor::aten::clamp_outf(context, self, min, max, out);
  }
};

/// Describes a test case, using tensors of the specified DTYPE.
// Runs test cases that are compatible with uint8_t, and thus all other real
// types. Cover the most cases here, since it's compatible with the most types.
// Runs test cases that are compatible with int8_t, and thus all signed real
TEST_F(OpClampOutTest, ByteTensors) {
  run_unsigned_integer_test_cases<ScalarType::Byte>();
}

TEST_F(OpClampOutTest, CharTensors) {
  run_unsigned_integer_test_cases<ScalarType::Char>();
  run_signed_integer_test_cases<ScalarType::Char>();
}

TEST_F(OpClampOutTest, ShortTensors) {
  run_unsigned_integer_test_cases<ScalarType::Short>();
  run_signed_integer_test_cases<ScalarType::Short>();
}

TEST_F(OpClampOutTest, IntTensors) {
  run_unsigned_integer_test_cases<ScalarType::Int>();
  run_signed_integer_test_cases<ScalarType::Int>();
}

TEST_F(OpClampOutTest, LongTensors) {
  run_unsigned_integer_test_cases<ScalarType::Long>();
  run_signed_integer_test_cases<ScalarType::Long>();
}

TEST_F(OpClampOutTest, HalfTensors) {
  // Note that the integer test cases test the situation where the min/max value
  // Scalars are integer types, demonstrating that floating point types can be
  // clamped to integer values.
  run_unsigned_integer_test_cases<ScalarType::Half>();
  run_signed_integer_test_cases<ScalarType::Half>();
  run_floating_point_test_cases<ScalarType::Half>();
}

TEST_F(OpClampOutTest, FloatTensors) {
  // Note that the integer test cases test the situation where the min/max value
  // Scalars are integer types, demonstrating that floating point types can be
  // clamped to integer values.
  run_unsigned_integer_test_cases<ScalarType::Float>();
  run_signed_integer_test_cases<ScalarType::Float>();
  run_floating_point_test_cases<ScalarType::Float>();
}

TEST_F(OpClampOutTest, DoubleTensors) {
  // Note that the integer test cases test the situation where the min/max value
  // Scalars are integer types, demonstrating that floating point types can be
  // clamped to integer values.
  run_unsigned_integer_test_cases<ScalarType::Double>();
  run_signed_integer_test_cases<ScalarType::Double>();
  run_floating_point_test_cases<ScalarType::Double>();
}

//
// Don't test every type, just a representative sample: unsigned int, signed
// int, floating point.
//

TEST_F(OpClampOutTest, ByteTensorNegativeClampDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle negative clamp on byte tensor";
  }
  // Cannot be represented by a uint8_t.
  expect_bad_clamp_value_dies<ScalarType::Byte>(-1);
}

TEST_F(OpClampOutTest, ByteTensorTooLargeClampDies) {
  // Cannot be represented by a uint8_t.
  expect_bad_clamp_value_dies<ScalarType::Byte>(256);
}

TEST_F(OpClampOutTest, ByteTensorFloatingPointClampDies) {
  // Cannot be represented by a uint8_t.
  expect_bad_clamp_value_dies<ScalarType::Byte>(2.2);
}

#ifndef USE_ATEN_LIB
TEST_F(OpClampOutTest, IntTensorTooSmallClampDies) {
  // Cannot be represented by a int32_t.
  expect_bad_clamp_value_dies<ScalarType::Int>(-2147483649);
}

TEST_F(OpClampOutTest, IntTensorTooLargeClampDies) {
  // Cannot be represented by a int32_t.
  expect_bad_clamp_value_dies<ScalarType::Int>(2147483648);
}
#endif

TEST_F(OpClampOutTest, IntTensorFloatingPointClampDies) {
  // Cannot be represented by a uint32_t.
  expect_bad_clamp_value_dies<ScalarType::Int>(2.2);
}

TEST_F(OpClampOutTest, FloatTensorTooSmallClampDies) {
  // Cannot be represented by a float.
  expect_bad_clamp_value_dies<ScalarType::Float>(-3.41e+38);
}

TEST_F(OpClampOutTest, FloatTensorTooLargeClampDies) {
  // Cannot be represented by a float.
  expect_bad_clamp_value_dies<ScalarType::Float>(3.41e+38);
}

TEST_F(OpClampOutTest, SimpleGeneratedCase) {
  TensorFactory<ScalarType::Float> tf;

  auto x = tf.make(
      {10, 10},
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  auto y = OptScalar(-0.5);
  auto z = OptScalar(0.5);
  Tensor expected_result = tf.make(
      {10, 10},
      {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
       0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5});

  Tensor out = tf.zeros({10, 10});
  Tensor ret = op_clamp_out(x, y, z, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpClampOutTest, DynamicShapeUpperBoundSameAsExpected) {
  TensorFactory<ScalarType::Float> tf;

  auto x = tf.make(
      {3, 2},
      {0.6984410881996155,
       0.5675464272499084,
       0.8352431654930115,
       0.2055988311767578,
       0.593172013759613,
       0.11234724521636963});
  auto y = OptScalar(-0.5);
  auto z = OptScalar(0.5);
  Tensor expected_result = tf.make(
      {3, 2}, {0.5, 0.5, 0.5, 0.2055988311767578, 0.5, 0.11234724521636963});

  Tensor out =
      tf.zeros({3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_clamp_out(x, y, z, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpClampOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  auto x = tf.make(
      {3, 2},
      {0.6984410881996155,
       0.5675464272499084,
       0.8352431654930115,
       0.2055988311767578,
       0.593172013759613,
       0.11234724521636963});
  auto y = OptScalar(-0.5);
  auto z = OptScalar(0.5);
  Tensor expected_result = tf.make(
      {3, 2}, {0.5, 0.5, 0.5, 0.2055988311767578, 0.5, 0.11234724521636963});

  Tensor out =
      tf.zeros({6, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor ret = op_clamp_out(x, y, z, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpClampOutTest, DynamicShapeUnbound) {
  GTEST_SKIP() << "Dynamic shape not supported";
  TensorFactory<ScalarType::Float> tf;

  auto x = tf.make(
      {3, 2},
      {0.6984410881996155,
       0.5675464272499084,
       0.8352431654930115,
       0.2055988311767578,
       0.593172013759613,
       0.11234724521636963});
  auto y = OptScalar(-0.5);
  auto z = OptScalar(0.5);
  Tensor expected_result = tf.make(
      {3, 2}, {0.5, 0.5, 0.5, 0.2055988311767578, 0.5, 0.11234724521636963});

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  Tensor ret = op_clamp_out(x, y, z, out);
  EXPECT_TENSOR_CLOSE(out, expected_result);
}

TEST_F(OpClampTensorOutTest, SmokeTest) {
  TensorFactory<ScalarType::Byte> tf_in;
  TensorFactory<ScalarType::Int> tf_min;
  TensorFactory<ScalarType::Char> tf_max;
  TensorFactory<ScalarType::Short> tf_out;

  Tensor in = tf_in.make({1, 1}, {3});
  Tensor min = tf_min.make({1, 3}, {0, 1, 4});
  Tensor max = tf_max.make({2, 1}, {2, 5});
  Tensor out = tf_out.zeros({2, 3});
  Tensor expected = tf_out.make({2, 3}, {2, 2, 2, 3, 3, 4});

  op_clamp_tensor_out(in, min, max, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpClampTensorOutTest, DowncastingSmokeTest) {
  TensorFactory<ScalarType::Byte> tf_in;
  TensorFactory<ScalarType::Short> tf_min;
  TensorFactory<ScalarType::Int> tf_max;
  TensorFactory<ScalarType::Char> tf_out;

  Tensor in = tf_in.make({}, {5});
  Tensor min = tf_min.make({}, {-129});
  Tensor max = tf_max.make({}, {300});
  Tensor out = tf_out.zeros({});
  Tensor expected = tf_out.make({}, {5});

  op_clamp_tensor_out(in, min, max, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpClampTensorOutTest, DowncastingSmokeTest2) {
  TensorFactory<ScalarType::Short> tf_in;
  TensorFactory<ScalarType::Short> tf_min;
  TensorFactory<ScalarType::Int> tf_max;
  TensorFactory<ScalarType::Char> tf_out;

  Tensor in = tf_in.make({}, {301});
  Tensor min = tf_min.make({}, {-129});
  Tensor max = tf_max.make({}, {300});
  Tensor out = tf_out.zeros({});
  Tensor expected = tf_out.make({}, {44});

  op_clamp_tensor_out(in, min, max, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpClampTensorOutTest, DowncastingSmokeTest3) {
  TensorFactory<ScalarType::Short> tf_in;
  TensorFactory<ScalarType::Short> tf_min;
  TensorFactory<ScalarType::Int> tf_max;
  TensorFactory<ScalarType::Char> tf_out;

  Tensor in = tf_in.make({}, {45});
  Tensor min = tf_min.make({}, {-129});
  Tensor max = tf_max.make({}, {300});
  Tensor out = tf_out.zeros({});
  Tensor expected = tf_out.make({}, {45});

  op_clamp_tensor_out(in, min, max, out);
  EXPECT_TENSOR_EQ(out, expected);
}
