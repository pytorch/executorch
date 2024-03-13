/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/NativeFunctions.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>
#include <cmath>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::native::allclose_out;
using torch::executor::testing::IsCloseTo;
using torch::executor::testing::TensorFactory;

const double default_atol{1e-08};
const double default_rtol{1e-05};

TEST(OpAllCloseTest, IdenticalFloatTensors) {
  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});
  Tensor b = tf_float.ones(/*sizes=*/{2, 2});
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a,
      b,
      default_rtol,
      default_atol,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], true);
}

TEST(OpAllCloseTest, IdenticalDoubleTensors) {
  TensorFactory<ScalarType::Double> tf_double;
  Tensor a = tf_double.ones(/*sizes=*/{2, 2});
  Tensor b = tf_double.ones(/*sizes=*/{2, 2});
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a,
      b,
      default_rtol,
      default_atol,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], true);
}

TEST(OpAllCloseTest, NonEqualFloatTensors) {
  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.make(/*sizes=*/{2, 2}, /*data=*/{1., 2., 3., 4.});
  Tensor b = tf_float.make(/*sizes=*/{2, 2}, /*data=*/{5., 6., 7., 8.});
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a,
      b,
      default_rtol,
      default_atol,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, NonEqualDoubleTensors) {
  TensorFactory<ScalarType::Double> tf_double;
  Tensor a = tf_double.make(/*sizes=*/{2, 2}, /*data=*/{1., 2., 3., 4.});
  Tensor b = tf_double.make(/*sizes=*/{2, 2}, /*data=*/{5., 6., 7., 8.});
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a,
      b,
      default_rtol,
      default_atol,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, IdenticalIntTensors) {
  TensorFactory<ScalarType::Int> tf_int;
  Tensor a = tf_int.ones(/*sizes=*/{2, 2});
  Tensor b = tf_int.ones(/*sizes=*/{2, 2});
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});
  allclose_out(
      a,
      b,
      default_rtol,
      default_atol,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], true);
}

TEST(OpAllCloseTest, NonEqualIntTensors) {
  TensorFactory<ScalarType::Int> tf_int;
  Tensor a = tf_int.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});
  Tensor b = tf_int.make(/*sizes=*/{2, 2}, /*data=*/{5, 6, 7, 8});
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});
  allclose_out(
      a,
      b,
      default_rtol,
      default_atol,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, IdenticalBoolTensors) {
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor a = tf_bool.ones(/*sizes=*/{2, 2});
  Tensor b = tf_bool.ones(/*sizes=*/{2, 2});
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a,
      b,
      default_rtol,
      default_atol,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);
  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], true);
}

TEST(OpAllCloseTest, NonEqualBoolTensors) {
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor a = tf_bool.ones(/*sizes=*/{2, 2});
  Tensor b = tf_bool.ones(/*sizes=*/{2, 2});
  auto b_data = b.data_ptr<bool>();
  b_data[0] = false;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a,
      b,
      default_rtol,
      default_atol,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);
  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, MismatchedInputShapesDeath) {
  TensorFactory<ScalarType::Int> tf_int;
  Tensor a = tf_int.ones(/*sizes=*/{2, 1});
  Tensor b = tf_int.ones(/*sizes=*/{2, 2});
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  ET_EXPECT_DEATH(
      allclose_out(
          a,
          b,
          default_rtol,
          default_atol,
          /*equal_nan=*/false,
          /*dummy_param=*/false,
          out),
      "");
}

TEST(OpAllCloseTest, MismatchedInputDtypesDeath) {
  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});

  TensorFactory<ScalarType::Int> tf_int;
  Tensor b = tf_int.ones(/*sizes=*/{2, 2});

  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  ET_EXPECT_DEATH(
      allclose_out(
          a,
          b,
          default_rtol,
          default_atol,
          /*equal_nan=*/false,
          /*dummy_param=*/false,
          out),
      "");
}

TEST(OpAllCloseTest, IncorrectOutputDtypeDeath) {
  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});
  Tensor b = tf_float.ones(/*sizes=*/{2, 2});
  Tensor out = tf_float.zeros(/*sizes=*/{1});

  ET_EXPECT_DEATH(
      allclose_out(
          a,
          b,
          default_rtol,
          default_atol,
          /*equal_nan=*/false,
          /*dummy_param=*/false,
          out),
      "");
}

TEST(OpAllCloseTest, IncorrectOutputShapeDeath) {
  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});
  Tensor b = tf_float.ones(/*sizes=*/{2, 2});
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{2, 2});

  ET_EXPECT_DEATH(
      allclose_out(
          a,
          b,
          default_rtol,
          default_atol,
          /*equal_nan=*/false,
          /*dummy_param=*/false,
          out),
      "");
}

TEST(OpAllCloseTest, FloatTensorsVaryWithinRelativeTolerance) {
  const double rtol = 1e-05;
  const double rdiff = 1e-06;

  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});
  Tensor b = tf_float.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<float>();
  auto b_data = b.data_ptr<float>();
  b_data[0] = a_data[0] * (1. + rdiff);

  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a, b, rtol, /*atol=*/0., /*equal_nan=*/false, /*dummy_param=*/false, out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], true);
}

TEST(OpAllCloseTest, DoubleTensorsVaryWithinRelativeTolerance) {
  const double rtol = 1e-05;
  const double rdiff = 1e-06;

  TensorFactory<ScalarType::Double> tf_double;
  Tensor a = tf_double.ones(/*sizes=*/{2, 2});
  Tensor b = tf_double.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<double>();
  auto b_data = b.data_ptr<double>();
  b_data[0] = a_data[0] * (1. + rdiff);
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a, b, rtol, /*atol=*/0., /*equal_nan=*/false, /*dummy_param=*/false, out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], true);
}

TEST(OpAllCloseTest, FloatTensorsVaryOutsideRelativeTolerance) {
  const double rtol = 1e-05;
  const double rdiff = 1e-04;

  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});
  Tensor b = tf_float.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<float>();
  auto b_data = b.data_ptr<float>();
  b_data[0] = a_data[0] * (1. + rdiff);
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a, b, rtol, /*atol=*/0., /*equal_nan=*/false, /*dummy_param=*/false, out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, DoubleTensorsVaryOutsideRelativeTolerance) {
  const double rtol = 1e-05;
  const double rdiff = 1e-04;

  TensorFactory<ScalarType::Double> tf_double;
  Tensor a = tf_double.ones(/*sizes=*/{2, 2});
  Tensor b = tf_double.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<double>();
  auto b_data = b.data_ptr<double>();
  b_data[0] = a_data[0] * (1. + rdiff);
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a, b, rtol, /*atol=*/0., /*equal_nan=*/false, /*dummy_param=*/false, out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, FloatTensorsVaryWithinAbsoluteTolerance) {
  const double atol = 1e-08;
  const double adiff = 1e-09;

  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});
  Tensor b = tf_float.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<float>();
  auto b_data = b.data_ptr<float>();
  b_data[0] = a_data[0] + adiff;
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a, b, /*rtol=*/0., atol, /*equal_nan=*/false, /*dummy_param=*/false, out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], true);
}

TEST(OpAllCloseTest, DoubleTensorsVaryWithinAbsoluteTolerance) {
  const double atol = 1e-08;
  const double adiff = 1e-09;

  TensorFactory<ScalarType::Double> tf_double;
  Tensor a = tf_double.ones(/*sizes=*/{2, 2});
  Tensor b = tf_double.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<double>();
  auto b_data = b.data_ptr<double>();
  b_data[0] = a_data[0] + adiff;
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a, b, /*rtol=*/0., atol, /*equal_nan=*/false, /*dummy_param=*/false, out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], true);
}

TEST(OpAllCloseTest, FloatTensorsVaryOutsideAbsoluteTolerance) {
  const double atol = 1e-08;
  const double adiff = 1e-07;

  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});
  Tensor b = tf_float.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<float>();
  auto b_data = b.data_ptr<float>();
  b_data[0] = a_data[0] + adiff;
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a, b, /*rtol=*/0., atol, /*equal_nan=*/false, /*dummy_param=*/false, out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, DoubleTensorsVaryOutsideAbsoluteTolerance) {
  const double atol = 1e-08;
  const double adiff = 1e-07;

  TensorFactory<ScalarType::Float> tf_double;
  Tensor a = tf_double.ones(/*sizes=*/{2, 2});
  Tensor b = tf_double.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<double>();
  auto b_data = b.data_ptr<double>();
  b_data[0] = a_data[0] + adiff;
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a, b, /*rtol=*/0., atol, /*equal_nan=*/false, /*dummy_param=*/false, out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, FloatTensorsVaryWithZeroTolerance) {
  TensorFactory<ScalarType::Float> tf_float;
  Tensor a = tf_float.ones(/*sizes=*/{2, 2});
  Tensor b = tf_float.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<float>();
  auto b_data = b.data_ptr<float>();
  b_data[0] = a_data[0] + 1e-07;
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a,
      b,
      /*rtol=*/0.,
      /*atol=*/0,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}

TEST(OpAllCloseTest, DoubleTensorsVaryWithZeroTolerance) {
  TensorFactory<ScalarType::Double> tf_double;
  Tensor a = tf_double.ones(/*sizes=*/{2, 2});
  Tensor b = tf_double.ones(/*sizes=*/{2, 2});

  auto a_data = a.data_ptr<double>();
  auto b_data = b.data_ptr<double>();
  b_data[0] = a_data[0] + 1e-09;
  TensorFactory<ScalarType::Bool> tf_bool;
  Tensor out = tf_bool.zeros(/*sizes=*/{1});

  allclose_out(
      a,
      b,
      /*rtol=*/0.,
      /*atol=*/0.,
      /*equal_nan=*/false,
      /*dummy_param=*/false,
      out);

  auto out_data = out.data_ptr<bool>();
  EXPECT_EQ(out_data[0], false);
}
