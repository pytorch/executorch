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
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::native::allclose_out;
using torch::executor::testing::TensorFactory;

const double default_atol{1e-08};
const double default_rtol{1e-05};

class OpAllCloseTest : public OperatorTest {
 protected:
  template <typename CTYPE, ScalarType DTYPE>
  void test_tensors_vary_tolerance(
      double rtol,
      double rdiff,
      double atol,
      double adiff,
      bool should_match) {
    TensorFactory<DTYPE> tf;
    Tensor a = tf.ones(/*sizes=*/{2, 2});
    Tensor b = tf.ones(/*sizes=*/{2, 2});

    auto a_data = a.data_ptr<CTYPE>();
    auto b_data = b.data_ptr<CTYPE>();
    b_data[0] = a_data[0] + adiff + a_data[0] * rdiff;

    TensorFactory<ScalarType::Bool> tf_bool;
    Tensor out = tf_bool.zeros(/*sizes=*/{1});

    allclose_out(
        a,
        b,
        rtol,
        atol,
        /*equal_nan=*/false,
        /*dummy_param=*/false,
        out);

    auto out_data = out.data_ptr<bool>();
    EXPECT_EQ(out_data[0], should_match)
        << a_data[0] << " doesn't match " << b_data[0] << "; dtype " << DTYPE;
  }
};

TEST_F(OpAllCloseTest, IdenticalFloatTensors) {
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

TEST_F(OpAllCloseTest, IdenticalDoubleTensors) {
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

TEST_F(OpAllCloseTest, NonEqualFloatTensors) {
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

TEST_F(OpAllCloseTest, NonEqualDoubleTensors) {
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

TEST_F(OpAllCloseTest, IdenticalIntTensors) {
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

TEST_F(OpAllCloseTest, NonEqualIntTensors) {
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

TEST_F(OpAllCloseTest, IdenticalBoolTensors) {
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

TEST_F(OpAllCloseTest, NonEqualBoolTensors) {
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

TEST_F(OpAllCloseTest, MismatchedInputShapesDeath) {
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

TEST_F(OpAllCloseTest, MismatchedInputDtypesDeath) {
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

TEST_F(OpAllCloseTest, IncorrectOutputDtypeDeath) {
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

TEST_F(OpAllCloseTest, IncorrectOutputShapeDeath) {
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

TEST_F(OpAllCloseTest, TensorsVaryWithinRelativeTolerance) {
#define TEST_ENTRY(ctype, dtype)                         \
  test_tensors_vary_tolerance<ctype, ScalarType::dtype>( \
      1e-01, 1e-02, 0, 0, true);
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAllCloseTest, TensorsVaryOutsideRelativeTolerance) {
#define TEST_ENTRY(ctype, dtype) \
  test_tensors_vary_tolerance<ctype, ScalarType::dtype>(1e-01, 1, 0, 0, false);
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAllCloseTest, TensorsVaryWithinAbsoluteTolerance) {
#define TEST_ENTRY(ctype, dtype)                         \
  test_tensors_vary_tolerance<ctype, ScalarType::dtype>( \
      0, 0, 1e-01, 1e-02, true);
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAllCloseTest, TensorsVaryOutsideAbsoluteTolerance) {
#define TEST_ENTRY(ctype, dtype) \
  test_tensors_vary_tolerance<ctype, ScalarType::dtype>(0, 0, 1e-01, 1, false);
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAllCloseTest, TensorsVaryWithZeroTolerance) {
#define TEST_ENTRY(ctype, dtype) \
  test_tensors_vary_tolerance<ctype, ScalarType::dtype>(0, 0, 0, 1e-01, false);
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}
