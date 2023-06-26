// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/kernel_types/kernel_types.h>
#include <executorch/core/kernel_types/testing/TensorFactory.h>
#include <executorch/core/kernel_types/testing/TensorUtil.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& _any_out(const Tensor& input, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::any_outf(context, input, out);
}

TEST(OpAnyOutTest, MismatchedDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }
  TensorFactory<ScalarType::Float> tff;
  const std::vector<int32_t> size{2, 2};

  Tensor in = tff.make(size, {0, 0, 1, 0});
  Tensor out = tff.ones(/*size=*/{1, 1});

  ET_EXPECT_KERNEL_FAILURE(_any_out(in, out));
}

template <ScalarType OUT_DTYPE>
void test_any_out_invalid_type() {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<OUT_DTYPE> tf_out;

  Tensor in = tf_float.make(
      {1, 4},
      {
          0,
          0,
          1,
          0,
      });
  Tensor out = tf_out.zeros(/*size=*/{0});

  ET_EXPECT_KERNEL_FAILURE(_any_out(in, out));
}

TEST(OpAnyOutTest, InvalidDtypeDies) {
#define TEST_ENTRY(ctype, dtype) test_any_out_invalid_type<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

template <ScalarType IN_DTYPE>
void test_any_out() {
  TensorFactory<IN_DTYPE> tf_in;
  TensorFactory<ScalarType::Bool> tf_bool;
  // clang-format off
  Tensor in = tf_in.make(
    {2, 4},
    {
      0, 1, 0, 1,
      1, 0, 1, 0
    });
  Tensor bool_false_in = tf_bool.make(
    {2, 4},
    {
      false, false, false, false,
      false, false, false, false,
    });
  Tensor bool_true_in = tf_bool.make(
    {2, 4},
    {
      true, true, true, true,
      true, true, true, true,
    });
  // clang-format on

  Tensor out = tf_bool.make({}, {false});

  _any_out(in, out);
  EXPECT_TENSOR_EQ(out, tf_bool.make({}, {true}));

  _any_out(bool_false_in, out);
  EXPECT_TENSOR_EQ(out, tf_bool.make({}, {false}));

  _any_out(bool_true_in, out);
  EXPECT_TENSOR_EQ(out, tf_bool.make({}, {true}));
}

TEST(OpAnyOutTest, AllRealInputTypePasses) {
#define TEST_ENTRY(ctype, dtype) test_any_out<ScalarType::dtype>();
  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}
