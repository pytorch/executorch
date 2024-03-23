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
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpLogicalNotOutTest : public OperatorTest {
 protected:
  Tensor& op_logical_not_out(const Tensor& input, Tensor& out) {
    return torch::executor::aten::logical_not_outf(context_, input, out);
  }

  template <ScalarType IN_DTYPE, ScalarType OUT_DTYPE>
  void test_logical_not_out() {
    TensorFactory<IN_DTYPE> tf_in;
    TensorFactory<OUT_DTYPE> tf_out;

    // clang-format off
    Tensor in = tf_in.make(
      {2, 4},
      {
        0, 1, 0, 1,
        1, 0, 1, 0
      });
    Tensor bool_in = tf_in.make(
      {2, 4},
      {
        false, true,  false, true,
        true,  false, true,  false,
      });
    // clang-format on

    Tensor out = tf_out.zeros({2, 4});
    Tensor bool_out = tf_out.zeros({2, 4});

    op_logical_not_out(in, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 4},
      {
        1, 0, 1, 0,
        0, 1, 0, 1
      }));
    // clang-format on

    op_logical_not_out(bool_in, out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(out, tf_out.make(
      {2, 4},
      {
        1, 0, 1, 0,
        0, 1, 0, 1
      }));
    // clang-format on

    op_logical_not_out(in, bool_out);
    // clang-format off
    EXPECT_TENSOR_CLOSE(bool_out, tf_out.make(
      {2, 4},
      {
        true,  false, true,  false,
        false, true,  false, true
      }));
    // clang-format on
  }

  template <ScalarType OUT_DTYPE>
  void test_logical_not_out_float() {
    TensorFactory<ScalarType::Float> tf_float;
    TensorFactory<OUT_DTYPE> tf_out;

    Tensor in = tf_float.make(
        {1, 4},
        {
            INFINITY,
            NAN,
            -INFINITY,
            0,
        });
    Tensor out = tf_out.zeros(/*size=*/{1, 4});

    op_logical_not_out(in, out);
    EXPECT_TENSOR_CLOSE(out, tf_out.make(/*size=*/{1, 4}, {0, 0, 0, 1}));
  }
};

TEST_F(OpLogicalNotOutTest, MismatchedDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }
  TensorFactory<ScalarType::Float> tff;
  const std::vector<int32_t> size{2, 2};

  Tensor in = tff.make(size, {0, 0, 1, 0});
  Tensor out = tff.zeros(/*size=*/{4, 1});

  ET_EXPECT_KERNEL_FAILURE(context_, op_logical_not_out(in, out));
}

TEST_F(OpLogicalNotOutTest, AllTypePasses) {
// Use a two layer switch to hanldle each possible data pair
#define TEST_KERNEL(INPUT_CTYPE, INPUT_DTYPE, OUTPUT_CTYPE, OUTPUT_DTYPE) \
  test_logical_not_out<ScalarType::INPUT_DTYPE, ScalarType::OUTPUT_DTYPE>();

#define TEST_ENTRY(INPUT_CTYPE, INPUT_DTYPE) \
  ET_FORALL_REAL_TYPES_WITH2(INPUT_CTYPE, INPUT_DTYPE, TEST_KERNEL);

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
#undef TEST_KERNEL
}

TEST_F(OpLogicalNotOutTest, FloatSpecificTest) {
// Float/double specific +/-Inf and NAN test
#define TEST_ENTRY_FLOAT_SPECIFIC_CASES(ctype, dtype) \
  test_logical_not_out_float<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY_FLOAT_SPECIFIC_CASES);
#undef TEST_ENTRY_FLOAT_SPECIFIC_CASES
}
