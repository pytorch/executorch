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
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using std::optional;
using torch::executor::testing::TensorFactory;

class OpAnyOutTest : public OperatorTest {
 protected:
  Tensor& op_any_all_out(const Tensor& input, Tensor& out) {
    return torch::executor::aten::any_outf(context_, input, out);
  }

  Tensor& op_any_dims_out(
      const Tensor& input,
      optional<ArrayRef<int64_t>> dim,
      bool keepdim,
      Tensor& out) {
    return torch::executor::aten::any_outf(context_, input, dim, keepdim, out);
  }

  Tensor&
  op_any_out(const Tensor& input, int64_t dim, bool keepdim, Tensor& out) {
    return torch::executor::aten::any_outf(context_, input, dim, keepdim, out);
  }

  template <ScalarType OUT_DTYPE>
  void test_any_all_out_invalid_type() {
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

    ET_EXPECT_KERNEL_FAILURE(context_, op_any_all_out(in, out));
  }

  template <ScalarType IN_DTYPE>
  void test_any_all_out() {
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

    op_any_all_out(in, out);
    EXPECT_TENSOR_EQ(out, tf_bool.make({}, {true}));

    op_any_all_out(bool_false_in, out);
    EXPECT_TENSOR_EQ(out, tf_bool.make({}, {false}));

    op_any_all_out(bool_true_in, out);
    EXPECT_TENSOR_EQ(out, tf_bool.make({}, {true}));
  }
};

TEST_F(OpAnyOutTest, MismatchedDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched dimensions";
  }
  TensorFactory<ScalarType::Float> tff;
  const std::vector<int32_t> size{2, 2};

  Tensor in = tff.make(size, {0, 0, 1, 0});
  Tensor out = tff.ones(/*size=*/{1, 1});

  ET_EXPECT_KERNEL_FAILURE(context_, op_any_all_out(in, out));
}

TEST_F(OpAnyOutTest, InvalidDtypeDies) {
#define TEST_ENTRY(ctype, dtype) \
  test_any_all_out_invalid_type<ScalarType::dtype>();
  ET_FORALL_FLOAT_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAnyOutTest, AllRealInputTypePasses) {
#define TEST_ENTRY(ctype, dtype) test_any_all_out<ScalarType::dtype>();
  ET_FORALL_REALHBBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpAnyOutTest, SmokeTestDims) {
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor self = tfBool.make({2, 3, 1}, {true, false, true, true, false, false});
  int64_t dims[3] = {0, 2};
  optional<ArrayRef<int64_t>> opt_dim_list{ArrayRef<int64_t>{dims, 2}};
  bool keepdim = true;
  Tensor out = tfBool.zeros({1, 3, 1});
  Tensor out_expected = tfBool.make({1, 3, 1}, {true, false, true});
  op_any_dims_out(self, opt_dim_list, keepdim, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpAnyOutTest, SmokeTest) {
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor self = tfBool.make({2, 3, 1}, {true, false, true, true, false, false});
  int64_t dim = 0;
  bool keepdim = false;
  Tensor out = tfBool.zeros({3, 1});
  Tensor out_expected = tfBool.make({3, 1}, {true, false, true});
  op_any_out(self, dim, keepdim, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpAnyOutTest, EmptyInput) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor x = tf.make({2, 0, 3}, {});
  optional<ArrayRef<int64_t>> dim_list = ArrayRef<int64_t>{};
  Tensor out = tfBool.make({2, 0, 3}, {});

  op_any_dims_out(x, dim_list, /*keepdim=*/true, out);
  EXPECT_TENSOR_CLOSE(out, tfBool.zeros({2, 0, 3}));

  out = tfBool.ones({2, 0, 3});
  op_any_dims_out(x, dim_list, /*keepdim=*/false, out);
  EXPECT_TENSOR_CLOSE(out, tfBool.zeros({2, 0, 3}));

  int64_t dims1[1] = {1};
  dim_list = ArrayRef<int64_t>{dims1, 1};
  out = tfBool.ones({2, 3});
  op_any_dims_out(x, dim_list, /*keepdim=*/false, out);
  EXPECT_TENSOR_CLOSE(out, tfBool.zeros({2, 3}));

  int64_t dims2[1] = {2};
  dim_list = ArrayRef<int64_t>{dims2, 1};
  out = tfBool.make({2, 0, 1}, {});
  op_any_dims_out(x, dim_list, /*keepdim=*/true, out);
  EXPECT_TENSOR_CLOSE(out, tfBool.make({2, 0, 1}, {}));
}

TEST_F(OpAnyOutTest, TestAnyDimsOutNullDimList) {
  TensorFactory<ScalarType::Int> tfInt;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor self = tfInt.make({2, 6}, {0, 2, 0, 3, 0, 1, 5, 0, 2, 0, 4, 0});
  optional<ArrayRef<int64_t>> opt_dim_list = std::nullopt;
  bool keepdim = false;
  Tensor out = tfBool.zeros({});
  Tensor out_expected = tfBool.make({}, {true});

  op_any_dims_out(self, opt_dim_list, keepdim, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpAnyOutTest, TestAnyDimsOutEmptyDimList) {
  TensorFactory<ScalarType::Int> tfInt;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor self = tfInt.make({2, 3}, {0, 2, 0, 0, 1, 5});
  int64_t dims[0] = {};
  size_t dims_size = 0;
  optional<ArrayRef<int64_t>> opt_dim_list{ArrayRef<int64_t>{dims, dims_size}};
  bool keepdim = false;
  Tensor out = tfBool.zeros({2, 3});
  Tensor out_expected =
      tfBool.make({2, 3}, {false, true, false, false, true, true});

  op_any_dims_out(self, opt_dim_list, keepdim, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
