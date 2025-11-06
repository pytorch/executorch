/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpNativeDropoutTest : public OperatorTest {
 protected:
  void op_native_dropout_out(
      const Tensor& self,
      double prob,
      executorch::aten::optional<bool> train,
      Tensor& out,
      Tensor& mask) {
    torch::executor::aten::native_dropout_outf(
        context_, self, prob, train, out, mask);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void test_dropout() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Bool> tf_bool;
    const std::vector<int32_t> sizes = {3, 2};
    Tensor in = tf.make(sizes, {1, 2, 3, 4, 5, 6});
    Tensor out = tf.zeros(sizes);
    Tensor mask = tf_bool.zeros(sizes);

    bool* const mask_data = mask.mutable_data_ptr<bool>();
    auto expect_no_drops = [&]() {
      EXPECT_TENSOR_CLOSE(out, in);
      for (const auto ii : c10::irange(mask.numel())) {
        EXPECT_TRUE(mask_data[ii]);
        mask_data[ii] = false;
      }
    };

    op_native_dropout_out(in, 0, true, out, mask);
    expect_no_drops();

    op_native_dropout_out(in, 0, false, out, mask);
    expect_no_drops();

    op_native_dropout_out(in, 1, false, out, mask);
    expect_no_drops();

    op_native_dropout_out(in, 1, true, out, mask);
    auto* const out_data = out.mutable_data_ptr<CTYPE>();
    for (const auto ii : c10::irange(out.numel())) {
      EXPECT_EQ(out_data[ii], CTYPE(0));
    }
    for (const auto ii : c10::irange(mask.numel())) {
      EXPECT_FALSE(mask_data[ii]);
      mask_data[ii] = 0;
    }
  }
};

TEST_F(OpNativeDropoutTest, Basic) {
#define TEST_ENTRY(ctype, dtype) test_dropout<ctype, ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpNativeDropoutTest, ProbabilityRangeCheck) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Bool> tf_bool;
  const std::vector<int32_t> sizes = {2, 3};
  Tensor a = tf_float.ones(sizes);
  Tensor out = tf_float.zeros(sizes);
  Tensor mask = tf_bool.zeros(sizes);
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_native_dropout_out(a, -1, true, out, mask));
}

TEST_F(OpNativeDropoutTest, MaskBoolCheck) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Byte> tf_byte;
  const std::vector<int32_t> sizes = {2, 3};
  Tensor a = tf_float.ones(sizes);
  Tensor out = tf_float.zeros(sizes);
  Tensor mask_byte = tf_byte.zeros(sizes);
  Tensor mask_float = tf_float.zeros(sizes);
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_native_dropout_out(a, 0.5, true, out, mask_byte));
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_native_dropout_out(a, 0.5, true, out, mask_float));
}
