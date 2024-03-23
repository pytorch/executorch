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
using exec_aten::IntArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::SupportedFeatures;
using torch::executor::testing::TensorFactory;

class OpConstantPadNDOutTest : public OperatorTest {
 protected:
  Tensor& op_constant_pad_nd_out(
      const Tensor& self,
      const IntArrayRef padding,
      const Scalar& value,
      Tensor& out) {
    return torch::executor::aten::constant_pad_nd_outf(
        context_, self, padding, value, out);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim2() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {2, 4, 6};
    const std::vector<int64_t> padding = {1, 1};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
  
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  1,  2,  3,  4,  7,
           7,  5,  6,  7,  8,  7,
           7,  1,  2,  3,  4,  7,
           7,  5,  6,  7,  8,  7,
  
           7,  1,  2,  3,  4,  7,
           7,  5,  6,  7,  8,  7,
           7,  1,  2,  3,  4,  7,
           7,  5,  6,  7,  8,  7,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim1() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {2, 6, 4};
    const std::vector<int64_t> padding = {0, 0, 2, 0};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
  
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  7,  7,
           7,  7,  7,  7,
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
  
           7,  7,  7,  7,
           7,  7,  7,  7,
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim0() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {3, 4, 4};
    const std::vector<int64_t> padding = {0, 0, 0, 0, 1, 0};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
  
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  7,  7,
           7,  7,  7,  7,
           7,  7,  7,  7,
           7,  7,  7,  7,
  
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
  
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim12() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {2, 6, 7};
    const std::vector<int64_t> padding = {2, 1, 0, 2};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
  
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
  
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim02() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {3, 4, 7};
    const std::vector<int64_t> padding = {2, 1, 0, 0, 0, 1};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
  
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
  
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
  
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  template <ScalarType DTYPE>
  void test_constant_pad_nd_out_dim012() {
    TensorFactory<DTYPE> tf;

    const std::vector<int32_t> sizes = {2, 4, 4};
    const std::vector<int32_t> sizes_out = {3, 5, 7};
    const std::vector<int64_t> padding = {2, 1, 1, 0, 0, 1};

    // clang-format off
    Tensor self = tf.make(
        sizes,
        {
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
  
           1,  2,  3,  4,
           5,  6,  7,  8,
           1,  2,  3,  4,
           5,  6,  7,  8,
        });
    // clang-format on

    // clang-format off
    Tensor expected = tf.make(
        sizes_out,
        {
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
  
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
           7,  7,  1,  2,  3,  4,  7,
           7,  7,  5,  6,  7,  8,  7,
  
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
           7,  7,  7,  7,  7,  7,  7,
        });
    // clang-format on

    IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());
    Tensor out = tf.zeros(sizes_out);

    // Valid input should give the expected output
    op_constant_pad_nd_out(self, padding_ref, 7, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }
};

TEST_F(OpConstantPadNDOutTest, TestPadDim2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim2<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim1) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim1<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim0) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim0<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim1And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim12<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim0And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim02<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, TestPadDim0And1And2) {
#define TEST_ENTRY(ctype, dtype) \
  test_constant_pad_nd_out_dim012<ScalarType::dtype>();

  ET_FORALL_REAL_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpConstantPadNDOutTest, DifferentInputOutputTypesFail) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Double> tf_out;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 6};
  const std::vector<int64_t> padding = {1, 1};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf_out.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST_F(OpConstantPadNDOutTest, OddNumberOfPaddingElementsFail) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {1, 1, 0};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST_F(OpConstantPadNDOutTest, TooManyPaddingElementsFail) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {3, 2, 1, 1, 2, 1, 1, 0};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_constant_pad_nd_out(self, padding_ref, 0, out));
}

TEST_F(OpConstantPadNDOutTest, IncorrectOutputShapeFail) {
  if (SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle reshape output";
  }

  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> sizes = {1, 4, 4};
  const std::vector<int32_t> sizes_out = {1, 4, 4};
  const std::vector<int64_t> padding = {1, 1};

  IntArrayRef padding_ref = IntArrayRef(padding.data(), padding.size());

  Tensor self = tf.ones(sizes);
  Tensor out = tf.zeros(sizes_out);

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_constant_pad_nd_out(self, padding_ref, 0, out));
}
