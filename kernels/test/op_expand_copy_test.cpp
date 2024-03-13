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
using exec_aten::IntArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpExpandOutTest : public OperatorTest {
 protected:
  Tensor& op_expand_copy_out(
      const Tensor& self,
      IntArrayRef sizes,
      bool implicit,
      Tensor& out) {
    return torch::executor::aten::expand_copy_outf(
        context_, self, sizes, implicit, out);
  }
};

TEST_F(OpExpandOutTest, NoOp) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 2});
  Tensor out = tf.zeros({2, 2});
  const std::vector<int64_t> dims{2, 2};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 2}));
}

TEST_F(OpExpandOutTest, PrependDims) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 2});
  Tensor out = tf.zeros({3, 3, 3, 2, 2});
  const std::vector<int64_t> dims{3, 3, 3, 2, 2};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);

  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({3, 3, 3, 2, 2}));
}

TEST_F(OpExpandOutTest, GrowExistingDim) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 1});
  Tensor out = tf.zeros({2, 92});

  const std::vector<int64_t> dims{2, 92};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 92}));
}

TEST_F(OpExpandOutTest, AllNegativeOnes) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 4, 12});
  Tensor out = tf.zeros({2, 4, 12});

  const std::vector<int64_t> dims{-1, -1, -1};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 4, 12}));
}

TEST_F(OpExpandOutTest, AllNegativeOnes2) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 1, 12});
  Tensor out = tf.zeros({2, 1, 12});

  const std::vector<int64_t> dims{-1, -1, -1};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 1, 12}));
}

TEST_F(OpExpandOutTest, EndsNegativeOnes) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 1, 12});
  Tensor out = tf.zeros({2, 14, 12});

  const std::vector<int64_t> dims{-1, 14, -1};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 14, 12}));
}

TEST_F(OpExpandOutTest, MoreNegativeOnes) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 14, 1});
  Tensor out = tf.zeros({2, 14, 12});

  const std::vector<int64_t> dims{-1, -1, 12};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.ones({2, 14, 12}));
}

TEST_F(OpExpandOutTest, BadExpandDimsTooSmall) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 14, 1});
  Tensor out = tf.ones({2, 14}); // undefined

  const std::vector<int64_t> dims{2};

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_expand_copy_out(a, {dims.data(), dims.size()}, false, out));
}

TEST_F(OpExpandOutTest, BadLeadingNegativeOnes) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 14, 1});
  Tensor out = tf.ones({2, 14, 1}); // undefined

  const std::vector<int64_t> dims{-1, -1, -1, -1, 2, 14, 1};

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_expand_copy_out(a, {dims.data(), dims.size()}, false, out));
}

TEST_F(OpExpandOutTest, ExpandDimsOneToN) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {2, 1}, /*data=*/{3, 3});
  Tensor out = tf.ones({2, 6});

  const std::vector<int64_t> dims{2, 6};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(
      out,
      tf.make(/*sizes*/ {2, 6}, /*data=*/{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}));
}

TEST_F(OpExpandOutTest, ExpandOneToNPlusNewDimUniform) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {2, 1}, /*data=*/{3, 3});
  Tensor out = tf.ones({2, 2, 6});

  const std::vector<int64_t> dims{2, 2, 6};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(
      out, tf.make(/*sizes*/ {2, 2, 6}, /*data=*/{3, 3, 3, 3, 3, 3, 3, 3,
                                                  3, 3, 3, 3, 3, 3, 3, 3,
                                                  3, 3, 3, 3, 3, 3, 3, 3}));
}

TEST_F(OpExpandOutTest, ExpandOneToNPlusNewDimDifferent) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {2, 1}, /*data=*/{1, 2});
  Tensor out = tf.ones({2, 2, 6});

  const std::vector<int64_t> dims{2, 2, 6};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(
      out, tf.make(/*sizes*/ {2, 2, 6}, /*data=*/{1, 1, 1, 1, 1, 1, 2, 2,
                                                  2, 2, 2, 2, 1, 1, 1, 1,
                                                  1, 1, 2, 2, 2, 2, 2, 2}));
}

TEST_F(OpExpandOutTest, ExpandOneToNPlusNewDimDifferentTwo) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {1, 2}, /*data=*/{42, 96});
  Tensor out = tf.ones({2, 6, 2});

  const std::vector<int64_t> dims{2, 6, 2};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(
      out,
      tf.make(/*sizes*/ {2, 6, 2}, /*data=*/{42, 96, 42, 96, 42, 96, 42, 96,
                                             42, 96, 42, 96, 42, 96, 42, 96,
                                             42, 96, 42, 96, 42, 96, 42, 96}));
}

TEST_F(OpExpandOutTest, BadOutDataTypeGoodShapeDeath) {
  TensorFactory<ScalarType::Int> tf_int;
  Tensor a = tf_int.make(/*sizes*/ {1, 2}, /*data=*/{42, 96});

  TensorFactory<ScalarType::Float> tf_float;
  Tensor out = tf_float.ones({2, 6, 2});

  const std::vector<int64_t> dims{2, 6, 2};

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_expand_copy_out(a, {dims.data(), dims.size()}, false, out));
}

TEST_F(OpExpandOutTest, BadOutShapeGoodDataTypeDeath) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle this";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {1, 2}, /*data=*/{42, 96});
  Tensor out = tf.ones({2, 6, 4});

  const std::vector<int64_t> dims{2, 6, 2};

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_expand_copy_out(a, {dims.data(), dims.size()}, false, out));
}

TEST_F(OpExpandOutTest, SingleToMany) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {1}, /*data=*/{42});
  Tensor out = tf.ones({4, 4, 4});

  const std::vector<int64_t> dims{4, 4, 4};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          /*sizes*/ {4, 4, 4},
          /*data=*/{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}));
}

TEST_F(OpExpandOutTest, ZeroDimInputExpand_1) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {}, /*data=*/{3});
  Tensor out = tf.ones({6});

  const std::vector<int64_t> dims{6};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make(/*sizes*/ {6}, /*data=*/{3, 3, 3, 3, 3, 3}));
}

TEST_F(OpExpandOutTest, ZeroDimInputExpand_2) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {}, /*data=*/{3});
  Tensor out = tf.ones({6, 2});

  const std::vector<int64_t> dims{6, 2};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(
      out,
      tf.make(/*sizes*/ {6, 2}, /*data=*/{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}));
}

TEST_F(OpExpandOutTest, ZeroDimInputZeroDimOutputExpand) {
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {}, /*data=*/{3});
  Tensor out = tf.ones({});

  const std::vector<int64_t> dims{};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(out, tf.make(/*sizes*/ {}, /*data=*/{3}));
}

#ifndef USE_ATEN_LIB
TEST_F(OpExpandOutTest, ResizedOutput) {
  // In this case, the output starts off with a different shape than is
  // expected. We are checking to see that dynamic shape support is working
  // correctly and that the output will be resized to the correct shape inside
  // the kernel.

  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.make(/*sizes*/ {2, 1, 1}, /*data=*/{42, 42});

  Tensor out =
      tf.zeros({2, 6, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  const std::vector<int64_t> dims{2, 3, 4};

  auto ret = op_expand_copy_out(a, {dims.data(), dims.size()}, false, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(
      out,
      tf.make(
          /*sizes*/ {2, 3, 4},
          /*data=*/{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42}));
}
#endif

TEST_F(OpExpandOutTest, ImplicitTrue) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle this";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor a = tf.ones({2, 2});
  Tensor out = tf.zeros({2, 2});
  const std::vector<int64_t> dims{2, 2};

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_expand_copy_out(a, {dims.data(), dims.size()}, true, out));
}

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(2, 1, 3)
res = x.expand(2, 5, 3)
op = "op_expand_copy_out"
opt_setup_params = """
  int64_t sizes[3] = {2, 5, 3};
  auto sizes_aref = IntArrayRef{sizes, 3};
"""
opt_extra_params = "sizes_aref, false,"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpExpandOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{2, 5, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 1, 3},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor expected = tf.make(
      {2, 5, 3},
      {0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636});

  int64_t sizes[3] = {2, 5, 3};
  auto sizes_aref = IntArrayRef{sizes, 3};

  Tensor out =
      tf.zeros({2, 5, 3}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_expand_copy_out(x, sizes_aref, false, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpExpandOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 1, 3},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor expected = tf.make(
      {2, 5, 3},
      {0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636});

  int64_t sizes[3] = {2, 5, 3};
  auto sizes_aref = IntArrayRef{sizes, 3};

  Tensor out = tf.zeros(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_expand_copy_out(x, sizes_aref, false, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpExpandOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 1, 3},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636});
  Tensor expected = tf.make(
      {2, 5, 3},
      {0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.49625658988952637, 0.7682217955589294,  0.08847743272781372,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
       0.13203048706054688, 0.30742281675338745, 0.6340786814689636});

  int64_t sizes[3] = {2, 5, 3};
  auto sizes_aref = IntArrayRef{sizes, 3};

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_expand_copy_out(x, sizes_aref, false, out);
  EXPECT_TENSOR_EQ(out, expected);
}
