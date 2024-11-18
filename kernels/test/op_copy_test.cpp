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
using exec_aten::MemoryFormat;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpCopyTest : public OperatorTest {
 protected:
  Tensor& op_copy_out(
      const Tensor& self,
      const Tensor& src,
      bool non_blocking,
      Tensor& out) {
    return torch::executor::aten::copy_outf(
        context_, self, src, non_blocking, out);
  }

  // test if copy.out works well under all kinds of legal input type.
  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    Tensor self = tf.make(/*sizes=*/{2, 4}, /*data=*/{2, 3, 2, 4, 1, 5, 1, 6});
    Tensor src = tf.make(/*sizes=*/{2, 4}, /*data=*/{2, 3, 2, 4, 1, 5, 1, 6});
    bool non_blocking = false;
    Tensor out_nullopt = tf.zeros(/*sizes=*/{2, 4});
    Tensor out_contiguous = tf.zeros(/*sizes=*/{2, 4});

    // we only support contiguous memory, the memory type shall be either
    // nullopt or MemoryFormat::Contiguous.
    Tensor out_nullopt_ret = op_copy_out(
        /*self=*/self,
        /*src=*/src,
        /*non_blocking=*/non_blocking,
        /*out=*/out_nullopt);
    Tensor out_contiguous_ret = op_copy_out(
        /*self=*/self,
        /*src=*/src,
        /*non_blocking=*/non_blocking,
        /*out=*/out_contiguous);

    // The original tensor a should share same value with the out variable and
    // return variable of copy function
    EXPECT_TENSOR_EQ(src, out_nullopt);
    EXPECT_TENSOR_EQ(src, out_nullopt_ret);

    EXPECT_TENSOR_EQ(src, out_contiguous);
    EXPECT_TENSOR_EQ(src, out_contiguous_ret);
  }

  template <class CTYPE, ScalarType DTYPE>
  void test_empty_input() {
    TensorFactory<DTYPE> tf;
    Tensor self = tf.make(/*sizes=*/{3, 0, 1, 2}, /*data=*/{});
    Tensor src = tf.make(/*sizes=*/{3, 0, 1, 2}, /*data=*/{});
    bool non_blocking = false;
    Tensor out = tf.zeros({3, 0, 1, 2});
    op_copy_out(self, src, non_blocking, out);
    // check a and out share same value, but are different object
    EXPECT_TENSOR_EQ(src, out);
  }

  /* %python
  import torch
  torch.manual_seed(0)
  self = torch.randint(10, (3, 4))
  src = torch.randint(10, (3, 4))
  non_blocking = False
  expected = src
  out_args = "out_shape, dynamism"

  copy_template = f"""
    {declare_tensor_factory("ScalarType::Int", "tf")}

    {declare_tensor_make_t("self", "tf")}
    {declare_tensor_make_t("src", "tf")}
    {declare_tensor_make_t("expected", "tf")}
    {declare_tensor_zeros("out_shape, dynamism", "tf", "out")}

    op_copy_out(self, src, $non_blocking$, out);
    EXPECT_TENSOR_EQ(out, expected);""" */

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(copy_template) */

    TensorFactory<ScalarType::Int> tf;

    Tensor self = tf.make({3, 4}, {4, 9, 3, 0, 3, 9, 7, 3, 7, 3, 1, 6});
    Tensor src = tf.make({3, 4}, {6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});
    Tensor expected = tf.make({3, 4}, {6, 9, 8, 6, 6, 8, 4, 3, 6, 9, 1, 4});
    Tensor out = tf.zeros(out_shape, dynamism);

    op_copy_out(self, src, false, out);
    EXPECT_TENSOR_EQ(out, expected);
  }
};

class OpCopyInplaceTest : public OperatorTest {
 protected:
  Tensor& op_copy_(Tensor& self, const Tensor& src, bool non_blocking) {
    return torch::executor::aten::copy_(context_, self, src, non_blocking);
  }
};

// regular test for copy.out
TEST_F(OpCopyTest, AllRealDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpCopyTest, EmptyInputSupported) {
#define TEST_ENTRY(ctype, dtype) test_empty_input<ctype, ScalarType::dtype>();
  ET_FORALL_REALHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpCopyTest, BroadCastSrcSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});
  Tensor src = tf.make(/*sizes=*/{1, 2}, /*data=*/{3, 3});
  bool non_blocking = false;
  Tensor out = tf.zeros({2, 2});
  op_copy_out(self, src, non_blocking, out);
  Tensor out_expected = tf.make(/*sizes=*/{2, 2}, /*data=*/{3, 3, 3, 3});
  EXPECT_TENSOR_EQ(out, out_expected);
}

TEST_F(OpCopyTest, BroadCastSrcMissingDimSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});
  Tensor src = tf.make(/*sizes=*/{1, 2}, /*data=*/{3, 3});
  bool non_blocking = false;
  Tensor out = tf.zeros({2, 2});
  op_copy_out(self, src, non_blocking, out);
  Tensor out_expected = tf.make(/*sizes=*/{2, 2}, /*data=*/{3, 3, 3, 3});
  EXPECT_TENSOR_EQ(out, out_expected);
}

TEST_F(OpCopyTest, BroadCastSelfcSupportedDie) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make(/*sizes=*/{1, 2}, /*data=*/{3, 3});
  Tensor src = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});
  bool non_blocking = false;
  Tensor out = tf.zeros({2, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, op_copy_out(self, src, non_blocking, out));
}

TEST_F(OpCopyTest, MismatchSelfSrcTypeSupported) {
  TensorFactory<ScalarType::Int> tf_self;
  TensorFactory<ScalarType::Float> tf_src;
  Tensor self =
      tf_self.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor src = tf_src.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf_src.zeros({3, 0, 1, 2});
  bool non_blocking = false;
  ET_EXPECT_KERNEL_FAILURE(context_, op_copy_out(self, src, non_blocking, out));
}

#ifndef USE_ATEN_LIB
TEST_F(OpCopyTest, ResizeOutSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor src = tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf.zeros(
      {4, 2, 2, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  bool non_blocking = false;
  op_copy_out(self, src, non_blocking, out);
  Tensor out_expected =
      tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  EXPECT_TENSOR_EQ(out, out_expected);
}

TEST_F(OpCopyTest, ResizeOutDie) {
  TensorFactory<ScalarType::Int> tf_self;
  TensorFactory<ScalarType::Float> tf_src;
  Tensor self =
      tf_self.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor src = tf_src.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf_src.zeros(
      {3, 2, 0}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  bool non_blocking = false;
  ET_EXPECT_KERNEL_FAILURE(context_, op_copy_out(self, src, non_blocking, out));
}
#endif

TEST_F(OpCopyTest, MismatchedSizesDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched sizes";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor self = tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor src = tf.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  bool non_blocking = false;
  Tensor out = tf.zeros({3, 2, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, op_copy_out(self, src, non_blocking, out));
}

TEST_F(OpCopyTest, MismatchedSrcOutTypesDie) {
  TensorFactory<ScalarType::Int> tf_in;
  TensorFactory<ScalarType::Float> tf_out;
  Tensor self = tf_in.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor src = tf_in.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  bool non_blocking = false;
  Tensor out = tf_out.zeros({3, 1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, op_copy_out(self, src, non_blocking, out));
}

// Only contiguous memory is supported, the memory type other than nullopt or
// MemoryFormat::Contiguous should not be allowed. The function is expected
// depth if using the illegal memory format.
TEST_F(OpCopyTest, BlockingDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle non-contiguous memory formats";
  }
  TensorFactory<ScalarType::Float> tf_in;
  TensorFactory<ScalarType::Float> tf_out;
  Tensor self = tf_in.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor src = tf_in.make(/*sizes=*/{3, 1, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  bool non_blocking = true;
  Tensor out = tf_out.zeros({3, 1, 1, 2});
  ET_EXPECT_KERNEL_FAILURE(context_, op_copy_out(self, src, non_blocking, out));
}

TEST_F(OpCopyTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpCopyTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpCopyTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}

TEST_F(OpCopyInplaceTest, SmokeTest) {
  TensorFactory<ScalarType::Int> tf;
  Tensor in = tf.zeros({2, 2});
  Tensor src = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});
  bool non_blocking = false;
  op_copy_(in, src, non_blocking);
  Tensor expected = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});
  EXPECT_TENSOR_EQ(in, expected);
}

TEST_F(OpCopyInplaceTest, BroadCastSrcSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor in = tf.make(/*sizes=*/{2, 2}, /*data=*/{1, 2, 3, 4});
  Tensor src = tf.make(/*sizes=*/{1, 2}, /*data=*/{3, 3});
  bool non_blocking = false;
  op_copy_(in, src, non_blocking);
  Tensor expected = tf.make(/*sizes=*/{2, 2}, /*data=*/{3, 3, 3, 3});
  EXPECT_TENSOR_EQ(in, expected);
}
