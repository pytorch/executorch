/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#include <executorch/kernels/test/TestUtil.h>

#include <gtest/gtest.h>
#include <cstdio>

using namespace ::testing;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpUnsqueezeTest : public OperatorTest {
 protected:
  Tensor& op_unsqueeze_copy_out(const Tensor& self, int64_t dim, Tensor& out) {
    return torch::executor::aten::unsqueeze_copy_outf(context_, self, dim, out);
  }

  template <class CTYPE, ScalarType DTYPE>
  void run_unsqueeze_test_cases(
      const Tensor& input,
      const std::vector<int64_t>& dims) {
    TensorFactory<DTYPE> tf;

    // DEBUG
    et_pal_init();

    for (int64_t dim : dims) {
      std::vector<int32_t> size_out = generate_size_out(input.sizes(), dim);
      Tensor out = tf.ones(size_out);
      Tensor ret = op_unsqueeze_copy_out(input, dim, out);

      // The following is just a check against itself.
      EXPECT_TENSOR_EQ(out, ret);
      EXPECT_TENSOR_DATA_EQ(input, out);
    }
  }

  // test if op_unsqueeze_copy_out works well under all kinds of legal input
  // type.
  template <class CTYPE, ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    Tensor input = tf.make(/*sizes=*/{2, 4}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});

    // All valid dims given the shape of the input
    // Legal dim for unsqueeze should be in [-(input.dim()+1), input.dim()]
    // Here input.dim == 2, so the range of legal dim for unsqueeze is [-3, 2]
    std::vector<int64_t> dims = {-3, -2, -1, 0, 1, 2};

    run_unsqueeze_test_cases<CTYPE, DTYPE>(input, dims);
  }

  template <class CTYPE, ScalarType DTYPE>
  void test_empty_input() {
    TensorFactory<DTYPE> tf;
    Tensor input = tf.make(/*sizes=*/{3, 0, 1, 2}, /*data=*/{});

    // All valid dims given the shape of the input
    // Legal dim for unsqueeze should be in [-(input.dim()+1), input.dim()]
    // Here input.dim == 4, so the range of legal dim for unsqueeze is [-5, 4]
    std::vector<int64_t> dims = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

    run_unsqueeze_test_cases<CTYPE, DTYPE>(input, dims);
  }

  // generate size of output based on input size and dim to be unsqueezed on.
  std::vector<int32_t> generate_size_out(
#ifdef USE_ATEN_LIB
      const c10::IntArrayRef& size_in,
#else
      const exec_aten::ArrayRef<int32_t>& size_in,
#endif
      int64_t dim) {
    std::vector<int32_t> size_out(size_in.size() + 1);

    // Support python-style negative indexing.
    if (dim < 0) {
      // Since we do not have out.dim() directly, calculate it from the input.
      dim += size_in.size() + 1;
    }
    EXPECT_GE(dim, 0);
    EXPECT_LT(dim, size_in.size() + 1);

    for (int32_t i = 0; i <= size_in.size(); i++) {
      if (i < dim) {
        size_out[i] = size_in[i];
      } else if (i > dim) {
        size_out[i] = size_in[i - 1];
      } else { // i == dim
        size_out[dim] = 1;
      }
    }

    return size_out;
  }
};

// regular test for op_unsqueeze_copy_out
TEST_F(OpUnsqueezeTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpUnsqueezeTest, EmptyInputSupported) {
#define TEST_ENTRY(ctype, dtype) test_empty_input<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpUnsqueezeTest, InputOutputMismatchedSizesDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched sizes";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor input = tf.make(/*sizes=*/{3, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  int64_t dim = 1;

  // unsqueese input on dim 1 should get tensor(3, 1, 1, 2)
  Tensor out = tf.ones(/*sizes=*/{3, 1, 1, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, op_unsqueeze_copy_out(input, dim, out));
  out = tf.ones(/*sizes=*/{3, 1, 1, 2, 1});
  ET_EXPECT_KERNEL_FAILURE(context_, op_unsqueeze_copy_out(input, dim, out));
}

TEST_F(OpUnsqueezeTest, DimOutputMismatchedSizesDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle mismatched sizes";
  }
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*sizes=*/{3, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf.ones(/*sizes=*/{3, 1, 2, 1});
  int64_t dim = 2;

  // The size of output should be [3,1,1,2], not [3,1,2,1], since dim=2 not 3
  ET_EXPECT_KERNEL_FAILURE(context_, op_unsqueeze_copy_out(input, dim, out));
}

TEST_F(OpUnsqueezeTest, MismatchedTypesDie) {
  TensorFactory<ScalarType::Int> tf_in;
  TensorFactory<ScalarType::Double> tf_out;
  Tensor input = tf_in.make(/*sizes=*/{3, 1, 2}, /*data=*/{1, 2, 3, 4, 5, 6});
  Tensor out = tf_out.ones(/*sizes=*/{3, 1, 2, 1});
  int64_t dim = 3;

  ET_EXPECT_KERNEL_FAILURE(context_, op_unsqueeze_copy_out(input, dim, out));
}

TEST_F(OpUnsqueezeTest, DimOutOfRangeDies) {
  TensorFactory<ScalarType::Int> tf;
  Tensor input = tf.make(/*sizes=*/{1, 1, 1}, /*data=*/{1});
  Tensor out = tf.ones(/*sizes=*/{1, 1, 1, 1});

  // Legal dim for unsqueeze should be in [-(input.dim()+1), input.dim()]
  // Here input.dim == 3, so the range of legal dim for unsqueeze is [-4, 3]
  std::vector<int64_t> illegal_dims = {
      -10, -9, -8, -7, -6, -5, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int64_t> legal_dims = {-4, -3, -2, -1, 0, 1, 2, 3};

  for (auto dim : legal_dims) {
    op_unsqueeze_copy_out(input, dim, out);
  }

  for (auto dim : illegal_dims) {
    ET_LOG(Info, "Checking dim %ld", dim);
    ET_EXPECT_KERNEL_FAILURE(context_, op_unsqueeze_copy_out(input, dim, out));
  }
}

#ifndef USE_ATEN_LIB
TEST_F(OpUnsqueezeTest, UpperBoundOutTensor) {
  TensorFactory<ScalarType::Float> tf;
  Tensor input = tf.make(/*sizes=*/{2, 4}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  Tensor out =
      tf.zeros({3, 4, 6}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  // All valid dims given the shape of the input
  // Legal dim for unsqueeze should be in [-(input.dim()+1), input.dim()]
  // Here input.dim == 2, so the range of legal dim for unsqueeze is [-3, 2]
  Tensor ref_out =
      tf.make(/*sizes=*/{1, 2, 4}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  op_unsqueeze_copy_out(input, -3, out);
  EXPECT_TENSOR_EQ(out, ref_out);

  ref_out = tf.make(/*sizes=*/{2, 1, 4}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  op_unsqueeze_copy_out(input, -2, out);
  EXPECT_TENSOR_EQ(out, ref_out);

  ref_out = tf.make(/*sizes=*/{2, 4, 1}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  op_unsqueeze_copy_out(input, -1, out);
  EXPECT_TENSOR_EQ(out, ref_out);

  ref_out = tf.make(/*sizes=*/{1, 2, 4}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  op_unsqueeze_copy_out(input, 0, out);
  EXPECT_TENSOR_EQ(out, ref_out);

  ref_out = tf.make(/*sizes=*/{2, 1, 4}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  op_unsqueeze_copy_out(input, 1, out);
  EXPECT_TENSOR_EQ(out, ref_out);

  ref_out = tf.make(/*sizes=*/{2, 4, 1}, /*data=*/{0, 1, 1, 1, 0, 1, 0, 1});
  op_unsqueeze_copy_out(input, 2, out);
  EXPECT_TENSOR_EQ(out, ref_out);
}
#endif

/* %python
import torch
torch.manual_seed(0)
x = torch.rand(2, 4)
res = torch.unsqueeze(x, 1)
op = "op_unsqueeze_copy_out"
opt_extra_params = "1,"
dtype = "ScalarType::Float"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpUnsqueezeTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{2, 1, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});
  Tensor expected = tf.make(
      {2, 1, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});

  Tensor out =
      tf.zeros({2, 1, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_unsqueeze_copy_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUnsqueezeTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});
  Tensor expected = tf.make(
      {2, 1, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});

  Tensor out =
      tf.zeros({5, 5, 5}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_unsqueeze_copy_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpUnsqueezeTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Float> tf;

  Tensor x = tf.make(
      {2, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});
  Tensor expected = tf.make(
      {2, 1, 4},
      {0.49625658988952637,
       0.7682217955589294,
       0.08847743272781372,
       0.13203048706054688,
       0.30742281675338745,
       0.6340786814689636,
       0.4900934100151062,
       0.8964447379112244});

  Tensor out = tf.zeros(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_unsqueeze_copy_out(x, 1, out);
  EXPECT_TENSOR_EQ(out, expected);
}
