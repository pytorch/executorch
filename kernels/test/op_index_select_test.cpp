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
#include <sys/types.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpIndexSelectOutTest : public OperatorTest {
 protected:
  Tensor& op_index_select_out(
      const Tensor& self,
      int64_t dim,
      const Tensor& index,
      Tensor& out) {
    return torch::executor::aten::index_select_outf(
        context_, self, dim, index, out);
  }

  template <class CTYPE, exec_aten::ScalarType DTYPE>
  void test_dtype() {
    TensorFactory<DTYPE> tf;
    TensorFactory<ScalarType::Long> tfl;

    // test index_select on dimension 0.

    // clang-format off
    Tensor x = tf.make(
        {3, 2, 4},
        {
          // all ones below are from x,
          // and all zeros are from y.
          // [0, :, :]
          1, 1, 1, 1, // [0, 0, :]
          0, 0, 0, 0, // [0, 1, :]

          // [1, :, :]
          1, 1, 1, 1, // [1, 0, :]
          0, 0, 0, 0, // [1, 1, :]

          // [2, :, :]
          1, 1, 1, 1, // [2, 0, :]
          0, 0, 0, 0, // [2, 1, :]
        });
    // clang-format on

    // Expected values for out_0 and ret_0 after the test are all ones(3, 4)
    // based on the above rules. So here we set the default value of out_0 as
    // zeros(3, 4) on purpose, to eliminate the influence to the final result
    // from initial value. Same for out_1 and ret_1.

    Tensor out_0 = tf.zeros({3, 1, 4});
    Tensor out_1 = tf.ones({3, 1, 4});
    Tensor index_0 = tfl.make({1}, {0});
    Tensor index_1 = tfl.make({1}, {1});
    Tensor ret_0 = op_index_select_out(x, /*dim=*/1, /*index=*/index_0, out_0);
    Tensor ret_1 = op_index_select_out(x, /*dim=*/1, /*index=*/index_1, out_1);

    EXPECT_TENSOR_EQ(ret_0, out_0);
    EXPECT_TENSOR_EQ(ret_1, out_1);

    EXPECT_TENSOR_EQ(ret_0, tf.ones({3, 1, 4}));
    EXPECT_TENSOR_EQ(ret_1, tf.zeros({3, 1, 4}));
  }

  void test_dynamic_shape(
      const std::vector<int32_t>& out_shape,
      enum torch::executor::TensorShapeDynamism dynamism) {
    /* %python
    %rewrite(index_select_template) */

    TensorFactory<ScalarType::Float> tf;
    TensorFactory<ScalarType::Long> tf_index;

    Tensor input = tf.make(
        {2, 3, 4},
        {0.49625658988952637,  0.7682217955589294,  0.08847743272781372,
         0.13203048706054688,  0.30742281675338745, 0.6340786814689636,
         0.4900934100151062,   0.8964447379112244,  0.455627977848053,
         0.6323062777519226,   0.3488934636116028,  0.40171730518341064,
         0.022325754165649414, 0.16885894536972046, 0.2938884496688843,
         0.518521785736084,    0.6976675987243652,  0.800011396408081,
         0.16102945804595947,  0.28226858377456665, 0.6816085577011108,
         0.9151939749717712,   0.39709991216659546, 0.8741558790206909});
    Tensor index = tf_index.make({2}, {0, 2});
    Tensor expected = tf.make(
        {2, 3, 2},
        {0.49625658988952637,
         0.08847743272781372,
         0.30742281675338745,
         0.4900934100151062,
         0.455627977848053,
         0.3488934636116028,
         0.022325754165649414,
         0.2938884496688843,
         0.6976675987243652,
         0.16102945804595947,
         0.6816085577011108,
         0.39709991216659546});
    Tensor out = tf.zeros(out_shape, dynamism);

    op_index_select_out(input, 2, index, out);
    EXPECT_TENSOR_CLOSE(out, expected);
  }

  // Run the test by selecting Tensor x on given dim and all available indexes
  // on that dimension
  void run_test_cases(
      const Tensor& x,
      ssize_t dim,
      const Tensor& index,
      const Tensor& expected) {
    // Generated out tensor sharing same size and dtype with expected tensor
    TensorFactory<ScalarType::Double> tf;

    const std::vector<int32_t> out_size(
        expected.sizes().begin(), expected.sizes().end());
    Tensor out = tf.ones(out_size);

    Tensor ret = op_index_select_out(x, dim, index, out);
    EXPECT_TENSOR_EQ(out, ret);
    EXPECT_TENSOR_EQ(ret, expected);
  }
};

TEST_F(OpIndexSelectOutTest, SelectFrontDimAllIndexes) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the tensor from the input at 0th dimension
  const std::vector<int32_t> out_size = {1, 3, 4};

  Tensor out = tf.zeros(out_size);
  Tensor index = tfl.make({1}, {0});
  // clang-format off
  Tensor expected = tf.make(
    out_size,
    {
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]
    }
  );
  // clang-format on

  run_test_cases(x, /*dim=*/0, /*index=*/index, expected);
}

TEST_F(OpIndexSelectOutTest, SelectMiddleDimAllIndexes) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the tensor from the input at 1st dimension
  const std::vector<int32_t> out_size = {2, 2, 4};

  Tensor out = tf.zeros(out_size);
  Tensor index = tfl.make({2}, {0, 2});
  // clang-format off
  Tensor expected = tf.make(
    out_size,
    {
          1.,   2.,   3.,   4., // [0, 0, :]
          9.,  10.,  11.,  12., // [0, 2, :]

         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -9., -10., -11., -12., // [1, 2, :]
    }
  );
  // clang-format on

  run_test_cases(x, /*dim=*/1, /*index=*/index, expected);
}

TEST_F(OpIndexSelectOutTest, SelectEndDimAllIndexes) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the tensor from the input at 0th dimension
  const std::vector<int32_t> out_size = {2, 3, 2};

  Tensor out = tf.zeros(out_size);
  Tensor index = tfl.make({2}, {0, 2});
  // clang-format off
  Tensor expected = tf.make(
    out_size,
    {
          // [0, :, :]
          1.,   3.,
          5.,   7.,
          9.,  11.,

          // [1, :, :]
         -1.,  -3.,
         -5.,  -7.,
         -9., -11.,
    }
  );
  // clang-format on
  run_test_cases(x, /*dim=*/2, /*index=*/index, expected);
}

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
TEST_F(OpIndexSelectOutTest, AllDtypesSupported) {
#define TEST_ENTRY(ctype, dtype) test_dtype<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
  // TODO: Also add tests for half, complex, quantized, and other types. Easiest
  // way to do that would be to make TensorFactory support zeros() and ones()
  // for those types.
}

//////////////////////////////////////////////////////////////////////////////
// The following tests focus on empty-size tensor and empty tensor.
// Here we first define the term:
// empty-size tensor: size is [] but do have data (e.g.tensor(5))
// empty tensor: size is not [] and the size of at least one
// dim is zero, and does not have data in it (e.g ones(1,0,2,3))

// In this test we are gonnna find if our select function support non-empty
// tensor input and empty-size tensor output.
TEST_F(OpIndexSelectOutTest, NonEmptyInputEmptyOutputWithMismatchDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;
  Tensor x = tf.make({10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor index = tfl.make({1}, {5});

  // Make an empty-size out tensor and demonstrate that it has data.
  Tensor out = tf.make({}, {0});
  EXPECT_EQ(out.numel(), 1);

  // pass the empty-size tensor to the function,
  Tensor expect = tf.make({}, {5});
  ET_EXPECT_KERNEL_FAILURE(
      context_, op_index_select_out(x, /*dim=*/0, /*index=*/index, out));
}

// This test focuses on the support for empty tensor (dim() > 0) input and empty
// tensor output
TEST_F(OpIndexSelectOutTest, EmptyInputEmptyOutputWithMatchingDimSupported) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor index = tfl.make({1}, {3});

  // Using empty tensors as input.
  Tensor x = tf.make({3, 0, 10, 3}, {});
  EXPECT_EQ(x.numel(), 0);

  // Output whose shape is appropriate for selecting along dim(2)
  Tensor out = tf.make({3, 0, 1, 3}, {});
  EXPECT_EQ(out.numel(), 0);

  Tensor ret = op_index_select_out(x, /*dim=*/2, /*index=*/index, out);
  EXPECT_EQ(ret.numel(), 0);
  // Success if it doesn't assert on the weird-shaped empty input and the
  // ret is still a empty array
}

///////////////////////////////////////////////////////////////////////

TEST_F(OpIndexSelectOutTest, DimOutOfBoundDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.ones({1, 1, 1});
  Tensor out = tf.zeros({1, 1, 1});
  Tensor index = tfl.make({1}, {0});

  // Some invalid dim values.
  const std::vector<int32_t> invalid_dims = {3, 4, 5, -4, -5, -6};
  for (ssize_t dim : invalid_dims) {
    ET_EXPECT_KERNEL_FAILURE(
        context_, op_index_select_out(x, dim, /*index=*/index, out));
  }
}

TEST_F(OpIndexSelectOutTest, MismatchedDtypesDies) {
  TensorFactory<ScalarType::Int> tf_int;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor x = tf_int.zeros({1, 2, 2});

  // Size is compatible to the output, but a mismatched dtype.
  Tensor out = tf_float.ones({1, 2, 2});
  Tensor index = tf_long.make({1}, {0});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_index_select_out(x, /*dim=*/0, /*index=*/index, out));
}

TEST_F(OpIndexSelectOutTest, OutMatchNumelLackDimAtEndDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({1, 2, 2, 1});
  Tensor index = tfl.make({1}, {0});

  // Out shares the same dtype and numel as the expected output, but a
  // mixmatched size (out.dim() should always equal to x.dim())
  Tensor out = tf.ones({1, 2, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_index_select_out(x, /*dim=*/0, /*index=*/index, out));
}

TEST_F(OpIndexSelectOutTest, OutMatchNumelExtraDimAtFrontDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({2, 2});
  Tensor index = tfl.make({1}, {0});

  // Out shares the same dtype as the expected output, but a
  // mismatched size
  Tensor out = tf.ones({1, 1, 2});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_index_select_out(x, /*dim=*/0, /*index=*/index, out));
}

TEST_F(OpIndexSelectOutTest, OutSizeMismatchDimDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle out with mismatched dimensions";
  }
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({2, 4, 7, 5});
  Tensor index = tfl.make({1}, {3});

  // Should be {2, 4, 1, 5} to match the x when calling index_select() with
  // dim 2.
  Tensor out = tf.zeros({2, 4, 7});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_index_select_out(x, /*dim=*/2, /*index=*/index, out));
}

TEST_F(OpIndexSelectOutTest, IndexWithInvalidDtypeDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Float> tff;

  Tensor x = tf.zeros({2, 4, 7, 5});
  Tensor index = tff.make({1}, {3});

  Tensor out = tf.zeros({2, 1, 7, 5});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_index_select_out(x, /*dim=*/1, /*index=*/index, out));
}

TEST_F(OpIndexSelectOutTest, IndexWithInvalidDimDies) {
  TensorFactory<ScalarType::Int> tf;
  TensorFactory<ScalarType::Long> tfl;

  Tensor x = tf.zeros({2, 4, 7, 5});
  // 2-D Tensor, will error out
  Tensor index = tfl.make({1, 1}, {3});

  Tensor out = tf.zeros({2, 1, 7, 5});

  ET_EXPECT_KERNEL_FAILURE(
      context_, op_index_select_out(x, /*dim=*/1, /*index=*/index, out));
}

#if !defined(USE_ATEN_LIB)
TEST_F(OpIndexSelectOutTest, UpperBoundOutTensor) {
  TensorFactory<ScalarType::Double> tf;
  TensorFactory<ScalarType::Long> tfl;
  // clang-format off
  Tensor x = tf.make(
      {2, 3, 4},
      {
          // [0, :, :]
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]

          // [1, :, :]
         -1.,  -2.,  -3.,  -4., // [1, 0, :]
         -5.,  -6.,  -7.,  -8., // [1, 1, :]
         -9., -10., -11., -12., // [1, 2, :]
      });
  // clang-format on

  // Try to select the tensor from the input at 0th dimension
  const std::vector<int32_t> out_size = {1, 3, 4};

  Tensor out =
      tf.zeros({2, 3, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor index = tfl.make({1}, {0});
  // clang-format off
  Tensor expected = tf.make(
    out_size,
    {
          1.,   2.,   3.,   4., // [0, 0, :]
          5.,   6.,   7.,   8., // [0, 1, :]
          9.,  10.,  11.,  12., // [0, 2, :]
    }
  );
  // clang-format on

  Tensor ret = op_index_select_out(x, 0, index, out);
  EXPECT_TENSOR_EQ(out, ret);
  EXPECT_TENSOR_EQ(ret, expected);
}
#endif

/* %python
import torch
torch.manual_seed(0)
input = torch.rand(2, 3, 4)
index = torch.tensor([0, 2])
dim = 2
expected = torch.index_select(input, dim, index)

index_select_template = f"""
  {declare_tensor_factory("ScalarType::Float", "tf")}
  {declare_tensor_factory("ScalarType::Long", "tf_index")}

  {declare_tensor_make_t("input", "tf")}
  {declare_tensor_make_t("index", "tf_index")}
  {declare_tensor_make_t("expected", "tf")}
  {declare_tensor_zeros("out_shape, dynamism", "tf", "out")}

  op_index_select_out(input, $dim$, index, out);
  EXPECT_TENSOR_CLOSE(out, expected);""" */

TEST_F(OpIndexSelectOutTest, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {2, 3, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpIndexSelectOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST_F(OpIndexSelectOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
