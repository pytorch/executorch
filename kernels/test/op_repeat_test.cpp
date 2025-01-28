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

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::IntArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpRepeatOutTest : public OperatorTest {
 protected:
  Tensor& op_repeat_out(const Tensor& self, IntArrayRef repeats, Tensor& out) {
    return torch::executor::aten::repeat_outf(context_, self, repeats, out);
  }

  template <typename CTYPE, ScalarType DTYPE>
  void run_dtype_tests() {
    TensorFactory<DTYPE> tf;
    // clang-format off
    Tensor x = tf.make(
        /*size=*/{2, 2},
        /*data=*/{
                    0, 1,
                    2, 3,
                  });
    std::vector<int64_t> repeats_vec = {3, 3, 3};
    exec_aten::ArrayRef<int64_t> repeats = {repeats_vec.data(), repeats_vec.size()};
    // clang-format on

    // Output tensor with the shape of the input tensor x repeated
    // - Its dimension shall equal to the length of repeat.
    // - For any dimension i ∈ [repeat.size()-x.dim(), repeat.size()),
    // out.size(i) = x.size(i) * repeat[i]
    // - For any dimension i ∈ [0, repeat.size()), out.size(i) = repeat[i]
    Tensor out = tf.zeros({3, 6, 6});

    // clang-format off
    // Repeat the input tensor along the specified `repeat` dimensions.
    Tensor expected = tf.make(
        /*sizes=*/ {3, 6, 6},
        /*data=*/
        {
          //[0, :, :]
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
  
          //[1, :, :]
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
  
          //[2, :, :]
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
          0, 1, 0, 1, 0, 1,
          2, 3, 2, 3, 2, 3,
        });
    // clang-format on

    Tensor ret = op_repeat_out(x, repeats, out);
    EXPECT_TENSOR_EQ(ret, out);
    EXPECT_TENSOR_EQ(ret, expected);
  }
};

TEST_F(OpRepeatOutTest, AllDtypesSupported) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel test fails";
  }
#define TEST_ENTRY(ctype, dtype) run_dtype_tests<ctype, ScalarType::dtype>();
  ET_FORALL_REAL_TYPES_AND(Bool, TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpRepeatOutTest, EmptyInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make(
      /*sizes=*/{3, 0, 2}, /*data=*/{});

  std::vector<int64_t> repeats_vec = {3, 4, 5, 6};
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  Tensor out = tf.ones(/*sizes=*/{3, 12, 0, 12});
  Tensor expected = tf.make(/*sizes=*/{3, 12, 0, 12}, /*data=*/{});

  Tensor ret = op_repeat_out(x, repeats, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expected);
}

TEST_F(OpRepeatOutTest, ZeroDimInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make(
      /*sizes=*/{}, /*data=*/{5});

  std::vector<int64_t> repeats_vec = {3, 4};
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  Tensor out = tf.ones(/*sizes=*/{3, 4});

  // clang-format off
  Tensor expected = tf.make(
    /*sizes=*/{3, 4},
    /*data=*/
    {
      5, 5, 5, 5,
      5, 5, 5, 5,
      5, 5, 5, 5,
    });
  // clang-format on

  Tensor ret = op_repeat_out(x, repeats, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expected);
}

TEST_F(OpRepeatOutTest, ZeroRepeatRegularInputSupported) {
  TensorFactory<ScalarType::Int> tf;
  Tensor x = tf.make(
      /*sizes=*/{3, 2}, /*data=*/{0, 1, 2, 3, 4, 5});

  std::vector<int64_t> repeats_vec = {3, 0, 6};
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  Tensor out = tf.ones(/*sizes=*/{3, 0, 12});
  Tensor expected = tf.make(/*sizes=*/{3, 0, 12}, /*data=*/{});

  Tensor ret = op_repeat_out(x, repeats, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expected);
}

TEST_F(OpRepeatOutTest, ZeroRepeatZeroDimInputSupported) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make(
      /*sizes=*/{}, /*data=*/{5});

  std::vector<int64_t> repeats_vec = {3, 0, 6};
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  Tensor out = tf.ones(/*sizes=*/{3, 0, 6});
  Tensor expected = tf.make(/*sizes=*/{3, 0, 6}, /*data=*/{});

  Tensor ret = op_repeat_out(x, repeats, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expected);
}

TEST_F(OpRepeatOutTest, RepeatTooShortDie) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make(
      /*sizes=*/{3, 2}, /*data=*/{0, 1, 2, 3, 4, 5});

  // The length of repeat vector shall not be shorter than x.dim().
  std::vector<int64_t> repeats_vec = {3};
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  Tensor out = tf.ones(/*sizes=*/{3, 0, 12});

  ET_EXPECT_KERNEL_FAILURE(context_, op_repeat_out(x, repeats, out));
}

TEST_F(OpRepeatOutTest, NegativeRepeatDie) {
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make(
      /*sizes=*/{3, 2}, /*data=*/{0, 1, 2, 3, 4, 5});

  // Try to create tensor with negative shape, die.
  std::vector<int64_t> repeats_vec = {3, -1};
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  Tensor out = tf.ones(/*sizes=*/{3, 1});

  ET_EXPECT_KERNEL_FAILURE(context_, op_repeat_out(x, repeats, out));
}

TEST_F(OpRepeatOutTest, WrongOutputShapeDie) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle wrong output shape";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.ones(
      /*sizes=*/{3, 2});

  std::vector<int64_t> repeats_vec = {3, 5, 6};
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  // The size of output shall be [3, 15, 12].
  Tensor out = tf.ones(/*sizes=*/{3, 5, 12});

  ET_EXPECT_KERNEL_FAILURE(context_, op_repeat_out(x, repeats, out));
}

TEST_F(OpRepeatOutTest, OutputDtypeMismatchedDie) {
  TensorFactory<ScalarType::Int> tf_in;
  TensorFactory<ScalarType::Float> tf_out;

  Tensor x = tf_in.ones(
      /*sizes=*/{3, 3});

  std::vector<int64_t> repeats_vec = {7, 5, 6};
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  Tensor out = tf_out.ones(/*sizes=*/{7, 15, 18});

  ET_EXPECT_KERNEL_FAILURE(context_, op_repeat_out(x, repeats, out));
}

// Right now we only support the dimension of input and output no larger
// than 16.
TEST_F(OpRepeatOutTest, TooManyDimensionsDies) {
  if (torch::executor::testing::SupportedFeatures::get()->is_aten) {
    GTEST_SKIP() << "ATen kernel can handle larger number of dimensions";
  }
  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.ones(
      /*sizes=*/{3, 2});

  auto repeats_vec = std::vector<int64_t>(17, 1);
  exec_aten::ArrayRef<int64_t> repeats = {
      repeats_vec.data(), repeats_vec.size()};

  // The size of output shall be [1, 1, .. total 15 * 1 .. , 1, 3, 2].
  auto output_shape = std::vector<int32_t>(15, 1);
  output_shape.push_back(3);
  output_shape.push_back(2);
  Tensor out = tf.ones(/*sizes=*/output_shape);

  ET_EXPECT_KERNEL_FAILURE(context_, op_repeat_out(x, repeats, out));
}

#if !defined(USE_ATEN_LIB)
TEST_F(OpRepeatOutTest, UpperBoundOutTensor) {
  TensorFactory<ScalarType::Float> tf;
  // clang-format off
  Tensor x = tf.make(
      /*size=*/{2, 2},
      /*data=*/{
                  0, 1,
                  2, 3,
                });
  std::vector<int64_t> repeats_vec = {3, 3, 3};
  exec_aten::ArrayRef<int64_t> repeats = {repeats_vec.data(), repeats_vec.size()};
  // clang-format on

  // Output tensor with the shape of the input tensor x repeated
  // - Its dimension shall equal to the length of repeat.
  // - For any dimension i ∈ [repeat.size()-x.dim(), repeat.size()), out.size(i)
  // = x.size(i) * repeat[i]
  // - For any dimension i ∈ [0, repeat.size()), out.size(i) = repeat[i]
  Tensor out =
      tf.zeros({5, 9, 9}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);

  // clang-format off
  // Repeat the input tensor along the specified `repeat` dimensions.
  Tensor expected = tf.make(
      /*sizes=*/ {3, 6, 6},
      /*data=*/
      {
        //[0, :, :]
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,

        //[1, :, :]
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,

        //[2, :, :]
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
        0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3,
      });
  // clang-format on

  Tensor ret = op_repeat_out(x, repeats, out);
  EXPECT_TENSOR_EQ(ret, out);
  EXPECT_TENSOR_EQ(ret, expected);
}
#endif

/* %python
import torch
torch.manual_seed(0)
x = torch.randint(10, (1, 2))
res = x.repeat(4, 2)
op = "op_repeat_out"
opt_setup_params = f"""
  {declare_array_ref([4, 2], "int64_t", "repeats")}
"""
opt_extra_params = "repeats,"
dtype = "ScalarType::Int"
check = "EXPECT_TENSOR_EQ" */

TEST_F(OpRepeatOutTest, DynamicShapeUpperBoundSameAsExpected) {
  /* %python
  out_args = "{4, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({1, 2}, {4, 9});
  Tensor expected =
      tf.make({4, 4}, {4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9});

  std::vector<int64_t> repeatsv = {4, 2};
  ArrayRef<int64_t> repeats(repeatsv.data(), repeatsv.size());

  Tensor out =
      tf.zeros({4, 4}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_repeat_out(x, repeats, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpRepeatOutTest, DynamicShapeUpperBoundLargerThanExpected) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({1, 2}, {4, 9});
  Tensor expected =
      tf.make({4, 4}, {4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9});

  std::vector<int64_t> repeatsv = {4, 2};
  ArrayRef<int64_t> repeats(repeatsv.data(), repeatsv.size());

  Tensor out =
      tf.zeros({10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  op_repeat_out(x, repeats, out);
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(OpRepeatOutTest, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape not supported";
  }
  /* %python
  out_args = "{1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND"
  %rewrite(unary_op) */

  TensorFactory<ScalarType::Int> tf;

  Tensor x = tf.make({1, 2}, {4, 9});
  Tensor expected =
      tf.make({4, 4}, {4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9, 4, 9});

  std::vector<int64_t> repeatsv = {4, 2};
  ArrayRef<int64_t> repeats(repeatsv.data(), repeatsv.size());

  Tensor out =
      tf.zeros({1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
  op_repeat_out(x, repeats, out);
  EXPECT_TENSOR_EQ(out, expected);
}
