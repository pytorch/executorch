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

class OpMaxPool2DWithIndicesBackwardOutTest : public OperatorTest {
 protected:
  executorch::aten::Tensor& op_max_pool2d_with_indices_backward_out(
      const executorch::aten::Tensor& grad_output,
      const executorch::aten::Tensor& input,
      executorch::aten::ArrayRef<int64_t> kernel_size,
      executorch::aten::ArrayRef<int64_t> stride,
      executorch::aten::ArrayRef<int64_t> padding,
      executorch::aten::ArrayRef<int64_t> dilation,
      bool ceil_mode,
      const executorch::aten::Tensor& indices,
      executorch::aten::Tensor& grad_input) {
    return torch::executor::aten::max_pool2d_with_indices_backward_outf(
        context_,
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
        grad_input);
  }

  template <executorch::aten::ScalarType DTYPE>
  void test_4d_dtype() {
    torch::executor::testing::TensorFactory<DTYPE> tf;
    torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Long>
        tfLong;

    executorch::aten::Tensor grad_output = tf.make(
        {2, 3, 4, 4},
        {69, 97, 97,  99,  69, 97, 97, 99,  12,  79, 85, 85, 77, 77, 85, 85,
         87, 73, 73,  68,  87, 94, 94, 68,  -30, 94, 94, 8,  71, 74, 77, 77,
         4,  -8, -12, -46, 87, 90, 90, -45, 87,  90, 90, 17, 63, 28, 88, 88,
         83, 83, 61,  61,  83, 83, 47, 49,  16,  47, 47, 74, 90, 90, 73, 74,
         41, 81, 81,  29,  84, 81, 81, 17,  84,  45, 99, 99, 16, 45, 99, 99,
         54, 54, 5,   29,  54, 68, 68, 29,  90,  90, 68, 90, 99, 99, 65, 90});

    executorch::aten::Tensor input = tf.make(
        {2, 3, 5, 5},
        {28,  -38, -7,  -13, 70,  53,  69,  97,  25,  99,  -72, -87, 79,  42,
         -24, -15, 12,  -86, 85,  0,   67,  77,  53,  -61, 50,  3,   42,  -37,
         51,  -60, 87,  32,  73,  68,  -84, -98, -30, 94,  1,   -86, -56, -68,
         74,  -51, 8,   71,  -53, 4,   77,  -89, 4,   -46, -46, -92, -85, -23,
         -8,  -12, -46, -88, 66,  87,  90,  -45, -78, 63,  28,  28,  -30, 17,
         -16, 5,   11,  88,  -47, 72,  32,  -7,  61,  -63, -22, 83,  -40, -78,
         49,  -39, -89, 47,  -61, 7,   16,  -96, -22, 8,   74,  12,  90,  73,
         -71, -10, 41,  1,   10,  -34, 29,  -27, 26,  81,  -8,  17,  84,  -23,
         -53, -26, -67, -90, 16,  45,  99,  56,  -87, -65, -79, 31,  79,  6,
         44,  -55, -5,  -68, -38, 54,  -3,  5,   29,  -39, 26,  68,  -24, -53,
         51,  90,  65,  43,  90,  -41, 99,  6,   -31, -94});

    ::std::vector<int64_t> kernel_size_vec = {2, 2};
    executorch::aten::ArrayRef<int64_t> kernel_size =
        executorch::aten::ArrayRef<int64_t>(
            kernel_size_vec.data(), kernel_size_vec.size());
    ::std::vector<int64_t> stride_vec = {1, 1};
    executorch::aten::ArrayRef<int64_t> stride =
        executorch::aten::ArrayRef<int64_t>(
            stride_vec.data(), stride_vec.size());
    ::std::vector<int64_t> padding_vec = {0, 0};
    executorch::aten::ArrayRef<int64_t> padding =
        executorch::aten::ArrayRef<int64_t>(
            padding_vec.data(), padding_vec.size());
    ::std::vector<int64_t> dilation_vec = {1, 1};
    executorch::aten::ArrayRef<int64_t> dilation =
        executorch::aten::ArrayRef<int64_t>(
            dilation_vec.data(), dilation_vec.size());
    bool ceil_mode = false;
    executorch::aten::Tensor indices = tfLong.make(
        {2, 3, 4, 4},
        {6, 7, 7, 9, 6,  7,  7,  9,  16, 12, 18, 18, 21, 21, 18, 18,
         5, 7, 7, 8, 5,  12, 12, 8,  11, 12, 12, 19, 20, 17, 23, 23,
         0, 6, 7, 8, 11, 12, 12, 13, 11, 12, 12, 19, 15, 16, 23, 23,
         6, 6, 3, 3, 6,  6,  12, 9,  15, 12, 12, 19, 21, 21, 22, 19,
         0, 7, 7, 4, 10, 7,  7,  9,  10, 17, 18, 18, 16, 17, 18, 18,
         6, 6, 8, 9, 6,  12, 12, 9,  16, 16, 12, 19, 21, 21, 17, 19});
    executorch::aten::Tensor grad_input = tf.zeros({2, 3, 5, 5});
    executorch::aten::Tensor grad_input_expected = tf.make(
        {2, 3, 5, 5},
        {0,   0,   0,   0,   0,   0,   138, 388, 0, 198, 0,  0,   79,  0,   0,
         0,   12,  0,   340, 0,   0,   154, 0,   0, 0,   0,  0,   0,   0,   0,
         174, 0,   146, 136, 0,   0,   -30, 376, 0, 0,   0,  0,   74,  0,   8,
         71,  0,   0,   154, 0,   4,   0,   0,   0, 0,   0,  -8,  -12, -46, 0,
         0,   174, 360, -45, 0,   63,  28,  0,   0, 17,  0,  0,   0,   176, 0,
         0,   0,   0,   122, 0,   0,   332, 0,   0, 49,  0,  0,   141, 0,   0,
         16,  0,   0,   0,   148, 0,   180, 73,  0, 0,   41, 0,   0,   0,   29,
         0,   0,   324, 0,   17,  168, 0,   0,   0, 0,   0,  16,  90,  396, 0,
         0,   0,   0,   0,   0,   0,   0,   0,   0, 0,   0,  162, 0,   5,   58,
         0,   0,   204, 0,   0,   0,   180, 65,  0, 180, 0,  198, 0,   0,   0});
    op_max_pool2d_with_indices_backward_out(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
        grad_input);
    EXPECT_TENSOR_CLOSE(grad_input, grad_input_expected);
  }

  template <executorch::aten::ScalarType DTYPE>
  void test_3d_dtype() {
    torch::executor::testing::TensorFactory<DTYPE> tf;
    torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Long>
        tfLong;

    executorch::aten::Tensor grad_output =
        tf.make({2, 5, 5}, {89, 89, 89, 20, 20, 89, 89, 86, 49, 80, 89, 89, 99,
                            99, 99, 84, 84, 86, 86, 86, 51, 86, 86, 86, 62, 42,
                            67, 85, 85, 85, 75, 75, 42, 42, 74, 75, 98, 98, 98,
                            61, 95, 98, 98, 98, 93, 88, 88, 13, 13, 67});

    executorch::aten::Tensor input = tf.make(
        {2, 12, 12},
        {73,  15,  30,  89,  -55, -62, 25,  -50, -47, 12,  -73, -89, 53,  -63,
         -44, 86,  53,  -84, -6,  20,  -24, -43, -11, -34, -7,  -13, 74,  33,
         -44, 49,  -59, -88, -46, -33, 48,  80,  38,  -58, 0,   -48, -46, -87,
         -66, 14,  -68, -77, -50, -15, 86,  89,  -37, 7,   -16, -6,  55,  40,
         -83, -77, -55, 32,  -17, -83, 43,  17,  2,   -51, 20,  -77, -68, -72,
         -47, -78, -49, -52, -7,  -25, -77, -8,  -3,  99,  71,  19,  21,  -47,
         44,  -90, -75, -87, 79,  -42, -90, 22,  2,   73,  -65, -50, -71, 19,
         -60, -91, -43, -60, 16,  86,  -93, -78, 82,  14,  20,  19,  33,  84,
         60,  41,  2,   -4,  -52, 74,  -40, -60, 88,  51,  -59, 49,  -81, -93,
         43,  -99, 40,  -84, 76,  27,  59,  -19, -55, -50, 81,  86,  -19, 51,
         70,  -90, 74,  62,  0,   -31, -71, 42,  42,  67,  26,  85,  -11, -34,
         -97, 5,   -45, -50, 74,  -62, -81, -84, 70,  33,  -27, -54, 94,  74,
         -30, 16,  39,  0,   0,   -80, 85,  42,  13,  -82, -30, -95, 34,  -60,
         -51, -10, -30, -65, -96, -95, 60,  -33, 67,  -88, -26, 75,  29,  -27,
         -28, 21,  -2,  -29, 11,  -68, -36, -85, -4,  9,   -31, -63, 98,  -1,
         17,  61,  -50, 41,  -18, -92, -50, -40, 14,  18,  22,  10,  58,  -86,
         -9,  5,   -69, -50, -26, 26,  57,  -94, -53, 98,  37,  35,  -20, -9,
         -13, -41, 41,  95,  82,  -71, -43, -37, -91, -14, -55, 52,  -30, 93,
         -26, 83,  2,   -63, 52,  31,  57,  42,  -2,  -45, 99,  -18, 38,  88,
         36,  -36, -35, 13,  -31, -50, 10,  -38, 1,   67,  3,   -87, 42,  -31,
         -77, -7,  -94, -99, 24,  -21, -98, 15});
    ::std::vector<int64_t> kernel_size_vec = {4, 3};
    executorch::aten::ArrayRef<int64_t> kernel_size =
        executorch::aten::ArrayRef<int64_t>(
            kernel_size_vec.data(), kernel_size_vec.size());
    ::std::vector<int64_t> stride_vec = {3, 2};
    executorch::aten::ArrayRef<int64_t> stride =
        executorch::aten::ArrayRef<int64_t>(
            stride_vec.data(), stride_vec.size());
    ::std::vector<int64_t> padding_vec = {2, 1};
    executorch::aten::ArrayRef<int64_t> padding =
        executorch::aten::ArrayRef<int64_t>(
            padding_vec.data(), padding_vec.size());
    ::std::vector<int64_t> dilation_vec = {1, 2};
    executorch::aten::ArrayRef<int64_t> dilation =
        executorch::aten::ArrayRef<int64_t>(
            dilation_vec.data(), dilation_vec.size());
    bool ceil_mode = false;
    executorch::aten::Tensor indices = tfLong.make(
        {2, 5, 5},
        {3,  3,  3,   19,  19,  49,  49,  15,  29,  35,  49,  49,  79,
         79, 79, 111, 111, 103, 103, 103, 121, 137, 137, 137, 143, 3,
         5,  7,  7,   7,   49,  49,  31,  31,  23,  49,  89,  89,  89,
         67, 97, 89,  89,  89,  107, 121, 121, 125, 125, 131});
    executorch::aten::Tensor grad_input = tf.zeros({2, 12, 12});
    executorch::aten::Tensor grad_input_expected = tf.make(
        {2, 12, 12},
        {0, 0,  0, 267, 0, 0,  0, 0,   0, 0, 0, 0,   0, 0,   0, 86, 0, 0,
         0, 40, 0, 0,   0, 0,  0, 0,   0, 0, 0, 49,  0, 0,   0, 0,  0, 80,
         0, 0,  0, 0,   0, 0,  0, 0,   0, 0, 0, 0,   0, 356, 0, 0,  0, 0,
         0, 0,  0, 0,   0, 0,  0, 0,   0, 0, 0, 0,   0, 0,   0, 0,  0, 0,
         0, 0,  0, 0,   0, 0,  0, 297, 0, 0, 0, 0,   0, 0,   0, 0,  0, 0,
         0, 0,  0, 0,   0, 0,  0, 0,   0, 0, 0, 0,   0, 258, 0, 0,  0, 0,
         0, 0,  0, 168, 0, 0,  0, 0,   0, 0, 0, 0,   0, 51,  0, 0,  0, 0,
         0, 0,  0, 0,   0, 0,  0, 0,   0, 0, 0, 258, 0, 0,   0, 0,  0, 62,
         0, 0,  0, 42,  0, 67, 0, 255, 0, 0, 0, 0,   0, 0,   0, 0,  0, 0,
         0, 0,  0, 0,   0, 74, 0, 0,   0, 0, 0, 0,   0, 84,  0, 0,  0, 0,
         0, 0,  0, 0,   0, 0,  0, 0,   0, 0, 0, 0,   0, 225, 0, 0,  0, 0,
         0, 0,  0, 0,   0, 0,  0, 0,   0, 0, 0, 0,   0, 61,  0, 0,  0, 0,
         0, 0,  0, 0,   0, 0,  0, 0,   0, 0, 0, 0,   0, 0,   0, 0,  0, 588,
         0, 0,  0, 0,   0, 0,  0, 95,  0, 0, 0, 0,   0, 0,   0, 0,  0, 93,
         0, 0,  0, 0,   0, 0,  0, 0,   0, 0, 0, 0,   0, 176, 0, 0,  0, 26,
         0, 0,  0, 0,   0, 67, 0, 0,   0, 0, 0, 0,   0, 0,   0, 0,  0, 0});

    op_max_pool2d_with_indices_backward_out(
        grad_output,
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
        grad_input);
    EXPECT_TENSOR_CLOSE(grad_input, grad_input_expected);
  }
};

TEST_F(OpMaxPool2DWithIndicesBackwardOutTest, SanityTest4D) {
#define TEST_ENTRY(ctype, dtype) \
  test_4d_dtype<executorch::aten::ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpMaxPool2DWithIndicesBackwardOutTest, SanityTest3D) {
#define TEST_ENTRY(ctype, dtype) \
  test_3d_dtype<executorch::aten::ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}
