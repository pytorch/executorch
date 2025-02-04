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

class OpMaxPool2DWithIndicesOutTest : public OperatorTest {
 protected:
  ::std::tuple<executorch::aten::Tensor&, executorch::aten::Tensor&>
  op_max_pool2d_with_indices_out(
      const executorch::aten::Tensor& self,
      executorch::aten::ArrayRef<int64_t> kernel_size,
      executorch::aten::ArrayRef<int64_t> stride,
      executorch::aten::ArrayRef<int64_t> padding,
      executorch::aten::ArrayRef<int64_t> dilation,
      bool ceil_mode,
      executorch::aten::Tensor& out,
      executorch::aten::Tensor& indices) {
    return torch::executor::aten::max_pool2d_with_indices_outf(
        context_,
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        out,
        indices);
  }

  template <executorch::aten::ScalarType DTYPE>
  void test_4d_dtype() {
    torch::executor::testing::TensorFactory<DTYPE> tf;
    torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Long>
        tfLong;

    executorch::aten::Tensor self = tf.make(
        {2, 3, 5, 5},
        {28.75,   -38.875, -7.0,    -13.5,   70.75,   53.75,   69.625,  97.375,
         25.375,  99.5,    -72.125, -87.25,  79.25,   42.0,    -24.75,  -15.5,
         12.5,    -86.0,   85.5,    -0.25,   67.125,  77.0,    53.375,  -61.125,
         50.0,    3.875,   42.25,   -37.375, 51.0,    -60.875, 87.0,    32.25,
         73.5,    68.875,  -84.375, -98.75,  -30.125, 94.25,   1.625,   -86.25,
         -56.5,   -68.0,   74.25,   -51.25,  8.125,   71.375,  -53.125, 4.875,
         77.5,    -89.875, 4.5,     -46.5,   -46.375, -92.625, -85.5,   -23.0,
         -8.875,  -12.0,   -46.625, -88.625, 66.75,   87.75,   90.25,   -45.0,
         -78.125, 63.25,   28.75,   28.125,  -30.375, 17.75,   -16.0,   5.0,
         11.125,  88.625,  -47.625, 72.25,   32.0,    -7.625,  61.625,  -63.125,
         -22.75,  83.125,  -40.375, -78.25,  49.5,    -39.125, -89.625, 47.875,
         -61.375, 7.75,    16.875,  -96.375, -22.5,   8.5,     74.25,   12.75,
         90.125,  73.875,  -71.75,  -10.0,   41.25,   1.125,   10.375,  -34.625,
         29.75,   -27.5,   26.625,  81.0,    -8.875,  17.625,  84.375,  -23.625,
         -53.875, -26.0,   -67.375, -90.75,  16.375,  45.625,  99.5,    56.25,
         -87.625, -65.5,   -79.75,  31.875,  79.75,   6.375,   44.625,  -55.25,
         -5.5,    -68.875, -38.625, 54.125,  -3.125,  5.75,    29.25,   -39.5,
         26.75,   68.25,   -24.625, -53.0,   51.0,    90.625,  65.375,  43.875,
         90.875,  -41.625, 99.875,  6.375,   -31.25,  -94.0});
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
    executorch::aten::Tensor out = tf.zeros({2, 3, 4, 4});
    executorch::aten::Tensor indices = tfLong.zeros({2, 3, 4, 4});
    executorch::aten::Tensor out_expected = tf.make(
        {2, 3, 4, 4},
        {69.625,  97.375, 97.375, 99.5,    69.625, 97.375, 97.375, 99.5,
         12.5,    79.25,  85.5,   85.5,    77.0,   77.0,   85.5,   85.5,
         87.0,    73.5,   73.5,   68.875,  87.0,   94.25,  94.25,  68.875,
         -30.125, 94.25,  94.25,  8.125,   71.375, 74.25,  77.5,   77.5,
         4.5,     -8.875, -12.0,  -46.625, 87.75,  90.25,  90.25,  -45.0,
         87.75,   90.25,  90.25,  17.75,   63.25,  28.75,  88.625, 88.625,
         83.125,  83.125, 61.625, 61.625,  83.125, 83.125, 47.875, 49.5,
         16.875,  47.875, 47.875, 74.25,   90.125, 90.125, 73.875, 74.25,
         41.25,   81.0,   81.0,   29.75,   84.375, 81.0,   81.0,   17.625,
         84.375,  45.625, 99.5,   99.5,    16.375, 45.625, 99.5,   99.5,
         54.125,  54.125, 5.75,   29.25,   54.125, 68.25,  68.25,  29.25,
         90.625,  90.625, 68.25,  90.875,  99.875, 99.875, 65.375, 90.875});
    executorch::aten::Tensor indices_expected = tfLong.make(
        {2, 3, 4, 4},
        {6, 7, 7, 9, 6,  7,  7,  9,  16, 12, 18, 18, 21, 21, 18, 18,
         5, 7, 7, 8, 5,  12, 12, 8,  11, 12, 12, 19, 20, 17, 23, 23,
         0, 6, 7, 8, 11, 12, 12, 13, 11, 12, 12, 19, 15, 16, 23, 23,
         6, 6, 3, 3, 6,  6,  12, 9,  15, 12, 12, 19, 21, 21, 22, 19,
         0, 7, 7, 4, 10, 7,  7,  9,  10, 17, 18, 18, 16, 17, 18, 18,
         6, 6, 8, 9, 6,  12, 12, 9,  16, 16, 12, 19, 21, 21, 17, 19});
    op_max_pool2d_with_indices_out(
        self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
    EXPECT_TENSOR_CLOSE(out, out_expected);
    EXPECT_TENSOR_CLOSE(indices, indices_expected);
  }
};

TEST_F(OpMaxPool2DWithIndicesOutTest, SanityTest4D) {
#define TEST_ENTRY(ctype, dtype) \
  test_4d_dtype<executorch::aten::ScalarType::dtype>();
  ET_FORALL_FLOATHBF16_TYPES(TEST_ENTRY);
#undef TEST_ENTRY
}

TEST_F(OpMaxPool2DWithIndicesOutTest, SanityTest4D_2) {
  torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Float>
      tfFloat;
  torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Long>
      tfLong;

  executorch::aten::Tensor self = tfFloat.make(
      {2, 3, 8, 8},
      {47.375,  -18.625, 12.0,    5.375,   -40.375, -75.875, 51.0,    -48.25,
       -5.0,    -50.625, -96.875, -53.25,  82.25,   0.125,   -13.125, 89.75,
       53.0,    67.125,  79.625,  55.0,    77.75,   -85.5,   35.25,   -44.625,
       25.625,  -44.625, -27.75,  75.875,  -96.375, 37.375,  67.0,    -73.75,
       -77.25,  86.5,    -27.5,   93.25,   -40.0,   -66.125, 57.5,    16.0,
       -94.125, 18.125,  -22.625, -94.25,  41.5,    52.875,  36.875,  0.875,
       92.75,   -94.375, -80.375, -55.625, -11.0,   -23.75,  -8.375,  33.625,
       -27.25,  -2.25,   70.25,   -60.375, -96.125, -33.25,  43.375,  -38.75,
       -58.5,   -32.375, -83.75,  -15.375, -4.375,  7.625,   58.875,  59.25,
       -12.5,   85.75,   67.125,  -32.0,   32.125,  41.0,    47.625,  72.25,
       -36.5,   5.125,   -30.75,  -37.625, -76.25,  96.25,   55.125,  -88.0,
       -78.0,   -18.125, -23.75,  -8.75,   -3.25,   82.0,    10.5,    48.625,
       -64.0,   11.5,    -66.75,  40.75,   38.25,   44.875,  21.0,    7.0,
       30.125,  -59.125, -31.5,   -56.0,   75.0,    -31.375, -25.0,   -23.75,
       48.625,  -20.5,   52.375,  -27.25,  84.75,   -39.125, -56.5,   -80.875,
       -60.5,   -26.125, 42.375,  82.625,  43.375,  -5.5,    73.375,  71.25,
       -12.25,  58.125,  -1.375,  -97.625, -43.375, 64.125,  -71.125, -55.25,
       5.375,   56.0,    20.0,    -70.125, 78.625,  20.625,  62.5,    71.375,
       93.0,    -50.25,  26.125,  -16.875, 50.375,  -96.25,  32.75,   53.375,
       -61.5,   -48.375, 72.375,  -15.5,   2.875,   -66.0,   -31.75,  -1.0,
       65.625,  -46.75,  69.875,  74.5,    53.25,   13.25,   -41.625, 96.625,
       75.875,  97.625,  -98.25,  -32.875, -75.0,   -31.75,  72.875,  -89.625,
       56.625,  10.375,  -40.5,   -53.625, 5.75,    -52.75,  19.5,    71.125,
       80.875,  16.25,   -32.625, 88.25,   -98.5,   92.5,    -68.25,  -83.25,
       43.375,  20.25,   1.875,   -69.25,  -92.75,  80.0,    -17.125, -52.125,
       -82.0,   64.5,    -7.875,  -0.125,  -74.5,   46.375,  -68.875, 63.125,
       88.0,    12.0,    87.75,   58.125,  -83.375, 21.125,  -34.0,   -60.125,
       37.75,   17.125,  -44.875, -22.875, 72.5,    -99.0,   -30.25,  81.125,
       -73.875, -90.25,  -30.0,   64.75,   -57.875, -4.0,    -33.125, -22.875,
       -74.0,   -46.0,   4.125,   -33.375, -72.625, 83.25,   25.125,  54.25,
       43.0,    -22.5,   -22.25,  -72.375, 80.25,   5.125,   -7.0,    73.5,
       -28.625, -33.75,  -91.125, -76.75,  -67.625, 79.25,   -56.625, 85.0,
       -53.125, 11.0,    -44.0,   -75.5,   -23.375, 54.5,    21.5,    36.5,
       33.0,    83.375,  81.625,  3.5,     -7.5,    7.125,   -87.5,   12.125,
       9.625,   -90.125, -3.75,   60.375,  -80.75,  -8.375,  59.25,   30.25,
       -75.75,  78.25,   13.5,    -69.125, 48.0,    -56.5,   -10.5,   -65.5,
       -80.875, 7.0,     64.375,  -81.0,   48.625,  -33.25,  -53.75,  -68.75,
       -12.0,   3.0,     -56.625, 17.0,    36.0,    28.25,   36.75,   -39.5,
       -23.5,   -53.875, -17.375, -84.375, 0.625,   12.125,  -45.5,   14.25,
       75.5,    -92.875, 9.125,   38.25,   20.375,  -46.625, 59.875,  -79.375,
       52.625,  6.75,    11.875,  -57.5,   91.75,   -55.5,   -86.875, 11.375,
       -41.0,   1.25,    41.375,  70.5,    43.625,  -65.625, 7.625,   -90.875,
       84.5,    26.0,    -52.25,  4.125,   27.75,   17.875,  -95.75,  66.5,
       -88.875, -67.125, 21.5,    -0.875,  35.875,  53.0,    62.0,    -78.5,
       -65.125, -57.375, 5.625,   54.75,   70.0,    -94.25,  50.625,  -0.25,
       89.5,    -32.0,   -83.125, -48.625, -23.0,   75.5,    44.0,    75.0,
       58.0,    18.125,  -94.75,  -69.375, 70.375,  -51.75,  -86.75,  81.5,
       75.75,   61.625,  -14.5,   -60.75,  -58.125, -3.25,   36.25,   -95.125});
  ::std::vector<int64_t> kernel_size_vec = {2, 3};
  executorch::aten::ArrayRef<int64_t> kernel_size =
      executorch::aten::ArrayRef<int64_t>(
          kernel_size_vec.data(), kernel_size_vec.size());
  ::std::vector<int64_t> stride_vec = {2, 1};
  executorch::aten::ArrayRef<int64_t> stride =
      executorch::aten::ArrayRef<int64_t>(stride_vec.data(), stride_vec.size());
  ::std::vector<int64_t> padding_vec = {0, 1};
  executorch::aten::ArrayRef<int64_t> padding =
      executorch::aten::ArrayRef<int64_t>(
          padding_vec.data(), padding_vec.size());
  ::std::vector<int64_t> dilation_vec = {2, 1};
  executorch::aten::ArrayRef<int64_t> dilation =
      executorch::aten::ArrayRef<int64_t>(
          dilation_vec.data(), dilation_vec.size());
  bool ceil_mode = false;
  executorch::aten::Tensor out = tfFloat.zeros({2, 3, 3, 8});
  executorch::aten::Tensor indices = tfLong.zeros({2, 3, 3, 8});
  executorch::aten::Tensor out_expected = tfFloat.make(
      {2, 3, 3, 8},
      {67.125, 79.625, 79.625, 79.625, 77.75,  77.75,  51.0,    51.0,    86.5,
       86.5,   93.25,  93.25,  93.25,  77.75,  57.5,   57.5,    92.75,   92.75,
       93.25,  93.25,  93.25,  57.5,   57.5,   57.5,   5.125,   5.125,   5.125,
       -4.375, 96.25,  96.25,  96.25,  59.25,  11.5,   11.5,    40.75,   40.75,
       96.25,  96.25,  96.25,  55.125, 48.625, 52.375, 52.375,  84.75,   84.75,
       84.75,  44.875, 21.0,   93.0,   93.0,   58.125, 50.375,  64.125,  64.125,
       64.125, 53.375, 93.0,   93.0,   74.5,   74.5,   74.5,    53.25,   96.625,
       96.625, 65.625, 69.875, 74.5,   74.5,   74.5,   53.25,   96.625,  96.625,
       88.0,   88.0,   87.75,  87.75,  80.0,   80.0,   80.0,    -17.125, 88.0,
       88.0,   87.75,  87.75,  64.75,  21.125, 21.125, -22.875, 43.0,    43.0,
       64.75,  80.25,  80.25,  80.25,  73.5,   73.5,   11.0,    11.0,    60.375,
       60.375, 60.375, 59.25,  59.25,  59.25,  9.625,  64.375,  64.375,  64.375,
       60.375, 59.25,  59.25,  59.25,  7.0,    64.375, 64.375,  64.375,  48.625,
       48.625, 14.25,  14.25,  84.5,   84.5,   26.0,   91.75,   91.75,   91.75,
       66.5,   66.5,   84.5,   84.5,   54.75,  70.0,   70.0,    70.0,    66.5,
       66.5,   58.0,   58.0,   54.75,  70.375, 70.375, 70.375,  81.5,    81.5});
  executorch::aten::Tensor indices_expected = tfLong.make(
      {2, 3, 3, 8},
      {17, 18, 18, 18, 20, 20, 6,  6,  33, 33, 35, 35, 35, 20, 38, 38, 48, 48,
       35, 35, 35, 38, 38, 38, 17, 17, 17, 4,  21, 21, 21, 7,  33, 33, 35, 35,
       21, 21, 21, 22, 48, 50, 50, 52, 52, 52, 37, 38, 16, 16, 1,  20, 5,  5,
       5,  23, 16, 16, 35, 35, 35, 36, 39, 39, 32, 34, 35, 35, 35, 36, 39, 39,
       16, 16, 18, 18, 5,  5,  5,  6,  16, 16, 18, 18, 35, 21, 21, 39, 48, 48,
       35, 52, 52, 52, 55, 55, 1,  1,  19, 19, 19, 22, 22, 22, 16, 34, 34, 34,
       19, 22, 22, 22, 33, 34, 34, 34, 36, 36, 55, 55, 16, 16, 17, 4,  4,  4,
       23, 23, 16, 16, 35, 36, 36, 36, 23, 23, 48, 48, 35, 52, 52, 52, 55, 55});
  op_max_pool2d_with_indices_out(
      self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  EXPECT_TENSOR_CLOSE(out, out_expected);
  EXPECT_TENSOR_CLOSE(indices, indices_expected);
}

TEST_F(OpMaxPool2DWithIndicesOutTest, SanityTest3D) {
  torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Float>
      tfFloat;
  torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Long>
      tfLong;

  executorch::aten::Tensor self = tfFloat.make(
      {2, 12, 12},
      {73.625,  15.5,    30.875,  89.25,   -55.625, -62.875, 25.0,    -50.75,
       -47.125, 12.125,  -73.125, -89.875, 53.625,  -63.125, -44.375, 86.0,
       53.625,  -84.125, -6.75,   20.125,  -24.25,  -43.5,   -11.125, -34.625,
       -7.5,    -13.0,   74.375,  33.75,   -44.875, 49.125,  -59.5,   -88.5,
       -46.5,   -33.0,   48.125,  80.875,  38.875,  -58.875, 0.875,   -48.625,
       -46.125, -87.25,  -66.625, 14.375,  -68.25,  -77.0,   -50.5,   -15.625,
       86.875,  89.875,  -37.25,  7.5,     -16.75,  -6.625,  55.875,  40.5,
       -83.875, -77.625, -55.375, 32.25,   -17.5,   -83.125, 43.375,  17.5,
       2.75,    -51.25,  20.25,   -77.375, -68.0,   -72.625, -47.5,   -78.875,
       -49.375, -52.125, -7.125,  -25.125, -77.5,   -8.625,  -3.125,  99.375,
       71.875,  19.625,  21.125,  -47.0,   44.5,    -90.625, -75.75,  -87.25,
       79.75,   -42.125, -90.0,   22.5,    2.5,     73.5,    -65.125, -50.375,
       -71.625, 19.25,   -60.125, -91.75,  -43.375, -60.875, 16.375,  86.875,
       -93.25,  -78.375, 82.5,    14.75,   20.125,  19.625,  33.875,  84.875,
       60.625,  41.5,    2.0,     -4.875,  -52.5,   74.375,  -40.125, -60.125,
       88.5,    51.875,  -59.75,  49.5,    -81.0,   -93.5,   43.0,    -99.625,
       40.375,  -84.0,   76.5,    27.5,    59.125,  -19.5,   -55.25,  -50.5,
       81.875,  86.0,    -19.75,  51.5,    70.875,  -90.5,   74.375,  62.5,
       -0.625,  -31.375, -71.25,  42.75,   42.5,    67.125,  26.125,  85.375,
       -11.75,  -34.375, -97.125, 5.875,   -45.25,  -50.125, 74.5,    -62.0,
       -81.5,   -84.875, 70.75,   33.375,  -27.5,   -54.25,  94.25,   74.625,
       -30.0,   16.875,  39.875,  -0.5,    0.25,    -80.125, 85.375,  42.5,
       13.125,  -82.375, -30.75,  -95.75,  34.75,   -60.125, -51.625, -10.375,
       -30.75,  -65.5,   -96.0,   -95.25,  60.125,  -33.125, 67.125,  -88.0,
       -26.125, 75.875,  29.5,    -27.75,  -28.875, 21.375,  -2.0,    -29.125,
       11.0,    -68.5,   -36.75,  -85.375, -4.625,  9.0,     -31.75,  -63.5,
       98.75,   -1.375,  17.125,  61.25,   -50.5,   41.375,  -18.375, -92.25,
       -50.375, -40.625, 14.0,    18.5,    22.625,  10.375,  58.875,  -86.0,
       -9.625,  5.125,   -69.625, -50.25,  -26.875, 26.0,    57.25,   -94.5,
       -53.125, 98.0,    37.375,  35.0,    -20.125, -9.375,  -13.375, -41.125,
       41.75,   95.125,  82.75,   -71.5,   -43.125, -37.75,  -91.25,  -14.0,
       -55.5,   52.5,    -30.125, 93.875,  -26.75,  83.125,  2.625,   -63.75,
       52.875,  31.25,   57.625,  42.75,   -2.875,  -45.5,   99.0,    -18.625,
       38.375,  88.125,  36.625,  -36.875, -35.25,  13.125,  -31.875, -50.875,
       10.5,    -38.75,  1.625,   67.125,  3.0,     -87.0,   42.0,    -31.25,
       -77.875, -7.125,  -94.0,   -99.0,   24.75,   -21.625, -98.375, 15.875});
  ::std::vector<int64_t> kernel_size_vec = {4, 3};
  executorch::aten::ArrayRef<int64_t> kernel_size =
      executorch::aten::ArrayRef<int64_t>(
          kernel_size_vec.data(), kernel_size_vec.size());
  ::std::vector<int64_t> stride_vec = {3, 2};
  executorch::aten::ArrayRef<int64_t> stride =
      executorch::aten::ArrayRef<int64_t>(stride_vec.data(), stride_vec.size());
  ::std::vector<int64_t> padding_vec = {2, 1};
  executorch::aten::ArrayRef<int64_t> padding =
      executorch::aten::ArrayRef<int64_t>(
          padding_vec.data(), padding_vec.size());
  ::std::vector<int64_t> dilation_vec = {1, 2};
  executorch::aten::ArrayRef<int64_t> dilation =
      executorch::aten::ArrayRef<int64_t>(
          dilation_vec.data(), dilation_vec.size());
  bool ceil_mode = false;
  executorch::aten::Tensor out = tfFloat.zeros({2, 5, 5});
  executorch::aten::Tensor indices = tfLong.zeros({2, 5, 5});
  executorch::aten::Tensor out_expected = tfFloat.make(
      {2, 5, 5},
      {89.25,  89.25,  89.25,  20.125, 20.125, 89.875, 89.875, 86.0,   49.125,
       80.875, 89.875, 89.875, 99.375, 99.375, 99.375, 84.875, 84.875, 86.875,
       86.875, 86.875, 51.875, 86.0,   86.0,   86.0,   62.5,   42.75,  67.125,
       85.375, 85.375, 85.375, 75.875, 75.875, 42.5,   42.5,   74.625, 75.875,
       98.0,   98.0,   98.0,   61.25,  95.125, 98.0,   98.0,   98.0,   93.875,
       88.125, 88.125, 13.125, 13.125, 67.125});
  executorch::aten::Tensor indices_expected = tfLong.make(
      {2, 5, 5}, {3,  3,  3,   19,  19,  49,  49,  15,  29,  35,  49,  49,  79,
                  79, 79, 111, 111, 103, 103, 103, 121, 137, 137, 137, 143, 3,
                  5,  7,  7,   7,   49,  49,  31,  31,  23,  49,  89,  89,  89,
                  67, 97, 89,  89,  89,  107, 121, 121, 125, 125, 131});
  op_max_pool2d_with_indices_out(
      self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  EXPECT_TENSOR_CLOSE(out, out_expected);
  EXPECT_TENSOR_CLOSE(indices, indices_expected);
}

TEST_F(OpMaxPool2DWithIndicesOutTest, CeilMode) {
  torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Float>
      tfFloat;
  torch::executor::testing::TensorFactory<executorch::aten::ScalarType::Long>
      tfLong;

  executorch::aten::Tensor self = tfFloat.make(
      {2, 7, 7}, {-7, -9,  -6,  -8,  -9, -9,  -6,  -10, -7,  -6,  -10, -7, -10,
                  -7, -8,  -10, -6,  -8, -8,  -10, -9,  -8,  -6,  -8,  -9, -8,
                  -8, -8,  -6,  -9,  -9, -8,  -8,  -8,  -8,  -7,  -7,  -6, -7,
                  -8, -6,  -8,  -9,  -7, -6,  -10, -6,  -9,  -6,

                  -7, -8,  -10, -10, -8, -8,  -10, -10, -6,  -10, -10, -7, -8,
                  -6, -10, -8,  -8,  -8, -9,  -6,  -9,  -7,  -10, -7,  -6, -9,
                  -8, -9,  -7,  -8,  -7, -10, -6,  -6,  -6,  -7,  -10, -8, -9,
                  -6, -9,  -9,  -8,  -8, -8,  -9,  -9,  -10, -8});

  ::std::vector<int64_t> kernel_size_vec = {2, 2};
  executorch::aten::ArrayRef<int64_t> kernel_size =
      executorch::aten::ArrayRef<int64_t>(
          kernel_size_vec.data(), kernel_size_vec.size());
  ::std::vector<int64_t> stride_vec = {2, 2};
  executorch::aten::ArrayRef<int64_t> stride =
      executorch::aten::ArrayRef<int64_t>(stride_vec.data(), stride_vec.size());
  ::std::vector<int64_t> padding_vec = {0, 0};
  executorch::aten::ArrayRef<int64_t> padding =
      executorch::aten::ArrayRef<int64_t>(
          padding_vec.data(), padding_vec.size());
  ::std::vector<int64_t> dilation_vec = {1, 1};
  executorch::aten::ArrayRef<int64_t> dilation =
      executorch::aten::ArrayRef<int64_t>(
          dilation_vec.data(), dilation_vec.size());

  bool ceil_mode = true;
  executorch::aten::Tensor out = tfFloat.zeros({2, 4, 4});
  executorch::aten::Tensor indices = tfLong.zeros({2, 4, 4});
  executorch::aten::Tensor out_expected = tfFloat.make(
      {2, 4, 4},
      {-7, -6,  -7, -6, -6, -6, -8, -8, -6, -6, -6, -8, -7, -6, -6, -6,

       -6, -10, -7, -6, -7, -6, -6, -9, -7, -7, -6, -6, -8, -8, -9, -8});

  op_max_pool2d_with_indices_out(
      self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  EXPECT_TENSOR_CLOSE(out, out_expected);

  ceil_mode = false;
  out = tfFloat.zeros({2, 3, 3});
  indices = tfLong.zeros({2, 3, 3});
  out_expected = tfFloat.make(
      {2, 3, 3},
      {-7,
       -6,
       -7,
       -6,
       -6,
       -8,
       -6,
       -6,
       -6,

       -6,
       -10,
       -7,
       -7,
       -6,
       -6,
       -7,
       -7,
       -6});

  op_max_pool2d_with_indices_out(
      self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
