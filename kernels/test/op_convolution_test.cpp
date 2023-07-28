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
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor& convolution_out(
    const Tensor& input,
    const Tensor& weight,
    const optional<Tensor>& bias,
    ArrayRef<int64_t> stride,
    ArrayRef<int64_t> padding,
    ArrayRef<int64_t> dilation,
    bool transposed,
    ArrayRef<int64_t> output_padding,
    int64_t groups,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::convolution_outf(
      context,
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      out);
}

/* Correctness Test Template for test code generation via Python */
/* %python
correctness_test_template = f"""
  {declare_tensor_factory("ScalarType::$DTYPE$", "tf")}

  {declare_tensor_make_t("input", "tf")}
  {declare_tensor_make_t("weight", "tf")}
  {declare_optional_tensor_make_t("bias", "tf")}
  {declare_tensor_make_t("expected", "tf")}
  Tensor out = tf.zeros($out_size$, $dynamism$);

  {declare_array_ref_t("stride", "int64_t")}
  {declare_array_ref_t("padding", "int64_t")}
  {declare_array_ref_t("dilation", "int64_t")}
  {declare_array_ref_t("output_padding", "int64_t")}

  convolution_out(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      $transposed$,
      output_padding,
      $groups$,
      out);
  EXPECT_TENSOR_CLOSE(out, expected);"""
*/

//
// Correctness Tests
//

TEST(OpConvCorrectnessTest, GenericSmokeTest) {
  TensorFactory<ScalarType::Int> tf;

  auto input = tf.make({1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto weight =
      tf.make({4, 2, 3}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  auto bias = tf.ones({4});
  auto expected = tf.make({1, 4, 2}, {80, 110, 206, 308, 332, 506, 458, 704});
  auto out = tf.zeros({1, 4, 2});

  int64_t stride[1] = {2};
  int64_t padding[1] = {0};
  int64_t dilation[1] = {1};
  int64_t output_padding[1] = {0};

  convolution_out(
      input,
      weight,
      exec_aten::optional<Tensor>(bias),
      exec_aten::ArrayRef<int64_t>{stride, 1},
      exec_aten::ArrayRef<int64_t>{padding, 1},
      exec_aten::ArrayRef<int64_t>{dilation, 1},
      false,
      exec_aten::ArrayRef<int64_t>{output_padding, 1},
      int64_t(1),
      out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

/* %python
import torch
torch.manual_seed(0)
input = (torch.randint(10, 100, (1, 2, 5)).to(torch.double) / 10.0);
weight = (torch.randint(10, 100, (4, 2, 3)).to(torch.double) / 10.0);
bias = torch.ones(4).to(torch.double)
stride = [2]
padding = [2]
dilation = [1]
transposed = False
output_padding = [0]
groups = 1
expected = torch.nn.functional.conv1d(
    input, weight, bias, stride, padding, dilation, groups)

DTYPE = "Float"
out_size = expected.size()
dynamism = "torch::executor::TensorShapeDynamism::STATIC"
*/
TEST(OpConvCorrectnessTest, NonZeroPadding) {
  /* %python
  %past-rewrite(correctness_test_template) */

  TensorFactory<ScalarType::Float> tf;

  Tensor input =
      tf.make({1, 2, 5}, {5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3});
  Tensor weight = tf.make(
      {4, 2, 3}, {8.1, 6.6, 1.6, 4.9, 3.8, 6.6, 4.6, 2.8, 2.4, 1.3, 3.6, 3.9,
                  8.1, 8.4, 5.4, 5.1, 8.9, 9.9, 7.9, 1.0, 1.1, 8.2, 6.3, 7.0});
  optional<Tensor> bias(tf.make({4}, {1.0, 1.0, 1.0, 1.0}));
  Tensor expected = tf.make(
      {1, 4, 4},
      {61.78,
       172.11,
       237.72,
       79.7,
       44.77,
       102.24,
       132.28,
       34.87,
       108.37,
       248.51,
       320.18,
       81.16,
       62.24,
       189.38,
       236.07,
       102.73});
  Tensor out =
      tf.zeros({1, 4, 4}, torch::executor::TensorShapeDynamism::STATIC);

  int64_t stride[] = {2};
  int64_t padding[] = {2};
  int64_t dilation[] = {1};
  int64_t output_padding[] = {0};

  convolution_out(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      output_padding,
      1,
      out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

/* %python
import torch
torch.manual_seed(0)
input = (torch.randint(10, 100, (3, 2, 5)).to(torch.double) / 10.0);
weight = (torch.randint(10, 100, (4, 2, 3)).to(torch.double) / 10.0);
bias = torch.ones(4).to(torch.double)
stride = [2]
padding = [2]
dilation = [1]
transposed = False
output_padding = [0]
groups = 1
expected = torch.nn.functional.conv1d(
    input, weight, bias, stride, padding, dilation, groups)

DTYPE = "Float"
out_size = expected.size()
dynamism = "torch::executor::TensorShapeDynamism::STATIC"
*/
TEST(OpConvCorrectnessTest, MultipleInputBatches) {
  /* %python
  %past-rewrite(correctness_test_template) */

  TensorFactory<ScalarType::Float> tf;

  Tensor input =
      tf.make({3, 2, 5}, {5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3,
                          8.1, 6.6, 1.6, 4.9, 3.8, 6.6, 4.6, 2.8, 2.4, 1.3,
                          3.6, 3.9, 8.1, 8.4, 5.4, 5.1, 8.9, 9.9, 7.9, 1.0});
  Tensor weight = tf.make(
      {4, 2, 3}, {1.1, 8.2, 6.3, 7.0, 6.5, 2.5, 9.2, 9.9, 8.1, 9.8, 4.8, 1.3,
                  2.6, 8.9, 1.1, 8.7, 2.3, 3.5, 4.2, 7.1, 5.0, 3.9, 3.3, 4.1});
  optional<Tensor> bias(tf.make({4}, {1.0, 1.0, 1.0, 1.0}));
  Tensor expected = tf.make(
      {3, 4, 4}, {54.77, 168.21, 208.92, 57.93, 55.01, 241.19, 312.18, 121.3,
                  34.59, 143.87, 201.88, 78.29, 60.39, 154.12, 194.07, 51.73,
                  68.53, 157.21, 105.33, 14.28, 75.19, 244.22, 135.66, 48.70,
                  33.01, 160.36, 87.38,  22.19, 68.56, 142.28, 85.68,  22.03,
                  36.43, 206.27, 235.96, 13.94, 36.79, 243.91, 338.66, 60.48,
                  22.81, 153.47, 210.56, 23.74, 39.91, 174.16, 190.44, 27.58});
  Tensor out =
      tf.zeros({3, 4, 4}, torch::executor::TensorShapeDynamism::STATIC);

  int64_t stride[] = {2};
  int64_t padding[] = {2};
  int64_t dilation[] = {1};
  int64_t output_padding[] = {0};

  convolution_out(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      output_padding,
      1,
      out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

/* %python
import torch
torch.manual_seed(0)
input = (torch.randint(10, 100, (1, 4, 8, 8)).to(torch.double) / 10.0);
weight = (torch.randint(10, 100, (2, 4, 3, 3)).to(torch.double) / 10.0);
bias = torch.ones(2).to(torch.double)
stride = [2, 2]
padding = [1, 1]
dilation = [1, 1]
transposed = False
output_padding = [0]
groups = 1
expected = torch.nn.functional.conv2d(
    input, weight, bias, stride, padding, dilation, groups)

DTYPE = "Float"
out_size = expected.size()
dynamism = "torch::executor::TensorShapeDynamism::STATIC"
*/
TEST(OpConvCorrectnessTest, 2DSanityCheck) {
  /* %python
  %past-rewrite(correctness_test_template) */

  TensorFactory<ScalarType::Float> tf;

  Tensor input = tf.make(
      {1, 4, 8, 8},
      {5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3, 8.1, 6.6, 1.6, 4.9,
       3.8, 6.6, 4.6, 2.8, 2.4, 1.3, 3.6, 3.9, 8.1, 8.4, 5.4, 5.1, 8.9, 9.9,
       7.9, 1.0, 1.1, 8.2, 6.3, 7.0, 6.5, 2.5, 9.2, 9.9, 8.1, 9.8, 4.8, 1.3,
       2.6, 8.9, 1.1, 8.7, 2.3, 3.5, 4.2, 7.1, 5.0, 3.9, 3.3, 4.1, 8.1, 6.0,
       3.3, 8.6, 6.6, 5.7, 5.9, 8.6, 7.3, 3.4, 9.5, 6.0, 6.8, 6.2, 1.8, 3.2,
       2.7, 7.5, 7.0, 8.0, 2.8, 5.1, 4.9, 8.6, 1.1, 9.0, 4.2, 9.9, 2.4, 5.3,
       4.9, 9.3, 2.9, 5.3, 8.9, 4.8, 9.5, 2.3, 9.2, 3.8, 6.5, 9.6, 2.6, 3.5,
       2.7, 9.2, 1.5, 7.6, 5.6, 8.5, 5.4, 7.0, 8.8, 5.1, 2.7, 1.8, 7.5, 4.4,
       2.4, 4.8, 1.4, 3.4, 8.9, 4.0, 4.7, 3.4, 2.5, 8.3, 8.3, 1.7, 2.3, 9.0,
       2.9, 2.9, 5.3, 7.1, 3.8, 7.1, 1.7, 9.8, 2.4, 4.1, 6.0, 8.4, 4.0, 1.4,
       7.9, 7.7, 4.0, 4.0, 9.1, 7.4, 4.9, 3.9, 3.5, 8.9, 2.2, 3.2, 8.2, 7.1,
       5.4, 2.9, 8.1, 5.1, 3.0, 9.3, 2.0, 3.6, 8.7, 6.6, 9.9, 3.1, 7.6, 3.4,
       4.1, 5.0, 8.5, 9.2, 7.5, 5.8, 6.1, 5.8, 4.1, 4.2, 9.8, 2.0, 7.3, 2.8,
       7.9, 8.2, 9.7, 9.0, 4.8, 7.8, 6.6, 5.8, 4.5, 7.8, 4.6, 8.5, 7.2, 4.4,
       1.2, 7.7, 2.2, 2.4, 2.9, 1.8, 2.5, 2.6, 3.4, 6.3, 9.3, 8.4, 3.0, 8.2,
       1.5, 2.1, 3.2, 5.8, 5.2, 6.4, 1.8, 7.3, 7.6, 1.5, 2.8, 7.8, 9.0, 5.5,
       4.1, 2.3, 3.0, 8.8, 7.1, 7.1, 9.1, 3.7, 6.2, 6.2, 2.2, 1.3, 4.3, 5.6,
       8.7, 6.8, 5.0, 9.5, 5.0, 5.3, 5.5, 4.5, 3.3, 6.6, 6.2, 8.2, 5.5, 8.5,
       2.9, 9.4, 8.3, 8.3});
  Tensor weight = tf.make(
      {2, 4, 3, 3},
      {4.7, 1.3, 7.8, 3.0, 9.7, 2.5, 3.8, 5.2, 4.4, 7.7, 2.3, 6.2,
       1.5, 9.5, 6.3, 4.9, 8.1, 9.8, 2.0, 6.6, 4.7, 2.4, 6.7, 5.6,
       2.9, 1.3, 7.8, 5.4, 2.4, 6.9, 6.4, 1.4, 8.9, 7.9, 7.5, 6.7,
       4.0, 8.3, 5.2, 4.0, 4.8, 7.6, 7.1, 5.9, 9.1, 9.6, 3.9, 6.8,
       7.6, 2.5, 8.1, 7.3, 7.5, 7.5, 9.3, 5.6, 5.2, 4.7, 4.5, 8.7,
       8.7, 1.3, 4.1, 4.5, 4.9, 6.5, 7.9, 4.6, 7.0, 8.0, 1.6, 3.5});
  optional<Tensor> bias(tf.make({2}, {1.0, 1.0}));
  Tensor expected = tf.make(
      {1, 2, 4, 4},
      {642.33, 714.6,   687.96,  717.12,  859.79, 939.27,  996.79,  1189.59,
       700.73, 1083.28, 1010.33, 1167.78, 776.33, 1138.92, 1073.43, 1140.64,
       539.83, 851.42,  754.16,  815.01,  822.66, 1191.95, 1063.46, 1330.28,
       662.97, 1240.69, 1254.52, 1281.46, 766.25, 1273.41, 1148.57, 1217.47});
  Tensor out =
      tf.zeros({1, 2, 4, 4}, torch::executor::TensorShapeDynamism::STATIC);

  int64_t stride[] = {2, 2};
  int64_t padding[] = {1, 1};
  int64_t dilation[] = {1, 1};
  int64_t output_padding[] = {0};

  convolution_out(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      output_padding,
      1,
      out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST(OpConvCorrectnessTest, 2DSanityCheckChannelsLast) {
  /* %python
  %past-rewrite(correctness_test_template) */

  TensorFactory<ScalarType::Float> tf;

  Tensor input = tf.make_channels_last(
      {1, 4, 8, 8},
      {5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3, 8.1, 6.6, 1.6, 4.9,
       3.8, 6.6, 4.6, 2.8, 2.4, 1.3, 3.6, 3.9, 8.1, 8.4, 5.4, 5.1, 8.9, 9.9,
       7.9, 1.0, 1.1, 8.2, 6.3, 7.0, 6.5, 2.5, 9.2, 9.9, 8.1, 9.8, 4.8, 1.3,
       2.6, 8.9, 1.1, 8.7, 2.3, 3.5, 4.2, 7.1, 5.0, 3.9, 3.3, 4.1, 8.1, 6.0,
       3.3, 8.6, 6.6, 5.7, 5.9, 8.6, 7.3, 3.4, 9.5, 6.0, 6.8, 6.2, 1.8, 3.2,
       2.7, 7.5, 7.0, 8.0, 2.8, 5.1, 4.9, 8.6, 1.1, 9.0, 4.2, 9.9, 2.4, 5.3,
       4.9, 9.3, 2.9, 5.3, 8.9, 4.8, 9.5, 2.3, 9.2, 3.8, 6.5, 9.6, 2.6, 3.5,
       2.7, 9.2, 1.5, 7.6, 5.6, 8.5, 5.4, 7.0, 8.8, 5.1, 2.7, 1.8, 7.5, 4.4,
       2.4, 4.8, 1.4, 3.4, 8.9, 4.0, 4.7, 3.4, 2.5, 8.3, 8.3, 1.7, 2.3, 9.0,
       2.9, 2.9, 5.3, 7.1, 3.8, 7.1, 1.7, 9.8, 2.4, 4.1, 6.0, 8.4, 4.0, 1.4,
       7.9, 7.7, 4.0, 4.0, 9.1, 7.4, 4.9, 3.9, 3.5, 8.9, 2.2, 3.2, 8.2, 7.1,
       5.4, 2.9, 8.1, 5.1, 3.0, 9.3, 2.0, 3.6, 8.7, 6.6, 9.9, 3.1, 7.6, 3.4,
       4.1, 5.0, 8.5, 9.2, 7.5, 5.8, 6.1, 5.8, 4.1, 4.2, 9.8, 2.0, 7.3, 2.8,
       7.9, 8.2, 9.7, 9.0, 4.8, 7.8, 6.6, 5.8, 4.5, 7.8, 4.6, 8.5, 7.2, 4.4,
       1.2, 7.7, 2.2, 2.4, 2.9, 1.8, 2.5, 2.6, 3.4, 6.3, 9.3, 8.4, 3.0, 8.2,
       1.5, 2.1, 3.2, 5.8, 5.2, 6.4, 1.8, 7.3, 7.6, 1.5, 2.8, 7.8, 9.0, 5.5,
       4.1, 2.3, 3.0, 8.8, 7.1, 7.1, 9.1, 3.7, 6.2, 6.2, 2.2, 1.3, 4.3, 5.6,
       8.7, 6.8, 5.0, 9.5, 5.0, 5.3, 5.5, 4.5, 3.3, 6.6, 6.2, 8.2, 5.5, 8.5,
       2.9, 9.4, 8.3, 8.3});
  Tensor weight = tf.make_channels_last(
      {2, 4, 3, 3},
      {4.7, 1.3, 7.8, 3.0, 9.7, 2.5, 3.8, 5.2, 4.4, 7.7, 2.3, 6.2,
       1.5, 9.5, 6.3, 4.9, 8.1, 9.8, 2.0, 6.6, 4.7, 2.4, 6.7, 5.6,
       2.9, 1.3, 7.8, 5.4, 2.4, 6.9, 6.4, 1.4, 8.9, 7.9, 7.5, 6.7,
       4.0, 8.3, 5.2, 4.0, 4.8, 7.6, 7.1, 5.9, 9.1, 9.6, 3.9, 6.8,
       7.6, 2.5, 8.1, 7.3, 7.5, 7.5, 9.3, 5.6, 5.2, 4.7, 4.5, 8.7,
       8.7, 1.3, 4.1, 4.5, 4.9, 6.5, 7.9, 4.6, 7.0, 8.0, 1.6, 3.5});
  optional<Tensor> bias(tf.make({2}, {1.0, 1.0}));
  Tensor expected = tf.make_channels_last(
      {1, 2, 4, 4},
      {624.92, 656.07, 710.91,  800.45,  622.48,  596.14,  831.26,  882.43,
       812.8,  947.49, 1069.65, 1155.81, 964.84,  1057.19, 1121.77, 1328.68,
       748.23, 799.7,  1090.23, 1203.45, 1043.71, 1124.75, 1140.41, 1265.35,
       688.62, 807.57, 1073.07, 1109.53, 1110,    1221.82, 1210.86, 1324.26});
  Tensor out = tf.full_channels_last({1, 2, 4, 4}, 0);

  int64_t stride[] = {2, 2};
  int64_t padding[] = {1, 1};
  int64_t dilation[] = {1, 1};
  int64_t output_padding[] = {0};

  convolution_out(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      output_padding,
      1,
      out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

/* %python
import torch
torch.manual_seed(0)
input = (torch.randint(10, 100, (1, 2, 5)).to(torch.double) / 10.0);
weight = (torch.randint(10, 100, (4, 2, 3)).to(torch.double) / 10.0);
bias = torch.ones(4).to(torch.double)
stride = [2]
padding = [0]
dilation = [1]
transposed = False
output_padding = [0]
groups = 1
expected = torch.nn.functional.conv1d(
    input, weight, bias, stride, padding, dilation, groups)

DTYPE = "Float"
out_size = "out_shape"
dynamism = "dynamism"
*/

void test_dynamic_shape(
    const std::vector<int32_t>& out_shape,
    enum torch::executor::TensorShapeDynamism dynamism) {
  /* %python
  %past-rewrite(correctness_test_template) */

  TensorFactory<ScalarType::Float> tf;

  Tensor input =
      tf.make({1, 2, 5}, {5.4, 1.9, 9.3, 7.0, 5.3, 7.9, 1.7, 8.3, 4.7, 7.3});
  Tensor weight = tf.make(
      {4, 2, 3}, {8.1, 6.6, 1.6, 4.9, 3.8, 6.6, 4.6, 2.8, 2.4, 1.3, 3.6, 3.9,
                  8.1, 8.4, 5.4, 5.1, 8.9, 9.9, 7.9, 1.0, 1.1, 8.2, 6.3, 7.0});
  optional<Tensor> bias(tf.make({4}, {1.0, 1.0, 1.0, 1.0}));
  Tensor expected = tf.make(
      {1, 4, 2},
      {172.11, 237.72, 102.24, 132.28, 248.51, 320.18, 189.38, 236.07});
  Tensor out = tf.zeros(out_shape, dynamism);

  int64_t stride[] = {2};
  int64_t padding[] = {0};
  int64_t dilation[] = {1};
  int64_t output_padding[] = {0};

  convolution_out(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      false,
      output_padding,
      1,
      out);
  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST(OpConvOut, DynamicShapeUpperBoundSameAsExpected) {
  test_dynamic_shape(
      {1, 4, 2}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpConvOut, DynamicShapeUpperBoundLargerThanExpected) {
  test_dynamic_shape(
      {10, 10, 10}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
}

TEST(OpConvOut, DynamicShapeUnbound) {
  if (!torch::executor::testing::SupportedFeatures::get()->output_resize) {
    GTEST_SKIP() << "Dynamic shape unbound not supported";
  }
  test_dynamic_shape(
      {1, 1, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_UNBOUND);
}
