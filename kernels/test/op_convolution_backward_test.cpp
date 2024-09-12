/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using IntArrayRef = exec_aten::ArrayRef<int64_t>;
using OptIntArrayRef = exec_aten::OptionalArrayRef<int64_t>;
using torch::executor::testing::TensorFactory;

class OpConvolutionBackwardOutTest : public OperatorTest {
 protected:
  std::tuple<Tensor&, Tensor&, Tensor&> op_convolution_backward_out(
      const Tensor& grad_output,
      const Tensor& input,
      const Tensor& weight,
      const OptIntArrayRef bias_sizes_opt,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      bool transposed,
      IntArrayRef output_padding,
      int64_t groups,
      std::array<bool, 3> output_mask_a,
      Tensor& grad_input,
      Tensor& grad_weight,
      Tensor& grad_bias) {
#ifndef USE_ATEN_LIB
    ArrayRef<bool> output_mask(output_mask_a.data(), output_mask_a.size());
#else
    std::array<bool, 3> output_mask = output_mask_a;
#endif
    return torch::executor::aten::convolution_backward_outf(
        context_,
        grad_output,
        input,
        weight,
        bias_sizes_opt,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
        grad_input,
        grad_weight,
        grad_bias);
  }
};

TEST_F(OpConvolutionBackwardOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tf;

  std::vector<float> grad_output_data = {
      10, 12, 87, 13, 34, 87, 55, 22, 48, 33, 29, 38, 60, 49, 88, 30,
      99, 19, 42, 37, 61, 31, 33, 58, 38, 23, 2,  33, 3,  21, 32, 2,
      30, 72, 10, 67, 92, 19, 11, 16, 65, 37, 60, 74, 4,  19, 45, 37};
  std::vector<float> input_data = {
      9,  89, 45, 39, 25, 2,  97, 55, 80, 24, 18, 33, 28, 89, 19, 16, 19, 33,
      69, 61, 34, 84, 58, 30, 33, 18, 75, 30, 6,  33, 42, 10, 80, 41, 66, 64,
      47, 51, 67, 62, 58, 10, 97, 71, 24, 44, 84, 34, 33, 54, 8,  73, 90, 15,
      21, 92, 55, 22, 56, 12, 10, 63, 32, 76, 65, 38, 95, 92, 22, 15, 37, 12,
      67, 14, 60, 44, 73, 74, 23, 4,  56, 64, 88, 90, 82, 32, 91, 3,  6,  87,
      55, 95, 7,  14, 24, 69, 52, 44, 14, 37, 75, 52, 37, 40, 25, 54, 4,  15,
      97, 51, 46, 28, 65, 95, 50, 82, 23, 39, 50, 55, 97, 52, 91, 16, 19, 49,
      61, 50, 42, 47, 87, 99, 9,  60, 22, 71, 47, 17, 0,  80, 28, 88, 93, 43,
      65, 25, 88, 67, 21, 89, 24, 81, 3,  71, 20, 34, 17, 17, 94, 10, 82, 25,
      10, 11, 7,  28, 77, 39, 74, 79, 17, 40, 67, 54, 49, 54, 21, 89, 17, 7,
      52, 64, 68, 80, 7,  72, 44, 35, 92, 47, 4,  13, 10, 43, 64, 66, 83, 49,
      81, 78, 58, 22, 86, 48, 35, 64, 98, 79, 8,  52, 56, 23, 38, 74, 16, 63,
      51, 70, 44, 28, 43, 13, 51, 85, 42, 29, 64, 26, 54, 91, 9,  96, 41, 56,
      7,  52, 27, 22, 69, 13, 8,  20, 22, 49, 66, 98, 77, 42, 54, 38, 70, 83,
      13, 8,  21, 56, 78, 37, 28, 69, 42, 30, 91, 5,  28, 15, 20, 14, 16, 39,
      95, 66, 4,  72, 52, 35, 54, 93, 87, 77, 3,  49, 82, 70, 84, 3,  73, 99,
      32, 95, 58, 65, 32, 75, 34, 22, 12, 84, 63, 72, 85, 66, 63, 27, 3,  73,
      45, 37, 61, 52, 41, 16, 37, 14, 80, 17, 48, 8,  87, 98, 69, 63, 92, 68,
      42, 63, 5,  22, 66, 91, 74, 11, 17, 45, 45, 33, 40, 85, 26, 75, 73, 81,
      54, 27, 80, 1,  44, 66, 10, 21, 15, 10, 76, 96, 0,  43, 39, 3,  57, 79,
      45, 64, 58, 92, 44, 42, 7,  28, 94, 4,  8,  22, 22, 31, 75, 44, 3,  70,
      83, 72, 87, 12, 20, 55, 84, 31, 50, 34, 25, 49, 29, 71, 57, 97, 25, 82,
      84, 42, 86, 41, 54, 92, 34, 30, 52, 34, 84, 25, 54, 37, 38, 26, 76, 82,
      34, 14, 85, 28, 93, 9};
  std::vector<float> weight_data = {
      2,  54, 9,  37, 0,  47, 70, 9,  84,  69, 56, 79, 25, 35, 54, 13,
      65, 46, 38, 28, 74, 27, 66, 61, 20,  60, 62, 58, 15, 44, 75, 55,
      7,  52, 13, 36, 39, 64, 62, 45, 100, 6,  79, 63, 63, 52, 37, 60,
      78, 12, 69, 2,  74, 56, 93, 39, 62,  22, 55, 67, 68, 74, 12, 69,
      15, 73, 28, 70, 86, 20, 90, 49, 52,  26, 58, 2,  82, 17, 70, 55,
      54, 83, 70, 11, 27, 9,  5,  42, 34,  62, 29, 94, 69, 81, 54, 4};
  std::vector<float> expected_grad_input_data = {
      1134,  7578,  686,   2682,  0, 4148,  7136,  2406,  8698, 0,
      3759,  6003,  2163,  2395,  0, 2929,  5830,  3469,  6955, 0,
      720,   6201,  495,   2063,  0, 5260,  5989,  3060,  7079, 0,
      9690,  3423,  3385,  1932,  0, 7644,  8499,  1323,  2613, 0,
      4334,  6624,  8532,  9719,  0, 5496,  8601,  1157,  2215, 0,
      4676,  7600,  6524,  10069, 0, 4047,  6117,  1612,  2567, 0,
      5931,  5651,  5669,  6623,  0, 7674,  3291,  2748,  1654, 0,
      10455, 4290,  4145,  796,   0, 9835,  5483,  11649, 5952, 0,
      7098,  5460,  3101,  2443,  0, 7788,  5909,  8582,  6298, 0,
      9462,  4845,  3041,  2067,  0, 7038,  6336,  10438, 6377, 0,
      7518,  8187,  2079,  2773,  0, 10036, 2642,  3952,  1166, 0,
      16014, 2250,  10025, 1908,  0, 9610,  298,   3868,  122,  0,
      16629, 4338,  11335, 3527,  0, 11514, 5965,  4762,  2207, 0,
      18552, 10755, 13309, 5996,  0, 12454, 6787,  4960,  2875, 0,
      8750,  6999,  3534,  3233,  0, 14160, 9399,  9595,  8922, 0,
      9110,  6567,  3820,  2351,  0, 12969, 11814, 9436,  5870, 0,
      7631,  7061,  2877,  2499,  0, 8553,  13527, 3631,  6863, 0,
      1361,  8634,  515,   3372,  0, 3394,  10206, 1504,  4112, 0,
      5505,  17421, 4702,  11891, 0, 4233,  11894, 1739,  5014, 0,
      11787, 14634, 8981,  10759, 0, 11777, 6701,  4719,  3111, 0,
      18459, 7761,  12044, 7627,  0, 11214, 4556,  4374,  1594, 0,
      604,   1908,  1506,  6102,  0, 2532,  4024,  1713,  6121, 0,
      1878,  1814,  4761,  5397,  0, 1127,  3885,  4373,  5832, 0,
      450,   1414,  1080,  4719,  0, 5210,  2683,  2765,  4252, 0,
      2390,  1668,  7710,  4257,  0, 378,   1698,  3276,  6021, 0,
      2866,  4881,  3547,  6822,  0, 502,   1238,  2784,  5199, 0,
      2496,  3975,  2700,  5004,  0, 1220,  1990,  3633,  5763, 0,
      4501,  2679,  4504,  5412,  0, 1968,  1376,  6246,  3669, 0,
      3130,  272,   9345,  1950,  0, 5167,  3278,  9097,  2138, 0,
      2446,  1946,  6942,  5460,  0, 5732,  3404,  7919,  5534, 0,
      2038,  1614,  6978,  4635,  0, 4544,  4839,  7367,  5574, 0,
      1242,  1922,  4842,  6333,  0, 1066,  236,   2236,  686,  0,
      17238, 2254,  10413, 1592,  0, 991,   30,    2206,  70,   0,
      18823, 6392,  12173, 2470,  0, 1142,  684,   2742,  1219, 0,
      21256, 11293, 12719, 7512,  0, 1303,  649,   2818,  1669, 0,
      898,   574,   2018,  1929,  0, 15720, 11989, 10517, 5972, 0,
      885,   781,   2210,  1281,  0, 14601, 12198, 7915,  4958, 0,
      856,   850,   1601,  1355,  0, 7039,  14083, 4113,  7490, 0,
      152,   927,   287,   1902,  0, 301,   1051,  886,   2346, 0,
      6821,  19615, 4491,  13281, 0, 424,   1146,  999,   2906, 0,
      15177, 15480, 8849,  12442, 0, 1222,  544,   2687,  1859, 0,
      20215, 9693,  11441, 4964,  0, 1206,  555,   2466,  860,  0};
  std::vector<float> expected_grad_weight_data = {
      9246,  22073, 12431, 19714, 11179, 19032, 8458,  6495,  18707, 13830,
      20445, 17089, 17124, 18710, 11827, 17236, 16824, 9008,  14086, 18834,
      17419, 16759, 13152, 9339,  13801, 20888, 13976, 27277, 13010, 23949,
      9838,  11220, 17658, 15019, 25337, 17583, 13270, 21754, 16908, 20563,
      20732, 13413, 20868, 27521, 19537, 21170, 15888, 10034, 19195, 16370,
      40243, 25890, 40472, 30460, 21228, 21625, 13289, 24435, 19876, 29816,
      24188, 23619, 13752, 16251, 18741, 19368, 24517, 34261, 27054, 31257,
      21238, 18909, 15776, 16881, 34604, 22534, 28101, 23834, 18479, 16469,
      12852, 16551, 14204, 29983, 20167, 24150, 14281, 17501, 15897, 16019,
      21661, 32765, 23874, 26527, 20463, 18661};
  std::vector<float> expected_grad_bias_data = {363, 438, 585, 501};

  auto grad_output = tf.make({2, 4, 3, 2}, grad_output_data);
  auto input = tf.make({2, 6, 7, 5}, input_data);
  auto weight = tf.make({4, 3, 4, 2}, weight_data);
  int64_t bias_sizes[1] = {4};
  int64_t stride[2] = {1, 2};
  int64_t padding[2] = {1, 0};
  int64_t dilation[2] = {2, 1};
  bool transposed = false;
  int64_t output_padding[2] = {0, 0};
  int64_t groups = 2;
  std::array<bool, 3> output_mask_a = {true, true, true};
  auto grad_input = tf.zeros({2, 6, 7, 5});
  auto grad_weight = tf.zeros({4, 3, 4, 2});
  auto grad_bias = tf.zeros({4});

  op_convolution_backward_out(
      grad_output,
      input,
      weight,
      IntArrayRef{bias_sizes, 1},
      IntArrayRef{stride, 2},
      IntArrayRef{padding, 2},
      IntArrayRef{dilation, 2},
      transposed,
      IntArrayRef{output_padding, 2},
      groups,
      output_mask_a,
      grad_input,
      grad_weight,
      grad_bias);

  auto expected_grad_input = tf.make({2, 6, 7, 5}, expected_grad_input_data);
  auto expected_grad_weight = tf.make({4, 3, 4, 2}, expected_grad_weight_data);
  auto expected_grad_bias = tf.make({4}, expected_grad_bias_data);

  EXPECT_TENSOR_CLOSE(grad_input, expected_grad_input);
  EXPECT_TENSOR_CLOSE(grad_weight, expected_grad_weight);
  EXPECT_TENSOR_CLOSE(grad_bias, expected_grad_bias);
}
