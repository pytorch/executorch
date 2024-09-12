/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <limits>

#include <executorch/extension/llm/custom_ops/op_sdpa.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::testing::TensorFactory;

exec_aten::Tensor op_sdpa_with_kv_cache(
    const exec_aten::Tensor& query,
    const exec_aten::Tensor& key,
    const exec_aten::Tensor& value,
    exec_aten::Tensor& key_cache,
    exec_aten::Tensor& value_cache,
    const int64_t start_pos,
    const int64_t seq_len,
    const exec_aten::optional<exec_aten::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    exec_aten::optional<double> scale,
    exec_aten::Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::native::sdpa_with_kv_cache_out(
      context,
      query,
      key,
      value,
      key_cache,
      value_cache,
      start_pos,
      seq_len,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
}

/*
SDPA with cache is equivalent of the following code
# q, = (batch size, q seq len, num heads, head dim)
# k, v = (batch size, kv seq len, num heads, head dim)
# k cache, v cache = (num layers, batch size, max seq length, num heads, head
dim) # attn_mask = [max seq length, max seq length]

def sdpa_with_cache(q, k, v, k_cache, v_cache, start_pos, attn_mask):
    attn_mask = attn_mask[start_pos].view((1, -1))
    q = q.transpose(1, 2)
    k_cache[:, start_pos] = k
    v_cache[:, start_pos] = v
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    sliced_k_cache = k_cache
    sliced_v_cache = v_cache
    sliced_k_cache = sliced_k_cache.transpose(1, 2)
    sliced_v_cache = sliced_v_cache.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, sliced_k_cache, sliced_v_cache,
attn_mask=attn_mask)
    out = out.transpose(1, 2)
*/

/*
Missing tests:
1. Test for different batch sizes
2. Mix 2 with attention_mask
3. No bool attention_mask
4. apply scaling
5. Different dtypes, fp16, bf16, double (or expect throw)
*/
TEST(OpScaledDotProductAttentionTest, BasicTest) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {1, 1, 4, 4},
      {0.8823,
       0.9150,
       0.3829,
       0.9593,
       0.3904,
       0.6009,
       0.2566,
       0.7936,
       0.9408,
       0.1332,
       0.9346,
       0.5936,
       0.8694,
       0.5677,
       0.7411,
       0.4294});
  exec_aten::Tensor key = tfFloat.make(
      {1, 1, 4, 4},
      {0.8854,
       0.5739,
       0.2666,
       0.6274,
       0.2696,
       0.4414,
       0.2969,
       0.8317,
       0.1053,
       0.2695,
       0.3588,
       0.1994,
       0.5472,
       0.0062,
       0.9516,
       0.0753});
  exec_aten::Tensor value = tfFloat.make(
      {1, 1, 4, 4},
      {0.8860,
       0.5832,
       0.3376,
       0.8090,
       0.5779,
       0.9040,
       0.5547,
       0.3423,
       0.6343,
       0.3644,
       0.7104,
       0.9464,
       0.7890,
       0.2814,
       0.7886,
       0.5895});
  exec_aten::Tensor key_cache_0 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor value_cache_0 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor key_cache_1 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor value_cache_1 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor key_cache_2 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor value_cache_2 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::optional<exec_aten::Tensor> attn_mask;
  double dropout_p = 0;
  bool is_causal = false;
  exec_aten::optional<double> scale;

  // start pos: 0 layer id 0
  exec_aten::Tensor ret_expected_0 = tfFloat.make(
      {1, 1, 4, 4},
      {0.8860,
       0.5832,
       0.3376,
       0.8090,
       0.5779,
       0.9040,
       0.5547,
       0.3423,
       0.6343,
       0.3644,
       0.7104,
       0.9464,
       0.7890,
       0.2814,
       0.7886,
       0.5895});

  std::vector<int32_t> out_size = {1, 1, 4, 4};
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      0,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_0, 1e-4, 1e-4);

  // start pos: 0 layer id 2
  exec_aten::Tensor ret_expected_1 = tfFloat.make(
      {1, 1, 4, 4},
      {0.8860,
       0.5832,
       0.3376,
       0.8090,
       0.5779,
       0.9040,
       0.5547,
       0.3423,
       0.6343,
       0.3644,
       0.7104,
       0.9464,
       0.7890,
       0.2814,
       0.7886,
       0.5895});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_2,
      value_cache_2,
      0,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_1, 1e-4, 1e-4);

  // start pos: 1 layer id 0
  exec_aten::Tensor ret_expected_2 = tfFloat.make(
      {1, 1, 4, 4},
      {0.8860,
       0.5832,
       0.3376,
       0.8090,
       0.5779,
       0.9040,
       0.5547,
       0.3423,
       0.6343,
       0.3644,
       0.7104,
       0.9464,
       0.7890,
       0.2814,
       0.7886,
       0.5895});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      1,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_2, 1e-4, 1e-4);

  // start pos: 1 layer id 1
  exec_aten::Tensor ret_expected_3 = tfFloat.make(
      {1, 1, 4, 4},
      {0.6486,
       0.4270,
       0.2472,
       0.5922,
       0.3669,
       0.5740,
       0.3522,
       0.2173,
       0.3635,
       0.2088,
       0.4071,
       0.5423,
       0.5110,
       0.1822,
       0.5107,
       0.3817});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_1,
      value_cache_1,
      1,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_3, 1e-4, 1e-4);

  // start pos: 2 layer id 1
  exec_aten::Tensor ret_expected_4 = tfFloat.make(
      {1, 1, 4, 4},
      {0.7490,
       0.4930,
       0.2854,
       0.6838,
       0.4489,
       0.7021,
       0.4308,
       0.2659,
       0.4622,
       0.2655,
       0.5176,
       0.6895,
       0.6202,
       0.2212,
       0.6199,
       0.4634});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_1,
      value_cache_1,
      2,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_4, 1e-4, 1e-4);

  // start pos: 2 layer id 2
  exec_aten::Tensor ret_expected_5 = tfFloat.make(
      {1, 1, 4, 4},
      {0.7490,
       0.4930,
       0.2854,
       0.6838,
       0.4489,
       0.7021,
       0.4308,
       0.2659,
       0.4622,
       0.2655,
       0.5176,
       0.6895,
       0.6202,
       0.2212,
       0.6199,
       0.4634});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_2,
      value_cache_2,
      2,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_5, 1e-4, 1e-4);
}

TEST(OpScaledDotProductAttentionTest, LargerTest) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {1, 1, 7, 4}, {0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566,
                     0.7936, 0.9408, 0.1332, 0.9346, 0.5936, 0.8694, 0.5677,
                     0.7411, 0.4294, 0.8854, 0.5739, 0.2666, 0.6274, 0.2696,
                     0.4414, 0.2969, 0.8317, 0.1053, 0.2695, 0.3588, 0.1994});
  exec_aten::Tensor key = tfFloat.make(
      {1, 1, 7, 4}, {0.5472, 0.0062, 0.9516, 0.0753, 0.8860, 0.5832, 0.3376,
                     0.8090, 0.5779, 0.9040, 0.5547, 0.3423, 0.6343, 0.3644,
                     0.7104, 0.9464, 0.7890, 0.2814, 0.7886, 0.5895, 0.7539,
                     0.1952, 0.0050, 0.3068, 0.1165, 0.9103, 0.6440, 0.7071});
  exec_aten::Tensor value = tfFloat.make(
      {1, 1, 7, 4}, {0.6581, 0.4913, 0.8913, 0.1447, 0.5315, 0.1587, 0.6542,
                     0.3278, 0.6532, 0.3958, 0.9147, 0.2036, 0.2018, 0.2018,
                     0.9497, 0.6666, 0.9811, 0.0874, 0.0041, 0.1088, 0.1637,
                     0.7025, 0.6790, 0.9155, 0.2418, 0.1591, 0.7653, 0.2979});
  exec_aten::Tensor key_cache_0 = tfFloat.zeros({1, 8, 7, 4});
  exec_aten::Tensor value_cache_0 = tfFloat.zeros({1, 8, 7, 4});
  exec_aten::Tensor key_cache_1 = tfFloat.zeros({1, 8, 7, 4});
  exec_aten::Tensor value_cache_1 = tfFloat.zeros({1, 8, 7, 4});
  exec_aten::Tensor key_cache_2 = tfFloat.zeros({1, 8, 7, 4});
  exec_aten::Tensor value_cache_2 = tfFloat.zeros({1, 8, 7, 4});
  exec_aten::optional<exec_aten::Tensor> attn_mask;
  double dropout_p = 0;
  bool is_causal = false;
  exec_aten::optional<double> scale;

  // start pos: 0 layer id 0
  exec_aten::Tensor ret_expected_0 = tfFloat.make(
      {1, 1, 7, 4}, {0.6581, 0.4913, 0.8913, 0.1447, 0.5315, 0.1587, 0.6542,
                     0.3278, 0.6532, 0.3958, 0.9147, 0.2036, 0.2018, 0.2018,
                     0.9497, 0.6666, 0.9811, 0.0874, 0.0041, 0.1088, 0.1637,
                     0.7025, 0.6790, 0.9155, 0.2418, 0.1591, 0.7653, 0.2979});

  std::vector<int32_t> out_size = {1, 1, 7, 4};
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      0,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_0, 1e-4, 1e-4);

  // start pos: 0 layer id 2
  exec_aten::Tensor ret_expected_1 = tfFloat.make(
      {1, 1, 7, 4}, {0.6581, 0.4913, 0.8913, 0.1447, 0.5315, 0.1587, 0.6542,
                     0.3278, 0.6532, 0.3958, 0.9147, 0.2036, 0.2018, 0.2018,
                     0.9497, 0.6666, 0.9811, 0.0874, 0.0041, 0.1088, 0.1637,
                     0.7025, 0.6790, 0.9155, 0.2418, 0.1591, 0.7653, 0.2979});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_2,
      value_cache_2,
      0,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_1, 1e-4, 1e-4);

  // start pos: 1 layer id 0
  exec_aten::Tensor ret_expected_2 = tfFloat.make(
      {1, 1, 7, 4}, {0.6581, 0.4913, 0.8913, 0.1447, 0.5315, 0.1587, 0.6542,
                     0.3278, 0.6532, 0.3958, 0.9147, 0.2036, 0.2018, 0.2018,
                     0.9497, 0.6666, 0.9811, 0.0874, 0.0041, 0.1088, 0.1637,
                     0.7025, 0.6790, 0.9155, 0.2418, 0.1591, 0.7653, 0.2979});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      1,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_2, 1e-4, 1e-4);

  // start pos: 1 layer id 1
  exec_aten::Tensor ret_expected_3 = tfFloat.make(
      {1, 1, 7, 4}, {0.4038, 0.3015, 0.5469, 0.0888, 0.3566, 0.1065, 0.4389,
                     0.2199, 0.4354, 0.2639, 0.6097, 0.1358, 0.1412, 0.1412,
                     0.6645, 0.4664, 0.6599, 0.0588, 0.0027, 0.0732, 0.0929,
                     0.3989, 0.3856, 0.5198, 0.1398, 0.0920, 0.4424, 0.1722});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_1,
      value_cache_1,
      1,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_3, 1e-4, 1e-4);

  // start pos: 2 layer id 1
  exec_aten::Tensor ret_expected_4 = tfFloat.make(
      {1, 1, 7, 4}, {0.5005, 0.3737, 0.6779, 0.1101, 0.4268, 0.1275, 0.5254,
                     0.2633, 0.5225, 0.3166, 0.7317, 0.1629, 0.1661, 0.1661,
                     0.7819, 0.5488, 0.7891, 0.0703, 0.0033, 0.0875, 0.1185,
                     0.5089, 0.4919, 0.6631, 0.1771, 0.1166, 0.5607, 0.2182});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_1,
      value_cache_1,
      2,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_4, 1e-4, 1e-4);

  // start pos: 2 layer id 2
  exec_aten::Tensor ret_expected_5 = tfFloat.make(
      {1, 1, 7, 4}, {0.5005, 0.3737, 0.6779, 0.1101, 0.4268, 0.1275, 0.5254,
                     0.2633, 0.5225, 0.3166, 0.7317, 0.1629, 0.1661, 0.1661,
                     0.7819, 0.5488, 0.7891, 0.0703, 0.0033, 0.0875, 0.1185,
                     0.5089, 0.4919, 0.6631, 0.1771, 0.1166, 0.5607, 0.2182});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_2,
      value_cache_2,
      2,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_5, 1e-4, 1e-4);
}

TEST(OpScaledDotProductAttentionTest, BasicTestWithAttnMask) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {1, 1, 4, 4},
      {0.8823,
       0.9150,
       0.3829,
       0.9593,
       0.3904,
       0.6009,
       0.2566,
       0.7936,
       0.9408,
       0.1332,
       0.9346,
       0.5936,
       0.8694,
       0.5677,
       0.7411,
       0.4294});
  exec_aten::Tensor key = tfFloat.make(
      {1, 1, 4, 4},
      {0.8854,
       0.5739,
       0.2666,
       0.6274,
       0.2696,
       0.4414,
       0.2969,
       0.8317,
       0.1053,
       0.2695,
       0.3588,
       0.1994,
       0.5472,
       0.0062,
       0.9516,
       0.0753});
  exec_aten::Tensor value = tfFloat.make(
      {1, 1, 4, 4},
      {0.8860,
       0.5832,
       0.3376,
       0.8090,
       0.5779,
       0.9040,
       0.5547,
       0.3423,
       0.6343,
       0.3644,
       0.7104,
       0.9464,
       0.7890,
       0.2814,
       0.7886,
       0.5895});
  exec_aten::Tensor attn_mask = tfFloat.make({1, 1}, {0});
  exec_aten::Tensor key_cache_0 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor value_cache_0 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor key_cache_1 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor value_cache_1 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor key_cache_2 = tfFloat.zeros({1, 5, 4, 4});
  exec_aten::Tensor value_cache_2 = tfFloat.zeros({1, 5, 4, 4});
  double dropout_p = 0;
  bool is_causal = false;
  exec_aten::optional<double> scale;

  // start pos: 0 layer id 0
  exec_aten::Tensor ret_expected_0 = tfFloat.make(
      {1, 1, 4, 4},
      {0.8860,
       0.5832,
       0.3376,
       0.8090,
       0.5779,
       0.9040,
       0.5547,
       0.3423,
       0.6343,
       0.3644,
       0.7104,
       0.9464,
       0.7890,
       0.2814,
       0.7886,
       0.5895});

  std::vector<int32_t> out_size = {1, 1, 4, 4};
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      0,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_0, 1e-4, 1e-4);

  // start pos: 0 layer id 2
  exec_aten::Tensor ret_expected_1 = tfFloat.make(
      {1, 1, 4, 4},
      {0.8860,
       0.5832,
       0.3376,
       0.8090,
       0.5779,
       0.9040,
       0.5547,
       0.3423,
       0.6343,
       0.3644,
       0.7104,
       0.9464,
       0.7890,
       0.2814,
       0.7886,
       0.5895});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_2,
      value_cache_2,
      0,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_1, 1e-4, 1e-4);

  attn_mask = tfFloat.make({1, 2}, {0, 0});
  // start pos: 1 layer id 0
  exec_aten::Tensor ret_expected_2 = tfFloat.make(
      {1, 1, 4, 4},
      {0.8860,
       0.5832,
       0.3376,
       0.8090,
       0.5779,
       0.9040,
       0.5547,
       0.3423,
       0.6343,
       0.3644,
       0.7104,
       0.9464,
       0.7890,
       0.2814,
       0.7886,
       0.5895});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      1,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_2, 1e-4, 1e-4);

  // start pos: 1 layer id 1
  exec_aten::Tensor ret_expected_3 = tfFloat.make(
      {1, 1, 4, 4},
      {0.6486,
       0.4270,
       0.2472,
       0.5922,
       0.3669,
       0.5740,
       0.3522,
       0.2173,
       0.3635,
       0.2088,
       0.4071,
       0.5423,
       0.5110,
       0.1822,
       0.5107,
       0.3817});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_1,
      value_cache_1,
      1,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_3, 1e-4, 1e-4);

  attn_mask = tfFloat.make({1, 3}, {0, 0, 0});
  // start pos: 2 layer id 1
  exec_aten::Tensor ret_expected_4 = tfFloat.make(
      {1, 1, 4, 4},
      {0.7490,
       0.4930,
       0.2854,
       0.6838,
       0.4489,
       0.7021,
       0.4308,
       0.2659,
       0.4622,
       0.2655,
       0.5176,
       0.6895,
       0.6202,
       0.2212,
       0.6199,
       0.4634});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_1,
      value_cache_1,
      2,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_4, 1e-4, 1e-4);

  // start pos: 2 layer id 2
  exec_aten::Tensor ret_expected_5 = tfFloat.make(
      {1, 1, 4, 4},
      {0.7490,
       0.4930,
       0.2854,
       0.6838,
       0.4489,
       0.7021,
       0.4308,
       0.2659,
       0.4622,
       0.2655,
       0.5176,
       0.6895,
       0.6202,
       0.2212,
       0.6199,
       0.4634});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_2,
      value_cache_2,
      2,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_5, 1e-4, 1e-4);
}

TEST(OpScaledDotProductAttentionTest, SequenceTest) {
  TensorFactory<exec_aten::ScalarType::Float> tfFloat;

  exec_aten::Tensor query = tfFloat.make(
      {1, 1, 8, 4},
      {0.1261, 0.5031, 0.1117, 0.3905, 0.3625, 0.9328, 0.6549, 0.4128,
       0.5845, 0.3557, 0.6965, 0.6978, 0.6343, 0.3051, 0.9266, 0.4278,
       0.3053, 0.8132, 0.9075, 0.9976, 0.6481, 0.3296, 0.7539, 0.9290,
       0.0096, 0.4381, 0.1590, 0.5932, 0.7068, 0.3967, 0.4582, 0.7251});
  exec_aten::Tensor key = tfFloat.make(
      {1, 1, 8, 4},
      {0.4160, 0.0801, 0.9001, 0.2483, 0.4451, 0.5472, 0.4700, 0.0297,
       0.7294, 0.2729, 0.2407, 0.6195, 0.2391, 0.2689, 0.3315, 0.3122,
       0.2912, 0.3652, 0.6299, 0.0954, 0.1974, 0.5073, 0.5695, 0.7761,
       0.1488, 0.6596, 0.7842, 0.7776, 0.0343, 0.3092, 0.0702, 0.1836});
  exec_aten::Tensor value = tfFloat.make(
      {1, 1, 8, 4},
      {0.7785, 0.4253, 0.7124, 0.2065, 0.5760, 0.1976, 0.7499, 0.2813,
       0.3746, 0.0662, 0.5017, 0.9747, 0.7427, 0.2332, 0.5067, 0.4452,
       0.0975, 0.8920, 0.5081, 0.6053, 0.2981, 0.2660, 0.5824, 0.6849,
       0.6121, 0.2590, 0.9854, 0.4264, 0.1938, 0.2661, 0.9922, 0.5000});

  exec_aten::Tensor key_cache_0 = tfFloat.zeros({1, 5, 8, 4});
  exec_aten::Tensor value_cache_0 = tfFloat.zeros({1, 5, 8, 4});

  exec_aten::optional<exec_aten::Tensor> attn_mask;
  double dropout_p = 0;
  bool is_causal = false;
  exec_aten::optional<double> scale;

  // start pos: 0 layer id 0
  exec_aten::Tensor ret_expected_0 = tfFloat.make(
      {1, 1, 8, 4},
      {0.7785, 0.4253, 0.7124, 0.2065, 0.5760, 0.1976, 0.7499, 0.2813,
       0.3746, 0.0662, 0.5017, 0.9747, 0.7427, 0.2332, 0.5067, 0.4452,
       0.0975, 0.8920, 0.5081, 0.6053, 0.2981, 0.2660, 0.5824, 0.6849,
       0.6121, 0.2590, 0.9854, 0.4264, 0.1938, 0.2661, 0.9922, 0.5000});

  std::vector<int32_t> out_size = {1, 1, 8, 4};
  exec_aten::Tensor out = tfFloat.zeros(out_size);
  exec_aten::Tensor ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      0,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_0, 1e-4, 1e-4);

  // start pos: 1 layer id 0
  query = tfFloat.make(
      {1, 1, 8, 4},
      {0.4321, 0.2919, 0.3689, 0.0789, 0.1027, 0.7926, 0.9277, 0.9772,
       0.1390, 0.7704, 0.1905, 0.7983, 0.8608, 0.8869, 0.8600, 0.8128,
       0.5097, 0.7297, 0.3211, 0.7177, 0.3393, 0.4916, 0.0648, 0.3693,
       0.2371, 0.3313, 0.1807, 0.0503, 0.5326, 0.8245, 0.9554, 0.7918});
  key = tfFloat.make(
      {1, 1, 8, 4},
      {0.2408, 0.0055, 0.6897, 0.7802, 0.0707, 0.6793, 0.9227, 0.5303,
       0.1988, 0.9099, 0.7135, 0.8311, 0.1619, 0.7910, 0.1585, 0.9947,
       0.2882, 0.8013, 0.6001, 0.6325, 0.4233, 0.7054, 0.2916, 0.0287,
       0.3079, 0.8918, 0.3684, 0.6572, 0.3151, 0.8751, 0.7992, 0.6765});
  value = tfFloat.make(
      {1, 1, 8, 4},
      {0.2444, 0.0914, 0.5188, 0.2067, 0.9111, 0.0195, 0.7234, 0.9985,
       0.7504, 0.6705, 0.0189, 0.9809, 0.4145, 0.0328, 0.9936, 0.2965,
       0.4646, 0.9576, 0.1534, 0.1463, 0.5813, 0.4331, 0.6152, 0.0806,
       0.5150, 0.2776, 0.2542, 0.0422, 0.7651, 0.5963, 0.0773, 0.8968});
  exec_aten::Tensor ret_expected_1 = tfFloat.make(
      {1, 1, 8, 4},
      {0.5203, 0.2639, 0.6188, 0.2066, 0.7836, 0.0872, 0.7335, 0.7256,
       0.5940, 0.4189, 0.2199, 0.9784, 0.5461, 0.1132, 0.7983, 0.3561,
       0.3125, 0.9305, 0.3003, 0.3364, 0.4355, 0.3471, 0.5983, 0.3918,
       0.5631, 0.2684, 0.6168, 0.2327, 0.5942, 0.4976, 0.3510, 0.7781});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      1,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_1, 1e-4, 1e-4);

  // start pos: 2 layer id 0
  query = tfFloat.make(
      {1, 1, 8, 4},
      {0.6508, 0.5928, 0.2064, 0.5754, 0.9818, 0.8429, 0.1106, 0.9564,
       0.5388, 0.7405, 0.8883, 0.9263, 0.1102, 0.9378, 0.1604, 0.5375,
       0.1506, 0.3904, 0.4773, 0.4402, 0.4210, 0.5394, 0.9932, 0.7905,
       0.7797, 0.7001, 0.8871, 0.4769, 0.5398, 0.6029, 0.0639, 0.0972});
  key = tfFloat.make(
      {1, 1, 8, 4},
      {0.5613, 0.3044, 0.4908, 0.3853, 0.5778, 0.8253, 0.3342, 0.9004,
       0.8948, 0.1163, 0.1139, 0.0955, 0.2260, 0.3054, 0.4624, 0.3784,
       0.2474, 0.3412, 0.3191, 0.9905, 0.3147, 0.1420, 0.7078, 0.4711,
       0.8828, 0.8124, 0.9594, 0.1338, 0.8214, 0.9196, 0.2531, 0.9596});
  value = tfFloat.make(
      {1, 1, 8, 4},
      {0.8748, 0.5055, 0.7411, 0.3252, 0.0639, 0.6264, 0.6491, 0.1732,
       0.7425, 0.0729, 0.9303, 0.9842, 0.6361, 0.1863, 0.7433, 0.5852,
       0.6360, 0.6643, 0.8807, 0.2851, 0.3875, 0.6364, 0.5545, 0.9032,
       0.2374, 0.4818, 0.5934, 0.3672, 0.8409, 0.5547, 0.0379, 0.4458});
  exec_aten::Tensor ret_expected_2 = tfFloat.make(
      {1, 1, 8, 4},
      {0.6350, 0.3426, 0.6582, 0.2484, 0.4391, 0.3419, 0.6962, 0.4399,
       0.6321, 0.3475, 0.3754, 0.9798, 0.5721, 0.1344, 0.7829, 0.4233,
       0.4122, 0.8394, 0.5040, 0.3304, 0.4066, 0.4378, 0.5820, 0.5922,
       0.4333, 0.3541, 0.6168, 0.2918, 0.6486, 0.4949, 0.2965, 0.6151});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      2,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_2, 1e-4, 1e-4);

  // start pos: 3 layer id 0
  query = tfFloat.make(
      {1, 1, 8, 4},
      {0.2732, 0.5486, 0.4419, 0.0040, 0.4089, 0.4521, 0.3526, 0.9594,
       0.3909, 0.8212, 0.6239, 0.0779, 0.6175, 0.9144, 0.1729, 0.1768,
       0.9894, 0.9018, 0.2211, 0.8009, 0.4360, 0.0070, 0.5376, 0.6615,
       0.3500, 0.6739, 0.0724, 0.8465, 0.9263, 0.7757, 0.5847, 0.6647});
  key = tfFloat.make(
      {1, 1, 8, 4},
      {0.1382, 0.3751, 0.4523, 0.2218, 0.1307, 0.8363, 0.8393, 0.0459,
       0.6591, 0.7034, 0.9750, 0.7893, 0.9597, 0.3363, 0.8502, 0.9067,
       0.0278, 0.0986, 0.6012, 0.7730, 0.2516, 0.5551, 0.4993, 0.6266,
       0.2313, 0.7820, 0.8325, 0.1531, 0.5048, 0.5014, 0.6606, 0.9658});
  value = tfFloat.make(
      {1, 1, 8, 4},
      {0.6466, 0.3864, 0.9491, 0.3097, 0.3548, 0.5341, 0.1192, 0.5544,
       0.1608, 0.5514, 0.5479, 0.5692, 0.0784, 0.0251, 0.7301, 0.9288,
       0.0563, 0.6852, 0.1319, 0.5313, 0.9652, 0.8793, 0.1344, 0.8093,
       0.7612, 0.4992, 0.9844, 0.3014, 0.3836, 0.2473, 0.5719, 0.6324});
  exec_aten::Tensor ret_expected_3 = tfFloat.make(
      {1, 1, 8, 4},
      {0.6441, 0.3571, 0.7319, 0.2624, 0.4506, 0.3619, 0.5749, 0.4930,
       0.4860, 0.3924, 0.4596, 0.8517, 0.4312, 0.1060, 0.7579, 0.5796,
       0.3507, 0.8063, 0.4223, 0.3597, 0.5522, 0.5558, 0.4665, 0.6486,
       0.5263, 0.3701, 0.6880, 0.2790, 0.6116, 0.4449, 0.3184, 0.6258});
  out = tfFloat.zeros(out_size);
  ret = op_sdpa_with_kv_cache(
      query,
      key,
      value,
      key_cache_0,
      value_cache_0,
      3,
      1,
      attn_mask,
      dropout_p,
      is_causal,
      scale,
      out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(ret, ret_expected_3, 1e-4, 1e-4);
}
