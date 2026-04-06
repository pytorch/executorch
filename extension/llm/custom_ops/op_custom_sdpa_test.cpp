/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests for the unfused SDPA code path (cpu_sdpa) dispatched when
// seq_len == 1 and inputs are non-quantized (the decode fast-path).
// These call custom_sdpa_out directly, not through sdpa_with_kv_cache.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::runtime::testing::TensorFactory;

namespace {

// Helper: call custom_sdpa_out. Inputs use [B, S, H, D] layout.
executorch::aten::Tensor call_custom_sdpa(
    const executorch::aten::Tensor& q,
    const executorch::aten::Tensor& k,
    const executorch::aten::Tensor& v,
    int64_t start_pos,
    const std::optional<executorch::aten::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    executorch::aten::Tensor& out) {
  executorch::runtime::KernelRuntimeContext ctx{};
  return torch::executor::native::custom_sdpa_out(
      ctx, q, k, v, start_pos, attn_mask, dropout_p, is_causal, scale, out);
}

/**
 * Naive reference SDPA for [B, S, H, D] layout.
 * Element [b,s,h,d] is at index b*S*H*D + s*H*D + h*D + d.
 * Only first num_valid_keys KV entries are used.
 */
void compute_reference_sdpa(
    const float* q_data,
    int B,
    int qS,
    int qH,
    int D,
    const float* k_data,
    int kvS,
    int kvH,
    const float* v_data,
    float* out_data,
    bool is_causal,
    int64_t start_pos,
    int num_valid_keys) {
  float scale = 1.0f / std::sqrt(static_cast<float>(D));
  int num_reps = qH / kvH;

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < qH; h++) {
      int kv_h = h / num_reps;
      for (int qs = 0; qs < qS; qs++) {
        // scores = Q @ K^T * scale
        std::vector<float> scores(num_valid_keys);
        for (int kvs = 0; kvs < num_valid_keys; kvs++) {
          float dot = 0;
          for (int d = 0; d < D; d++) {
            float qv = q_data[b * qS * qH * D + qs * qH * D + h * D + d];
            float kv = k_data[b * kvS * kvH * D + kvs * kvH * D + kv_h * D + d];
            dot += qv * kv;
          }
          scores[kvs] = dot * scale;
        }

        // Causal mask
        if (is_causal) {
          int64_t valid = std::min(
              start_pos + qs + 1, static_cast<int64_t>(num_valid_keys));
          for (int64_t j = valid; j < num_valid_keys; j++) {
            scores[j] = -std::numeric_limits<float>::infinity();
          }
        }

        // Softmax
        float max_val = *std::max_element(scores.begin(), scores.end());
        float sum = 0;
        for (auto& s : scores) {
          s = std::exp(s - max_val);
          sum += s;
        }
        if (sum > 0) {
          for (auto& s : scores) {
            s /= sum;
          }
        }

        // output = scores @ V
        for (int d = 0; d < D; d++) {
          float val = 0;
          for (int kvs = 0; kvs < num_valid_keys; kvs++) {
            float vv = v_data[b * kvS * kvH * D + kvs * kvH * D + kv_h * D + d];
            val += scores[kvs] * vv;
          }
          out_data[b * qS * qH * D + qs * qH * D + h * D + d] = val;
        }
      }
    }
  }
}

} // namespace

// With a single KV entry (start_pos=0), output must equal V[0].
TEST(OpCustomSdpaTest, DecodeSingleKV) {
  TensorFactory<executorch::aten::ScalarType::Float> tf;

  executorch::aten::Tensor q = tf.make(
      {1, 1, 2, 4},
      {0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566, 0.7936});

  executorch::aten::Tensor k = tf.make(
      {1, 1, 2, 4},
      {0.8854, 0.5739, 0.2666, 0.6274, 0.2696, 0.4414, 0.2969, 0.8317});

  executorch::aten::Tensor v = tf.make(
      {1, 1, 2, 4},
      {0.6343, 0.3644, 0.7104, 0.9464, 0.7890, 0.2814, 0.7886, 0.5895});

  // softmax of a single score is always 1.0, so output == V
  executorch::aten::Tensor expected = tf.make(
      {1, 1, 2, 4},
      {0.6343, 0.3644, 0.7104, 0.9464, 0.7890, 0.2814, 0.7886, 0.5895});

  executorch::aten::Tensor out = tf.zeros({1, 1, 2, 4});
  call_custom_sdpa(q, k, v, /*start_pos=*/0, {}, 0.0, false, {}, out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(out, expected, 1e-6, 1e-6);
}

// Decode with 3 valid KV entries, verified against reference computation.
TEST(OpCustomSdpaTest, DecodeNonCausal) {
  TensorFactory<executorch::aten::ScalarType::Float> tf;

  // Q: [B=1, S=1, H=2, D=4]
  executorch::aten::Tensor q = tf.make(
      {1, 1, 2, 4},
      {0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566, 0.7936});

  // K, V: [B=1, kv_len=4, H=2, D=4], first 3 entries valid
  executorch::aten::Tensor k = tf.make(
      {1, 4, 2, 4},
      {0.8854, 0.5739, 0.2666, 0.6274, 0.2696, 0.4414, 0.2969, 0.8317,
       0.1053, 0.2695, 0.3588, 0.1994, 0.5472, 0.0062, 0.9516, 0.0753,
       0.8860, 0.5832, 0.3376, 0.8090, 0.5779, 0.9040, 0.5547, 0.3423,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000});

  executorch::aten::Tensor v = tf.make(
      {1, 4, 2, 4},
      {0.6343, 0.3644, 0.7104, 0.9464, 0.7890, 0.2814, 0.7886, 0.5895,
       0.7539, 0.1952, 0.0050, 0.3068, 0.1165, 0.9103, 0.6440, 0.7071,
       0.6581, 0.4913, 0.8913, 0.1447, 0.5315, 0.1587, 0.6542, 0.3278,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000});

  int64_t start_pos = 2;
  int num_valid = 3;

  std::vector<float> ref(8, 0.0f);
  compute_reference_sdpa(
      q.const_data_ptr<float>(),
      1,
      1,
      2,
      4,
      k.const_data_ptr<float>(),
      4,
      2,
      v.const_data_ptr<float>(),
      ref.data(),
      false,
      start_pos,
      num_valid);

  executorch::aten::Tensor expected = tf.make({1, 1, 2, 4}, ref);
  executorch::aten::Tensor out = tf.zeros({1, 1, 2, 4});
  call_custom_sdpa(q, k, v, start_pos, {}, 0.0, false, {}, out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(out, expected, 1e-4, 1e-4);
}

// GQA: 4 query heads sharing 2 KV heads.
TEST(OpCustomSdpaTest, DecodeGQA) {
  TensorFactory<executorch::aten::ScalarType::Float> tf;

  // Q: [B=1, S=1, H_q=4, D=4]
  executorch::aten::Tensor q = tf.make(
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

  // K: [B=1, kv_len=3, H_kv=2, D=4]
  executorch::aten::Tensor k =
      tf.make({1, 3, 2, 4}, {0.8854, 0.5739, 0.2666, 0.6274, 0.2696, 0.4414,
                             0.2969, 0.8317, 0.1053, 0.2695, 0.3588, 0.1994,
                             0.5472, 0.0062, 0.9516, 0.0753, 0.8860, 0.5832,
                             0.3376, 0.8090, 0.5779, 0.9040, 0.5547, 0.3423});

  // V: [B=1, kv_len=3, H_kv=2, D=4]
  executorch::aten::Tensor v =
      tf.make({1, 3, 2, 4}, {0.6343, 0.3644, 0.7104, 0.9464, 0.7890, 0.2814,
                             0.7886, 0.5895, 0.7539, 0.1952, 0.0050, 0.3068,
                             0.1165, 0.9103, 0.6440, 0.7071, 0.6581, 0.4913,
                             0.8913, 0.1447, 0.5315, 0.1587, 0.6542, 0.3278});

  int64_t start_pos = 2;
  int num_valid = 3;

  std::vector<float> ref(16, 0.0f);
  compute_reference_sdpa(
      q.const_data_ptr<float>(),
      1,
      1,
      4,
      4,
      k.const_data_ptr<float>(),
      3,
      2,
      v.const_data_ptr<float>(),
      ref.data(),
      false,
      start_pos,
      num_valid);

  executorch::aten::Tensor expected = tf.make({1, 1, 4, 4}, ref);
  executorch::aten::Tensor out = tf.zeros({1, 1, 4, 4});
  call_custom_sdpa(q, k, v, start_pos, {}, 0.0, false, {}, out);
  EXPECT_TENSOR_CLOSE_WITH_TOL(out, expected, 1e-4, 1e-4);
}

// For seq_len=1, causal mask doesn't restrict any positions
// (all start_pos+1 entries are visible), so result must match non-causal.
TEST(OpCustomSdpaTest, DecodeCausalMatchesNonCausal) {
  TensorFactory<executorch::aten::ScalarType::Float> tf;

  executorch::aten::Tensor q = tf.make(
      {1, 1, 2, 4},
      {0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566, 0.7936});

  executorch::aten::Tensor k = tf.make(
      {1, 4, 2, 4},
      {0.8854, 0.5739, 0.2666, 0.6274, 0.2696, 0.4414, 0.2969, 0.8317,
       0.1053, 0.2695, 0.3588, 0.1994, 0.5472, 0.0062, 0.9516, 0.0753,
       0.8860, 0.5832, 0.3376, 0.8090, 0.5779, 0.9040, 0.5547, 0.3423,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000});

  executorch::aten::Tensor v = tf.make(
      {1, 4, 2, 4},
      {0.6343, 0.3644, 0.7104, 0.9464, 0.7890, 0.2814, 0.7886, 0.5895,
       0.7539, 0.1952, 0.0050, 0.3068, 0.1165, 0.9103, 0.6440, 0.7071,
       0.6581, 0.4913, 0.8913, 0.1447, 0.5315, 0.1587, 0.6542, 0.3278,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000});

  int64_t start_pos = 2;

  executorch::aten::Tensor out_nc = tf.zeros({1, 1, 2, 4});
  call_custom_sdpa(q, k, v, start_pos, {}, 0.0, false, {}, out_nc);

  executorch::aten::Tensor out_c = tf.zeros({1, 1, 2, 4});
  call_custom_sdpa(q, k, v, start_pos, {}, 0.0, true, {}, out_c);

  EXPECT_TENSOR_CLOSE_WITH_TOL(out_c, out_nc, 1e-6, 1e-6);
}
