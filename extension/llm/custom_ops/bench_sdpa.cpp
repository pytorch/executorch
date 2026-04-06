/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Benchmark for SDPA (scaled dot-product attention) implementations.
 *
 * Compares:
 *   - CustomSDPA: ExecuTorch's tiled flash attention (custom_sdpa_out)
 *   - StandardSDPA: Standalone GEMM-based SDPA (no tiling, 3-pass softmax)
 *
 * StandardSDPA supports both [B,S,H,D] (standard) and [B,H,S,D] (transposed)
 * layouts via BLAS leading dimension parameters, allowing isolation of
 * algorithm vs layout effects.
 */

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/kernel/thread_parallel_interface.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::testing::TensorFactory;

namespace {

// Fill a float tensor with random data in [0, 1)
void fill_random(Tensor& t, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float* data = t.mutable_data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    data[i] = dist(gen);
  }
}

// Core GEMM-based SDPA: Q @ K^T -> scale -> causal mask -> softmax -> @ V.
// Supports both [B,H,S,D] (transposed) and [B,S,H,D] layouts via BLAS
// leading dimension. scores_buf must hold batch*Hq*q_seq_len*kvSize floats.
//
// NOTE: executorch::cpublas::gemm uses COLUMN-MAJOR (Fortran) convention
// internally (CblasColMajor). Our data is row-major. The conversion is:
//   row-major C[M,N] = A[M,K] * trans(B)[K,N]  (B stored as [N,K])
//   becomes col-major: gemm(Trans, NoTrans, N, M, K, a, B, ldb, A, lda, b, C,
//   ldc)
// where lda/ldb/ldc are the row-major strides (row strides = col-major ld).
void run_standard_sdpa(
    const float* q_data,
    const float* k_data,
    const float* v_data,
    float* out_data,
    float* scores_buf,
    int64_t batch,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t max_seq_len,
    int64_t start_pos,
    int64_t q_seq_len,
    bool is_transposed) {
  using executorch::cpublas::TransposeType;

  const int64_t kvSize = start_pos + q_seq_len;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int64_t heads_per_group = Hq / Hkv;

  // Row strides (= col-major leading dimensions for our memory layout)
  const int64_t ldq = is_transposed ? D : Hq * D;
  const int64_t ldk = is_transposed ? D : Hkv * D;
  const int64_t ldv = is_transposed ? D : Hkv * D;
  const int64_t ldo = is_transposed ? D : Hq * D;

  torch::executor::parallel_for(
      0, batch * Hq, 1, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          const int64_t b = idx / Hq;
          const int64_t h = idx % Hq;
          const int64_t kv_h = h / heads_per_group;

          const float* q_ptr;
          const float* k_ptr;
          const float* v_ptr;
          float* out_ptr;
          if (is_transposed) {
            q_ptr = q_data + (b * Hq + h) * q_seq_len * D;
            k_ptr = k_data + (b * Hkv + kv_h) * max_seq_len * D;
            v_ptr = v_data + (b * Hkv + kv_h) * max_seq_len * D;
            out_ptr = out_data + (b * Hq + h) * q_seq_len * D;
          } else {
            q_ptr = q_data + b * q_seq_len * Hq * D + h * D;
            k_ptr = k_data + b * max_seq_len * Hkv * D + kv_h * D;
            v_ptr = v_data + b * max_seq_len * Hkv * D + kv_h * D;
            out_ptr = out_data + b * q_seq_len * Hq * D + h * D;
          }
          float* scores = scores_buf + idx * q_seq_len * kvSize;

          // Row-major: scores[qSeqLen,kvSize] = Q[qSeqLen,D] @ K^T[D,kvSize]
          // Col-major: gemm(Trans, NoTrans, kvSize, qSeqLen, D, ...)
          executorch::cpublas::gemm(
              TransposeType::Transpose,
              TransposeType::NoTranspose,
              kvSize,
              q_seq_len,
              D,
              1.0f,
              k_ptr,
              ldk,
              q_ptr,
              ldq,
              0.0f,
              scores,
              kvSize);

          // Scale, causal mask, and softmax per query row
          for (int64_t qi = 0; qi < q_seq_len; ++qi) {
            float* row = scores + qi * kvSize;
            const int64_t valid = std::min(start_pos + qi + 1, kvSize);

            for (int64_t j = 0; j < valid; ++j) {
              row[j] *= scale;
            }
            for (int64_t j = valid; j < kvSize; ++j) {
              row[j] = -std::numeric_limits<float>::infinity();
            }

            float max_val = row[0];
            for (int64_t j = 1; j < kvSize; ++j) {
              max_val = std::max(max_val, row[j]);
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < kvSize; ++j) {
              row[j] = std::exp(row[j] - max_val);
              sum += row[j];
            }
            const float inv_sum = 1.0f / sum;
            for (int64_t j = 0; j < kvSize; ++j) {
              row[j] *= inv_sum;
            }
          }

          // Row-major: output[qSeqLen,D] = scores[qSeqLen,kvSize] @ V[kvSize,D]
          // Col-major: gemm(NoTrans, NoTrans, D, qSeqLen, kvSize, ...)
          executorch::cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::NoTranspose,
              D,
              q_seq_len,
              kvSize,
              1.0f,
              v_ptr,
              ldv,
              scores,
              kvSize,
              0.0f,
              out_ptr,
              ldo);
        }
      });
}

// ONNX Runtime GQA-style SDPA, faithfully ported from
// onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h.
// Differences from run_standard_sdpa:
//   1. Scale in GEMM alpha (no separate scaling pass)
//   2. Scores buffer padded to max_seq_len cols (ONNX's present_buffer_seq_len)
//   3. Causal mask: zero out future positions, softmax on valid window only
//   4. Output in [B, S, Hq, D] with stride Hq*D (ONNX's interleaved BNSH->BSNH)
//
// When is_transposed=true, inputs are [B,H,S,D]; output is [B,S,Hq,D].
// When is_transposed=false, inputs are [B,S,H,D]; output is [B,S,Hq,D].
// Output is always [B, S, Hq, D] to match ONNX's actual output format.
void run_onnx_gqa_sdpa(
    const float* q_data,
    const float* k_data,
    const float* v_data,
    float* out_data, // always [B, q_seq_len, Hq, D]
    float* scores_buf, // must hold batch*Hq*q_seq_len*max_seq_len floats
    int64_t batch,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t max_seq_len,
    int64_t start_pos,
    int64_t q_seq_len,
    bool is_transposed) {
  using executorch::cpublas::TransposeType;

  const int64_t total_seqlen = start_pos + q_seq_len;
  const float alpha = 1.0f / std::sqrt(static_cast<float>(D));
  const int64_t heads_per_group = Hq / Hkv;
  const int64_t hidden_size = Hq * D; // output row stride (ONNX convention)

  // Input strides depend on layout
  const int64_t ldq = is_transposed ? D : Hq * D;
  const int64_t ldk = is_transposed ? D : Hkv * D;
  const int64_t ldv = is_transposed ? D : Hkv * D;
  // Output is always [B, S, Hq, D] so ldo = Hq * D = hidden_size
  const int64_t ldo = hidden_size;

  torch::executor::parallel_for(
      0, batch * Hq, 1, [&](int64_t begin, int64_t end) {
        for (int64_t idx = begin; idx < end; ++idx) {
          const int64_t b = idx / Hq;
          const int64_t h = idx % Hq;
          const int64_t kv_h = h / heads_per_group;

          const float* q_ptr;
          const float* k_ptr;
          const float* v_ptr;
          if (is_transposed) {
            q_ptr = q_data + (b * Hq + h) * q_seq_len * D;
            k_ptr = k_data + (b * Hkv + kv_h) * max_seq_len * D;
            v_ptr = v_data + (b * Hkv + kv_h) * max_seq_len * D;
          } else {
            q_ptr = q_data + b * q_seq_len * Hq * D + h * D;
            k_ptr = k_data + b * max_seq_len * Hkv * D + kv_h * D;
            v_ptr = v_data + b * max_seq_len * Hkv * D + kv_h * D;
          }
          // Output always [B, S, Hq, D]: head h writes at stride hidden_size
          float* out_ptr = out_data + b * q_seq_len * hidden_size + h * D;

          // Scores padded to max_seq_len columns (ONNX convention)
          float* scores = scores_buf + idx * q_seq_len * max_seq_len;

          // GEMM 1: Q @ K^T with scale in alpha
          executorch::cpublas::gemm(
              TransposeType::Transpose,
              TransposeType::NoTranspose,
              total_seqlen,
              q_seq_len,
              D,
              alpha,
              k_ptr,
              ldk,
              q_ptr,
              ldq,
              0.0f,
              scores,
              max_seq_len);

          // Causal mask + narrow softmax (ONNX style):
          // Zero future positions, softmax only on valid [0, causal_len).
          for (int64_t qi = 0; qi < q_seq_len; ++qi) {
            float* row = scores + qi * max_seq_len;
            const int64_t causal_len =
                std::min(start_pos + qi + 1, total_seqlen);

            for (int64_t j = causal_len; j < total_seqlen; ++j) {
              row[j] = 0.0f;
            }

            float max_val = row[0];
            for (int64_t j = 1; j < causal_len; ++j) {
              max_val = std::max(max_val, row[j]);
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < causal_len; ++j) {
              row[j] = std::exp(row[j] - max_val);
              sum += row[j];
            }
            const float inv_sum = 1.0f / sum;
            for (int64_t j = 0; j < causal_len; ++j) {
              row[j] *= inv_sum;
            }
          }

          // GEMM 2: scores @ V -> output
          executorch::cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::NoTranspose,
              D,
              q_seq_len,
              total_seqlen,
              1.0f,
              v_ptr,
              ldv,
              scores,
              max_seq_len,
              0.0f,
              out_ptr,
              ldo);
        }
      });
}

// Return max |a - b| across all elements.
float max_abs_diff(const float* a, const float* b, int64_t n) {
  float d = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    d = std::max(d, std::abs(a[i] - b[i]));
  }
  return d;
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
  return max_abs_diff(
      a.const_data_ptr<float>(), b.const_data_ptr<float>(), a.numel());
}

// Validate a single config: run StandardSDPA and custom_sdpa_out on the same
// inputs, check outputs match within tolerance. Returns false on mismatch.
// Only tests standard [B,S,H,D] layout (is_transposed=false).
bool validate_config(
    int64_t batch,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t max_seq_len,
    int64_t start_pos,
    int64_t q_seq_len,
    float atol) {
  TensorFactory<ScalarType::Float> tf;
  std::mt19937 gen(42);

  // Standard [B, S, H, D] layout
  Tensor q =
      tf.zeros({(int32_t)batch, (int32_t)q_seq_len, (int32_t)Hq, (int32_t)D});
  Tensor k = tf.zeros(
      {(int32_t)batch, (int32_t)max_seq_len, (int32_t)Hkv, (int32_t)D});
  Tensor v = tf.zeros(
      {(int32_t)batch, (int32_t)max_seq_len, (int32_t)Hkv, (int32_t)D});

  fill_random(q, gen);
  fill_random(k, gen);
  fill_random(v, gen);

  // Reference: ET custom_sdpa_out (10-param signature, standard layout)
  Tensor out_ref =
      tf.zeros({(int32_t)batch, (int32_t)q_seq_len, (int32_t)Hq, (int32_t)D});
  KernelRuntimeContext ctx{};
  torch::executor::native::custom_sdpa_out(
      ctx, q, k, v, start_pos, std::nullopt, 0.0, true, std::nullopt, out_ref);

  // Test: GEMM-based standard SDPA
  Tensor out_test =
      tf.zeros({(int32_t)batch, (int32_t)q_seq_len, (int32_t)Hq, (int32_t)D});
  int64_t kvSize = start_pos + q_seq_len;
  std::vector<float> scores_buf(batch * Hq * q_seq_len * kvSize);
  run_standard_sdpa(
      q.const_data_ptr<float>(),
      k.const_data_ptr<float>(),
      v.const_data_ptr<float>(),
      out_test.mutable_data_ptr<float>(),
      scores_buf.data(),
      batch,
      Hq,
      Hkv,
      D,
      max_seq_len,
      start_pos,
      q_seq_len,
      false /* is_transposed */);

  float diff = max_abs_diff(out_ref, out_test);
  const char* mode = q_seq_len == 1 ? "decode" : "prefill";
  if (diff > atol) {
    fprintf(
        stderr,
        "FAIL: StandardSDPA standard %s (B=%ld Hq=%ld Hkv=%ld D=%ld sp=%ld sl=%ld) "
        "max_abs_diff=%.6e > atol=%.6e\n",
        mode,
        (long)batch,
        (long)Hq,
        (long)Hkv,
        (long)D,
        (long)start_pos,
        (long)q_seq_len,
        diff,
        atol);
    return false;
  }
  fprintf(
      stderr,
      "PASS: StandardSDPA standard %s (B=%ld Hq=%ld Hkv=%ld D=%ld sp=%ld sl=%ld) "
      "max_abs_diff=%.6e\n",
      mode,
      (long)batch,
      (long)Hq,
      (long)Hkv,
      (long)D,
      (long)start_pos,
      (long)q_seq_len,
      diff);

  // Also validate ONNX GQA variant. Output is always [B, S, Hq, D].
  // Since we only test standard [B,S,H,D] layout, out_ref is already
  // [B,S,Hq,D] — just copy directly to ref_bshd (no transpose needed).
  Tensor out_onnx =
      tf.zeros({(int32_t)batch, (int32_t)q_seq_len, (int32_t)Hq, (int32_t)D});
  std::vector<float> onnx_scores_buf(batch * Hq * q_seq_len * max_seq_len);
  run_onnx_gqa_sdpa(
      q.const_data_ptr<float>(),
      k.const_data_ptr<float>(),
      v.const_data_ptr<float>(),
      out_onnx.mutable_data_ptr<float>(),
      onnx_scores_buf.data(),
      batch,
      Hq,
      Hkv,
      D,
      max_seq_len,
      start_pos,
      q_seq_len,
      false /* is_transposed */);

  // out_ref is already [B, S, Hq, D] (standard layout), compare directly
  std::vector<float> ref_bshd(batch * q_seq_len * Hq * D);
  const float* ref_ptr = out_ref.const_data_ptr<float>();
  std::copy(ref_ptr, ref_ptr + batch * q_seq_len * Hq * D, ref_bshd.data());

  float onnx_diff = max_abs_diff(
      out_onnx.const_data_ptr<float>(),
      ref_bshd.data(),
      batch * q_seq_len * Hq * D);
  if (onnx_diff > atol) {
    fprintf(
        stderr,
        "FAIL: OnnxGQA standard %s (B=%ld Hq=%ld Hkv=%ld D=%ld sp=%ld sl=%ld) "
        "max_abs_diff=%.6e > atol=%.6e\n",
        mode,
        (long)batch,
        (long)Hq,
        (long)Hkv,
        (long)D,
        (long)start_pos,
        (long)q_seq_len,
        onnx_diff,
        atol);
    return false;
  }
  fprintf(
      stderr,
      "PASS: OnnxGQA standard %s (B=%ld Hq=%ld Hkv=%ld D=%ld sp=%ld sl=%ld) "
      "max_abs_diff=%.6e\n",
      mode,
      (long)batch,
      (long)Hq,
      (long)Hkv,
      (long)D,
      (long)start_pos,
      (long)q_seq_len,
      onnx_diff);

  return true;
}

// Run all validation tests. Aborts if any fail.
void run_validation_tests() {
  fprintf(stderr, "--- Validating StandardSDPA vs custom_sdpa_out ---\n");
  // Online softmax (flash) vs 3-pass softmax can differ at ~1e-5 for float32;
  // use atol=1e-3 to be safe across various kv sizes.
  constexpr float kAtol = 1e-3f;
  bool all_passed = true;

  // Decode configs (q_seq_len=1), standard [B,S,H,D] layout only
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 1, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 64, 1, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 128, 1, kAtol);

  // Prefill configs
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 16, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 64, kAtol);

  // Non-GQA (Hq == Hkv)
  all_passed &= validate_config(1, 8, 8, 64, 256, 64, 1, kAtol);

  if (!all_passed) {
    fprintf(stderr, "VALIDATION FAILED — benchmark results are unreliable\n");
    std::abort();
  }
  fprintf(stderr, "--- All validation tests passed ---\n\n");
}

} // namespace

namespace {

// Benchmark fixture for custom_sdpa_out. Always uses standard [B,S,H,D] layout.
class SDPABenchFixture : public benchmark::Fixture {
 public:
  // Args: {batch, num_heads_q, num_heads_kv, head_dim, max_seq_len, start_pos,
  //        query_seq_len}
  void SetUp(benchmark::State& state) override {
    int64_t batch = state.range(0);
    int64_t num_heads_q = state.range(1);
    int64_t num_heads_kv = state.range(2);
    int64_t head_dim = state.range(3);
    int64_t max_seq_len = state.range(4);
    int64_t start_pos = state.range(5);
    int64_t q_seq_len = state.range(6);

    std::mt19937 gen(42);

    // Standard [B, S, H, D] layout
    q_.emplace(tf_.zeros(
        {(int32_t)batch,
         (int32_t)q_seq_len,
         (int32_t)num_heads_q,
         (int32_t)head_dim}));
    k_cache_.emplace(tf_.zeros(
        {(int32_t)batch,
         (int32_t)max_seq_len,
         (int32_t)num_heads_kv,
         (int32_t)head_dim}));
    v_cache_.emplace(tf_.zeros(
        {(int32_t)batch,
         (int32_t)max_seq_len,
         (int32_t)num_heads_kv,
         (int32_t)head_dim}));
    output_.emplace(tf_.zeros(
        {(int32_t)batch,
         (int32_t)q_seq_len,
         (int32_t)num_heads_q,
         (int32_t)head_dim}));

    fill_random(*q_, gen);
    fill_random(*k_cache_, gen);
    fill_random(*v_cache_, gen);

    start_pos_ = start_pos;
  }

  void TearDown(benchmark::State&) override {
    q_.reset();
    k_cache_.reset();
    v_cache_.reset();
    output_.reset();
  }

  TensorFactory<ScalarType::Float> tf_;
  std::optional<Tensor> q_;
  std::optional<Tensor> k_cache_;
  std::optional<Tensor> v_cache_;
  std::optional<Tensor> output_;
  int64_t start_pos_ = 0;
};

// Benchmark custom_sdpa with causal masking (standard [B,S,H,D] layout)
BENCHMARK_DEFINE_F(SDPABenchFixture, CustomSDPA)
(benchmark::State& state) {
  for (auto _ : state) {
    KernelRuntimeContext ctx{};
    torch::executor::native::custom_sdpa_out(
        ctx,
        *q_,
        *k_cache_,
        *v_cache_,
        start_pos_,
        std::nullopt, // attn_mask
        0.0, // dropout_p
        true, // is_causal
        std::nullopt, // scale
        *output_);
  }
}

// Standalone GEMM-based SDPA benchmark. Supports both [B,H,S,D] and [B,S,H,D]
// layouts via BLAS leading dimension, isolating algorithm vs layout effects.
class StandardSDPABenchFixture : public benchmark::Fixture {
 public:
  // Args: {batch, num_heads_q, num_heads_kv, head_dim, max_seq_len, start_pos,
  //        query_seq_len, is_transposed}
  void SetUp(benchmark::State& state) override {
    int64_t batch = state.range(0);
    int64_t num_heads_q = state.range(1);
    int64_t num_heads_kv = state.range(2);
    int64_t head_dim = state.range(3);
    int64_t max_seq_len = state.range(4);
    int64_t start_pos = state.range(5);
    int64_t q_seq_len = state.range(6);
    bool is_transposed = state.range(7) != 0;

    std::mt19937 gen(42);

    if (is_transposed) {
      // [B, H, S, D]
      q_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)num_heads_q,
           (int32_t)q_seq_len,
           (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)num_heads_kv,
           (int32_t)max_seq_len,
           (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)num_heads_kv,
           (int32_t)max_seq_len,
           (int32_t)head_dim}));
      output_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)num_heads_q,
           (int32_t)q_seq_len,
           (int32_t)head_dim}));
    } else {
      // [B, S, H, D]
      q_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)q_seq_len,
           (int32_t)num_heads_q,
           (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)max_seq_len,
           (int32_t)num_heads_kv,
           (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)max_seq_len,
           (int32_t)num_heads_kv,
           (int32_t)head_dim}));
      output_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)q_seq_len,
           (int32_t)num_heads_q,
           (int32_t)head_dim}));
    }

    fill_random(*q_, gen);
    fill_random(*k_cache_, gen);
    fill_random(*v_cache_, gen);

    batch_ = batch;
    num_heads_q_ = num_heads_q;
    num_heads_kv_ = num_heads_kv;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    start_pos_ = start_pos;
    q_seq_len_ = q_seq_len;
    kv_size_ = start_pos + q_seq_len;
    is_transposed_ = is_transposed;

    int64_t total_units = batch * num_heads_q;
    scores_buf_.resize(total_units * q_seq_len * kv_size_);
  }

  void TearDown(benchmark::State&) override {
    q_.reset();
    k_cache_.reset();
    v_cache_.reset();
    output_.reset();
    scores_buf_.clear();
  }

  TensorFactory<ScalarType::Float> tf_;
  std::optional<Tensor> q_;
  std::optional<Tensor> k_cache_;
  std::optional<Tensor> v_cache_;
  std::optional<Tensor> output_;
  std::vector<float> scores_buf_;
  int64_t batch_ = 0;
  int64_t num_heads_q_ = 0;
  int64_t num_heads_kv_ = 0;
  int64_t head_dim_ = 0;
  int64_t max_seq_len_ = 0;
  int64_t start_pos_ = 0;
  int64_t q_seq_len_ = 0;
  int64_t kv_size_ = 0;
  bool is_transposed_ = false;
};

// Benchmark standard (non-tiled) SDPA with GEMM.
// Both layouts supported via BLAS leading dimension:
//   [B, H, S, D]: head data is contiguous, ld = D
//   [B, S, H, D]: head data is strided, ld = H * D
BENCHMARK_DEFINE_F(StandardSDPABenchFixture, StandardSDPA)
(benchmark::State& state) {
  const float* q_data = q_->const_data_ptr<float>();
  const float* k_data = k_cache_->const_data_ptr<float>();
  const float* v_data = v_cache_->const_data_ptr<float>();
  float* out_data = output_->mutable_data_ptr<float>();

  for (auto _ : state) {
    run_standard_sdpa(
        q_data,
        k_data,
        v_data,
        out_data,
        scores_buf_.data(),
        batch_,
        num_heads_q_,
        num_heads_kv_,
        head_dim_,
        max_seq_len_,
        start_pos_,
        q_seq_len_,
        is_transposed_);
  }
}

// ONNX Runtime GQA-style benchmark. Faithfully matches the algorithm from
// gqa_attention_base.h: scale-in-alpha, padded scores buffer, narrow softmax,
// and output in [B, S, Hq, D] with stride Hq*D.
class OnnxGQABenchFixture : public benchmark::Fixture {
 public:
  // Args: {batch, num_heads_q, num_heads_kv, head_dim, max_seq_len, start_pos,
  //        query_seq_len, is_transposed}
  void SetUp(benchmark::State& state) override {
    int64_t batch = state.range(0);
    int64_t num_heads_q = state.range(1);
    int64_t num_heads_kv = state.range(2);
    int64_t head_dim = state.range(3);
    int64_t max_seq_len = state.range(4);
    int64_t start_pos = state.range(5);
    int64_t q_seq_len = state.range(6);
    bool is_transposed = state.range(7) != 0;

    std::mt19937 gen(42);

    if (is_transposed) {
      q_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)num_heads_q,
           (int32_t)q_seq_len,
           (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)num_heads_kv,
           (int32_t)max_seq_len,
           (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)num_heads_kv,
           (int32_t)max_seq_len,
           (int32_t)head_dim}));
    } else {
      q_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)q_seq_len,
           (int32_t)num_heads_q,
           (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)max_seq_len,
           (int32_t)num_heads_kv,
           (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros(
          {(int32_t)batch,
           (int32_t)max_seq_len,
           (int32_t)num_heads_kv,
           (int32_t)head_dim}));
    }
    // Output always [B, S, Hq, D] (ONNX convention)
    output_.emplace(tf_.zeros(
        {(int32_t)batch,
         (int32_t)q_seq_len,
         (int32_t)num_heads_q,
         (int32_t)head_dim}));

    fill_random(*q_, gen);
    fill_random(*k_cache_, gen);
    fill_random(*v_cache_, gen);

    batch_ = batch;
    num_heads_q_ = num_heads_q;
    num_heads_kv_ = num_heads_kv;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    start_pos_ = start_pos;
    q_seq_len_ = q_seq_len;
    is_transposed_ = is_transposed;

    // Scores buffer padded to max_seq_len columns (ONNX convention)
    int64_t total_units = batch * num_heads_q;
    scores_buf_.resize(total_units * q_seq_len * max_seq_len);
  }

  void TearDown(benchmark::State&) override {
    q_.reset();
    k_cache_.reset();
    v_cache_.reset();
    output_.reset();
    scores_buf_.clear();
  }

  TensorFactory<ScalarType::Float> tf_;
  std::optional<Tensor> q_;
  std::optional<Tensor> k_cache_;
  std::optional<Tensor> v_cache_;
  std::optional<Tensor> output_;
  std::vector<float> scores_buf_;
  int64_t batch_ = 0;
  int64_t num_heads_q_ = 0;
  int64_t num_heads_kv_ = 0;
  int64_t head_dim_ = 0;
  int64_t max_seq_len_ = 0;
  int64_t start_pos_ = 0;
  int64_t q_seq_len_ = 0;
  bool is_transposed_ = false;
};

BENCHMARK_DEFINE_F(OnnxGQABenchFixture, OnnxGQA)
(benchmark::State& state) {
  const float* q_data = q_->const_data_ptr<float>();
  const float* k_data = k_cache_->const_data_ptr<float>();
  const float* v_data = v_cache_->const_data_ptr<float>();
  float* out_data = output_->mutable_data_ptr<float>();

  for (auto _ : state) {
    run_onnx_gqa_sdpa(
        q_data,
        k_data,
        v_data,
        out_data,
        scores_buf_.data(),
        batch_,
        num_heads_q_,
        num_heads_kv_,
        head_dim_,
        max_seq_len_,
        start_pos_,
        q_seq_len_,
        is_transposed_);
  }
}

/*
 * Benchmark configurations modeled after Llama 3 8B (GQA: 32 q heads, 8 kv
 * heads, head_dim=128). We test decode (seq_len=1) and prefill scenarios at
 * various cache fill levels.
 */

// --- custom_sdpa: standard [B,S,H,D] layout ---
// Args: {batch, Hq, Hkv, D, MaxS, StartPos, SeqLen}
BENCHMARK_REGISTER_F(SDPABenchFixture, CustomSDPA)
    // Decode at various cache positions
    ->Args({1, 32, 8, 128, 2048, 0, 1})
    ->Args({1, 32, 8, 128, 2048, 64, 1})
    ->Args({1, 32, 8, 128, 2048, 256, 1})
    ->Args({1, 32, 8, 128, 2048, 512, 1})
    ->Args({1, 32, 8, 128, 2048, 1024, 1})
    // Prefill
    ->Args({1, 32, 8, 128, 2048, 0, 128})
    ->Args({1, 32, 8, 128, 2048, 0, 512})
    // Llama 2 style (32 heads, no GQA)
    ->Args({1, 32, 32, 128, 2048, 256, 1})
    ->ArgNames({"B", "Hq", "Hkv", "D", "MaxS", "StartPos", "SeqLen"});

// --- Standard SDPA (GEMM-based, both layouts) ---
// Args: {batch, Hq, Hkv, D, MaxS, StartPos, SeqLen, Transposed}
BENCHMARK_REGISTER_F(StandardSDPABenchFixture, StandardSDPA)
    // Standard layout decode at various cache positions
    ->Args({1, 32, 8, 128, 2048, 0, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 64, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 256, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 512, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 1024, 1, 0})
    // Transposed layout decode at same positions
    ->Args({1, 32, 8, 128, 2048, 0, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 64, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 256, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 512, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 1024, 1, 1})
    // Standard layout prefill
    ->Args({1, 32, 8, 128, 2048, 0, 128, 0})
    ->Args({1, 32, 8, 128, 2048, 0, 512, 0})
    // Transposed layout prefill
    ->Args({1, 32, 8, 128, 2048, 0, 128, 1})
    ->Args({1, 32, 8, 128, 2048, 0, 512, 1})
    // Llama 2 style (32 heads, no GQA)
    ->Args({1, 32, 32, 128, 2048, 256, 1, 0})
    ->Args({1, 32, 32, 128, 2048, 256, 1, 1})
    ->ArgNames({"B", "Hq", "Hkv", "D", "MaxS", "StartPos", "SeqLen", "Trans"});

// --- ONNX Runtime GQA-style SDPA ---
// Same configs as StandardSDPA. Differences: scale-in-alpha, padded scores
// buffer (ld=MaxS), narrow softmax, output in [B,S,Hq,D] with stride Hq*D.
BENCHMARK_REGISTER_F(OnnxGQABenchFixture, OnnxGQA)
    // Standard layout decode at various cache positions
    ->Args({1, 32, 8, 128, 2048, 0, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 64, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 256, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 512, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 1024, 1, 0})
    // Transposed layout decode at same positions
    ->Args({1, 32, 8, 128, 2048, 0, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 64, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 256, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 512, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 1024, 1, 1})
    // Standard layout prefill
    ->Args({1, 32, 8, 128, 2048, 0, 128, 0})
    ->Args({1, 32, 8, 128, 2048, 0, 512, 0})
    // Transposed layout prefill
    ->Args({1, 32, 8, 128, 2048, 0, 128, 1})
    ->Args({1, 32, 8, 128, 2048, 0, 512, 1})
    // Llama 2 style (32 heads, no GQA)
    ->Args({1, 32, 32, 128, 2048, 256, 1, 0})
    ->Args({1, 32, 32, 128, 2048, 256, 1, 1})
    ->ArgNames({"B", "Hq", "Hkv", "D", "MaxS", "StartPos", "SeqLen", "Trans"});

} // namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  run_validation_tests();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
