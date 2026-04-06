// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/*
 * Benchmark to compare performance of transposed vs standard KV cache layout
 * for custom_sdpa and update_cache ops.
 *
 * Standard layout:    [Batch, Seq, Heads, HeadDim]  (is_seq_dim_2=false)
 * Transposed layout:  [Batch, Heads, Seq, HeadDim]  (is_seq_dim_2=true)
 *
 * The hypothesis is that transposed cache improves GEMM performance because:
 *   - In attn_score @ V: V stride along S_kv changes from H*D to D
 *   - In Q @ K^T: K stride similarly improves from H*D to D
 */

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/extension/llm/custom_ops/op_update_cache.h>
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

// Core GEMM-based SDPA: Q @ K^T → scale → causal mask → softmax → @ V.
// Supports both [B,H,S,D] (transposed) and [B,S,H,D] layouts via BLAS
// leading dimension. scores_buf must hold batch*Hq*q_seq_len*kvSize floats.
//
// NOTE: executorch::cpublas::gemm uses COLUMN-MAJOR (Fortran) convention
// internally (CblasColMajor). Our data is row-major. The conversion is:
//   row-major C[M,N] = A[M,K] * trans(B)[K,N]  (B stored as [N,K])
//   becomes col-major: gemm(Trans, NoTrans, N, M, K, α, B, ldb, A, lda, β, C, ldc)
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
              TransposeType::Transpose, TransposeType::NoTranspose,
              kvSize, q_seq_len, D,
              1.0f, k_ptr, ldk, q_ptr, ldq,
              0.0f, scores, kvSize);

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
              TransposeType::NoTranspose, TransposeType::NoTranspose,
              D, q_seq_len, kvSize,
              1.0f, v_ptr, ldv, scores, kvSize,
              0.0f, out_ptr, ldo);
        }
      });
}

// ONNX Runtime GQA-style SDPA, faithfully ported from
// onnxruntime/contrib_ops/cpu/bert/gqa_attention_base.h.
// Differences from run_standard_sdpa:
//   1. Scale in GEMM alpha (no separate scaling pass)
//   2. Scores buffer padded to max_seq_len cols (ONNX's present_buffer_seq_len)
//   3. Causal mask: zero out future positions, softmax on valid window only
//   4. Output in [B, S, Hq, D] with stride Hq*D (ONNX's interleaved BNSH→BSNH)
//
// When is_transposed=true, inputs are [B,H,S,D]; output is [B,S,Hq,D].
// When is_transposed=false, inputs are [B,S,H,D]; output is [B,S,Hq,D].
// Output is always [B, S, Hq, D] to match ONNX's actual output format.
void run_onnx_gqa_sdpa(
    const float* q_data,
    const float* k_data,
    const float* v_data,
    float* out_data,       // always [B, q_seq_len, Hq, D]
    float* scores_buf,     // must hold batch*Hq*q_seq_len*max_seq_len floats
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
  const int64_t hidden_size = Hq * D;  // output row stride (ONNX convention)

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
          float* out_ptr =
              out_data + b * q_seq_len * hidden_size + h * D;

          // Scores padded to max_seq_len columns (ONNX convention)
          float* scores = scores_buf + idx * q_seq_len * max_seq_len;

          // GEMM 1: Q @ K^T with scale in alpha.
          // ONNX row-major: GemmEx(NoTrans, Trans, S, total_seqlen, H,
          //                        alpha, Q, H, K, H, 0, probs, max_seq_len)
          // ET col-major equivalent:
          executorch::cpublas::gemm(
              TransposeType::Transpose, TransposeType::NoTranspose,
              total_seqlen, q_seq_len, D,
              alpha, k_ptr, ldk, q_ptr, ldq,
              0.0f, scores, max_seq_len);

          // Causal mask + narrow softmax (ONNX style):
          // Zero future positions, softmax only on valid [0, causal_len).
          for (int64_t qi = 0; qi < q_seq_len; ++qi) {
            float* row = scores + qi * max_seq_len;
            const int64_t causal_len =
                std::min(start_pos + qi + 1, total_seqlen);

            // Zero out positions after causal boundary
            for (int64_t j = causal_len; j < total_seqlen; ++j) {
              row[j] = 0.0f;
            }

            // Softmax over valid window only
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

          // GEMM 2: scores @ V → output
          // ONNX row-major: GemmEx(NoTrans, NoTrans, S, H, total_seqlen,
          //                        1.0, probs, max_seq_len, V, H, 0, out,
          //                        hidden_size)
          // ET col-major equivalent:
          executorch::cpublas::gemm(
              TransposeType::NoTranspose, TransposeType::NoTranspose,
              D, q_seq_len, total_seqlen,
              1.0f, v_ptr, ldv, scores, max_seq_len,
              0.0f, out_ptr, ldo);
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
bool validate_config(
    int64_t batch,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t max_seq_len,
    int64_t start_pos,
    int64_t q_seq_len,
    bool is_transposed,
    float atol) {
  TensorFactory<ScalarType::Float> tf;
  std::mt19937 gen(42);

  Tensor q = is_transposed
      ? tf.zeros(
            {(int32_t)batch, (int32_t)Hq, (int32_t)q_seq_len, (int32_t)D})
      : tf.zeros(
            {(int32_t)batch, (int32_t)q_seq_len, (int32_t)Hq, (int32_t)D});
  Tensor k = is_transposed
      ? tf.zeros(
            {(int32_t)batch, (int32_t)Hkv, (int32_t)max_seq_len, (int32_t)D})
      : tf.zeros(
            {(int32_t)batch, (int32_t)max_seq_len, (int32_t)Hkv, (int32_t)D});
  Tensor v = is_transposed
      ? tf.zeros(
            {(int32_t)batch, (int32_t)Hkv, (int32_t)max_seq_len, (int32_t)D})
      : tf.zeros(
            {(int32_t)batch, (int32_t)max_seq_len, (int32_t)Hkv, (int32_t)D});

  fill_random(q, gen);
  fill_random(k, gen);
  fill_random(v, gen);

  // Reference: ET custom_sdpa_out
  Tensor out_ref = is_transposed
      ? tf.zeros(
            {(int32_t)batch, (int32_t)Hq, (int32_t)q_seq_len, (int32_t)D})
      : tf.zeros(
            {(int32_t)batch, (int32_t)q_seq_len, (int32_t)Hq, (int32_t)D});
  KernelRuntimeContext ctx{};
  torch::executor::native::custom_sdpa_out(
      ctx,
      q, k, v,
      start_pos,
      std::nullopt, 0.0, true, std::nullopt,
      is_transposed, is_transposed, is_transposed,
      out_ref);

  // Test: GEMM-based standard SDPA
  Tensor out_test = is_transposed
      ? tf.zeros(
            {(int32_t)batch, (int32_t)Hq, (int32_t)q_seq_len, (int32_t)D})
      : tf.zeros(
            {(int32_t)batch, (int32_t)q_seq_len, (int32_t)Hq, (int32_t)D});
  int64_t kvSize = start_pos + q_seq_len;
  std::vector<float> scores_buf(batch * Hq * q_seq_len * kvSize);
  run_standard_sdpa(
      q.const_data_ptr<float>(),
      k.const_data_ptr<float>(),
      v.const_data_ptr<float>(),
      out_test.mutable_data_ptr<float>(),
      scores_buf.data(),
      batch, Hq, Hkv, D, max_seq_len, start_pos, q_seq_len, is_transposed);

  float diff = max_abs_diff(out_ref, out_test);
  const char* layout = is_transposed ? "transposed" : "standard";
  const char* mode = q_seq_len == 1 ? "decode" : "prefill";
  if (diff > atol) {
    fprintf(
        stderr,
        "FAIL: StandardSDPA %s %s (B=%ld Hq=%ld Hkv=%ld D=%ld sp=%ld sl=%ld) "
        "max_abs_diff=%.6e > atol=%.6e\n",
        layout, mode, (long)batch, (long)Hq, (long)Hkv, (long)D,
        (long)start_pos, (long)q_seq_len, diff, atol);
    return false;
  }
  fprintf(
      stderr,
      "PASS: StandardSDPA %s %s (B=%ld Hq=%ld Hkv=%ld D=%ld sp=%ld sl=%ld) "
      "max_abs_diff=%.6e\n",
      layout, mode, (long)batch, (long)Hq, (long)Hkv, (long)D,
      (long)start_pos, (long)q_seq_len, diff);

  // Also validate ONNX GQA variant. Output is always [B, S, Hq, D], so
  // we compare per-element against out_ref rearranged to the same layout.
  // For simplicity, run ONNX SDPA with the same inputs, then compare
  // element-wise against a reference produced in [B, S, Hq, D] format.
  Tensor out_onnx =
      tf.zeros({(int32_t)batch, (int32_t)q_seq_len, (int32_t)Hq, (int32_t)D});
  std::vector<float> onnx_scores_buf(batch * Hq * q_seq_len * max_seq_len);
  run_onnx_gqa_sdpa(
      q.const_data_ptr<float>(),
      k.const_data_ptr<float>(),
      v.const_data_ptr<float>(),
      out_onnx.mutable_data_ptr<float>(),
      onnx_scores_buf.data(),
      batch, Hq, Hkv, D, max_seq_len, start_pos, q_seq_len, is_transposed);

  // Build reference in [B, S, Hq, D] from out_ref which may be in [B,H,S,D]
  // or [B,S,H,D]. For [B,S,H,D], it's already [B,S,Hq,D]. For [B,H,S,D],
  // transpose to [B,S,Hq,D].
  std::vector<float> ref_bshd(batch * q_seq_len * Hq * D);
  const float* ref_ptr = out_ref.const_data_ptr<float>();
  if (is_transposed) {
    // [B, H, S, D] → [B, S, H, D]
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t h = 0; h < Hq; ++h) {
        for (int64_t s = 0; s < q_seq_len; ++s) {
          const float* src = ref_ptr + ((b * Hq + h) * q_seq_len + s) * D;
          float* dst = ref_bshd.data() + ((b * q_seq_len + s) * Hq + h) * D;
          std::copy(src, src + D, dst);
        }
      }
    }
  } else {
    // Already [B, S, H, D]
    std::copy(ref_ptr, ref_ptr + batch * q_seq_len * Hq * D, ref_bshd.data());
  }

  float onnx_diff = max_abs_diff(
      out_onnx.const_data_ptr<float>(), ref_bshd.data(),
      batch * q_seq_len * Hq * D);
  if (onnx_diff > atol) {
    fprintf(
        stderr,
        "FAIL: OnnxGQA %s %s (B=%ld Hq=%ld Hkv=%ld D=%ld sp=%ld sl=%ld) "
        "max_abs_diff=%.6e > atol=%.6e\n",
        layout, mode, (long)batch, (long)Hq, (long)Hkv, (long)D,
        (long)start_pos, (long)q_seq_len, onnx_diff, atol);
    return false;
  }
  fprintf(
      stderr,
      "PASS: OnnxGQA %s %s (B=%ld Hq=%ld Hkv=%ld D=%ld sp=%ld sl=%ld) "
      "max_abs_diff=%.6e\n",
      layout, mode, (long)batch, (long)Hq, (long)Hkv, (long)D,
      (long)start_pos, (long)q_seq_len, onnx_diff);
  return true;
}

// Run all validation tests. Aborts if any fail.
void run_validation_tests() {
  fprintf(stderr, "--- Validating StandardSDPA vs custom_sdpa_out ---\n");
  // Use moderate dimensions for speed: Hq=8, Hkv=2 (GQA), D=64
  // Online softmax (flash) vs 3-pass softmax can differ at ~1e-5 for float32;
  // use atol=1e-3 to be safe across various kv sizes.
  constexpr float kAtol = 1e-3f;
  bool all_passed = true;

  // Decode configs (q_seq_len=1)
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 1, true, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 1, false, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 64, 1, true, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 64, 1, false, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 128, 1, true, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 128, 1, false, kAtol);

  // Prefill configs
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 16, true, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 16, false, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 64, true, kAtol);
  all_passed &= validate_config(1, 8, 2, 64, 256, 0, 64, false, kAtol);

  // Non-GQA (Hq == Hkv)
  all_passed &= validate_config(1, 8, 8, 64, 256, 64, 1, true, kAtol);
  all_passed &= validate_config(1, 8, 8, 64, 256, 64, 1, false, kAtol);

  if (!all_passed) {
    fprintf(stderr, "VALIDATION FAILED — benchmark results are unreliable\n");
    std::abort();
  }
  fprintf(stderr, "--- All validation tests passed ---\n\n");
}

} // namespace

// Benchmark fixture that sets up tensors for SDPA benchmarking.
// Uses std::optional because ExecuTorch Tensor has a deleted default ctor.
class SDPABenchFixture : public benchmark::Fixture {
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
      q_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_q,
                            (int32_t)q_seq_len, (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                  (int32_t)max_seq_len, (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                  (int32_t)max_seq_len, (int32_t)head_dim}));
      output_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_q,
                                 (int32_t)q_seq_len, (int32_t)head_dim}));
    } else {
      // [B, S, H, D]
      q_.emplace(tf_.zeros({(int32_t)batch, (int32_t)q_seq_len,
                            (int32_t)num_heads_q, (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                  (int32_t)num_heads_kv, (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                  (int32_t)num_heads_kv, (int32_t)head_dim}));
      output_.emplace(tf_.zeros({(int32_t)batch, (int32_t)q_seq_len,
                                 (int32_t)num_heads_q, (int32_t)head_dim}));
    }

    fill_random(*q_, gen);
    fill_random(*k_cache_, gen);
    fill_random(*v_cache_, gen);

    start_pos_ = start_pos;
    is_transposed_ = is_transposed;
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
  bool is_transposed_ = false;
};

// Benchmark custom_sdpa with causal masking
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
        0.0,          // dropout_p
        true,         // is_causal
        std::nullopt, // scale
        is_transposed_,
        is_transposed_,
        is_transposed_,
        *output_);
  }
}

// Benchmark fixture for update_cache
class UpdateCacheBenchFixture : public benchmark::Fixture {
 public:
  // Args: {batch, num_heads, head_dim, max_seq_len, start_pos,
  //        update_seq_len, is_transposed}
  void SetUp(benchmark::State& state) override {
    int64_t batch = state.range(0);
    int64_t num_heads = state.range(1);
    int64_t head_dim = state.range(2);
    int64_t max_seq_len = state.range(3);
    int64_t start_pos = state.range(4);
    int64_t update_seq_len = state.range(5);
    bool is_transposed = state.range(6) != 0;

    std::mt19937 gen(42);

    if (is_transposed) {
      // [B, H, S, D]
      value_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads,
                                (int32_t)update_seq_len, (int32_t)head_dim}));
      cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads,
                                (int32_t)max_seq_len, (int32_t)head_dim}));
    } else {
      // [B, S, H, D]
      value_.emplace(tf_.zeros({(int32_t)batch, (int32_t)update_seq_len,
                                (int32_t)num_heads, (int32_t)head_dim}));
      cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                (int32_t)num_heads, (int32_t)head_dim}));
    }

    fill_random(*value_, gen);
    fill_random(*cache_, gen);
    // Output is a dummy placeholder (unused by update_cache_out)
    update_output_.emplace(tf_.zeros({1}));

    start_pos_ = start_pos;
    is_transposed_ = is_transposed;
  }

  void TearDown(benchmark::State&) override {
    value_.reset();
    cache_.reset();
    update_output_.reset();
  }

  TensorFactory<ScalarType::Float> tf_;
  std::optional<Tensor> value_;
  std::optional<Tensor> cache_;
  std::optional<Tensor> update_output_;
  int64_t start_pos_ = 0;
  bool is_transposed_ = false;
};

// Benchmark update_cache
BENCHMARK_DEFINE_F(UpdateCacheBenchFixture, UpdateCache)
(benchmark::State& state) {
  for (auto _ : state) {
    KernelRuntimeContext ctx{};
    torch::executor::native::update_cache_out(
        ctx,
        *value_,
        *cache_,
        start_pos_,
        is_transposed_,
        *update_output_);
  }
}

// Combined update_cache + custom_sdpa fixture
class CombinedBenchFixture : public benchmark::Fixture {
 public:
  // Args: {batch, num_heads_q, num_heads_kv, head_dim, max_seq_len, start_pos,
  //        seq_len, is_transposed}
  void SetUp(benchmark::State& state) override {
    int64_t batch = state.range(0);
    int64_t num_heads_q = state.range(1);
    int64_t num_heads_kv = state.range(2);
    int64_t head_dim = state.range(3);
    int64_t max_seq_len = state.range(4);
    int64_t start_pos = state.range(5);
    int64_t seq_len = state.range(6);
    bool is_transposed = state.range(7) != 0;

    std::mt19937 gen(42);

    if (is_transposed) {
      // [B, H, S, D]
      q_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_q,
                            (int32_t)seq_len, (int32_t)head_dim}));
      k_proj_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                 (int32_t)seq_len, (int32_t)head_dim}));
      v_proj_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                 (int32_t)seq_len, (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                  (int32_t)max_seq_len, (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                  (int32_t)max_seq_len, (int32_t)head_dim}));
      output_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_q,
                                 (int32_t)seq_len, (int32_t)head_dim}));
    } else {
      // [B, S, H, D]
      q_.emplace(tf_.zeros({(int32_t)batch, (int32_t)seq_len,
                            (int32_t)num_heads_q, (int32_t)head_dim}));
      k_proj_.emplace(tf_.zeros({(int32_t)batch, (int32_t)seq_len,
                                 (int32_t)num_heads_kv, (int32_t)head_dim}));
      v_proj_.emplace(tf_.zeros({(int32_t)batch, (int32_t)seq_len,
                                 (int32_t)num_heads_kv, (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                  (int32_t)num_heads_kv, (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                  (int32_t)num_heads_kv, (int32_t)head_dim}));
      output_.emplace(tf_.zeros({(int32_t)batch, (int32_t)seq_len,
                                 (int32_t)num_heads_q, (int32_t)head_dim}));
    }

    fill_random(*q_, gen);
    fill_random(*k_proj_, gen);
    fill_random(*v_proj_, gen);
    fill_random(*k_cache_, gen);
    fill_random(*v_cache_, gen);

    update_output_.emplace(tf_.zeros({1}));
    start_pos_ = start_pos;
    is_transposed_ = is_transposed;
  }

  void TearDown(benchmark::State&) override {
    q_.reset();
    k_proj_.reset();
    v_proj_.reset();
    k_cache_.reset();
    v_cache_.reset();
    output_.reset();
    update_output_.reset();
  }

  TensorFactory<ScalarType::Float> tf_;
  std::optional<Tensor> q_;
  std::optional<Tensor> k_proj_;
  std::optional<Tensor> v_proj_;
  std::optional<Tensor> k_cache_;
  std::optional<Tensor> v_cache_;
  std::optional<Tensor> output_;
  std::optional<Tensor> update_output_;
  int64_t start_pos_ = 0;
  bool is_transposed_ = false;
};

// Benchmark combined update_cache + custom_sdpa
BENCHMARK_DEFINE_F(CombinedBenchFixture, CombinedUpdateSDPA)
(benchmark::State& state) {
  for (auto _ : state) {
    KernelRuntimeContext ctx{};
    torch::executor::native::update_cache_out(
        ctx, *k_proj_, *k_cache_, start_pos_, is_transposed_, *update_output_);
    torch::executor::native::update_cache_out(
        ctx, *v_proj_, *v_cache_, start_pos_, is_transposed_, *update_output_);
    torch::executor::native::custom_sdpa_out(
        ctx,
        *q_,
        *k_cache_,
        *v_cache_,
        start_pos_,
        std::nullopt, // attn_mask
        0.0,          // dropout_p
        true,         // is_causal
        std::nullopt, // scale
        is_transposed_,
        is_transposed_,
        is_transposed_,
        *output_);
  }
}

// Standard (ONNX-style) SDPA benchmark using GEMM.
// No tiling — single GEMM per head for Q@K^T and scores@V, with standard
// 3-pass softmax. Supports both [B, H, S, D] and [B, S, H, D] layouts via
// BLAS leading dimension, so we can isolate algorithm vs layout effects.
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
      q_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_q,
                            (int32_t)q_seq_len, (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                  (int32_t)max_seq_len, (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                  (int32_t)max_seq_len, (int32_t)head_dim}));
      output_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_q,
                                 (int32_t)q_seq_len, (int32_t)head_dim}));
    } else {
      // [B, S, H, D]
      q_.emplace(tf_.zeros({(int32_t)batch, (int32_t)q_seq_len,
                            (int32_t)num_heads_q, (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                  (int32_t)num_heads_kv, (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                  (int32_t)num_heads_kv, (int32_t)head_dim}));
      output_.emplace(tf_.zeros({(int32_t)batch, (int32_t)q_seq_len,
                                 (int32_t)num_heads_q, (int32_t)head_dim}));
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

    // Per-work-unit scores buffer: [q_seq_len, kv_size] per (batch, head)
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
        q_data, k_data, v_data, out_data, scores_buf_.data(),
        batch_, num_heads_q_, num_heads_kv_, head_dim_,
        max_seq_len_, start_pos_, q_seq_len_, is_transposed_);
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
      q_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_q,
                            (int32_t)q_seq_len, (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                  (int32_t)max_seq_len, (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)num_heads_kv,
                                  (int32_t)max_seq_len, (int32_t)head_dim}));
    } else {
      q_.emplace(tf_.zeros({(int32_t)batch, (int32_t)q_seq_len,
                            (int32_t)num_heads_q, (int32_t)head_dim}));
      k_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                  (int32_t)num_heads_kv, (int32_t)head_dim}));
      v_cache_.emplace(tf_.zeros({(int32_t)batch, (int32_t)max_seq_len,
                                  (int32_t)num_heads_kv, (int32_t)head_dim}));
    }
    // Output always [B, S, Hq, D] (ONNX convention)
    output_.emplace(tf_.zeros({(int32_t)batch, (int32_t)q_seq_len,
                               (int32_t)num_heads_q, (int32_t)head_dim}));

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
        q_data, k_data, v_data, out_data, scores_buf_.data(),
        batch_, num_heads_q_, num_heads_kv_, head_dim_,
        max_seq_len_, start_pos_, q_seq_len_, is_transposed_);
  }
}

/*
 * Benchmark configurations modeled after Llama 3 8B (GQA: 32 q heads, 8 kv
 * heads, head_dim=128). We test decode (seq_len=1) and prefill scenarios at
 * various cache fill levels, comparing standard vs transposed layout.
 */

// --- custom_sdpa: Decode (seq_len=1) ---
// Args: {batch, Hq, Hkv, D, MaxS, StartPos, SeqLen, Transposed}
BENCHMARK_REGISTER_F(SDPABenchFixture, CustomSDPA)
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
    ->ArgNames(
        {"B", "Hq", "Hkv", "D", "MaxS", "StartPos", "SeqLen", "Trans"});

// --- update_cache ---
// Args: {batch, H, D, MaxS, StartPos, SeqLen, Transposed}
BENCHMARK_REGISTER_F(UpdateCacheBenchFixture, UpdateCache)
    // Decode (seq_len=1)
    ->Args({1, 8, 128, 2048, 0, 1, 0})
    ->Args({1, 8, 128, 2048, 256, 1, 0})
    ->Args({1, 8, 128, 2048, 1024, 1, 0})
    ->Args({1, 8, 128, 2048, 0, 1, 1})
    ->Args({1, 8, 128, 2048, 256, 1, 1})
    ->Args({1, 8, 128, 2048, 1024, 1, 1})
    // Prefill
    ->Args({1, 8, 128, 2048, 0, 128, 0})
    ->Args({1, 8, 128, 2048, 0, 512, 0})
    ->Args({1, 8, 128, 2048, 0, 128, 1})
    ->Args({1, 8, 128, 2048, 0, 512, 1})
    ->ArgNames({"B", "H", "D", "MaxS", "StartPos", "SeqLen", "Trans"});

// --- Combined: update_cache + custom_sdpa ---
// Args: {batch, Hq, Hkv, D, MaxS, StartPos, SeqLen, Transposed}
BENCHMARK_REGISTER_F(CombinedBenchFixture, CombinedUpdateSDPA)
    // Decode at various positions
    ->Args({1, 32, 8, 128, 2048, 0, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 256, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 512, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 1024, 1, 0})
    ->Args({1, 32, 8, 128, 2048, 0, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 256, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 512, 1, 1})
    ->Args({1, 32, 8, 128, 2048, 1024, 1, 1})
    // Prefill
    ->Args({1, 32, 8, 128, 2048, 0, 128, 0})
    ->Args({1, 32, 8, 128, 2048, 0, 128, 1})
    ->ArgNames(
        {"B", "Hq", "Hkv", "D", "MaxS", "StartPos", "SeqLen", "Trans"});

// --- Standard SDPA (ONNX-style, GEMM-based) ---
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
    ->ArgNames(
        {"B", "Hq", "Hkv", "D", "MaxS", "StartPos", "SeqLen", "Trans"});

// --- ONNX Runtime GQA-style SDPA ---
// Same configs as above. Differences: scale-in-alpha, padded scores buffer
// (ld=MaxS), narrow softmax, output in [B,S,Hq,D] with stride Hq*D.
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
    ->ArgNames(
        {"B", "Hq", "Hkv", "D", "MaxS", "StartPos", "SeqLen", "Trans"});

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  run_validation_tests();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
