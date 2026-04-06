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

#include <optional>
#include <random>
#include <vector>

#include <executorch/extension/llm/custom_ops/op_sdpa.h>
#include <executorch/extension/llm/custom_ops/op_update_cache.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

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
        *output_);
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

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
