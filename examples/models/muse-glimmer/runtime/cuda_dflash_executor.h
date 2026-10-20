/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/examples/models/muse-glimmer/runtime/backend_factory.h>

#include <unordered_map>

namespace executorch::extension::llm {

// Exact packed-export metadata contract. Geometry tensors use the ordinary
// get_n_caches/get_kv_heads/get_head_dims/get_windows names with get_draft_.
namespace cuda_dflash_detail {
inline constexpr char kForward[] = "draft_forward";
inline constexpr char kBlock[] = "get_block_size";
inline constexpr char kMask[] = "get_mask_token_id";
inline constexpr char kTargetLayers[] = "get_n_target_layers";
inline constexpr char kMaxOwners[] = "get_draft_max_owners";
inline constexpr char kMaxWrite[] = "get_draft_max_step_tokens";
inline constexpr char kMaxCells[] = "get_draft_offgraph_kv_max_cells";

struct GreedyBurst {
  std::vector<batching::Token> tokens;
  std::size_t accepted = 0;
  bool bonus = false;
};
// Verification returns only new predictions, never the consumed anchor.
std::optional<GreedyBurst> verify_greedy(
    const std::vector<batching::Token>& proposals,
    const std::vector<batching::Token>& predictions);
bool trim_hidden(
    std::vector<std::uint8_t>& hidden,
    std::size_t row_bytes,
    int draft_frontier,
    int position);

// Reusable owned storage for borrowed target taps. Capture precedes every
// sampler/next slice; host destinations stay alive until fence().
class HiddenSnapshot {
 public:
  ~HiddenSnapshot();
  HiddenSnapshot() = default;
  HiddenSnapshot(const HiddenSnapshot&) = delete;
  HiddenSnapshot& operator=(const HiddenSnapshot&) = delete;
  HiddenSnapshot(HiddenSnapshot&&) = delete;
  HiddenSnapshot& operator=(HiddenSnapshot&&) = delete;
  bool begin(std::size_t rows, std::size_t row_bytes);
  bool capture(const aten::Tensor& taps, std::size_t offset, std::size_t rows);
  bool copy_rows(
      std::vector<std::uint8_t>& destination,
      std::size_t offset,
      std::size_t rows);
  bool fence();
  bool on_device() const {
    return device_source_.value_or(false);
  }

 private:
  void* device_ = nullptr;
  void* stream_ = nullptr;
  std::size_t capacity_ = 0, rows_ = 0, row_bytes_ = 0, captured_ = 0;
  std::vector<std::uint8_t> host_;
  std::optional<bool> device_source_;
  bool stream_set_ = false;
};

struct DraftContext {
  std::int32_t id;
  int position;
  batching::Token anchor;
  const std::vector<std::uint8_t>* hidden;
  int count;
};
struct DraftPlan {
  std::vector<std::int64_t> tokens, positions, kv_order;
  std::vector<std::int32_t> ids;
  std::vector<std::uint8_t> hidden;
  std::vector<int> counts, ends;
};
std::optional<DraftPlan> pack_draft(
    const std::vector<DraftContext>& owners,
    int block,
    batching::Token mask,
    int context_limit,
    int max_context,
    std::size_t row_bytes);
} // namespace cuda_dflash_detail

struct CudaDFlashStats {
  std::uint64_t executions = 0;
  std::uint64_t draft_forwards = 0;
  std::uint64_t warmed_context_tokens = 0;
  // Calls are packed execute_raw attempts; forwards count slice callbacks.
  std::uint64_t pending_decode_batches = 0;
  std::uint64_t pending_decode_owners = 0;
  std::uint64_t target_preamble_calls = 0;
  std::uint64_t target_verify_calls = 0;
  std::uint64_t target_preamble_forwards = 0;
  std::uint64_t target_verify_forwards = 0;
  // Actual CUDA transfers, excluding static padding and CPU-resident taps.
  std::uint64_t hidden_rows_d2h = 0, hidden_bytes_d2h = 0;
  std::uint64_t hidden_rows_d2d = 0, hidden_bytes_d2d = 0;
  std::uint64_t cycles = 0; // per-owner verification cycles
  std::uint64_t proposed = 0;
  std::uint64_t accepted = 0;
  std::uint64_t rejected = 0; // cycles ended by a rejection, not tokens
  std::uint64_t bonuses = 0;
  std::uint64_t failed_batches = 0;
};

// Text-only model adapter over the existing CUDA target executor. Draft runs
// eagerly; target methods retain their ordinary static CUDA graphs. Sampling
// other than exactly temperature zero is rejected, not approximated.
class MuseGlimmerCudaDFlashExecutor final : public batching::Executor {
 public:
  static runtime::Result<MuseGlimmerBackend> create(
      const MuseGlimmerBackendConfig& config);
  MuseGlimmerCudaDFlashExecutor(const MuseGlimmerCudaDFlashExecutor&) = delete;
  MuseGlimmerCudaDFlashExecutor& operator=(
      const MuseGlimmerCudaDFlashExecutor&) = delete;
  MuseGlimmerCudaDFlashExecutor(MuseGlimmerCudaDFlashExecutor&&) = delete;
  MuseGlimmerCudaDFlashExecutor& operator=(MuseGlimmerCudaDFlashExecutor&&) =
      delete;
  ~MuseGlimmerCudaDFlashExecutor() override;
  std::size_t preferred_batch_tokens() const override;
  bool initialize() override;
  std::optional<batching::SessionId> open_session() override;
  void close_session(batching::SessionId session) override;
  std::optional<batching::SessionId> clone(
      batching::SessionId source,
      batching::Position upto) override;
  void set_sampling(
      batching::SessionId session,
      const batching::SamplingParams& params,
      std::optional<std::uint64_t> seed) override;
  bool execute(const batching::BatchInput& batch, batching::BatchOutput& out)
      override;
  // Engine-thread-only, between executes. Counters measure actual adapter work.
  CudaDFlashStats stats() const {
    return stats_;
  }

 private:
  friend struct CudaDFlashTestPeer;
  using Target = ::executorch::backends::cuda::batching::CudaExecutor;
  using RawStep = ::executorch::backends::cuda::batching::RawTargetStep;
  struct State {
    State(std::int32_t target, std::int32_t draft)
        : target_id(target), draft_id(draft) {}
    std::int32_t target_id;
    std::int32_t draft_id;
    int frontier = 0;
    // Last emitted correction/bonus, not yet consumed by the target cache.
    std::optional<batching::Token> pending;
    // Host-owned committed BF16 rows [draft frontier, target frontier).
    // Verification appends only the consumed anchor and accepted proposals.
    std::vector<std::uint8_t> hidden;
    std::optional<batching::SamplingParams> sampling;
    std::optional<std::uint64_t> seed;
    bool poisoned = false;
  };
  MuseGlimmerCudaDFlashExecutor(
      std::unique_ptr<Target> target,
      std::shared_ptr<cache::Cache> draft_cache,
      int block,
      int proposals,
      batching::Token mask,
      int max_owners,
      int fresh_context,
      int max_sessions,
      int max_context,
      int hidden_width);
  bool rewind(State& state, int position);
  bool run_target(
      const RawStep& step,
      const std::vector<batching::SessionId>& token_owners,
      std::vector<batching::Token>& predictions,
      bool drain_context);
  bool drain(batching::SessionId owner);
  bool draft(
      const std::vector<batching::SessionId>& owners,
      const std::vector<batching::Token>& anchors,
      int context_limit,
      std::vector<std::vector<batching::Token>>* proposals);
  bool speculate(
      const batching::BatchInput& batch,
      batching::BatchOutput& out,
      bool pending_decode);
  bool fail_batch(
      const batching::BatchInput& batch,
      batching::BatchOutput& out);

  // Both guards exist before either cache delegate loads. Destruction releases
  // the target-owned Module/delegates before removing the draft registry entry.
  std::shared_ptr<cache::Cache> draft_cache_;
  cache::InstallGuard draft_install_;
  std::unique_ptr<Target> target_;
  cache::BatchControl* const draft_ctl_;
  int block_, proposals_, max_owners_, fresh_context_, max_sessions_,
      max_context_, hidden_width_;
  batching::Token mask_;
  std::size_t row_bytes_;
  batching::SessionId next_session_ = 1;
  std::unordered_map<batching::SessionId, State> sessions_;
  CudaDFlashStats stats_;
  cuda_dflash_detail::HiddenSnapshot verification_hidden_;
  bool initialized_ = false;
  // A failed co-batch invalidates every draft owner. Only model restart
  // recovers.
  bool failed_ = false;
};

} // namespace executorch::extension::llm
