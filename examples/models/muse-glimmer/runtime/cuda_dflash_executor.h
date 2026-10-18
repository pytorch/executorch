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

#include <array>
#include <unordered_map>

namespace executorch::extension::llm {

class MuseGlimmerModelEmbeddings;

// The preparation spec for prompts to `target`. When the program has
// vision_encoder, also installs an image embedder on it; `embeddings` then
// receives it, for adapters that build their own target steps.
runtime::Result<std::shared_ptr<const MuseGlimmerPreparationSpec>>
make_cuda_preparation(
    ::executorch::backends::cuda::batching::CudaExecutor& target,
    const MuseGlimmerBackendConfig& config,
    std::int32_t max_context,
    std::shared_ptr<MuseGlimmerModelEmbeddings>* embeddings = nullptr);

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
// Static draft_forward_{N}: N owners, each with at most one block of context.
inline constexpr char kStaticForwardPrefix[] = "draft_forward_";
// On-device verification of [owners, width] candidates; see
// model/dflash_token_sampler.py.
inline constexpr char kVerify[] = "dflash_verify";
inline constexpr char kVerifyArgmax[] = "dflash_verify_argmax";
inline constexpr int kOwnerLimit = 4;

struct Burst {
  std::vector<batching::Token> tokens;
  std::size_t accepted = 0;
  bool bonus = false;
};
// The new tokens of one verified block: the accepted proposals, then the
// correction or bonus. Never the consumed anchor.
std::optional<Burst> make_burst(
    const std::vector<batching::Token>& proposals,
    std::int64_t accepted,
    std::int64_t token,
    std::int32_t vocab_size);
// A uniform in [0, 1) that depends only on the seed, position and purpose, so
// a seed reproduces a generation however its requests are batched.
float positional_uniform(
    std::uint64_t seed,
    batching::Position position,
    int purpose);
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
// Widens a plan whose context fits one block per owner to the static
// draft_forward_{N} shape: unused context rows go last, after every block, so
// the cache writes them to its scratch cell and no query attends them.
bool pad_to_static(
    DraftPlan& plan,
    int owners,
    int block,
    std::size_t row_bytes);

// Reusable device memory for small host-built inputs.
class DeviceStaging {
 public:
  ~DeviceStaging();
  DeviceStaging() = default;
  DeviceStaging(const DeviceStaging&) = delete;
  DeviceStaging& operator=(const DeviceStaging&) = delete;
  // Device copies of each (data, bytes) part, ordered on stream. Earlier
  // uploads must be consumed first: they share one allocation.
  std::optional<std::vector<void*>> upload(
      const std::vector<std::pair<const void*, std::size_t>>& parts,
      void* stream);

 private:
  void* device_ = nullptr;
  std::size_t capacity_ = 0;
};
} // namespace cuda_dflash_detail

struct CudaDFlashStats {
  std::uint64_t executions = 0;
  std::uint64_t draft_forwards = 0;
  std::uint64_t static_draft_forwards = 0;
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
  std::uint64_t rejected = 0;
  std::uint64_t bonuses = 0;
  std::uint64_t failed_batches = 0;
};

// Text-only model adapter over the existing CUDA target executor. Decode drafts
// run the static draft_forward_{N} in CUDA graphs, as the target's static
// methods do; longer contexts drain through the dynamic draft_forward. Drafts
// are greedy; dflash_verify accepts or resamples them under each session's
// sampling on the device, so only token IDs return to the host.
class MuseGlimmerCudaDFlashExecutor final : public batching::Executor {
 public:
  static runtime::Result<MuseGlimmerBackend> create(
      const MuseGlimmerBackendConfig& config);
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
      std::vector<batching::Token>& predictions);
  // Runs the verification step and the on-device verifier; results holds
  // [accepted, correction] per owner.
  bool verify(
      const RawStep& step,
      const std::vector<batching::SessionId>& owners,
      const std::vector<std::int64_t>& candidates,
      std::vector<std::int64_t>& results);
  std::vector<float> verify_params(const State& state, batching::Position first)
      const;
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
  cuda_dflash_detail::DeviceStaging verify_inputs_;
  // static_drafts_[n]: the program has draft_forward_{n}.
  std::array<bool, cuda_dflash_detail::kOwnerLimit + 1> static_drafts_{};
  bool initialized_ = false;
  // A failed co-batch invalidates every draft owner. Only model restart
  // recovers.
  bool failed_ = false;
};

} // namespace executorch::extension::llm
