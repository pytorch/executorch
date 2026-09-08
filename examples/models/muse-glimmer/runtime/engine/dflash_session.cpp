/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/muse-glimmer/runtime/engine/dflash_session.h>

#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#endif

namespace executorch::extension::llm {
namespace {

using ::executorch::extension::from_blob;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;
using SizesType = executorch::aten::SizesType;
using Clock = std::chrono::steady_clock;

constexpr const char* kTargetForwardFromEmbeddings =
    "target_forward_from_embeddings";
constexpr const char* kTargetPrefillFromEmbeddings =
    "target_prefill_from_embeddings";
constexpr const char* kEmbedText = "embed_text";
constexpr const char* kDraftForward = "draft_forward";
constexpr const char* kDraftPrefill = "draft_prefill";
constexpr const char* kMaxContextLen = "get_max_seq_len";

bool valid_temperature(float temperature) {
  return temperature == -1.0f || (temperature >= 0.0f && temperature <= 2.0f);
}

bool valid_top_p(float top_p) {
  return top_p > 0.0f && top_p <= 1.0f;
}

bool same_sampling(const SamplingConfig& lhs, const SamplingConfig& rhs) {
  return lhs.temperature == rhs.temperature && lhs.top_p == rhs.top_p &&
      lhs.top_k == rhs.top_k && lhs.seed == rhs.seed;
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

class ProbabilityWorkers {
 public:
  explicit ProbabilityWorkers(int64_t worker_count) {
    workers_.reserve(worker_count);
    for (int64_t worker_index = 0; worker_index < worker_count;
         ++worker_index) {
      auto worker = std::make_unique<Worker>();
      worker->thread = std::thread([this, worker_index, worker = worker.get()] {
        run_worker(worker_index, worker->workspace);
      });
      workers_.push_back(std::move(worker));
    }
  }

  ~ProbabilityWorkers() {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      stopping_ = true;
      ++generation_;
    }
    start_.notify_all();
    for (auto& worker : workers_) {
      worker->thread.join();
    }
  }

  ProbabilityWorkers(const ProbabilityWorkers&) = delete;
  ProbabilityWorkers& operator=(const ProbabilityWorkers&) = delete;

  void fill(
      const executorch::aten::Tensor& logits,
      int64_t row_count,
      int64_t logits_row_offset,
      int64_t probabilities_row_offset,
      double temperature,
      int32_t top_k,
      double top_p,
      std::vector<std::vector<float>>& probabilities) {
    const int64_t vocab_size = logits.size(logits.dim() - 1);
    {
      std::lock_guard<std::mutex> guard(mutex_);
      logits_ = static_cast<const float*>(logits.const_data_ptr());
      vocab_size_ = vocab_size;
      row_count_ = row_count;
      logits_row_offset_ = logits_row_offset;
      probabilities_row_offset_ = probabilities_row_offset;
      temperature_ = temperature;
      top_k_ = top_k;
      top_p_ = top_p;
      probabilities_ = &probabilities;
      row_stride_ = workers_.size() + (row_count > workers_.size() ? 1 : 0);
      completed_ = 0;
      ++generation_;
    }
    start_.notify_all();
    if (row_count > static_cast<int64_t>(workers_.size())) {
      for (int64_t row = workers_.size(); row < row_count; row += row_stride_) {
        muse_glimmer::fill_sampling_probabilities(
            logits_ + (logits_row_offset + row) * vocab_size,
            vocab_size,
            temperature,
            top_k,
            top_p,
            probabilities[probabilities_row_offset + row],
            caller_workspace_);
      }
    }
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return completed_ == workers_.size(); });
  }

 private:
  struct Worker {
    muse_glimmer::SamplingWorkspace workspace;
    std::thread thread;
  };

  void run_worker(
      int64_t worker_index,
      muse_glimmer::SamplingWorkspace& workspace) {
    uint64_t observed_generation = 0;
    while (true) {
      const float* logits = nullptr;
      int64_t vocab_size = 0;
      int64_t row_count = 0;
      int64_t logits_row_offset = 0;
      int64_t probabilities_row_offset = 0;
      int64_t row_stride = 0;
      double temperature = 0.0;
      int32_t top_k = 0;
      double top_p = 1.0;
      std::vector<std::vector<float>>* probabilities = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_.wait(lock, [this, &observed_generation] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_) {
          return;
        }
        observed_generation = generation_;
        logits = logits_;
        vocab_size = vocab_size_;
        row_count = row_count_;
        logits_row_offset = logits_row_offset_;
        probabilities_row_offset = probabilities_row_offset_;
        row_stride = row_stride_;
        temperature = temperature_;
        top_k = top_k_;
        top_p = top_p_;
        probabilities = probabilities_;
      }

      for (int64_t row = worker_index; row < row_count; row += row_stride) {
        muse_glimmer::fill_sampling_probabilities(
            logits + (logits_row_offset + row) * vocab_size,
            vocab_size,
            temperature,
            top_k,
            top_p,
            (*probabilities)[probabilities_row_offset + row],
            workspace);
      }

      {
        std::lock_guard<std::mutex> guard(mutex_);
        ++completed_;
        if (completed_ == workers_.size()) {
          done_.notify_one();
        }
      }
    }
  }

  std::vector<std::unique_ptr<Worker>> workers_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  bool stopping_ = false;
  uint64_t generation_ = 0;
  size_t completed_ = 0;
  const float* logits_ = nullptr;
  int64_t vocab_size_ = 0;
  int64_t row_count_ = 0;
  int64_t logits_row_offset_ = 0;
  int64_t probabilities_row_offset_ = 0;
  int64_t row_stride_ = 0;
  double temperature_ = 0.0;
  int32_t top_k_ = 0;
  double top_p_ = 1.0;
  std::vector<std::vector<float>>* probabilities_ = nullptr;
  muse_glimmer::SamplingWorkspace caller_workspace_;
};

class DFlashSession final : public LLMSession, public DFlashMultimodalSession {
 public:
  explicit DFlashSession(DFlashSessionConfig config)
      : config_(std::move(config)),
        block_length_(
            config_.block_length > 0 ? config_.block_length
                                     : config_.block_size),
        n_draft_(config_.n_draft > 0 ? config_.n_draft : block_length_ - 1),
        hidden_dtype_(
            config_.activation_dtype == "bfloat16"
                ? executorch::aten::ScalarType::BFloat16
                : executorch::aten::ScalarType::Half),
        rng_(std::random_device{}()) {}

  ~DFlashSession() override {
    probability_workers_.reset();
    if (config_.mutable_state != nullptr &&
        config_.session_token != kMuseGlimmerNoMutableSession) {
      config_.mutable_state->destroy_session(config_.session_token);
    }
    if (config_.live_sessions != nullptr) {
      config_.live_sessions->fetch_sub(1);
    }
  }

  Error prefill_tokens(
      const std::vector<uint64_t>& tokens,
      const SamplingConfig* initial_sampling) override {
    // Moving the staged value up front makes image staging one-shot on every
    // prefill exit path, including validation failures.
    std::optional<PreparedMuseGlimmerImage> image = std::move(staged_image_);
    staged_image_.reset();
    preserve_staged_image_on_next_reset_ = false;

    if (poisoned_) {
      ET_LOG(Error, "DFlashSession is poisoned; reset before prefill");
      return Error::InvalidState;
    }
    if (tokens.empty()) {
      ET_LOG(Error, "DFlash prefill requires at least one token");
      return Error::InvalidArgument;
    }

    int64_t next_image_row = 0;
    if (image.has_value()) {
      ET_CHECK_OR_RETURN_ERROR(
          target_pos_ == 0 && logical_pos_ == 0 && cached_ctx_len_ == 0 &&
              hidden_.empty(),
          InvalidState,
          "DFlash image prefill requires a freshly reset session");
      ET_CHECK_OK_OR_RETURN_ERROR(validate_staged_image(tokens, *image));
    }

    SamplingConfig sampling = sampling_;
    if (initial_sampling != nullptr) {
      sampling = *initial_sampling;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(validate_sampling(sampling));
    sampling_ = sampling;
    sampling_initialized_ = true;
    seed_rng(sampling_);
    if (effective_temperature() > 0.0 && n_draft_ > 1 &&
        probability_workers_ == nullptr) {
      probability_workers_ = std::make_unique<ProbabilityWorkers>(n_draft_);
    }
    stop_.store(false, std::memory_order_relaxed);

    const int64_t token_count = static_cast<int64_t>(tokens.size());
    const int64_t max_context = max_context_len();
    if (max_context > 0 && logical_pos_ + token_count >= max_context) {
      ET_LOG(Error, "DFlash prefill would leave no room to generate");
      return Error::InvalidArgument;
    }

    // A request may end while verified lookahead is still queued. The worker's
    // suffix starts at logical_pos_, so retain only target hidden state through
    // that caller-visible boundary and overwrite speculative cache slots there.
    ET_CHECK_OK_OR_RETURN_ERROR(discard_unreturned_lookahead());

    int64_t offset = 0;
#ifdef EXECUTORCH_BUILD_CUDA
    const int64_t retained_hidden_start = config_.draft_sliding_window > 0
        ? std::max<int64_t>(
              logical_pos_,
              logical_pos_ + token_count - config_.draft_sliding_window)
        : logical_pos_;
    if (retained_hidden_start > logical_pos_) {
      hidden_.clear();
      cached_ctx_len_ = retained_hidden_start;
    }
#endif
    while (offset < token_count) {
      int64_t chunk = token_count - offset;
#ifndef EXECUTORCH_BUILD_CUDA
      chunk = std::min(
          chunk,
          config_.max_prefill_chunk > 0 ? config_.max_prefill_chunk : chunk);
#endif
      bool use_target_prefill = false;
#ifdef EXECUTORCH_BUILD_CUDA
      if (config_.max_target_prefill_chunk > 0 &&
          chunk >= config_.min_target_prefill_chunk) {
        chunk = std::min(chunk, config_.max_target_prefill_chunk);
        const int64_t tail = token_count - offset - chunk;
        if (tail > 0 && tail < config_.min_target_prefill_chunk) {
          chunk -= config_.min_target_prefill_chunk - tail;
        }
        use_target_prefill = chunk >= config_.min_target_prefill_chunk;
      } else {
        chunk = std::min<int64_t>(chunk, kCudaDFlashHiddenRows);
      }
#endif
      const bool is_final_chunk = offset + chunk == token_count;
#ifdef EXECUTORCH_BUILD_CUDA
      const int64_t retain_from_row = std::min<int64_t>(
          chunk, std::max<int64_t>(0, retained_hidden_start - logical_pos_));
#else
      const int64_t retain_from_row = 0;
#endif
      auto result = run_target_from_embeddings(
          tokens.data() + offset,
          chunk,
          target_pos_,
          sampling_,
          /*sample_last=*/is_final_chunk,
          image.has_value() ? &*image : nullptr,
          image.has_value() ? &next_image_row : nullptr,
          /*timing=*/nullptr,
          use_target_prefill,
          retain_from_row);
      if (result.error() != Error::Ok) {
        poisoned_ = true;
        return result.error();
      }
      if (is_final_chunk) {
        pending_token_ = result->sampled_token;
      }
      append_hidden(result->hidden);
      target_pos_ += chunk;
      logical_pos_ += chunk;
      offset += chunk;
    }

    ET_CHECK_OR_RETURN_ERROR(
        !image.has_value() || next_image_row == image->num_soft_tokens,
        InvalidState,
        "DFlash image prefill did not consume every image embedding row");
    prev_decode_token_ = tokens.back();
    trim_hidden_window();
    const Error prime_error = prime_draft_cache();
    if (prime_error != Error::Ok) {
      poisoned_ = true;
      return prime_error;
    }
    return Error::Ok;
  }

  Result<DecodeResult> decode_one(const SamplingConfig& sampling) override {
    if (poisoned_) {
      ET_LOG(Error, "DFlashSession is poisoned; reset before decode");
      return Error::InvalidState;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(validate_sampling(sampling));
    if (sampling_initialized_ && !same_sampling(sampling_, sampling)) {
      ET_LOG(
          Error,
          "Cannot change DFlash sampling after prefill; prefill a suffix first");
      return Error::InvalidState;
    }
    if (!sampling_initialized_) {
      sampling_ = sampling;
      seed_rng(sampling_);
      sampling_initialized_ = true;
    }

    if (stop_.load(std::memory_order_relaxed)) {
      return DecodeResult{0, "", /*is_eos=*/false, /*is_terminal=*/true};
    }
    if (!pending_token_.has_value() && ready_tokens_.empty()) {
      ET_LOG(Error, "DFlash decode requires prefill");
      return Error::InvalidState;
    }

    if (ready_tokens_.empty()) {
      if (is_eos(*pending_token_)) {
        const uint64_t eos = *pending_token_;
        pending_token_.reset();
        return terminal_result(eos);
      }
      const Error cycle_error = run_decode_cycle();
      if (cycle_error != Error::Ok) {
        poisoned_ = true;
        return cycle_error;
      }
    }

    if (ready_tokens_.empty()) {
      poisoned_ = true;
      ET_LOG(Error, "DFlash cycle produced no committed token");
      return Error::InvalidState;
    }
    const uint64_t token = ready_tokens_.front();
    ready_tokens_.pop_front();
    auto decoded = decode_piece(token);
    if (decoded.error() != Error::Ok) {
      poisoned_ = true;
      return decoded.error();
    }
    ++logical_pos_;
    prev_decode_token_ = token;
    return DecodeResult{
        token,
        std::move(decoded.get()),
        /*is_eos=*/false,
        /*is_terminal=*/false};
  }

  int64_t position() const override {
    return logical_pos_;
  }

  Error reset() override {
    ready_tokens_.clear();
    pending_token_.reset();
    hidden_.clear();
    hidden_dim_ = 0;
    cached_ctx_len_ = 0;
    target_pos_ = 0;
    logical_pos_ = 0;
    prev_decode_token_.reset();
    sampling_initialized_ = false;
    poisoned_ = false;
    if (preserve_staged_image_on_next_reset_) {
      // Staging may precede the worker's full-request reset. Preserve only
      // across that single reset; any later reset clears the one-shot value.
      preserve_staged_image_on_next_reset_ = false;
    } else {
      staged_image_.reset();
    }
    if (config_.timing != nullptr) {
      config_.timing->reset();
    }
    stop_.store(false, std::memory_order_relaxed);
    return Error::Ok;
  }

  void stop() override {
    stop_.store(true, std::memory_order_relaxed);
  }

  void stage_image_for_next_prefill(PreparedMuseGlimmerImage image) override {
    staged_image_ = std::move(image);
    preserve_staged_image_on_next_reset_ = true;
  }

  void clear_staged_image() override {
    staged_image_.reset();
    preserve_staged_image_on_next_reset_ = false;
  }

 private:
  struct TargetResult {
    uint64_t sampled_token = 0;
    std::vector<uint16_t> hidden;
  };

  Error validate_staged_image(
      const std::vector<uint64_t>& tokens,
      const PreparedMuseGlimmerImage& image) const {
    ET_CHECK_OR_RETURN_ERROR(
        image.num_soft_tokens > 0 && image.hidden_dim > 0,
        InvalidArgument,
        "DFlash staged image dimensions must be positive");
    ET_CHECK_OR_RETURN_ERROR(
        image.embeddings.size() % image.num_soft_tokens == 0 &&
            image.embeddings.size() / image.num_soft_tokens == image.hidden_dim,
        InvalidArgument,
        "DFlash staged image data size does not match its dimensions");

    // A contiguous run of <|patch|> tokens identifies the image span.
    std::optional<int64_t> first;
    int64_t last = -1;
    int64_t patch_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(tokens.size()); ++i) {
      if (tokens[i] == kMuseGlimmerImagePatchTokenId) {
        if (!first.has_value()) {
          first = i;
        }
        last = i;
        ++patch_count;
      }
    }
    ET_CHECK_OR_RETURN_ERROR(
        patch_count == image.num_soft_tokens,
        InvalidArgument,
        "DFlash image span patch count does not match staged embeddings");
    ET_CHECK_OR_RETURN_ERROR(
        first.has_value() && last - *first + 1 == patch_count,
        InvalidArgument,
        "DFlash image span must be one contiguous run of patch tokens");
    return Error::Ok;
  }

  Error validate_sampling(const SamplingConfig& sampling) const {
    if (!valid_temperature(sampling.temperature) ||
        !valid_top_p(sampling.top_p) || sampling.top_k < 0) {
      ET_LOG(Error, "Invalid DFlash sampling configuration");
      return Error::InvalidArgument;
    }
    return Error::Ok;
  }

  void seed_rng(const SamplingConfig& sampling) {
    if (sampling.seed == 0) {
      std::random_device random;
      std::seed_seq seed{random(), random()};
      rng_.seed(seed);
      return;
    }
    std::seed_seq seed{
        static_cast<uint32_t>(sampling.seed),
        static_cast<uint32_t>(sampling.seed >> 32)};
    rng_.seed(seed);
  }

  int64_t max_context_len() const {
    const auto it = config_.metadata.find(kMaxContextLen);
    return it == config_.metadata.end() ? 0 : it->second;
  }

  bool is_eos(uint64_t token) const {
    return !config_.ignore_eos &&
        config_.eos_ids.find(token) != config_.eos_ids.end();
  }

  Result<std::string> decode_piece(uint64_t token) {
    const uint64_t previous = prev_decode_token_.value_or(token);
    auto decoded = config_.tokenizer->decode(previous, token);
    if (!decoded.ok()) {
      ET_LOG(Error, "DFlash tokenizer decode failed");
      return Error::InvalidArgument;
    }
    return std::move(*decoded);
  }

  Result<DecodeResult> terminal_result(uint64_t token) {
    auto decoded = decode_piece(token);
    ET_CHECK_OK_OR_RETURN_ERROR(decoded.error());
    return DecodeResult{
        token,
        std::move(decoded.get()),
        /*is_eos=*/true,
        /*is_terminal=*/true};
  }

  Error discard_unreturned_lookahead() {
    ET_CHECK_OR_RETURN_ERROR(
        logical_pos_ >= cached_ctx_len_,
        InvalidState,
        "DFlash logical position precedes draft cache cursor");
    const int64_t retained_rows = logical_pos_ - cached_ctx_len_;
    ET_CHECK_OR_RETURN_ERROR(
        retained_rows <= hidden_rows(),
        InvalidState,
        "DFlash hidden backlog does not cover logical position");
    if (hidden_dim_ > 0) {
      hidden_.resize(retained_rows * hidden_dim_);
    }
    target_pos_ = logical_pos_;
    ready_tokens_.clear();
    pending_token_.reset();
    return Error::Ok;
  }

  int64_t hidden_rows() const {
    return hidden_dim_ == 0
        ? 0
        : static_cast<int64_t>(hidden_.size()) / hidden_dim_;
  }

  void append_hidden(const std::vector<uint16_t>& values) {
    hidden_.insert(hidden_.end(), values.begin(), values.end());
    trim_hidden_window();
  }

  void trim_hidden_window() {
    if (config_.draft_sliding_window <= 0 || hidden_dim_ == 0 ||
        hidden_rows() <= config_.draft_sliding_window) {
      return;
    }
    const int64_t drop = hidden_rows() - config_.draft_sliding_window;
    hidden_.erase(hidden_.begin(), hidden_.begin() + drop * hidden_dim_);
    cached_ctx_len_ += drop;
  }

  Error prime_draft_cache() {
#ifdef EXECUTORCH_BUILD_CUDA
    if (config_.has_draft_prefill) {
      while (hidden_rows() - kCudaDFlashHiddenRows >=
             config_.min_draft_prefill_chunk) {
        const int64_t rows_to_feed = std::min(
            hidden_rows() - kCudaDFlashHiddenRows,
            config_.max_draft_prefill_chunk);
        ET_CHECK_OK_OR_RETURN_ERROR(run_draft_prefill(rows_to_feed));
        hidden_.erase(
            hidden_.begin(), hidden_.begin() + rows_to_feed * hidden_dim_);
        cached_ctx_len_ += rows_to_feed;
      }
    }
    constexpr int64_t prime_chunk = kCudaDFlashHiddenRows;
#else
    const int64_t prime_chunk = config_.max_prefill_chunk;
#endif
    ET_CHECK_OR_RETURN_ERROR(
        prime_chunk > 0, InvalidProgram, "Invalid DFlash prime chunk");
    while (hidden_rows() > prime_chunk) {
      ET_CHECK_OK_OR_RETURN_ERROR(
          run_draft(/*discard_output=*/true, prime_chunk).error());
      const int64_t consumed = std::min(hidden_rows(), prime_chunk);
      hidden_.erase(hidden_.begin(), hidden_.begin() + consumed * hidden_dim_);
      cached_ctx_len_ += consumed;
    }
    return Error::Ok;
  }

  Error run_draft_prefill(int64_t rows_to_feed) {
    ET_CHECK_OR_RETURN_ERROR(
        pending_token_.has_value() && hidden_dim_ > 0 &&
            rows_to_feed >= config_.min_draft_prefill_chunk &&
            rows_to_feed <= config_.max_draft_prefill_chunk &&
            rows_to_feed <= hidden_rows(),
        InvalidState,
        "Invalid DFlash prefill hidden rows");
    std::vector<int64_t> draft_tokens(block_length_, config_.mask_token_id);
    draft_tokens[0] = static_cast<int64_t>(*pending_token_);
    std::vector<int64_t> draft_position = {cached_ctx_len_};
    auto token_tensor = from_blob(
        draft_tokens.data(),
        {1, static_cast<SizesType>(block_length_)},
        executorch::aten::ScalarType::Long);
    auto hidden_tensor = from_blob(
        hidden_.data(),
        {1,
         static_cast<SizesType>(rows_to_feed),
         static_cast<SizesType>(hidden_dim_)},
        hidden_dtype_);
    auto position_tensor = from_blob(
        draft_position.data(), {1}, executorch::aten::ScalarType::Long);
    std::vector<EValue> inputs = {
        EValue(token_tensor), EValue(hidden_tensor), EValue(position_tensor)};
    std::lock_guard<std::mutex> guard(*config_.exec_mutex);
    auto outputs = execute_active(kDraftPrefill, inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());
    ET_CHECK_OR_RETURN_ERROR(
        !outputs->empty() && (*outputs)[0].isTensor(),
        InvalidProgram,
        "draft_prefill must return a tensor");
    return Error::Ok;
  }

  Error run_decode_cycle() {
    ET_CHECK_OR_RETURN_ERROR(
        pending_token_.has_value(),
        InvalidState,
        "DFlash cycle requires a pending token");
    const int64_t room =
        max_context_len() > 0 ? max_context_len() - target_pos_ : block_length_;
    ET_CHECK_OR_RETURN_ERROR(
        room > 0, InvalidArgument, "DFlash context capacity exhausted");
#ifdef EXECUTORCH_BUILD_CUDA
    if (room < n_draft_ + 1) {
      return run_target_only_cycle();
    }
#else
    if (room == 1) {
      return run_target_only_cycle();
    }
#endif

    DFlashCycleTiming cycle_timing;
    DFlashCycleTiming* const timing =
        config_.timing == nullptr ? nullptr : &cycle_timing;
    const auto cycle_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();

    auto draft_logits =
        run_draft(/*discard_output=*/false, hidden_rows(), timing);
    ET_CHECK_OK_OR_RETURN_ERROR(draft_logits.error());
    const int64_t verify_len = std::min<int64_t>(n_draft_ + 1, room);
    std::vector<uint64_t> candidates(verify_len);
    candidates[0] = *pending_token_;
    if (draft_probs_.size() < verify_len) {
      draft_probs_.resize(verify_len);
    }
    const bool stochastic = effective_temperature() > 0.0;
    const bool draft_argmax = stochastic && config_.draft_argmax;
    const auto draft_sampling_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    if (draft_argmax) {
      const auto& logits = *draft_logits.get();
      const int64_t vocab_size = logits.size(logits.dim() - 1);
      const float* logits_data =
          static_cast<const float*>(logits.const_data_ptr());
      for (int64_t i = 1; i < verify_len; ++i) {
        candidates[i] = muse_glimmer::argmax_index(
            logits_data + i * vocab_size, vocab_size);
      }
    } else if (stochastic && probability_workers_ != nullptr) {
      probability_workers_->fill(
          *draft_logits.get(),
          verify_len - 1,
          /*logits_row_offset=*/1,
          /*probabilities_row_offset=*/1,
          effective_temperature(),
          sampling_.top_k,
          sampling_.top_p,
          draft_probs_);
      const auto& logits = *draft_logits.get();
      const int64_t vocab_size = logits.size(logits.dim() - 1);
      for (int64_t i = 1; i < verify_len; ++i) {
        candidates[i] = muse_glimmer::categorical_sample(
            rng_, draft_probs_[i].data(), vocab_size);
      }
    } else {
      for (int64_t i = 1; i < verify_len; ++i) {
        candidates[i] = muse_glimmer::sample_token(
            rng_,
            *draft_logits.get(),
            i,
            effective_temperature(),
            sampling_.top_k,
            sampling_.top_p,
            stochastic ? &draft_probs_[i] : nullptr,
            /*probs_only=*/false,
            &sampling_workspace_);
      }
    }
    if (timing != nullptr) {
      timing->draft_sampling_ms +=
          elapsed_ms(draft_sampling_start, Clock::now());
    }

    auto verify = run_target(
        candidates.data(),
        verify_len,
        target_pos_,
        sampling_,
        /*sample_last=*/false,
        timing);
    ET_CHECK_OK_OR_RETURN_ERROR(verify.error());
    const auto& verify_logits = *last_logits_tensor_;

    int64_t committed = verify_len;
    uint64_t correction = 0;
    bool rejected = false;
    const auto accept_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    if (stochastic) {
      if (target_probs_.size() < verify_len) {
        target_probs_.resize(verify_len);
      }
      if (probability_workers_ != nullptr) {
        probability_workers_->fill(
            verify_logits,
            verify_len,
            /*logits_row_offset=*/0,
            /*probabilities_row_offset=*/0,
            effective_temperature(),
            sampling_.top_k,
            sampling_.top_p,
            target_probs_);
      }
    }
    for (int64_t i = 0; i < verify_len - 1; ++i) {
      bool accepted = false;
      if (!stochastic) {
        const uint64_t target_pick = muse_glimmer::sample_token(
            rng_, verify_logits, i, 0.0, sampling_.top_k, sampling_.top_p);
        accepted = target_pick == candidates[i + 1];
        if (!accepted) {
          committed = i + 1;
          correction = target_pick;
        }
      } else {
        if (probability_workers_ == nullptr) {
          muse_glimmer::sample_token(
              rng_,
              verify_logits,
              i,
              effective_temperature(),
              sampling_.top_k,
              sampling_.top_p,
              &target_probs_[i],
              /*probs_only=*/true,
              &sampling_workspace_);
        }
        const uint64_t token = candidates[i + 1];
        const float p = target_probs_[i][token];
        const float q = draft_argmax ? 1.0f : draft_probs_[i + 1][token];
        const float accept_probability =
            q > 0.0f ? std::min(1.0f, p / q) : 1.0f;
        accepted =
            muse_glimmer::accept_with_probability(rng_, accept_probability);
        if (!accepted) {
          committed = i + 1;
          correction = draft_argmax
              ? muse_glimmer::sample_excluding_token_in_place(
                    rng_, target_probs_[i], token)
              : muse_glimmer::sample_from_residual_in_place(
                    rng_, target_probs_[i], draft_probs_[i + 1]);
        }
      }
      if (timing != nullptr) {
        if (timing->draft_attempts_by_row.size() <= static_cast<size_t>(i)) {
          timing->draft_attempts_by_row.resize(i + 1);
          timing->draft_accepts_by_row.resize(i + 1);
        }
        ++timing->draft_attempts_by_row[i];
        if (accepted) {
          ++timing->draft_accepts_by_row[i];
        }
      }
      if (!accepted) {
        rejected = true;
        break;
      }
    }
    if (!rejected) {
      if (stochastic && probability_workers_ != nullptr) {
        const int64_t vocab_size = verify_logits.size(verify_logits.dim() - 1);
        correction = muse_glimmer::categorical_sample(
            rng_, target_probs_[verify_len - 1].data(), vocab_size);
      } else {
        correction = muse_glimmer::sample_token(
            rng_,
            verify_logits,
            verify_len - 1,
            effective_temperature(),
            sampling_.top_k,
            sampling_.top_p,
            /*out_probs=*/nullptr,
            /*probs_only=*/false,
            &sampling_workspace_);
      }
    }
    if (timing != nullptr) {
      timing->accept_correction_ms += elapsed_ms(accept_start, Clock::now());
    }

    const auto commit_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    const int64_t old_hidden_rows = hidden_rows();
    cached_ctx_len_ += old_hidden_rows;
    hidden_ = std::move(verify->hidden);
    hidden_.resize(committed * hidden_dim_);

    int64_t nonterminal_committed = committed;
    std::optional<uint64_t> terminal;
    for (int64_t i = 0; i < committed; ++i) {
      if (is_eos(candidates[i])) {
        nonterminal_committed = i;
        terminal = candidates[i];
        break;
      }
      ready_tokens_.push_back(candidates[i]);
    }
    if (nonterminal_committed != committed) {
      hidden_.resize(nonterminal_committed * hidden_dim_);
      correction = *terminal;
    }
    target_pos_ += nonterminal_committed;
    pending_token_ = correction;
    if (timing != nullptr) {
      timing->state_commit_ms += elapsed_ms(commit_start, Clock::now());
      timing->total_cycle_ms += elapsed_ms(cycle_start, Clock::now());
      config_.timing->record(*timing);
    }
    return Error::Ok;
  }

  Error run_target_only_cycle() {
    DFlashCycleTiming cycle_timing;
    DFlashCycleTiming* const timing =
        config_.timing == nullptr ? nullptr : &cycle_timing;
    const auto cycle_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    const uint64_t anchor = *pending_token_;
    auto result = run_target(
        &anchor,
        1,
        target_pos_,
        sampling_,
        /*sample_last=*/true,
        timing);
    ET_CHECK_OK_OR_RETURN_ERROR(result.error());
    const auto commit_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    append_hidden(result->hidden);
    ready_tokens_.push_back(anchor);
    pending_token_ = result->sampled_token;
    ++target_pos_;
    if (timing != nullptr) {
      timing->state_commit_ms += elapsed_ms(commit_start, Clock::now());
      timing->total_cycle_ms += elapsed_ms(cycle_start, Clock::now());
      timing->target_only = true;
      config_.timing->record(*timing);
    }
    return Error::Ok;
  }

  double effective_temperature() const {
    return sampling_.temperature < 0.0f ? 0.0 : sampling_.temperature;
  }

  Result<TargetResult> run_target(
      const uint64_t* tokens,
      int64_t count,
      int64_t start_pos,
      const SamplingConfig& sampling,
      bool sample_last,
      DFlashCycleTiming* timing = nullptr,
      bool use_target_prefill = false,
      int64_t retain_from_row = 0) {
    return run_target_from_embeddings(
        tokens,
        count,
        start_pos,
        sampling,
        sample_last,
        /*image=*/nullptr,
        /*next_image_row=*/nullptr,
        timing,
        use_target_prefill,
        retain_from_row);
  }

  Result<TargetResult> run_target_from_embeddings(
      const uint64_t* tokens,
      int64_t count,
      int64_t start_pos,
      const SamplingConfig& sampling,
      bool sample_last,
      const PreparedMuseGlimmerImage* image,
      int64_t* next_image_row,
      DFlashCycleTiming* timing = nullptr,
      bool use_target_prefill = false,
      int64_t retain_from_row = 0) {
    std::vector<int64_t> token_data(tokens, tokens + count);
    std::vector<int64_t> positions(count);
    std::iota(positions.begin(), positions.end(), start_pos);
    auto token_tensor = from_blob(
        token_data.data(),
        {1, static_cast<SizesType>(count)},
        executorch::aten::ScalarType::Long);
    auto position_tensor = from_blob(
        positions.data(),
        {static_cast<SizesType>(count)},
        executorch::aten::ScalarType::Long);

    const auto execute_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    std::lock_guard<std::mutex> guard(*config_.exec_mutex);
    auto embed_outputs = execute_active(kEmbedText, {EValue(token_tensor)});
    ET_CHECK_OK_OR_RETURN_ERROR(embed_outputs.error());
    ET_CHECK_OR_RETURN_ERROR(
        embed_outputs->size() == 1 && (*embed_outputs)[0].isTensor(),
        InvalidProgram,
        "embed_text must return one tensor");
    const auto& embed_tensor = (*embed_outputs)[0].toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        embed_tensor.dim() == 3 && embed_tensor.size(0) == 1 &&
            embed_tensor.size(1) == count &&
            embed_tensor.scalar_type() == hidden_dtype_,
        InvalidProgram,
        "embed_text must return [1, T, H] in the activation dtype");
    const int64_t embed_dim = embed_tensor.size(2);
    ET_CHECK_OR_RETURN_ERROR(
        embed_dim > 0 && (image == nullptr || embed_dim == image->hidden_dim),
        InvalidArgument,
        "embed_text hidden dimension does not match the staged image");

    std::vector<uint16_t> embeddings;
    TensorPtr embeddings_tensor;
    std::vector<EValue> target_inputs;
    if (image != nullptr) {
      ET_CHECK_OR_RETURN_ERROR(
          next_image_row != nullptr,
          InvalidState,
          "DFlash image splice requires an image row cursor");
      embeddings.resize(count * embed_dim);
      ET_CHECK_OK_OR_RETURN_ERROR(copy_bytes_to_host(
          embeddings.data(),
          embed_tensor.const_data_ptr(),
          embeddings.size() * sizeof(uint16_t)));
      for (int64_t row = 0; row < count; ++row) {
        if (tokens[row] != kMuseGlimmerImagePatchTokenId) {
          continue;
        }
        ET_CHECK_OR_RETURN_ERROR(
            *next_image_row < image->num_soft_tokens,
            InvalidArgument,
            "DFlash prefill has more patch tokens than image rows");
        std::memcpy(
            embeddings.data() + row * embed_dim,
            image->embeddings.data() + *next_image_row * embed_dim,
            embed_dim * sizeof(uint16_t));
        ++*next_image_row;
      }
      embeddings_tensor = from_blob(
          embeddings.data(),
          {1, static_cast<SizesType>(count), static_cast<SizesType>(embed_dim)},
          hidden_dtype_);
      target_inputs = {EValue(embeddings_tensor), EValue(position_tensor)};
    } else {
      target_inputs = {(*embed_outputs)[0], EValue(position_tensor)};
    }

    auto outputs = execute_active(
        use_target_prefill ? kTargetPrefillFromEmbeddings
                           : kTargetForwardFromEmbeddings,
        target_inputs);
    if (timing != nullptr) {
      timing->target_execute_ms += elapsed_ms(execute_start, Clock::now());
    }
    ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());
    return process_target_outputs(
        *outputs,
        count,
        sampling,
        sample_last,
        timing,
        use_target_prefill,
        retain_from_row);
  }

  Result<TargetResult> process_target_outputs(
      const std::vector<EValue>& outputs,
      int64_t count,
      const SamplingConfig& sampling,
      bool sample_last,
      DFlashCycleTiming* timing,
      bool use_target_prefill,
      int64_t retain_from_row) {
    ET_CHECK_OR_RETURN_ERROR(
        outputs.size() >= 2 && outputs[0].isTensor() && outputs[1].isTensor(),
        InvalidProgram,
        "DFlash target must return logits and hidden tensors");

    const auto logits_copy_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    auto logits =
        copy_float_tensor_to_host(outputs[0].toTensor(), last_logits_storage_);
    ET_CHECK_OK_OR_RETURN_ERROR(logits.error());
    last_logits_tensor_ = std::move(logits.get());
    if (timing != nullptr) {
      timing->target_logits_copy_ms +=
          elapsed_ms(logits_copy_start, Clock::now());
    }
    uint64_t sampled = 0;
    if (sample_last) {
      const auto sampling_start =
          timing == nullptr ? Clock::time_point{} : Clock::now();
      sampled = muse_glimmer::sample_token(
          rng_,
          *last_logits_tensor_,
          use_target_prefill ? 0 : count - 1,
          sampling.temperature < 0.0f ? 0.0 : sampling.temperature,
          sampling.top_k,
          sampling.top_p,
          /*out_probs=*/nullptr,
          /*probs_only=*/false,
          &sampling_workspace_);
      if (timing != nullptr) {
        timing->accept_correction_ms +=
            elapsed_ms(sampling_start, Clock::now());
      }
    }

    const auto& hidden_tensor = outputs[1].toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        hidden_tensor.dim() == 3 && hidden_tensor.size(0) == 1 &&
            hidden_tensor.size(1) == count &&
            hidden_tensor.scalar_type() == hidden_dtype_,
        InvalidProgram,
        "DFlash target hidden must be [1, T, H] in the activation dtype");
    if (hidden_dim_ == 0) {
      hidden_dim_ = hidden_tensor.size(2);
    }
    ET_CHECK_OR_RETURN_ERROR(
        hidden_tensor.size(2) == hidden_dim_,
        InvalidProgram,
        "DFlash target hidden dimension changed");
    ET_CHECK_OR_RETURN_ERROR(
        retain_from_row >= 0 && retain_from_row <= count,
        InvalidArgument,
        "Invalid retained target hidden row offset");
    const int64_t retained_rows = count - retain_from_row;
    std::vector<uint16_t> hidden(retained_rows * hidden_dim_);
    const auto hidden_copy_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    ET_CHECK_OK_OR_RETURN_ERROR(copy_bytes_to_host(
        hidden.data(),
        static_cast<const uint16_t*>(hidden_tensor.const_data_ptr()) +
            retain_from_row * hidden_dim_,
        hidden.size() * sizeof(uint16_t)));
    if (timing != nullptr) {
      timing->target_hidden_copy_ms +=
          elapsed_ms(hidden_copy_start, Clock::now());
    }
    return TargetResult{sampled, std::move(hidden)};
  }

  Result<TensorPtr> copy_float_tensor_to_host(
      const executorch::aten::Tensor& source,
      std::vector<float>& storage) {
    ET_CHECK_OR_RETURN_ERROR(
        source.scalar_type() == executorch::aten::ScalarType::Float,
        InvalidProgram,
        "DFlash logits must be float32");
    storage.resize(source.numel());
    ET_CHECK_OK_OR_RETURN_ERROR(copy_bytes_to_host(
        storage.data(),
        source.const_data_ptr(),
        storage.size() * sizeof(float)));
    std::vector<SizesType> sizes(source.dim());
    for (int32_t i = 0; i < source.dim(); ++i) {
      sizes[i] = source.size(i);
    }
    return from_blob(
        storage.data(), sizes, executorch::aten::ScalarType::Float);
  }

  Error
  copy_bytes_to_host(void* destination, const void* source, size_t bytes) {
#ifdef EXECUTORCH_BUILD_CUDA
    cudaPointerAttributes attributes{};
    const bool on_device =
        cudaPointerGetAttributes(&attributes, source) == cudaSuccess &&
        attributes.type == cudaMemoryTypeDevice;
    if (on_device) {
      return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost) ==
              cudaSuccess
          ? Error::Ok
          : Error::Internal;
    }
#endif
    std::memcpy(destination, source, bytes);
    return Error::Ok;
  }

  Result<TensorPtr> run_draft(
      bool discard_output,
      int64_t rows_to_feed,
      DFlashCycleTiming* timing = nullptr) {
    ET_CHECK_OR_RETURN_ERROR(
        pending_token_.has_value() && hidden_dim_ > 0 && hidden_rows() > 0 &&
            rows_to_feed > 0 && rows_to_feed <= hidden_rows(),
        InvalidState,
        "DFlash draft requires pending token and target hidden state");
    const int64_t draft_len = block_length_;
    std::vector<int64_t> draft_tokens(draft_len, config_.mask_token_id);
    draft_tokens[0] = static_cast<int64_t>(*pending_token_);
    std::vector<int64_t> draft_position = {cached_ctx_len_};

    const uint16_t* hidden_data = hidden_.data();
    int64_t hidden_input_rows = rows_to_feed;
#ifdef EXECUTORCH_BUILD_CUDA
    std::vector<uint16_t> padded_hidden(kCudaDFlashHiddenRows * hidden_dim_, 0);
    ET_CHECK_OR_RETURN_ERROR(
        hidden_input_rows <= kCudaDFlashHiddenRows,
        InvalidState,
        "CUDA DFlash hidden backlog exceeds %" PRId64 " rows",
        kCudaDFlashHiddenRows);
    std::memcpy(
        padded_hidden.data(),
        hidden_data,
        rows_to_feed * hidden_dim_ * sizeof(uint16_t));
    hidden_data = padded_hidden.data();
    hidden_input_rows = kCudaDFlashHiddenRows;
#endif
    auto token_tensor = from_blob(
        draft_tokens.data(),
        {1, static_cast<SizesType>(draft_len)},
        executorch::aten::ScalarType::Long);
    auto hidden_tensor = from_blob(
        const_cast<uint16_t*>(hidden_data),
        {1,
         static_cast<SizesType>(hidden_input_rows),
         static_cast<SizesType>(hidden_dim_)},
        hidden_dtype_);
    auto position_tensor = from_blob(
        draft_position.data(), {1}, executorch::aten::ScalarType::Long);
    std::vector<EValue> inputs = {
        EValue(token_tensor), EValue(hidden_tensor), EValue(position_tensor)};
#ifdef EXECUTORCH_BUILD_CUDA
    std::vector<int64_t> valid_length = {rows_to_feed};
    auto valid_length_tensor =
        from_blob(valid_length.data(), {1}, executorch::aten::ScalarType::Long);
    inputs.push_back(EValue(valid_length_tensor));
#endif
    const auto execute_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    std::lock_guard<std::mutex> guard(*config_.exec_mutex);
    auto outputs = execute_active(kDraftForward, inputs);
    if (timing != nullptr) {
      timing->draft_execute_ms += elapsed_ms(execute_start, Clock::now());
    }
    ET_CHECK_OK_OR_RETURN_ERROR(outputs.error());
    ET_CHECK_OR_RETURN_ERROR(
        !outputs->empty() && (*outputs)[0].isTensor(),
        InvalidProgram,
        "draft_forward must return logits");
    if (discard_output) {
      return TensorPtr{};
    }
    const auto logits_copy_start =
        timing == nullptr ? Clock::time_point{} : Clock::now();
    auto host_logits = copy_float_tensor_to_host(
        (*outputs)[0].toTensor(), draft_logits_storage_);
    ET_CHECK_OK_OR_RETURN_ERROR(host_logits.error());
    draft_logits_tensor_ = std::move(host_logits.get());
    if (timing != nullptr) {
      timing->draft_logits_copy_ms +=
          elapsed_ms(logits_copy_start, Clock::now());
    }
    return draft_logits_tensor_;
  }

  Result<std::vector<EValue>> execute_active(
      const char* method,
      const std::vector<EValue>& inputs) {
    return config_.mutable_state != nullptr
        ? config_.mutable_state->with_active_session(
              config_.session_token,
              [&]() { return config_.module->execute(method, inputs); })
        : config_.module->execute(method, inputs);
  }

  DFlashSessionConfig config_;
  int64_t block_length_;
  int64_t n_draft_;
  executorch::aten::ScalarType hidden_dtype_;
  std::mt19937 rng_;
  SamplingConfig sampling_;
  bool sampling_initialized_ = false;
  bool poisoned_ = false;
  bool preserve_staged_image_on_next_reset_ = false;
  std::optional<PreparedMuseGlimmerImage> staged_image_;
  std::atomic<bool> stop_{false};

  int64_t logical_pos_ = 0;
  int64_t target_pos_ = 0;
  int64_t cached_ctx_len_ = 0;
  int64_t hidden_dim_ = 0;
  std::vector<uint16_t> hidden_;
  std::deque<uint64_t> ready_tokens_;
  std::optional<uint64_t> pending_token_;
  std::optional<uint64_t> prev_decode_token_;

  std::vector<float> last_logits_storage_;
  TensorPtr last_logits_tensor_;
  std::vector<float> draft_logits_storage_;
  TensorPtr draft_logits_tensor_;
  std::vector<std::vector<float>> draft_probs_;
  std::vector<std::vector<float>> target_probs_;
  muse_glimmer::SamplingWorkspace sampling_workspace_;
  std::unique_ptr<ProbabilityWorkers> probability_workers_;
};

} // namespace

Result<std::unique_ptr<LLMSession>> create_dflash_session(
    DFlashSessionConfig config) {
  ET_CHECK_OR_RETURN_ERROR(
      config.module != nullptr && config.exec_mutex != nullptr &&
          config.live_sessions != nullptr && config.tokenizer != nullptr,
      InvalidArgument,
      "DFlash session requires module, mutex, live-session counter, and tokenizer");
  ET_CHECK_OR_RETURN_ERROR(
      config.block_size >= 2 && config.block_length >= 2 &&
          config.block_length <= config.block_size && config.n_draft > 0 &&
          config.n_draft < config.block_length,
      InvalidArgument,
      "Invalid DFlash block configuration");
  ET_CHECK_OR_RETURN_ERROR(
      config.activation_dtype == "bfloat16" ||
          config.activation_dtype == "float16",
      InvalidProgram,
      "DFlash hidden dtype must be bfloat16 or float16");
  return std::unique_ptr<LLMSession>(new DFlashSession(std::move(config)));
}

} // namespace executorch::extension::llm
