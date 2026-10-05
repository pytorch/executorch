/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/serving/detail/prompt_preparer.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <gtest/gtest.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

using namespace executorch::extension::llm;
using namespace executorch::extension::llm::serving;
using namespace std::chrono_literals;

namespace {

class Gate {
 public:
  void hold() {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = true;
    entered_ = false;
  }
  void enter() {
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [&] { return !held_; });
  }
  bool wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, 5s, [&] { return entered_; });
  }
  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = false;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool held_ = false;
  bool entered_ = false;
};

class Tokenizer final : public tokenizers::Tokenizer {
 public:
  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }
  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t bos, int8_t eos) const override {
    EXPECT_EQ(bos, 0);
    EXPECT_EQ(eos, 0);
    return std::vector<uint64_t>(text.begin(), text.end());
  }
  tokenizers::Result<std::string> decode(uint64_t, uint64_t, bool)
      const override {
    return std::string("A");
  }
  tokenizers::Result<std::string> id_to_piece(uint64_t) const override {
    return tokenizers::Error::Internal;
  }
  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override {
    return tokenizers::Error::Internal;
  }
};

// A deliberately private payload: serving can only retain and slice it.
class Payload final : public batching::PreparedInput {
 public:
  Payload(std::vector<float> rows, std::shared_ptr<std::atomic<int>> live)
      : rows(std::move(rows)), live_(std::move(live)) {
    ++*live_;
  }
  ~Payload() override {
    --*live_;
  }
  std::size_t position_count() const override {
    return rows.size();
  }
  std::size_t retained_bytes() const override {
    return sizeof(*this) + rows.capacity() * sizeof(float);
  }
  static const void* tag() {
    static const char tag = 0;
    return &tag;
  }
  const void* compatibility_tag() const override {
    return tag();
  }
  const std::vector<float> rows;

 private:
  std::shared_ptr<std::atomic<int>> live_;
};

batching::PreparationConfig preparation_limits() {
  batching::PreparationConfig config;
  config.max_positions = 128;
  config.max_retained_bytes = 4096;
  config.max_workspace_bytes = 65536;
  config.max_total_retained_bytes =
      4 * (config.max_workspace_bytes + config.max_retained_bytes);
  config.max_images = 1;
  return config;
}

class ImageExecutor final : public batching::Executor {
 public:
  explicit ImageExecutor(
      batching::PreparationConfig config = preparation_limits())
      : batching::Executor(config) {}
  Gate preparing;
  Gate executing;
  std::atomic<int> preparations{0};
  std::atomic<int> opens{0};
  std::atomic<int> clones{0};
  std::atomic<int> executions{0};
  std::atomic<bool> fail_prepare{false};
  std::atomic<bool> empty_prepare{false};
  std::shared_ptr<std::atomic<int>> live =
      std::make_shared<std::atomic<int>>(0);
  std::size_t image_positions = 4;
  std::vector<std::vector<float>> prepared_rows;
  std::vector<std::vector<float>> executed_rows;
  std::thread::id prepare_thread;

  bool prepare(
      const batching::PreparationInput& input,
      batching::PreparedInputPtr& out) override {
    ++preparations;
    prepare_thread = std::this_thread::get_id();
    preparing.enter();
    std::vector<float> rows;
    for (const auto& segment : input.segments) {
      if (auto* ids = segment.try_get_tokens()) {
        rows.insert(rows.end(), ids->begin(), ids->end());
      } else if (auto* image = segment.try_get_image()) {
        rows.insert(
            rows.end(),
            image_positions,
            1000.0f + image->get_uint8_data().front());
      } else {
        return false;
      }
    }
    prepared_rows.push_back(rows);
    if (empty_prepare.load()) {
      rows.clear();
    }
    out = std::make_shared<Payload>(std::move(rows), live);
    return !fail_prepare.load();
  }
  bool wrap_tokens(
      std::shared_ptr<const std::vector<batching::Token>> tokens,
      batching::PreparedInputPtr& out) override {
    out = std::make_shared<Payload>(
        std::vector<float>(tokens->begin(), tokens->end()), live);
    return true;
  }
  bool accepts(const batching::PreparedInput& input) const override {
    return input.compatibility_tag() == Payload::tag();
  }
  std::optional<batching::SessionId> open_session() override {
    const auto id = ++opens;
    positions_[id] = 0;
    return id;
  }
  void close_session(batching::SessionId id) override {
    positions_.erase(id);
  }
  std::optional<batching::SessionId> clone(
      batching::SessionId,
      batching::Position) override {
    ++clones;
    return std::nullopt;
  }
  void set_sampling(
      batching::SessionId,
      const batching::SamplingParams&,
      std::optional<uint64_t>) override {}
  bool execute(const batching::BatchInput& input, batching::BatchOutput& out)
      override {
    if (!validate_batch(input)) {
      return false;
    }
    ++executions;
    executing.enter();
    out.outputs.clear();
    for (const auto& item : input.inputs) {
      const auto& rows = static_cast<const Payload&>(*item.prepared).rows;
      const auto start = static_cast<int64_t>(item.position) + item.offset;
      EXPECT_EQ(start, positions_.at(item.sid));
      positions_[item.sid] = start + item.size;
      executed_rows.emplace_back(
          rows.begin() + item.offset, rows.begin() + item.offset + item.size);
      if (item.produce_output) {
        out.outputs.emplace_back(batching::Output{item.sid, {100}});
      } else {
        out.outputs.emplace_back(std::nullopt);
      }
    }
    return true;
  }

 private:
  std::map<batching::SessionId, int64_t> positions_;
};

struct Events {
  std::optional<TerminalEvent> terminal;
  std::string text;
  int terminals = 0;
  void accept(GenerationEvent event) {
    if (auto* value = std::get_if<TextEvent>(&event)) {
      text += value->text;
    } else {
      terminal = std::get<TerminalEvent>(event);
      ++terminals;
    }
  }
};

PromptInput tokens(std::vector<uint64_t> ids) {
  return {{make_token_input(std::move(ids))}};
}
EncodedImage png(uint8_t value = 1, uint32_t width = 1, uint32_t height = 1) {
  // Complete IHDR framing plus a synthetic marker for the injected fake codec.
  std::vector<uint8_t> data{
      137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 'I', 'H', 'D', 'R'};
  for (auto dimension : {width, height}) {
    for (int shift : {24, 16, 8, 0}) {
      data.push_back(static_cast<uint8_t>(dimension >> shift));
    }
  }
  data.insert(data.end(), {8, 0, 0, 0, 0, 0, 0, 0, 0, value});
  return {std::move(data), "image/png"};
}
MultimodalInput image(uint8_t value = 1) {
  return make_encoded_image_input(png(value));
}

class ImagePreparationTest : public ::testing::Test {
 protected:
  ImageExecutor executor;
  Tokenizer tokenizer;
  ServingRuntimeConfig config;
  std::unique_ptr<ServingRuntime> runtime;
  std::atomic<int> cpu_calls{0};
  std::atomic<int> cpu_allocations{0};
  std::thread::id cpu_thread;
  Gate terminal_gate;
  void SetUp() override {
    config.max_sessions = 4;
    config.max_context_length = 64;
    config.max_images = 1;
    config.max_image_preprocessed_bytes = 1;
    config.prefix_cache_capacity = 2;
    config.image_preprocessor =
        [this, bound = config.max_image_preprocessed_bytes](
            const EncodedImage& source) -> executorch::runtime::Result<Image> {
      ++cpu_calls;
      cpu_thread = std::this_thread::get_id();
      // Synthetic fixture only; this is not a real compressed-image decoder.
      if (source.data.size() != 34 || source.data.back() == 0 || bound < 1) {
        return executorch::runtime::Error::InvalidArgument;
      }
      ++cpu_allocations;
      return Image(std::vector<uint8_t>{source.data.back()}, 1, 1, 1);
    };
  }
  void start() {
    runtime = std::make_unique<ServingRuntime>(
        executor,
        batching::DecodeFirstScheduler::create(4, 2, 2),
        tokenizer,
        config);
  }
  void TearDown() override {
    executor.preparing.release();
    executor.executing.release();
    terminal_gate.release();
    runtime.reset();
    EXPECT_EQ(executor.live->load(), 0);
  }
  RequestHandle submit(
      Events& events,
      PromptInput prompt,
      std::optional<std::string> key = "s",
      int budget = 1) {
    GenerationOptions options;
    options.max_new_tokens = budget;
    auto result = runtime->generate(
        std::move(key),
        std::move(prompt),
        options,
        [&events](GenerationEvent event) { events.accept(std::move(event)); });
    if (auto* error = std::get_if<ServingError>(&result)) {
      ADD_FAILURE() << error->message;
      return {};
    }
    return std::get<RequestHandle>(result);
  }
};

TEST_F(ImagePreparationTest, OrderedPrivateImagePayloadAndImageOnlyPrompt) {
  start();
  EXPECT_EQ(runtime->info().max_images, 1u);
  Events mixed;
  submit(
      mixed,
      {{make_text_input("x"), image(7), make_token_input({9})}},
      "mixed",
      2)
      .wait();
  ASSERT_TRUE(mixed.terminal);
  EXPECT_EQ(mixed.text, "AA");
  EXPECT_EQ(mixed.terminal->stats.prompt_tokens, 2u);
  EXPECT_EQ(mixed.terminal->stats.prompt_positions, 6u);
  EXPECT_EQ(mixed.terminal->stats.prefilled_prompt_positions, 6u);
  EXPECT_EQ(mixed.terminal->stats.prefilled_prompt_tokens, 0u);
  EXPECT_EQ(executor.preparations.load(), 1);
  EXPECT_EQ(
      executor.prepared_rows.front(),
      (std::vector<float>{120, 1007, 1007, 1007, 1007, 9}));
  EXPECT_NE(cpu_thread, executor.prepare_thread);
  EXPECT_EQ(executor.clones.load(), 0);
  Events only;
  submit(only, {{image(8)}}, "only").wait();
  ASSERT_TRUE(only.terminal);
  EXPECT_EQ(only.terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(only.terminal->stats.prompt_tokens, 0u);
  EXPECT_EQ(only.terminal->stats.prompt_positions, 4u);
  EXPECT_EQ(only.text, "A");
  EXPECT_EQ(executor.preparations.load(), 2);
  EXPECT_EQ(executor.clones.load(), 0);
}

TEST_F(
    ImagePreparationTest,
    TextAlwaysPreparesFullPromptBeforeSuffixSelection) {
  config.prefix_cache_capacity = 0;
  start();
  Events first;
  submit(first, tokens({10, 11})).wait();
  Events next;
  submit(next, tokens({10, 11, 100, 12})).wait();
  EXPECT_EQ(executor.preparations.load(), 2);
  ASSERT_EQ(executor.prepared_rows.size(), 2u);
  EXPECT_EQ(
      executor.prepared_rows.back(), (std::vector<float>{10, 11, 100, 12}));
  ASSERT_TRUE(next.terminal);
  EXPECT_EQ(next.terminal->stats.reused_prompt_positions, 2u);
  EXPECT_EQ(next.terminal->stats.prefilled_prompt_positions, 2u);
  EXPECT_EQ(executor.opens.load(), 1);
}

TEST_F(ImagePreparationTest, CpuAndExecutorFailuresPreserveResidentHistory) {
  config.prefix_cache_capacity = 0;
  start();
  Events seed;
  submit(seed, tokens({10, 11})).wait();
  Events cpu_failed;
  submit(cpu_failed, {{image(0)}}).wait();
  EXPECT_EQ(executor.preparations.load(), 1);
  ASSERT_TRUE(cpu_failed.terminal->error);
  executor.fail_prepare = true;
  Events encoder_failed;
  submit(encoder_failed, {{image(1)}}).wait();
  EXPECT_EQ(executor.opens.load(), 1);
  EXPECT_EQ(executor.live->load(), 0);
  executor.fail_prepare = false;
  executor.empty_prepare = true;
  Events invalid;
  submit(invalid, {{image(1)}}).wait();
  EXPECT_EQ(executor.opens.load(), 1);
  executor.empty_prepare = false;
  Events continuation;
  submit(continuation, tokens({10, 11, 100, 12})).wait();
  EXPECT_EQ(continuation.terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(executor.opens.load(), 1);
}

TEST_F(
    ImagePreparationTest,
    ExpandedContextCheckedBeforeDestructiveReplacement) {
  config.max_context_length = 6;
  config.prefix_cache_capacity = 0;
  start();
  Events seed;
  submit(seed, tokens({10, 11})).wait();
  Events too_large;
  submit(too_large, {{make_token_input({1, 2}), image()}}).wait();
  ASSERT_TRUE(too_large.terminal->error);
  EXPECT_EQ(too_large.terminal->error->code, ErrorCode::InvalidArgument);
  EXPECT_EQ(executor.opens.load(), 1);
  Events continuation;
  submit(continuation, tokens({10, 11, 100, 12})).wait();
  EXPECT_EQ(continuation.terminal->stats.session_reset_reason, "exact_prefix");
}

TEST_F(ImagePreparationTest, EqualPatchCountsNeverIdentifyImageHistory) {
  start();
  Events first;
  submit(first, {{make_token_input({10}), image(1)}}).wait();
  Events changed;
  submit(
      changed,
      {{make_token_input({10}), image(2), make_token_input({100, 12})}})
      .wait();
  EXPECT_EQ(changed.terminal->stats.session_reset_reason, "image_history");
  EXPECT_EQ(changed.terminal->stats.reused_prompt_positions, 0u);
  EXPECT_EQ(executor.opens.load(), 2);
  EXPECT_EQ(executor.clones.load(), 0);
  Events text;
  submit(text, tokens({10, 100, 12, 100, 13})).wait();
  EXPECT_EQ(text.terminal->stats.session_reset_reason, "image_history");
  EXPECT_EQ(text.terminal->stats.reused_prompt_positions, 0u);
  EXPECT_EQ(executor.opens.load(), 3);
  EXPECT_EQ(executor.clones.load(), 0);
}

TEST_F(ImagePreparationTest, LimitsRejectBeforeCpuOrModelPreparation) {
  config.max_image_encoded_bytes = 34;
  auto oversized = png();
  oversized.data.push_back(0);
  start();
  for (auto prompt : std::vector<PromptInput>{
           {{image(), image()}},
           {{make_encoded_image_input(std::move(oversized))}},
           {{make_encoded_image_input(EncodedImage{{1}, "image/gif"})}},
           {{make_image_input(Image{})}}}) {
    Events failed;
    submit(failed, std::move(prompt)).wait();
    ASSERT_TRUE(failed.terminal->error);
    EXPECT_EQ(failed.terminal->error->code, ErrorCode::InvalidArgument);
  }
  EXPECT_EQ(executor.preparations.load(), 0);
  EXPECT_EQ(executor.opens.load(), 0);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
}

TEST_F(ImagePreparationTest, SourceHeaderBoundsRunBeforeCpuAllocation) {
  start();
  std::vector<EncodedImage> rejected{
      png(1, 4097, 1),
      png(1, 1, 4097),
      png(1, 4096, 4096),
      png(1, 0, 1),
      png(1, UINT32_MAX, UINT32_MAX)};
  auto wrong_mime = png();
  wrong_mime.mime_type = "image/jpeg";
  rejected.push_back(std::move(wrong_mime));
  for (std::size_t length = 0; length < 33; ++length) {
    auto truncated = png();
    truncated.data.resize(length);
    rejected.push_back(std::move(truncated));
  }
  auto bad_chunk = png();
  bad_chunk.data[11] = 12;
  rejected.push_back(std::move(bad_chunk));
  for (auto& source : rejected) {
    Events event;
    submit(event, {{make_encoded_image_input(std::move(source))}}).wait();
    ASSERT_TRUE(event.terminal->error);
    EXPECT_EQ(event.terminal->error->code, ErrorCode::InvalidArgument);
  }
  EXPECT_EQ(cpu_calls.load(), 0);
  EXPECT_EQ(executor.preparations.load(), 0);
  EXPECT_EQ(executor.opens.load(), 0);
}

TEST(ImageHeaderTest, JpegMarkerTraversalChecksBoundsAndTruncation) {
  using executorch::extension::llm::serving::detail::bounded_image_header;
  const EncodedImage jpeg{
      {0xff, 0xd8, 0xff, 0xe0, 0, 4, 0, 0,    0xff, 0xc0, 0,   11,
       8,    0,    2,    0,    3, 1, 1, 0x11, 0,    0xff, 0xd9},
      "image/jpeg"};
  EXPECT_TRUE(bounded_image_header(jpeg, 3, 6));
  EXPECT_FALSE(bounded_image_header(jpeg, 2, 6));
  EXPECT_FALSE(bounded_image_header(jpeg, 3, 5));
  for (std::size_t length = 0; length < 21; ++length) {
    auto truncated = jpeg;
    truncated.data.resize(length);
    EXPECT_FALSE(bounded_image_header(truncated, 4096, 4194304)) << length;
  }
  for (const auto marker :
       {0xc0,
        0xc1,
        0xc2,
        0xc3,
        0xc5,
        0xc6,
        0xc7,
        0xc9,
        0xca,
        0xcb,
        0xcd,
        0xce,
        0xcf}) {
    auto frame = jpeg;
    frame.data[9] = marker;
    EXPECT_TRUE(bounded_image_header(frame, 3, 6));
  }
  for (const auto length : {0, 1, 255}) {
    auto invalid = jpeg;
    invalid.data[5] = length;
    EXPECT_FALSE(bounded_image_header(invalid, 4096, 4194304));
  }
  auto scan_before_frame = jpeg;
  scan_before_frame.data[3] = 0xda;
  EXPECT_FALSE(bounded_image_header(scan_before_frame, 4096, 4194304));
  auto zero_height = jpeg;
  zero_height.data[14] = 0;
  EXPECT_FALSE(bounded_image_header(zero_height, 4096, 4194304));
  auto wrong_magic = jpeg;
  wrong_magic.data[0] = 0;
  EXPECT_FALSE(bounded_image_header(wrong_magic, 4096, 4194304));
}

TEST_F(ImagePreparationTest, ImageAtPromptEndClearsTextPredecessor) {
  auto prepared = executorch::extension::llm::serving::detail::prepare_prompt(
      tokenizer,
      {{make_token_input({0}), image(), make_text_input("")}},
      64,
      &config,
      &executor.preparation_config());
  ASSERT_TRUE(prepared.ok());
  EXPECT_EQ(prepared->tokens, (std::vector<uint64_t>{0}));
  EXPECT_FALSE(prepared->previous_token);
  auto with_text = executorch::extension::llm::serving::detail::prepare_prompt(
      tokenizer,
      {{image(), make_token_input({7})}},
      64,
      &config,
      &executor.preparation_config());
  ASSERT_TRUE(with_text.ok());
  EXPECT_EQ(with_text->previous_token, 7u);
}

TEST_F(ImagePreparationTest, MalformedHookOutputNeverReachesExecutor) {
  config.image_preprocessor =
      [bound = config.max_image_preprocessed_bytes](
          const EncodedImage& source) -> executorch::runtime::Result<Image> {
    if (bound < 1) {
      return executorch::runtime::Error::InvalidArgument;
    }
    switch (source.data.back()) {
      case 1:
        return Image(std::vector<uint8_t>{1}, 4097, 1, 1);
      case 2:
        return Image(std::vector<uint8_t>{1}, 4096, 4096, 1);
      case 3:
        return Image(std::vector<uint8_t>{1}, 1, 1, 5);
      default:
        return Image(std::vector<uint8_t>{1}, 2, 1, 1);
    }
  };
  start();
  for (uint8_t value : {1, 2, 3, 4}) {
    Events event;
    submit(event, {{image(value)}}).wait();
    ASSERT_TRUE(event.terminal->error);
  }
  EXPECT_EQ(executor.preparations.load(), 0);
  EXPECT_EQ(executor.opens.load(), 0);
}

TEST_F(ImagePreparationTest, CapabilityRequiresOptInAndCpuHook) {
  config.max_images = 0;
  start();
  EXPECT_EQ(runtime->info().max_images, 0u);
  Events disabled;
  submit(disabled, {{image()}}).wait();
  EXPECT_EQ(cpu_calls.load(), 0);
  EXPECT_EQ(executor.preparations.load(), 0);
  runtime.reset();
  config.max_images = 1;
  config.image_preprocessor = {};
  start();
  EXPECT_EQ(runtime->info().max_images, 0u);
  Events missing;
  submit(missing, {{image()}}).wait();
  EXPECT_EQ(executor.preparations.load(), 0);
}

TEST_F(ImagePreparationTest, UnfitImageBoundDisablesOnlyImagesBeforeCpu) {
  const auto workspace = executor.preparation_config().max_workspace_bytes;
  const auto overhead =
      sizeof(batching::PreparationInput) + sizeof(MultimodalInput);
  for (const auto bound :
       {std::size_t{0},
        workspace - overhead + 1,
        workspace,
        std::numeric_limits<std::size_t>::max()}) {
    SCOPED_TRACE(bound);
    config.max_image_preprocessed_bytes = bound;
    start();
    EXPECT_EQ(runtime->info().max_images, 0u);
    Events rejected;
    submit(rejected, {{image()}}).wait();
    ASSERT_TRUE(rejected.terminal);
    ASSERT_TRUE(rejected.terminal->error);
    EXPECT_EQ(rejected.terminal->error->code, ErrorCode::InvalidArgument);
    EXPECT_EQ(cpu_calls.load(), 0);
    EXPECT_EQ(cpu_allocations.load(), 0);
    EXPECT_EQ(runtime->info().active_sessions, 0u);
    Events text;
    submit(text, tokens({10, 11})).wait();
    ASSERT_TRUE(text.terminal);
    EXPECT_EQ(text.terminal->finish_reason, FinishReason::Length);
    EXPECT_TRUE(runtime->info().ready);
    runtime.reset();
  }
  config.max_image_preprocessed_bytes = workspace - overhead;
  start();
  EXPECT_EQ(runtime->info().max_images, 1u);
  Events boundary;
  submit(boundary, {{image()}}).wait();
  ASSERT_TRUE(boundary.terminal);
  EXPECT_EQ(boundary.terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(cpu_calls.load(), 1);
  EXPECT_EQ(cpu_allocations.load(), 1);
}

TEST_F(ImagePreparationTest, InvalidImageLimitsDoNotDisableText) {
  const auto valid = config;
  for (const auto setting :
       {&ServingRuntimeConfig::max_images,
        &ServingRuntimeConfig::max_image_encoded_bytes,
        &ServingRuntimeConfig::max_image_dimension,
        &ServingRuntimeConfig::max_image_pixels}) {
    config = valid;
    config.*setting = std::numeric_limits<std::size_t>::max();
    start();
    EXPECT_EQ(runtime->info().max_images, 0u);
    Events rejected;
    submit(rejected, {{image()}}).wait();
    ASSERT_TRUE(rejected.terminal);
    ASSERT_TRUE(rejected.terminal->error);
    EXPECT_EQ(rejected.terminal->error->code, ErrorCode::InvalidArgument);
    EXPECT_EQ(cpu_calls.load(), 0);
    EXPECT_EQ(cpu_allocations.load(), 0);
    Events text;
    submit(text, tokens({10, 11})).wait();
    ASSERT_TRUE(text.terminal);
    EXPECT_EQ(text.terminal->finish_reason, FinishReason::Length);
    runtime.reset();
  }
}

TEST_F(
    ImagePreparationTest,
    FullMixedPromptBudgetRejectsBeforeCpuAndPreservesHistory) {
  batching::PreparationInput staged;
  staged.segments.emplace_back(Image{});
  staged.segments.emplace_back(std::vector<uint64_t>{'x'});
  staged.segments.emplace_back(std::vector<uint64_t>{7, 8});
  const auto& limits = executor.preparation_config();
  const auto bytes = limits.input_retained_bytes(staged);
  ASSERT_TRUE(bytes);
  config.max_image_preprocessed_bytes = limits.max_workspace_bytes - *bytes + 1;
  config.prefix_cache_capacity = 0;
  start();
  EXPECT_EQ(runtime->info().max_images, 1u);
  Events seed;
  submit(seed, tokens({10, 11})).wait();
  Events rejected;
  submit(rejected, {{image(), make_text_input("x"), make_token_input({7, 8})}})
      .wait();
  ASSERT_TRUE(rejected.terminal);
  ASSERT_TRUE(rejected.terminal->error);
  EXPECT_EQ(rejected.terminal->error->code, ErrorCode::InvalidArgument);
  EXPECT_EQ(cpu_calls.load(), 0);
  EXPECT_EQ(cpu_allocations.load(), 0);
  EXPECT_EQ(executor.preparations.load(), 1);
  EXPECT_EQ(executor.opens.load(), 1);
  Events continued;
  submit(continued, tokens({10, 11, 100, 12})).wait();
  ASSERT_TRUE(continued.terminal);
  EXPECT_EQ(continued.terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(executor.opens.load(), 1);
  EXPECT_EQ(executor.live->load(), 0);
}

TEST_F(ImagePreparationTest, StagedCapacityPlusDeclaredBoundIsInclusive) {
  batching::PreparationInput staged;
  staged.segments.emplace_back(Image{});
  staged.segments.emplace_back(std::vector<uint64_t>{'x'});
  staged.segments.emplace_back(std::vector<uint64_t>{7, 8});
  auto limits = preparation_limits();
  const auto staged_bytes = limits.input_retained_bytes(staged);
  ASSERT_TRUE(staged_bytes);
  // The declaration, not the hook's smaller actual output, must fit.
  config.max_image_preprocessed_bytes = 32;
  const auto boundary = *staged_bytes + config.max_image_preprocessed_bytes;
  const PromptInput prompt{
      {image(), make_text_input("x"), make_token_input({7, 8})}};
  limits.max_workspace_bytes = boundary - 1;
  auto rejected = executorch::extension::llm::serving::detail::prepare_prompt(
      tokenizer, prompt, 64, &config, &limits);
  EXPECT_FALSE(rejected.ok());
  EXPECT_EQ(cpu_calls.load(), 0);
  EXPECT_EQ(cpu_allocations.load(), 0);
  for (const auto workspace : {boundary, boundary + 1}) {
    limits.max_workspace_bytes = workspace;
    auto prepared = executorch::extension::llm::serving::detail::prepare_prompt(
        tokenizer, prompt, 64, &config, &limits);
    ASSERT_TRUE(prepared.ok());
    EXPECT_EQ(prepared->tokens, (std::vector<uint64_t>{'x', 7, 8}));
    EXPECT_EQ(prepared->previous_token, 8u);
    ASSERT_EQ(prepared->input.segments.size(), 3u);
    EXPECT_EQ(prepared->input.segments.capacity(), staged.segments.capacity());
    EXPECT_EQ(
        prepared->input.segments.front().get_image().get_uint8_data(),
        (std::vector<uint8_t>{1}));
    EXPECT_EQ(limits.input_retained_bytes(prepared->input), *staged_bytes + 1);
  }
  EXPECT_EQ(cpu_calls.load(), 2);
  EXPECT_EQ(cpu_allocations.load(), 2);
}

TEST_F(ImagePreparationTest, InvalidSuffixNeverInvokesCpuHook) {
  start();
  for (auto prompt : std::vector<PromptInput>{
           {{image(), make_text_input(std::string(65, 'x'))}},
           {{image(), make_token_input(std::vector<uint64_t>(65, 7))}},
           {{image(), make_image_input(Image{})}}}) {
    Events rejected;
    submit(rejected, std::move(prompt)).wait();
    ASSERT_TRUE(rejected.terminal);
    ASSERT_TRUE(rejected.terminal->error);
    EXPECT_EQ(rejected.terminal->error->code, ErrorCode::InvalidArgument);
  }
  EXPECT_EQ(cpu_calls.load(), 0);
  EXPECT_EQ(cpu_allocations.load(), 0);
  EXPECT_EQ(executor.preparations.load(), 0);
  EXPECT_EQ(executor.opens.load(), 0);
}

TEST_F(ImagePreparationTest, FloatOutputCapacityUsesBytes) {
  config.max_image_preprocessed_bytes = sizeof(float);
  config.image_preprocessor =
      [this, bound = config.max_image_preprocessed_bytes](
          const EncodedImage& source) -> executorch::runtime::Result<Image> {
    ++cpu_calls;
    if (bound < sizeof(float)) {
      return executorch::runtime::Error::InvalidArgument;
    }
    ++cpu_allocations;
    std::vector<float> pixels(1, 1.0f);
    if (source.data.back() == 2) {
      // Deliberate contract violation, still within executor workspace.
      pixels.reserve(2);
    }
    return Image(std::move(pixels), 1, 1, 1);
  };
  const auto& limits = executor.preparation_config();
  auto prepared = executorch::extension::llm::serving::detail::prepare_prompt(
      tokenizer, {{image()}}, 64, &config, &limits);
  ASSERT_TRUE(prepared.ok());
  EXPECT_EQ(
      prepared->input.segments.front().get_image().get_float_data(),
      (std::vector<float>{1.0f}));
  auto rejected = executorch::extension::llm::serving::detail::prepare_prompt(
      tokenizer, {{image(2)}}, 64, &config, &limits);
  EXPECT_FALSE(rejected.ok());
  EXPECT_EQ(cpu_calls.load(), 2);
}

TEST_F(ImagePreparationTest, PreparingRequestHoldsClaimAndDoesNotBlockControl) {
  config.prefix_cache_capacity = 0;
  start();
  executor.preparing.hold();
  Events first;
  auto pending = submit(first, {{image()}});
  ASSERT_TRUE(executor.preparing.wait());
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_EQ(executor.opens.load(), 0);
  Events busy;
  auto overlap = submit(busy, tokens({1}));
  auto control = runtime->close_session_async("absent");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  overlap.wait();
  ASSERT_TRUE(busy.terminal->error);
  EXPECT_EQ(busy.terminal->error->code, ErrorCode::SessionBusy);
  pending.cancel();
  executor.preparing.release();
  pending.wait();
  EXPECT_EQ(first.terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(executor.opens.load(), 0);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
}

TEST_F(
    ImagePreparationTest,
    CloseFencesPreparationAndTerminalCallbackLifetime) {
  start();
  executor.preparing.hold();
  terminal_gate.hold();
  Events event;
  GenerationOptions options;
  options.max_new_tokens = 1;
  auto result =
      runtime->generate("s", {{image()}}, options, [&](GenerationEvent next) {
        const bool terminal = std::holds_alternative<TerminalEvent>(next);
        event.accept(std::move(next));
        if (terminal) {
          terminal_gate.enter();
        }
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  auto handle = std::get<RequestHandle>(result);
  ASSERT_TRUE(executor.preparing.wait());
  auto closed = runtime->close_session_async("s");
  auto control = runtime->close_session_async("absent");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_EQ(closed.wait_for(0s), std::future_status::timeout);
  executor.preparing.release();
  ASSERT_TRUE(terminal_gate.wait());
  EXPECT_EQ(closed.wait_for(0s), std::future_status::timeout);
  EXPECT_FALSE(handle.done());
  EXPECT_EQ(executor.opens.load(), 0);
  terminal_gate.release();
  ASSERT_EQ(closed.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(closed.get());
  EXPECT_TRUE(handle.done());
  EXPECT_EQ(event.terminals, 1);
}

TEST_F(
    ImagePreparationTest,
    ResetDuringPreparationDefersOpenWithoutBlockingPeers) {
  config.prefix_cache_capacity = 0;
  start();
  executor.preparing.hold();
  Events first;
  auto pending = submit(first, {{image()}});
  ASSERT_TRUE(executor.preparing.wait());
  auto reset = runtime->reset_session_async("s");
  auto control = runtime->close_session_async("absent");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_EQ(executor.opens.load(), 0);
  EXPECT_EQ(reset.wait_for(0s), std::future_status::timeout);
  executor.preparing.release();
  ASSERT_EQ(reset.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(reset.get());
  pending.wait();
  EXPECT_EQ(first.terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(executor.opens.load(), 1);
  Events next;
  submit(next, tokens({1, 2})).wait();
  EXPECT_EQ(next.terminal->stats.session_reset_reason, "new");
  EXPECT_EQ(executor.opens.load(), 1);
}

TEST_F(
    ImagePreparationTest,
    QueuedPreparationCancellationSkipsExecutorAndOpen) {
  config.prefix_cache_capacity = 0;
  start();
  executor.preparing.hold();
  Events first;
  auto pending = submit(first, {{image()}}, "first");
  ASSERT_TRUE(executor.preparing.wait());
  Events second;
  auto cancelled = submit(second, {{image()}}, "second");
  auto control = runtime->close_session_async("absent");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  cancelled.cancel();
  executor.preparing.release();
  pending.wait();
  cancelled.wait();
  EXPECT_EQ(executor.preparations.load(), 1);
  EXPECT_EQ(executor.opens.load(), 1);
  EXPECT_EQ(second.terminal->finish_reason, FinishReason::Cancelled);
}

TEST_F(
    ImagePreparationTest,
    CancellationAfterPreparationPublicationSkipsDecoderOpen) {
  start();
  executor.preparing.hold();
  terminal_gate.hold();
  Events event;
  auto handle = submit(event, {{image()}});
  ASSERT_TRUE(executor.preparing.wait());
  runtime->open_session_async("barrier", [&](LifecycleResult result) {
    EXPECT_FALSE(result);
    terminal_gate.enter();
  });
  // The second reservation proves control is awaiting the unrelated open.
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (runtime->info().active_sessions != 2 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_EQ(runtime->info().active_sessions, 2u);
  // Its callback holds consumption of the already-published preparation.
  executor.preparing.release();
  ASSERT_TRUE(terminal_gate.wait());
  handle.cancel();
  terminal_gate.release();
  handle.wait();
  EXPECT_EQ(event.terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(executor.opens.load(), 1);
  EXPECT_EQ(executor.executions.load(), 0);
  EXPECT_EQ(executor.live->load(), 0);
}

TEST_F(
    ImagePreparationTest,
    RawImageReservationRejectsPeersBeforeCpuAllocation) {
  const auto limits = preparation_limits();
  config.max_prepared_bytes =
      limits.max_workspace_bytes + limits.max_retained_bytes;
  config.prefix_cache_capacity = 0;
  config.max_image_preprocessed_bytes = 48 * 1024;
  config.image_preprocessor =
      [this, bound = config.max_image_preprocessed_bytes](
          const EncodedImage&) -> executorch::runtime::Result<Image> {
    ++cpu_calls;
    if (bound < 48 * 1024) {
      return executorch::runtime::Error::InvalidArgument;
    }
    ++cpu_allocations;
    return Image(std::vector<uint8_t>(48 * 1024, 1), 256, 192, 1);
  };
  start();
  executor.preparing.hold();
  Events first;
  auto pending = submit(first, {{image()}}, "first");
  ASSERT_TRUE(executor.preparing.wait());
  EXPECT_EQ(cpu_calls.load(), 1);
  Events second;
  auto denied = submit(second, {{image()}}, "second");
  denied.wait();
  ASSERT_TRUE(second.terminal->error);
  EXPECT_EQ(second.terminal->error->code, ErrorCode::CapacityExceeded);
  EXPECT_EQ(cpu_calls.load(), 1);
  EXPECT_EQ(cpu_allocations.load(), 1);
  EXPECT_EQ(executor.preparations.load(), 1);
  EXPECT_EQ(executor.opens.load(), 0);
  pending.cancel();
  executor.preparing.release();
  pending.wait();
  EXPECT_EQ(first.terminal->finish_reason, FinishReason::Cancelled);
  Events third;
  submit(third, {{image()}}, "second").wait();
  EXPECT_EQ(third.terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(cpu_calls.load(), 2);
  EXPECT_EQ(cpu_allocations.load(), 2);
}

TEST_F(
    ImagePreparationTest,
    ReturnedCapacityAboveDeclaredBoundNeverQueuesOrChangesHistory) {
  config.prefix_cache_capacity = 0;
  config.image_preprocessor =
      [this](const EncodedImage&) -> executorch::runtime::Result<Image> {
    ++cpu_calls;
    // Deliberately violate the trusted-hook contract: size fits B, capacity
    // does not. The result still fits workspace, so only B can reject it.
    ++cpu_allocations;
    std::vector<uint8_t> bytes{1};
    bytes.reserve(2);
    return Image(std::move(bytes), 1, 1, 1);
  };
  start();
  Events seed;
  submit(seed, tokens({10, 11})).wait();
  Events rejected;
  submit(rejected, {{image()}}).wait();
  ASSERT_TRUE(rejected.terminal->error);
  EXPECT_EQ(rejected.terminal->error->code, ErrorCode::InvalidArgument);
  EXPECT_EQ(executor.preparations.load(), 1);
  EXPECT_EQ(executor.opens.load(), 1);
  EXPECT_EQ(cpu_calls.load(), 1);
  EXPECT_EQ(cpu_allocations.load(), 1);
  Events continued;
  submit(continued, tokens({10, 11, 100, 12})).wait();
  EXPECT_EQ(continued.terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(executor.opens.load(), 1);
}

TEST(ServingPreparationBudgetTest, CheckedReservationCannotWrapBeforeCpu) {
  auto limits = preparation_limits();
  limits.max_retained_bytes = std::numeric_limits<std::size_t>::max() - 1024;
  limits.max_workspace_bytes = 2048;
  limits.max_total_retained_bytes = std::numeric_limits<std::size_t>::max();
  ImageExecutor executor(limits);
  Tokenizer tokenizer;
  ServingRuntimeConfig config;
  config.max_prepared_bytes = std::numeric_limits<std::size_t>::max();
  config.max_images = 1;
  config.max_image_preprocessed_bytes = 1;
  int cpu_calls = 0;
  config.image_preprocessor =
      [&, bound = config.max_image_preprocessed_bytes](
          const EncodedImage&) -> executorch::runtime::Result<Image> {
    ++cpu_calls;
    if (bound < 1) {
      return executorch::runtime::Error::InvalidArgument;
    }
    return Image(std::vector<uint8_t>{1}, 1, 1, 1);
  };
  ServingRuntime runtime(
      executor,
      batching::DecodeFirstScheduler::create(4, 2, 2),
      tokenizer,
      config);
  Events event;
  auto result =
      runtime.generate("s", {{image()}}, {}, [&](GenerationEvent value) {
        event.accept(std::move(value));
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  std::get<RequestHandle>(result).wait();
  ASSERT_TRUE(event.terminal->error);
  EXPECT_EQ(event.terminal->error->code, ErrorCode::CapacityExceeded);
  EXPECT_EQ(cpu_calls, 0);
  EXPECT_EQ(executor.preparations.load(), 0);
  EXPECT_EQ(runtime.info().active_sessions, 0u);
}

TEST_F(ImagePreparationTest, AggregateReservationIncludesRetiringTerminal) {
  const auto limits = preparation_limits();
  config.max_prepared_bytes =
      limits.max_workspace_bytes + limits.max_retained_bytes;
  config.prefix_cache_capacity = 0;
  start();
  Events seed;
  submit(seed, tokens({10, 11})).wait();
  terminal_gate.hold();
  Events first;
  GenerationOptions options;
  options.max_new_tokens = 1;
  auto result = runtime->generate(
      "image", {{image()}}, options, [&](GenerationEvent event) {
        const bool terminal = std::holds_alternative<TerminalEvent>(event);
        first.accept(std::move(event));
        if (terminal) {
          terminal_gate.enter();
        }
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  auto handle = std::get<RequestHandle>(result);
  ASSERT_TRUE(terminal_gate.wait());
  EXPECT_GT(executor.live->load(), 0);
  Events second;
  auto denied = submit(second, {{image()}}, "s");
  auto barrier = runtime->close_session_async("absent");
  ASSERT_EQ(barrier.wait_for(5s), std::future_status::ready);
  ASSERT_TRUE(denied.error());
  EXPECT_EQ(denied.error()->code, ErrorCode::CapacityExceeded);
  EXPECT_EQ(executor.preparations.load(), 2);
  EXPECT_EQ(cpu_calls.load(), 1);
  EXPECT_EQ(cpu_allocations.load(), 1);
  terminal_gate.release();
  handle.wait();
  denied.wait();
  EXPECT_EQ(executor.live->load(), 0);
  Events continued;
  submit(continued, tokens({10, 11, 100, 12})).wait();
  ASSERT_TRUE(continued.terminal);
  EXPECT_EQ(continued.terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(executor.opens.load(), 2);
  Events third;
  submit(third, {{image()}}, "other").wait();
  EXPECT_EQ(third.terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(cpu_calls.load(), 2);
  EXPECT_EQ(cpu_allocations.load(), 2);
}

TEST_F(ImagePreparationTest, CancelledSliceRetainsOpaqueBackingThroughExecute) {
  executor.image_positions = 6;
  start();
  executor.executing.hold();
  Events event;
  auto handle = submit(event, {{image()}});
  ASSERT_TRUE(executor.executing.wait());
  EXPECT_GT(executor.live->load(), 0);
  handle.cancel();
  EXPECT_FALSE(handle.done());
  EXPECT_GT(executor.live->load(), 0);
  executor.executing.release();
  handle.wait();
  EXPECT_EQ(event.terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(event.terminal->stats.prompt_positions, 6u);
  EXPECT_EQ(event.terminal->stats.prefilled_prompt_positions, 4u);
  EXPECT_EQ(executor.live->load(), 0);
}

TEST_F(ImagePreparationTest, ShutdownDuringPreparationSettlesOnceWithoutOpen) {
  start();
  executor.preparing.hold();
  Events event;
  auto handle = submit(event, {{image()}});
  ASSERT_TRUE(executor.preparing.wait());
  auto stopped = std::async(std::launch::async, [&] { runtime->shutdown(); });
  EXPECT_EQ(stopped.wait_for(20ms), std::future_status::timeout);
  executor.preparing.release();
  ASSERT_EQ(stopped.wait_for(5s), std::future_status::ready);
  stopped.get();
  EXPECT_TRUE(handle.done());
  EXPECT_EQ(event.terminals, 1);
  EXPECT_EQ(event.terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(executor.opens.load(), 0);
  EXPECT_EQ(executor.live->load(), 0);
}

} // namespace
