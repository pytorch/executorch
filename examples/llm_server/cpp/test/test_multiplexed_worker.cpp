/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Cppcheck's default model cannot expand GoogleTest registration macros.
// The CMake test target checks their C++ syntax and executes the tests.

// cppcheck-suppress-file syntaxError

#include <executorch/examples/llm_server/cpp/multiplexed_worker.h>
#include <executorch/examples/llm_server/cpp/multiplexed_worker_test.h>
#include <executorch/examples/llm_server/cpp/test_image_executor.h>

#include <poll.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/tokenizer.h>

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>

namespace {
namespace batching = executorch::extension::llm::batching;
namespace serving = executorch::extension::llm::serving;
using executorch::examples::llm_server::MultiplexedWorkerConfig;
using executorch::examples::llm_server::testing::Checkpoint;
using executorch::examples::llm_server::testing::ImageExecutor;
using executorch::examples::llm_server::testing::run_multiplexed_worker;
using executorch::examples::llm_server::testing::WorkerTestHooks;
using Json = nlohmann::json;
using namespace std::chrono_literals;

class Tokenizer : public tokenizers::Tokenizer {
 public:
  std::string piece = "x\n\"";
  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }
  tokenizers::Result<std::string> decode(uint64_t, uint64_t, bool)
      const override {
    return piece;
  }
  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t, int8_t) const override {
    return std::vector<uint64_t>(text.begin(), text.end());
  }
  tokenizers::Result<std::string> id_to_piece(uint64_t) const override {
    return piece;
  }
  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override {
    return uint64_t{1};
  }
};

class Executor : public ImageExecutor {
 public:
  explicit Executor(bool images = false) : ImageExecutor(images) {}

  // Installed before start(); invoked only on the engine thread.
  std::function<void(std::size_t)> before_execute;
  std::function<void(std::size_t)> before_prepare;
  std::atomic<std::size_t> prepare_calls{0};
  std::atomic<std::size_t> clone_calls{0};
  bool allow_clones = false;

  bool prepare(
      const batching::PreparationInput& input,
      batching::PreparedInputPtr& out) override {
    const auto call = ++prepare_calls;
    if (before_prepare)
      before_prepare(call);
    return ImageExecutor::prepare(input, out);
  }

  std::optional<batching::SessionId> clone(
      batching::SessionId,
      batching::Position) override {
    ++clone_calls;
    // FakeExecutor has no KV storage; an independent row is a sufficient
    // snapshot for tests that only observe whether caching is attempted.
    return allow_clones ? open_session() : std::nullopt;
  }

  std::vector<std::vector<batching::Token>> supplied() const {
    std::lock_guard<std::mutex> lock(supplied_mutex_);
    return supplied_;
  }

  bool execute(const batching::BatchInput& batch, batching::BatchOutput& out)
      override {
    if (before_execute)
      before_execute(++execute_count_);
    std::this_thread::sleep_for(2ms);
    {
      std::lock_guard<std::mutex> lock(supplied_mutex_);
      for (const auto& input : batch.inputs) {
        std::vector<batching::Token> tokens;
        for (std::size_t i = 0; i < input.size; ++i)
          tokens.push_back(input_token(*input.prepared, input.offset + i));
        supplied_.push_back(std::move(tokens));
      }
    }
    return ImageExecutor::execute(batch, out);
  }

 private:
  std::size_t execute_count_ = 0;
  mutable std::mutex supplied_mutex_;
  std::vector<std::vector<batching::Token>> supplied_;
};

class CheckpointGate {
 public:
  void signal() {
    std::lock_guard<std::mutex> lock(mutex_);
    reached_ = true;
    cv_.notify_all();
  }

  void pause() {
    signal();
    std::unique_lock<std::mutex> lock(mutex_);
    EXPECT_TRUE(cv_.wait_for(lock, 5s, [this] { return released_; }));
  }

  bool wait(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [this] { return reached_; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool reached_ = false;
  bool released_ = false;
};

class ProtocolTest : public ::testing::Test {
 protected:
  explicit ProtocolTest(bool images = false)
      : executor_(images), images_(images) {}

  void start(
      MultiplexedWorkerConfig config = {},
      std::size_t max_sessions = 4,
      std::size_t max_requests = 8,
      std::size_t max_pending_operations = 64) {
    signal(SIGPIPE, SIG_IGN);
    ASSERT_EQ(pipe(input_), 0);
    ASSERT_EQ(pipe(output_), 0);
    serving::ServingRuntimeConfig runtime_config;
    runtime_config.max_sessions = max_sessions;
    runtime_config.max_context_length = 8192;
    runtime_config.max_requests = max_requests;
    runtime_config.max_pending_operations = max_pending_operations;
    runtime_config.max_events_per_request = 128;
    runtime_config.max_tokens_per_request = 8192;
    if (images_)
      ImageExecutor::enable_images(runtime_config);
    if (configure_runtime_)
      configure_runtime_(runtime_config);
    runtime_ = std::make_unique<serving::ServingRuntime>(
        executor_,
        batching::DecodeFirstScheduler::create(16, 8, 8),
        tokenizer_,
        runtime_config);
    result_ = std::async(std::launch::async, [&, config] {
      return run_multiplexed_worker(
          *runtime_, input_[0], output_[1], config, hooks_);
    });
    auto ready = receive();
    ready_ = ready;
    ASSERT_TRUE(ready.value("ready", false));
    ASSERT_TRUE(ready.value("multiplexed", false));
    ASSERT_FALSE(ready.contains("request_id"));
    ASSERT_EQ(ready.at("max_named_sessions"), max_sessions);
    ASSERT_EQ(ready.at("max_inflight_requests"), config.max_inflight_requests);
  }

  std::shared_ptr<CheckpointGate> gate() {
    auto result = std::make_shared<CheckpointGate>();
    gates_.push_back(result);
    return result;
  }

  void TearDown() override {
    for (const auto& checkpoint : gates_)
      checkpoint->release();
    executor_.release();
    if (input_[1] >= 0) {
      close(input_[1]);
      input_[1] = -1;
    }
    // Unblock a stalled writer even after an assertion fails.
    if (output_[0] >= 0) {
      close(output_[0]);
      output_[0] = -1;
    }
    if (result_.valid())
      (void)result_.get();
    runtime_.reset();
    for (int fd : {input_[0], input_[1], output_[0], output_[1]})
      if (fd >= 0)
        close(fd);
    EXPECT_EQ(executor_.open_count(), 0);
    EXPECT_EQ(executor_.opened().size(), executor_.closed().size());
  }

  void send_raw(const std::string& bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
      auto count =
          write(input_[1], bytes.data() + offset, bytes.size() - offset);
      ASSERT_GT(count, 0);
      offset += count;
    }
  }
  void send(Json message) {
    send_raw(message.dump() + "\n");
  }
  Json generate(uint64_t id, int count = 3) {
    return {
        {"op", "generate"},
        {"request_id", id},
        {"prompt", "hi"},
        {"max_new_tokens", count}};
  }
  Json receive() {
    std::string frame;
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (std::chrono::steady_clock::now() < deadline) {
      pollfd fd{output_[0], POLLIN, 0};
      if (poll(&fd, 1, 20) <= 0)
        continue;
      char c;
      if (read(output_[0], &c, 1) != 1)
        break;
      if (c == '\n')
        return Json::parse(frame);
      frame += c;
    }
    ADD_FAILURE() << "timed out waiting for complete JSONL record: " << frame;
    return Json::object();
  }
  bool wait_execute() {
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!executor_.in_execute() &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    return executor_.in_execute();
  }
  int finish() {
    close(input_[1]);
    input_[1] = -1;
    if (result_.wait_for(5s) != std::future_status::ready) {
      ADD_FAILURE() << "worker failed to stop";
      return -1;
    }
    return result_.get();
  }

  Json ready_;
  std::function<void(serving::ServingRuntimeConfig&)> configure_runtime_;
  WorkerTestHooks hooks_;
  std::vector<std::shared_ptr<CheckpointGate>> gates_;
  Tokenizer tokenizer_;
  Executor executor_;
  std::unique_ptr<serving::ServingRuntime> runtime_;
  std::future<int> result_;
  int input_[2]{-1, -1}, output_[2]{-1, -1};
  const bool images_;
};

std::vector<std::uint8_t> png_bytes(bool white = false) {
  // Complete 1x1 PNG; header mutations below exercise bounds, not a codec.
  std::vector<std::uint8_t> bytes = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
      0x08, 0x04, 0x00, 0x00, 0x00, 0xb5, 0x1c, 0x0c, 0x02, 0x00, 0x00, 0x00,
      0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x60, 0xf8, 0x0f, 0x00,
      0x01, 0x02, 0x01, 0x00, 0x42, 0xbe, 0xbc, 0x68, 0x00, 0x00, 0x00, 0x00,
      0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
  if (white) {
    const std::uint8_t idat[] = {
        0x78,
        0x9c,
        0x63,
        0xf8,
        0xff,
        0x1f,
        0x00,
        0x03,
        0x00,
        0x01,
        0xff,
        0xfc,
        0x25,
        0xdc,
        0x51};
    std::copy(std::begin(idat), std::end(idat), bytes.begin() + 41);
  }
  return bytes;
}

std::string base64(const std::vector<std::uint8_t>& bytes) {
  constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const unsigned a = bytes[i];
    const unsigned b = i + 1 < bytes.size() ? bytes[i + 1] : 0;
    const unsigned c = i + 2 < bytes.size() ? bytes[i + 2] : 0;
    result += alphabet[a >> 2];
    result += alphabet[((a & 3) << 4) | (b >> 4)];
    result += i + 1 < bytes.size() ? alphabet[((b & 15) << 2) | (c >> 6)] : '=';
    result += i + 2 < bytes.size() ? alphabet[c & 63] : '=';
  }
  return result;
}

Json image_segment(const std::vector<std::uint8_t>& bytes = png_bytes()) {
  return {{"image", {{"mime_type", "image/png"}, {"data", base64(bytes)}}}};
}

class ImageProtocolTest : public ProtocolTest {
 protected:
  ImageProtocolTest() : ProtocolTest(true) {}

  Json image_request(uint64_t id, int count = 3) {
    auto request = generate(id, count);
    request.erase("prompt");
    request["prompt_segments"] = Json::array(
        {{{"text", "abcdef"}},
         image_segment(),
         {{"ids", {7, 8}}},
         {{"text", "z"}}});
    return request;
  }

  Json terminal(uint64_t id) {
    for (;;) {
      auto message = receive();
      EXPECT_EQ(message.value("request_id", uint64_t{0}), id) << message;
      if (!message.contains("token"))
        return message;
    }
  }

  void expect_invalid(const Json& request, const std::string& reason = {}) {
    SCOPED_TRACE(request.dump());
    send(request);
    const auto error = receive();
    EXPECT_EQ(error.at("request_id"), request.at("request_id"));
    EXPECT_EQ(error.at("code"), "invalid_argument") << error;
    EXPECT_FALSE(error.contains("done"));
    if (!reason.empty())
      EXPECT_NE(
          error.at("error").get<std::string>().find(reason), std::string::npos)
          << error;
  }
};

TEST_F(ProtocolTest, ImagesAreUnsupportedByDefault) {
  start();
  EXPECT_FALSE(ready_.at("supports_images").get<bool>());
  EXPECT_EQ(ready_.at("max_images"), 0);
  EXPECT_EQ(runtime_->info().max_images, 0u);
  auto request = generate(1);
  request.erase("prompt");
  request["prompt_segments"] = Json::array({image_segment()});
  send(request);
  const auto error = receive();
  EXPECT_EQ(error.at("request_id"), 1);
  EXPECT_EQ(error.at("code"), "invalid_argument");
  EXPECT_EQ(executor_.prepare_calls.load(), 0u);
  EXPECT_TRUE(executor_.opened().empty());
  EXPECT_EQ(finish(), 0);
}

TEST_F(
    ImageProtocolTest,
    InterleavedSegmentsPreservePositionsAndTokenFeedback) {
  executor_.stop_token = 1000;
  start();
  EXPECT_TRUE(ready_.at("supports_images").get<bool>());
  EXPECT_EQ(ready_.at("max_images"), 1);
  EXPECT_EQ(ready_.at("max_image_bytes"), 512 * 1024);
  EXPECT_EQ(ready_.at("max_image_dimension"), 4096);
  EXPECT_EQ(ready_.at("max_image_pixels"), 4 * 1024 * 1024);
  send(image_request(1));
  std::string text;
  Json done;
  do {
    done = receive();
    ASSERT_EQ(done.at("request_id"), 1);
    ASSERT_FALSE(done.contains("error")) << done;
    if (done.contains("token"))
      text += done.at("token").get<std::string>();
  } while (done.contains("token"));
  ASSERT_TRUE(done.value("done", false));
  EXPECT_EQ(text, tokenizer_.piece + tokenizer_.piece + tokenizer_.piece);
  EXPECT_EQ(done.at("finish_reason"), "length");
  EXPECT_EQ(done.at("generated_token_ids"), Json::array({1000, 1000, 1000}));
  EXPECT_EQ(done.at("completion_tokens"), 3);
  EXPECT_EQ(done.at("prompt_tokens"), 9);
  EXPECT_EQ(done.at("prompt_positions"), 13);
  // Opaque image positions have no tokenizer-token mapping for partial work.
  EXPECT_EQ(done.at("prefilled_prompt_tokens"), 0);
  EXPECT_EQ(done.at("prefilled_prompt_positions"), 13);
  EXPECT_EQ(done.at("reused_prompt_tokens"), 0);
  EXPECT_EQ(done.at("reused_prompt_positions"), 0);
  EXPECT_EQ(executor_.prepare_calls.load(), 1u);
  const auto seen = executor_.seen();
  const auto supplied = executor_.supplied();
  ASSERT_EQ(seen.size(), 4u);
  ASSERT_EQ(supplied.size(), seen.size());
  EXPECT_EQ(seen[0].effective_position(), 0);
  EXPECT_EQ(seen[0].size, 8u);
  EXPECT_EQ(seen[1].effective_position(), 8);
  EXPECT_EQ(seen[1].size, 5u);
  EXPECT_EQ(seen[2].effective_position(), 13);
  EXPECT_EQ(seen[3].effective_position(), 14);
  // The image straddles the 8-position prefill boundary.
  std::uint8_t hash = 0;
  for (auto byte : png_bytes())
    hash = static_cast<std::uint8_t>(hash * 31 + byte);
  EXPECT_EQ(
      supplied[0],
      (std::vector<batching::Token>{
          'a', 'b', 'c', 'd', 'e', 'f', hash, batching::Token(hash) + 1}));
  EXPECT_EQ(
      supplied[1],
      (std::vector<batching::Token>{
          batching::Token(hash) + 2, batching::Token(hash) + 3, 7, 8, 'z'}));
  EXPECT_EQ(supplied[2], (std::vector<batching::Token>{1000}));
  EXPECT_EQ(supplied[3], (std::vector<batching::Token>{1000}));
  send({{"op", "close"}, {"request_id", 2}, {"session_id", "absent"}});
  const auto fence = receive();
  EXPECT_EQ(fence.at("request_id"), 2); // No second generation terminal.
  EXPECT_TRUE(fence.value("closed", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ImageProtocolTest, StrictImageFieldsAndCanonicalBase64AreIsolated) {
  start();
  uint64_t id = 1;
  const auto valid = image_segment().at("image");
  std::vector<Json> invalid_images = {
      nullptr,
      true,
      "data",
      Json::object(),
      {{"mime_type", "image/png"}},
      {{"data", valid.at("data")}},
      {{"mime_type", "image/gif"}, {"data", valid.at("data")}},
      {{"mime_type", "image/jpeg"}, {"data", valid.at("data")}},
      {{"mime_type", true}, {"data", valid.at("data")}},
      {{"mime_type", "image/png"}, {"data", 1}},
      {{"mime_type", "image/png"},
       {"data", valid.at("data")},
       {"url", "file:///x"}}};
  for (const auto* data :
       {"",
        "a",
        "abc",
        "!!!!",
        "AA A",
        "AA\nA",
        "AA_A",
        "AA-A",
        "=AAA",
        "A===",
        "AA==AAAA",
        "AB==",
        "AAB="}) {
    invalid_images.push_back({{"mime_type", "image/png"}, {"data", data}});
  }
  // Nonzero pad bits on otherwise valid PNG data must not be accepted.
  auto bad_padding = valid.at("data").get<std::string>();
  bad_padding[bad_padding.size() - 2] = 'J'; // canonical final I= -> J=
  invalid_images.push_back({{"mime_type", "image/png"}, {"data", bad_padding}});
  for (const auto& image : invalid_images) {
    auto request = image_request(id++);
    request["prompt_segments"] = Json::array({{{"image", image}}});
    const bool pad_bits = image.is_object() && image.contains("data") &&
        (image.at("data") == "AB==" || image.at("data") == "AAB=");
    expect_invalid(request, pad_bits ? "canonical base64" : "");
  }
  for (const auto& segments : std::vector<Json>{
           Json::array(),
           Json::array({Json::object()}),
           Json::array({image_segment(), image_segment()}),
           Json::array({{{"image", valid}, {"text", "x"}}}),
           Json::array({{{"image", valid}, {"ids", {1}}}}),
           Json::array({{{"image", valid}, {"unknown", 1}}})}) {
    auto request = image_request(id++);
    request["prompt_segments"] = segments;
    expect_invalid(request);
  }
  EXPECT_EQ(executor_.prepare_calls.load(), 0u);
  EXPECT_TRUE(executor_.opened().empty());
  send(image_request(id, 1));
  EXPECT_TRUE(terminal(id).value("done", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ImageProtocolTest, DuplicateImageKeysAreRejectedBeforePreparation) {
  start();
  const auto image = image_segment().at("image").dump();
  const auto data = image_segment().at("image").at("data").dump();
  const std::vector<std::string> segments = {
      "{\"image\":" + image + ",\"image\":" + image + "}",
      "{\"image\":{\"mime_type\":\"image/png\",\"mime_type\":\"image/png\",\"data\":" +
          data + "}}",
      "{\"image\":{\"mime_type\":\"image/png\",\"data\":" + data +
          ",\"data\":" + data + "}}"};
  uint64_t id = 1;
  for (const auto& segment : segments) {
    send_raw(
        "{\"op\":\"generate\",\"request_id\":" + std::to_string(id) +
        ",\"max_new_tokens\":1,\"prompt_segments\":[" + segment + "]}\n");
    const auto error = receive();
    EXPECT_EQ(error.at("request_id"), id++);
    EXPECT_EQ(error.at("code"), "invalid_argument") << error;
  }
  EXPECT_EQ(executor_.prepare_calls.load(), 0u);
  EXPECT_TRUE(executor_.opened().empty());
  EXPECT_EQ(finish(), 0);
}

TEST_F(
    ImageProtocolTest,
    EncodedByteDimensionAndPixelBoundsPrecedePreparation) {
  configure_runtime_ = [](auto& config) {
    config.max_image_encoded_bytes = png_bytes().size();
    config.max_image_dimension = 4;
    config.max_image_pixels = 8;
  };
  start();
  EXPECT_EQ(ready_.at("max_image_bytes"), png_bytes().size());
  EXPECT_EQ(ready_.at("max_image_dimension"), 4);
  EXPECT_EQ(ready_.at("max_image_pixels"), 8);
  std::vector<std::vector<std::uint8_t>> invalid;
  auto bytes = png_bytes();
  bytes.push_back(0);
  invalid.push_back(bytes);
  bytes = png_bytes();
  bytes[19] = 5; // Width above per-dimension bound.
  invalid.push_back(bytes);
  bytes = png_bytes();
  bytes[23] = 5; // Height above per-dimension bound.
  invalid.push_back(bytes);
  bytes = png_bytes();
  bytes[19] = bytes[23] = 3; // Individually bounded, but 9 > 8 pixels.
  invalid.push_back(bytes);
  bytes = png_bytes();
  bytes[19] = 0; // Zero width.
  invalid.push_back(bytes);
  bytes = png_bytes();
  bytes.resize(8); // Signature without IHDR.
  invalid.push_back(bytes);
  bytes = png_bytes();
  bytes[12] = 'X'; // First chunk must be IHDR.
  invalid.push_back(bytes);
  bytes = png_bytes();
  bytes.resize(24); // Truncated IHDR.
  invalid.push_back(bytes);
  uint64_t id = 1;
  for (const auto& source : invalid) {
    auto request = image_request(id++);
    request["prompt_segments"] = Json::array({image_segment(source)});
    expect_invalid(request);
  }
  EXPECT_EQ(executor_.prepare_calls.load(), 0u);
  EXPECT_TRUE(executor_.opened().empty());
  auto request = image_request(id, 1);
  request["prompt_segments"] = Json::array({image_segment()});
  send(request); // Exact byte bound and an image-only prompt are accepted.
  const auto done = terminal(id);
  ASSERT_TRUE(done.value("done", false)) << done;
  EXPECT_EQ(done.at("prompt_tokens"), 0);
  EXPECT_EQ(done.at("prompt_positions"), 4);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ImageProtocolTest, EqualPatchCountsColdReplayAndBypassPrefixCapture) {
  configure_runtime_ = [](auto& config) { config.prefix_cache_capacity = 2; };
  executor_.stop_token = 1000;
  executor_.allow_clones = true;
  start();
  auto seed = image_request(10, 1);
  seed["prompt_segments"].erase(seed["prompt_segments"].begin() + 1);
  send(seed);
  ASSERT_TRUE(terminal(10).value("done", false));
  const auto cached_clones = executor_.clone_calls.load();
  ASSERT_GT(cached_clones, 0u);
  const auto seed_slices = executor_.seen().size();
  auto first = image_request(1, 1);
  first["session_id"] = "images";
  send(first);
  ASSERT_TRUE(terminal(1).value("done", false));
  const auto changed = png_bytes(true); // New pixel, same four synthetic rows.
  auto replay = first;
  replay["request_id"] = 2;
  replay["prompt_segments"][1] = image_segment(changed);
  send(replay);
  const auto done = terminal(2);
  ASSERT_TRUE(done.value("done", false)) << done;
  EXPECT_EQ(done.at("prompt_positions"), 13);
  EXPECT_EQ(done.at("reused_prompt_positions"), 0);
  EXPECT_EQ(done.at("prefilled_prompt_positions"), 13);
  EXPECT_EQ(done.at("reused_prompt_tokens"), 0);
  const auto all_seen = executor_.seen();
  const std::vector<Executor::Seen> seen(
      all_seen.begin() + seed_slices, all_seen.end());
  ASSERT_EQ(seen.size(), 4u);
  EXPECT_NE(seen[0].session, seen[2].session);
  EXPECT_EQ(seen[2].effective_position(), 0);
  EXPECT_EQ(seen[2].size, 8u);
  EXPECT_EQ(executor_.prepare_calls.load(), 3u);
  EXPECT_EQ(executor_.clone_calls.load(), cached_clones);
  EXPECT_NE(
      executor_.supplied()[seed_slices], executor_.supplied()[seed_slices + 2]);
  replay["request_id"] = 3;
  replay["session_id"] = "new-images";
  send(replay);
  const auto fresh = terminal(3);
  ASSERT_TRUE(fresh.value("done", false)) << fresh;
  EXPECT_EQ(fresh.at("reused_prompt_positions"), 0);
  EXPECT_EQ(fresh.at("prefilled_prompt_positions"), 13);
  EXPECT_EQ(executor_.clone_calls.load(), cached_clones);
  // Even removing the image from a resident history requires cold replay.
  auto text = generate(4, 1);
  text["session_id"] = "images";
  send(text);
  const auto text_done = terminal(4);
  ASSERT_TRUE(text_done.value("done", false)) << text_done;
  EXPECT_EQ(text_done.at("reused_prompt_tokens"), 0);
  EXPECT_EQ(text_done.at("prompt_positions"), 2);
  EXPECT_EQ(executor_.seen().back().effective_position(), 0);
  EXPECT_NE(executor_.seen().back().session, seen[2].session);
  EXPECT_EQ(finish(), 0);
}

TEST_F(
    ImageProtocolTest,
    QueuedAndInFlightPreparationCancellationAvoidsPrefill) {
  auto preparing = gate();
  auto queued = gate();
  executor_.before_prepare = [preparing](std::size_t call) {
    if (call == 1)
      preparing->pause();
  };
  hooks_.checkpoint = [queued](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::Bound && id == 2)
      queued->signal();
  };
  start();
  send(image_request(1));
  ASSERT_TRUE(preparing->wait());
  send(image_request(2));
  ASSERT_TRUE(queued->wait());
  // This control-thread fence proves the second preparation has been queued
  // to Runner's inbox; its engine cannot submit scheduler work while blocked.
  send({{"op", "close"}, {"request_id", 6}, {"session_id", "unrelated"}});
  const auto fence = receive();
  ASSERT_EQ(fence.at("request_id"), 6);
  ASSERT_TRUE(fence.value("closed", false));
  send({{"op", "cancel"}, {"request_id", 3}, {"target_request_id", 2}});
  EXPECT_EQ(receive().at("request_id"), 3);
  send({{"op", "cancel"}, {"request_id", 4}, {"target_request_id", 1}});
  // Queued cancellation may already publish its terminal before the next ACK.
  std::map<uint64_t, int> terminals;
  bool acknowledged = false;
  while (!acknowledged) {
    auto message = receive();
    const auto id = message.at("request_id").get<uint64_t>();
    if (id == 4) {
      EXPECT_TRUE(message.value("cancelled", false));
      acknowledged = true;
    } else {
      ASSERT_TRUE(id == 1 || id == 2) << message;
      EXPECT_TRUE(message.value("cancelled", false));
      EXPECT_EQ(++terminals[id], 1);
    }
  }
  EXPECT_TRUE(executor_.seen().empty());
  EXPECT_TRUE(executor_.opened().empty());
  preparing->release();
  while (terminals.size() < 2) {
    auto message = receive();
    const auto id = message.at("request_id").get<uint64_t>();
    ASSERT_TRUE(id == 1 || id == 2) << message;
    EXPECT_TRUE(message.value("done", false));
    EXPECT_TRUE(message.value("cancelled", false));
    EXPECT_EQ(message.at("completion_tokens"), 0);
    EXPECT_EQ(++terminals[id], 1);
  }
  EXPECT_EQ(executor_.prepare_calls.load(), 1u);
  EXPECT_TRUE(executor_.seen().empty());
  send(image_request(5, 1));
  EXPECT_TRUE(terminal(5).value("done", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ImageProtocolTest, CloseDuringPreparationFencesReplacementAndTerminal) {
  auto preparing = gate();
  executor_.before_prepare = [preparing](std::size_t call) {
    if (call == 1)
      preparing->pause();
  };
  start();
  auto old = image_request(1);
  old["session_id"] = "images";
  send(old);
  ASSERT_TRUE(preparing->wait());
  send({{"op", "close"}, {"request_id", 2}, {"session_id", "images"}});
  auto next = image_request(3, 1);
  next["session_id"] = "images";
  send(next);
  send({{"op", "close"}, {"request_id", 4}, {"session_id", "unrelated"}});
  const auto unrelated = receive();
  ASSERT_EQ(unrelated.at("request_id"), 4);
  ASSERT_TRUE(unrelated.value("closed", false));
  EXPECT_EQ(executor_.prepare_calls.load(), 1u);
  EXPECT_TRUE(executor_.seen().empty());
  EXPECT_TRUE(executor_.opened().empty());
  preparing->release();
  const auto cancelled = receive();
  ASSERT_EQ(cancelled.at("request_id"), 1);
  EXPECT_TRUE(cancelled.value("done", false));
  EXPECT_TRUE(cancelled.value("cancelled", false));
  EXPECT_EQ(cancelled.at("completion_tokens"), 0);
  const auto closed = receive();
  ASSERT_EQ(closed.at("request_id"), 2);
  EXPECT_TRUE(closed.value("closed", false));
  const auto replacement = terminal(3);
  ASSERT_TRUE(replacement.value("done", false)) << replacement;
  EXPECT_FALSE(replacement.value("cancelled", false));
  EXPECT_EQ(replacement.at("reused_prompt_positions"), 0);
  EXPECT_EQ(replacement.at("prefilled_prompt_positions"), 13);
  EXPECT_EQ(executor_.prepare_calls.load(), 2u);
  EXPECT_EQ(executor_.opened().size(), 1u);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, InterleavedIdsAndOutOfOrderCompletion) {
  start();
  send(generate(20, 40));
  send(generate(10, 2)); // Reserved earlier, arriving later is legal.
  std::vector<uint64_t> completed;
  std::map<uint64_t, std::string> text;
  while (completed.size() != 2) {
    auto msg = receive();
    ASSERT_TRUE(msg.contains("request_id"));
    const auto id = msg.at("request_id").get<uint64_t>();
    ASSERT_TRUE(id == 20 || id == 10);
    ASSERT_FALSE(msg.contains("error")) << msg;
    if (msg.contains("token"))
      text[id] += msg.at("token").get<std::string>();
    else {
      ASSERT_TRUE(msg.value("done", false));
      EXPECT_EQ(msg.at("finish_reason"), "length");
      EXPECT_EQ(msg.at("completion_tokens"), id == 10 ? 2 : 40);
      EXPECT_TRUE(msg.contains("generated_token_ids"));
      completed.push_back(id);
    }
  }
  EXPECT_EQ(completed, (std::vector<uint64_t>{10, 20}));
  EXPECT_EQ(text[10], tokenizer_.piece + tokenizer_.piece);
  EXPECT_EQ(text[20].size(), tokenizer_.piece.size() * 40);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, LifecycleAndTargetedCancellationWhileGenerationActive) {
  start();
  executor_.hold();
  send(generate(1, 100));
  ASSERT_TRUE(wait_execute());
  send({{"op", "close"}, {"request_id", 2}, {"session_id", "absent"}});
  auto closed = receive();
  EXPECT_EQ(closed.at("request_id"), 2);
  EXPECT_TRUE(closed.value("closed", false));
  send({{"op", "cancel"}, {"request_id", 3}, {"target_request_id", 1}});
  auto ack = receive();
  EXPECT_EQ(ack.at("request_id"), 3);
  EXPECT_TRUE(ack.value("cancelled", false));
  send(generate(6, 2));
  executor_.release();
  int completed = 0;
  while (completed != 2) {
    auto done = receive();
    if (done.contains("token"))
      continue;
    ASSERT_TRUE(done.value("done", false)) << done;
    const bool cancelled = done.at("request_id") == 1;
    EXPECT_EQ(done.value("cancelled", false), cancelled);
    EXPECT_EQ(done.at("finish_reason"), cancelled ? "stop" : "length");
    ++completed;
  }
  send({{"op", "cancel"}, {"request_id", 4}, {"target_request_id", 1}});
  EXPECT_TRUE(receive().value("cancelled", false));
  send({{"op", "cancel"}, {"request_id", 5}, {"target_request_id", 999}});
  EXPECT_TRUE(receive().value("cancelled", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, BusyAndMissingSessionErrorsKeepTheirWireCodes) {
  start();
  executor_.hold();
  auto active = generate(1, 100);
  active["session_id"] = "owned";
  send(active);
  ASSERT_TRUE(wait_execute());
  auto overlap = generate(2);
  overlap["session_id"] = "owned";
  send(overlap);
  auto busy = receive();
  EXPECT_EQ(busy.at("request_id"), 2);
  EXPECT_EQ(busy.at("code"), "session_busy");
  send({{"op", "reset"}, {"request_id", 3}, {"session_id", "absent"}});
  auto missing = receive();
  EXPECT_EQ(missing.at("request_id"), 3);
  EXPECT_EQ(missing.at("code"), "session_not_found");
  send({{"op", "cancel"}, {"request_id", 4}, {"target_request_id", 1}});
  EXPECT_TRUE(receive().value("cancelled", false));
  executor_.release();
  Json done;
  do {
    done = receive();
  } while (done.contains("token"));
  EXPECT_TRUE(done.value("cancelled", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, EofSettlesOutstandingGenerationAndLifecycleCommands) {
  start();
  executor_.hold();
  send(generate(1, 100));
  ASSERT_TRUE(wait_execute());
  send({{"op", "open"}, {"request_id", 2}, {"session_id", "other"}});
  send({{"op", "reset"}, {"request_id", 3}, {"session_id", "other"}});
  send({{"op", "close"}, {"request_id", 4}, {"session_id", "other"}});
  close(input_[1]);
  input_[1] = -1;
  executor_.release();
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 0);
  std::map<uint64_t, int> terminals;
  while (terminals.size() != 4) {
    const auto message = receive();
    ASSERT_TRUE(message.contains("request_id")) << message;
    if (message.contains("token"))
      continue;
    const auto id = message.at("request_id").get<uint64_t>();
    ASSERT_GE(id, 1u);
    ASSERT_LE(id, 4u);
    EXPECT_EQ(++terminals[id], 1);
    EXPECT_TRUE(
        message.contains("error") || message.value("done", false) ||
        message.value("opened", false) || message.value("reset", false) ||
        message.value("closed", false))
        << message;
  }
  EXPECT_EQ(executor_.open_count(), 0);
}

TEST_F(ProtocolTest, IncompleteUtf8IsReplacedWithoutFailingOtherRequests) {
  tokenizer_.piece = "\xE2";
  start();
  send(generate(1, 1));
  send(generate(2, 1));
  std::map<uint64_t, std::string> text;
  int completed = 0;
  while (completed != 2) {
    auto msg = receive();
    ASSERT_TRUE(msg.contains("request_id"));
    ASSERT_FALSE(msg.contains("error")) << msg;
    const auto id = msg.at("request_id").get<uint64_t>();
    if (msg.contains("token"))
      text[id] += msg.at("token").get<std::string>();
    else {
      ASSERT_TRUE(msg.value("done", false));
      ++completed;
    }
  }
  EXPECT_EQ(text[1], "\xEF\xBF\xBD");
  EXPECT_EQ(text[2], "\xEF\xBF\xBD");
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, AdmissionFailureDoesNotPoisonOtherRequests) {
  start({}, 1);
  send({{"op", "open"}, {"request_id", 1}, {"session_id", "owned"}});
  EXPECT_TRUE(receive().value("opened", false));
  send(generate(2));
  auto error = receive();
  EXPECT_EQ(error.at("request_id"), 2);
  EXPECT_EQ(error.at("code"), "capacity_exhausted");
  send({{"op", "reset"}, {"request_id", 3}, {"session_id", "owned"}});
  EXPECT_TRUE(receive().value("reset", false));
  auto request = generate(4);
  request["session_id"] = "owned";
  request.erase("prompt");
  request["prompt_segments"] =
      Json::array({{{"text", "h"}}, {{"ids", {7, 8}}}});
  send(request);
  Json done;
  do {
    done = receive();
  } while (done.contains("token"));
  EXPECT_TRUE(done.value("done", false));
  EXPECT_EQ(done.at("prompt_tokens"), 3);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, StrictFieldsAndUnsupportedModalitiesAreIsolated) {
  start();
  uint64_t id = 1;
  std::vector<Json> invalid;
  for (auto field :
       {"seed", "temperature", "top_p", "top_k", "max_new_tokens"}) {
    auto request = generate(id++);
    request[field] = true;
    invalid.push_back(request);
  }
  for (auto value :
       {Json(-2),
        Json(1.5),
        Json(uint64_t{1} << 40),
        Json(std::numeric_limits<uint64_t>::max())}) {
    auto request = generate(id++);
    request["max_new_tokens"] = value;
    invalid.push_back(request);
  }
  auto image = generate(id++);
  image["image"] = "ignored?";
  invalid.push_back(image);
  image = generate(id++);
  image.erase("prompt");
  image["prompt_segments"] =
      Json::array({{{"type", "image"}, {"image", "data"}}});
  invalid.push_back(image);
  auto both = generate(id++);
  both["prompt_segments"] = Json::array();
  invalid.push_back(both);
  auto bad_ids = generate(id++);
  bad_ids.erase("prompt");
  bad_ids["prompt_segments"] = Json::array({{{"ids", {1, -1}}}});
  invalid.push_back(bad_ids);
  auto ambiguous = generate(id++);
  ambiguous.erase("prompt");
  ambiguous["prompt_segments"] = Json::array({{{"text", "h"}, {"ids", {1}}}});
  invalid.push_back(ambiguous);
  auto hidden_image = generate(id++);
  hidden_image.erase("prompt");
  hidden_image["prompt_segments"] =
      Json::array({{{"text", "h"}, {"image", "data"}}});
  invalid.push_back(hidden_image);
  invalid.push_back(
      {{"op", "open"}, {"request_id", id++}, {"session_id", false}});
  invalid.push_back(
      {{"op", "cancel"}, {"request_id", id++}, {"target_request_id", 1.5}});
  for (const auto& request : invalid) {
    send(request);
    const auto error = receive();
    EXPECT_EQ(error.at("request_id"), request.at("request_id"));
    EXPECT_EQ(error.at("code"), "invalid_argument") << error;
  }
  send({{"op", "close"}, {"request_id", id}, {"session_id", "absent"}});
  EXPECT_TRUE(receive().value("closed", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, ZeroSeedRemainsUnsetAndPositiveSeedIsPreserved) {
  start();
  uint64_t request_id = 1;
  for (uint64_t seed : {uint64_t{0}, std::numeric_limits<uint64_t>::max()}) {
    auto request = generate(request_id++, 1);
    request["seed"] = seed;
    send(request);
    Json done;
    do {
      done = receive();
    } while (done.contains("token"));
    ASSERT_TRUE(done.value("done", false)) << done;
    const auto seen = executor_.seen();
    ASSERT_FALSE(seen.empty());
    EXPECT_EQ(
        seen.back().sampling_seed,
        seed == 0 ? std::nullopt : std::optional<uint64_t>(seed));
  }
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, OversizedInputFailsBoundedly) {
  MultiplexedWorkerConfig config;
  config.max_frame_bytes = 1024;
  start(config);
  send_raw(std::string(1025, 'x') + "\n");
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 1);
}

class InvalidIdTest : public ProtocolTest,
                      public ::testing::WithParamInterface<const char*> {};

TEST_P(InvalidIdTest, RejectsUncorrelatableFrame) {
  start();
  send_raw(
      std::string("{\"request_id\":") + GetParam() + ",\"op\":\"close\"}\n");
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 1);
}

INSTANTIATE_TEST_SUITE_P(
    StrictIds,
    InvalidIdTest,
    ::testing::Values(
        "true",
        "0",
        "-1",
        "1.5",
        "18446744073709551616",
        "null",
        "\"1\""));

TEST_F(ProtocolTest, IncompleteInputFailsAtEof) {
  start();
  send_raw("{\"request_id\":1");
  EXPECT_EQ(finish(), 1);
}

TEST_F(ProtocolTest, MaxUint64IdAndAutoBudgetAreAccepted) {
  start();
  auto request = generate(std::numeric_limits<uint64_t>::max(), -1);
  request["stop"] = Json::array({"x"});
  send(request);
  auto done = receive();
  EXPECT_EQ(done.at("request_id"), std::numeric_limits<uint64_t>::max());
  EXPECT_TRUE(done.value("done", false));
  EXPECT_EQ(done.at("finish_reason"), "stop");
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, OversizedTerminalIsAnExplicitError) {
  MultiplexedWorkerConfig config;
  config.max_frame_bytes = 1024;
  start(config);
  send(generate(1, 300));
  Json terminal;
  do {
    terminal = receive();
  } while (terminal.contains("token"));
  EXPECT_EQ(terminal.at("request_id"), 1);
  EXPECT_EQ(terminal.at("code"), "frame_too_large");
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, DuplicateInflightIdIsRejectedWithoutAnotherAdmission) {
  start();
  executor_.hold();
  send(generate(1));
  ASSERT_TRUE(wait_execute());
  send(generate(1));
  executor_.release();
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 1);
  EXPECT_EQ(executor_.opened().size(), 1u);
}

TEST_F(ProtocolTest, BacklogCancelsOnlyAffectedRequestAndKeepsTerminalBudget) {
  MultiplexedWorkerConfig config;
  config.token_bytes_per_request = 1;
  start(config);
  send(generate(1, 100));
  send({{"op", "close"}, {"request_id", 2}, {"session_id", "absent"}});
  bool failed = false, closed = false;
  for (int i = 0; i < 2; ++i) {
    const auto msg = receive();
    if (msg.at("request_id") == 1) {
      EXPECT_EQ(msg.at("code"), "slow_consumer");
      failed = true;
    } else {
      EXPECT_TRUE(msg.value("closed", false));
      closed = true;
    }
  }
  EXPECT_TRUE(failed && closed);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, OverflowBeforeBindReplaysCancellationAndWaitsForTerminal) {
  auto binding = gate();
  auto overflow = gate();
  auto engine = gate();
  auto terminal = gate();
  executor_.before_execute = [engine](std::size_t call) {
    if (call == 2)
      engine->pause();
  };
  hooks_.checkpoint = [binding, overflow, terminal](
                          Checkpoint point, uint64_t id) {
    if (id != 1)
      return;
    if (point == Checkpoint::BeforeBind)
      binding->pause();
    if (point == Checkpoint::OverflowLatched)
      overflow->signal();
    if (point == Checkpoint::TerminalEnqueued)
      terminal->signal();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  config.token_bytes_per_request = 1;
  start(config, 1, 1);
  auto request = generate(1, 100);
  request["session_id"] = "owned";
  send(request);
  ASSERT_TRUE(binding->wait());
  ASSERT_TRUE(engine->wait());
  ASSERT_TRUE(overflow->wait());
  EXPECT_FALSE(terminal->wait(0ms));
  EXPECT_EQ(executor_.seen().size(), 1u);
  EXPECT_EQ(executor_.open_count(), 1);
  EXPECT_TRUE(executor_.closed().empty());
  pollfd descriptor{output_[0], POLLIN, 0};
  EXPECT_EQ(poll(&descriptor, 1, 0), 0);

  binding->release();
  send({{"op", "cancel"}, {"request_id", 2}, {"target_request_id", 99}});
  // Reaching the next input proves handle assignment and latch replay returned.
  const auto fence = receive();
  ASSERT_EQ(fence.at("request_id"), 2);
  ASSERT_TRUE(fence.value("cancelled", false));
  send(generate(3, 1));
  const auto overload = receive();
  ASSERT_EQ(overload.at("request_id"), 3);
  EXPECT_EQ(overload.at("code"), "capacity_exhausted");
  EXPECT_EQ(overload.at("error"), "worker operation capacity exhausted");
  EXPECT_FALSE(terminal->wait(0ms));
  engine->release();
  ASSERT_TRUE(terminal->wait());
  const auto failed = receive();
  EXPECT_EQ(failed.at("request_id"), 1);
  EXPECT_EQ(failed.at("code"), "slow_consumer");
  // Without replay, the invalid pre-bind handle silently loses cancellation
  // and all 100 forwards run, even though the wire still reports overflow.
  EXPECT_LE(executor_.seen().size(), 2u);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, LateOverflowPreservesCommittedNamedSessionHistory) {
  auto binding = gate();
  auto overflow = gate();
  auto terminal = gate();
  // Deterministic output ID, not a runtime stop token. Set before engine start.
  executor_.stop_token = 1000;
  hooks_.checkpoint = [binding, overflow, terminal](
                          Checkpoint point, uint64_t id) {
    if (id != 1)
      return;
    if (point == Checkpoint::BeforeBind)
      binding->pause();
    if (point == Checkpoint::OverflowLatched)
      overflow->signal();
    if (point == Checkpoint::TerminalEnqueued)
      terminal->signal();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  config.token_bytes_per_request = 1;
  start(config, 1, 1);
  auto request = generate(1, 1);
  request["session_id"] = "owned";
  send(request);
  ASSERT_TRUE(binding->wait());
  ASSERT_TRUE(overflow->wait());
  ASSERT_TRUE(terminal->wait());
  const auto failed = receive();
  ASSERT_EQ(failed.at("request_id"), 1);
  ASSERT_EQ(failed.at("code"), "slow_consumer");
  EXPECT_EQ(executor_.opened().size(), 1u);
  EXPECT_TRUE(executor_.closed().empty());

  auto continuation = generate(2, 1);
  continuation["session_id"] = "owned";
  continuation.erase("prompt");
  continuation["prompt_segments"] =
      Json::array({{{"ids", {104, 105, 1000, 106}}}});
  // Suppress text for this request so its tiny transport budget does not fail.
  continuation["stop"] = Json::array({"x"});
  send(continuation);
  binding->release();
  const auto done = receive();
  ASSERT_EQ(done.at("request_id"), 2);
  ASSERT_TRUE(done.value("done", false)) << done;
  EXPECT_EQ(done.at("session_reset_reason"), "exact_prefix");
  EXPECT_EQ(done.at("reused_prompt_tokens"), 2);
  EXPECT_EQ(executor_.opened().size(), 1u);
  EXPECT_TRUE(executor_.closed().empty());
  const auto seen = executor_.seen();
  ASSERT_GE(seen.size(), 2u);
  EXPECT_EQ(seen[1].session, seen[0].session);
  EXPECT_EQ(seen[1].effective_position(), 2);
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, PermanentlyUnreadOutputExpiresAndCleansUp) {
  MultiplexedWorkerConfig config;
  config.write_timeout = 100ms;
  config.token_frames_per_request = 128;
  config.token_bytes_per_request = 1024 * 1024;
  tokenizer_.piece = std::string(128 * 1024, 'x');
  start(config);
  send(generate(1, 100));
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 1);
  EXPECT_EQ(executor_.open_count(), 0);
}

TEST_F(ProtocolTest, PublishedControlReleasesCapacityBeforeWriterContinues) {
  auto published = gate();
  auto admitted = gate();
  hooks_.checkpoint = [published, admitted](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::TerminalPublished && id == 1)
      published->pause();
    if (point == Checkpoint::Admitted && id == 2)
      admitted->signal();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 4, 1);
  send({{"op", "cancel"}, {"request_id", 1}, {"target_request_id", 99}});
  const auto first = receive();
  ASSERT_EQ(first.at("request_id"), 1);
  ASSERT_TRUE(first.value("cancelled", false));
  ASSERT_TRUE(published->wait());
  send({{"op", "cancel"}, {"request_id", 2}, {"target_request_id", 99}});
  ASSERT_TRUE(admitted->wait());
  published->release();
  const auto second = receive();
  EXPECT_EQ(second.at("request_id"), 2);
  EXPECT_TRUE(second.value("cancelled", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, PublishedGenerationReusesWireIdBeforeCallbackReturns) {
  auto sink = gate();
  auto terminals = std::make_shared<std::atomic<int>>(0);
  auto first_handle = std::make_shared<std::promise<serving::RequestHandle>>();
  auto next_handle = std::make_shared<std::promise<serving::RequestHandle>>();
  auto first_bound = first_handle->get_future();
  auto next_bound = next_handle->get_future();
  hooks_.handle_bound = [first_handle, next_handle, bindings = 0](
                            uint64_t id,
                            const serving::RequestHandle& handle) mutable {
    if (id == 1) {
      if (++bindings == 1)
        first_handle->set_value(handle);
      else if (bindings == 2)
        next_handle->set_value(handle);
    }
  };
  hooks_.checkpoint = [sink, terminals](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::TerminalEnqueued && id == 1 && ++*terminals == 1)
      sink->pause();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 1, 1);
  auto request = generate(1, 1);
  request["session_id"] = "owned";
  send(request);
  ASSERT_TRUE(sink->wait());
  ASSERT_EQ(first_bound.wait_for(5s), std::future_status::ready);
  const auto handle = first_bound.get();
  ASSERT_NE(handle.id(), 0u);
  EXPECT_FALSE(handle.done());
  ASSERT_EQ(receive().at("token"), tokenizer_.piece);
  const auto done = receive();
  ASSERT_EQ(done.at("request_id"), 1);
  ASSERT_TRUE(done.value("done", false));

  // Both transport admission and runtime admission are reusable, even though
  // the old callback still owns its operation and done() is false.
  send(request);
  ASSERT_EQ(next_bound.wait_for(5s), std::future_status::ready);
  const auto replacement = next_bound.get();
  ASSERT_NE(replacement.id(), 0u);
  EXPECT_NE(replacement.id(), handle.id());
  EXPECT_FALSE(handle.done());
  send({{"op", "cancel"}, {"request_id", 2}, {"target_request_id", 99}});
  const auto ack = receive();
  ASSERT_EQ(ack.at("request_id"), 2);
  ASSERT_TRUE(ack.value("cancelled", false));
  EXPECT_FALSE(handle.done());
  sink->release();
  EXPECT_EQ(receive().at("token"), tokenizer_.piece);
  const auto next = receive();
  EXPECT_EQ(next.at("request_id"), 1);
  EXPECT_TRUE(next.value("done", false)) << next;
  EXPECT_EQ(finish(), 0);
  EXPECT_TRUE(handle.done());
  EXPECT_TRUE(replacement.done());
}

class LifecycleOrderingTest : public ProtocolTest,
                              public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    watchdog_ = std::thread([finished = finished_.get_future()] {
      if (finished.wait_for(30s) != std::future_status::ready) {
        ADD_FAILURE() << "lifecycle ordering test or teardown deadlocked";
        std::abort();
      }
    });
  }

  void TearDown() override {
    // Release every gate before joining the worker, including on ASSERT exit.
    // Keep the watchdog alive through the base fixture's runtime shutdown.
    ProtocolTest::TearDown();
    finished_.set_value();
    watchdog_.join();
  }

 private:
  std::promise<void> finished_;
  std::thread watchdog_;
};

TEST_P(
    LifecycleOrderingTest,
    CallbackFenceAndWriterFifoOrderPrequeuedReplacementAndHistory) {
  auto old_callback = gate();
  auto ack_enqueued = gate();
  auto ack_newline = gate();
  auto ack_published = gate();
  auto open_enqueued = gate();
  auto replacement_enqueued = gate();
  auto old_handle = std::make_shared<std::promise<serving::RequestHandle>>();
  auto next_handle = std::make_shared<std::promise<serving::RequestHandle>>();
  auto old_bound = old_handle->get_future();
  auto next_bound = next_handle->get_future();
  hooks_.handle_bound = [old_handle, next_handle](
                            uint64_t id, const serving::RequestHandle& handle) {
    if (id == 1)
      old_handle->set_value(handle);
    if (id == 4)
      next_handle->set_value(handle);
  };
  hooks_.checkpoint = [old_callback,
                       ack_enqueued,
                       ack_published,
                       open_enqueued,
                       replacement_enqueued](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::TerminalEnqueued) {
      if (id == 1)
        old_callback->pause();
      if (id == 2)
        ack_enqueued->signal();
      if (id == 3)
        open_enqueued->signal();
      if (id == 4)
        replacement_enqueued->signal();
    }
    if (point == Checkpoint::TerminalPublished && id == 2)
      ack_published->signal();
  };
  hooks_.terminal_write_error = [ack_newline](uint64_t id) {
    // This hook runs before the final-byte write acquires the worker mutex.
    // Blocking here leaves both the reader and runtime callbacks runnable.
    if (id == 2)
      ack_newline->pause();
    return 0;
  };
  executor_.stop_token =
      1000; // Deterministic output, not a runtime stop token.
  start();
  auto first = generate(1, 1);
  first["session_id"] = "owned";
  send(first);
  ASSERT_TRUE(old_callback->wait());
  ASSERT_EQ(old_bound.wait_for(5s), std::future_status::ready);
  const auto old = old_bound.get();
  ASSERT_NE(old.id(), 0u);

  const auto old_text = receive();
  ASSERT_EQ(old_text.at("request_id"), 1);
  ASSERT_EQ(old_text.at("token"), tokenizer_.piece);
  const auto old_terminal = receive();
  ASSERT_EQ(old_terminal.at("request_id"), 1);
  ASSERT_TRUE(old_terminal.value("done", false)) << old_terminal;
  EXPECT_EQ(old_terminal.at("finish_reason"), "length");
  EXPECT_EQ(old_terminal.at("generated_token_ids"), Json::array({1000}));
  EXPECT_FALSE(
      old.done()); // Complete JSONL publication is not callback return.

  send(
      {{"op", GetParam() ? "reset" : "close"},
       {"request_id", 2},
       {"session_id", "owned"}});
  send({{"op", "open"}, {"request_id", 3}, {"session_id", "owned"}});
  auto replacement = generate(4, 1);
  replacement["session_id"] = "owned";
  replacement.erase("prompt");
  // An extension of the old committed history must nevertheless start cold.
  replacement["prompt_segments"] =
      Json::array({{{"ids", {104, 105, 1000, 106}}}});
  send(replacement);
  ASSERT_EQ(next_bound.wait_for(5s), std::future_status::ready);
  const auto next = next_bound.get();
  ASSERT_NE(next.id(), 0u);
  EXPECT_NE(next.id(), old.id());
  EXPECT_FALSE(old.done());

  // Unlike a cancel reply, this ACK witnesses runtime control-thread progress
  // past the deferred fence and both later same-key submissions. Successful
  // handle assignment above proves runtime acceptance, not just reader input.
  send({{"op", "close"}, {"request_id", 5}, {"session_id", "unrelated"}});
  const auto control = receive();
  ASSERT_EQ(control.at("request_id"), 5);
  ASSERT_TRUE(control.value("closed", false)) << control;
  EXPECT_FALSE(old.done());
  EXPECT_FALSE(next.done());
  EXPECT_FALSE(ack_enqueued->wait(0ms));
  EXPECT_FALSE(open_enqueued->wait(0ms));
  EXPECT_FALSE(replacement_enqueued->wait(0ms));
  EXPECT_EQ(executor_.seen().size(), 1u);

  old_callback->release();
  ASSERT_TRUE(ack_newline->wait());
  ASSERT_TRUE(ack_enqueued->wait());
  EXPECT_TRUE(old.done());
  // The fence covers callback cleanup, not pipe I/O: later accepted work can
  // finish enqueueing while the ACK still lacks its JSONL delimiter. FIFO must
  // keep that work behind the ACK on the actual wire.
  ASSERT_TRUE(open_enqueued->wait());
  ASSERT_TRUE(replacement_enqueued->wait());
  EXPECT_FALSE(ack_published->wait(0ms));
  ack_newline->release();
  const auto ack = receive();
  ASSERT_EQ(ack.at("request_id"), 2);
  ASSERT_TRUE(ack.value(GetParam() ? "reset" : "closed", false)) << ack;
  const auto opened = receive();
  ASSERT_EQ(opened.at("request_id"), 3);
  ASSERT_TRUE(opened.value("opened", false)) << opened;
  const auto next_text = receive();
  ASSERT_EQ(next_text.at("request_id"), 4);
  ASSERT_EQ(next_text.at("token"), tokenizer_.piece);
  const auto next_terminal = receive();
  ASSERT_EQ(next_terminal.at("request_id"), 4);
  ASSERT_TRUE(next_terminal.value("done", false)) << next_terminal;
  EXPECT_EQ(next_terminal.at("finish_reason"), "length");
  EXPECT_FALSE(next_terminal.value("cancelled", false));
  EXPECT_EQ(next_terminal.at("completion_tokens"), 1);
  EXPECT_EQ(next_terminal.at("generated_token_ids"), Json::array({1000}));
  EXPECT_EQ(next_terminal.at("session_reset_reason"), "new");
  EXPECT_EQ(next_terminal.at("prompt_tokens"), 4);
  EXPECT_EQ(next_terminal.at("reused_prompt_tokens"), 0);
  EXPECT_EQ(next_terminal.at("prefilled_prompt_tokens"), 4);
  const auto cold = executor_.seen();
  ASSERT_EQ(cold.size(), 2u);
  EXPECT_NE(cold[1].session, cold[0].session);
  EXPECT_EQ(cold[1].effective_position(), 0);
  EXPECT_EQ(cold[1].size, 4u);
  EXPECT_EQ(executor_.opened().size(), 2u);
  EXPECT_EQ(
      executor_.closed(), (std::vector<batching::SessionId>{cold[0].session}));

  // Also submit after observing the ACK: replacement history, including its
  // pending generated token, must survive the old callback's late cleanup.
  auto continuation = replacement;
  continuation["request_id"] = 6;
  continuation["prompt_segments"] =
      Json::array({{{"ids", {104, 105, 1000, 106, 1000, 107}}}});
  send(continuation);
  const auto continued_text = receive();
  ASSERT_EQ(continued_text.at("request_id"), 6);
  ASSERT_EQ(continued_text.at("token"), tokenizer_.piece);
  const auto continued = receive();
  ASSERT_EQ(continued.at("request_id"), 6);
  ASSERT_TRUE(continued.value("done", false)) << continued;
  EXPECT_EQ(continued.at("finish_reason"), "length");
  EXPECT_EQ(continued.at("session_reset_reason"), "exact_prefix");
  EXPECT_EQ(continued.at("prompt_tokens"), 6);
  EXPECT_EQ(continued.at("reused_prompt_tokens"), 4);
  EXPECT_EQ(continued.at("prefilled_prompt_tokens"), 2);
  const auto warm = executor_.seen();
  ASSERT_GE(warm.size(), 3u);
  EXPECT_EQ(warm[2].session, cold[1].session);
  EXPECT_EQ(warm[2].effective_position(), 4);
  EXPECT_EQ(executor_.opened().size(), 2u);
  EXPECT_EQ(executor_.closed().size(), 1u);

  send({{"op", "close"}, {"request_id", 7}, {"session_id", "unrelated"}});
  const auto drained = receive();
  ASSERT_EQ(drained.at("request_id"), 7);
  ASSERT_TRUE(drained.value("closed", false)) << drained;
  EXPECT_EQ(finish(), 0);
  EXPECT_TRUE(old.done());
  EXPECT_TRUE(next.done());
}

INSTANTIATE_TEST_SUITE_P(
    CloseAndReset,
    LifecycleOrderingTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Reset" : "Close";
    });

TEST_F(ProtocolTest, TerminalPublishesBeforeBindButReaderWaitsForAssignment) {
  auto binding = gate();
  auto sink = gate();
  auto next_admitted = gate();
  auto admissions = std::make_shared<std::atomic<int>>(0);
  auto bindings = std::make_shared<std::atomic<int>>(0);
  auto bound = std::make_shared<std::atomic<bool>>(false);
  auto terminals = std::make_shared<std::atomic<int>>(0);
  hooks_.checkpoint =
      [binding, sink, next_admitted, admissions, bindings, bound, terminals](
          Checkpoint point, uint64_t id) {
        if (id != 1)
          return;
        if (point == Checkpoint::BeforeBind && ++*bindings == 1)
          binding->pause();
        if (point == Checkpoint::Bound)
          bound->store(true);
        if (point == Checkpoint::TerminalEnqueued && ++*terminals == 1)
          sink->pause();
        if (point == Checkpoint::Admitted && ++*admissions == 2) {
          EXPECT_TRUE(bound->load());
          next_admitted->signal();
        }
      };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 4, 1);
  send(generate(1, 1));
  ASSERT_TRUE(binding->wait());
  ASSERT_TRUE(sink->wait());
  ASSERT_EQ(receive().at("token"), tokenizer_.piece);
  const auto done = receive();
  ASSERT_EQ(done.at("request_id"), 1);
  ASSERT_TRUE(done.value("done", false));
  EXPECT_FALSE(bound->load());
  send(generate(1, 1));
  EXPECT_FALSE(next_admitted->wait(50ms));
  EXPECT_EQ(admissions->load(), 1);

  // Only the original assignment is gated. Publication already retired its
  // wire identity, but the serial reader cannot consume the next command yet.
  binding->release();
  ASSERT_TRUE(next_admitted->wait());
  sink->release();
  EXPECT_EQ(receive().at("token"), tokenizer_.piece);
  const auto next = receive();
  EXPECT_EQ(next.at("request_id"), 1);
  EXPECT_TRUE(next.value("done", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, PublishedLifecycleAckReentersWithOnePendingPermit) {
  auto callback = gate();
  auto terminals = std::make_shared<std::atomic<int>>(0);
  hooks_.checkpoint = [callback, terminals](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::TerminalEnqueued && id == 1 && ++*terminals == 1)
      callback->pause();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 1, 1, 1);
  send({{"op", "close"}, {"request_id", 1}, {"session_id", "absent"}});
  ASSERT_TRUE(callback->wait());
  const auto first = receive();
  ASSERT_EQ(first.at("request_id"), 1);
  ASSERT_TRUE(first.value("closed", false));
  send({{"op", "open"}, {"request_id", 1}, {"session_id", "owned"}});
  send({{"op", "cancel"}, {"request_id", 2}, {"target_request_id", 99}});
  // This reader fence follows submission. A retained runtime permit would
  // instead put an inline capacity error ahead of this ACK in the writer FIFO.
  const auto fence = receive();
  ASSERT_EQ(fence.at("request_id"), 2);
  ASSERT_TRUE(fence.value("cancelled", false));
  callback->release();
  const auto next = receive();
  EXPECT_EQ(next.at("request_id"), 1);
  EXPECT_TRUE(next.value("opened", false)) << next;
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, EofWaitsForPublishedLifecycleCallbackToReturn) {
  auto callback = gate();
  hooks_.checkpoint = [callback](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::TerminalEnqueued && id == 1)
      callback->pause();
  };
  start({}, 1, 1, 1);
  send({{"op", "close"}, {"request_id", 1}, {"session_id", "absent"}});
  ASSERT_TRUE(callback->wait());
  const auto ack = receive();
  ASSERT_EQ(ack.at("request_id"), 1);
  ASSERT_TRUE(ack.value("closed", false));
  close(input_[1]);
  input_[1] = -1;
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (runtime_->info().ready && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  ASSERT_FALSE(runtime_->info().ready);
  EXPECT_EQ(result_.wait_for(50ms), std::future_status::timeout);
  callback->release();
  ASSERT_EQ(result_.wait_for(5s), std::future_status::ready);
  EXPECT_EQ(result_.get(), 0);
}

TEST_F(ProtocolTest, TerminalNewlineRetriesKeepAdmissionReserved) {
  auto retry = gate();
  auto admitted = gate();
  auto attempts = std::make_shared<std::atomic<int>>(0);
  hooks_.terminal_write_error = [retry, attempts](uint64_t id) {
    if (id != 1)
      return 0;
    const auto attempt = ++*attempts;
    if (attempt == 1)
      return EAGAIN;
    if (attempt == 2)
      return EINTR;
    retry->pause();
    return 0;
  };
  hooks_.checkpoint = [admitted](Checkpoint point, uint64_t id) {
    if (point == Checkpoint::Admitted && id == 2)
      admitted->signal();
  };
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config, 4, 1);
  send(generate(1, 1));
  ASSERT_TRUE(retry->wait());
  // The terminal payload is written, but its LF has not been published.
  // This must still take the transport's overload path, not reach runtime.
  send(generate(2, 1));
  ASSERT_TRUE(admitted->wait());
  retry->release();
  EXPECT_EQ(receive().at("token"), tokenizer_.piece);
  const auto done = receive();
  EXPECT_EQ(done.at("request_id"), 1);
  EXPECT_TRUE(done.value("done", false));
  const auto overload = receive();
  EXPECT_EQ(overload.at("request_id"), 2);
  EXPECT_EQ(overload.at("code"), "capacity_exhausted");
  EXPECT_EQ(overload.at("error"), "worker operation capacity exhausted");
  EXPECT_EQ(attempts->load(), 3);
  send(generate(3, 1));
  EXPECT_EQ(receive().at("token"), tokenizer_.piece);
  EXPECT_TRUE(receive().value("done", false));
  EXPECT_EQ(finish(), 0);
}

TEST_F(ProtocolTest, OperationCapacityReservesCancellationResponses) {
  MultiplexedWorkerConfig config;
  config.max_inflight_requests = 1;
  start(config);
  executor_.hold();
  send(generate(1, 100));
  ASSERT_TRUE(wait_execute());
  send(generate(2));
  EXPECT_EQ(receive().at("code"), "capacity_exhausted");
  send({{"op", "cancel"}, {"request_id", 3}, {"target_request_id", 1}});
  EXPECT_TRUE(receive().value("cancelled", false));
  executor_.release();
  Json done;
  do {
    done = receive();
  } while (done.contains("token"));
  EXPECT_TRUE(done.value("cancelled", false));
  EXPECT_EQ(finish(), 0);
}
// Subprocess mode for Python validation regressions. The inherited socket is
// separate from JSONL: 'A' witnesses active execution, and 'R' releases it.
int run_validation_worker(int gate_fd) {
  signal(SIGPIPE, SIG_IGN);
  Tokenizer tokenizer;
  Executor executor;
  executor.before_execute = [gate_fd](std::size_t call) {
    if (call != 1)
      return;
    pollfd gate{gate_fd, POLLIN, 0};
    char release = 0;
    if (write(gate_fd, "A", 1) != 1 || poll(&gate, 1, 10000) <= 0 ||
        read(gate_fd, &release, 1) != 1 || release != 'R') {
      std::_Exit(2);
    }
  };
  serving::ServingRuntimeConfig config;
  config.max_sessions = 4;
  config.max_context_length = 128;
  serving::ServingRuntime runtime(
      executor,
      batching::DecodeFirstScheduler::create(16, 8, 8),
      tokenizer,
      config);
  return executorch::examples::llm_server::run_multiplexed_worker(
      runtime, STDIN_FILENO, STDOUT_FILENO);
}
} // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--validation-worker") {
    return run_validation_worker(std::stoi(argv[2]));
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
