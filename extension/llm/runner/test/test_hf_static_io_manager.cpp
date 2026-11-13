/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// HfStaticIOManager drives a program with Hugging Face's static-cache ABI
// (export_hf_static_program.py): it owns the caches, which the program
// writes in place, and builds each step's tokens, masks and positions. The
// program's greedy tokens can only match the expected ones if every step's
// writes land in the IOManager's buffers.

#include <executorch/extension/llm/runner/io_manager/hf_static_io_manager.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::llm::HfStaticIOManager;
using executorch::runtime::Error;

namespace {

constexpr int64_t kPrefill = 4;

std::vector<int64_t> int_list(Module& module, const char* method) {
  auto result = module.execute(method);
  EXPECT_EQ(result.error(), Error::Ok);
  std::vector<int64_t> values;
  if (!result.ok()) {
    return {};
  }
  for (const auto& value : *result) {
    values.push_back(value.toInt());
  }
  return values;
}

// Exercise both float32 and float16 cache storage.
class HfStaticIOManagerTest : public ::testing::TestWithParam<const char*> {
 protected:
  void SetUp() override {
    const char* path = std::getenv(GetParam());
    ASSERT_NE(path, nullptr);
    module_ = std::make_unique<Module>(path);
    io_ = std::make_unique<HfStaticIOManager>(*module_);
    auto methods = module_->method_names();
    ASSERT_EQ(methods.error(), Error::Ok);
    std::tie(prefill_, decode_) = io_->method_names(*methods);
    ASSERT_EQ(io_->load(prefill_, decode_), Error::Ok);
  }

  // Runs `tokens` at `start` through `method`; returns the greedy next token.
  int64_t
  step(const std::string& method, std::vector<int64_t> tokens, int64_t start) {
    auto ids = from_blob(
        tokens.data(),
        {1, static_cast<executorch::aten::SizesType>(tokens.size())},
        executorch::aten::ScalarType::Long);
    auto pos = from_blob(&start, {1}, executorch::aten::ScalarType::Long);
    auto inputs = method == prefill_ ? io_->prepare_prefill(ids, pos, method)
                                     : io_->prepare_decode(ids, pos, method);
    EXPECT_EQ(inputs.error(), Error::Ok);
    if (!inputs.ok()) {
      return -1;
    }
    auto outputs = module_->execute(method, *inputs);
    EXPECT_EQ(outputs.error(), Error::Ok);
    if (!outputs.ok()) {
      return -1;
    }
    if (outputs->size() == 13) {
      for (size_t c = 0; c < 6; ++c) {
        EXPECT_EQ(
            (*outputs)[7 + c].toTensor().const_data_ptr(),
            (*inputs)[3 + c].toTensor().const_data_ptr());
      }
    }
    EXPECT_EQ(io_->update_decode(*outputs, method), Error::Ok);
    auto logits = io_->logits(*outputs, method);
    EXPECT_EQ(logits.error(), Error::Ok);
    if (!logits.ok()) {
      return -1;
    }
    const float* row = logits->const_data_ptr<float>();
    return std::max_element(row, row + logits->numel()) - row;
  }

  std::vector<int64_t> generate(const std::string& prompt_method) {
    const auto prompt = int_list(*module_, "get_test_prompt");
    const auto expected = int_list(*module_, "get_test_expected");
    const int64_t chunk = io_->max_tokens_per_step(prompt_method);
    int64_t start = 0, next = 0;
    while (start < static_cast<int64_t>(prompt.size())) {
      const auto end = std::min<size_t>(prompt.size(), start + chunk);
      std::vector<int64_t> part(prompt.begin() + start, prompt.begin() + end);
      next = step(prompt_method, part, start);
      start = end;
    }
    std::vector<int64_t> generated;
    for (size_t i = 0; i < expected.size(); ++i) {
      generated.push_back(next);
      next = step(decode_, {next}, start++);
    }
    return generated;
  }

  std::unique_ptr<Module> module_;
  std::unique_ptr<HfStaticIOManager> io_;
  std::string prefill_, decode_;
};

TEST_P(HfStaticIOManagerTest, PicksPrefillAndDecode) {
  EXPECT_EQ(prefill_, "prefill");
  EXPECT_EQ(decode_, "decode");
  EXPECT_EQ(io_->max_tokens_per_step(prefill_), kPrefill);
  EXPECT_EQ(io_->max_tokens_per_step(decode_), 1);
}

TEST_P(HfStaticIOManagerTest, GeneratesExpectedTokens) {
  EXPECT_EQ(generate(prefill_), int_list(*module_, "get_test_expected"));
}

TEST_P(HfStaticIOManagerTest, DecodeOnlyPromptMatches) {
  // The same caches serve both methods: a prompt fed one token at a time
  // through decode gives the same continuation.
  EXPECT_EQ(generate(decode_), int_list(*module_, "get_test_expected"));
}

TEST_P(HfStaticIOManagerTest, ResetClearsTheCaches) {
  const auto first = generate(prefill_);
  ASSERT_EQ(io_->reset(prefill_, decode_), Error::Ok);
  EXPECT_EQ(generate(prefill_), first);
}

TEST_P(HfStaticIOManagerTest, RejectsTooManyTokens) {
  std::vector<int64_t> tokens(kPrefill + 1, 1);
  int64_t start = 0;
  auto ids = from_blob(
      tokens.data(),
      {1, static_cast<executorch::aten::SizesType>(tokens.size())},
      executorch::aten::ScalarType::Long);
  auto pos = from_blob(&start, {1}, executorch::aten::ScalarType::Long);
  EXPECT_EQ(
      io_->prepare_prefill(ids, pos, prefill_).error(), Error::InvalidArgument);
}

TEST_P(HfStaticIOManagerTest, RejectsAPromptChunkPastTheCache) {
  // A prompt chunk writes all its rows, padding included.
  std::vector<int64_t> tokens = {1};
  int64_t start = 16 - kPrefill + 1;
  auto ids =
      from_blob(tokens.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto pos = from_blob(&start, {1}, executorch::aten::ScalarType::Long);
  EXPECT_EQ(
      io_->prepare_prefill(ids, pos, prefill_).error(), Error::OutOfResources);
}

TEST_P(HfStaticIOManagerTest, RejectsNegativePositions) {
  std::vector<int64_t> tokens = {1};
  int64_t start = -1;
  auto ids =
      from_blob(tokens.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto pos = from_blob(&start, {1}, executorch::aten::ScalarType::Long);
  EXPECT_EQ(
      io_->prepare_decode(ids, pos, decode_).error(), Error::InvalidArgument);
}

TEST_P(HfStaticIOManagerTest, RejectsPositionsPastTheCache) {
  std::vector<int64_t> tokens = {1};
  int64_t start = 16; // the program's context length
  auto ids =
      from_blob(tokens.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto pos = from_blob(&start, {1}, executorch::aten::ScalarType::Long);
  EXPECT_EQ(
      io_->prepare_decode(ids, pos, decode_).error(), Error::OutOfResources);
}

INSTANTIATE_TEST_SUITE_P(
    Caches,
    HfStaticIOManagerTest,
    ::testing::Values(
        "ET_HF_STATIC_TINY_PATH",
        "ET_HF_STATIC_TINY_HALF_PATH",
        "ET_HF_STATIC_TINY_STATE_PATH"));

TEST_P(HfStaticIOManagerTest, PrepareSetsEveryCounterFromStart) {
  int64_t token = 1, start = 7;
  auto ids = from_blob(&token, {1, 1}, executorch::aten::ScalarType::Long);
  auto pos = from_blob(&start, {1}, executorch::aten::ScalarType::Long);
  auto inputs = io_->prepare_prefill(ids, pos, prefill_);
  ASSERT_TRUE(inputs.ok());
  for (size_t c = 5; c < inputs->size(); c += 3) {
    EXPECT_EQ((*inputs)[c].toTensor().const_data_ptr<int64_t>()[0], start);
    (*inputs)[c].toTensor().mutable_data_ptr<int64_t>()[0] = 100;
  }
  start = 9;
  auto decode_inputs = io_->prepare_decode(ids, pos, decode_);
  ASSERT_TRUE(decode_inputs.ok());
  for (size_t c = 5; c < decode_inputs->size(); c += 3) {
    EXPECT_EQ(
        (*decode_inputs)[c].toTensor().const_data_ptr<int64_t>()[0], start);
  }
}

TEST_P(HfStaticIOManagerTest, GeneratesAgainWithoutAnExplicitCacheReset) {
  const auto first = generate(prefill_);
  EXPECT_EQ(first, int_list(*module_, "get_test_expected"));
  EXPECT_EQ(generate(prefill_), first);
}

std::string fixture_path(const std::string& name) {
  const auto* dir = std::getenv("ET_HF_STATIC_FIXTURES_DIR");
  return std::string(dir ? dir : "") + "/" + name + ".pte";
}

class HfStaticMalformedTest : public ::testing::TestWithParam<const char*> {};

TEST_P(HfStaticMalformedTest, RejectsTheProgramAtLoad) {
  Module module(fixture_path(std::string("HfStatic_") + GetParam()));
  HfStaticIOManager io(module);
  EXPECT_EQ(io.load("prefill", "decode"), Error::InvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    Contract,
    HfStaticMalformedTest,
    ::testing::Values(
        "zero_version",
        "version_type",
        "missing_io_contract_version",
        "dtype_mismatch",
        "dtype_unknown",
        "dtype_type",
        "missing_cache_dtype",
        "layers_mismatch",
        "layers_zero",
        "layers_type",
        "missing_n_layers",
        "context_zero",
        "context_short",
        "context_type",
        "context_large",
        "missing_max_seq_len",
        "tokens_batch",
        "mask_rows",
        "mask_batch",
        "mask_heads",
        "mask_width",
        "position_rows",
        "position_batch",
        "cache_batch",
        "kv_shape",
        "method_shape",
        "method_dtype",
        "logits_rank",
        "logits_last_only",
        "logits_rows",
        "method_vocab",
        "state_count",
        "state_shape"),
    [](const ::testing::TestParamInfo<const char*>& info) {
      return info.param;
    });

TEST(HfStaticContractTest, RejectsNewerContractVersion) {
  Module module(fixture_path("HfStatic_newer_version"));
  HfStaticIOManager io(module);
  EXPECT_EQ(io.load("prefill", "decode"), Error::NotSupported);
}

TEST(HfStaticContractTest, BuildsMultipleMasksAndMarksOnlyWindowPadding) {
  Module module(fixture_path("HfStatic_windows"));
  HfStaticIOManager io(module);
  ASSERT_EQ(io.load("prefill", "decode"), Error::Ok);
  std::vector<int64_t> tokens = {7, 9};
  int64_t start = 5;
  auto ids =
      from_blob(tokens.data(), {1, 2}, executorch::aten::ScalarType::Long);
  auto pos = from_blob(&start, {1}, executorch::aten::ScalarType::Long);
  auto inputs = io.prepare_prefill(ids, pos, "prefill");
  ASSERT_TRUE(inputs.ok());
  EXPECT_EQ((*inputs)[0].toTensor().const_data_ptr<int64_t>()[3], 9);
  EXPECT_EQ((*inputs)[3].toTensor().const_data_ptr<int64_t>()[3], 8);
  for (size_t m : {1, 2}) {
    const auto& mask = (*inputs)[m].toTensor();
    const int64_t width = mask.size(3);
    for (int64_t r = 0; r < kPrefill; ++r) {
      for (int64_t c = 0; c < width; ++c) {
        const bool visible = !(m == 2 && r >= 2) && c <= start + r;
        EXPECT_EQ(mask.const_data_ptr<bool>()[r * width + c], visible);
      }
    }
  }
  auto outputs = module.execute("prefill", *inputs);
  ASSERT_TRUE(outputs.ok());
  for (size_t c = 4; c < inputs->size(); ++c) {
    EXPECT_EQ(
        (*outputs)[c - 4].toTensor().const_data_ptr(),
        (*inputs)[c].toTensor().const_data_ptr());
  }
}

TEST(HfStaticContractTest, AllWindowCachesUseMetadataContext) {
  Module module(fixture_path("HfStatic_all_window"));
  HfStaticIOManager io(module);
  ASSERT_EQ(io.load("prefill", "decode"), Error::Ok);
  int64_t token = 1, start = 10;
  auto ids = from_blob(&token, {1, 1}, executorch::aten::ScalarType::Long);
  auto pos = from_blob(&start, {1}, executorch::aten::ScalarType::Long);
  auto inputs = io.prepare_prefill(ids, pos, "prefill");
  ASSERT_TRUE(inputs.ok());
  const auto& mask = (*inputs)[1].toTensor();
  for (int64_t r = 0; r < kPrefill; ++r) {
    for (int64_t c = 0; c < mask.size(3); ++c) {
      EXPECT_EQ(mask.const_data_ptr<bool>()[r * mask.size(3) + c], r == 0);
    }
  }
}

class TinyTokenizer : public ::tokenizers::Tokenizer {
 public:
  TinyTokenizer() {
    initialized_ = true;
    vocab_size_ = 32;
    eos_tok_ = 100;
  }
  ::tokenizers::Error load(const std::string&) override {
    return ::tokenizers::Error::Ok;
  }
  ::tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t, int8_t) const override {
    if (text == "tail") {
      return std::vector<uint64_t>{7, 9};
    }
    return std::vector<uint64_t>{3, 7, 1, 9, 4, 12};
  }
  ::tokenizers::Result<std::string> decode(uint64_t, uint64_t token, bool)
      const override {
    return std::to_string(token);
  }
  ::tokenizers::Result<std::string> id_to_piece(uint64_t token) const override {
    return std::to_string(token);
  }
  ::tokenizers::Result<uint64_t> piece_to_id(
      const std::string&) const override {
    return uint64_t{0};
  }
};

class ObservedHfStaticIOManager : public HfStaticIOManager {
 public:
  explicit ObservedHfStaticIOManager(Module& module)
      : HfStaticIOManager(module) {}
  Error reset(const std::string& prefill, const std::string& decode) override {
    ++resets;
    return HfStaticIOManager::reset(prefill, decode);
  }
  int resets = 0;
};

TEST_P(HfStaticIOManagerTest, RunnerResetRegeneratesIdenticalTokens) {
  const char* path = std::getenv(GetParam());
  auto module = std::make_unique<Module>(path);
  auto io = std::make_unique<ObservedHfStaticIOManager>(*module);
  auto* observed = io.get();
  ASSERT_EQ(io->load("prefill", "decode"), Error::Ok);
  auto tokenizer = std::make_unique<TinyTokenizer>();
  auto metadata = executorch::extension::llm::get_llm_metadata(
      tokenizer.get(), module.get());
  ASSERT_TRUE(metadata.ok());
  auto stats = std::make_unique<executorch::extension::llm::Stats>();
  auto prefill =
      std::make_unique<executorch::extension::llm::TextDecoderRunner>(
          module.get(), io.get(), "prefill");
  auto decode = std::make_unique<executorch::extension::llm::TextDecoderRunner>(
      module.get(), io.get(), "decode");
  auto prefiller = std::make_unique<executorch::extension::llm::TextPrefiller>(
      prefill.get(), true, true, kPrefill);
  auto generator =
      std::make_unique<executorch::extension::llm::TextTokenGenerator>(
          tokenizer.get(),
          decode.get(),
          true,
          std::make_unique<std::unordered_set<uint64_t>>(),
          stats.get());
  executorch::extension::llm::TextLLMRunner runner(
      *metadata,
      std::move(tokenizer),
      std::move(module),
      std::move(prefill),
      std::move(prefiller),
      std::move(io),
      std::move(generator),
      std::move(stats),
      -1.0f,
      std::move(decode));
  executorch::extension::llm::GenerationConfig config;
  config.echo = false;
  config.ignore_eos = true;
  config.max_new_tokens = 6;
  config.temperature = 0.0f;
  std::vector<int64_t> first, second;
  ASSERT_EQ(
      runner.generate(
          "prompt",
          config,
          [&](const std::string& s) { first.push_back(std::stoll(s)); }),
      Error::Ok);
  runner.reset();
  EXPECT_EQ(observed->resets, 1);
  ASSERT_EQ(
      runner.generate(
          "prompt",
          config,
          [&](const std::string& s) { second.push_back(std::stoll(s)); }),
      Error::Ok);
  EXPECT_EQ(first.size(), 6);
  EXPECT_EQ(first, int_list(*module_, "get_test_expected"));
  EXPECT_EQ(second, first);
}

TEST(HfStaticRunnerTest, LoadUsesTheSelectedMethodNames) {
  auto module = std::make_unique<Module>(fixture_path("HfStaticTiny"));
  auto io = std::make_unique<HfStaticIOManager>(*module);
  auto tokenizer = std::make_unique<TinyTokenizer>();
  auto stats = std::make_unique<executorch::extension::llm::Stats>();
  auto prefill =
      std::make_unique<executorch::extension::llm::TextDecoderRunner>(
          module.get(), io.get(), "prefill");
  auto decode = std::make_unique<executorch::extension::llm::TextDecoderRunner>(
      module.get(), io.get(), "decode");
  auto prefiller = std::make_unique<executorch::extension::llm::TextPrefiller>(
      prefill.get(), true, true, kPrefill);
  auto generator =
      std::make_unique<executorch::extension::llm::TextTokenGenerator>(
          tokenizer.get(),
          decode.get(),
          true,
          std::make_unique<std::unordered_set<uint64_t>>(),
          stats.get());
  executorch::extension::llm::TextLLMRunner runner(
      {},
      std::move(tokenizer),
      std::move(module),
      std::move(prefill),
      std::move(prefiller),
      std::move(io),
      std::move(generator),
      std::move(stats),
      -1.0f,
      std::move(decode));
  EXPECT_EQ(runner.load(), Error::Ok);
}

TEST(HfStaticRunnerTest, RejectsUnknownAndMalformedContractLabels) {
  for (const auto* name : {"unknown_contract", "contract_type"}) {
    auto runner = executorch::extension::llm::create_text_llm_runner(
        fixture_path(std::string("HfStatic_") + name),
        std::make_unique<TinyTokenizer>());
    EXPECT_EQ(runner, nullptr);
  }
}

TEST(HfStaticRunnerTest, RejectsMalformedRuntimeMetadataWithoutAborting) {
  for (const auto* name :
       {"context_type",
        "layers_type",
        "version_type",
        "dtype_type",
        "newer_version"}) {
    auto runner = executorch::extension::llm::create_text_llm_runner(
        fixture_path(std::string("HfStatic_") + name),
        std::make_unique<TinyTokenizer>());
    EXPECT_EQ(runner, nullptr);
  }
}

TEST(HfStaticRunnerTest, RejectsOversizedExplicitContextBudget) {
  auto runner = executorch::extension::llm::create_text_llm_runner(
      fixture_path("HfStatic_context_budget_large"),
      std::make_unique<TinyTokenizer>());
  EXPECT_EQ(runner, nullptr);
}

TEST_P(HfStaticIOManagerTest, RunnerUsesDecodeForNearFullContextTail) {
  const char* path = std::getenv(GetParam());
  auto runner = executorch::extension::llm::create_text_llm_runner(
      path, std::make_unique<TinyTokenizer>());
  ASSERT_NE(runner, nullptr);
  executorch::extension::llm::GenerationConfig config;
  config.echo = false;
  config.ignore_eos = true;
  config.temperature = 0.0f;
  config.max_new_tokens = 8;
  ASSERT_EQ(runner->generate("prompt", config), Error::Ok); // position 13
  config.max_new_tokens = 1;
  std::vector<int64_t> tail;
  EXPECT_EQ(
      runner->generate(
          "tail",
          config,
          [&](const std::string& s) { tail.push_back(std::stoll(s)); }),
      Error::Ok);
  EXPECT_EQ(tail.size(), 1);
  // Independent decode-only prefill, with identical history.
  ASSERT_EQ(io_->reset(prefill_, decode_), Error::Ok);
  const auto prompt = int_list(*module_, "get_test_prompt");
  int64_t next = 0, start = 0;
  for (const auto token : prompt) {
    next = step(decode_, {token}, start++);
  }
  for (int i = 0; i < 7; ++i) {
    next = step(decode_, {next}, start++);
  }
  next = step(decode_, {7}, start++);
  next = step(decode_, {9}, start++);
  EXPECT_EQ(tail, std::vector<int64_t>{next});
}

struct AllocatorCounts {
  size_t allocated = 0, bound = 0, freed = 0;
  bool fallback = false;
};

class CountingAllocator : public executorch::extension::llm::IOBufferAllocator {
 public:
  explicit CountingAllocator(AllocatorCounts& counts) : counts_(counts) {}
  void* allocate(size_t nbytes, size_t alignment) override {
    ++counts_.allocated;
    if (counts_.fallback) {
      return nullptr;
    }
    void* block = ::operator new(nbytes, std::align_val_t(alignment));
    blocks_[block] = nbytes;
    return block;
  }
  void bind(void* block, void* tensor_data) override {
    ASSERT_TRUE(blocks_.count(block));
    auto offset =
        static_cast<std::byte*>(tensor_data) - static_cast<std::byte*>(block);
    EXPECT_GE(offset, 0);
    EXPECT_LT(static_cast<size_t>(offset), blocks_.at(block));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(tensor_data) % 64, 0);
    ++counts_.bound;
  }
  void deallocate(void* block) override {
    EXPECT_EQ(blocks_.erase(block), 1);
    ++counts_.freed;
    ::operator delete(block, std::align_val_t(64));
  }

 private:
  AllocatorCounts& counts_;
  std::unordered_map<void*, size_t> blocks_;
};

TEST(HfStaticContractTest, AllocatorBindsAndFreesOnlyItsOwnBlocks) {
  auto previous = executorch::extension::llm::io_buffer_allocator_factory();
  for (bool fallback : {false, true}) {
    AllocatorCounts counts;
    counts.fallback = fallback;
    executorch::extension::llm::set_io_buffer_allocator(
        [&counts] { return std::make_unique<CountingAllocator>(counts); });
    {
      Module module(fixture_path("HfStaticTiny"));
      HfStaticIOManager io(module);
      EXPECT_EQ(io.load("prefill", "decode"), Error::Ok);
      EXPECT_EQ(counts.allocated, 3);
      EXPECT_EQ(counts.bound, fallback ? 0 : 14);
    }
    EXPECT_EQ(counts.freed, fallback ? 0 : counts.allocated);
  }
  executorch::extension::llm::set_io_buffer_allocator(std::move(previous));
}

} // namespace
