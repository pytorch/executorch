/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/embedding_materializer.h>
#include <executorch/examples/models/muse-glimmer/vision/preprocess.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/tokenizer.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>

namespace llm = executorch::extension::llm;
namespace batch = llm::batching;
namespace serving = llm::serving;
namespace ext = executorch::extension;
using executorch::aten::ScalarType;
using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {
void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::shared_ptr<const llm::MuseGlimmerPreparationSpec> spec(
    ScalarType dtype = ScalarType::Half,
    bool vision = true) {
  return std::make_shared<const llm::MuseGlimmerPreparationSpec>(
      llm::MuseGlimmerPreparationSpec{
          dtype, 2, 100, 4, 201000, vision, 16, {1024, 128, 4096}});
}
std::shared_ptr<llm::MuseGlimmerPreparedInput> image_input(
    std::shared_ptr<const llm::MuseGlimmerPreparationSpec> identity,
    uint8_t pixel = 1,
    int32_t width = 56,
    int32_t height = 28) {
  return std::make_shared<llm::MuseGlimmerPreparedInput>(
      std::move(identity),
      std::vector<batch::Token>{200000, 11, 200092, 200092, 12},
      llm::MuseGlimmerRGBImage{
          std::vector<uint8_t>(width * height * 3, pixel), width, height},
      llm::MuseGlimmerImageGrid{height, width, 2},
      llm::MuseGlimmerImageSpan{2, 2});
}

class Tokenizer final : public tokenizers::Tokenizer {
 public:
  mutable size_t encodes = 0;
  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }
  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t, int8_t) const override {
    ++encodes;
    if (text == "bos")
      return std::vector<uint64_t>{200000, 11};
    std::vector<uint64_t> tokens;
    for (unsigned char c : text)
      tokens.push_back(c);
    return tokens;
  }
  tokenizers::Result<std::string> decode(uint64_t, uint64_t value, bool)
      const override {
    return std::to_string(value);
  }
  tokenizers::Result<std::string> id_to_piece(uint64_t value) const override {
    return std::to_string(value);
  }
  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override {
    return tokenizers::Error::Internal;
  }
};

serving::ModelPreparationResult prepare(
    const nlohmann::json& request,
    const serving::PromptPreparationContext& context,
    std::shared_ptr<const llm::MuseGlimmerPreparationSpec> identity) {
  auto source = llm::prepare_muse_glimmer_prompt(request, context, identity);
  if (auto* error = std::get_if<serving::ServingError>(&source))
    return *error;
  return llm::prepare_muse_glimmer_input(
      context, std::get<serving::PromptInput>(source), std::move(identity));
}

class PositionalTableFile final {
 public:
  PositionalTableFile()
      : path_((std::filesystem::temp_directory_path() /
               "muse-glimmer-pos-embed-XXXXXX")
                  .string()),
        fd_(mkstemp(path_.data())) {
    require(fd_ >= 0, "cannot create positional-table fixture");
  }
  ~PositionalTableFile() {
    close(fd_);
    std::remove(path_.c_str());
  }
  PositionalTableFile(const PositionalTableFile&) = delete;
  PositionalTableFile& operator=(const PositionalTableFile&) = delete;

  const std::string& path() const {
    return path_;
  }
  void resize(int64_t bytes) const {
    require(
        ftruncate(fd_, bytes) == 0, "cannot resize positional-table fixture");
  }

 private:
  std::string path_;
  int fd_;
};

void materialization() {
  for (auto dtype : {ScalarType::Half, ScalarType::BFloat16}) {
    auto identity = spec(dtype);
    auto backing = image_input(identity);
    auto before = backing->suffix(1);
    auto inside = backing->suffix(3);
    auto after = backing->suffix(4);
    require(
        before && inside && after && before->size() == 4 &&
            inside->size() == 2 && after->size() == 1,
        "valid suffix rejected or sized incorrectly");
    require(
        !backing->suffix(5) && !backing->suffix(6),
        "empty/OOB suffix accepted");
    require(backing->suffix(0)->size() == 5, "full suffix lost positions");
    llm::MuseGlimmerMaterializer materializer(identity);
    int encodes = 0;
    int embeds = 0;
    auto encode = [&](const llm::MuseGlimmerRGBImage&)
        -> Result<llm::PreparedMuseGlimmerImage> {
      ++encodes;
      return llm::PreparedMuseGlimmerImage{{101, 102, 103, 104}, 2, 2, 0};
    };
    auto embed =
        [&](const std::vector<int64_t>& tokens) -> Result<ext::TensorPtr> {
      ++embeds;
      auto result =
          ext::zeros({1, static_cast<int32_t>(tokens.size()), 2}, dtype);
      auto* data = static_cast<uint16_t*>(result->mutable_data_ptr());
      for (size_t i = 0; i < tokens.size(); ++i)
        data[2 * i] = data[2 * i + 1] = static_cast<uint16_t>(tokens[i]);
      return result;
    };
    auto text_tail =
        materializer.materialize({{1, false, 0, 1, after, 0}}, encode, embed);
    require(
        text_tail.ok() && encodes == 0 &&
            static_cast<const uint16_t*>((*text_tail)->const_data_ptr())[0] ==
                12,
        "text suffix encoded unused image or selected wrong token");
    embeds = 0;
    auto raw = std::make_shared<const std::vector<batch::Token>>(
        std::vector<batch::Token>{99});
    auto a = materializer.materialize(
        {{1, false, 0, 3, batch::PreparedInputPtr(backing), 0}}, encode, embed);
    require(a.ok(), "first slice failed");
    const auto* ad = static_cast<const uint16_t*>((*a)->const_data_ptr());
    require(
        ad[2] == 11 && ad[4] == 101 && ad[5] == 102,
        "text/image first splice wrong");
    auto b = materializer.materialize(
        {{1, true, 3, 2, batch::PreparedInputPtr(backing), 0},
         {2, true, 0, 1, raw, 50}},
        encode,
        embed);
    require(b.ok(), "split image plus raw feedback failed");
    const auto* bd = static_cast<const uint16_t*>((*b)->const_data_ptr());
    require(
        bd[0] == 103 && bd[1] == 104 && bd[2] == 12 && bd[4] == 99,
        "offset-based scatter wrong");
    require(
        encodes == 1 && embeds == 2,
        "image not once or text not gathered once per step");
    require(ad[4] == 101, "returned physical tensor was borrowed");
    auto repeated = materializer.materialize(
        {{3, false, 2, 2, batch::PreparedInputPtr(backing), 0}}, encode, embed);
    require(
        repeated.ok() && encodes == 1 && embeds == 2,
        "image-only replay used embed/cursor");
    auto other = image_input(identity);
    auto pair = materializer.materialize(
        {{1, false, 2, 1, batch::PreparedInputPtr(backing), 0},
         {2, false, 3, 1, batch::PreparedInputPtr(other), 0}},
        encode,
        embed);
    require(
        pair.ok() && encodes == 2,
        "packed independent backing encoding incorrect");
    auto nested = before->suffix(1);
    auto late = backing->suffix(3);
    require(
        nested && late && nested->size() == 3 && late->size() == 2,
        "nested or post-encode suffix size wrong");
    backing.reset();
    for (const auto& view : {before, inside, after, nested, late}) {
      require(
          view->kind() == llm::MuseGlimmerPreparedInput::kind_tag() &&
              view->last_prompt_token() == 12 && view->prefix_identity() &&
              view->prefix_identity()->size() == view->size(),
          "suffix type, tail, or identity changed");
    }
    auto slices = materializer.materialize(
        {{1, false, 1, 1, before, 30},
         {2, false, 0, 1, inside, 40},
         {3, false, 1, 1, nested, 50},
         {4, false, 0, 1, late, 60}},
        encode,
        embed);
    require(
        slices.ok() && encodes == 2 && embeds == 2,
        "suffix cache not shared after original destruction");
    const auto* sd = static_cast<const uint16_t*>((*slices)->const_data_ptr());
    require(
        sd[0] == 101 && sd[2] == 103 && sd[4] == 103 && sd[6] == 103,
        "view-relative or nested nonzero offset splice wrong");
    require(
        !materializer.materialize({{1, false, 1, 2, inside, 0}}, encode, embed)
             .ok(),
        "suffix-local bounds ignored");
  }
}

void preparation() {
  Tokenizer tokenizer;
  auto identity = spec();
  serving::PromptPreparationContext context{
      tokenizer, 99, [] { return false; }};
  auto text = prepare({{"prompt", "bos"}}, context, identity);
  auto text_prompt = std::get<batch::PreparedInputPtr>(text)->prefix_identity();
  require(
      *std::get<batch::TokenSpan>(text_prompt->spans[0]).tokens ==
          std::vector<uint64_t>({200000, 11}),
      "BOS duplicated");
  // Valid 1x1 PNG, decoded under independent byte/pixel limits.
  nlohmann::json image{
      {"encoding", "base64"},
      {"mime_type", "image/png"},
      {"data",
       "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aY1sAAAAASUVORK5CYII="}};
  auto prepared =
      prepare({{"prompt", "a<img>b"}, {"image", image}}, context, identity);
  auto opaque = std::get<batch::PreparedInputPtr>(prepared);
  require(
      opaque->last_prompt_token() == 98 && opaque->size() == 4,
      "decoder layout/previous token incorrect");
  auto end_image =
      prepare({{"prompt", "a<img>"}, {"image", image}}, context, identity);
  require(
      std::get<batch::PreparedInputPtr>(end_image)->last_prompt_token() ==
          200092,
      "image tail must preserve the real patch layout token");
  context.max_prompt_positions = 3;
  require(
      std::holds_alternative<serving::ServingError>(prepare(
          {{"prompt", "a<img>b"}, {"image", image}}, context, identity)),
      "expanded limit ignored");
  context.max_prompt_positions = 99;
  image["data"] = "not-base64";
  require(
      std::holds_alternative<serving::ServingError>(
          llm::prepare_muse_glimmer_prompt(
              {{"prompt", "<img>"}, {"image", image}}, context, identity)),
      "malformed image accepted");
  context.cancelled = [] { return true; };
  require(
      std::holds_alternative<serving::ServingError>(
          llm::prepare_muse_glimmer_prompt(
              {{"prompt", "a"}}, context, identity)),
      "cancel ignored");
}

void mixed_identity() {
  auto identity = spec();
  auto original = image_input(identity);
  auto prefix = original->prefix_identity();
  require(
      prefix && prefix->size() == 5 && prefix->spans.size() == 3,
      "mixed identity does not cover exact layout");
  // Independent SHA-256 fixture for the canonical 56x28 image.
  const batch::ContentKey expected_key = {
      0xfc, 0x2b, 0x2a, 0x3a, 0xe1, 0x5e, 0xe2, 0xe4, 0x3e, 0x9a, 0x21,
      0x0b, 0xad, 0xe5, 0x6f, 0x16, 0x54, 0x16, 0x5c, 0xab, 0x11, 0x33,
      0x0a, 0x55, 0x5d, 0x62, 0x07, 0x8d, 0x11, 0x08, 0xe7, 0xa7};
  require(
      std::get<batch::OpaqueSpan>(prefix->spans[1]).key == expected_key,
      "canonical image digest changed");
  auto same = image_input(identity)->prefix_identity();
  require(
      batch::common_prefix(*prefix, *same) == 5,
      "separate equal-pixel allocations have different identity");
  for (auto changed :
       {image_input(identity, 2), image_input(identity, 1, 28, 56)}) {
    auto different = changed->prefix_identity();
    require(
        batch::common_prefix(*prefix, *different) == 2 &&
            std::get<batch::OpaqueSpan>(prefix->spans[1]).key !=
                std::get<batch::OpaqueSpan>(different->spans[1]).key,
        "changed pixels/geometry retained opaque key or lost text prefix");
  }
  auto suffix = original->suffix(3)->prefix_identity();
  const auto& image = std::get<batch::OpaqueSpan>(suffix->spans[0]);
  const auto& tail = std::get<batch::TokenSpan>(suffix->spans[1]);
  require(
      suffix->size() == 2 && suffix->spans.size() == 2 && image.offset == 1 &&
          image.size == 1 &&
          image.key == std::get<batch::OpaqueSpan>(prefix->spans[1]).key &&
          tail.size == 1 && (*tail.tokens)[tail.offset] == 12,
      "partial image identity lost content-relative offset or text tail");
  auto fresh_spec = spec();
  std::weak_ptr<const llm::MuseGlimmerPreparationSpec> weak_spec = fresh_spec;
  auto owner = image_input(fresh_spec);
  auto retained = owner->prefix_identity();
  auto retained_suffix = owner->suffix(3)->prefix_identity();
  auto tokens = std::get<batch::TokenSpan>(retained->spans[0]).tokens;
  owner.reset();
  fresh_spec.reset();
  require(weak_spec.expired(), "identity retained execution backing/spec");
  require(
      *tokens == std::vector<batch::Token>({200000, 11, 200092, 200092, 12}) &&
          batch::common_prefix(*retained, *prefix) == 5 &&
          batch::common_prefix(*retained_suffix, *suffix) == 2,
      "identity tokens or digest died with execution backing");
}

void source_and_model_preparation() {
  Tokenizer tokenizer;
  auto identity = spec();
  serving::PromptPreparationContext context{
      tokenizer, 99, [] { return false; }};
  // RGB8 PNG, 2x1: red, then (0,255,127); no image fixture files.
  nlohmann::json png{
      {"encoding", "base64"},
      {"mime_type", "image/png"},
      {"data",
       "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAD0lEQVR4nGP4"
       "z8DA8L8eAAh+An41RIHhAAAAAElFTkSuQmCC"}};
  const nlohmann::json request{{"prompt", "bos<img>x"}, {"image", png}};
  auto source = std::get<serving::PromptInput>(
      llm::prepare_muse_glimmer_prompt(request, context, identity));
  require(
      tokenizer.encodes == 0 && source.segments.size() == 3 &&
          source.segments[0].get_text() == "bos" &&
          source.segments[2].get_text() == "x",
      "source stage tokenized or reordered text");
  const auto& decoded = source.segments[1].get_image();
  require(
      decoded.width() == 2 && decoded.height() == 1 &&
          decoded.channels() == 3 &&
          decoded.get_uint8_data() ==
              std::vector<uint8_t>({255, 0, 0, 255, 0, 127}),
      "source PNG did not decode to CHW");
  auto direct = source;
  direct.segments[1] = llm::make_image_input(
      llm::Image(std::vector<uint8_t>{255, 0, 0, 255, 0, 127}, 2, 1, 3));
  batch::PrefixIdentityPtr previous;
  int encodes = 0;
  for (const auto& input : {source, direct}) {
    auto prepared = std::get<batch::PreparedInputPtr>(
        llm::prepare_muse_glimmer_input(context, input, identity));
    auto prefix = prepared->prefix_identity();
    require(
        !previous ||
            batch::common_prefix(*previous, *prefix) == prepared->size(),
        "PNG and direct CHW disagree on identity");
    previous = prefix;
    auto encoded = llm::MuseGlimmerMaterializer(identity).materialize(
        {{1, false, 2, 1, prepared, 0}},
        [&](const llm::MuseGlimmerRGBImage& rgb)
            -> Result<llm::PreparedMuseGlimmerImage> {
          ++encodes;
          require(
              rgb.width == 2 && rgb.height == 1 &&
                  rgb.rgb == std::vector<uint8_t>({255, 0, 0, 0, 255, 127}),
              "model preparation did not restore HWC");
          return llm::PreparedMuseGlimmerImage{{7, 8}, 1, 2, 0};
        },
        [](const auto&) -> Result<ext::TensorPtr> { return Error::Internal; });
    require(encoded.ok(), "colored image-only materialization failed");
  }
  require(encodes == 2, "independent preparations shared embedding cache");
  auto ids = std::get<serving::PromptInput>(llm::prepare_muse_glimmer_prompt(
      {{"prompt_segments", {{{"ids", {200000, 11}}}, {{"text", "x"}}}}},
      context,
      identity));
  auto normalized = std::get<batch::PreparedInputPtr>(
                        llm::prepare_muse_glimmer_input(context, ids, identity))
                        ->prefix_identity();
  require(
      ids.segments[0].get_tokens() == std::vector<uint64_t>({200000, 11}) &&
          *std::get<batch::TokenSpan>(normalized->spans[0]).tokens ==
              std::vector<uint64_t>({200000, 11, 120}),
      "ordered IDs/text or BOS normalization changed");
  auto rejects = [&](const serving::PromptInput& input) {
    return std::holds_alternative<serving::ServingError>(
        llm::prepare_muse_glimmer_input(context, input, identity));
  };
  for (auto bad :
       {llm::Image{},
        llm::Image(std::vector<uint8_t>{1}, 1, 1, 3),
        llm::Image(std::vector<uint8_t>(4), 1, 1, 4),
        llm::Image(std::vector<float>(3), 1, 1, 3),
        llm::Image(std::vector<uint8_t>(129 * 3), 129, 1, 3),
        llm::Image(std::vector<uint8_t>(65 * 65 * 3), 65, 65, 3)}) {
    require(
        rejects({{llm::make_image_input(bad)}}),
        "direct malformed image bypassed validation");
  }
  for (auto invalid :
       {serving::PromptInput{{llm::make_audio_input(llm::Audio{})}},
        serving::PromptInput{{direct.segments[1], direct.segments[1]}},
        serving::PromptInput{{llm::make_token_input({201000})}},
        serving::PromptInput{{llm::make_token_input({200092})}}}) {
    require(
        rejects(invalid), "direct modality/image/token validation bypassed");
  }
  for (auto invalid :
       {nlohmann::json{{"prompt_segments", {{{"audio", "x"}}}}},
        nlohmann::json{{"prompt_segments", {{{"ids", {-1}}}}}},
        nlohmann::json{{"prompt", "<img><img>"}, {"image", png}}}) {
    require(
        std::holds_alternative<serving::ServingError>(
            llm::prepare_muse_glimmer_prompt(invalid, context, identity)),
        "source validation bypassed");
  }
  auto text_only = spec(ScalarType::Half, false);
  require(
      std::holds_alternative<serving::ServingError>(
          llm::prepare_muse_glimmer_prompt(request, context, text_only)) &&
          std::holds_alternative<serving::ServingError>(
              llm::prepare_muse_glimmer_input(context, direct, text_only)),
      "vision-disabled source/model accepted image");
  context.max_prompt_positions = 3;
  require(rejects(direct), "direct expanded context limit ignored");
  context.max_prompt_positions = 99;
  context.cancelled = [] { return true; };
  require(rejects(direct), "direct cancellation ignored");
}

class ServingExecutor final : public batch::testing::FakeExecutor {
 public:
  std::shared_ptr<const llm::MuseGlimmerPreparationSpec> identity = spec();
  std::atomic<size_t> last_size{0}, clones{0};
  std::atomic<batch::Token> last_token{0};
  bool accepts(const batch::PreparedInput& input) const override {
    return input.kind() == llm::MuseGlimmerPreparedInput::kind_tag() &&
        static_cast<const llm::MuseGlimmerPreparedInput&>(input).compatible(
            *identity);
  }
  std::optional<batch::SessionId> clone(batch::SessionId, batch::Position)
      override {
    // Only routing is simulated here; materialization is checked separately.
    ++clones;
    return open_session();
  }
  bool execute(const batch::BatchInput& input, batch::BatchOutput& output)
      override {
    for (const auto& slice : input.inputs) {
      if (auto* ptr = std::get_if<batch::PreparedInputPtr>(&slice.payload)) {
        const auto& prepared = *ptr;
        if (!prepared || !accepts(*prepared) || !prepared->prefix_identity() ||
            prepared->prefix_identity()->size() != prepared->size() ||
            slice.offset + slice.size > prepared->size())
          return false;
        last_size = prepared->size();
        last_token = prepared->last_prompt_token();
      }
    }
    return FakeExecutor::execute(input, output);
  }
};

void serving_reuse() {
  ServingExecutor executor;
  Tokenizer tokenizer;
  serving::ServingRuntimeConfig config;
  config.max_sessions = 2;
  config.max_context_length = 100;
  config.prefix_cache_capacity = 1;
  serving::ServingRuntime runtime(
      executor,
      batch::DecodeFirstScheduler::create(4, 1, 3),
      tokenizer,
      config,
      [&](const auto& context, const auto& input) {
        return llm::prepare_muse_glimmer_input(
            context, input, executor.identity);
      });
  serving::PromptInput prompt{
      {llm::make_text_input("bos"),
       llm::make_image_input(
           llm::Image(std::vector<uint8_t>(56 * 28 * 3, 1), 56, 28, 3)),
       llm::make_text_input("b")}};
  auto generate = [&](const char* key, serving::PromptInput input) {
    auto terminal = std::make_shared<std::promise<serving::TerminalEvent>>();
    auto result = terminal->get_future();
    serving::GenerationOptions options;
    options.max_new_tokens = 1;
    options.seed = 42;
    auto admitted = runtime.generate(
        key,
        std::move(input),
        options,
        [terminal](serving::GenerationEvent event) {
          if (auto* end = std::get_if<serving::TerminalEvent>(&event))
            terminal->set_value(*end);
        });
    require(
        std::holds_alternative<serving::RequestHandle>(admitted),
        "generation rejected");
    require(
        result.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
        "generation timed out");
    std::get<serving::RequestHandle>(admitted).wait();
    auto end = result.get();
    require(
        !end.error && end.finish_reason == serving::FinishReason::Length,
        "fake MG generation failed");
    return end.stats;
  };
  auto first = generate("named", prompt);
  require(
      first.prompt_tokens == 5 && first.reused_prompt_tokens == 0 &&
          first.generated_token_ids && first.generated_token_ids->size() == 1,
      "initial mixed generation failed");
  auto continued = prompt;
  continued.segments.push_back(
      llm::make_token_input(*first.generated_token_ids));
  continued.segments.push_back(llm::make_text_input("c"));
  auto next = generate("named", std::move(continued));
  require(
      next.reused_prompt_tokens == 5 && next.prefilled_prompt_tokens == 2 &&
          next.session_reset_reason == "exact_prefix" && executor.clones == 1 &&
          executor.last_size == 1 && executor.last_token == 99,
      "named mixed continuation did not execute MG suffix");
  auto hit = generate("snapshot", prompt);
  require(
      hit.reused_prompt_tokens == 4 && hit.prefilled_prompt_tokens == 1 &&
          executor.clones == 3 && executor.last_size == 1 &&
          executor.last_token == 98,
      "mixed snapshot did not reuse image identity and execute MG suffix");
}

void unsafe_inputs() {
  auto identity = spec();
  batch::PreparedInputPtr backing = image_input(identity);
  llm::MuseGlimmerMaterializer materializer(identity);
  require(!image_input(nullptr)->compatible(*identity), "null spec accepted");
  llm::MuseGlimmerPreparedInput malformed(
      identity,
      std::vector<batch::Token>{200000, 11, 200092, 12, 12},
      llm::MuseGlimmerRGBImage{std::vector<uint8_t>(56 * 28 * 3, 1), 56, 28},
      llm::MuseGlimmerImageGrid{28, 56, 2},
      llm::MuseGlimmerImageSpan{2, 2});
  require(!malformed.compatible(*identity), "malformed patch layout accepted");
  int encodes = 0;
  auto encode = [&](const llm::MuseGlimmerRGBImage& image)
      -> Result<llm::PreparedMuseGlimmerImage> {
    require(!image.rgb.empty(), "RGB missing on retry");
    ++encodes;
    return llm::PreparedMuseGlimmerImage{{1, 2, 3, 4}, 2, 2, 0};
  };
  auto embed =
      [](const std::vector<int64_t>& tokens) -> Result<ext::TensorPtr> {
    return ext::zeros(
        {1, static_cast<int32_t>(tokens.size()), 2}, ScalarType::Half);
  };
  require(
      !materializer.materialize({{1, true, 4, 2, backing, 0}}, encode, embed)
           .ok(),
      "OOB slice accepted");
  require(
      !materializer.materialize({{1, true, 0, 5, backing, 0}}, encode, embed)
           .ok(),
      "physical width exceeded");
  require(
      !llm::MuseGlimmerMaterializer(spec())
           .materialize({{1, false, 0, 2, backing, 0}}, encode, embed)
           .ok(),
      "wrong executor materialized");
  auto bad_embed =
      [](const std::vector<int64_t>& tokens) -> Result<ext::TensorPtr> {
    return ext::zeros(
        {1, static_cast<int32_t>(tokens.size()), 2}, ScalarType::Float);
  };
  require(
      !materializer
           .materialize({{1, false, 0, 2, backing, 0}}, encode, bad_embed)
           .ok(),
      "wrong dtype accepted");
  uint16_t strided_data[4] = {};
  auto strided_embed =
      [&](const std::vector<int64_t>&) -> Result<ext::TensorPtr> {
    return ext::make_tensor_ptr(
        {1, 2, 2}, strided_data, {0, 2, 1}, {4, 1, 2}, ScalarType::Half);
  };
  require(
      !materializer
           .materialize({{1, false, 0, 2, backing, 0}}, encode, strided_embed)
           .ok(),
      "strided embedding output accepted as contiguous storage");
  auto bad_encode = [](const llm::MuseGlimmerRGBImage& image)
      -> Result<llm::PreparedMuseGlimmerImage> {
    require(!image.rgb.empty(), "RGB missing before failed encode");
    return llm::PreparedMuseGlimmerImage{{1, 2}, 1, 2, 0};
  };
  require(
      !materializer
           .materialize({{1, false, 2, 1, backing, 0}}, bad_encode, embed)
           .ok(),
      "wrong vision row count accepted");
  require(
      materializer
              .materialize(
                  {{1, false, 0, 1, backing->suffix(2), 0}}, encode, embed)
              .ok() &&
          encodes == 1,
      "vision retry failed");
  require(
      materializer.materialize({{1, false, 2, 1, backing, 0}}, encode, embed)
              .ok() &&
          encodes == 1,
      "vision replay failed or re-encoded");

  namespace vision = executorch::examples::muse_glimmer_vision;
  const int64_t elements = static_cast<int64_t>(vision::kPosGrid) *
      vision::kPosGrid * vision::kLatent;
  const int64_t bytes = elements * sizeof(float);
  PositionalTableFile file;
  for (int64_t size :
       {bytes - static_cast<int64_t>(sizeof(float)), bytes + 1}) {
    file.resize(size);
    bool rejected = false;
    try {
      (void)vision::load_pos_embed_table(file.path());
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "malformed positional-table byte length accepted");
  }
  file.resize(bytes);
  // Invalid dimensions must fail before executing this non-program file.
  ext::Module module(file.path());
  std::mutex execution_mutex;
  llm::MuseGlimmerVisionRuntimeConfig config;
  config.module = &module;
  config.execution_mutex = &execution_mutex;
  config.pos_embed_path = file.path();
  config.expected_hidden_dim = 2;
  config.max_image_dimension = 2;
  config.max_image_pixels = 2;
  llm::MuseGlimmerVisionRuntime runtime(config);
  require(
      runtime.prepare_decoded_image(nullptr, 1, 1).error() ==
          Error::InvalidArgument,
      "null decoded image accepted");
  const uint8_t rgb[12] = {};
  for (const auto& dimensions :
       std::vector<std::pair<int32_t, int32_t>>{{0, 1}, {3, 1}, {2, 2}}) {
    require(
        runtime.prepare_decoded_image(rgb, dimensions.first, dimensions.second)
                .error() == Error::InvalidArgument,
        "invalid decoded dimensions reached preprocessing");
  }
}
} // namespace

int main() {
  try {
    materialization();
    preparation();
    mixed_identity();
    source_and_model_preparation();
    serving_reuse();
    unsafe_inputs();
    std::cout
        << "Muse Glimmer batching preparation/materialization tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
