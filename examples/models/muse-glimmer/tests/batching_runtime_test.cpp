/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/embedding_materializer.h>
#include <executorch/examples/models/muse-glimmer/vision/preprocess.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/tokenizer.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
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
    ScalarType dtype = ScalarType::Half) {
  return std::make_shared<const llm::MuseGlimmerPreparationSpec>(
      llm::MuseGlimmerPreparationSpec{
          dtype, 2, 100, 4, 201000, true, 16, {1024, 128, 4096}});
}
std::shared_ptr<llm::MuseGlimmerPreparedInput> image_input(
    std::shared_ptr<const llm::MuseGlimmerPreparationSpec> identity) {
  return std::make_shared<llm::MuseGlimmerPreparedInput>(
      std::move(identity),
      std::vector<batch::Token>{200000, 11, 200092, 200092, 12},
      llm::MuseGlimmerRGBImage{std::vector<uint8_t>(56 * 28 * 3, 1), 56, 28},
      llm::MuseGlimmerImageGrid{28, 56, 2},
      llm::MuseGlimmerImageSpan{2, 2});
}

class Tokenizer final : public tokenizers::Tokenizer {
 public:
  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }
  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t, int8_t) const override {
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
  }
}

void preparation() {
  Tokenizer tokenizer;
  auto identity = spec();
  serving::PromptPreparationContext context{
      tokenizer, 99, [] { return false; }};
  auto text =
      llm::prepare_muse_glimmer_prompt({{"prompt", "bos"}}, context, identity);
  auto& text_prompt =
      std::get<serving::PromptInput>(std::get<serving::GenerationPrompt>(text));
  require(
      text_prompt.segments[0].get_tokens() ==
          std::vector<uint64_t>({200000, 11}),
      "BOS duplicated");
  // Valid 1x1 PNG, decoded under independent byte/pixel limits.
  nlohmann::json image{
      {"encoding", "base64"},
      {"mime_type", "image/png"},
      {"data",
       "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aY1sAAAAASUVORK5CYII="}};
  auto prepared = llm::prepare_muse_glimmer_prompt(
      {{"prompt", "a<img>b"}, {"image", image}}, context, identity);
  auto& opaque = std::get<serving::PreparedPromptInput>(
      std::get<serving::GenerationPrompt>(prepared));
  require(
      opaque.previous_token == 98 && opaque.input->size() == 4,
      "decoder layout/previous token incorrect");
  auto end_image = llm::prepare_muse_glimmer_prompt(
      {{"prompt", "a<img>"}, {"image", image}}, context, identity);
  require(
      std::get<serving::PreparedPromptInput>(
          std::get<serving::GenerationPrompt>(end_image))
              .previous_token == 200092,
      "image tail must preserve the real patch layout token");
  context.max_prompt_positions = 3;
  require(
      std::holds_alternative<serving::ServingError>(
          llm::prepare_muse_glimmer_prompt(
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

void unsafe_inputs() {
  auto identity = spec();
  batch::PreparedInputPtr backing = image_input(identity);
  llm::MuseGlimmerMaterializer materializer(identity);
  auto encode = [](const llm::MuseGlimmerRGBImage&)
      -> Result<llm::PreparedMuseGlimmerImage> {
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
  auto bad_encode = [](const llm::MuseGlimmerRGBImage&)
      -> Result<llm::PreparedMuseGlimmerImage> {
    return llm::PreparedMuseGlimmerImage{{1, 2}, 1, 2, 0};
  };
  require(
      !materializer
           .materialize({{1, false, 2, 1, backing, 0}}, bad_encode, embed)
           .ok(),
      "wrong vision row count accepted");

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
    unsafe_inputs();
    std::cout
        << "Muse Glimmer batching preparation/materialization tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
