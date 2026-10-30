/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/runners/batching_bootstrap.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

DEFINE_string(
    prompt,
    "The meaning of life is",
    "Prompt text; one <img> marker for an image");
DEFINE_string(image_path, "", "Optional local JPEG/PNG image");
DEFINE_int32(max_new_tokens, 128, "Maximum generated tokens");
DEFINE_double(temperature, 0, "Sampling temperature");
DEFINE_double(top_p, 1, "Nucleus sampling probability");
DEFINE_int32(top_k, 0, "Top-k sampling; zero disables");

namespace {
std::string base64(const std::vector<uint8_t>& bytes) {
  constexpr char alphabet[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((bytes.size() + 2) / 3 * 4);
  for (size_t i = 0; i < bytes.size(); i += 3) {
    const uint32_t value = (static_cast<uint32_t>(bytes[i]) << 16) |
        (i + 1 < bytes.size() ? static_cast<uint32_t>(bytes[i + 1]) << 8 : 0) |
        (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    out.push_back(alphabet[(value >> 18) & 63]);
    out.push_back(alphabet[(value >> 12) & 63]);
    out.push_back(i + 1 < bytes.size() ? alphabet[(value >> 6) & 63] : '=');
    out.push_back(i + 2 < bytes.size() ? alphabet[value & 63] : '=');
  }
  return out;
}
} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    namespace llm = executorch::extension::llm;
    auto batching_runtime = llm::create_muse_glimmer_batching_runtime();
    nlohmann::json request{{"prompt", FLAGS_prompt}};
    if (!FLAGS_image_path.empty()) {
      std::ifstream file(FLAGS_image_path, std::ios::binary | std::ios::ate);
      const auto count = file.tellg();
      if (!file || count <= 0 ||
          static_cast<uint64_t>(count) > batching_runtime->backend.preparation
                                             ->image_limits.max_encoded_bytes) {
        throw std::runtime_error("image file missing, empty, or oversized");
      }
      std::vector<uint8_t> bytes(static_cast<size_t>(count));
      file.seekg(0);
      file.read(reinterpret_cast<char*>(bytes.data()), count);
      if (!file)
        throw std::runtime_error("could not read image file");
      request["image"] = {
          {"encoding", "base64"},
          {"mime_type", bytes[0] == 0x89 ? "image/png" : "image/jpeg"},
          {"data", base64(bytes)}};
    }
    llm::serving::GenerationOptions options;
    options.max_new_tokens = FLAGS_max_new_tokens;
    options.sampling = {
        static_cast<float>(FLAGS_temperature),
        static_cast<float>(FLAGS_top_p),
        FLAGS_top_k};
    // Opening waits for executor initialization without running model methods.
    if (auto error =
            batching_runtime->runtime->open_session_async("solo").get()) {
      throw std::runtime_error(error->message);
    }
    llm::serving::PromptPreparation prepare =
        [request = std::move(request),
         spec = batching_runtime->backend.preparation](
            const llm::serving::PromptPreparationContext& context) {
          return llm::prepare_muse_glimmer_prompt(request, context, spec);
        };
    auto generated = batching_runtime->runtime->generate(
        "solo",
        std::move(prepare),
        options,
        [](llm::serving::GenerationEvent event) {
          if (auto* text = std::get_if<llm::serving::TextEvent>(&event))
            std::cout << text->text << std::flush;
        });
    if (const auto* error =
            std::get_if<llm::serving::ServingError>(&generated)) {
      throw std::runtime_error(error->message);
    }
    auto& handle = std::get<llm::serving::RequestHandle>(generated);
    handle.wait();
    if (auto error = handle.error())
      throw std::runtime_error(error->message);
    std::cout << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "batching runner failed: " << error.what() << '\n';
    return 1;
  }
}
