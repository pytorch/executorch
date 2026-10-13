/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * C API implementation over the Exynos LLM runner. Keeps the model(s) loaded
 * across generate() calls so consecutive inferences do not re-prepare.
 */

#include <executorch/examples/samsung/executor_runner/ennllm.h>

#include <executorch/runtime/platform/log.h>

#include <executorch/examples/samsung/executor_runner/llm/runner/runner.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

namespace llm = ::executorch::extension::llm;

struct enn_runner {
  std::unique_ptr<llm::IRunner> impl;
};

namespace {

std::optional<std::string> opt_or_null(const char* s) {
  if (s == nullptr || *s == '\0') {
    return std::nullopt;
  }
  return std::string(s);
}

std::string str_or_empty(const char* s) {
  return s == nullptr ? std::string() : std::string(s);
}

} // namespace

extern "C" {

enn_runner* enn_llm_create(
    const char* model_path,
    const char* data_path,
    const char* tokenizer_path,
    const char* pos_embedding_path,
    const char* config_path,
    const char* vision_model_path) {
  if (model_path == nullptr || tokenizer_path == nullptr ||
      config_path == nullptr) {
    ET_LOG(Error, "enn_llm_create: model/tokenizer/config path required");
    return nullptr;
  }

  example::RunnerPaths paths{
      .model_path = model_path,
      .tokenizer_path = tokenizer_path,
      .pos_embedding_path = opt_or_null(pos_embedding_path),
      .data_path = opt_or_null(data_path),
      .vision_model_path = opt_or_null(vision_model_path)};

  example::RunnerConfig cfg{.config_path = config_path};

  auto runner = example::create_llm_runner(paths, std::move(cfg));
  if (runner == nullptr) {
    ET_LOG(Error, "enn_llm_create: failed to create LLM runner");
    return nullptr;
  }

  return new enn_runner{std::move(runner)};
}

void enn_llm_destroy(enn_runner* runner) {
  delete runner;
}

int32_t enn_llm_generate(
    enn_runner* runner,
    const char* prompt,
    const char* image_path,
    const char* prompt_before_image,
    int32_t seq_len,
    float temperature,
    int32_t num_bos,
    int32_t num_eos,
    int32_t verbose,
    enn_token_cb on_token,
    void* userdata) {
  if (runner == nullptr || prompt == nullptr || *prompt == '\0') {
    ET_LOG(Error, "enn_llm_generate: invalid argument");
    return ENN_ERR_INVALID_ARG;
  }

  auto* lrunner = dynamic_cast<example::LLMRunner*>(runner->impl.get());
  if (lrunner == nullptr) {
    ET_LOG(Error, "enn_llm_generate: unexpected runner type");
    return ENN_ERR_INVALID_ARG;
  }

  // Start each generation from a clean KV-cache.
  lrunner->reset();

  // Image inputs are threaded through the runner's config (read by generate).
  lrunner->get_config().image_path = str_or_empty(image_path);
  lrunner->get_config().prompt_before_image = str_or_empty(prompt_before_image);
  lrunner->get_config().verbose = verbose != 0;

  llm::GenerationConfig gen_cfg{
      .seq_len = seq_len >= 0 ? seq_len : -1,
      .temperature = temperature,
      .num_bos = num_bos,
      .num_eos = num_eos};

  auto err = runner->impl->generate(
      prompt,
      gen_cfg,
      [on_token, userdata](const std::string& piece) {
        if (on_token != nullptr) {
          on_token(piece.data(), static_cast<int32_t>(piece.size()), userdata);
        }
      },
      {});
  if (err != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "enn_llm_generate: generation failed");
    return ENN_ERR_GENERATE;
  }
  return ENN_OK;
}

} // extern "C"
