/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/examples/samsung/executor_runner/llm/runner/runner.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

namespace llm = ::executorch::extension::llm;

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(data_path, "", "Data file for the model.");

DEFINE_string(pos_embedding_path, "", "rotary embedding stuff.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(config_path, "", "Path to the model configuration JSON file.");

DEFINE_string(vision_model_path, "", "Path to the Gemma 4 vision encoder PTE.");

DEFINE_string(image_path, "", "Path to an input image (Gemma 4 multimodal).");

DEFINE_string(
    prompt_before_image,
    "",
    "Text placed before the image in the user turn (Gemma 4 multimodal).");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_double(
    temperature,
    0.8,
    "Temperature; Default is 0.8. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    cpu_threads,
    -1,
    "Number of CPU threads for inference. Defaults to -1, which implies we'll use a heuristic to derive the # of performant cores for a specific device.");

DEFINE_int32(
    num_bos,
    0,
    "Number of BOS tokens to prepend to the prompt. Defaults to 0. If > 0, the prompt will be prepended with BOS tokens. This is useful for models that expect one or more BOS token at the start.");

DEFINE_int32(
    num_eos,
    0,
    "Number of EOS tokens to append to the prompt. Defaults to 0. If > 0, the prompt will be appended with EOS tokens. This is useful for models that expect one or more EOS token at the end.");

DEFINE_bool(is_warmup, false, "Whether to run a warmup run.");

DEFINE_bool(is_decode_only, false, "Whether to run with decode method only.");

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const char* model_path = FLAGS_model_path.c_str();

  std::optional<std::string> data_path = std::nullopt;
  if (!FLAGS_data_path.empty()) {
    data_path = FLAGS_data_path.c_str();
  }
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  std::optional<std::string> pos_embedding_path = std::nullopt;
  if (!FLAGS_pos_embedding_path.empty()) {
    pos_embedding_path = FLAGS_pos_embedding_path;
  }

  const char* config_path = FLAGS_config_path.c_str();
  std::optional<std::string> vision_model_path = std::nullopt;
  if (!FLAGS_vision_model_path.empty()) {
    vision_model_path = FLAGS_vision_model_path;
  }

  const char* prompt = FLAGS_prompt.c_str();

  if (FLAGS_prompt.empty()) {
    ET_LOG(Error, "Input prompt cannot be empty.");
    return static_cast<int32_t>(executorch::runtime::Error::InvalidArgument);
  }
  ET_LOG(Info, "Input prompt message: %s", prompt);

  float temperature = static_cast<float>(FLAGS_temperature);
  int32_t seq_len = FLAGS_seq_len;
  int32_t cpu_threads = FLAGS_cpu_threads;

  bool is_warmup = FLAGS_is_warmup;
  bool is_decode_only = FLAGS_is_decode_only;

  int num_bos = FLAGS_num_bos;
  int num_eos = FLAGS_num_eos;

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  example::RunnerPaths runner_paths{
      .model_path = model_path,
      .tokenizer_path = tokenizer_path,
      .pos_embedding_path = pos_embedding_path,
      .data_path = data_path,
      .vision_model_path = vision_model_path};
  example::RunnerConfig runner_config{
      .config_path = config_path,
      .is_decode_only = is_decode_only,
      .verbose = true,
      .image_path = FLAGS_image_path,
      .prompt_before_image = FLAGS_prompt_before_image};
  std::unique_ptr<llm::IRunner> runner =
      example::create_llm_runner(runner_paths, std::move(runner_config));
  ET_CHECK_MSG(runner != nullptr, "Failed to create LLM runner");

  // TODO: support warmup interface

  executorch::extension::llm::GenerationConfig config{
      .seq_len = seq_len,
      .temperature = temperature,
      .num_bos = num_bos,
      .num_eos = num_eos};
  auto error = runner->generate(prompt, config, {}, {});
  ET_CHECK_MSG(
      error == executorch::runtime::Error::Ok,
      "Failed to generate token from LLM runner");

  return 0;
}
