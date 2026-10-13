/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/examples/samsung/executor_runner/llama/runner/runner.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

namespace llm = ::executorch::extension::llm;

DEFINE_string(
    model_path,
    "llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(data_path, "", "Data file for the model.");

DEFINE_string(
    tokenizer_type,
    "tiktoken",
    "Tokenizer type. One of ['bpe', 'tiktoken'].");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(pos_embedding_path, "", "rotary embedding stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

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

DEFINE_bool(warmup, false, "Whether to run a warmup run.");

DEFINE_bool(decode_only, false, "Whether to run with decode method only.");

DEFINE_bool(use_gemma, false, "Use Gemma model specific processing.");

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  std::optional<std::string> data_path = std::nullopt;
  if (!FLAGS_data_path.empty()) {
    data_path = FLAGS_data_path.c_str();
  }

  const char* tokenizer_type = FLAGS_tokenizer_type.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  std::optional<const std::string> pos_embedding_path =
      FLAGS_pos_embedding_path.empty() ? std::nullopt
                                       : std::make_optional(FLAGS_pos_embedding_path.c_str());

  ET_LOG(Info, "prompt : %s", FLAGS_prompt.c_str());
  const char* prompt = FLAGS_prompt.c_str();

  float temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  int32_t cpu_threads = FLAGS_cpu_threads;

  bool warmup = FLAGS_warmup;

  bool decode_only = FLAGS_decode_only;

  bool use_gemma = FLAGS_use_gemma;

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

  // create llama runner
  example::RunnerPaths runner_paths{
      .model_path = model_path,
      .tokenizer_path = tokenizer_path,
      .pos_embedding_path = pos_embedding_path,
      .data_path = data_path};
  example::RunnerConfig runner_config{
      .tokenizer_type = tokenizer_type,
      .decode_only = decode_only,
      .use_gemma = use_gemma};
  std::unique_ptr<llm::IRunner> runner = example::create_llama_runner(runner_paths, runner_config);

  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create llama runner");
    return 1;
  }

  if (warmup) {
    ET_LOG(
        Info,
        "Can not support warm-up (TODO : IRunner doesn't have warmup interface (pure virtual)");
#if 0
    auto error = runner->warmup(prompt, /*max_new_tokens=*/seq_len);
    if (error != executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to warmup llama runner");
      return 1;
    }
#endif
  }

  // generate
  executorch::extension::llm::GenerationConfig config{
      .seq_len = seq_len, .temperature = temperature,
      .num_bos = num_bos, .num_eos = num_eos};
  auto error = runner->generate(prompt, config, {}, {});
  if (error != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to generate token from llama runner");
    return 1;
  }

  return 0;
}
