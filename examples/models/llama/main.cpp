/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <executorch/examples/models/llama/runner/runner.h>
#include <gflags/gflags.h>
#include <sstream>
#include <vector>

#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

DEFINE_string(
    model_path,
    "llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    data_paths,
    "",
    "Data files for the model. If multiple files are provided, they should be comma separated.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    128,
    "DEPRECATED: Please use max_seq_len instead. Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    max_new_tokens,
    -1,
    "Total number of tokens to generate, excluding the prompt, will be capped by max_seq_len - # prompt tokens.");

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

DEFINE_bool(
    ignore_eos,
    false,
    "Whether to ignore EOS token and continue generating until max_new_tokens is reached.");

DEFINE_string(
    etdump_path,
    "etdump.in",
    "If an etdump path is provided, generate an ETDump file at the specified path for profiling purposes.");

DEFINE_string(
    method_name,
    "forward",
    "Method name to execute in the model (e.g., 'forward', 'lora_forward').");

// Helper function to parse comma-separated string lists
std::vector<std::string> parseStringList(const std::string& input) {
  std::vector<std::string> result;
  if (input.empty()) {
    return result;
  }

  std::stringstream ss(input);
  std::string item;
  while (std::getline(ss, item, ',')) {
    // Trim whitespace
    item.erase(0, item.find_first_not_of(" \t"));
    item.erase(item.find_last_not_of(" \t") + 1);
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  std::vector<std::string> data_paths = parseStringList(FLAGS_data_paths);

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();

  float temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  int32_t cpu_threads = FLAGS_cpu_threads;

  bool warmup = FLAGS_warmup;

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

#ifdef ET_EVENT_TRACER_ENABLED
  // Create ETDumpGen and get raw pointer reference for later access
  auto etdump_gen_ptr = std::make_unique<executorch::etdump::ETDumpGen>();
  executorch::etdump::ETDumpGen* etdump_gen = etdump_gen_ptr.get();
#endif

  // create llama runner
  std::unique_ptr<::executorch::extension::llm::TextLLMRunner> runner =
      example::create_llama_runner(
          model_path,
          tokenizer_path,
          data_paths,
          temperature,
#ifdef ET_EVENT_TRACER_ENABLED
          std::move(etdump_gen_ptr),
#else
          nullptr,
#endif
          FLAGS_method_name);

  if (runner == nullptr) {
    ET_LOG(Error, "Failed to create llama runner");
    return 1;
  }

  if (warmup) {
    int32_t warmup_max_new_tokens =
        FLAGS_max_new_tokens != -1 ? FLAGS_max_new_tokens : seq_len;
    auto error =
        runner->warmup(prompt, /*max_new_tokens=*/warmup_max_new_tokens);
    if (error != executorch::runtime::Error::Ok) {
      ET_LOG(Error, "Failed to warmup llama runner");
      return 1;
    }
  }
  // generate
  executorch::extension::llm::GenerationConfig config{
      .temperature = temperature};

  config.ignore_eos = FLAGS_ignore_eos;

  if (FLAGS_max_new_tokens != -1) {
    config.max_new_tokens = FLAGS_max_new_tokens;
  } else {
    ET_LOG(
        Info,
        "max_new_tokens not provided, falling back to seq_len=%d. "
        "Consider using --max_new_tokens instead of --seq_len for specifying generation length.",
        seq_len);
    config.seq_len = seq_len;
  }

  auto error = runner->generate(prompt, config);
  if (error != executorch::runtime::Error::Ok) {
    ET_LOG(Error, "Failed to run llama runner");
    return 1;
  }

#ifdef ET_EVENT_TRACER_ENABLED
  if (etdump_gen != nullptr) {
    executorch::etdump::ETDumpResult result = etdump_gen->get_etdump_data();
    if (result.buf != nullptr && result.size > 0) {
      FILE* f = fopen(FLAGS_etdump_path.c_str(), "w+");
      if (f == nullptr) {
        ET_LOG(
            Error,
            "Failed to open etdump file at path: %s",
            FLAGS_etdump_path.c_str());
      } else {
        fwrite((uint8_t*)result.buf, 1, result.size, f);
        fclose(f);
        ET_LOG(Info, "ETDump file written to: %s", FLAGS_etdump_path.c_str());
      }
      free(result.buf);
    }
  }
#endif

  return 0;
}
