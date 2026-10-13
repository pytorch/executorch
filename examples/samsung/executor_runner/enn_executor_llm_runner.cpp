/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <executorch/examples/samsung/executor_runner/exynos_llm_wrapper.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#if defined(ET_USE_THREADPOOL)
#include <executorch/extension/threadpool/cpuinfo_utils.h>
#include <executorch/extension/threadpool/threadpool.h>
#endif

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

DEFINE_string(system_prompt, "", "System prompt. Empty means none.");

DEFINE_string(
    prompt_file,
    "",
    "Read the prompt from this file instead of --prompt.");

DEFINE_string(
    system_prompt_file,
    "",
    "Read the system prompt from this file instead of --system_prompt.");

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

DEFINE_int32(
    max_token_limit,
    0,
    "If > 0, print each token from the callback and call enn_llm_stop() right after the N-th generated token. Used to verify the stop API.");

struct StopContext {
  enn_runner* runner;
  int32_t limit;
  int32_t num_pieces = 0;
  int32_t num_tokens = 0;
};

static void on_token_with_limit(const char* text, int32_t len, void* userdata) {
  auto* ctx = static_cast<StopContext*>(userdata);
  fwrite(text, 1, static_cast<size_t>(len), stdout);
  fflush(stdout);
  // The first callback is the echoed input prompt, not a generated token.
  if (ctx->num_pieces++ == 0) {
    return;
  }
  // The runner emits a trailing "\n" after generation ends; ignore it.
  if (ctx->num_tokens >= ctx->limit) {
    return;
  }
  if (++ctx->num_tokens == ctx->limit) {
    enn_llm_stop(ctx->runner);
    printf("\n[STOP] enn_llm_stop called at token %d\n", ctx->num_tokens);
    fflush(stdout);
  }
}

static bool read_file(const std::string& path, std::string* out) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    ET_LOG(Error, "Failed to open %s", path.c_str());
    return false;
  }
  std::stringstream ss;
  ss << file.rdbuf();
  *out = ss.str();
  return true;
}

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if ((!FLAGS_prompt_file.empty() &&
       !read_file(FLAGS_prompt_file, &FLAGS_prompt)) ||
      (!FLAGS_system_prompt_file.empty() &&
       !read_file(FLAGS_system_prompt_file, &FLAGS_system_prompt))) {
    return ENN_ERR_INVALID_ARG;
  }

  if (FLAGS_prompt.empty()) {
    ET_LOG(Error, "Input prompt cannot be empty.");
    return ENN_ERR_INVALID_ARG;
  }
  ET_LOG(Info, "Input prompt message: %s", FLAGS_prompt.c_str());

#if defined(ET_USE_THREADPOOL)
  uint32_t num_performant_cores = FLAGS_cpu_threads == -1
      ? ::executorch::extension::cpuinfo::get_num_performant_cores()
      : static_cast<uint32_t>(FLAGS_cpu_threads);
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  if (num_performant_cores > 0) {
    ::executorch::extension::threadpool::get_threadpool()
        ->_unsafe_reset_threadpool(num_performant_cores);
  }
#endif

  enn_runner* runner = enn_llm_create(
      FLAGS_model_path.c_str(),
      FLAGS_data_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_pos_embedding_path.c_str(),
      FLAGS_config_path.c_str(),
      FLAGS_vision_model_path.c_str(),
      FLAGS_is_decode_only);
  ET_CHECK_MSG(runner != nullptr, "Failed to create LLM runner");

  // TODO: support warmup interface

  const bool stop_test = FLAGS_max_token_limit > 0;
  StopContext stop_ctx{runner, FLAGS_max_token_limit};
  int32_t error = enn_llm_generate(
      runner,
      FLAGS_prompt.c_str(),
      FLAGS_system_prompt.c_str(),
      FLAGS_image_path.c_str(),
      FLAGS_prompt_before_image.c_str(),
      FLAGS_seq_len,
      static_cast<float>(FLAGS_temperature),
      FLAGS_num_bos,
      FLAGS_num_eos,
      /*echo=*/1,
      /*verbose=*/!stop_test,
      stop_test ? on_token_with_limit : nullptr,
      stop_test ? &stop_ctx : nullptr);
  enn_llm_destroy(runner);
  if (stop_test) {
    printf(
        "\n[STOP] limit=%d, generated=%d\n",
        stop_ctx.limit,
        stop_ctx.num_tokens);
  }
  ET_CHECK_MSG(error == ENN_OK, "Failed to generate token from LLM runner");

  return 0;
}
