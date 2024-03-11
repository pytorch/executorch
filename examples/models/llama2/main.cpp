/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/examples/models/llama2/runner/runner.h>

#include <executorch/backends/xnnpack/threadpool/cpuinfo_utils.h>
#include <executorch/backends/xnnpack/threadpool/threadpool.h>

DEFINE_string(
    model_path,
    "llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

DEFINE_double(
    temperature,
    0.8f,
    "Temperature; Default is 0.8f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();

  double temperature = FLAGS_temperature;

  int32_t seq_len = FLAGS_seq_len;

  uint32_t num_performant_cores =
      torch::executorch::cpuinfo::get_num_performant_cores();
  ET_LOG(
      Info, "Resetting threadpool with num threads = %d", num_performant_cores);
  torch::executorch::threadpool::get_threadpool()->_unsafe_reset_threadpool(
      num_performant_cores);
  // create llama runner
  ::torch::executor::Runner runner(model_path, tokenizer_path, temperature);

  // generate
  runner.generate(prompt, seq_len);

  return 0;
}
