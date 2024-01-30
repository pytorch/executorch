/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/examples/models/llama2/runner/runner.h>

DEFINE_string(
    model_path,
    "llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_bool(
    eos,
    false,
    "Whether to append an end-of-sentence token to the end of the prompt");

DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");

DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");

int32_t main(int32_t argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Create a loader to get the data of the program file. There are other
  // DataLoaders that use mmap() or point32_t to data that's already in memory,
  // and users can create their own DataLoaders to load from arbitrary sources.
  const char* model_path = FLAGS_model_path.c_str();

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();

  const char* prompt = FLAGS_prompt.c_str();

  // create llama runner
  ::torch::executor::Runner runner(model_path, tokenizer_path);

  // generate
  runner.generate(prompt, FLAGS_eos);

  return 0;
}
