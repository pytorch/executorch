/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run Llama3.2 1B/3B with Qualcomm AI Engine Direct.
 *
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama3_2/runner/runner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#include <fstream>

DEFINE_string(
    model_path,
    "qnn_llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    output_path,
    "outputs.txt",
    "Executorch inference data output path.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");
DEFINE_string(prompt, "The answer to the ultimate question is", "Prompt.");
DEFINE_string(
    system_prompt,
    "",
    "Tells the model what kind of assistant it should be. For example, You are a helpful AI assistant for travel tips and recommendations. Default is None");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");
DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");

DEFINE_int32(
    eval_mode,
    0,
    "0: PromptProcessor(batch_prefill) / 1: TokenGenerator(kv) / 2: HybridMode (TBD)");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // create llama runner
  example::Runner runner(
      {FLAGS_model_path},
      FLAGS_tokenizer_path.c_str(),
      FLAGS_temperature,
      FLAGS_eval_mode);

  // generate tokens & store inference output
  std::ofstream fout(FLAGS_output_path.c_str());
  runner.generate(
      FLAGS_prompt,
      FLAGS_system_prompt,
      FLAGS_seq_len,
      [&](const std::string& piece) { fout << piece; });
  fout.close();
  return 0;
}
