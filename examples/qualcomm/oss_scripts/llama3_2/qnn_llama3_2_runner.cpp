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
#include <vector>

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
    "0: PromptProcessor(prefill) / 1: TokenGenerator(kv) / 2: HybridMode (prefill+kv)");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // create llama runner
  example::Runner runner(
      {FLAGS_model_path},
      FLAGS_tokenizer_path.c_str(),
      FLAGS_prompt.c_str(),
      FLAGS_system_prompt.c_str(),
      FLAGS_temperature,
      FLAGS_eval_mode);
  std::vector<char> buf;
  buf.reserve(5 * FLAGS_seq_len); // assume each token is around 5 char
  std::ofstream fout(FLAGS_output_path.c_str());
  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };
  // generate tokens & store inference output
  runner.generate(FLAGS_seq_len, callback);
  fout.write(buf.data(), buf.size());
  fout.close();
  return 0;
}
