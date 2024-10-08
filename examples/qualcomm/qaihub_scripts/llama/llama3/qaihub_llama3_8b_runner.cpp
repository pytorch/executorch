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
 * This tool can run Llama3 8b with Qualcomm AI Engine Direct.
 *
 * User could specify arguments like desired prompt, eval_mode, etc.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/qaihub_scripts/llama/runner/runner.h>
#include <executorch/runtime/platform/log.h>

#include <gflags/gflags.h>

#include <fstream>

DEFINE_string(sharded_1_path, "", "Path to 1st sharded pte file");
DEFINE_string(sharded_2_path, "", "Path to 2nd sharded pte file");
DEFINE_string(sharded_3_path, "", "Path to 3rd sharded pte file");
DEFINE_string(sharded_4_path, "", "Path to 4th sharded pte file");
DEFINE_string(sharded_5_path, "", "Path to 5th sharded pte file");

DEFINE_string(freq_cos_path, "", "Path to precomputed position embeddings");
DEFINE_string(freq_sin_path, "", "Path to precomputed position embeddings");

DEFINE_string(output_path, "outputs", "Executorch inference data output path.");
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
    eval_mode,
    0,
    "0: PromptProcessor / 1: TokenGenerator / 2: MixedMode (TBD)");
DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output). Defaults to max_seq_len. If the number of input tokens + seq_len > max_seq_len, the output will be truncated to max_seq_len tokens.");
DEFINE_double(logits_scale, 0.0, "Path to logits scale file");
DEFINE_int32(logits_offset, 0, "Path to logits offset file");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> models_path = {
      FLAGS_sharded_1_path,
      FLAGS_sharded_2_path,
      FLAGS_sharded_3_path,
      FLAGS_sharded_4_path,
      FLAGS_sharded_5_path};
  std::vector<std::string> pos_embs_path = {
      FLAGS_freq_cos_path, FLAGS_freq_sin_path};

  // create llama runner
  example::Runner runner(
      models_path,
      pos_embs_path,
      {4, 8, 8, 8, 4},
      FLAGS_tokenizer_path.c_str(),
      FLAGS_eval_mode,
      FLAGS_temperature,
      FLAGS_logits_scale,
      FLAGS_logits_offset);

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
