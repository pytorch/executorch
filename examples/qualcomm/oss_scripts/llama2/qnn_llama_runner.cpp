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
 * This tool can run ExecuTorch model files with Qualcomm AI Engine Direct.
 *
 * User could specify arguments like desired prompt, temperature, etc.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama2/runner/runner.h>
#include <executorch/runtime/platform/log.h>

#include <gflags/gflags.h>

#include <fstream>
#include <vector>

DEFINE_string(
    model_path,
    "qnn_llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    output_folder_path,
    "outputs",
    "Executorch inference data output path.");

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

using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char* prompt = FLAGS_prompt.c_str();
  double temperature = FLAGS_temperature;
  int32_t seq_len = FLAGS_seq_len;

  // create llama runner
  example::Runner runner(FLAGS_model_path, tokenizer_path, temperature);
  ET_CHECK_MSG(runner.load() == Error::Ok, "Runner failed to load method");

  // MethodMeta describes the memory requirements of the method.
  Result<MethodMeta> method_meta = runner.get_method_meta();
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta 0x%x",
      (unsigned int)method_meta.error());
  ET_CHECK_MSG(
      runner.mem_alloc(MemoryAllocator::kDefaultAlignment, seq_len) ==
          Error::Ok,
      "Runner failed to allocate memory");

  // generate tokens
  std::string inference_output;
  // prompt are determined by command line arguments
  // pos_ids, atten_mask are infered inside runner
  runner.generate(prompt, seq_len, [&](const std::string& piece) {
    inference_output += piece;
  });

  size_t inference_index = 0;
  auto output_file_name = FLAGS_output_folder_path + "/output_" +
      std::to_string(inference_index++) + "_0.raw";
  std::ofstream fout(output_file_name.c_str());
  fout << inference_output;
  fout.close();

  return 0;
}
