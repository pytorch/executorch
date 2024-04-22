/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run ExecuTorch model files with Qualcomm AI Engine Direct
 * and the portable kernels.
 *
 * User could specify arguments like desired input data, iterfations, etc.
 * Currently we assume that the outputs are all fp32 tensors.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/llama2/runner/runner.h>
#include <executorch/extension/runner_util/managed_tensor.h>
#include <executorch/runtime/platform/log.h>

#include <gflags/gflags.h>

#include <fstream>

DEFINE_string(
    model_path,
    "qnn_llama2.pte",
    "Model serialized in flatbuffer format.");

DEFINE_string(
    output_folder_path,
    "outputs",
    "Executorch inference data output path.");

DEFINE_string(input_list_path, "input_list.txt", "Model input list path.");

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

int main(int argc, char** argv) {
  using namespace torch::executor;

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* model_path = FLAGS_model_path.c_str();
  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char* prompt = FLAGS_prompt.c_str();
  double temperature = FLAGS_temperature;
  int32_t seq_len = FLAGS_seq_len;

  // create llama runner
  Runner runner(model_path, tokenizer_path, temperature);
  ET_CHECK_MSG(runner.load() == Error::Ok, "Runner failed to load method");

  // MethodMeta describes the memory requirements of the method.
  Result<MethodMeta> method_meta = runner.method_meta();
  ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta 0x%x",
      (unsigned int)method_meta.error());

  // Fill in data for input
  std::ifstream input_list(FLAGS_input_list_path);
  ET_CHECK_MSG(input_list.is_open(), "Failed to open input_list.txt");

  auto split = [](std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
      token = s.substr(pos_start, pos_end - pos_start);
      pos_start = pos_end + delim_len;
      res.push_back(token);
    }
    res.push_back(s.substr(pos_start));
    return res;
  };

  std::string file_path;
  size_t inference_index = 0, num_inputs = method_meta->num_inputs();
  std::vector<std::vector<char>> inputs(num_inputs);
  while (std::getline(input_list, file_path)) {
    auto input_files = split(file_path, " ");
    if (input_files.size() == 0) {
      break;
    }
    // inputs: [tokens, pos_ids, kv_mask, *k_cache, *v_cache]
    // tokens are determined by command line arguments
    // pos_ids are infered inside runner
    std::vector<ManagedTensor> managed_inputs;
    for (int input_index = 2; input_index < num_inputs; ++input_index) {
      Result<TensorInfo> tensor_meta =
          method_meta->input_tensor_meta(input_index);

      std::ifstream fin(input_files[input_index], std::ios::binary);
      fin.seekg(0, fin.end);
      size_t file_size = fin.tellg();

      ET_CHECK_MSG(
          file_size == tensor_meta->nbytes(),
          "Input(%d) size mismatch. file bytes: %zu, tensor bytes: %zu",
          input_index,
          file_size,
          tensor_meta->nbytes());

      inputs[input_index].resize(tensor_meta->nbytes());
      fin.close();

      auto tensor_shape = tensor_meta->sizes();
      std::vector<exec_aten::SizesType> sizes(
          tensor_shape.data(), tensor_shape.data() + tensor_shape.size());

      managed_inputs.emplace_back(ManagedTensor(
          inputs[input_index].data(), 128, sizes, tensor_meta->scalar_type()));
    }

    // generate tokens
    std::string inference_output;
    runner.generate(
        prompt, seq_len, managed_inputs, [&](const std::string& piece) {
          inference_output += piece;
        });

    auto output_file_name = FLAGS_output_folder_path + "/output_" +
        std::to_string(inference_index++) + "_0.raw";
    std::ofstream fout(output_file_name.c_str());
    fout << inference_output;
    fout.close();
  }

  return 0;
}
