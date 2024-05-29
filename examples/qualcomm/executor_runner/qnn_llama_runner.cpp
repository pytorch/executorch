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
#include <sstream>
#include <vector>

DEFINE_string(
    model_paths,
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

  std::vector<std::string> model_path_list;
  std::istringstream f(FLAGS_model_paths);
  std::string s;    
  while (getline(f, s, ',')) {
    model_path_list.push_back(s);
  }

  const char* tokenizer_path = FLAGS_tokenizer_path.c_str();
  const char* prompt = FLAGS_prompt.c_str();
  double temperature = FLAGS_temperature;
  int32_t seq_len = FLAGS_seq_len;

  // create llama runner
  Runner runner(model_path_list, tokenizer_path, temperature);
  ET_CHECK_MSG(runner.load() == Error::Ok, "Runner failed to load method");

  // MethodMeta describes the memory requirements of the method.
  std::vector<Result<MethodMeta>> method_metas = runner.get_methods_meta();
  for(auto& method_meta: method_metas){
    ET_CHECK_MSG(
      method_meta.ok(),
      "Failed to get method_meta 0x%x",
      (unsigned int)method_meta.error());
  }

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
  size_t inference_index = 0;
  std::vector<std::vector<float>> freqs_inputs(2);
  std::vector<std::vector<std::vector<char>>> inputs(method_metas.size()-2);

  for (int i = 1; i < method_metas.size()-1; i++){
    size_t num_inputs = method_metas[i]->num_inputs();
    inputs[i-1].resize(num_inputs);
  }

  while (std::getline(input_list, file_path)) {
    auto input_files = split(file_path, " ");
    if (input_files.size() == 0) {
      break;
    }
    // inputs: [tokens, pos_ids, freqs_cos, freqs_sin, atten_mask, *k_cache, *v_cache]
    // tokens are determined by command line arguments
    // pos_ids, atten_mask are infered inside runner

    for (int input_index = 2; input_index < 4; ++input_index) {
      std::ifstream fin(input_files[input_index], std::ios::binary);
      fin.seekg(0, fin.end);
      size_t file_size = fin.tellg();

      freqs_inputs[input_index-2].resize(file_size / sizeof(float));
      fin.seekg(0, fin.beg);
      fin.read(reinterpret_cast<char*>(freqs_inputs[input_index-2].data()), file_size);
      fin.close();
    }

    std::vector<std::vector<ManagedTensor>> managed_kv_inputs(method_metas.size()-2);
    for (int i = 1; i < method_metas.size()-1; ++i){
      size_t num_inputs = method_metas[i]->num_inputs();
      const int k_caches_end = (num_inputs - 4) / 2;

      // TODO: need to handle batch size != 1
      // k caches init
      for (int input_index = 4; input_index < k_caches_end; ++input_index) {
        Result<TensorInfo> tensor_meta =
            method_metas[i]->input_tensor_meta(input_index);
        int file_index = (i-1) * (num_inputs - 4) + input_index + 1;
        std::ifstream fin(input_files[file_index], std::ios::binary);
        fin.seekg(0, fin.end);
        size_t file_size = fin.tellg();

        ET_CHECK_MSG(
            file_size == tensor_meta->nbytes(),
            "Input(%d) size mismatch. file bytes: %zu, tensor bytes: %zu",
            file_index,
            file_size,
            tensor_meta->nbytes());

        // to simplify kv_cache update logic, we use (bsz, head_dim+2, seq)
        // for fast pointer shifting
        // head_dim+1 is the buffer of last word
        // head_dim+2 is for output
        inputs[i-1][input_index].resize(tensor_meta->nbytes() + 2*(tensor_meta->nbytes()/tensor_meta->sizes()[1]));
        fin.close();

        auto tensor_shape = tensor_meta->sizes();
        std::vector<exec_aten::SizesType> sizes(
            tensor_shape.data(), tensor_shape.data() + tensor_shape.size());

        managed_kv_inputs[i-1].emplace_back(ManagedTensor(
            inputs[i-1][input_index].data(), 128, sizes, tensor_meta->scalar_type()));
      }

      // v caches init
      for (int input_index = k_caches_end; input_index < num_inputs; ++input_index) {
        Result<TensorInfo> tensor_meta =
            method_metas[i]->input_tensor_meta(input_index);
        int file_index = (i-1) * (num_inputs - 4) + input_index + 1;
        std::ifstream fin(input_files[file_index], std::ios::binary);
        fin.seekg(0, fin.end);
        size_t file_size = fin.tellg();

        ET_CHECK_MSG(
            file_size == tensor_meta->nbytes(),
            "Input(%d) size mismatch. file bytes: %zu, tensor bytes: %zu",
            file_index,
            file_size,
            tensor_meta->nbytes());

        // to simplify v_cache update logic, we use (bsz, 2*max_seq_len, head_dim)
        // for fast pointer shifting
        inputs[i-1][input_index].resize(2*tensor_meta->nbytes());
        fin.close();

        auto tensor_shape = tensor_meta->sizes();
        std::vector<exec_aten::SizesType> sizes(
            tensor_shape.data(), tensor_shape.data() + tensor_shape.size());

        managed_kv_inputs[i-1].emplace_back(ManagedTensor(
            inputs[i-1][input_index].data(), 128, sizes, tensor_meta->scalar_type()));
      }
    }

    // generate tokens
    std::string inference_output;
    runner.generate(
        prompt, seq_len, managed_kv_inputs, freqs_inputs, [&](const std::string& piece) {
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
