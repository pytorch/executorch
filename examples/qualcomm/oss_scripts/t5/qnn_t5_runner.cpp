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
 * This tool can run t5 with Qualcomm AI Engine Direct.
 *
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/t5/runner/runner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#include <fstream>
#include <vector>

DEFINE_string(
    model_path,
    "t5_qnn.pte",
    "t5 model serialized in flatbuffer format.");

DEFINE_string(
    tokenizer_model_path,
    "tokenizer.model",
    "The tokenizer is saved from T5Tokenize.save_pretrained for tokenizer.");
DEFINE_string(
    input_list_path,
    "input_list.txt",
    "Input list storing file name of encoded results.");
DEFINE_int32(
    seq_len,
    128,
    "Maximum sequence length for the generated output.  Defaults to use the model's `max_cache_size` attribute. Will be truncated to maximal cache size if larger than `max_cache_size`.");

DEFINE_string(
    output_folder_path,
    "outputs",
    "Executorch inference data output path.");

std::vector<std::vector<std::vector<uint8_t>>> parse_input_list_file(
    const std::string& input_list_path) {
  std::vector<std::vector<std::vector<uint8_t>>> bufs;
  std::ifstream input_list(input_list_path);

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

  if (!input_list.is_open()) {
    ET_LOG(Error, "Unable to open file");
    return bufs;
  }

  std::string file_path;
  while (std::getline(input_list, file_path)) {
    auto input_files = split(file_path, " ");
    int num_inputs = input_files.size();
    if (num_inputs == 0) {
      break;
    }

    bufs.emplace_back();
    bufs.back().resize(num_inputs);
    for (int input_index = 0; input_index < num_inputs; ++input_index) {
      std::ifstream fin(input_files[input_index], std::ios::binary);
      if (!fin.is_open()) {
        ET_LOG(
            Error, "Could not open file %s", input_files[input_index].c_str());
        continue;
      }

      fin.seekg(0, std::ios::end);
      size_t file_size = fin.tellg();
      fin.seekg(0, std::ios::beg);
      bufs.back()[input_index].resize(file_size);

      if (!fin.read(
              reinterpret_cast<char*>(bufs.back()[input_index].data()),
              file_size)) {
        ET_LOG(
            Error, "Could not read file %s", input_files[input_index].c_str());
        continue;
      }

      fin.close();
    }
  }

  input_list.close();
  return bufs;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::vector<std::vector<uint8_t>>> multi_turns_input_buffers =
      parse_input_list_file(FLAGS_input_list_path);

  for (int iter = 0; iter < multi_turns_input_buffers.size(); ++iter) {
    std::vector<char> bufs;
    bufs.reserve(5 * FLAGS_seq_len); // assume each token is around 5 char
    auto callback = [&](const std::string& piece) {
      for (const char c : piece) {
        bufs.push_back(c);
      }
    };

    example::Runner runner(FLAGS_model_path, FLAGS_tokenizer_model_path);
    // generate tokens
    runner.generate(FLAGS_seq_len, multi_turns_input_buffers[iter], callback);
    auto output_file_name =
        FLAGS_output_folder_path + "/output_" + std::to_string(iter) + ".txt";
    std::ofstream fout(output_file_name);
    fout.write(bufs.data(), bufs.size());
    fout.close();
  }

  return 0;
}
