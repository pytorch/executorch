/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>

#include <executorch/runtime/platform/assert.h>
#include <gflags/gflags.h>

#include "qnn_executorch.h"

DEFINE_string(
    model_path,
    "model.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    output_folder_path,
    ".",
    "Executorch inference data output path.");
DEFINE_string(input_list_path, "input_list.txt", "Model input list path.");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  // fastrpc related
  // adsp
  const int adsp_domain_id = 0;
  // signed PD
  const int enable_unsigned_pd = 0;
  // domain uri
  std::string domain_uri(qnn_executorch_URI);
  domain_uri += "&_dom=adsp";
  // init session
  struct remote_rpc_control_unsigned_module data;
  data.domain = adsp_domain_id;
  data.enable = enable_unsigned_pd;
  int err = AEE_SUCCESS;
  ET_CHECK_MSG(
      AEE_SUCCESS ==
          (err = remote_session_control(
               DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data))),
      "remote_session_control failed: 0x%x",
      err);
  // start session
  remote_handle64 handle = -1;
  ET_CHECK_MSG(
      AEE_SUCCESS == (err = qnn_executorch_open(domain_uri.data(), &handle)),
      "qnn_executorch_open failed: 0x%x",
      err);
  // load model
  const char* model_path = FLAGS_model_path.c_str();
  qnn_executorch_load(handle, model_path);

  // prepare io
  std::vector<std::vector<uint8_t>> input_data, output_data;
  std::vector<tensor> input_tensor, output_tensor;
  for (int i = 0;; ++i) {
    int nbytes = 0;
    qnn_executorch_get_input_size(handle, model_path, i, &nbytes);
    if (nbytes == -1) {
      break;
    }
    input_data.emplace_back(std::vector<uint8_t>(nbytes));
    input_tensor.emplace_back(
        tensor({input_data.back().data(), (int)input_data.back().size()}));
  }
  for (int i = 0;; ++i) {
    int nbytes = 0;
    qnn_executorch_get_output_size(handle, model_path, i, &nbytes);
    if (nbytes == -1) {
      break;
    }
    output_data.emplace_back(std::vector<uint8_t>(nbytes));
    output_tensor.emplace_back(
        tensor({output_data.back().data(), (int)output_data.back().size()}));
  }

  // prepare input data
  std::ifstream input_list(FLAGS_input_list_path);
  // TODO: should check IO info via fastrpc first
  if (input_list.is_open()) {
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
    int inference_index = 0;
    while (std::getline(input_list, file_path)) {
      auto input_files = split(file_path, " ");
      if (input_files.size() == 0) {
        break;
      }
      size_t num_inputs = input_files.size();
      for (int i = 0; i < num_inputs; ++i) {
        std::ifstream fin(input_files[i], std::ios::binary);
        fin.seekg(0, fin.end);
        size_t file_size = fin.tellg();
        fin.seekg(0, fin.beg);
        fin.read((char*)input_data[i].data(), file_size);
        fin.close();
      }
      qnn_executorch_set_input(
          handle, model_path, input_tensor.data(), input_tensor.size());
      qnn_executorch_execute(handle, model_path);
      qnn_executorch_get_output(
          handle, model_path, output_tensor.data(), output_tensor.size());
      for (size_t i = 0; i < output_tensor.size(); i++) {
        auto output_file_name = FLAGS_output_folder_path + "/output_" +
            std::to_string(inference_index) + "_" + std::to_string(i) + ".raw";
        std::ofstream fout(output_file_name.c_str(), std::ios::binary);
        fout.write(
            (const char*)output_tensor[i].data, output_tensor[i].dataLen);
        fout.close();
      }
    }
  }

  // unload model
  qnn_executorch_unload(handle, model_path);
  // tear down
  qnn_executorch_close(handle);
  return 0;
}
