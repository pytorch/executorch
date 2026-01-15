/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/log.h>

#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using ::executorch::aten::ScalarType;
using ::executorch::extension::llm::Image;
using ::executorch::extension::llm::MultimodalInput;

namespace example {

inline std::vector<std::string> load_raw_files(
    const std::string& input_list_file_path) {
  std::vector<std::string> input_files;

  std::ifstream input_list(input_list_file_path);
  ET_CHECK_MSG(
      input_list.is_open(),
      "Failed to open input list file: %s",
      input_list_file_path.c_str());

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

  std::string file_path_line;
  while (std::getline(input_list, file_path_line)) {
    if (!file_path_line.empty() && file_path_line.back() == '\r') {
      file_path_line.pop_back();
    }
    if (file_path_line.empty()) {
      continue;
    }

    auto line_files = split(file_path_line, " ");
    if (line_files.empty()) {
      continue;
    }

    input_files.insert(input_files.end(), line_files.begin(), line_files.end());
  }
  return input_files;
}

void load_image(
    const std::string& image_path,
    Image& image,
    const std::vector<int32_t>& expected_size,
    const ScalarType& expected_dtype) {
  const size_t n = expected_size.size();
  ET_CHECK_MSG(n >= 3, "expected dim should at least be 3, but got %zu", n);
  const int32_t channels = expected_size[n - 3];
  const int32_t height = expected_size[n - 2];
  const int32_t width = expected_size[n - 1];

  size_t num_elems = std::accumulate(
      expected_size.begin(),
      expected_size.end(),
      size_t{1},
      std::multiplies<size_t>());

  std::streamsize expected_length = num_elems * sizeof(float);

  std::ifstream file(image_path, std::ios::binary | std::ios::ate);
  ET_CHECK_MSG(
      file.is_open(), "Failed to open input file: %s", image_path.c_str());

  std::streamsize file_size = file.tellg();
  ET_CHECK_MSG(
      file_size == expected_length,
      "Input image size mismatch. file bytes: %ld, expected bytes: %zu (file: "
      "%s)",
      file_size,
      expected_length,
      image_path.c_str());
  file.seekg(0, std::ios::beg);
  std::vector<float> buffer(num_elems);
  file.read(reinterpret_cast<char*>(buffer.data()), expected_length);
  file.close();

  image = Image(std::move(buffer), width, height, channels);
  ET_LOG(
      Info,
      "image Channels: %" PRId32 ", Height: %" PRId32 ", Width: %" PRId32,
      image.channels(),
      image.height(),
      image.width());
}

} // namespace example
