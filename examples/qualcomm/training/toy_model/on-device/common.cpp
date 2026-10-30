/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common.h"

#include <cstdint>
#include <fstream>

#include <executorch/runtime/platform/log.h>

using executorch::aten::SizesType;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;

namespace qnn_toy_model_training {

nlohmann::json load_json(const std::string& path) {
  std::ifstream file(path);
  ET_CHECK_MSG(file.is_open(), "Failed to open %s", path.c_str());
  nlohmann::json j;
  file >> j;
  return j;
}

std::unordered_map<std::string, TensorPtr> load_safetensors(
    const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  ET_CHECK_MSG(file.is_open(), "Failed to open %s", path.c_str());

  uint64_t header_size = 0;
  file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
  ET_CHECK_MSG(file.good(), "Failed to read header size from %s", path.c_str());

  std::string header_bytes(header_size, '\0');
  file.read(header_bytes.data(), static_cast<std::streamsize>(header_size));
  ET_CHECK_MSG(file.good(), "Failed to read header from %s", path.c_str());
  const nlohmann::json header = nlohmann::json::parse(header_bytes);

  const auto data_start =
      static_cast<std::streamoff>(sizeof(header_size) + header_size);

  std::unordered_map<std::string, TensorPtr> tensors;
  for (const auto& [name, info] : header.items()) {
    if (name == "__metadata__") {
      continue;
    }
    const auto dtype = info.at("dtype").get<std::string>();
    ET_CHECK_MSG(
        dtype == "F32",
        "Unsupported safetensors dtype %s for tensor %s (expected F32).",
        dtype.c_str(),
        name.c_str());

    const auto shape = info.at("shape").get<std::vector<int64_t>>();
    const std::vector<SizesType> sizes(shape.begin(), shape.end());

    const auto offsets = info.at("data_offsets").get<std::vector<size_t>>();
    ET_CHECK_MSG(
        offsets.size() == 2 && offsets.at(1) >= offsets.at(0),
        "Invalid data_offsets for tensor %s in %s.",
        name.c_str(),
        path.c_str());
    const size_t nbytes = offsets.at(1) - offsets.at(0);

    size_t numel = 1;
    for (const auto s : sizes) {
      numel *= static_cast<size_t>(s);
    }
    ET_CHECK_MSG(
        nbytes == numel * sizeof(float),
        "Tensor %s in %s: data_offsets span %zu bytes, expected %zu for its shape.",
        name.c_str(),
        path.c_str(),
        nbytes,
        numel * sizeof(float));

    std::vector<float> data(nbytes / sizeof(float));
    file.seekg(data_start + static_cast<std::streamoff>(offsets.at(0)));
    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(nbytes));
    ET_CHECK_MSG(
        file.good(),
        "Failed to read tensor data for %s from %s",
        name.c_str(),
        path.c_str());

    tensors.emplace(name, make_tensor_ptr(sizes, std::move(data)));
  }
  return tensors;
}

std::vector<TensorPtr> load_weights(
    const nlohmann::json& manifest,
    const std::string& safetensors_path) {
  auto tensors = load_safetensors(safetensors_path);
  std::vector<TensorPtr> weights;
  for (const auto& name : manifest.at("param_names")) {
    const auto key = name.get<std::string>();
    const auto it = tensors.find(key);
    ET_CHECK_MSG(
        it != tensors.end(),
        "Missing weight %s in safetensors file.",
        key.c_str());
    weights.push_back(it->second);
  }
  return weights;
}

std::vector<SizesType> shape_from_json(const nlohmann::json& shape_array) {
  const auto shape = shape_array.get<std::vector<int64_t>>();
  return std::vector<SizesType>(shape.begin(), shape.end());
}

TensorPtr make_random_tensor(
    const std::vector<SizesType>& shape,
    std::mt19937& rng) {
  size_t n = 1;
  for (const auto s : shape) {
    n *= static_cast<size_t>(s);
  }
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> data(n);
  for (auto& v : data) {
    v = dist(rng);
  }
  return make_tensor_ptr(shape, std::move(data));
}

void save_safetensors(
    const std::string& path,
    const std::vector<std::string>& names,
    const std::vector<TensorPtr>& tensors) {
  ET_CHECK_MSG(
      names.size() == tensors.size(),
      "save_safetensors: %zu names but %zu tensors.",
      names.size(),
      tensors.size());

  // Build header JSON: each entry has dtype, shape, data_offsets.
  nlohmann::json header;
  size_t offset = 0;
  for (size_t i = 0; i < names.size(); i++) {
    const auto& t = *tensors[i];
    const size_t nbytes = t.numel() * sizeof(float);
    std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
    header[names[i]] = {
        {"dtype", "F32"},
        {"shape", shape},
        {"data_offsets", {offset, offset + nbytes}},
    };
    offset += nbytes;
  }

  const std::string header_str = header.dump();
  // safetensors spec: header must be padded to a multiple of 8 bytes with
  // spaces.
  const size_t padded = ((header_str.size() + 7) / 8) * 8;
  std::string header_padded = header_str;
  header_padded.resize(padded, ' ');

  std::ofstream file(path, std::ios::binary);
  ET_CHECK_MSG(file.is_open(), "Failed to open %s for writing.", path.c_str());

  const uint64_t header_size = static_cast<uint64_t>(padded);
  file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
  file.write(header_padded.data(), static_cast<std::streamsize>(padded));
  for (const auto& t : tensors) {
    const size_t nbytes = t->numel() * sizeof(float);
    file.write(
        reinterpret_cast<const char*>(t->const_data_ptr<float>()),
        static_cast<std::streamsize>(nbytes));
  }
  ET_CHECK_MSG(file.good(), "Failed to write safetensors to %s.", path.c_str());
}

std::string manifest_dir(const std::string& manifest_path) {
  const auto slash = manifest_path.find_last_of('/');
  return slash == std::string::npos ? "." : manifest_path.substr(0, slash);
}

} // namespace qnn_toy_model_training
