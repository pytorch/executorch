/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

DEFINE_string(model_path, "", "Path to .pte file");
DEFINE_string(data_path, "", "Path to .ptd file (for CUDA delegate)");
DEFINE_string(input_dir, "", "Directory with input .bin files");
DEFINE_string(output_dir, "", "Directory to write output .bin files");

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

static std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", path.c_str());
    exit(1);
  }
  std::size_t size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<char> buf(size);
  f.read(buf.data(), static_cast<std::streamsize>(size));
  return buf;
}

static void write_file(const std::string& path, const void* data, size_t len) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(data), len);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_path.empty()) {
    fprintf(stderr, "Error: --model_path required\n");
    return 1;
  }

  std::unique_ptr<Module> module;
  if (!FLAGS_data_path.empty()) {
    module = std::make_unique<Module>(
        FLAGS_model_path,
        FLAGS_data_path,
        Module::LoadMode::MmapUseMlockIgnoreErrors);
  } else {
    module = std::make_unique<Module>(
        FLAGS_model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  }

  auto load_err = module->load();
  if (load_err != Error::Ok) {
    fprintf(stderr, "Failed to load model: 0x%x\n", static_cast<int>(load_err));
    return 1;
  }

  std::vector<EValue> inputs;

  if (!FLAGS_input_dir.empty()) {
    std::string path = FLAGS_input_dir + "/x.bin";
    static std::vector<char> input_buf = read_file(path);

    // Infer rows from file size: each row is 8 bf16 elements = 16 bytes
    constexpr int kCols = 8;
    constexpr int kElemSize = 2; // bf16
    int rows = static_cast<int>(input_buf.size()) / (kCols * kElemSize);

    static executorch::extension::TensorPtr input_tensor;
    input_tensor = from_blob(
        input_buf.data(), {rows, kCols}, exec_aten::ScalarType::BFloat16);
    inputs.push_back(*input_tensor);
  } else {
    fprintf(stderr, "Error: --input_dir required\n");
    return 1;
  }

  auto result = module->execute("forward", inputs);
  if (!result.ok()) {
    fprintf(stderr, "Forward failed: 0x%x\n", static_cast<int>(result.error()));
    return 1;
  }

  auto outputs = result.get();
  for (size_t i = 0; i < outputs.size(); i++) {
    if (!outputs[i].isTensor())
      continue;
    const auto& t = outputs[i].toTensor();
    printf("Output %zu: [", i);
    for (int d = 0; d < t.dim(); d++)
      printf("%d%s", static_cast<int>(t.size(d)), d < t.dim() - 1 ? "," : "");
    printf("] dtype=%d\n", static_cast<int>(t.scalar_type()));

    if (!FLAGS_output_dir.empty()) {
      std::string path =
          FLAGS_output_dir + "/output_" + std::to_string(i) + ".bin";
      write_file(path, t.const_data_ptr(), t.nbytes());
      printf("  Saved to %s (%zu bytes)\n", path.c_str(), (size_t)t.nbytes());
    }
  }

  printf("SUCCESS\n");
  return 0;
}
