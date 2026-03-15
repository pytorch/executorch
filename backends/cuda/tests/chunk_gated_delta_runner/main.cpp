/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

DEFINE_string(model_path, "", "Path to .pte file");
DEFINE_string(data_path, "", "Path to .ptd directory (for CUDA delegate)");
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

  constexpr int B = 1, T = 128, H = 4, K = 64, V = 64;

  std::vector<EValue> inputs;

  if (!FLAGS_input_dir.empty()) {
    // Load inputs from binary files
    struct TensorSpec {
      const char* name;
      std::vector<exec_aten::SizesType> shape;
      exec_aten::ScalarType dtype;
    };
    TensorSpec specs[] = {
        {"q", {B, T, H, K}, exec_aten::ScalarType::BFloat16},
        {"k", {B, T, H, K}, exec_aten::ScalarType::BFloat16},
        {"v", {B, T, H, V}, exec_aten::ScalarType::BFloat16},
        {"g", {B, T, H}, exec_aten::ScalarType::BFloat16},
        {"beta", {B, T, H}, exec_aten::ScalarType::BFloat16},
        {"initial_state", {B, H, K, V}, exec_aten::ScalarType::BFloat16},
    };

    // Keep data and TensorPtrs alive for the duration of execution
    static std::vector<std::vector<char>> input_bufs;
    static std::vector<executorch::extension::TensorPtr> input_tensors;
    input_bufs.resize(6);
    input_tensors.clear();

    for (int i = 0; i < 6; i++) {
      std::string path = FLAGS_input_dir + "/" + specs[i].name + ".bin";
      input_bufs[i] = read_file(path);
      input_tensors.push_back(
          from_blob(input_bufs[i].data(), specs[i].shape, specs[i].dtype));
      inputs.push_back(*input_tensors.back());
    }
  } else {
    // Generate deterministic test inputs
    auto to_bf16 = [](float f) -> uint16_t {
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(float));
      return static_cast<uint16_t>(bits >> 16);
    };

    static std::vector<uint16_t> qk_data(B * T * H * K);
    for (size_t i = 0; i < qk_data.size(); i++)
      qk_data[i] = to_bf16(static_cast<float>(i % 100) * 0.01f - 0.5f);
    static auto v_data = std::vector<uint16_t>(qk_data.begin(), qk_data.end());
    static std::vector<uint16_t> g_data(B * T * H, to_bf16(-0.5f));
    static std::vector<uint16_t> beta_data(B * T * H, to_bf16(0.5f));
    static std::vector<uint16_t> state_data(B * H * K * V, to_bf16(0.0f));

    static std::vector<executorch::extension::TensorPtr> default_tensors;
    default_tensors.clear();
    default_tensors.push_back(from_blob(
        qk_data.data(), {B, T, H, K}, exec_aten::ScalarType::BFloat16));
    default_tensors.push_back(from_blob(
        qk_data.data(), {B, T, H, K}, exec_aten::ScalarType::BFloat16));
    default_tensors.push_back(from_blob(
        v_data.data(), {B, T, H, V}, exec_aten::ScalarType::BFloat16));
    default_tensors.push_back(
        from_blob(g_data.data(), {B, T, H}, exec_aten::ScalarType::BFloat16));
    default_tensors.push_back(from_blob(
        beta_data.data(), {B, T, H}, exec_aten::ScalarType::BFloat16));
    default_tensors.push_back(from_blob(
        state_data.data(), {B, H, K, V}, exec_aten::ScalarType::BFloat16));
    for (auto& t : default_tensors)
      inputs.push_back(*t);
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
      // Output tensors are on host memory (CUDA delegate copies back to CPU)
      std::string path =
          FLAGS_output_dir + "/output_" + std::to_string(i) + ".bin";
      write_file(path, t.const_data_ptr(), t.nbytes());
      printf("  Saved to %s (%zu bytes)\n", path.c_str(), (size_t)t.nbytes());
    }
  }

  printf("SUCCESS\n");
  return 0;
}
