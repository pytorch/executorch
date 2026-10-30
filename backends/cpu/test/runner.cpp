// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/KernelProvider.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool_guard.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  constexpr int max_preferences = 8;
  if (argc < 3 || argc > 3 + max_preferences) {
    fprintf(
        stderr,
        "Usage: %s model.pte golden.bin [provider[/implementation] ...]\n",
        argv[0]);
    return 1;
  }
  executorch::extension::threadpool::NoThreadPoolGuard threadpool_guard;
  executorch::runtime::BackendOptions<1 + 2 * max_preferences> options;
  executorch::runtime::LoadBackendOptionsMap backends;
  executorch::extension::Module module(argv[1]);
  if (argc > 3) {
    using executorch::runtime::Error;
    if (options.set_option("preference_count", argc - 3) != Error::Ok) {
      return 1;
    }
    for (int index = 0; index < argc - 3; ++index) {
      const std::string entry = argv[index + 3];
      const auto separator = entry.find('/');
      const auto provider = entry.substr(0, separator);
      const auto implementation = separator == std::string::npos
          ? std::string{}
          : entry.substr(separator + 1);
      if (provider.empty() ||
          provider.size() >= executorch::runtime::kMaxOptionValueLength ||
          implementation.size() >= executorch::runtime::kMaxOptionValueLength) {
        return 1;
      }
      // set_option requires const char(&)[N].
      // NOLINTNEXTLINE(facebook-hte-CArray)
      char key[executorch::runtime::kMaxOptionKeyLength];
      std::snprintf(key, sizeof(key), "preferred_provider_%d", index);
      if (options.set_option(key, provider.c_str()) != Error::Ok) {
        return 1;
      }
      std::snprintf(key, sizeof(key), "preferred_implementation_%d", index);
      if (options.set_option(key, implementation.c_str()) != Error::Ok) {
        return 1;
      }
    }
    if (backends.set_options("CpuBackend", options.view()) != Error::Ok ||
        module.load(backends) != Error::Ok) {
      return 1;
    }
  }
  auto meta = module.method_meta("forward");
  if (!meta.ok() || meta->num_inputs() != 1 || meta->num_outputs() != 1) {
    fprintf(stderr, "Expected one input and one output\n");
    return 1;
  }
  auto input_meta = meta->input_tensor_meta(0);
  if (!input_meta.ok()) {
    return 1;
  }
  std::vector<executorch::aten::SizesType> sizes(
      input_meta->sizes().begin(), input_meta->sizes().end());
  std::vector<float> data(input_meta->nbytes() / sizeof(float));
  for (size_t index = 0; index < data.size(); ++index) {
    data[index] = static_cast<float>(index % 251) / 250.0f;
  }
  auto input = executorch::extension::from_blob(data.data(), sizes);
  for (int run = 0; run < 2; ++run) {
    auto outputs = module.forward(executorch::runtime::EValue(*input));
    if (!outputs.ok()) {
      fprintf(
          stderr,
          "Execution failed: 0x%x\n",
          static_cast<unsigned>(outputs.error()));
      return 1;
    }
    const auto& output = outputs->at(0).toTensor();
    std::vector<float> golden(output.numel());
    std::ifstream file(argv[2], std::ios::binary);
    // Golden files store native FP32 bytes.
    // cppcheck-suppress invalidPointerCast
    if (!file.read(reinterpret_cast<char*>(golden.data()), output.nbytes()) ||
        file.peek() != std::ifstream::traits_type::eof()) {
      fprintf(stderr, "Golden file size mismatch\n");
      return 1;
    }
    const auto* actual = output.const_data_ptr<float>();
    float max_error = 0;
    for (size_t index = 0; index < golden.size(); ++index) {
      const float error = std::abs(actual[index] - golden[index]);
      max_error = std::max(max_error, error);
      if (!std::isfinite(actual[index]) || !std::isfinite(golden[index]) ||
          error > 1e-4f + 1e-4f * std::abs(golden[index])) {
        fprintf(
            stderr,
            "Mismatch at %zu: actual=%g golden=%g\n",
            index,
            actual[index],
            golden[index]);
        return 1;
      }
    }
    printf(
        "Golden match: run=%d elements=%zu max_abs_error=%g\n",
        run,
        golden.size(),
        max_error);
  }
  return 0;
}
