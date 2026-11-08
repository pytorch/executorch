/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Runs two Metal programs in one process. The second is loaded and run, then
// the first is loaded, run and destroyed, and the second is run again.
// Destroying the first must free everything it made, and nothing of the
// second's. Like executor_runner, every input is all ones, and the outputs of
// the second program's last run are written to <output_file>-<index>.bin
// (tests/test_modules.py).
//
// Usage: metal_multi_delegate_runner <first.pte> <second.pte> <output_file>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Result;

namespace {

Result<std::vector<EValue>> run_with_ones(Module& module) {
  auto meta = module.method_meta("forward");
  if (!meta.ok()) {
    return meta.error();
  }
  std::vector<TensorPtr> tensors;
  std::vector<EValue> inputs;
  for (size_t i = 0; i < meta->num_inputs(); ++i) {
    auto input = meta->input_tensor_meta(i);
    if (!input.ok()) {
      return input.error();
    }
    const auto sizes = input->sizes();
    tensors.push_back(executorch::extension::ones(
        {sizes.begin(), sizes.end()}, input->scalar_type()));
    inputs.emplace_back(tensors.back());
  }
  return module.execute("forward", inputs);
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(
        stderr, "Usage: %s <first.pte> <second.pte> <output_file>\n", argv[0]);
    return 1;
  }
  executorch::runtime::runtime_init();

  Module second(argv[2]);
  if (!run_with_ones(second).ok()) {
    ET_LOG(Error, "The second program failed to run");
    return 1;
  }
  // The handles the shims hold for the second program.
  const size_t handles = executorch::backends::metal::tensors.size();

  auto first = std::make_unique<Module>(argv[1]);
  if (!run_with_ones(*first).ok()) {
    ET_LOG(Error, "The first program failed to run");
    return 1;
  }
  // Destroys the first program's delegate.
  first.reset();
  if (executorch::backends::metal::tensors.size() != handles) {
    ET_LOG(
        Error,
        "Destroying the first program left %zu handles, expected %zu",
        executorch::backends::metal::tensors.size(),
        handles);
    return 1;
  }

  auto outputs = run_with_ones(second);
  if (!outputs.ok()) {
    ET_LOG(Error, "The second program failed after the first was destroyed");
    return 1;
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    if (!(*outputs)[i].isTensor()) {
      continue;
    }
    const auto& tensor = (*outputs)[i].toTensor();
    const std::string path =
        std::string(argv[3]) + "-" + std::to_string(i) + ".bin";
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr ||
        std::fwrite(tensor.const_data_ptr(), 1, tensor.nbytes(), file) !=
            tensor.nbytes()) {
      ET_LOG(Error, "Could not write %s", path.c_str());
      if (file != nullptr) {
        std::fclose(file);
      }
      return 1;
    }
    std::fclose(file);
  }
  return 0;
}
