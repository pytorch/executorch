/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * ExecuTorch runner for Cadence Xtensa cores, intended to run on the
 * Xtensa Instruction Set Simulator (xt-run / xt-run --turbo).
 *
 * Reads a .pte from the host filesystem via xt-run semi-hosting,
 * executes the first method with all-ones inputs (via
 * prepare_input_tensors), and prints the outputs.
 *
 * Argument parsing is plain argv inspection — gflags pulls in
 * mkdir(2), which Xtensa newlib does not declare, breaking
 * cross-compile. Mirrors the same approach Arm and NXP take in their
 * embedded runners.
 *
 * Usage:
 *   xt-run --turbo cadence_executor_runner --model_path=add.pte
 *   xt-run --mem_model --summary cadence_executor_runner --model_path=add.pte
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
// patternlint-disable executorch-cpp-nostdinc
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {

// 18 KB has historically been enough for the cadence "hello world"
// models (add, simple MLP). Bump if you hit MemoryAllocator overflow
// at load_method time.
constexpr std::size_t kMethodAllocatorBytes = 18 * 1024U;
uint8_t method_allocator_pool[kMethodAllocatorBytes];

const char* parse_model_path(int argc, char** argv) {
  constexpr char kFlag[] = "--model_path=";
  constexpr std::size_t kFlagLen = sizeof(kFlag) - 1;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], kFlag, kFlagLen) == 0) {
      // Static so the returned pointer stays valid after parse returns.
      static std::string path{argv[i] + kFlagLen};
      return path.c_str();
    }
  }
  return "model.pte";
}

bool slurp(const char* path, std::vector<uint8_t>* out) {
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    ET_LOG(Error, "fopen('%s') failed", path);
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (sz <= 0) {
    ET_LOG(Error, "model file '%s' is empty or stat failed", path);
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<std::size_t>(sz));
  std::size_t n = std::fread(out->data(), 1, sz, f);
  std::fclose(f);
  if (static_cast<long>(n) != sz) {
    ET_LOG(Error, "fread short on '%s': %zu/%ld", path, n, sz);
    return false;
  }
  ET_LOG(Info, "Loaded %ld bytes from %s", sz, path);
  return true;
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  std::vector<uint8_t> model;
  const char* path = parse_model_path(argc, argv);
  if (!slurp(path, &model)) {
    return 1;
  }

  auto loader =
      executorch::extension::BufferDataLoader(model.data(), model.size());

  Result<executorch::runtime::Program> program =
      executorch::runtime::Program::load(&loader);
  if (!program.ok()) {
    ET_LOG(Error, "Program::load failed: 0x%" PRIx32, program.error());
    return 1;
  }
  ET_LOG(Info, "Model buffer loaded, has %u methods", program->num_methods());

  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Running method %s", method_name);

  Result<executorch::runtime::MethodMeta> method_meta =
      program->method_meta(method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Error,
        "method_meta('%s') failed: 0x%x",
        method_name,
        (unsigned int)method_meta.error());
    return 1;
  }

  executorch::runtime::MemoryAllocator method_allocator(
      sizeof(method_allocator_pool), method_allocator_pool);

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<executorch::runtime::Span<uint8_t>> planned_spans;
  const std::size_t num_planned = method_meta->num_memory_planned_buffers();
  for (std::size_t id = 0; id < num_planned; ++id) {
    const std::size_t buffer_size = static_cast<std::size_t>(
        method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(Info, "Setting up planned buffer %zu, size %zu", id, buffer_size);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }
  executorch::runtime::HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  executorch::runtime::MemoryManager memory_manager(
      &method_allocator, &planned_memory);

  Result<executorch::runtime::Method> method =
      program->load_method(method_name, &memory_manager);
  if (!method.ok()) {
    ET_LOG(
        Error,
        "load_method('%s') failed: 0x%" PRIx32,
        method_name,
        method.error());
    return 1;
  }
  ET_LOG(Info, "Method loaded.");

  auto cleanup = executorch::extension::prepare_input_tensors(*method);
  if (!cleanup.ok()) {
    ET_LOG(
        Error,
        "prepare_input_tensors failed: 0x%x",
        (unsigned int)cleanup.error());
    return 1;
  }
  ET_LOG(Info, "Starting model execution...");

  Error status = method->execute();
  if (status != Error::Ok) {
    ET_LOG(Error, "execute() failed for '%s': 0x%" PRIx32, method_name, status);
    return 1;
  }
  ET_LOG(Info, "Model executed successfully.");

  std::vector<executorch::runtime::EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
  if (status != Error::Ok) {
    ET_LOG(
        Error,
        "get_outputs() failed for '%s': 0x%" PRIx32,
        method_name,
        status);
    return 1;
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (!outputs[i].isTensor()) {
      ET_LOG(Info, "output[%zu]: non-tensor", i);
      continue;
    }
    const auto& t = outputs[i].toTensor();
    const float* p = t.const_data_ptr<float>();
    const std::size_t n = t.numel() < 20 ? t.numel() : 20;
    ET_LOG(Info, "First %zu elements of output %zu:", n, i);
    for (std::size_t j = 0; j < n; ++j) {
      ET_LOG(Info, "  %f", p[j]);
    }
  }
  return 0;
}
