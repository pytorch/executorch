/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/gemma4/runner/gemma4_spec_runner.h>

#include <executorch/backends/webgpu/runtime/WebGPUBackend.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define ET_WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define ET_WASM_EXPORT
#endif

namespace {

using ::executorch::backends::webgpu::webgpu_backend_execution_attestation_json;
using ::executorch::examples::gemma4::Gemma4SpecRunner;
using ::executorch::runtime::Error;

constexpr const char* kMethodName = "k2_round";
constexpr size_t kExpectedTensorDataPaths = 3;

std::unique_ptr<Gemma4SpecRunner> runner;
std::string execution_attestation_json;
std::string profile_json;

std::vector<std::string> split_paths(const char* paths) {
  std::vector<std::string> result;
  if (paths == nullptr) {
    return result;
  }
  const std::string value(paths);
  size_t start = 0;
  while (start <= value.size()) {
    const size_t end = value.find('\n', start);
    const std::string path = value.substr(start, end - start);
    if (!path.empty()) {
      result.push_back(path);
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return result;
}

} // namespace

extern "C" {

ET_WASM_EXPORT int et_init() {
  if (runner == nullptr) {
    runner = std::make_unique<Gemma4SpecRunner>();
  }
  return 1;
}

ET_WASM_EXPORT int et_load(
    const char* pte_path,
    const char* tensor_data_paths,
    const char* method) {
  if (runner == nullptr || pte_path == nullptr || method == nullptr ||
      std::string(method) != kMethodName) {
    return 0;
  }
  auto paths = split_paths(tensor_data_paths);
  if (paths.size() != kExpectedTensorDataPaths) {
    return 0;
  }
  return runner->load(pte_path, std::move(paths)) == Error::Ok ? 1 : 0;
}

ET_WASM_EXPORT int et_unload() {
  if (runner == nullptr) {
    return 1;
  }
  const Error error = runner->unload();
  runner.reset();
  return error == Error::Ok ? 1 : 0;
}

ET_WASM_EXPORT int et_reset() {
  return runner != nullptr && runner->reset() == Error::Ok ? 1 : 0;
}

ET_WASM_EXPORT int
et_prefill_batch(const int32_t* ids, int32_t count, int32_t start_position) {
  const int64_t actual_count = count < 0 ? -static_cast<int64_t>(count) : count;
  if (runner == nullptr || ids == nullptr || actual_count <= 0 ||
      actual_count > INT32_MAX || start_position < 0) {
    return -1;
  }
  std::vector<int64_t> input_ids(ids, ids + actual_count);
  auto result = runner->prefill(input_ids, start_position);
  return result.ok() ? static_cast<int32_t>(result.get()) : -2;
}

ET_WASM_EXPORT void et_prefill_step(int32_t token, int32_t position) {
  if (runner != nullptr) {
    (void)runner->prefill_step(token, position);
  }
}

ET_WASM_EXPORT int et_step(int32_t token, int32_t position) {
  if (runner == nullptr) {
    return -1;
  }
  auto result = runner->step(token, position);
  return result.ok() ? static_cast<int32_t>(result.get()) : -2;
}

ET_WASM_EXPORT int et_mtp_execute_count() {
  return runner == nullptr ? 0 : static_cast<int>(runner->execute_count());
}

ET_WASM_EXPORT int et_mtp_accepted_drafts() {
  return runner == nullptr ? 0 : static_cast<int>(runner->accepted_drafts());
}

ET_WASM_EXPORT int et_mtp_buffered_tokens() {
  return runner == nullptr ? 0 : static_cast<int>(runner->buffered_tokens());
}

ET_WASM_EXPORT int et_mtp_execute(
    const int32_t* ids,
    int32_t count,
    int32_t start_position,
    int32_t is_round,
    int32_t donor_length,
    int32_t* output) {
  if (runner == nullptr || ids == nullptr || output == nullptr || count <= 0 ||
      start_position < 0 || (is_round != 0 && is_round != 1)) {
    return 0;
  }
  std::vector<int64_t> input_ids(ids, ids + count);
  std::vector<int64_t> positions;
  positions.reserve(count);
  for (int32_t index = 0; index < count; ++index) {
    positions.push_back(static_cast<int64_t>(start_position) + index);
  }
  auto result =
      runner->execute(input_ids, positions, is_round != 0, donor_length);
  if (!result.ok()) {
    return 0;
  }
  output[0] = static_cast<int32_t>(result->candidates[0]);
  output[1] = static_cast<int32_t>(result->candidates[1]);
  output[2] = static_cast<int32_t>(result->target_greedy[0]);
  output[3] = static_cast<int32_t>(result->target_greedy[1]);
  output[4] = static_cast<int32_t>(result->target_greedy[2]);
  output[5] = static_cast<int32_t>(result->match_count);
  output[6] = static_cast<int32_t>(result->bonus);
  return 1;
}

ET_WASM_EXPORT const char* et_mtp_execution_attestation() {
  execution_attestation_json = webgpu_backend_execution_attestation_json();
  return execution_attestation_json.c_str();
}

ET_WASM_EXPORT void et_profile_enable(int enabled) {
  if (runner != nullptr) {
    runner->set_profiling_enabled(enabled != 0);
  }
}

ET_WASM_EXPORT const char* et_profile() {
  if (runner == nullptr) {
    static const std::string unsupported =
        "{\"schemaVersion\":1,\"supported\":false,"
        "\"fresh\":false,\"valid\":false,\"context_generation\":0,"
        "\"querypool_generation\":0,\"execute_generation\":0,"
        "\"total_kernel_ms\":0,"
        "\"pass_span_ms\":0,\"interpass_gap_ms\":0,\"perop\":[]}";
    return unsupported.c_str();
  }
  profile_json = runner->profile_json();
  return profile_json.c_str();
}

} // extern "C"
