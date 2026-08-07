/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runner/webgpu_model_loader.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#ifdef WGPU_BACKEND_ENABLE_PROFILING
#include <executorch/backends/webgpu/runtime/WebGPUQueryPool.h>
#endif
#include <executorch/extension/tensor/tensor_ptr.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define GEMMA4_WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define GEMMA4_WASM_EXPORT
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using executorch::aten::ScalarType;
using executorch::backends::webgpu::WebGPUContext;
using executorch::backends::webgpu::WebGPUModelLoadSpec;
using executorch::backends::webgpu::compare_and_set_default_webgpu_context;
using executorch::backends::webgpu::create_webgpu_context;
using executorch::backends::webgpu::destroy_webgpu_context;
#ifdef WGPU_BACKEND_ENABLE_PROFILING
using executorch::backends::webgpu::g_last_route_conflict_count;
using executorch::backends::webgpu::g_last_route_mask;
#endif
using executorch::backends::webgpu::get_default_webgpu_context;
using executorch::backends::webgpu::load_webgpu_model;
using executorch::extension::Module;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

constexpr char kMethodName[] = "text_decoder";
constexpr size_t kExpectedPtdCount = 3;
constexpr int kMaxInputLength = 512;
constexpr int kMaxSequenceLength = 8960;

WebGPUContext g_context;
bool g_owns_default_context = false;
std::unique_ptr<Module> g_module;
int g_last_prefill_tokens = 0;

void reset_runtime_observations() {
  g_last_prefill_tokens = 0;
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  if (g_context.querypool) {
    g_context.querypool->reset(0);
  }
  g_last_route_mask = 0;
  g_last_route_conflict_count = 0;
#endif
}

void unload_model() {
  g_module.reset();
  reset_runtime_observations();
}

void release_context() {
  if (!g_owns_default_context) {
    return;
  }
  compare_and_set_default_webgpu_context(&g_context, nullptr);
  destroy_webgpu_context(g_context);
  g_owns_default_context = false;
}

bool is_long_tensor(const executorch::runtime::Result<TensorInfo>& info) {
  return info.ok() && info.get().scalar_type() == ScalarType::Long;
}

bool validate_method_meta(const MethodMeta& meta) {
  if (meta.num_inputs() != 2 || meta.num_outputs() != 1 ||
      !meta.uses_backend("VulkanBackend")) {
    return false;
  }
  const auto input_ids_tag = meta.input_tag(0);
  const auto input_pos_tag = meta.input_tag(1);
  const auto output_tag = meta.output_tag(0);
  if (!input_ids_tag.ok() || input_ids_tag.get() != Tag::Tensor ||
      !input_pos_tag.ok() || input_pos_tag.get() != Tag::Tensor ||
      !output_tag.ok() || output_tag.get() != Tag::Tensor) {
    return false;
  }

  const auto input_ids = meta.input_tensor_meta(0);
  const auto input_pos = meta.input_tensor_meta(1);
  const auto output = meta.output_tensor_meta(0);
  if (!is_long_tensor(input_ids) || !is_long_tensor(input_pos) ||
      !is_long_tensor(output)) {
    return false;
  }
  const auto input_ids_sizes = input_ids.get().sizes();
  const auto input_pos_sizes = input_pos.get().sizes();
  const auto output_sizes = output.get().sizes();
  return input_ids_sizes.size() == 2 && input_ids_sizes[0] == 1 &&
      input_pos_sizes.size() == 1 && output_sizes.size() == 2 &&
      output_sizes[0] == 1 && output_sizes[1] == 1 &&
      output.get().nbytes() == sizeof(int64_t);
}

bool validate_loaded_method(Module& module) {
  auto meta = module.method_meta(kMethodName);
  return meta.ok() && validate_method_meta(meta.get());
}

std::vector<std::string> parse_ptd_paths(const char* data_paths) {
  std::vector<std::string> ptd_paths;
  if (data_paths == nullptr) {
    return ptd_paths;
  }
  std::istringstream stream{std::string(data_paths)};
  std::string path;
  while (std::getline(stream, path)) {
    if (!path.empty()) {
      ptd_paths.push_back(std::move(path));
    }
  }
  return ptd_paths;
}

int read_compact_token(const std::vector<EValue>& values) {
  if (values.size() != 1 || !values[0].isTensor()) {
    return -1;
  }
  const auto& output = values[0].toTensor();
  if (output.scalar_type() != ScalarType::Long || output.dim() != 2 ||
      output.size(0) != 1 || output.size(1) != 1 || output.numel() != 1 ||
      output.nbytes() != sizeof(int64_t)) {
    return -1;
  }
  const int64_t token = output.const_data_ptr<int64_t>()[0];
  if (token < 0 || token > std::numeric_limits<int>::max()) {
    return -1;
  }
  return static_cast<int>(token);
}

int execute_tokens(const int* tokens, int count, int position) {
  if (!g_module || tokens == nullptr || count <= 0 ||
      count > kMaxInputLength || position < 0 ||
      position > kMaxSequenceLength - count) {
    return -1;
  }

  std::vector<int64_t> input_ids(static_cast<size_t>(count));
  std::vector<int64_t> input_pos(static_cast<size_t>(count));
  for (int index = 0; index < count; ++index) {
    input_ids[static_cast<size_t>(index)] = tokens[index];
    input_pos[static_cast<size_t>(index)] = position + index;
  }
  auto input_ids_tensor = make_tensor_ptr(
      {1, count}, std::move(input_ids), {}, {}, ScalarType::Long);
  auto input_pos_tensor = make_tensor_ptr(
      {count}, std::move(input_pos), {}, {}, ScalarType::Long);
  auto result = g_module->execute(
      kMethodName,
      {EValue(std::move(input_ids_tensor)), EValue(std::move(input_pos_tensor))});
  if (!result.ok()) {
    std::printf("Gemma4 text_decoder failed: %d\n", (int)result.error());
    return -1;
  }
  return read_compact_token(result.get());
}

} // namespace

extern "C" {

GEMMA4_WASM_EXPORT int et_init() {
  unload_model();
  release_context();
  try {
    g_context = create_webgpu_context();
  } catch (const std::exception& error) {
    std::printf("Gemma4 WebGPU initialization failed: %s\n", error.what());
    return 0;
  }
  if (!compare_and_set_default_webgpu_context(nullptr, &g_context)) {
    destroy_webgpu_context(g_context);
    std::printf("Gemma4 WebGPU context is already owned\n");
    return 0;
  }
  g_owns_default_context = true;
  return 1;
}

GEMMA4_WASM_EXPORT int et_load(
    const char* pte_path,
    const char* data_paths,
    const char* method_name) {
  unload_model();
  if (!g_owns_default_context || pte_path == nullptr || *pte_path == '\0' ||
      (method_name != nullptr && *method_name != '\0' &&
       std::string(method_name) != kMethodName)) {
    return 0;
  }
  auto ptd_paths = parse_ptd_paths(data_paths);
  if (ptd_paths.size() != kExpectedPtdCount) {
    std::printf(
        "Gemma4 requires exactly %zu ordered PTDs, got %zu\n",
        kExpectedPtdCount,
        ptd_paths.size());
    return 0;
  }

  WebGPUModelLoadSpec spec;
  spec.pte_path = pte_path;
  spec.ptd_paths = std::move(ptd_paths);
  spec.required_methods = {kMethodName};
  spec.load_mode = Module::LoadMode::File;
  auto next = load_webgpu_model(std::move(spec));
  if (!next.ok() || !validate_loaded_method(*next.get())) {
    std::printf("Gemma4 PTE or text_decoder ABI validation failed\n");
    return 0;
  }
  g_module = std::move(next.get());
  reset_runtime_observations();
  return 1;
}

GEMMA4_WASM_EXPORT int et_unload() {
  unload_model();
  return 1;
}

GEMMA4_WASM_EXPORT int et_reset() {
  if (!g_module) {
    return 0;
  }
  g_module->unload_method(kMethodName);
  const Error error = g_module->load_method(kMethodName);
  if (error != Error::Ok || !validate_loaded_method(*g_module)) {
    unload_model();
    return 0;
  }
  reset_runtime_observations();
  return 1;
}

GEMMA4_WASM_EXPORT int et_step(int token, int position) {
  return execute_tokens(&token, 1, position);
}

GEMMA4_WASM_EXPORT void et_prefill_step(int token, int position) {
  (void)execute_tokens(&token, 1, position);
}

GEMMA4_WASM_EXPORT int
et_prefill_batch(const int* tokens, int count, int position) {
  if (count == 0 || count == std::numeric_limits<int>::min()) {
    return -1;
  }
  const bool discard_output = count < 0;
  const int live_count = discard_output ? -count : count;
  const int token = execute_tokens(tokens, live_count, position);
  if (token < 0) {
    return -1;
  }
  g_last_prefill_tokens = live_count;
  return discard_output ? 0 : token;
}

GEMMA4_WASM_EXPORT int et_get_last_prefill_token_count() {
  return g_last_prefill_tokens;
}

GEMMA4_WASM_EXPORT int et_get_route_contract_version() {
  return 3;
}

GEMMA4_WASM_EXPORT int et_get_last_route_mask() {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  return static_cast<int>(g_last_route_mask);
#else
  return 0;
#endif
}

GEMMA4_WASM_EXPORT int et_get_last_route_conflict_count() {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  return static_cast<int>(g_last_route_conflict_count);
#else
  return 0;
#endif
}

GEMMA4_WASM_EXPORT void et_profile_enable(int enabled) {
#if defined(_WIN32)
  _putenv_s("WEBGPU_TIMESTAMP_QUERY", enabled ? "1" : "");
#else
  if (enabled) {
    setenv("WEBGPU_TIMESTAMP_QUERY", "1", 1);
  } else {
    unsetenv("WEBGPU_TIMESTAMP_QUERY");
  }
#endif
}

GEMMA4_WASM_EXPORT const char* et_profile() {
  static std::string output;
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  const auto* context = get_default_webgpu_context();
  if (context == nullptr || !context->querypool ||
      !context->querypool->results_valid()) {
    output =
        "{\"perop\":[],\"total_kernel_ms\":0,\"pass_span_ms\":0,"
        "\"interpass_gap_ms\":0,\"supported\":false}";
    return output.c_str();
  }

  std::map<std::string, std::pair<double, uint32_t>> totals;
  double total_ns = 0.0;
  uint64_t first_begin = std::numeric_limits<uint64_t>::max();
  uint64_t last_end = 0;
  for (const auto& duration : context->querypool->results()) {
    totals[duration.kernel_name].first += duration.execution_duration_ns / 1e6;
    totals[duration.kernel_name].second++;
    total_ns += static_cast<double>(duration.execution_duration_ns);
    first_begin = std::min(first_begin, duration.start_time_ns);
    last_end = std::max(last_end, duration.end_time_ns);
  }
  const double total_ms = total_ns / 1e6;
  const double span_ms = last_end > first_begin
      ? static_cast<double>(last_end - first_begin) / 1e6
      : 0.0;
  const double gap_ms = std::max(0.0, span_ms - total_ms);
  output = "{\"perop\":[";
  bool first = true;
  for (const auto& entry : totals) {
    char item[320];
    const double percent =
        total_ms > 0.0 ? 100.0 * entry.second.first / total_ms : 0.0;
    std::snprintf(
        item,
        sizeof(item),
        "%s{\"op\":\"%s\",\"ms\":%.4f,\"calls\":%u,\"pct\":%.1f}",
        first ? "" : ",",
        entry.first.c_str(),
        entry.second.first,
        entry.second.second,
        percent);
    output += item;
    first = false;
  }
  char tail[256];
  std::snprintf(
      tail,
      sizeof(tail),
      "],\"total_kernel_ms\":%.4f,\"pass_span_ms\":%.4f,"
      "\"interpass_gap_ms\":%.4f,\"supported\":true}",
      total_ms,
      span_ms,
      gap_ms);
  output += tail;
#else
  output =
      "{\"perop\":[],\"total_kernel_ms\":0,\"pass_span_ms\":0,"
      "\"interpass_gap_ms\":0,\"supported\":false}";
#endif
  return output.c_str();
}

} // extern "C"

int main() {
  return 0;
}
