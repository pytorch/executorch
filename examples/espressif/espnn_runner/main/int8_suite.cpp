/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <runner.h>

#include <cstdio>
#include <cstring>
#include <new>

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <sdkconfig.h>

#include <espnn/int8_interpreter.h>

namespace executorch {
namespace examples {
namespace espressif {

using ::executorch::backends::espressif::Int8Interpreter;
using ::executorch::backends::espressif::Int8RuntimeOptions;
using ::executorch::backends::espressif::Int8TensorInfo;
using ::executorch::backends::espressif::KernelMode;
using ::executorch::backends::espressif::MAX_GRAPH_IO;
using ::executorch::backends::espressif::read_le_u32;

namespace {

constexpr size_t BUNDLE_HEADER_SIZE = 12;
constexpr size_t BUNDLE_NAME_BYTES = 32;
constexpr size_t BUNDLE_RECORD_SIZE = BUNDLE_NAME_BYTES + 16;
constexpr size_t XTV2_HEADER_SIZE = 20;

bool in_bounds(size_t offset, size_t length, size_t total) {
  return offset <= total && length <= total - offset;
}

struct VectorsV2 {
  uint32_t samples;
  uint32_t inputs;
  uint32_t outputs;
  uint32_t input_bytes[MAX_GRAPH_IO];
  uint32_t output_bytes[MAX_GRAPH_IO];
  size_t stride;
  const uint8_t* data;
};

bool parse_vectors_v2(const uint8_t* data, size_t size, VectorsV2& v) {
  if (size < XTV2_HEADER_SIZE || std::memcmp(data, "XTV2", 4) != 0 ||
      read_le_u32(data + 4) != 2) {
    return false;
  }
  v.samples = read_le_u32(data + 8);
  v.inputs = read_le_u32(data + 12);
  v.outputs = read_le_u32(data + 16);
  if (v.samples == 0 || v.inputs == 0 || v.outputs == 0 ||
      v.inputs > MAX_GRAPH_IO || v.outputs > MAX_GRAPH_IO)
    return false;
  const size_t table = XTV2_HEADER_SIZE + 4U * (v.inputs + v.outputs);
  if (size < table)
    return false;
  v.stride = 0;
  for (uint32_t i = 0; i < v.inputs; ++i) {
    v.input_bytes[i] = read_le_u32(data + XTV2_HEADER_SIZE + 4 * i);
    v.stride += v.input_bytes[i];
  }
  for (uint32_t i = 0; i < v.outputs; ++i) {
    v.output_bytes[i] =
        read_le_u32(data + XTV2_HEADER_SIZE + 4 * (v.inputs + i));
    v.stride += v.output_bytes[i];
  }
  v.data = data + table;
  return v.stride != 0 && (size - table) / v.stride >= v.samples &&
      table + v.stride * v.samples == size;
}

struct CaseResult {
  uint32_t exact_samples;
  uint32_t mismatched;
  int32_t worst;
  bool ansi_exact;
  int64_t total_us;
  int64_t bench_min_us;
  int64_t bench_max_us;
  int64_t bench_total_us;
  uint32_t bench_iterations;
};

uint32_t compare(const int8_t* a, const int8_t* b, size_t n, int32_t& worst) {
  uint32_t mismatches = 0;
  for (size_t i = 0; i < n; ++i) {
    int32_t d = static_cast<int32_t>(a[i]) - b[i];
    if (d < 0)
      d = -d;
    if (d != 0)
      ++mismatches;
    if (d > worst)
      worst = d;
  }
  return mismatches;
}

bool run_sample(
    Int8Interpreter& interpreter,
    const VectorsV2& v,
    uint32_t index,
    uint32_t& mismatched,
    int32_t& worst) {
  const uint8_t* sample = v.data + v.stride * index;
  for (uint32_t i = 0; i < v.inputs; ++i) {
    std::memcpy(interpreter.input_data(i), sample, v.input_bytes[i]);
    sample += v.input_bytes[i];
  }
  if (!interpreter.invoke())
    return false;
  mismatched = 0;
  for (uint32_t o = 0; o < v.outputs; ++o) {
    mismatched += compare(
        interpreter.output_data(o),
        reinterpret_cast<const int8_t*>(sample),
        v.output_bytes[o],
        worst);
    sample += v.output_bytes[o];
  }
  return true;
}

bool run_case(
    const char* name,
    const uint8_t* blob,
    size_t blob_size,
    const uint8_t* vectors,
    size_t vectors_size,
    uint32_t index,
    uint32_t count) {
  VectorsV2 v{};
  if (!parse_vectors_v2(vectors, vectors_size, v)) {
    std::printf(
        "SUITE CASE %u/%u name=%s status=FAIL reason=vectors\n",
        static_cast<unsigned>(index),
        static_cast<unsigned>(count),
        name);
    return false;
  }
  Int8Interpreter* interpreter = new (std::nothrow) Int8Interpreter();
  Int8RuntimeOptions options{};
#if CONFIG_ESPNN_RUNNER_DUAL_CORE
  options.dual_core = true;
#endif
#if CONFIG_ESPNN_RUNNER_CONSTANTS_TO_PSRAM
  options.copy_constants_to_psram = true;
#endif
  if (interpreter == nullptr || !interpreter->load(blob, blob_size, options)) {
    std::printf(
        "SUITE CASE %u/%u name=%s status=FAIL reason=load\n",
        static_cast<unsigned>(index),
        static_cast<unsigned>(count),
        name);
    delete interpreter;
    return false;
  }
  bool shapes_ok = interpreter->input_count() == v.inputs &&
      interpreter->output_count() == v.outputs;
  for (uint32_t i = 0; shapes_ok && i < v.inputs; ++i) {
    Int8TensorInfo info{};
    shapes_ok =
        interpreter->input_info(i, info) && info.bytes == v.input_bytes[i];
  }
  for (uint32_t o = 0; shapes_ok && o < v.outputs; ++o) {
    Int8TensorInfo info{};
    shapes_ok =
        interpreter->output_info(o, info) && info.bytes == v.output_bytes[o];
  }
  if (!shapes_ok) {
    std::printf(
        "SUITE CASE %u/%u name=%s status=FAIL reason=io-shapes\n",
        static_cast<unsigned>(index),
        static_cast<unsigned>(count),
        name);
    delete interpreter;
    return false;
  }
  CaseResult result{};
  bool ok = true;
  for (uint32_t s = 0; ok && s < v.samples; ++s) {
    uint32_t mismatched = 0;
    ok = run_sample(*interpreter, v, s, mismatched, result.worst);
    result.mismatched += mismatched;
    if (ok && mismatched == 0)
      ++result.exact_samples;
    result.total_us += interpreter->last_invoke_us();
  }
  result.ansi_exact = true;
#if CONFIG_ESPNN_RUNNER_VERIFY_ANSI
  if (ok) {
    uint32_t mismatched = 0;
    int32_t worst = 0;
    interpreter->set_kernel_mode(KernelMode::REFERENCE);
    ok = run_sample(*interpreter, v, 0, mismatched, worst);
    interpreter->set_kernel_mode(KernelMode::OPTIMIZED);
    result.ansi_exact = ok && mismatched == 0;
  }
#endif
  result.bench_min_us = INT64_MAX;
  for (uint32_t i = 0; ok && i < CONFIG_ESPNN_RUNNER_BENCH_ITERATIONS; ++i) {
    ok = interpreter->invoke();
    const int64_t elapsed = interpreter->last_invoke_us();
    result.bench_total_us += elapsed;
    if (elapsed < result.bench_min_us)
      result.bench_min_us = elapsed;
    if (elapsed > result.bench_max_us)
      result.bench_max_us = elapsed;
    ++result.bench_iterations;
  }
  if (result.bench_iterations == 0)
    result.bench_min_us = 0;
  const bool pass =
      ok && result.exact_samples == v.samples && result.ansi_exact;
  std::printf(
      "SUITE CASE %u/%u name=%s ops=%u inputs=%u outputs=%u samples=%u exact=%u "
      "mismatched=%u worst=%ld ansi=%d us=%lld min_us=%lld max_us=%lld iterations=%u "
      "arena=%u arena_internal=%d constants_in_ram=%d blob=%u status=%s\n",
      static_cast<unsigned>(index),
      static_cast<unsigned>(count),
      name,
      static_cast<unsigned>(interpreter->graph().header.num_operations),
      static_cast<unsigned>(v.inputs),
      static_cast<unsigned>(v.outputs),
      static_cast<unsigned>(v.samples),
      static_cast<unsigned>(result.exact_samples),
      static_cast<unsigned>(result.mismatched),
      static_cast<long>(result.worst),
      result.ansi_exact ? 1 : 0,
      static_cast<long long>(
          result.bench_iterations
              ? result.bench_total_us / result.bench_iterations
              : result.total_us / v.samples),
      static_cast<long long>(result.bench_min_us),
      static_cast<long long>(result.bench_max_us),
      static_cast<unsigned>(result.bench_iterations),
      static_cast<unsigned>(interpreter->arena_bytes()),
      interpreter->arena_in_internal_ram() ? 1 : 0,
      interpreter->constants_copied() ? 1 : 0,
      static_cast<unsigned>(blob_size),
      pass ? "PASS" : "FAIL");
  if (!pass)
    interpreter->print_profile();
  delete interpreter;
  return pass;
}

} // namespace

bool run_int8_suite(const uint8_t* data, size_t size) {
  std::printf("\n========================================\n");
  std::printf(" ESP-NN INT8 OPERATOR SUITE\n");
  std::printf("========================================\n");
  if (!is_int8_bundle(data, size) || read_le_u32(data + 4) != 1) {
    std::printf("SUITE RESULT: FAIL reason=bundle\n");
    return false;
  }
  const uint32_t count = read_le_u32(data + 8);
  if (count == 0 ||
      !in_bounds(
          BUNDLE_HEADER_SIZE,
          static_cast<size_t>(count) * BUNDLE_RECORD_SIZE,
          size)) {
    std::printf("SUITE RESULT: FAIL reason=bundle-table\n");
    return false;
  }
  uint32_t passed = 0;
  const int64_t started = esp_timer_get_time();
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* record =
        data + BUNDLE_HEADER_SIZE + static_cast<size_t>(i) * BUNDLE_RECORD_SIZE;
    char name[BUNDLE_NAME_BYTES + 1]{};
    std::memcpy(name, record, BUNDLE_NAME_BYTES);
    const uint32_t blob_offset = read_le_u32(record + BUNDLE_NAME_BYTES);
    const uint32_t blob_size = read_le_u32(record + BUNDLE_NAME_BYTES + 4);
    const uint32_t vectors_offset = read_le_u32(record + BUNDLE_NAME_BYTES + 8);
    const uint32_t vectors_size = read_le_u32(record + BUNDLE_NAME_BYTES + 12);
    if (!in_bounds(blob_offset, blob_size, size) ||
        !in_bounds(vectors_offset, vectors_size, size)) {
      std::printf(
          "SUITE CASE %u/%u name=%s status=FAIL reason=bounds\n",
          static_cast<unsigned>(i + 1),
          static_cast<unsigned>(count),
          name);
      continue;
    }
    if (run_case(
            name,
            data + blob_offset,
            blob_size,
            data + vectors_offset,
            vectors_size,
            i + 1,
            count))
      ++passed;
  }
  std::printf(
      "Internal RAM free %u, PSRAM free %u, suite time %lld ms\n",
      static_cast<unsigned>(
          heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
      static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)),
      static_cast<long long>((esp_timer_get_time() - started) / 1000));
  std::printf(
      "SUITE RESULT: %s cases=%u passed=%u\n",
      passed == count ? "PASS" : "FAIL",
      static_cast<unsigned>(count),
      static_cast<unsigned>(passed));
  return passed == count;
}

} // namespace espressif
} // namespace examples
} // namespace executorch
