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

#include <cmath>
#include <cstdio>
#include <cstring>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <esp_heap_caps.h>
#include <esp_rom_crc.h>
#include <esp_timer.h>
#include <sdkconfig.h>

#include <espnn/espnn_backend.h>
#include <espnn/xts3_graph.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace executorch {
namespace examples {
namespace espressif {

using ::executorch::backends::espressif::delegate_stats;
using ::executorch::backends::espressif::DelegateStats;
using ::executorch::backends::espressif::read_le_u32;
using ::executorch::backends::espressif::register_espnn_backend;
using ::executorch::backends::espressif::reset_delegate_stats;

namespace {

constexpr size_t BUNDLE_HEADER_SIZE = 12;
constexpr size_t BUNDLE_NAME_BYTES = 32;
constexpr size_t BUNDLE_RECORD_SIZE = BUNDLE_NAME_BYTES + 16;
constexpr size_t XTE1_HEADER_SIZE = 28;
constexpr uint32_t XTE1_FLAG_EXACT = 1;
constexpr uint32_t XTE1_FLAG_CRC = 2;
constexpr size_t MAX_IO = 16;
constexpr size_t MAX_PLANNED_BUFFERS = 8;

bool in_bounds(size_t offset, size_t length, size_t total) {
  return offset <= total && length <= total - offset;
}

size_t element_size(uint32_t dtype) {
  if (dtype == static_cast<uint32_t>(ScalarType::Float))
    return 4;
  if (dtype == static_cast<uint32_t>(ScalarType::Char))
    return 1;
  if (dtype == static_cast<uint32_t>(ScalarType::Long))
    return 8;
  if (dtype == static_cast<uint32_t>(ScalarType::Int))
    return 4;
  return 0;
}

struct Vectors {
  uint32_t samples;
  uint32_t inputs;
  uint32_t outputs;
  float min_sqnr_db;
  uint32_t flags;
  uint32_t dtype[2 * MAX_IO];
  uint32_t numel[2 * MAX_IO];
  size_t stride;
  const uint8_t* data;
};

bool parse_vectors(const uint8_t* data, size_t size, Vectors& v) {
  if (size < XTE1_HEADER_SIZE || std::memcmp(data, "XTE1", 4) != 0 ||
      read_le_u32(data + 4) != 1) {
    return false;
  }
  v.samples = read_le_u32(data + 8);
  v.inputs = read_le_u32(data + 12);
  v.outputs = read_le_u32(data + 16);
  const uint32_t sqnr_bits = read_le_u32(data + 20);
  std::memcpy(&v.min_sqnr_db, &sqnr_bits, sizeof(float));
  v.flags = read_le_u32(data + 24);
  if (v.samples == 0 || v.inputs == 0 || v.outputs == 0 || v.inputs > MAX_IO ||
      v.outputs > MAX_IO) {
    return false;
  }
  const size_t table = XTE1_HEADER_SIZE + 8U * (v.inputs + v.outputs);
  if (size < table)
    return false;
  v.stride = 0;
  for (uint32_t i = 0; i < v.inputs + v.outputs; ++i) {
    v.dtype[i] = read_le_u32(data + XTE1_HEADER_SIZE + 8 * i);
    v.numel[i] = read_le_u32(data + XTE1_HEADER_SIZE + 8 * i + 4);
    const size_t bytes = element_size(v.dtype[i]) * v.numel[i];
    if (bytes == 0)
      return false;
    v.stride += (i >= v.inputs && (v.flags & XTE1_FLAG_CRC) != 0) ? 4U : bytes;
  }
  v.data = data + table;
  return (size - table) / v.stride >= v.samples &&
      table + v.stride * v.samples == size;
}

struct Pools {
  uint8_t* method_pool = nullptr;
  uint8_t* temp_pool = nullptr;
  uint8_t* planned[MAX_PLANNED_BUFFERS]{};
  size_t planned_count = 0;
  size_t planned_bytes = 0;

  ~Pools() {
    if (method_pool != nullptr)
      heap_caps_free(method_pool);
    if (temp_pool != nullptr)
      heap_caps_free(temp_pool);
    for (size_t i = 0; i < planned_count; ++i) {
      if (planned[i] != nullptr)
        heap_caps_free(planned[i]);
    }
  }
};

void* psram(size_t bytes) {
  void* pointer =
      heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (pointer == nullptr)
    pointer = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_8BIT);
  return pointer;
}

struct OutputScore {
  double signal;
  double noise;
  float max_abs;
  uint32_t mismatched;
};

void score_output(
    const Tensor& actual,
    uint32_t dtype,
    const uint8_t* expected,
    size_t count,
    OutputScore& score) {
  if (dtype == static_cast<uint32_t>(ScalarType::Float)) {
    const float* got = actual.const_data_ptr<float>();
    for (size_t i = 0; i < count; ++i) {
      float want = 0.0f;
      std::memcpy(&want, expected + 4 * i, sizeof(float));
      const double difference = static_cast<double>(got[i]) - want;
      score.signal += static_cast<double>(want) * want;
      score.noise += difference * difference;
      const float magnitude = static_cast<float>(std::fabs(difference));
      if (magnitude > score.max_abs)
        score.max_abs = magnitude;
      if (got[i] != want)
        ++score.mismatched;
    }
    return;
  }
  const uint8_t* got = static_cast<const uint8_t*>(actual.const_data_ptr());
  const size_t bytes = count * element_size(dtype);
  for (size_t i = 0; i < bytes; ++i) {
    if (got[i] != expected[i])
      ++score.mismatched;
  }
}

bool run_case(
    const char* name,
    const uint8_t* pte,
    size_t pte_size,
    const uint8_t* vectors,
    size_t vectors_size,
    uint32_t index,
    uint32_t count) {
  Vectors v{};
  if (!parse_vectors(vectors, vectors_size, v)) {
    std::printf(
        "ET CASE %u/%u name=%s status=FAIL reason=vectors\n",
        static_cast<unsigned>(index),
        static_cast<unsigned>(count),
        name);
    return false;
  }
  reset_delegate_stats();
  const int64_t load_started = esp_timer_get_time();
  BufferDataLoader loader(pte, pte_size);
  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    std::printf(
        "ET CASE %u/%u name=%s status=FAIL reason=program-load error=0x%x\n",
        static_cast<unsigned>(index),
        static_cast<unsigned>(count),
        name,
        static_cast<unsigned>(program.error()));
    return false;
  }
  Result<MethodMeta> meta = program->method_meta("forward");
  if (!meta.ok() || meta->num_memory_planned_buffers() > MAX_PLANNED_BUFFERS) {
    std::printf(
        "ET CASE %u/%u name=%s status=FAIL reason=method-meta\n",
        static_cast<unsigned>(index),
        static_cast<unsigned>(count),
        name);
    return false;
  }
  Pools pools;
  const size_t method_bytes =
      static_cast<size_t>(CONFIG_ESPNN_RUNNER_METHOD_POOL_KB) * 1024U;
  const size_t temp_bytes =
      static_cast<size_t>(CONFIG_ESPNN_RUNNER_TEMP_POOL_KB) * 1024U;
  pools.method_pool = static_cast<uint8_t*>(psram(method_bytes));
  pools.temp_pool = static_cast<uint8_t*>(psram(temp_bytes));
  Span<uint8_t> spans[MAX_PLANNED_BUFFERS];
  pools.planned_count = meta->num_memory_planned_buffers();
  bool allocated = pools.method_pool != nullptr && pools.temp_pool != nullptr;
  for (size_t i = 0; allocated && i < pools.planned_count; ++i) {
    const size_t bytes =
        static_cast<size_t>(meta->memory_planned_buffer_size(i).get());
    pools.planned[i] = static_cast<uint8_t*>(psram(bytes == 0 ? 16 : bytes));
    allocated = pools.planned[i] != nullptr;
    spans[i] = Span<uint8_t>(pools.planned[i], bytes);
    pools.planned_bytes += bytes;
  }
  if (!allocated) {
    std::printf(
        "ET CASE %u/%u name=%s status=FAIL reason=memory\n",
        static_cast<unsigned>(index),
        static_cast<unsigned>(count),
        name);
    return false;
  }
  MemoryAllocator method_allocator(
      static_cast<uint32_t>(method_bytes), pools.method_pool);
  MemoryAllocator temp_allocator(
      static_cast<uint32_t>(temp_bytes), pools.temp_pool);
  HierarchicalAllocator planned(
      Span<Span<uint8_t>>(spans, pools.planned_count));
  MemoryManager memory(&method_allocator, &planned, &temp_allocator);
  bool pass = false;
  {
    Result<Method> method = program->load_method("forward", &memory);
    if (!method.ok()) {
      std::printf(
          "ET CASE %u/%u name=%s status=FAIL reason=method-load error=0x%x\n",
          static_cast<unsigned>(index),
          static_cast<unsigned>(count),
          name,
          static_cast<unsigned>(method.error()));
      return false;
    }
    const int64_t load_us = esp_timer_get_time() - load_started;
    if (method->inputs_size() != v.inputs ||
        method->outputs_size() != v.outputs) {
      std::printf(
          "ET CASE %u/%u name=%s status=FAIL reason=io-count\n",
          static_cast<unsigned>(index),
          static_cast<unsigned>(count),
          name);
      return false;
    }
    EValue inputs[MAX_IO];
    OutputScore scores[MAX_IO]{};
    int64_t total_us = 0;
    int64_t fastest = INT64_MAX;
    bool ok = true;
    for (uint32_t s = 0; ok && s < v.samples; ++s) {
      const uint8_t* sample = v.data + v.stride * s;
      ok = method->get_inputs(inputs, v.inputs) == Error::Ok;
      for (uint32_t i = 0; ok && i < v.inputs; ++i) {
        Tensor tensor = inputs[i].toTensor();
        const size_t bytes = element_size(v.dtype[i]) * v.numel[i];
        ok = static_cast<uint32_t>(tensor.scalar_type()) == v.dtype[i] &&
            tensor.nbytes() == bytes;
        if (ok)
          std::memcpy(tensor.mutable_data_ptr(), sample, bytes);
        sample += bytes;
      }
      if (!ok) {
        std::printf(
            "ET CASE %u/%u name=%s status=FAIL reason=input-shape\n",
            static_cast<unsigned>(index),
            static_cast<unsigned>(count),
            name);
        return false;
      }
      const int64_t started = esp_timer_get_time();
      const Error status = method->execute();
      const int64_t elapsed = esp_timer_get_time() - started;
      if (status != Error::Ok) {
        std::printf(
            "ET CASE %u/%u name=%s status=FAIL reason=execute error=0x%x\n",
            static_cast<unsigned>(index),
            static_cast<unsigned>(count),
            name,
            static_cast<unsigned>(status));
        return false;
      }
      total_us += elapsed;
      if (elapsed < fastest)
        fastest = elapsed;
      for (uint32_t o = 0; o < v.outputs; ++o) {
        const uint32_t slot = v.inputs + o;
        const Tensor tensor = method->get_output(o).toTensor();
        const size_t bytes = element_size(v.dtype[slot]) * v.numel[slot];
        if (static_cast<uint32_t>(tensor.scalar_type()) != v.dtype[slot] ||
            tensor.nbytes() != bytes) {
          std::printf(
              "ET CASE %u/%u name=%s status=FAIL reason=output-shape\n",
              static_cast<unsigned>(index),
              static_cast<unsigned>(count),
              name);
          return false;
        }
        if ((v.flags & XTE1_FLAG_CRC) != 0) {
          const uint32_t crc = esp_rom_crc32_le(
              0, static_cast<const uint8_t*>(tensor.const_data_ptr()), bytes);
          if (crc != read_le_u32(sample))
            ++scores[o].mismatched;
          sample += 4;
          continue;
        }
        score_output(tensor, v.dtype[slot], sample, v.numel[slot], scores[o]);
        sample += bytes;
      }
    }
    int64_t bench_min = INT64_MAX;
    int64_t bench_max = 0;
    int64_t bench_total = 0;
    uint32_t bench_iterations = 0;
    for (uint32_t i = 0; ok && i < CONFIG_ESPNN_RUNNER_BENCH_ITERATIONS; ++i) {
      const int64_t started = esp_timer_get_time();
      ok = method->execute() == Error::Ok;
      const int64_t elapsed = esp_timer_get_time() - started;
      bench_total += elapsed;
      if (elapsed < bench_min)
        bench_min = elapsed;
      if (elapsed > bench_max)
        bench_max = elapsed;
      ++bench_iterations;
    }
    if (bench_iterations == 0)
      bench_min = fastest;
    double worst_sqnr = INFINITY;
    float max_abs = 0.0f;
    uint32_t mismatched = 0;
    for (uint32_t o = 0; o < v.outputs; ++o) {
      mismatched += scores[o].mismatched;
      if (scores[o].max_abs > max_abs)
        max_abs = scores[o].max_abs;
      if (scores[o].noise > 0.0 && scores[o].signal > 0.0) {
        const double sqnr =
            10.0 * std::log10(scores[o].signal / scores[o].noise);
        if (sqnr < worst_sqnr)
          worst_sqnr = sqnr;
      } else if (scores[o].noise > 0.0) {
        worst_sqnr = -INFINITY;
      }
    }
    const DelegateStats stats = delegate_stats();
    const uint32_t runs = v.samples + bench_iterations;
    const int64_t mean_us = bench_iterations ? bench_total / bench_iterations
                                             : total_us / v.samples;
    const int64_t delegate_us = stats.calls ? stats.total_us / runs : 0;
    pass = (v.flags & XTE1_FLAG_EXACT) != 0 ? mismatched == 0
                                            : worst_sqnr >= v.min_sqnr_db;
    std::printf(
        "ET CASE %u/%u name=%s inputs=%u outputs=%u samples=%u delegates=%u "
        "delegate_calls=%u us=%lld min_us=%lld max_us=%lld iterations=%u delegate_us=%lld "
        "cpu_us=%lld sqnr_db=%.1f "
        "max_abs=%.6g mismatched=%u exact_required=%d min_sqnr=%.1f load_us=%lld "
        "planned=%u arena=%u status=%s\n",
        static_cast<unsigned>(index),
        static_cast<unsigned>(count),
        name,
        static_cast<unsigned>(v.inputs),
        static_cast<unsigned>(v.outputs),
        static_cast<unsigned>(v.samples),
        static_cast<unsigned>(stats.delegates),
        static_cast<unsigned>(stats.calls / runs),
        static_cast<long long>(mean_us),
        static_cast<long long>(bench_min < fastest ? bench_min : fastest),
        static_cast<long long>(bench_iterations ? bench_max : mean_us),
        static_cast<unsigned>(bench_iterations),
        static_cast<long long>(delegate_us),
        static_cast<long long>(mean_us - delegate_us),
        std::isinf(worst_sqnr) ? (worst_sqnr > 0 ? 999.0 : -999.0) : worst_sqnr,
        static_cast<double>(max_abs),
        static_cast<unsigned>(mismatched),
        (v.flags & XTE1_FLAG_EXACT) != 0 ? 1 : 0,
        static_cast<double>(v.min_sqnr_db),
        static_cast<long long>(load_us),
        static_cast<unsigned>(pools.planned_bytes),
        static_cast<unsigned>(stats.arena_bytes),
        pass ? "PASS" : "FAIL");
  }
  return pass;
}

} // namespace

bool run_pte_suite(const uint8_t* data, size_t size) {
  std::printf("\n========================================\n");
  std::printf(
      " EXECUTORCH RUNTIME SUITE (ESP-NN delegate + portable CPU kernels)\n");
  std::printf("========================================\n");
  executorch::runtime::runtime_init();
  if (!register_espnn_backend()) {
    std::printf("ET RESULT: FAIL reason=backend-registration\n");
    return false;
  }
  if (!is_pte_bundle(data, size) || read_le_u32(data + 4) != 1) {
    std::printf("ET RESULT: FAIL reason=bundle\n");
    return false;
  }
  const uint32_t count = read_le_u32(data + 8);
  if (count == 0 ||
      !in_bounds(
          BUNDLE_HEADER_SIZE,
          static_cast<size_t>(count) * BUNDLE_RECORD_SIZE,
          size)) {
    std::printf("ET RESULT: FAIL reason=bundle-table\n");
    return false;
  }
  uint32_t passed = 0;
  const int64_t started = esp_timer_get_time();
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* record =
        data + BUNDLE_HEADER_SIZE + static_cast<size_t>(i) * BUNDLE_RECORD_SIZE;
    char name[BUNDLE_NAME_BYTES + 1]{};
    std::memcpy(name, record, BUNDLE_NAME_BYTES);
    const uint32_t pte_offset = read_le_u32(record + BUNDLE_NAME_BYTES);
    const uint32_t pte_size = read_le_u32(record + BUNDLE_NAME_BYTES + 4);
    const uint32_t vectors_offset = read_le_u32(record + BUNDLE_NAME_BYTES + 8);
    const uint32_t vectors_size = read_le_u32(record + BUNDLE_NAME_BYTES + 12);
    if (!in_bounds(pte_offset, pte_size, size) ||
        !in_bounds(vectors_offset, vectors_size, size)) {
      std::printf(
          "ET CASE %u/%u name=%s status=FAIL reason=bounds\n",
          static_cast<unsigned>(i + 1),
          static_cast<unsigned>(count),
          name);
      continue;
    }
    if (run_case(
            name,
            data + pte_offset,
            pte_size,
            data + vectors_offset,
            vectors_size,
            i + 1,
            count)) {
      ++passed;
    }
    std::printf(
        "ET MEMORY after=%s internal_free=%u psram_free=%u\n",
        name,
        static_cast<unsigned>(
            heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
  }
  std::printf(
      "ET RESULT: %s cases=%u passed=%u time_ms=%lld\n",
      passed == count ? "PASS" : "FAIL",
      static_cast<unsigned>(count),
      static_cast<unsigned>(passed),
      static_cast<long long>((esp_timer_get_time() - started) / 1000));
  return passed == count;
}

} // namespace espressif
} // namespace examples
} // namespace executorch
