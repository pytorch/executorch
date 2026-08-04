/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_memory_allocator.h"
#include "image.h"
#include "model_pte.h"

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

const size_t method_allocation_pool_size =
    ET_SEGMENTATION_METHOD_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    section("input_data_sec"),
    aligned(16))) method_allocation_pool[method_allocation_pool_size];

const size_t temp_allocation_pool_size =
    ET_SEGMENTATION_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    section(".bss.tensor_arena"),
    aligned(16))) temp_allocation_pool[temp_allocation_pool_size];

#if defined(ET_SEGMENTATION_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
extern "C" {
size_t ethosu_fast_scratch_size =
    ET_SEGMENTATION_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((section(".bss.ethosu_scratch"), aligned(16)))
dedicated_sram[ET_SEGMENTATION_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];
unsigned char* ethosu_fast_scratch = dedicated_sram;
}
#endif

namespace {

#if defined(ET_SEGMENTATION_MASK_THRESHOLD)
constexpr float kMaskThreshold = ET_SEGMENTATION_MASK_THRESHOLD;
#else
constexpr float kMaskThreshold = 0.0f;
#endif

#if defined(ET_SEGMENTATION_SEMIHOSTING_OUTPUT) && \
    (defined(__arm__) || defined(__thumb__))
constexpr uint32_t kSemihostingSysWrite0 = 0x04;
constexpr uint32_t kSemihostingSysExitExtended = 0x20;
constexpr uint32_t kAdpStoppedApplicationExit = 0x20026;

uint32_t semihosting_call(uint32_t operation_code, const void* argument) {
  uint32_t result;
  asm volatile(
      "mov r0, %[operation]\n"
      "mov r1, %[argument]\n"
      "bkpt 0xab\n"
      "mov %[result], r0\n"
      : [result] "=r"(result)
      : [operation] "r"(operation_code), [argument] "r"(argument)
      : "r0", "r1", "memory");
  return result;
}

void semihosting_write0(const char* message) {
  semihosting_call(kSemihostingSysWrite0, message);
}
#endif

void write_runtime_line(const char* message) {
#if defined(ET_SEGMENTATION_SEMIHOSTING_OUTPUT) && \
    (defined(__arm__) || defined(__thumb__))
  semihosting_write0(message);
  semihosting_write0("\n");
#else
  (void)message;
#endif
}

void write_runtime_format(const char* format, ...) {
  char line[768];
  va_list args;
  va_start(args, format);
  vsnprintf(line, sizeof(line), format, args);
  va_end(args);
  write_runtime_line(line);
}

void request_runtime_exit(int code) {
#if defined(ET_SEGMENTATION_SEMIHOSTING_OUTPUT) && \
    (defined(__arm__) || defined(__thumb__))
  const uint32_t exit_block[2] = {
      kAdpStoppedApplicationExit,
      static_cast<uint32_t>(code),
  };
  semihosting_call(kSemihostingSysExitExtended, exit_block);
#else
  (void)code;
#endif
}

uint32_t update_hash(uint32_t hash, uint8_t value) {
  hash ^= value;
  hash *= 16777619u;
  return hash;
}

#if defined(ET_SEGMENTATION_DUMP_MASK)
void dump_mask_rle(const std::vector<uint8_t>& mask) {
  ET_LOG(Info, "Segmentation mask RLE begin");
  write_runtime_line("Segmentation mask RLE begin");
  char line[512];
  size_t line_len = 0;
  line[0] = '\0';

  size_t index = 0;
  while (index < mask.size()) {
    const uint8_t class_id = mask[index];
    size_t run_len = 1;
    while (index + run_len < mask.size() && mask[index + run_len] == class_id) {
      ++run_len;
    }

    char entry[32];
    const int entry_len = snprintf(
        entry,
        sizeof(entry),
        "%u:%zu,",
        static_cast<unsigned>(class_id),
        run_len);
    if (entry_len <= 0) {
      break;
    }
    if (line_len + static_cast<size_t>(entry_len) >= sizeof(line)) {
      ET_LOG(Info, "Segmentation mask RLE chunk %s", line);
      write_runtime_format("Segmentation mask RLE chunk %s", line);
      line_len = 0;
      line[0] = '\0';
    }
    line_len += static_cast<size_t>(
        snprintf(line + line_len, sizeof(line) - line_len, "%s", entry));
    index += run_len;
  }
  if (line_len > 0) {
    ET_LOG(Info, "Segmentation mask RLE chunk %s", line);
    write_runtime_format("Segmentation mask RLE chunk %s", line);
  }
  ET_LOG(Info, "Segmentation mask RLE end");
  write_runtime_line("Segmentation mask RLE end");
}
#endif

void summarize_segmentation_output(const Tensor& out) {
  ET_CHECK_MSG(
      out.dim() == 4,
      "Expected mask logits with shape [1, 1, height, width], got rank %zd",
      out.dim());
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Float,
      "Expected float mask logits, got dtype %d",
      out.scalar_type());
  ET_CHECK_MSG(out.size(0) == 1, "Only batch size 1 is supported.");
  ET_CHECK_MSG(
      out.size(1) == 1,
      "MobileSAM fixed-prompt export expects one mask channel, got %zd",
      out.size(1));

  const size_t height = static_cast<size_t>(out.size(2));
  const size_t width = static_cast<size_t>(out.size(3));
  const auto strides = out.strides();
  const float* data = out.const_data_ptr<float>();

  size_t foreground_pixels = 0;
#if defined(ET_SEGMENTATION_DUMP_MASK)
  std::vector<uint8_t> mask(height * width, 0);
#endif
  uint32_t mask_hash = 2166136261u;
  float min_score = data[0];
  float max_score = data[0];
  double score_sum = 0.0;
  const float threshold_sweep[] = {
      0.0f,
      -2.0f,
      -4.0f,
      -5.0f,
      -6.0f,
      -7.0f,
      -8.0f,
      -10.0f,
      -12.0f,
      -14.0f,
  };
  size_t threshold_sweep_counts
      [sizeof(threshold_sweep) / sizeof(threshold_sweep[0])] = {};

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      const float score = data[y * strides[2] + x * strides[3]];
      min_score = std::min(min_score, score);
      max_score = std::max(max_score, score);
      score_sum += score;
      for (size_t i = 0;
           i < sizeof(threshold_sweep) / sizeof(threshold_sweep[0]);
           ++i) {
        threshold_sweep_counts[i] += score > threshold_sweep[i] ? 1 : 0;
      }
      const uint8_t mask_value = score > kMaskThreshold ? 1 : 0;
      foreground_pixels += mask_value;
#if defined(ET_SEGMENTATION_DUMP_MASK)
      mask[y * width + x] = mask_value;
#endif
      mask_hash = update_hash(mask_hash, mask_value);
    }
  }

  ET_LOG(Info, "Output mask logits shape = [1, 1, %zu, %zu]", height, width);
  write_runtime_format(
      "Output mask logits shape = [1, 1, %zu, %zu]", height, width);
  ET_LOG(
      Info,
      "Segmentation input image = %zu x %zu x %zu",
      image_width,
      image_height,
      image_channels);
  write_runtime_format(
      "Segmentation input image = %zu x %zu x %zu",
      image_width,
      image_height,
      image_channels);
  ET_LOG(Info, "Mask threshold = %.4f", static_cast<double>(kMaskThreshold));
  write_runtime_format(
      "Mask threshold = %.4f", static_cast<double>(kMaskThreshold));
  ET_LOG(
      Info,
      "Mask logits min/max/mean = %.6f / %.6f / %.6f",
      static_cast<double>(min_score),
      static_cast<double>(max_score),
      score_sum / static_cast<double>(height * width));
  write_runtime_format(
      "Mask logits min/max/mean = %.6f / %.6f / %.6f",
      static_cast<double>(min_score),
      static_cast<double>(max_score),
      score_sum / static_cast<double>(height * width));
  for (size_t i = 0; i < sizeof(threshold_sweep) / sizeof(threshold_sweep[0]);
       ++i) {
    write_runtime_format(
        "Threshold %.1f foreground pixels = %zu",
        static_cast<double>(threshold_sweep[i]),
        threshold_sweep_counts[i]);
  }
  ET_LOG(Info, "Segmentation mask hash = 0x%08" PRIx32, mask_hash);
  write_runtime_format("Segmentation mask hash = 0x%08" PRIx32, mask_hash);
  ET_LOG(Info, "Mask foreground pixels = %zu", foreground_pixels);
  write_runtime_format("Mask foreground pixels = %zu", foreground_pixels);
  ET_LOG(
      Info, "Mask background pixels = %zu", height * width - foreground_pixels);
  write_runtime_format(
      "Mask background pixels = %zu", height * width - foreground_pixels);

#if defined(ET_SEGMENTATION_DUMP_MASK)
  dump_mask_rle(mask);
#endif
}

} // namespace

int main() {
  executorch::runtime::runtime_init();
  ET_LOG(Info, "Runtime initialized");
  write_runtime_line("MobileSAM Ethos-U example started");
  BufferDataLoader loader(model_pte, sizeof(model_pte));
  ET_LOG(Info, "Size of the model = %zu", sizeof(model_pte));
  write_runtime_format("Model size = %zu bytes", sizeof(model_pte));
  write_runtime_line("Loading ExecuTorch program");
  Result<Program> program = Program::load(&loader);
  ET_CHECK_MSG(program.ok(), "Program::load failed: 0x%x", program.error());
  write_runtime_line("Program loaded");

  const auto method_name_result = program->get_method_name(0);
  ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
  const char* method_name = *method_name_result;
  ET_LOG(Info, "Running method %s", method_name);
  write_runtime_format("Running method %s", method_name);

  Result<MethodMeta> method_meta_result = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta_result.ok(),
      "method_meta lookup failed: 0x%x",
      method_meta_result.error());

  ArmMemoryAllocator method_allocator(
      method_allocation_pool_size, method_allocation_pool);
  ArmMemoryAllocator temp_allocator(
      temp_allocation_pool_size, temp_allocation_pool);

  std::vector<uint8_t*> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  const size_t num_memory_planned_buffers =
      method_meta_result->num_memory_planned_buffers();
  ET_LOG(Info, "num_memory_planned_buffers = %zu", num_memory_planned_buffers);
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    const size_t buffer_size =
        method_meta_result->memory_planned_buffer_size(id).get();
    ET_LOG(Info, "Planned memory buffer_size %zu %zu bytes", id, buffer_size);

    uint8_t* buffer = reinterpret_cast<uint8_t*>(
        method_allocator.allocate(buffer_size, 16UL));
    ET_CHECK_MSG(
        buffer != nullptr,
        "Could not allocate memory for memory planned buffer size %zu",
        buffer_size);
    planned_buffers.push_back(buffer);
    planned_spans.push_back({planned_buffers.back(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);
  write_runtime_line("Loading method");
  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(method.ok(), "load_method failed: 0x%x", method.error());
  write_runtime_line("Method loaded");

  const size_t num_inputs = method->inputs_size();
  ET_LOG(Info, "Number of input tensors = %zu", num_inputs);
  ET_CHECK_MSG(
      num_inputs == 1,
      "The segmentation model has a single input tensor, but the provided model has %zu input tensors",
      num_inputs);

  EValue* input_evalues = method_allocator.allocateList<EValue>(num_inputs);
  Error err = method->get_inputs(input_evalues, num_inputs);
  ET_CHECK_MSG(err == Error::Ok, "get_inputs failed");
  Tensor& input_tensor = input_evalues[0].toTensor();
  const size_t expected_elems = input_tensor.numel();
  const size_t image_elements = sizeof(image_data) / sizeof(image_data[0]);
  ET_CHECK_MSG(
      expected_elems == image_elements,
      "Input tensor expects %zu elements, but image_data has %zu elements",
      expected_elems,
      image_elements);
  ET_CHECK_MSG(
      input_tensor.scalar_type() == ScalarType::Float,
      "Expected float input tensor, got dtype %d",
      input_tensor.scalar_type());

  float* input_data = input_tensor.mutable_data_ptr<float>();
  write_runtime_format("Copying %zu input elements", expected_elems);
  for (size_t i = 0; i < expected_elems; ++i) {
    input_data[i] = image_data[i];
  }

  write_runtime_line("Running model execution");
  Error status_inference = method->execute();
  ET_CHECK_MSG(
      status_inference == Error::Ok,
      "Inference failed 0x%" PRIx32,
      status_inference);
  write_runtime_line("Inference finished");

  const size_t num_outputs = method->outputs_size();
  std::vector<EValue> outputs(num_outputs);
  Error status_outputs = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK_MSG(
      status_outputs == Error::Ok,
      "get_outputs failed 0x%" PRIx32,
      status_outputs);

  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].isTensor()) {
      summarize_segmentation_output(outputs[i].toTensor());
      ET_LOG(Info, "Model executed successfully.");
      write_runtime_line("Model executed successfully.");
      request_runtime_exit(0);
      return 0;
    }
  }

  ET_CHECK_MSG(false, "No tensor output found.");
  request_runtime_exit(1);
  return 1;
}
