/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <vector>

#include "arm_memory_allocator.h"

#include "audio.h"
#include "model_pte.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

constexpr size_t kSampleRate = 16000;
constexpr size_t kWindowSize = 512;
constexpr size_t kContextSize = 64;
constexpr size_t kInputSize = kWindowSize + kContextSize;
constexpr size_t kHiddenDim = 128;
constexpr size_t kStateSize = 2 * kHiddenDim;
constexpr float kFrameDuration =
    static_cast<float>(kWindowSize) / static_cast<float>(kSampleRate);

#if defined(ENABLE_SEMIHOSTING_OUTPUT)
int semihosting_call(int operation, void* arguments) {
  // cppcheck-suppress syntaxError
  register int r0 __asm("r0") = operation;
  register void* r1 __asm("r1") = arguments;
  __asm volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
  return r0;
}

void write_semihosting_file(
    const char* output_path,
    const void* data,
    size_t size_bytes) {
  uintptr_t open_args[] = {
      reinterpret_cast<uintptr_t>(output_path),
      5,
      std::strlen(output_path),
  };
  int handle = semihosting_call(0x01, open_args);
  ET_CHECK_MSG(handle >= 0, "Failed to open %s", output_path);

  uintptr_t write_args[] = {
      static_cast<uintptr_t>(handle),
      reinterpret_cast<uintptr_t>(data),
      size_bytes,
  };
  int unwritten_bytes = semihosting_call(0x05, write_args);

  uintptr_t close_args[] = {static_cast<uintptr_t>(handle)};
  int close_status = semihosting_call(0x02, close_args);

  ET_CHECK_MSG(unwritten_bytes == 0, "Failed to write all probability values");
  ET_CHECK_MSG(close_status == 0, "Failed to close %s", output_path);
}

void write_probabilities(
    const char* output_path,
    const std::vector<float>& probabilities) {
  write_semihosting_file(
      output_path, probabilities.data(), probabilities.size() * sizeof(float));
}

#endif

const size_t method_allocation_pool_size = 2 * 1024 * 1024;
unsigned char __attribute__((
    section("input_data_sec"),
    aligned(16))) method_allocation_pool[method_allocation_pool_size];

const size_t temp_allocation_pool_size = 1 * 1024 * 1024;
unsigned char __attribute__((
    section(".bss.tensor_arena"),
    aligned(16))) temp_allocation_pool[temp_allocation_pool_size];

void copy_input_frame(
    Tensor& input_tensor,
    std::array<float, kContextSize>& context,
    size_t frame_index) {
  float* input_data = input_tensor.mutable_data_ptr<float>();
  std::memcpy(input_data, context.data(), kContextSize * sizeof(float));

  size_t offset = frame_index * kWindowSize;
  size_t remaining = audio_data_len > offset ? audio_data_len - offset : 0;
  size_t chunk_len = std::min(kWindowSize, remaining);

  if (chunk_len > 0) {
    std::memcpy(
        input_data + kContextSize,
        audio_data + offset,
        chunk_len * sizeof(float));
  }
  if (chunk_len < kWindowSize) {
    std::memset(
        input_data + kContextSize + chunk_len,
        0,
        (kWindowSize - chunk_len) * sizeof(float));
  }

  if (chunk_len >= kContextSize) {
    std::memcpy(
        context.data(),
        audio_data + offset + chunk_len - kContextSize,
        kContextSize * sizeof(float));
  } else if (chunk_len > 0) {
    size_t keep = kContextSize - chunk_len;
    std::memmove(
        context.data(), context.data() + chunk_len, keep * sizeof(float));
    std::memcpy(
        context.data() + keep, audio_data + offset, chunk_len * sizeof(float));
  }
}

int main() {
  executorch::runtime::runtime_init();
  ET_LOG(Info, "Runtime initialized");
  BufferDataLoader loader(model_pte, sizeof(model_pte));
  ET_LOG(Info, "Size of the model = %zu", sizeof(model_pte));
  Result<Program> program = Program::load(&loader);
  ET_CHECK_MSG(program.ok(), "Program::load failed: 0x%x", program.error());

  const auto method_name_result = program->get_method_name(0);
  ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
  const char* method_name = *method_name_result;
  ET_LOG(Info, "Running method %s", method_name);

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
  size_t num_memory_planned_buffers =
      method_meta_result->num_memory_planned_buffers();
  ET_LOG(Info, "num_memory_planned_buffers = %zu", num_memory_planned_buffers);
  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
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

  Result<Method> method = program->load_method(method_name, &memory_manager);
  ET_CHECK_MSG(method.ok(), "load_method failed: 0x%x", method.error());

  size_t num_inputs = method->inputs_size();
  ET_LOG(Info, "Number of input tensors = %zu", num_inputs);
  ET_CHECK_MSG(num_inputs == 2, "Silero VAD expects audio and state inputs");

  EValue* input_evalues = method_allocator.allocateList<EValue>(num_inputs);
  Error input_status = method->get_inputs(input_evalues, num_inputs);
  ET_CHECK_MSG(input_status == Error::Ok, "Get inputs failed");
  Tensor& audio_input = input_evalues[0].toTensor();
  Tensor& state_input = input_evalues[1].toTensor();

  ET_CHECK_MSG(
      audio_input.scalar_type() == ScalarType::Float,
      "Audio input must be float");
  ET_CHECK_MSG(
      state_input.scalar_type() == ScalarType::Float,
      "State input must be float");
  ET_CHECK_MSG(
      audio_input.numel() == kInputSize,
      "Audio input expects %zu elements, got %zu",
      kInputSize,
      audio_input.numel());
  ET_CHECK_MSG(
      state_input.numel() == kStateSize,
      "State input expects %zu elements, got %zu",
      kStateSize,
      state_input.numel());

  std::array<float, kContextSize> context{};
  std::array<float, kStateSize> state{};
  size_t num_frames = (audio_data_len + kWindowSize - 1) / kWindowSize;
#if defined(ENABLE_SEMIHOSTING_OUTPUT)
  std::vector<float> probabilities(num_frames, 0.0f);
#endif
  size_t num_outputs = method->outputs_size();
  ET_CHECK_MSG(num_outputs == 2, "Silero VAD expects probability and state");
  std::vector<EValue> outputs(num_outputs);

  bool speech_active = false;
  size_t speech_start_frame = 0;
  size_t speech_frames = 0;
  size_t num_segments = 0;

  ET_LOG(
      Info,
      "Running %zu audio samples (%zu frames), threshold=%f",
      audio_data_len,
      num_frames,
      static_cast<double>(VAD_THRESHOLD));

  for (size_t frame_index = 0; frame_index < num_frames; ++frame_index) {
    copy_input_frame(audio_input, context, frame_index);
    std::memcpy(
        state_input.mutable_data_ptr<float>(),
        state.data(),
        kStateSize * sizeof(float));

    Error inference_status = method->execute();
    ET_CHECK_MSG(
        inference_status == Error::Ok,
        "Inference failed 0x%" PRIx32,
        inference_status);

    Error outputs_status = method->get_outputs(outputs.data(), outputs.size());
    ET_CHECK_MSG(
        outputs_status == Error::Ok,
        "get_outputs failed 0x%" PRIx32,
        outputs_status);

    Tensor probability_output = outputs[0].toTensor();
    Tensor state_output = outputs[1].toTensor();
    float probability = probability_output.const_data_ptr<float>()[0];
#if defined(ENABLE_SEMIHOSTING_OUTPUT)
    probabilities[frame_index] = probability;
#endif

    std::memcpy(
        state.data(),
        state_output.const_data_ptr<float>(),
        kStateSize * sizeof(float));

    bool is_speech = probability > VAD_THRESHOLD;
    if (is_speech) {
      speech_frames++;
      if (!speech_active) {
        speech_active = true;
        speech_start_frame = frame_index;
      }
    } else if (speech_active) {
      speech_active = false;
      ET_LOG(
          Info,
          "SEGMENT %.3f %.3f speech",
          static_cast<double>(speech_start_frame * kFrameDuration),
          static_cast<double>(frame_index * kFrameDuration));
      num_segments++;
    }

    ET_LOG(
        Info,
        "PROB %.3f %f %s",
        static_cast<double>(frame_index * kFrameDuration),
        static_cast<double>(probability),
        is_speech ? "speech" : "silence");
  }

  if (speech_active) {
    ET_LOG(
        Info,
        "SEGMENT %.3f %.3f speech",
        static_cast<double>(speech_start_frame * kFrameDuration),
        static_cast<double>(num_frames * kFrameDuration));
    num_segments++;
  }

  float speech_percent = num_frames == 0 ? 0.0f
                                         : 100.0f *
          static_cast<float>(speech_frames) / static_cast<float>(num_frames);
  ET_LOG(
      Info,
      "%zu segments, %zu frames, %.1fs",
      num_segments,
      num_frames,
      static_cast<double>(num_frames * kFrameDuration));
  ET_LOG(
      Info,
      "Speech: %zu/%zu frames (%.1f%%)",
      speech_frames,
      num_frames,
      static_cast<double>(speech_percent));

#if defined(ENABLE_SEMIHOSTING_OUTPUT)
  const char* output_path = "vad_probs.bin";
  ET_LOG(Info, "Writing probability dump to %s", output_path);
  write_probabilities(output_path, probabilities);
#endif

  ET_LOG(Info, "\04");
  return 0;
}
