/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

#if defined(ET_EVENT_TRACER_ENABLED)
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif

#include "arm_memory_allocator.h"
#include "model_qparams.h"
#include "platform.h"
#include "platform/fvp_lcd.h"
#include "platform/vsi_camera.h"

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

#define ET_METHOD_POOL_SECTION "input_data_sec"

constexpr uint32_t kInputWidth = 128;
constexpr uint32_t kInputHeight = 128;
constexpr uint32_t kGridSize = 7;
constexpr uint32_t kOutputSize = kGridSize * kGridSize * 5;
constexpr float kMinimumConfidence = 0.4F;
constexpr float kLargeBoxStartArea = 0.5F;
constexpr float kFullScreenBoxArea = 0.9F;
constexpr float kFullScreenConfidence = 0.7F;
constexpr float kNmsIouThreshold = 0.5F;

alignas(16) __attribute__((section(ET_METHOD_POOL_SECTION))) uint8_t
    method_pool[ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE];
alignas(32) __attribute__((section(ET_METHOD_POOL_SECTION))) uint8_t
    camera_frame[kCameraFrameSize];
alignas(16) __attribute__((section(".bss.tensor_arena")))
uint8_t temp_pool[ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];
#if defined(ET_EVENT_TRACER_ENABLED)
constexpr size_t kETDumpBufferSize = 192 * 1024;
alignas(64) __attribute__((section(".bss.etdump"))) uint8_t
    etdump_buffer[kETDumpBufferSize];
#endif

struct Detection {
  float x1;
  float y1;
  float x2;
  float y2;
  float confidence;
};

void write_exact(const uint8_t* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    platform_putc(data[i]);
  }
}

void write_text(const char* text) {
  while (*text != '\0') {
    platform_putc(static_cast<uint8_t>(*text++));
  }
}

void write_decimal_u64(uint64_t value) {
  uint8_t digits[20];
  size_t size = 0;
  do {
    digits[size++] = static_cast<uint8_t>('0' + value % 10);
    value /= 10;
  } while (value != 0);
  while (size != 0) {
    platform_putc(digits[--size]);
  }
}

void send_inference_time(uint64_t microseconds) {
  write_text("Inference time: ");
  write_decimal_u64(microseconds);
  write_text(" us\n");
}

#if defined(ET_EVENT_TRACER_ENABLED)
uint32_t checksum(const uint8_t* data, size_t size) {
  uint32_t value = 2166136261U;
  for (size_t i = 0; i < size; ++i) {
    value = (value ^ data[i]) * 16777619U;
  }
  return value;
}

void write_hex_u32(uint32_t value) {
  constexpr char digits[] = "0123456789ABCDEF";
  for (int shift = 28; shift >= 0; shift -= 4) {
    platform_putc(digits[(value >> shift) & 0xF]);
  }
}

void write_hex_bytes(const uint8_t* data, size_t size) {
  constexpr char digits[] = "0123456789ABCDEF";
  for (size_t i = 0; i < size; ++i) {
    platform_putc(digits[data[i] >> 4]);
    platform_putc(digits[data[i] & 0xF]);
  }
}

void send_etdump(const executorch::etdump::ETDumpResult& result) {
  const auto* data = static_cast<const uint8_t*>(result.buf);
  write_exact(reinterpret_cast<const uint8_t*>("ETDP"), 4);
  write_hex_u32(static_cast<uint32_t>(result.size));
  write_hex_u32(checksum(data, result.size));
  platform_putc('\n');
  write_hex_bytes(data, result.size);
  platform_putc('\n');
}
#endif

int8_t quantize_input(float value) {
  const float quantized = std::nearbyint(value / model_qparams::kInputScale) +
      model_qparams::kInputZeroPoint;
  return static_cast<int8_t>(std::clamp(
      quantized,
      static_cast<float>(model_qparams::kInputMinimum),
      static_cast<float>(model_qparams::kInputMaximum)));
}

void preprocess(const uint8_t* source, int8_t* destination) {
  constexpr float mean[3] = {0.485F, 0.456F, 0.406F};
  constexpr float standard_deviation[3] = {0.229F, 0.224F, 0.225F};
  for (uint32_t y = 0; y < kInputHeight; ++y) {
    const float source_y =
        (static_cast<float>(y) + 0.5F) * kCameraHeight / kInputHeight - 0.5F;
    const int32_t y0 = std::max(
        static_cast<int32_t>(0), static_cast<int32_t>(std::floor(source_y)));
    const int32_t y1 =
        std::min(y0 + 1, static_cast<int32_t>(kCameraHeight - 1));
    const float wy = std::max(0.0F, source_y - y0);
    for (uint32_t x = 0; x < kInputWidth; ++x) {
      const float source_x =
          (static_cast<float>(x) + 0.5F) * kCameraWidth / kInputWidth - 0.5F;
      const int32_t x0 = std::max(
          static_cast<int32_t>(0), static_cast<int32_t>(std::floor(source_x)));
      const int32_t x1 =
          std::min(x0 + 1, static_cast<int32_t>(kCameraWidth - 1));
      const float wx = std::max(0.0F, source_x - x0);
      for (uint32_t channel = 0; channel < 3; ++channel) {
        const auto sample = [&](int32_t sx, int32_t sy) {
          return static_cast<float>(
              source[(sy * kCameraWidth + sx) * kCameraChannels + channel]);
        };
        const float top =
            sample(x0, y0) + wx * (sample(x1, y0) - sample(x0, y0));
        const float bottom =
            sample(x0, y1) + wx * (sample(x1, y1) - sample(x0, y1));
        const float pixel = (top + wy * (bottom - top)) / 255.0F;
        const float normalized =
            (pixel - mean[channel]) / standard_deviation[channel];
        destination
            [channel * kInputWidth * kInputHeight + y * kInputWidth + x] =
                quantize_input(normalized);
      }
    }
  }
}

float sigmoid(float value) {
  return 1.0F / (1.0F + std::exp(-value));
}

float intersection_over_union(const Detection& left, const Detection& right) {
  const float width =
      std::max(0.0F, std::min(left.x2, right.x2) - std::max(left.x1, right.x1));
  const float height =
      std::max(0.0F, std::min(left.y2, right.y2) - std::max(left.y1, right.y1));
  const float intersection = width * height;
  const float left_area = (left.x2 - left.x1) * (left.y2 - left.y1);
  const float right_area = (right.x2 - right.x1) * (right.y2 - right.y1);
  return intersection / (left_area + right_area - intersection);
}

size_t decode_and_filter(const int8_t* output, Detection* detections) {
  Detection candidates[kGridSize * kGridSize];
  size_t candidate_count = 0;
  for (uint32_t row = 0; row < kGridSize; ++row) {
    for (uint32_t column = 0; column < kGridSize; ++column) {
      const int8_t* box = output + (row * kGridSize + column) * 5;
      const auto dequantize = [](int8_t value) {
        return (static_cast<int32_t>(value) - model_qparams::kOutputZeroPoint) *
            model_qparams::kOutputScale;
      };
      const float confidence = sigmoid(dequantize(box[0]));
      const float center_x = (sigmoid(dequantize(box[1])) + column) / kGridSize;
      const float center_y = (sigmoid(dequantize(box[2])) + row) / kGridSize;
      const float width = sigmoid(dequantize(box[3]));
      const float height = sigmoid(dequantize(box[4]));
      Detection detection{
          std::clamp(center_x - width / 2, 0.0F, 1.0F),
          std::clamp(center_y - height / 2, 0.0F, 1.0F),
          std::clamp(center_x + width / 2, 0.0F, 1.0F),
          std::clamp(center_y + height / 2, 0.0F, 1.0F),
          confidence};
      const float area =
          (detection.x2 - detection.x1) * (detection.y2 - detection.y1);
      const float area_scale = std::clamp(
          (area - kLargeBoxStartArea) /
              (kFullScreenBoxArea - kLargeBoxStartArea),
          0.0F,
          1.0F);
      const float threshold = kMinimumConfidence +
          area_scale * (kFullScreenConfidence - kMinimumConfidence);
      if (confidence >= threshold) {
        candidates[candidate_count++] = detection;
      }
    }
  }

  std::sort(
      candidates,
      candidates + candidate_count,
      [](const Detection& left, const Detection& right) {
        return left.confidence > right.confidence;
      });
  size_t detection_count = 0;
  for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
    bool keep = true;
    for (size_t selected = 0; selected < detection_count; ++selected) {
      if (intersection_over_union(candidates[candidate], detections[selected]) >
          kNmsIouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) {
      detections[detection_count++] = candidates[candidate];
    }
  }
  return detection_count;
}

void render_detections(const int8_t* output) {
  Detection detections[kGridSize * kGridSize];
  const size_t count = decode_and_filter(output, detections);
  lcd_draw_rgb888(camera_frame, kCameraWidth, kCameraHeight);
  for (size_t i = 0; i < count; ++i) {
    const Detection& detection = detections[i];
    lcd_draw_detection(
        static_cast<uint32_t>(detection.x1 * (kCameraWidth - 1)),
        static_cast<uint32_t>(detection.y1 * (kCameraHeight - 1)),
        static_cast<uint32_t>(detection.x2 * (kCameraWidth - 1)),
        static_cast<uint32_t>(detection.y2 * (kCameraHeight - 1)),
        static_cast<uint32_t>(std::lround(detection.confidence * 100.0F)));
  }
  write_text("Detections: ");
  write_decimal_u64(count);
  write_text("\n");
}

} // namespace

// Called by the ExecuTorch runtime through the platform abstraction interface.
// cppcheck-suppress unusedFunction
void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t,
    ET_UNUSED et_pal_log_level_t,
    ET_UNUSED const char*,
    ET_UNUSED const char*,
    ET_UNUSED size_t,
    ET_UNUSED const char*,
    ET_UNUSED size_t) {}

// Called by the ExecuTorch runtime through the platform abstraction interface.
// cppcheck-suppress unusedFunction
void et_pal_init() {}

// Called by the ExecuTorch runtime through the platform abstraction interface.
// cppcheck-suppress unusedFunction
et_timestamp_t et_pal_current_ticks() {
  return platform_cycle_count();
}

// Called by the ExecuTorch runtime through the platform abstraction interface.
// cppcheck-suppress unusedFunction
et_tick_ratio_t et_pal_ticks_to_ns_multiplier() {
  return {1, 1};
}

extern "C" void person_detection_main() {
  platform_init();
  executorch::runtime::runtime_init();
  write_text("Initializing VSI camera and MPS3 LCD\n");
  ET_CHECK_MSG(
      camera_init(camera_frame, sizeof(camera_frame)), "Camera init failed");
  ET_CHECK_MSG(lcd_init(), "LCD init failed");

  static const PlatformModel model = platform_model();
  static BufferDataLoader loader(model.data, model.size);
  static Result<Program> program = Program::load(&loader);
  ET_CHECK_MSG(
      program.ok(),
      "Program load failed: 0x%x",
      static_cast<unsigned int>(program.error()));
  static const auto method_name = program->get_method_name(0);
  ET_CHECK_MSG(method_name.ok(), "PTE has no method");
  static const auto method_meta = program->method_meta(*method_name);
  ET_CHECK_MSG(method_meta.ok(), "Method metadata load failed");

  static ArmMemoryAllocator method_allocator(sizeof(method_pool), method_pool);
  static ArmMemoryAllocator temp_allocator(sizeof(temp_pool), temp_pool);
  static Span<Span<uint8_t>> planned_spans = [&]() {
    const size_t count = method_meta->num_memory_planned_buffers();
    auto* spans = method_allocator.allocateList<Span<uint8_t>>(count);
    ET_CHECK_MSG(spans != nullptr || count == 0, "Span allocation failed");
    for (size_t i = 0; i < count; ++i) {
      const size_t size = method_meta->memory_planned_buffer_size(i).get();
      auto* buffer = static_cast<uint8_t*>(method_allocator.allocate(size, 16));
      ET_CHECK_MSG(buffer != nullptr, "Planned-memory allocation failed");
      new (&spans[i]) Span<uint8_t>(buffer, size);
    }
    return Span<Span<uint8_t>>(spans, count);
  }();
  static HierarchicalAllocator planned_memory(planned_spans);
  static MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

#if defined(ET_EVENT_TRACER_ENABLED)
  static executorch::etdump::ETDumpGen etdump(
      {etdump_buffer, sizeof(etdump_buffer)});
  executorch::runtime::EventTracer* event_tracer = &etdump;
#else
  executorch::runtime::EventTracer* event_tracer = nullptr;
#endif

  static Result<Method> method =
      program->load_method(*method_name, &memory_manager, event_tracer);
  ET_CHECK_MSG(
      method.ok(),
      "Method load failed: 0x%x",
      static_cast<unsigned int>(method.error()));
  static const auto input_meta = method->method_meta().input_tensor_meta(0);
  ET_CHECK_MSG(input_meta.ok(), "Input metadata load failed");
  ET_CHECK_MSG(
      method->method_meta().num_inputs() == 1 &&
          input_meta->scalar_type() == ScalarType::Char &&
          input_meta->is_memory_planned() &&
          input_meta->nbytes() == 3 * kInputWidth * kInputHeight,
      "Expected one memory-planned 1x3x128x128 int8 input");
  ET_CHECK_MSG(method->outputs_size() == 1, "Expected one model output");

  static EValue bound_input;
  ET_CHECK_MSG(
      method->get_inputs(&bound_input, 1) == Error::Ok &&
          bound_input.isTensor(),
      "Could not access planned input");
  static Tensor input = bound_input.toTensor();
  static EValue output_evalue;
  write_text("Ready for VSI frames\n");

  for (;;) {
    const CameraCaptureResult capture = camera_capture();
    if (capture == CameraCaptureResult::EndOfStream) {
      write_text("Video input ended\n");
      for (;;) {
      }
    }
    if (capture == CameraCaptureResult::Error) {
      write_text("VSI camera error\n");
      for (;;) {
      }
    }

    preprocess(camera_frame, input.mutable_data_ptr<int8_t>());
#if defined(ET_EVENT_TRACER_ENABLED)
    etdump.reset();
#endif
    const uint32_t start_cycles = platform_cycle_count();
    const Error execute_error = method->execute();
    const uint32_t elapsed_cycles = platform_cycle_count() - start_cycles;
    send_inference_time(platform_cycles_to_microseconds(elapsed_cycles));
#if defined(ET_EVENT_TRACER_ENABLED)
    send_etdump(etdump.get_etdump_data());
#endif

    ET_CHECK_MSG(execute_error == Error::Ok, "Inference failed");
    ET_CHECK_MSG(
        method->get_outputs(&output_evalue, 1) == Error::Ok &&
            output_evalue.isTensor(),
        "Could not read model output");
    const Tensor output = output_evalue.toTensor();
    ET_CHECK_MSG(
        output.scalar_type() == ScalarType::Char &&
            output.nbytes() == kOutputSize,
        "Expected 245-byte int8 output");
    render_detections(output.const_data_ptr<int8_t>());
    platform_flush();
    temp_allocator.reset();
  }
}
