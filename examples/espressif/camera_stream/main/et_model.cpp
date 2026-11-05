/*
 * Copyright 2026 Mwaniki Njagi.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <et_model.h>

#include <cstdio>
#include <new>

#include <esp_heap_caps.h>
#include <esp_timer.h>

#include <espnn/espnn_backend.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

namespace executorch {
namespace examples {
namespace espressif {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::extension::BufferDataLoader;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::HierarchicalAllocator;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::MemoryManager;
using ::executorch::runtime::Method;
using ::executorch::runtime::MethodMeta;
using ::executorch::runtime::Program;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;
using ::executorch::runtime::etensor::TensorImpl;

namespace {

constexpr size_t METHOD_POOL_BYTES = 256 * 1024;
constexpr size_t TEMP_POOL_BYTES = 128 * 1024;

void* psram_alloc(size_t bytes) {
  void* pointer =
      heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (pointer == nullptr)
    pointer = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_8BIT);
  return pointer;
}

bool float_tensor(Result<::executorch::runtime::TensorInfo> meta, size_t rank) {
  return meta.ok() && meta->scalar_type() == ScalarType::Float &&
      meta->sizes().size() == rank;
}

} // namespace

bool EtModel::load(const uint8_t* data, size_t size) {
  ::executorch::runtime::runtime_init();
  auto* loader = new (std::nothrow) BufferDataLoader(data, size);
  if (loader == nullptr)
    return false;
  Result<Program> program = Program::load(loader);
  if (!program.ok()) {
    std::printf("#ERR Program::load failed: 0x%x\n", unsigned(program.error()));
    return false;
  }
  auto* owned_program = new (std::nothrow) Program(std::move(*program));
  if (owned_program == nullptr)
    return false;
  Result<MethodMeta> meta = owned_program->method_meta("forward");
  if (!meta.ok() || meta->num_inputs() != 1 || meta->num_outputs() != 1 ||
      !float_tensor(meta->input_tensor_meta(0), 4) ||
      !float_tensor(meta->output_tensor_meta(0), 2)) {
    std::printf(
        "#ERR model must map one float [1,3,H,W] image to float [1,classes]\n");
    return false;
  }
  auto input_meta = meta->input_tensor_meta(0);
  auto output_meta = meta->output_tensor_meta(0);
  if (input_meta->sizes()[0] != 1 || input_meta->sizes()[1] != 3 ||
      output_meta->sizes()[0] != 1) {
    std::printf("#ERR model input must be [1,3,H,W]\n");
    return false;
  }

  auto* method_pool = static_cast<uint8_t*>(psram_alloc(METHOD_POOL_BYTES));
  auto* temp_pool = static_cast<uint8_t*>(psram_alloc(TEMP_POOL_BYTES));
  const size_t planned_count = meta->num_memory_planned_buffers();
  auto* planned_spans = new (std::nothrow) Span<uint8_t>[planned_count + 1];
  if (method_pool == nullptr || temp_pool == nullptr ||
      planned_spans == nullptr) {
    std::printf("#ERR runtime pools could not be allocated\n");
    return false;
  }
  size_t planned_bytes = 0;
  for (size_t i = 0; i < planned_count; ++i) {
    Result<int64_t> bytes = meta->memory_planned_buffer_size(i);
    if (!bytes.ok() || *bytes < 0)
      return false;
    auto* buffer =
        static_cast<uint8_t*>(psram_alloc(static_cast<size_t>(*bytes)));
    if (buffer == nullptr) {
      std::printf(
          "#ERR planned buffer %u (%lld bytes) could not be allocated\n",
          unsigned(i),
          static_cast<long long>(*bytes));
      return false;
    }
    planned_spans[i] = {buffer, static_cast<size_t>(*bytes)};
    planned_bytes += static_cast<size_t>(*bytes);
  }

  auto* method_allocator =
      new (std::nothrow) MemoryAllocator(METHOD_POOL_BYTES, method_pool);
  auto* temp_allocator =
      new (std::nothrow) MemoryAllocator(TEMP_POOL_BYTES, temp_pool);
  auto* planned =
      new (std::nothrow) HierarchicalAllocator({planned_spans, planned_count});
  if (method_allocator == nullptr || temp_allocator == nullptr ||
      planned == nullptr)
    return false;
  auto* memory = new (std::nothrow)
      MemoryManager(method_allocator, planned, temp_allocator);
  if (memory == nullptr)
    return false;

  const auto before = ::executorch::backends::espressif::delegate_stats();
  Result<Method> method = owned_program->load_method("forward", memory);
  if (!method.ok()) {
    std::printf("#ERR load_method failed: 0x%x\n", unsigned(method.error()));
    return false;
  }
  const auto after = ::executorch::backends::espressif::delegate_stats();
  method_ = new (std::nothrow) Method(std::move(*method));
  if (method_ == nullptr)
    return false;

  info_.input_height = input_meta->sizes()[2];
  info_.input_width = input_meta->sizes()[3];
  info_.classes = static_cast<size_t>(output_meta->sizes()[1]);
  info_.planned_bytes = planned_bytes;
  info_.delegates = after.delegates - before.delegates;
  info_.delegate_arena_bytes = after.arena_bytes - before.arena_bytes;

  const size_t pixels =
      static_cast<size_t>(info_.input_width) * info_.input_height;
  input_ = static_cast<float*>(psram_alloc(sizeof(float) * pixels * 3));
  if (input_ == nullptr)
    return false;
  sizes_[0] = 1;
  sizes_[1] = 3;
  sizes_[2] = info_.input_height;
  sizes_[3] = info_.input_width;
  for (int i = 0; i < 4; ++i)
    dim_order_[i] = static_cast<TensorImpl::DimOrderType>(i);
  strides_[3] = 1;
  strides_[2] = info_.input_width;
  strides_[1] = static_cast<TensorImpl::StridesType>(pixels);
  strides_[0] = static_cast<TensorImpl::StridesType>(pixels * 3);
  input_impl_ = new (std::nothrow)
      TensorImpl(ScalarType::Float, 4, sizes_, input_, dim_order_, strides_);
  return input_impl_ != nullptr;
}

bool EtModel::set_input_rgb888(
    const uint8_t* rgb,
    const float mean[3],
    const float stddev[3]) {
  if (input_ == nullptr || rgb == nullptr)
    return false;
  float lut[3][256];
  for (int c = 0; c < 3; ++c) {
    for (int v = 0; v < 256; ++v)
      lut[c][v] = (static_cast<float>(v) / 255.0f - mean[c]) / stddev[c];
  }
  const size_t pixels =
      static_cast<size_t>(info_.input_width) * info_.input_height;
  float* red = input_;
  float* green = input_ + pixels;
  float* blue = input_ + 2 * pixels;
  for (size_t i = 0; i < pixels; ++i) {
    red[i] = lut[0][rgb[i * 3 + 0]];
    green[i] = lut[1][rgb[i * 3 + 1]];
    blue[i] = lut[2][rgb[i * 3 + 2]];
  }
  return true;
}

bool EtModel::run() {
  if (method_ == nullptr)
    return false;
  const int64_t started = esp_timer_get_time();
  const auto before = ::executorch::backends::espressif::delegate_stats();
  Error error = method_->set_input(EValue(Tensor(input_impl_)), 0);
  if (error == Error::Ok)
    error = method_->execute();
  const auto after = ::executorch::backends::espressif::delegate_stats();
  last_run_us_ = esp_timer_get_time() - started;
  last_delegate_us_ = after.total_us - before.total_us;
  if (error != Error::Ok) {
    std::printf("#ERR execute failed: 0x%x\n", unsigned(error));
    return false;
  }
  const EValue& output = method_->get_output(0);
  return output.isTensor() &&
      output.toTensor().numel() == static_cast<ssize_t>(info_.classes);
}

const float* EtModel::logits() const {
  return method_ != nullptr
      ? method_->get_output(0).toTensor().const_data_ptr<float>()
      : nullptr;
}

} // namespace espressif
} // namespace examples
} // namespace executorch
