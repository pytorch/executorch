/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Implementation of et_model.h over executorch::runtime::Program/Method.
//
// Program/Method directly rather than extension::Module: Module keeps its
// methods in std::string-keyed maps and shared_ptr state on the global
// heap, while this API places runtime buffers with heap_caps when they
// are allocated in et_model_load_method. Module is still vendored in this
// component for advanced users; see README.md.

#include "et_model.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <optional>
#include <utility>

#include <esp_heap_caps.h>
#include <esp_log.h>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

namespace {

constexpr char kTag[] = "et_model";
constexpr size_t kDefaultArenaSize = 256 * 1024;
constexpr size_t kArenaAlignment = 16;
constexpr size_t kPteAlignment = 16;
constexpr size_t kMaxPlannedBuffers = 16;

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
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
using executorch::runtime::TensorInfo;
using SpanU8 = executorch::runtime::Span<uint8_t>;

bool valid_location(et_mem_location_t loc) {
  return static_cast<unsigned>(loc) <= ET_MEM_PSRAM;
}

uint32_t caps_for(et_mem_location_t loc) {
  switch (loc) {
    case ET_MEM_INTERNAL:
      return MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    case ET_MEM_PSRAM:
      return MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    case ET_MEM_DEFAULT:
    default:
      return MALLOC_CAP_DEFAULT;
  }
}

// Zero-sized planned buffers still need a valid pointer for their span.
uint8_t* caps_alloc(size_t size, et_mem_location_t loc) {
  return static_cast<uint8_t*>(heap_caps_aligned_alloc(
      kArenaAlignment, size == 0 ? 1 : size, caps_for(loc)));
}

} // namespace

struct et_model {
  explicit et_model(const et_model_config_t& c)
      : cfg(c), loader(c.pte, c.pte_size) {}

  void fail(const char* what, Error err) {
    snprintf(
        error,
        sizeof(error),
        "%s failed (error 0x%x)",
        what,
        static_cast<unsigned>(err));
    ESP_LOGE(kTag, "%s", error);
  }
  void fail(const char* what) {
    snprintf(error, sizeof(error), "%s", what);
    ESP_LOGE(kTag, "%s", error);
  }

  void free_method() {
    meta.reset();
    method.reset();
    manager.reset();
    planned.reset();
    for (size_t i = 0; i < num_planned; i++) {
      heap_caps_free(planned_ptrs[i]);
      planned_ptrs[i] = nullptr;
    }
    num_planned = 0;
    temp_allocator.reset();
    heap_caps_free(temp_arena);
    temp_arena = nullptr;
    allocator.reset();
    heap_caps_free(arena);
    arena = nullptr;
  }

  et_model_config_t cfg;
  BufferDataLoader loader;
  std::optional<Program> program;

  // Loaded-method state (all zero/null until et_model_load_method).
  uint8_t* arena = nullptr;
  std::optional<MemoryAllocator> allocator;
  uint8_t* temp_arena = nullptr;
  std::optional<MemoryAllocator> temp_allocator;
  uint8_t* planned_ptrs[kMaxPlannedBuffers] = {};
  SpanU8 planned_spans[kMaxPlannedBuffers];
  size_t num_planned = 0;
  std::optional<HierarchicalAllocator> planned;
  std::optional<MemoryManager> manager;
  std::optional<Method> method;
  std::optional<MethodMeta> meta;

  char error[128] = "no error";
};

namespace {

bool fill_info(
    et_model* model,
    const Result<TensorInfo>& info,
    et_tensor_info_t* out) {
  if (!info.ok()) {
    model->fail("tensor_meta", info.error());
    return false;
  }
  if (out == nullptr) {
    model->fail("null et_tensor_info_t");
    return false;
  }
  if (info->sizes().size() > ET_MODEL_MAX_DIMS) {
    model->fail("tensor has more than ET_MODEL_MAX_DIMS dims");
    return false;
  }
  memset(out, 0, sizeof(*out));
  out->scalar_type = static_cast<int32_t>(info->scalar_type());
  out->ndim = info->sizes().size();
  for (size_t i = 0; i < out->ndim; i++) {
    out->dims[i] = info->sizes()[i];
  }
  out->nbytes = info->nbytes();
  return true;
}

// Resolves `name` (NULL = first method). The returned name is either
// `name` or owned by the program.
bool method_meta_for(
    et_model* model,
    const char* name,
    std::optional<MethodMeta>* out_meta,
    const char** out_name) {
  const char* resolved = name;
  if (resolved == nullptr) {
    auto name_res = model->program->get_method_name(0);
    if (!name_res.ok()) {
      model->fail("program has no methods");
      return false;
    }
    resolved = *name_res;
  }
  auto meta_res = model->program->method_meta(resolved);
  if (!meta_res.ok()) {
    model->fail("method_meta", meta_res.error());
    return false;
  }
  out_meta->emplace(std::move(meta_res.get()));
  *out_name = resolved;
  return true;
}

bool require_method(et_model* model) {
  if (!model->method) {
    model->fail("no method loaded");
    return false;
  }
  return true;
}

} // namespace

extern "C" {

et_model_t* et_model_create(const et_model_config_t* cfg) {
  if (cfg == nullptr || cfg->pte == nullptr || cfg->pte_size == 0) {
    ESP_LOGE(kTag, "et_model_create: invalid config");
    return nullptr;
  }
  if (reinterpret_cast<uintptr_t>(cfg->pte) % kPteAlignment != 0) {
    ESP_LOGE(
        kTag,
        "et_model_create: pte %p must be %zu-byte aligned",
        cfg->pte,
        kPteAlignment);
    return nullptr;
  }
  if (!valid_location(cfg->arena_location) ||
      !valid_location(cfg->temp_location) ||
      !valid_location(cfg->planned_location)) {
    ESP_LOGE(kTag, "et_model_create: invalid et_mem_location_t");
    return nullptr;
  }
  executorch::runtime::runtime_init();

  std::unique_ptr<et_model> model(new (std::nothrow) et_model(*cfg));
  if (!model) {
    ESP_LOGE(kTag, "et_model_create: out of memory");
    return nullptr;
  }
  auto program_res = Program::load(
      &model->loader,
      cfg->verify ? Program::Verification::InternalConsistency
                  : Program::Verification::Minimal);
  if (!program_res.ok()) {
    model->fail("Program::load", program_res.error());
    return nullptr;
  }
  model->program.emplace(std::move(program_res.get()));
  ESP_LOGI(
      kTag,
      "loaded program: %zu bytes, %zu method(s)",
      cfg->pte_size,
      static_cast<size_t>(model->program->num_methods()));
  return model.release();
}

void et_model_destroy(et_model_t* model) {
  if (model == nullptr) {
    return;
  }
  model->free_method();
  delete model;
}

size_t et_model_num_methods(et_model_t* model) {
  if (model == nullptr) {
    return 0;
  }
  return static_cast<size_t>(model->program->num_methods());
}

const char* et_model_method_name(et_model_t* model, size_t index) {
  if (model == nullptr) {
    return nullptr;
  }
  auto res = model->program->get_method_name(index);
  if (!res.ok()) {
    model->fail("get_method_name", res.error());
    return nullptr;
  }
  return *res;
}

bool et_model_load_method(et_model_t* model, const char* name) {
  if (model == nullptr) {
    return false;
  }
  model->free_method();

  std::optional<MethodMeta> meta;
  const char* resolved = nullptr;
  if (!method_meta_for(model, name, &meta, &resolved)) {
    return false;
  }
  const size_t num_planned = meta->num_memory_planned_buffers();
  if (num_planned > kMaxPlannedBuffers) {
    model->fail("too many planned buffers");
    return false;
  }

  const size_t arena_size =
      model->cfg.arena_size ? model->cfg.arena_size : kDefaultArenaSize;
  model->arena = caps_alloc(arena_size, model->cfg.arena_location);
  if (model->arena == nullptr) {
    model->fail("allocating method arena failed");
    return false;
  }
  model->allocator.emplace(static_cast<uint32_t>(arena_size), model->arena);

  const size_t temp_arena_size = model->cfg.temp_arena_size;
  if (temp_arena_size != 0) {
    model->temp_arena = caps_alloc(temp_arena_size, model->cfg.temp_location);
    if (model->temp_arena == nullptr) {
      model->free_method();
      model->fail("allocating temp arena failed");
      return false;
    }
    model->temp_allocator.emplace(
        static_cast<uint32_t>(temp_arena_size), model->temp_arena);
  }

  for (size_t i = 0; i < num_planned; i++) {
    auto size_res = meta->memory_planned_buffer_size(i);
    if (!size_res.ok()) {
      model->free_method();
      model->fail("memory_planned_buffer_size", size_res.error());
      return false;
    }
    const size_t bytes = static_cast<size_t>(size_res.get());
    uint8_t* buf = caps_alloc(bytes, model->cfg.planned_location);
    if (buf == nullptr) {
      model->free_method();
      model->fail("allocating planned buffer failed");
      return false;
    }
    model->planned_ptrs[i] = buf;
    model->planned_spans[i] = SpanU8(buf, bytes);
    model->num_planned = i + 1;
  }
  model->planned.emplace(
      executorch::runtime::Span<SpanU8>(model->planned_spans, num_planned));
  // A null temp allocator makes Method fall back to PAL allocation.
  model->manager.emplace(
      &*model->allocator,
      &*model->planned,
      model->temp_allocator ? &*model->temp_allocator : nullptr);

  auto method_res = model->program->load_method(resolved, &*model->manager);
  if (!method_res.ok()) {
    Error err = method_res.error();
    model->free_method();
    model->fail("load_method", err);
    return false;
  }
  model->method.emplace(std::move(method_res.get()));
  model->meta.emplace(std::move(*meta));
  ESP_LOGI(
      kTag,
      "loaded method '%s': arena %zu B, temp %zu B, %zu planned buffer(s)",
      resolved,
      arena_size,
      temp_arena_size,
      num_planned);
  return true;
}

bool et_model_planned_info(
    et_model_t* model,
    const char* name,
    size_t* out_num_buffers,
    size_t* out_total_bytes) {
  if (model == nullptr) {
    return false;
  }
  std::optional<MethodMeta> meta;
  if (name == nullptr) {
    if (!model->meta) {
      model->fail("no method loaded");
      return false;
    }
    meta = model->meta;
  } else {
    const char* resolved = nullptr;
    if (!method_meta_for(model, name, &meta, &resolved)) {
      return false;
    }
  }
  size_t total = 0;
  for (size_t i = 0; i < meta->num_memory_planned_buffers(); i++) {
    auto size_res = meta->memory_planned_buffer_size(i);
    if (!size_res.ok()) {
      model->fail("memory_planned_buffer_size", size_res.error());
      return false;
    }
    total += static_cast<size_t>(size_res.get());
  }
  if (out_num_buffers != nullptr) {
    *out_num_buffers = meta->num_memory_planned_buffers();
  }
  if (out_total_bytes != nullptr) {
    *out_total_bytes = total;
  }
  return true;
}

size_t et_model_num_inputs(et_model_t* model) {
  return (model != nullptr && model->meta) ? model->meta->num_inputs() : 0;
}

size_t et_model_num_outputs(et_model_t* model) {
  return (model != nullptr && model->meta) ? model->meta->num_outputs() : 0;
}

bool et_model_input_info(
    et_model_t* model,
    size_t index,
    et_tensor_info_t* out) {
  if (model == nullptr || !require_method(model)) {
    return false;
  }
  return fill_info(model, model->meta->input_tensor_meta(index), out);
}

bool et_model_output_info(
    et_model_t* model,
    size_t index,
    et_tensor_info_t* out) {
  if (model == nullptr || !require_method(model)) {
    return false;
  }
  return fill_info(model, model->meta->output_tensor_meta(index), out);
}

bool et_model_set_input(
    et_model_t* model,
    size_t index,
    const void* data,
    size_t nbytes) {
  if (model == nullptr || !require_method(model)) {
    return false;
  }
  auto info = model->meta->input_tensor_meta(index);
  if (!info.ok()) {
    model->fail("input_tensor_meta", info.error());
    return false;
  }
  if (info->nbytes() != nbytes) {
    model->fail("input size does not match model");
    return false;
  }
  const size_t ndim = info->sizes().size();
  if (ndim > ET_MODEL_MAX_DIMS) {
    model->fail("input has more than ET_MODEL_MAX_DIMS dims");
    return false;
  }
  // Method copies planned inputs and shares only the data pointer of
  // non-planned ones, so the tensor wrapper can live on the stack.
  int32_t sizes[ET_MODEL_MAX_DIMS];
  uint8_t dim_order[ET_MODEL_MAX_DIMS];
  for (size_t d = 0; d < ndim; d++) {
    sizes[d] = info->sizes()[d];
    dim_order[d] = info->dim_order()[d];
  }
  TensorImpl impl(
      info->scalar_type(),
      static_cast<ssize_t>(ndim),
      sizes,
      const_cast<void*>(data),
      dim_order);
  Error err = model->method->set_input(EValue(Tensor(&impl)), index);
  if (err != Error::Ok) {
    model->fail("set_input", err);
    return false;
  }
  return true;
}

bool et_model_execute(et_model_t* model) {
  if (model == nullptr || !require_method(model)) {
    return false;
  }
  Error err = model->method->execute();
  if (err != Error::Ok) {
    model->fail("execute", err);
    return false;
  }
  return true;
}

const void*
et_model_output_data(et_model_t* model, size_t index, size_t* out_nbytes) {
  if (out_nbytes != nullptr) {
    *out_nbytes = 0;
  }
  if (model == nullptr || !require_method(model)) {
    return nullptr;
  }
  if (index >= model->method->outputs_size()) {
    model->fail("output index out of range");
    return nullptr;
  }
  const EValue& output = model->method->get_output(index);
  if (!output.isTensor()) {
    model->fail("output is not a tensor");
    return nullptr;
  }
  const Tensor& tensor = output.toTensor();
  if (out_nbytes != nullptr) {
    *out_nbytes = tensor.nbytes();
  }
  return tensor.const_data_ptr();
}

const char* et_model_last_error(et_model_t* model) {
  return model != nullptr ? model->error : "null model";
}

} // extern "C"
