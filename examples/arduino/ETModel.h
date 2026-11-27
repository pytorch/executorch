/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * A small wrapper over the ExecuTorch runtime for sketches that just want to
 * run a model.
 *
 * Loading a method by hand means asking the program for its method name,
 * asking that for its metadata, counting its planned buffers, allocating a
 * span array, sizing and allocating each buffer, wrapping them in a
 * HierarchicalAllocator and pairing that with a MemoryAllocator in a
 * MemoryManager. That is around sixty lines before any inference happens, and
 * none of it is a decision the sketch author makes.
 *
 *   alignas(16) static uint8_t pool[8 * 1024];
 *   static ETModel model(model_pte, sizeof(model_pte), pool, sizeof(pool));
 *
 *   if (!model.begin()) { Serial.println(model.error()); }
 *   model.setInput(0, values, 3);
 *   if (!model.run()) { Serial.println(model.error()); }
 *   const float* out = model.output(0);
 *
 * error() returns a readable reason rather than a code, because a bare hex
 * status is what makes these failures expensive to diagnose. The full runtime
 * API stays available through program() and method() for anything this does
 * not cover.
 */

#pragma once

#include <ExecuTorch.h>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <utility>

class ETModel {
 public:
  // The model buffer and the pool must outlive this object, which on Arduino
  // means both are normally static or global.
  ETModel(
      const uint8_t* model_data,
      size_t model_size,
      uint8_t* pool,
      size_t pool_size)
      : loader_(model_data, model_size),
        allocator_(static_cast<uint32_t>(pool_size), pool) {}

  // Loads the program and its first method. Call once, from setup().
  bool begin() {
    auto program = executorch::runtime::Program::load(&loader_);
    if (!program.ok()) {
      return fail("Program::load", static_cast<int>(program.error()));
    }
    program_.emplace(std::move(program.get()));

    auto name = program_->get_method_name(0);
    if (!name.ok()) {
      return fail("get_method_name", static_cast<int>(name.error()));
    }
    method_name_ = *name;

    auto meta = program_->method_meta(method_name_);
    if (!meta.ok()) {
      return fail("method_meta", static_cast<int>(meta.error()));
    }

    // Planned buffers hold the intermediate tensors the memory planner sized
    // at export time. They come out of the same pool as the method itself.
    const size_t count = meta->num_memory_planned_buffers();
    auto* spans = static_cast<Span*>(allocator_.allocate(count * sizeof(Span)));
    if (spans == nullptr) {
      return fail("allocating the planned-buffer table", pool_hint());
    }
    for (size_t i = 0; i < count; i++) {
      auto size = meta->memory_planned_buffer_size(i);
      if (!size.ok()) {
        return fail(
            "memory_planned_buffer_size", static_cast<int>(size.error()));
      }
      auto bytes = static_cast<size_t>(size.get());
      auto* buffer = static_cast<uint8_t*>(allocator_.allocate(bytes));
      if (buffer == nullptr) {
        return fail("allocating a planned buffer", pool_hint());
      }
      spans[i] = Span(buffer, bytes);
    }

    planned_.emplace(executorch::runtime::Span<Span>(spans, count));
    manager_.emplace(&allocator_, &*planned_);

    auto method = program_->load_method(method_name_, &*manager_);
    if (!method.ok()) {
      // Much the most common failure, and the pool is nearly always why.
      return fail("load_method", static_cast<int>(method.error()));
    }
    method_.emplace(std::move(method.get()));
    return true;
  }

  // Points input `index` at `data` without copying it, so `data` must stay
  // alive across run(). Passing a `const` array in flash works for every model
  // here, but an operator that writes to its input would be writing to
  // read-only memory; copy into a RAM buffer first if that is a risk. Not
  // copying is the right default on a board with this little RAM -- the
  // keyword spotting input alone is 2 KB.
  //
  // The shape and dtype come from the model, so only the element count is
  // checked against it.
  bool setInput(size_t index, const float* data, size_t count) {
    if (!method_) {
      return fail("setInput before begin()", 0);
    }
    if (index >= kMaxInputs) {
      return fail("input index above the built-in limit", kMaxInputs);
    }
    auto meta = program_->method_meta(method_name_);
    auto info = meta->input_tensor_meta(index);
    if (!info.ok()) {
      return fail("input_tensor_meta", static_cast<int>(info.error()));
    }
    size_t expected = 1;
    for (size_t d = 0; d < info->sizes().size(); d++) {
      sizes_[index][d] = info->sizes()[d];
      dim_order_[index][d] = info->dim_order()[d];
      expected *= static_cast<size_t>(info->sizes()[d]);
    }
    if (expected != count) {
      return fail(
          "input element count does not match the model",
          static_cast<int>(expected));
    }
    impls_[index].emplace(
        info->scalar_type(),
        static_cast<ssize_t>(info->sizes().size()),
        sizes_[index],
        const_cast<float*>(data),
        dim_order_[index]);
    tensors_[index].emplace(&*impls_[index]);
    auto status = method_->set_input(
        executorch::runtime::EValue(*tensors_[index]), index);
    if (status != executorch::runtime::Error::Ok) {
      return fail("set_input", static_cast<int>(status));
    }
    return true;
  }

  bool run() {
    if (!method_) {
      return fail("run before begin()", 0);
    }
    auto status = method_->execute();
    if (status != executorch::runtime::Error::Ok) {
      return fail("execute", static_cast<int>(status));
    }
    return true;
  }

  // Valid until the next run().
  const float* output(size_t index = 0) {
    const auto* tensor = outputTensor(index);
    return tensor ? tensor->const_data_ptr<float>() : nullptr;
  }

  size_t outputCount(size_t index = 0) {
    const auto* tensor = outputTensor(index);
    return tensor ? static_cast<size_t>(tensor->numel()) : 0;
  }

  // Index of the largest output element, which is what a classifier wants.
  int argmax(size_t index = 0) {
    const float* values = output(index);
    size_t count = outputCount(index);
    if (values == nullptr || count == 0) {
      return -1;
    }
    int best = 0;
    for (size_t i = 1; i < count; i++) {
      if (values[i] > values[best]) {
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  // Why the last call returned false. Never null.
  const char* error() const {
    return error_[0] ? error_ : "no error";
  }

  // For anything this wrapper does not cover.
  executorch::runtime::Program* program() {
    return program_ ? &*program_ : nullptr;
  }
  executorch::runtime::Method* method() {
    return method_ ? &*method_ : nullptr;
  }

 private:
  using Span = executorch::runtime::Span<uint8_t>;
  static constexpr size_t kMaxInputs = 4;
  static constexpr size_t kMaxDims = 8;

  const executorch::aten::Tensor* outputTensor(size_t index) {
    if (!method_) {
      fail("output before begin()", 0);
      return nullptr;
    }
    auto status = method_->get_outputs(&output_, 1 + index);
    if (status != executorch::runtime::Error::Ok || !output_.isTensor()) {
      fail("get_outputs", static_cast<int>(status));
      return nullptr;
    }
    out_tensor_.emplace(output_.toTensor());
    return &*out_tensor_;
  }

  // A pool shortfall is by far the most likely cause of a null allocation, and
  // the runtime already logged the exact numbers through et_arduino_log.
  int pool_hint() const {
    return static_cast<int>(allocator_.size());
  }

  bool fail(const char* what, int detail) {
    snprintf(error_, sizeof(error_), "%s failed (0x%x)", what, detail);
    return false;
  }

  executorch::extension::BufferDataLoader loader_;
  executorch::runtime::MemoryAllocator allocator_;
  std::optional<executorch::runtime::Program> program_;
  std::optional<executorch::runtime::HierarchicalAllocator> planned_;
  std::optional<executorch::runtime::MemoryManager> manager_;
  std::optional<executorch::runtime::Method> method_;
  const char* method_name_ = nullptr;

  std::optional<executorch::aten::TensorImpl> impls_[kMaxInputs];
  std::optional<executorch::aten::Tensor> tensors_[kMaxInputs];
  int32_t sizes_[kMaxInputs][kMaxDims] = {};
  uint8_t dim_order_[kMaxInputs][kMaxDims] = {};

  executorch::runtime::EValue output_;
  std::optional<executorch::aten::Tensor> out_tensor_;
  char error_[96] = {};
};
