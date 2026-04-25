/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * MobileNetV2 Image Classification using ExecuTorch with Ethos-U NPU
 */

#include <executorch/examples/arm/executor_runner/arm_memory_allocator.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <cstring>
#include <vector>

#include "model_pte.h"
#include "mv2_input.h"

// Runtime copy of the model blob so that the Ethos-U DMA can access command
// stream and weights. On Corstone FVP the original model_pte[] lives in flash
// which is not DMA-accessible. On boards where MRAM is DMA-accessible (e.g.
// Alif Ensemble), this copy can be bypassed via board-specific linker config.
alignas(16) static unsigned char model_pte_runtime[sizeof(model_pte)];
static bool model_pte_runtime_initialized = false;

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferCleanup;
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
using executorch::runtime::Tag;
using executorch::runtime::TensorInfo;

#if !defined(ET_ARM_METHOD_ALLOCATOR_POOL_SIZE)
#define ET_ARM_METHOD_ALLOCATOR_POOL_SIZE (1572864)
#endif
const size_t method_allocation_pool_size = ET_ARM_METHOD_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    aligned(16))) method_allocation_pool[method_allocation_pool_size];

#if !defined(ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
#define ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE (1572864)
#endif

#if !defined(ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
#define ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x600
#endif

const size_t temp_allocation_pool_size =
    ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((
    section(".bss.tensor_arena"),
    aligned(16))) temp_allocation_pool[temp_allocation_pool_size];

#if defined(ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
extern "C" {
size_t ethosu_fast_scratch_size =
    ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE;
unsigned char __attribute__((section(".bss.ethosu_scratch"), aligned(16)))
dedicated_sram[ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];
unsigned char* ethosu_fast_scratch = dedicated_sram;
}
#endif

#define MV2_NUM_OUTPUT_CLASSES 1000
#define MV2_TOP_K 5

namespace {

Result<BufferCleanup> prepare_input_tensors(
    Method& method,
    MemoryAllocator& allocator,
    const uint8_t* input_data,
    size_t input_size) {
  MethodMeta method_meta = method.method_meta();
  size_t num_inputs = method_meta.num_inputs();
  size_t num_allocated = 0;

  void** inputs =
      static_cast<void**>(allocator.allocate(num_inputs * sizeof(void*)));
  ET_CHECK_OR_RETURN_ERROR(
      inputs != nullptr,
      MemoryAllocationFailed,
      "Could not allocate memory for pointers to input buffers.");

  for (size_t i = 0; i < num_inputs; i++) {
    auto tag = method_meta.input_tag(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tag.error());

    if (tag.get() != Tag::Tensor) {
      ET_LOG(Debug, "Skipping non-tensor input %zu", i);
      continue;
    }
    Result<TensorInfo> tensor_meta = method_meta.input_tensor_meta(i);
    ET_CHECK_OK_OR_RETURN_ERROR(tensor_meta.error());

    void* data_ptr = allocator.allocate(tensor_meta->nbytes());
    ET_CHECK_OR_RETURN_ERROR(
        data_ptr != nullptr,
        MemoryAllocationFailed,
        "Could not allocate memory for input buffers.");
    inputs[num_allocated++] = data_ptr;

    Error err = Error::Ok;
    ScalarType scalar_type = tensor_meta->scalar_type();
    size_t num_elements = 1;
    auto sizes = tensor_meta->sizes();
    for (size_t k = 0; k < sizes.size(); k++) {
      num_elements *= sizes[k];
    }

    ET_LOG(
        Info,
        "Input tensor: scalar_type=%s, numel=%lu, nbytes=%lu",
        executorch::runtime::toString(scalar_type),
        static_cast<unsigned long>(num_elements),
        static_cast<unsigned long>(tensor_meta->nbytes()));

    if (scalar_type == ScalarType::Float && input_size == num_elements) {
      ET_LOG(
          Info,
          "Converting uint8 input (%lu elements) to float32",
          static_cast<unsigned long>(input_size));
      float* float_data = static_cast<float*>(data_ptr);
      for (size_t j = 0; j < input_size; j++) {
        float_data[j] = (static_cast<float>(input_data[j]) - 128.0f) / 128.0f;
      }
    } else if (input_size == tensor_meta->nbytes()) {
      ET_LOG(
          Info,
          "Copying input data to tensor (%lu bytes)",
          static_cast<unsigned long>(input_size));
      std::memcpy(data_ptr, input_data, input_size);
    } else {
      ET_LOG(
          Error,
          "Input size (%lu) and tensor size (%lu elements, %lu bytes) mismatch!",
          static_cast<unsigned long>(input_size),
          static_cast<unsigned long>(num_elements),
          static_cast<unsigned long>(tensor_meta->nbytes()));
      err = Error::InvalidArgument;
    }

    TensorImpl impl = TensorImpl(
        tensor_meta.get().scalar_type(),
        tensor_meta.get().sizes().size(),
        const_cast<TensorImpl::SizesType*>(tensor_meta.get().sizes().data()),
        data_ptr,
        const_cast<TensorImpl::DimOrderType*>(
            tensor_meta.get().dim_order().data()));
    Tensor t(&impl);

    if (err == Error::Ok) {
      err = method.set_input(t, i);
    }

    if (err != Error::Ok) {
      ET_LOG(
          Error, "Failed to prepare input %zu: 0x%" PRIx32, i, (uint32_t)err);
      BufferCleanup cleanup({inputs, num_allocated});
      return err;
    }
  }
  return BufferCleanup({inputs, num_allocated});
}

void print_top_k(const std::vector<EValue>& outputs) {
  if (outputs.empty() || !outputs[0].isTensor()) {
    ET_LOG(Error, "Output is not a tensor");
    return;
  }

  Tensor output_tensor = outputs[0].toTensor();
  ScalarType scalar_type = output_tensor.scalar_type();
  size_t num_classes = output_tensor.numel();

  ET_LOG(
      Info,
      "Output tensor: scalar_type=%s, numel=%lu",
      executorch::runtime::toString(scalar_type),
      static_cast<unsigned long>(num_classes));

  int top_indices[MV2_TOP_K] = {0};
  float top_values[MV2_TOP_K];

  for (int j = 0; j < MV2_TOP_K; j++) {
    top_values[j] = -1e9f;
  }

  for (size_t i = 0; i < num_classes; i++) {
    float val;
    switch (scalar_type) {
      case ScalarType::Float:
        val = output_tensor.const_data_ptr<float>()[i];
        break;
      case ScalarType::Int:
        val = static_cast<float>(output_tensor.const_data_ptr<int>()[i]);
        break;
      case ScalarType::Char:
        val = static_cast<float>(output_tensor.const_data_ptr<int8_t>()[i]);
        break;
      case ScalarType::Byte:
        val = static_cast<float>(output_tensor.const_data_ptr<uint8_t>()[i]);
        break;
      default:
        ET_LOG(
            Error,
            "Unsupported output scalar type: %s",
            executorch::runtime::toString(scalar_type));
        return;
    }

    for (int j = 0; j < MV2_TOP_K; j++) {
      if (val > top_values[j]) {
        for (int m = MV2_TOP_K - 1; m > j; m--) {
          top_values[m] = top_values[m - 1];
          top_indices[m] = top_indices[m - 1];
        }
        top_values[j] = val;
        top_indices[j] = static_cast<int>(i);
        break;
      }
    }
  }

  ET_LOG(Info, "\nTop-%d predictions:", MV2_TOP_K);
  for (int j = 0; j < MV2_TOP_K; j++) {
    ET_LOG(
        Info,
        "  [%d] class %d: %.4f",
        j + 1,
        top_indices[j],
        static_cast<double>(top_values[j]));
  }
}

} // namespace

int main(void) {
  printk("\n========================================\n");
  printk("ExecuTorch MobileNetV2 Classification Demo\n");
  printk("========================================\n\n");

  executorch::runtime::runtime_init();

  size_t pte_size = sizeof(model_pte);
  ET_LOG(
      Info,
      "Model PTE at %p, Size: %lu bytes",
      model_pte,
      static_cast<unsigned long>(pte_size));

  if (!model_pte_runtime_initialized) {
    std::memcpy(model_pte_runtime, model_pte, sizeof(model_pte));
    model_pte_runtime_initialized = true;
  }
  const void* program_data = model_pte_runtime;
  size_t program_data_len = pte_size;

  auto loader = BufferDataLoader(program_data, program_data_len);
  ET_LOG(
      Info,
      "Model data loaded. Size: %lu bytes.",
      static_cast<unsigned long>(program_data_len));

  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    ET_LOG(
        Info,
        "Program loading failed @ 0x%p: 0x%" PRIx32,
        program_data,
        static_cast<uint32_t>(program.error()));
    return 1;
  }

  ET_LOG(
      Info,
      "Model loaded, has %lu methods",
      static_cast<unsigned long>(program->num_methods()));

  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "Running method: %s", method_name);

  Result<MethodMeta> method_meta = program->method_meta(method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Info,
        "Failed to get method_meta for %s: 0x%x",
        method_name,
        (unsigned int)method_meta.error());
    return 1;
  }

  ET_LOG(
      Info,
      "Method allocator pool size: %lu bytes.",
      static_cast<unsigned long>(method_allocation_pool_size));

  ArmMemoryAllocator method_allocator(
      method_allocation_pool_size, method_allocation_pool);

  std::vector<uint8_t*> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(
        Info,
        "Setting up planned buffer %lu, size %lu.",
        static_cast<unsigned long>(id),
        static_cast<unsigned long>(buffer_size));

    uint8_t* buffer =
        reinterpret_cast<uint8_t*>(method_allocator.allocate(buffer_size));
    ET_CHECK_MSG(
        buffer != nullptr,
        "Could not allocate planned buffer size %lu",
        static_cast<unsigned long>(buffer_size));
    planned_buffers.push_back(buffer);
    planned_spans.push_back({planned_buffers.back(), buffer_size});
  }

  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  ArmMemoryAllocator temp_allocator(
      temp_allocation_pool_size, temp_allocation_pool);

  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  ET_LOG(Info, "Loading method...");
  executorch::runtime::EventTracer* event_tracer_ptr = nullptr;

  Result<Method> method =
      program->load_method(method_name, &memory_manager, event_tracer_ptr);

  if (!method.ok()) {
    ET_LOG(
        Info,
        "Loading of method %s failed with status 0x%" PRIx32,
        method_name,
        static_cast<uint32_t>(method.error()));
    return 1;
  }
  ET_LOG(Info, "Method '%s' loaded successfully", method_name);

  ET_LOG(
      Info,
      "Preparing input: static RGB image (%lu bytes)",
      static_cast<unsigned long>(sizeof(mv2_input_data)));

  {
    static auto prepared_inputs = ::prepare_input_tensors(
        *method, method_allocator, mv2_input_data, sizeof(mv2_input_data));

    if (!prepared_inputs.ok()) {
      ET_LOG(
          Info,
          "Preparing input failed: 0x%" PRIx32,
          static_cast<uint32_t>(prepared_inputs.error()));
      return 1;
    }
  }

  ET_LOG(Info, "\n--- Starting inference ---");
  uint32_t start_time = k_uptime_get_32();

  Error status = method->execute();

  uint32_t end_time = k_uptime_get_32();
  uint32_t inference_time = end_time - start_time;

  if (status != Error::Ok) {
    ET_LOG(
        Info,
        "Execution failed: 0x%" PRIx32,
        static_cast<uint32_t>(status));
    return 1;
  }
  ET_LOG(Info, "Inference completed in %u ms", inference_time);

  std::vector<EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK(status == Error::Ok);

  ET_LOG(Info, "\n--- Classification Results ---");
  print_top_k(outputs);

  ET_LOG(Info, "\n========================================");
  ET_LOG(Info, "MobileNetV2 Demo Complete");
  ET_LOG(
      Info,
      "Model size: %lu bytes",
      static_cast<unsigned long>(pte_size));
  ET_LOG(
      Info,
      "Input: 224x224x3 RGB image (%lu bytes)",
      static_cast<unsigned long>(sizeof(mv2_input_data)));
  ET_LOG(
      Info,
      "Output: %d ImageNet classes (top-%d shown)",
      MV2_NUM_OUTPUT_CLASSES,
      MV2_TOP_K);
  ET_LOG(Info, "Inference time: %u ms", inference_time);
  ET_LOG(Info, "========================================\n");

  return 0;
}
