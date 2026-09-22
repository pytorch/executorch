/* Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <stdio.h>
#include <algorithm>
#include <cinttypes>
#include <vector>

#include "arm_memory_allocator.h"

#if defined(ET_COMPILED_PTE)
#include "model_pte.h"
#elif !defined(ET_MODEL_PTE_ADDR) || !defined(ET_MODEL_PTE_SIZE)
#error "Fixed-address models require ET_MODEL_PTE_ADDR and ET_MODEL_PTE_SIZE"
#endif

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

#if !defined(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE)
#define ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE (4 * 1024 * 1024)
#endif

// Bare-metal targets do not have virtual memory or a large general-purpose
// heap. These statically sized arenas make the runtime's memory use explicit.
// The linker script places the persistent method pool and temporary tensor
// arena in memory that is accessible to both the CPU and Ethos-U.
alignas(16) __attribute__((section("input_data_sec"))) unsigned char method_pool
    [ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE];
alignas(16)
    __attribute__((section(".bss.tensor_arena"))) unsigned char temp_pool
        [ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE];

// Override the platform logging hook so ExecuTorch diagnostics are visible on
// the FVP UART (and on the corresponding output device on real hardware).
[[maybe_unused]] void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  fprintf(
      stderr,
      "%c [executorch:%s:%lu] %s\n",
      level,
      filename,
      static_cast<unsigned long>(line),
      message);
}

int main() {
  // Runtime initialization registers the platform abstraction before any
  // Program or Method objects are created.
  executorch::runtime::runtime_init();

#if defined(ET_COMPILED_PTE)
  // The normal example build converts the PTE into a C array and links it into
  // the ELF. sizeof(model_pte) therefore gives the loader an exact bound.
  const uint8_t* model_data = model_pte;
  const size_t model_size = sizeof(model_pte);
#else
  // A production system can keep a large PTE in flash or another
  // memory-mapped region instead. The address is supplied by the build.
  const uint8_t* model_data =
      reinterpret_cast<const uint8_t*>(ET_MODEL_PTE_ADDR);
  const size_t model_size = ET_MODEL_PTE_SIZE;
#endif

  // BufferDataLoader lets Program read the serialized PTE directly from its
  // current memory location, without first copying the whole model.
  BufferDataLoader loader(model_data, model_size);
  Result<Program> program = Program::load(&loader);
  ET_CHECK_MSG(program.ok(), "Program::load failed: 0x%x", program.error());

  // Classic ML exports normally contain a single inference method. Reading
  // its metadata first tells us how much planned memory must be provided
  // before the method itself can be loaded.
  const auto method_name = program->get_method_name(0);
  ET_CHECK_MSG(method_name.ok(), "Program has no methods");
  const auto method_meta = program->method_meta(*method_name);
  ET_CHECK_MSG(method_meta.ok(), "Could not load method metadata");

  // Persistent allocations, including method state and input storage, come
  // from method_pool. Temporary kernel allocations come from temp_pool and
  // only need to remain valid during execute().
  ArmMemoryAllocator method_allocator(sizeof(method_pool), method_pool);
  ArmMemoryAllocator temp_allocator(sizeof(temp_pool), temp_pool);

  // The exporter memory-plans intermediate and output tensors into one or more
  // numbered buffers. Allocate every requested buffer from the persistent
  // arena and preserve the spans for the lifetime of the loaded Method.
  std::vector<Span<uint8_t>> planned_spans;
  planned_spans.reserve(method_meta->num_memory_planned_buffers());
  for (size_t i = 0; i < method_meta->num_memory_planned_buffers(); ++i) {
    const size_t size = method_meta->memory_planned_buffer_size(i).get();
    auto* buffer = static_cast<uint8_t*>(method_allocator.allocate(size, 16));
    ET_CHECK_MSG(buffer != nullptr, "Could not allocate planned buffer %lu", i);
    planned_spans.push_back({buffer, size});
  }

  // HierarchicalAllocator maps the PTE's planned-buffer IDs to the spans above.
  // MemoryManager combines planned, persistent, and temporary allocation into
  // the three memory classes used while loading and executing a method.
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  // Keep Program, all allocators, planned_spans, and Method alive until after
  // inference. The Method retains references to memory owned by these objects.
  Result<Method> method = program->load_method(*method_name, &memory_manager);
  ET_CHECK_MSG(method.ok(), "Could not load method: 0x%x", method.error());

  // Graph inputs are caller-owned and are not generally backed by the
  // memory-planned buffers. Allocate each input from method_pool, construct a
  // tensor with the PTE's exact shape and scalar type, fill it with
  // deterministic ones, and bind it with set_input(). Using the method arena
  // rather than malloc is important because the bare-metal libc heap is
  // intentionally small and cannot hold common image inputs such as
  // MobileNetV2's.
  const auto loaded_method_meta = method->method_meta();
  for (size_t i = 0; i < loaded_method_meta.num_inputs(); ++i) {
    const auto input_tag = loaded_method_meta.input_tag(i);
    ET_CHECK_MSG(
        input_tag.ok(),
        "Could not read input %lu metadata",
        static_cast<unsigned long>(i));
    ET_CHECK_MSG(
        input_tag.get() == executorch::runtime::Tag::Tensor,
        "Input %lu is not a tensor",
        static_cast<unsigned long>(i));

    auto tensor_meta = loaded_method_meta.input_tensor_meta(i);
    ET_CHECK_MSG(
        tensor_meta.ok(),
        "Could not read input %lu tensor metadata",
        static_cast<unsigned long>(i));
    void* input_buffer = method_allocator.allocate(tensor_meta->nbytes(), 16);
    ET_CHECK_MSG(
        input_buffer != nullptr,
        "Could not allocate input %lu",
        static_cast<unsigned long>(i));
    TensorImpl input_impl(
        tensor_meta->scalar_type(),
        tensor_meta->sizes().size(),
        const_cast<TensorImpl::SizesType*>(tensor_meta->sizes().data()),
        input_buffer,
        const_cast<TensorImpl::DimOrderType*>(tensor_meta->dim_order().data()));
    Tensor input(&input_impl);
    switch (input.scalar_type()) {
#define FILL_INPUT(cpp_type, scalar_type)                \
  case ScalarType::scalar_type: {                        \
    cpp_type* data = input.mutable_data_ptr<cpp_type>(); \
    std::fill(data, data + input.numel(), cpp_type(1));  \
    break;                                               \
  }
      ET_FORALL_REALHBBF16_TYPES(FILL_INPUT)
#undef FILL_INPUT
      default:
        ET_CHECK_MSG(false, "Unsupported input tensor type");
    }
    ET_CHECK_MSG(
        method->set_input(input, i) == Error::Ok,
        "Could not prepare input %lu",
        static_cast<unsigned long>(i));
  }

  // All input and planned-buffer storage is now bound, so the method can safely
  // execute in place. Delegate calls run on Ethos-U and any remaining
  // operators run on the CPU.
  ET_CHECK_MSG(method->execute() == Error::Ok, "Inference failed");

  // Output EValues reference tensors in planned memory. They do not own or copy
  // the tensor data, so consume them while Method and planned_memory are alive.
  std::vector<EValue> outputs(method->outputs_size());
  ET_CHECK_MSG(
      method->get_outputs(outputs.data(), outputs.size()) == Error::Ok,
      "Could not get outputs");
  ET_LOG(
      Info,
      "Inference complete: %lu output(s)",
      static_cast<unsigned long>(outputs.size()));
  for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
    ET_CHECK_MSG(outputs[output_index].isTensor(), "Output is not a tensor");
    Tensor output = outputs[output_index].toTensor();

    // Print every element so this minimal runner is useful with classification
    // logits as well as small integer or boolean test models. Tensor storage is
    // interpreted according to ScalarType rather than as untyped bytes.
    for (size_t element_index = 0; element_index < output.numel();
         ++element_index) {
      printf(
          "Output[%lu][%lu]: ",
          static_cast<unsigned long>(output_index),
          static_cast<unsigned long>(element_index));
      switch (output.scalar_type()) {
        case ScalarType::Byte:
          printf(
              "%u\n",
              static_cast<unsigned int>(
                  output.const_data_ptr<uint8_t>()[element_index]));
          break;
        case ScalarType::Char:
          printf("%d\n", output.const_data_ptr<int8_t>()[element_index]);
          break;
        case ScalarType::Short:
          printf("%d\n", output.const_data_ptr<int16_t>()[element_index]);
          break;
        case ScalarType::Int:
          printf("%d\n", output.const_data_ptr<int32_t>()[element_index]);
          break;
        case ScalarType::Long:
          printf(
              "%" PRId64 "\n", output.const_data_ptr<int64_t>()[element_index]);
          break;
        case ScalarType::Half:
          printf(
              "%f\n",
              static_cast<double>(output.const_data_ptr<
                                  executorch::aten::Half>()[element_index]));
          break;
        case ScalarType::Float:
          printf("%f\n", output.const_data_ptr<float>()[element_index]);
          break;
        case ScalarType::Double:
          printf("%f\n", output.const_data_ptr<double>()[element_index]);
          break;
        case ScalarType::Bool:
          printf(
              "%s\n",
              output.const_data_ptr<bool>()[element_index] ? "true" : "false");
          break;
        case ScalarType::BFloat16:
          printf(
              "%f\n",
              static_cast<double>(
                  output.const_data_ptr<
                      executorch::aten::BFloat16>()[element_index]));
          break;
        default:
          ET_CHECK_MSG(false, "Unsupported output tensor type");
      }
    }
  }

  // ASCII EOT asks the FVP UART model to stop the simulation cleanly.
  printf("\04");
  return 0;
}
