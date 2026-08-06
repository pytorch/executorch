/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <errno.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <memory>
#include <set>
#include <vector>

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

// newlib-nano printf does not reliably handle the z length modifier. Values
// logged by this app use unsigned long with %lu instead of size_t with %zu.
using printf_size_t = unsigned long;

#include "arm_memory_allocator.h"

#include "image.h"
#include "model_pte.h"

// This application is for a model fine tuned on the Oxford-IIIT Pet
// dataset(https://huggingface.co/datasets/timm/oxford-iiit-pet/blob/main/README.md)
// The dataset contains the following 37 pet breed labels (cats and dogs).
constexpr const char* labels[] = {
    "abyssinian",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "bengal",
    "birman",
    "bombay",
    "boxer",
    "british_shorthair",
    "chihuahua",
    "egyptian_mau",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "maine_coon",
    "miniature_pinscher",
    "newfoundland",
    "persian",
    "pomeranian",
    "pug",
    "ragdoll",
    "russian_blue",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "siamese",
    "sphynx",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier"};

const size_t method_allocation_pool_size = 1 * 1024 * 1024;
unsigned char __attribute__((
    section("input_data_sec"),
    aligned(16))) method_allocation_pool[method_allocation_pool_size];

/*
The to_edge_tranform_and_lower step reports
    Total SRAM used                               1291.80 KiB
therefore, we allocate 1.3MB in the temporary allocation pool store the peak
intermediate tensor for the inference.
*/
const size_t temp_allocation_pool_size = 1.3 * 1024 * 1024;
unsigned char __attribute__((
    section(".bss.tensor_arena"),
    aligned(16))) temp_allocation_pool[temp_allocation_pool_size];

// cppcheck-suppress unusedFunction
void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  printf_size_t log_line = line;
  fprintf(
      stderr,
      "%c [executorch:%s:%lu %s()] %s\n",
      level,
      filename,
      log_line,
      function,
      message);
}

int main() {
  executorch::runtime::runtime_init();
  ET_LOG(Info, "Runtime initialized");
  printf_size_t model_pte_size = sizeof(model_pte);
  BufferDataLoader loader(model_pte, model_pte_size);
  ET_LOG(Info, "Size of the model = %lu", model_pte_size);
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

  std::vector<uint8_t*> planned_buffers; // Owns the memory
  std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
  printf_size_t num_memory_planned_buffers =
      method_meta_result->num_memory_planned_buffers();
  ET_LOG(Info, "num_memory_planned_buffers = %lu", num_memory_planned_buffers);
  for (printf_size_t id = 0; id < num_memory_planned_buffers; ++id) {
    printf_size_t buffer_size =
        method_meta_result->memory_planned_buffer_size(id).get();
    ET_LOG(Info, "Planned memory buffer_size %lu %lu bytes", id, buffer_size);

    uint8_t* buffer = reinterpret_cast<uint8_t*>(
        method_allocator.allocate(buffer_size, 16UL));

    ET_CHECK_MSG(
        buffer != nullptr,
        "Could not allocate memory for memory planned buffer size %lu",
        buffer_size);
    planned_buffers.push_back(buffer);
    planned_spans.push_back({planned_buffers.back(), buffer_size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  Result<Method> method = program->load_method(method_name, &memory_manager);

  printf_size_t num_inputs = method->inputs_size();
  ET_LOG(Info, "Number of input tensors = %lu", num_inputs);
  ET_CHECK_MSG(
      num_inputs == 1,
      "DEiT-Tiny has a single input tensor, but the provided model has more input tensors");

  EValue* input_evalues = method_allocator.allocateList<EValue>(num_inputs);
  Error err = method->get_inputs(input_evalues, num_inputs);
  ET_CHECK_MSG(err == Error::Ok, "Get inputs failed");
  Tensor& tensor =
      input_evalues[0].toTensor(); // DEiT-Tiny has a single input tensor.
  printf_size_t expected_elems = tensor.numel();

  printf_size_t image_elements = sizeof(image_data) /
      sizeof(image_data[0]); // number of elements of the array in image.h
  ET_CHECK_MSG(
      expected_elems == image_elements,
      "Input tensor expects %lu elements, but image_data has %lu elements",
      expected_elems,
      image_elements);

  switch (tensor.scalar_type()) {
    case ScalarType::Float: {
      float* dst = tensor.mutable_data_ptr<float>();
      for (size_t j = 0; j < tensor.numel(); ++j) {
        dst[j] = image_data[j];
      }
      break;
    }
    default:
      ET_CHECK_MSG(
          false,
          "Input tensor datatype is not float. The image data we want to populate in the input tensor is float");
      break;
  }
  Error status_inference = method->execute(); // run inference
  ET_CHECK_MSG(
      status_inference == Error::Ok,
      "Inference failed 0x%" PRIx32,
      status_inference);

  size_t num_outputs = method->outputs_size();
  std::vector<EValue> outputs(num_outputs);
  Error status_outputs = method->get_outputs(outputs.data(), outputs.size());
  ET_CHECK_MSG(
      status_outputs == Error::Ok,
      "get_outputs failed 0x%" PRIx32,
      status_outputs);

  std::set<std::pair<float, printf_size_t>> set_confidence_idx;
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (!outputs[i].isTensor())
      continue;
    Tensor out = outputs[i].toTensor();
    switch (out.scalar_type()) {
      case ScalarType::Float: {
        // When we generate the pte file in the AoT flow, we use float32
        // datatype as input to the model(with Q/DQ nodes around every
        // operator). Therefore, we only handle the float32 case in the
        // application logic.
        const float* data = out.const_data_ptr<float>();
        for (printf_size_t j = 0; j < out.numel(); ++j)
          set_confidence_idx.insert({data[j], j});
        break;
      }
      default:
        ET_LOG(
            Info, "Output tensor has unsupported dtype %d", out.scalar_type());
        break;
    }
  }
  printf_size_t printed = 0;
  printf_size_t topK = 5;
  printf_size_t num_labels = sizeof(labels) / sizeof(labels[0]);
  ET_LOG(
      Info,
      "Top %lu classes in descending order(highest probability is at the top)",
      topK);
  for (auto it = set_confidence_idx.rbegin();
       it != set_confidence_idx.rend() && printed < topK;
       ++it, ++printed) {
    printf_size_t class_id = it->second;
    const char* class_name =
        (class_id < num_labels) ? labels[class_id] : "unknown";
    ET_LOG(
        Info,
        "Class %lu ( %s ) with score of %f",
        class_id,
        class_name,
        it->first);
  }
  return 0;
}
