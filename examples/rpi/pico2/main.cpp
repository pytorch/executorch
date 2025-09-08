/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Model data
#include "model_pte.h"

// Pico includes
#include "pico/stdio_usb.h"
#include "pico/stdlib.h"

// Executorch includes
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

// Std c++ includes
#include <memory>

using namespace executorch::runtime;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using ScalarType = executorch::runtime::etensor::ScalarType;
using executorch::runtime::runtime_init;

// Define GPIO pins for indicators
const uint INDICATOR_PIN_1 = 25; // Onboard LED
const uint INDICATOR_PIN_2 = 22; // External LED
const uint INDICATOR_PIN_3 = 23; // Onboard LED

void init_gpio_pins() {
  gpio_init(INDICATOR_PIN_1);
  gpio_set_dir(INDICATOR_PIN_1, GPIO_OUT);
  gpio_init(INDICATOR_PIN_2);
  gpio_set_dir(INDICATOR_PIN_2, GPIO_OUT);
  gpio_init(INDICATOR_PIN_3);
  gpio_set_dir(INDICATOR_PIN_3, GPIO_OUT);
}

void wait_for_usb() {
  const int kMaxRetryCount = 10;
  int retry_usb_count = 0;
  while (!stdio_usb_connected() && retry_usb_count++ < kMaxRetryCount) {
    printf("Retry again! USB not connected \n");
    sleep_ms(1000);
  }
}

// Helper function to blink an indicator pin on the pico board a given number of
// times
void blink_indicator(uint pin, int times, int delay_ms = 100) {
  for (int i = 0; i < times; ++i) {
    gpio_put(pin, 1);
    sleep_ms(delay_ms);
    gpio_put(pin, 0);
    sleep_ms(delay_ms);
  }
}

bool load_and_prepare_model(
    std::unique_ptr<Program>& program_ptr,
    std::unique_ptr<Method>& method_ptr,
    MemoryManager& memory_manager) {
  executorch::extension::BufferDataLoader loader(model_pte, model_pte_len);
  auto program_result = Program::load(&loader);
  if (!program_result.ok()) {
    printf("Failed to load model: %d\n", (int)program_result.error());
    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }
  program_ptr = std::make_unique<Program>(std::move(*program_result));
  auto method_name_result = program_ptr->get_method_name(0);
  if (!method_name_result.ok()) {
    printf("Failed to get method name: %d\n", (int)method_name_result.error());
    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }
  auto method_result =
      program_ptr->load_method(*method_name_result, &memory_manager);
  if (!method_result.ok()) {
    printf("Failed to load method: %d\n", (int)method_result.error());
    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }
  method_ptr = std::make_unique<Method>(std::move(*method_result));
  printf("Method loaded [%s]\n", *method_name_result);
  return true;
}

bool validate_and_set_inputs(Method& method) {
  float input_data_0[4] = {4.0, 109.0, 13.0, 123.0};
  float input_data_1[4] = {9.0, 27.0, 11.0, 8.0};
  TensorImpl::SizesType sizes[1] = {4};
  TensorImpl::DimOrderType dim_order[] = {0};
  TensorImpl impl0(ScalarType::Float, 1, sizes, input_data_0, dim_order);
  TensorImpl impl1(ScalarType::Float, 1, sizes, input_data_1, dim_order);
  Tensor input0(&impl0);
  Tensor input1(&impl1);
  if (method.set_input(input0, 0) != Error::Ok) {
    printf("Failed to set input0\n");
    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }
  if (method.set_input(input1, 1) != Error::Ok) {
    printf("Failed to set input1\n");
    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }
  return true;
}

bool run_inference(Method& method) {
  if (!validate_and_set_inputs(method)) {
    return false; // Input validation or setting failed
  }

  if (method.execute() != Error::Ok) {
    printf("Failed to execute\n");
    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }
  const EValue& output = method.get_output(0);
  if (output.isTensor()) {
    const float* out_data = output.toTensor().const_data_ptr<float>();
    printf(
        "Output: %f, %f, %f, %f\n",
        out_data[0],
        out_data[1],
        out_data[2],
        out_data[3]);
  } else {
    printf("Output is not a tensor!\n");
    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }
  return true;
}

int executor_runner() {
  init_gpio_pins();
  stdio_init_all();
  sleep_ms(1000);

  wait_for_usb();
  runtime_init();

  static uint8_t method_allocator_pool[32 * 1024]; // 32KB
  static uint8_t activation_pool[64 * 1024]; // 64KB
  MemoryAllocator method_allocator(
      sizeof(method_allocator_pool), method_allocator_pool);
  method_allocator.enable_profiling("method allocator");
  Span<uint8_t> memory_planned_buffers[1]{
      {activation_pool, sizeof(activation_pool)}};
  HierarchicalAllocator planned_memory({memory_planned_buffers, 1});
  MemoryManager memory_manager(&method_allocator, &planned_memory);

  std::unique_ptr<Program> program_ptr;
  std::unique_ptr<Method> method_ptr;
  if (!load_and_prepare_model(program_ptr, method_ptr, memory_manager)) {
    printf("Failed to load and prepare model\n");
    return 1;
  }
  if (!run_inference(*method_ptr)) {
    printf("Failed to run inference\n");
    return 1;
  }

  // If everything went well, it will blink the indicator pin
  blink_indicator(INDICATOR_PIN_1, 10, 500);
  return 0;
}

int main() {
  return executor_runner();
}
