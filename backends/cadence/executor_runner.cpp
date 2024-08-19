/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This is a simple executor_runner that boots up the DSP, configures the serial
 * port, sends a bunch of test messages to the M33 core and then loads the model
 * defined in model_pte.h. It runs this model using the ops available in
 * cadence/ops directory.
 */

#include <fsl_debug_console.h>
#include "fsl_device_registers.h"
#include "fsl_mu.h"

#include "board_hifi4.h"
#include "model_pte.h"
#include "pin_mux.h"

#include <memory>
// patternlint-disable executorch-cpp-nostdinc
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/runner_util/inputs.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

static uint8_t method_allocator_pool[18 * 1024U]; // 4 MB

using namespace torch::executor;
#include <xtensa/config/core.h>

#define APP_MU MUB
/* Flag indicates Core Boot Up*/
#define BOOT_FLAG 0x01U
/* Channel transmit and receive register */
#define CHN_MU_REG_NUM 0U
/* How many message is used to test message sending */
#define MSG_LENGTH 32U

using torch::executor::Error;
using torch::executor::Result;

void LED_INIT();
void LED_TOGGLE();

void LED_INIT() {
  CLOCK_EnableClock(kCLOCK_HsGpio0);
  RESET_PeripheralReset(kHSGPIO0_RST_SHIFT_RSTn);
  gpio_pin_config_t pin_config = {kGPIO_DigitalOutput, LOGIC_LED_OFF};
  GPIO_PinInit(
      BOARD_LED_RED_GPIO,
      BOARD_LED_RED_GPIO_PORT,
      BOARD_LED_RED_GPIO_PIN,
      &pin_config);
}

void LED_TOGGLE() {
  LED_RED_TOGGLE();
}

/*!
 * @brief Function to create delay for Led blink.
 */
void delay(void) {
  volatile uint32_t i = 0;
  for (i = 0; i < 50000000; ++i) {
    __NOP();
  }
}

void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  PRINTF("\r%s\n", message);
}

int main(int argc, char** argv) {
  /* Init board hardware. */
  BOARD_InitBootPins();

  /* Initialize LED */
  LED_INIT();

  /* MUB init */
  MU_Init(APP_MU);

  /* Send flag to Core 0 to indicate Core 1 has startup */
  MU_SetFlags(APP_MU, BOOT_FLAG);

  BOARD_InitDebugConsole();
  ET_LOG(Info, "Booted up in DSP.");

  torch::executor::runtime_init();

  auto loader =
      torch::executor::util::BufferDataLoader(model_pte, sizeof(model_pte));

  Result<torch::executor::Program> program =
      torch::executor::Program::load(&loader);
  if (!program.ok()) {
    ET_LOG(
        Error,
        "ET: Program loading failed @ 0x%p: 0x%" PRIx32,
        model_pte,
        program.error());
  }

  ET_LOG(
      Info, "AET: Model buffer loaded, has %u methods", program->num_methods());

  const char* method_name = nullptr;
  {
    const auto method_name_result = program->get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;
  }
  ET_LOG(Info, "ET: Running method %s", method_name);

  Result<torch::executor::MethodMeta> method_meta =
      program->method_meta(method_name);
  if (!method_meta.ok()) {
    ET_LOG(
        Error,
        "ET: Failed to get method_meta for %s: 0x%x",
        method_name,
        (unsigned int)method_meta.error());
  }

  torch::executor::MemoryAllocator method_allocator{
      torch::executor::MemoryAllocator(
          sizeof(method_allocator_pool), method_allocator_pool)};

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers; // Owns the memory
  std::vector<torch::executor::Span<uint8_t>>
      planned_spans; // Passed to the allocator
  size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();

  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
    ET_LOG(
        Info, "ET: Setting up planned buffer %zu, size %zu.", id, buffer_size);

    planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans.push_back({planned_buffers.back().get(), buffer_size});
  }

  torch::executor::HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});

  torch::executor::MemoryManager memory_manager(
      &method_allocator, &planned_memory);

  Result<torch::executor::Method> method =
      program->load_method(method_name, &memory_manager);
  if (!method.ok()) {
    ET_LOG(
        Error,
        "Loading of method %s failed with status 0x%" PRIx32,
        method_name,
        method.error());
  }

  ET_LOG(Info, "Method loaded.");
  torch::executor::util::prepare_input_tensors(*method);
  ET_LOG(Info, "Starting the model execution...");

  Error status = method->execute();
  ET_LOG(Info, "Executed model");
  if (status != Error::Ok) {
    ET_LOG(
        Error,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        status);
  } else {
    ET_LOG(Info, "Model executed successfully.");
  }

  std::vector<EValue> outputs(method->outputs_size());
  status = method->get_outputs(outputs.data(), outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    float* out_data_ptr = (float*)outputs[i].toTensor().const_data_ptr();
    ET_LOG(Info, "First 20 elements of output %d", i);
    for (size_t j = 0; j < 20; j++) {
      ET_LOG(Info, "%f \n", out_data_ptr[j]);
    }
  }
  delay();
  LED_TOGGLE();

  return 0;
}
