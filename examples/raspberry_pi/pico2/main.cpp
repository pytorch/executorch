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
  printf("Loading model data (%u bytes)...\n", model_pte_len);

  executorch::extension::BufferDataLoader loader(model_pte, model_pte_len);
  auto program_result = Program::load(&loader);
  if (!program_result.ok()) {
    printf("❌ Failed to load model: error %d\n", (int)program_result.error());

    // Print more detailed error info
    switch (program_result.error()) {
      case Error::InvalidProgram:
        printf("   → Invalid program format\n");
        break;
      case Error::InvalidState:
        printf("   → Invalid state\n");
        break;
      case Error::NotSupported:
        printf("   → Feature not supported\n");
        break;
      case Error::NotFound:
        printf("   → Resource not found\n");
        break;
      case Error::InvalidArgument:
        printf("   → Invalid argument\n");
        break;
      default:
        printf("   → Unknown error code: %d\n", (int)program_result.error());
    }

    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }

  program_ptr = std::make_unique<Program>(std::move(*program_result));
  printf("✅ Program loaded successfully\n");

  // Get method count and names
  printf("📊 Program info:\n");
  printf("   Method count: %zu\n", program_ptr->num_methods());

  auto method_name_result = program_ptr->get_method_name(0);
  if (!method_name_result.ok()) {
    printf(
        "❌ Failed to get method name: error %d\n",
        (int)method_name_result.error());
    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }

  printf("   Method 0 name: %s\n", *method_name_result);

  // Try to load the method - this is where operator errors usually happen
  printf("🔄 Loading method '%s'...\n", *method_name_result);
  auto method_result =
      program_ptr->load_method(*method_name_result, &memory_manager);

  if (!method_result.ok()) {
    printf("❌ Failed to load method: error %d\n", (int)method_result.error());

    // More detailed method loading errors
    switch (method_result.error()) {
      case Error::InvalidProgram:
        printf("   → Method has invalid program structure\n");
        break;
      case Error::InvalidState:
        printf("   → Method in invalid state\n");
        break;
      case Error::NotSupported:
        printf("   → Method uses unsupported operators\n");
        printf(
            "   → This usually means missing operators in selective build!\n");
        break;
      case Error::NotFound:
        printf("   → Method resource not found\n");
        break;
      case Error::MemoryAllocationFailed:
        printf("   → Not enough memory to load method\n");
        break;
      case Error::OperatorMissing:
        printf("   → Operator missing\n");
        break;
      default:
        printf("   → Unknown method error: %d\n", (int)method_result.error());
    }

    blink_indicator(INDICATOR_PIN_1, 10);
    return false;
  }

  method_ptr = std::make_unique<Method>(std::move(*method_result));
  printf("✅ Method '%s' loaded successfully\n", *method_name_result);
  return true;
}

bool run_inference(Method& method) {
  printf(
      "🔥 ExecuTorch MLP MNIST Demo (Neural network) on Pico2 (microcontroller) 🔥\n");

  // ASCII art for digit '0' (28x28)
  const char* ascii_digit_0[28] = {
      "                            ", "        ############        ",
      "      ##################    ", "    ######################  ",
      "   ######################## ", "  ####                ####  ",
      " ####                  #### ", " ####                  #### ",
      "####                    ####", "####                    ####",
      "####                    ####", "####                    ####",
      "####                    ####", "####                    ####",
      "####                    ####", "####                    ####",
      "####                    ####", "####                    ####",
      "####                    ####", "####                    ####",
      " ####                  #### ", " ####                  #### ",
      "  ####                ####  ", "   ######################## ",
      "    ######################  ", "      ##################    ",
      "        ############        ", "                            "};

  const char* ascii_digit_1[28] = {
      "            ####            ", "           #####            ",
      "          ######            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "            ####            ", "            ####            ",
      "        ############        ", "        ############        ",
      "        ############        ", "                            "};

  const char* ascii_digit_4[28] = {
      "                            ", "               ####         ",
      "              #####         ", "             ######         ",
      "            #######         ", "           #### ####        ",
      "          ####  ####        ", "         ####   ####        ",
      "        ####    ####        ", "       ####     ####        ",
      "      ####      ####        ", "     ####       ####        ",
      "    ####        ####        ", "   ####         ####        ",
      "  ######################    ", "  ######################    ",
      "  ######################    ", "                ####        ",
      "                ####        ", "                ####        ",
      "                ####        ", "                ####        ",
      "                ####        ", "                ####        ",
      "                ####        ", "                ####        ",
      "                ####        ", "                            "};

  const char* ascii_digit_7[28] = {
      "############################", "############################",
      "                        ####", "                       #### ",
      "                      ####  ", "                     ####   ",
      "                    ####    ", "                   ####     ",
      "                  ####      ", "                 ####       ",
      "                ####        ", "               ####         ",
      "              ####          ", "             ####           ",
      "            ####            ", "           ####             ",
      "          ####              ", "         ####               ",
      "        ####                ", "       ####                 ",
      "      ####                  ", "     ####                   ",
      "    ####                    ", "   ####                     ",
      "  ####                      ", " ####                       ",
      "####                        ", "###                         "};

  // Test patterns
  struct TestCase {
    const char** pattern;
    const char* name;
    int expected_digit;
  };

  TestCase test_cases[] = {
      {ascii_digit_0, "Digit 0", 0},
      {ascii_digit_1, "Digit 1", 1},
      {ascii_digit_4, "Digit 4", 4},
      {ascii_digit_7, "Digit 7", 7}};

  printf("🧪 Testing all supported digits:\n\n");

  for (int test = 0; test < 4; test++) {
    const char** ascii_digit = test_cases[test].pattern;
    const char* digit_name = test_cases[test].name;
    int expected = test_cases[test].expected_digit;

    // Display the ASCII digit
    printf("=== %s ===\n", digit_name);
    for (int i = 0; i < 28; i++) {
      printf("%s\n", ascii_digit[i]);
    }
    printf("\n");

    // Convert ASCII to 28x28 float tensor
    float input_data[784]; // 28*28 = 784
    for (int row = 0; row < 28; row++) {
      for (int col = 0; col < 28; col++) {
        char pixel = ascii_digit[row][col];
        input_data[row * 28 + col] = (pixel == '#') ? 1.0f : 0.0f;
      }
    }

    // Count white pixels
    int white_pixels = 0;
    for (int i = 0; i < 784; i++) {
      if (input_data[i] > 0.5f)
        white_pixels++;
    }
    printf("Input stats: %d white pixels out of 784 total\n", white_pixels);

    // Create input tensor: [1, 28, 28]
    TensorImpl::SizesType input_sizes[3] = {1, 28, 28};
    TensorImpl::DimOrderType dim_order[3] = {0, 1, 2};

    TensorImpl input_impl(
        ScalarType::Float,
        3, // 3 dimensions: [batch, height, width]
        input_sizes, // [1, 28, 28]
        input_data,
        dim_order);
    Tensor input(&input_impl);

    // Set input and run inference
    printf("Running neural network inference...\n");

    auto result = method.set_input(input, 0);
    if (result != Error::Ok) {
      printf("❌ Failed to set input: error %d\n", (int)result);
      return false;
    }

    result = method.execute();
    if (result != Error::Ok) {
      printf("❌ Failed to execute: error %d\n", (int)result);
      return false;
    }

    auto output_evalue = method.get_output(0);
    if (!output_evalue.isTensor()) {
      printf("❌ Output is not a tensor\n");
      return false;
    }

    // Extract tensor from EValue
    Tensor output = output_evalue.toTensor();
    float* output_data = output.mutable_data_ptr<float>();

    // Find digit with highest score
    int predicted_digit = 0;
    float max_score = output_data[0];
    for (int i = 1; i < 10; i++) {
      if (output_data[i] > max_score) {
        max_score = output_data[i];
        predicted_digit = i;
      }
    }

    // Display results
    printf("✅ Neural network results:\n");
    for (int i = 0; i < 10; i++) {
      printf("  Digit %d: %.3f", i, output_data[i]);
      if (i == predicted_digit)
        printf(" ← PREDICTED");
      printf("\n");
    }

    // Check if correct
    printf("\n🎯 PREDICTED: %d (Expected: %d) ", predicted_digit, expected);
    if (predicted_digit == expected) {
      printf("✅ CORRECT!\n");
    } else {
      printf("❌ WRONG!\n");
    }

    printf("\n==================================================\n\n");
  }

  printf(
      "🎉 All tests complete! ExecuTorch inference of neural network works on Pico2!\n");
  return true;
}

int executor_runner() {
  init_gpio_pins();
  stdio_init_all();
  sleep_ms(1000);

  wait_for_usb();
  runtime_init();

  // Fit within Pico2's 520KB SRAM limit
  static uint8_t
      method_allocator_pool[200 * 1024]; // 200KB - plenty for method metadata
  static uint8_t activation_pool[200 * 1024]; // 200KB - plenty for activations
  // Total: 400KB directly allocated to ExecuTorch, leaves 120KB for other uses

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
