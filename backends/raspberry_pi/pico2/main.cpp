#include "pico/stdio_usb.h"
#include "pico/stdlib.h"

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

// Declare your model data (from simple_addmodule_pte.c)
extern const uint8_t model_pte[] __attribute__((aligned(8)));
extern const unsigned int model_pte_len;

// Define GPIO pins for indicators
const uint INDICATOR_PIN_1 = 25; // Onboard LED
const uint INDICATOR_PIN_2 = 22; // External LED
const uint INDICATOR_PIN_3 = 23; // Onboard LED

static uint8_t method_allocator_pool[1024];
static uint8_t activation_pool[512];

void blink_indicator(uint pin, int times, int delay_ms = 100) {
  gpio_init(pin);
  gpio_set_dir(pin, GPIO_OUT);
  for (int i = 0; i < times; ++i) {
    gpio_put(pin, 1);
    sleep_ms(delay_ms);
    gpio_put(pin, 0);
    sleep_ms(delay_ms);
  }
}

int main() {
  using namespace executorch::extension;
  using namespace executorch::runtime;
  using executorch::aten::Tensor;
  using executorch::aten::TensorImpl;
  using ScalarType = executorch::runtime::etensor::ScalarType;
  using executorch::runtime::runtime_init;

  stdio_init_all();
 
  // Give host time to enumerate USB serial
  sleep_ms(1000);

  int retry_usb_count= 0;
  while (!stdio_usb_connected() && retry_usb_count++ < 10) {
    printf("Retry again! USB not connected \n");
    sleep_ms(100); // Check every 100 ms
  }
  runtime_init();

  executorch::extension::BufferDataLoader loader(model_pte, model_pte_len);
  MemoryAllocator method_allocator(sizeof(method_allocator_pool), method_allocator_pool);
  method_allocator.enable_profiling("method allocator");

  Span<uint8_t> memory_planned_buffers[1]{{activation_pool, sizeof(activation_pool)}};
  HierarchicalAllocator planned_memory({memory_planned_buffers, 1});

  MemoryManager memory_manager(&method_allocator, &planned_memory);
  auto program_result = Program::load(&loader);
  if (!program_result.ok()) {
    printf("Failed to load model: %d\n", (int)program_result.error());
    blink_indicator(INDICATOR_PIN_1, 10);
    return 1;
  }
  Program program = std::move(*program_result);

  const char* method_name = nullptr;
  // Get method name (usually "forward")
  auto method_name_result = program.get_method_name(0);
  if (!method_name_result.ok()) {
    printf("Failed to get method name: %d\n", (int)method_name_result.error());
    blink_indicator(INDICATOR_PIN_1, 10);
    return 1;
  }

  method_name = *method_name_result;
  // Load method
  auto method_result = program.load_method(method_name, &memory_manager);
  if (!method_result.ok()) {
    printf("Failed to load method: %d\n", (int)method_result.error());
    blink_indicator(INDICATOR_PIN_1, 10);
    return 1;
  }
  printf("Method loaded [%s]\n", method_name);

  auto method = std::move(*method_result);
  // Prepare input tensor
  float input_data_0[4] = {4.0, 109.0, 13.0, 123.0};
  float input_data_1[4] = {9.0, 27.0, 11.0, 8.0};
  TensorImpl::SizesType sizes[1] = {4};
  TensorImpl::DimOrderType dim_order[] = {0};
  TensorImpl impl0(ScalarType::Float, 1, sizes, input_data_0, dim_order);
  TensorImpl impl1(ScalarType::Float, 1, sizes, input_data_1, dim_order);
  Tensor input0(&impl0);
  Tensor input1(&impl1);

  // Set input
  auto set_input_error0 = method.set_input(input0, 0);
  auto set_input_error1 = method.set_input(input1, 1);
  if (set_input_error0 != Error::Ok || set_input_error1 != Error::Ok) {
    printf("Failed to set input(s)\n");
    blink_indicator(INDICATOR_PIN_1, 10);
    return 1;
  }

  // Run inference
  auto exec_error = method.execute();
  if (exec_error != Error::Ok) {
    printf("Failed to execute: %d\n", (int)exec_error);
    blink_indicator(INDICATOR_PIN_1, 10);
    return 1;
  }

  // Get output
  const EValue& output = method.get_output(0);
  if (output.isTensor()) {
    const auto& out_tensor = output.toTensor();
    const float* out_data = out_tensor.const_data_ptr<float>();
    printf("Output: %f, %f, %f, %f\n", out_data[0], out_data[1], out_data[2], out_data[3]);
  } else {
    printf("Output is not a tensor!\n");
    blink_indicator(INDICATOR_PIN_1, 10);
  }

  blink_indicator(INDICATOR_PIN_1, 100, 500); // Blink onboard LED 10 times (~10 secs) to indicate success
  return 0;
}
