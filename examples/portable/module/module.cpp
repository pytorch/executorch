#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <iostream>
using namespace ::executorch::extension;

int main(int argc, char** argv) {
  // Create a Module.
  Module module("/data/users/lfq/executorch/mobilenet_v3_small.pte");

  // Wrap the input data with a Tensor.
  float input[1 * 3 * 224 * 224] = {4.f};
  auto tensor = from_blob(input, {1, 3, 224, 224});

  auto err = module.set_inputs({tensor});

  // Perform an inference.
  const auto result = module.forward();

  // Check for success or failure.
  if (result.ok()) {
    // Retrieve the output data.
    const auto output = result->at(0).toTensor().const_data_ptr<float>();
    std::cout << "Output: " << output[0] << std::endl;
  }
  return 0;
}
