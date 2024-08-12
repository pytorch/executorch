// The module takes in a string as input and emits a string as output.

#pragma once
#include <cstdint>
#include <vector>

namespace torch::executor {

struct Image {
  // Assuming NCHW format
  std::vector<uint8_t> data;
  int32_t width;
  int32_t height;
  int32_t channels;
};

} // namespace torch::executor