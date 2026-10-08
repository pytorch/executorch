// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "test_case.h"

#include <sstream>

namespace executorch {
namespace vulkan {
namespace prototyping {

// ReferenceKey implementation
ReferenceKey ReferenceKey::from_test_case(const TestCase& tc) {
  std::ostringstream oss;

  // Serialize inputs that affect reference computation
  // Skip: storage_type, memory_layout, string values (like impl_selector)
  for (size_t i = 0; i < tc.inputs().size(); ++i) {
    const ValueSpec& input = tc.inputs()[i];
    oss << "i" << i << ":";

    if (input.is_tensor()) {
      // For tensors: sizes, dtype, data_gen_type, is_constant
      oss << "T[";
      for (size_t j = 0; j < input.sizes.size(); ++j) {
        if (j > 0) {
          oss << ",";
        }
        oss << input.sizes[j];
      }
      oss << "]d" << static_cast<int>(input.dtype);
      oss << "g" << static_cast<int>(input.data_gen_type);
      oss << "c" << (input.is_constant() ? 1 : 0);
      oss << "n" << (input.is_none() ? 1 : 0);
    } else if (input.is_int()) {
      oss << "I" << input.get_int_value();
    } else if (input.is_float()) {
      oss << "F" << input.get_float_value();
    } else if (input.is_bool()) {
      oss << "B" << (input.get_bool_value() ? 1 : 0);
    } else if (input.is_int_list()) {
      oss << "L[";
      const auto& list = input.get_int_list();
      for (size_t j = 0; j < list.size(); ++j) {
        if (j > 0) {
          oss << ",";
        }
        oss << list[j];
      }
      oss << "]";
    }
    // Skip string inputs (like impl_selector) as they don't affect reference
    oss << ";";
  }

  // Also include output shapes for completeness
  for (size_t i = 0; i < tc.outputs().size(); ++i) {
    const ValueSpec& output = tc.outputs()[i];
    oss << "o" << i << ":[";
    for (size_t j = 0; j < output.sizes.size(); ++j) {
      if (j > 0) {
        oss << ",";
      }
      oss << output.sizes[j];
    }
    oss << "]d" << static_cast<int>(output.dtype) << ";";
  }

  ReferenceKey key;
  key.key_string = oss.str();
  return key;
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
