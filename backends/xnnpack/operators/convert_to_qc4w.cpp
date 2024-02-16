/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor convert_to_qc4w(at::Tensor x) {
  std::vector<int64_t> sizes = x.sizes().vec();
  TORCH_CHECK(sizes.size() == 2, "Expecting 2D tensor");
  TORCH_CHECK(sizes[1] % 2 == 0);
  TORCH_CHECK(
      x.options().dtype() == at::kByte, "Input tensor must be of type uint8.");
  sizes[1] = sizes[1] / 2;
  at::Tensor output = at::empty(sizes, x.options().dtype());
  uint8_t* x_ptr = x.data_ptr<uint8_t>();
  uint8_t* output_ptr = output.data_ptr<uint8_t>();
  for (int i = 0; i < output.numel(); ++i) {
    int32_t input_i = i * 2;
    int32_t input_i_plus_1 = i * 2 + 1;
    output_ptr[i] = (x_ptr[input_i_plus_1] << 4) | (x_ptr[input_i]);
  }
  return output;
}

TORCH_LIBRARY_FRAGMENT(xnnpack, m) {
  m.def("convert_to_qc4w", &convert_to_qc4w);
}
