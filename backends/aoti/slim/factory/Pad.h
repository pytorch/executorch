#pragma once

#include <executorch/backends/aoti/slim/factory/Empty.h>
#include <executorch/backends/aoti/slim/util/ArrayRefUtil.h>

namespace executorch::backends::aoti::slim {

inline SlimTensor constant_pad_nd(
    const SlimTensor& self,
    IntArrayRef pad,
    const executorch::backends::aoti::slim::c10::Scalar& value) {
  ET_CHECK_MSG(pad.size() % 2 == 0, "Length of pad must be even");

  IntArrayRef input_sizes = self.sizes();
  int64_t l_inp = self.dim();
  int64_t l_pad = static_cast<int64_t>(pad.size()) / 2;
  int64_t l_diff = l_inp - l_pad;

  ET_CHECK_MSG(
      l_pad <= l_inp,
      "Length of pad should be no more than twice the input's dimension.");

  bool all_pads_non_positive = true;
  SlimTensor c_input = self;
  for (int64_t i = l_diff; i < l_inp; i++) {
    int64_t pad_idx = 2 * (l_inp - i - 1);

    if (pad[pad_idx] < 0) {
      c_input =
          c_input.narrow(i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
    } else if (pad[pad_idx] != 0) {
      all_pads_non_positive = false;
    }
    if (pad[pad_idx + 1] < 0) {
      c_input = c_input.narrow(i, 0, c_input.size(i) + pad[pad_idx + 1]);
    } else if (pad[pad_idx + 1] != 0) {
      all_pads_non_positive = false;
    }
  }

  // if none of the pads are positive we can optimize and just return the result
  // of calling .narrow() on the input
  if (all_pads_non_positive) {
    return c_input.clone_contiguous();
  }

  // calculate the new shape for the output tensor
  std::vector<int64_t> new_shape;
  new_shape.reserve(l_diff);
  for (int64_t i = 0; i < l_diff; i++) {
    new_shape.emplace_back(input_sizes[i]);
  }

  for (const auto i :
       executorch::backends::aoti::slim::c10::irange((size_t)l_pad)) {
    auto pad_idx = pad.size() - ((i + 1) * 2);
    auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
    ET_CHECK_MSG(
        new_dim > 0,
        "The input size plus negative padding resulted in a negative output size. "
        "Check your input.");
    new_shape.emplace_back(new_dim);
  }

  SlimTensor output =
      empty(makeArrayRef(new_shape), self.dtype(), self.device());
  output.fill_(value);

  // create a view into the center of the output tensor
  SlimTensor c_output = output;
  for (const auto i :
       executorch::backends::aoti::slim::c10::irange(l_diff, l_inp)) {
    auto pad_idx = 2 * (l_inp - i - 1);
    if (pad[pad_idx] > 0) {
      c_output =
          c_output.narrow(i, pad[pad_idx], c_output.size(i) - pad[pad_idx]);
    }
    if (pad[pad_idx + 1] > 0) {
      c_output = c_output.narrow(i, 0, c_output.size(i) - pad[pad_idx + 1]);
    }
  }
  // copy the input data into the center view
  c_output.copy_(c_input);
  return output;
}

inline SlimTensor pad(
    const SlimTensor& self,
    IntArrayRef pad,
    std::string_view mode,
    std::optional<double> value) {
  if (mode == "constant") {
    return constant_pad_nd(self, pad, value.value_or(0.0));
  }
  ET_CHECK_MSG(
      false, "Unsupported padding mode. Only constant mode is available.");
}

} // namespace executorch::backends::aoti::slim
