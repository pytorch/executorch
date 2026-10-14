/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::cuda::batching {

// One exported forward method a slice of a batch can run through.
struct ForwardMethod {
  std::string name;
  // The most tokens the method takes. A static method takes exactly this
  // many, so a narrower slice is padded up to it; the dynamic method takes the
  // slice as it is.
  int max_tokens;
  bool is_static;
};

struct StepSlice {
  int offset;
  // The slice's real tokens.
  int length;
  // Index into the method table.
  int method;
  // The tokens the method runs: max_tokens for a static method, whose rows
  // past `length` are padding, and `length` for the dynamic one.
  int width;
};

// Cuts a batch of `total` packed tokens into forwards the exported methods can
// run, in order, so a slice attends every cell its predecessors wrote.
//
// `methods` must be sorted by max_tokens, ascending. Slices take up to the
// widest method's max_tokens each, and each runs through the method with the
// smallest max_tokens that still holds it. The dynamic method is only ever
// reached for slices wider than every static method, so it must be exported
// from (at most) one token past the widest static method. The widest method
// must take at least one token, or no slice would advance through the batch.
inline std::vector<StepSlice> plan_slices(
    int total,
    const std::vector<ForwardMethod>& methods) {
  std::vector<StepSlice> slices;
  if (methods.empty()) {
    return slices;
  }
  const int widest = methods.back().max_tokens;
  ET_CHECK_MSG(
      widest > 0,
      "plan_slices: the widest method must take at least one token, got %d",
      widest);
  for (int offset = 0; offset < total;) {
    const int length = std::min(widest, total - offset);
    int index = 0;
    while (methods[index].max_tokens < length) {
      ++index;
    }
    const ForwardMethod& method = methods[index];
    slices.push_back(
        {offset, length, index, method.is_static ? method.max_tokens : length});
    offset += length;
  }
  return slices;
}

} // namespace executorch::backends::cuda::batching
