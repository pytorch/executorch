// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/utils/Print.h>

#include <cstddef>
#include <string>

namespace ptn {

std::string to_string(const TensorMeta& meta) {
  std::string s = scalar_type_name(meta.dtype);
  s += "[";
  for (size_t i = 0; i < meta.sizes.size(); ++i) {
    if (i != 0) {
      s += ",";
    }
    s += std::to_string(meta.sizes[i]);
  }
  s += "]";
  return s;
}

} // namespace ptn
