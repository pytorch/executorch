// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/Scalar.h>

#include <stdexcept>

namespace ptn {

int64_t Scalar::to_int() const {
  const int64_t* v = std::get_if<int64_t>(&value_);
  if (v == nullptr) {
    throw std::runtime_error("Scalar::to_int: scalar is not an Int");
  }
  return *v;
}

double Scalar::to_double() const {
  const double* v = std::get_if<double>(&value_);
  if (v == nullptr) {
    throw std::runtime_error("Scalar::to_double: scalar is not a Double");
  }
  return *v;
}

bool Scalar::to_bool() const {
  const bool* v = std::get_if<bool>(&value_);
  if (v == nullptr) {
    throw std::runtime_error("Scalar::to_bool: scalar is not a Bool");
  }
  return *v;
}

} // namespace ptn
