// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/Value.h>

#include <stdexcept>

namespace ptn {

const TensorMeta& Value::tensor_meta() const {
  const TensorMeta* m = std::get_if<TensorMeta>(&value_);
  if (m == nullptr) {
    throw std::runtime_error("Value::tensor_meta: value is not a Tensor");
  }
  return *m;
}

const Scalar& Value::scalar() const {
  const Scalar* s = std::get_if<Scalar>(&value_);
  if (s == nullptr) {
    throw std::runtime_error("Value::scalar: value is not a Scalar");
  }
  return *s;
}

const std::vector<ValueRef>& Value::content_refs() const {
  const std::vector<ValueRef>* refs =
      std::get_if<std::vector<ValueRef>>(&value_);
  if (refs == nullptr) {
    throw std::runtime_error("Value::content_refs: value is not a List");
  }
  return *refs;
}

} // namespace ptn
