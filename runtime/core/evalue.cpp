/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/evalue.h>

namespace executorch {
namespace runtime {
template <>
executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>
BoxedEvalueList<std::optional<executorch::aten::Tensor>>::get() const {
  for (typename executorch::aten::ArrayRef<
           std::optional<executorch::aten::Tensor>>::size_type i = 0;
       i < wrapped_vals_.size();
       i++) {
    if (wrapped_vals_[i] == nullptr) {
      unwrapped_vals_[i] = executorch::aten::nullopt;
    } else {
      unwrapped_vals_[i] =
          wrapped_vals_[i]->to<std::optional<executorch::aten::Tensor>>();
    }
  }
  return executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>{
      unwrapped_vals_, wrapped_vals_.size()};
}

// Specialization note: unlike the generic tryGet, a null wrapped_vals_[i]
// here is a valid encoding of None (matching the get() specialization above,
// which mirrors parseListOptionalType's "absent index" convention). Only an
// element whose tag is neither None nor Tensor is treated as an error.
template <>
Result<executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>>
BoxedEvalueList<std::optional<executorch::aten::Tensor>>::tryGet() const {
  for (typename executorch::aten::ArrayRef<
           std::optional<executorch::aten::Tensor>>::size_type i = 0;
       i < wrapped_vals_.size();
       i++) {
    if (wrapped_vals_[i] == nullptr) {
      unwrapped_vals_[i] = std::nullopt;
      continue;
    }
    auto r = wrapped_vals_[i]->tryToOptional<executorch::aten::Tensor>();
    if (!r.ok()) {
      return r.error();
    }
    unwrapped_vals_[i] = std::move(r.get());
  }
  return executorch::aten::ArrayRef<std::optional<executorch::aten::Tensor>>{
      unwrapped_vals_, wrapped_vals_.size()};
}
} // namespace runtime
} // namespace executorch
