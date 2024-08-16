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
exec_aten::ArrayRef<exec_aten::optional<exec_aten::Tensor>>
BoxedEvalueList<exec_aten::optional<exec_aten::Tensor>>::get() const {
  for (typename exec_aten::ArrayRef<
           exec_aten::optional<exec_aten::Tensor>>::size_type i = 0;
       i < wrapped_vals_.size();
       i++) {
    if (wrapped_vals_[i] == nullptr) {
      unwrapped_vals_[i] = exec_aten::nullopt;
    } else {
      unwrapped_vals_[i] =
          wrapped_vals_[i]->to<exec_aten::optional<exec_aten::Tensor>>();
    }
  }
  return exec_aten::ArrayRef<exec_aten::optional<exec_aten::Tensor>>{
      unwrapped_vals_, wrapped_vals_.size()};
}
} // namespace runtime
} // namespace executorch
