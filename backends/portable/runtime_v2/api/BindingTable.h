/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Buffer.h>

#include <cstdint>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Side table mapping value_id -> Buffer* for tensor storage backings only.
 * Non-tensor EValues (scalars, lists) live in LoadedDelegate::values; not
 * in this table. See §4.3 of PORTABLE_BACKEND_API_PROPOSAL.md.
 *
 * The TensorImpl::data_ptr in the corresponding EValue is a denormalized
 * cache of the bound Buffer's host_ptr(); kept consistent at bind time.
 *
 * Bindings are populated once at init (prebind_owned_buffers) and live
 * for the LoadedDelegate's lifetime. Per-execute IO does NOT change
 * bindings — upload_from_host re-aliases the bound Buffer in place.
 */
class BindingTable {
 public:
  // Returns the storage Buffer for this value's tensor data.
  // Returns nullptr if value_id is not a tensor (e.g. a scalar) or hasn't
  // been bound yet.
  Buffer* get(uint32_t value_id) const {
    auto it = map_.find(value_id);
    return it != map_.end() ? it->second : nullptr;
  }

  void bind(uint32_t value_id, Buffer* buf) { map_[value_id] = buf; }

  // Used during init for diagnostics / iteration.
  size_t size() const { return map_.size(); }

 private:
  std::unordered_map<uint32_t, Buffer*> map_;
};

using BindingView = const BindingTable&;

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
