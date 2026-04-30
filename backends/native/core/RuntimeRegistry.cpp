/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/core/RuntimeRegistry.h>

namespace executorch {
namespace backends {
namespace native {

RuntimeRegistry::RuntimeRegistry(
    std::vector<std::unique_ptr<Runtime>> providers)
    : owned_(std::move(providers)) {
  all_.reserve(owned_.size());
  available_.reserve(owned_.size());

  for (RuntimeId id = 0; id < owned_.size(); ++id) {
    Runtime* p = owned_[id].get();
    // Stamp the RuntimeId on the Runtime via friend access.
    p->id_ = id;
    all_.push_back(p);
    if (p->is_available_on_device()) {
      available_.push_back(p);
    }
  }
}

} // namespace native
} // namespace backends
} // namespace executorch
