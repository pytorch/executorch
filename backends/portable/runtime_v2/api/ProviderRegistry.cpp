/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/portable/runtime_v2/api/ProviderRegistry.h>

namespace executorch {
namespace backends {
namespace portable_v2 {

ProviderRegistry::ProviderRegistry(
    std::vector<std::unique_ptr<Provider>> providers)
    : owned_(std::move(providers)) {
  all_.reserve(owned_.size());
  available_.reserve(owned_.size());

  for (RuntimeId id = 0; id < owned_.size(); ++id) {
    Provider* p = owned_[id].get();
    // Stamp the RuntimeId on the Provider via friend access.
    p->id_ = id;
    all_.push_back(p);
    if (p->is_available_on_device()) {
      available_.push_back(p);
    }
  }
}

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
