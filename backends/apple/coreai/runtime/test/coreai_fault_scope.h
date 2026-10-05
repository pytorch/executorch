/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cerrno>
#include "coreai_storage.h"

namespace executorch::backends::coreai::testing {

struct StorageFaultScope {
  StorageOperation operation;
  int error = EIO;
  int match_index = 1;
  int matches = 0;
  int hits = 0;
  bool armed = true;
  void (^observe)(StorageOperation) = nil;

  explicit StorageFaultScope(StorageOperation op)
      : operation(op),
        previous_(current_),
        previous_callback_(storage_fault_callback) {
    current_ = this;
    storage_fault_callback = inject;
  }
  ~StorageFaultScope() {
    storage_fault_callback = previous_callback_;
    current_ = previous_;
  }
  StorageFaultScope(const StorageFaultScope&) = delete;
  StorageFaultScope& operator=(const StorageFaultScope&) = delete;

 private:
  inline static thread_local StorageFaultScope* current_ = nullptr;
  StorageFaultScope* previous_;
  StorageFaultCallback previous_callback_;

  static int inject(StorageOperation operation) {
    auto& scope = *current_;
    if (scope.observe != nil)
      scope.observe(operation);
    if (scope.armed && scope.operation == operation &&
        ++scope.matches == scope.match_index) {
      ++scope.hits;
      return scope.error;
    }
    return 0;
  }
};

} // namespace executorch::backends::coreai::testing
