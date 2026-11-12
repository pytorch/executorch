/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#import <Foundation/Foundation.h>
#include <cerrno>
#include "coreai_fault_injection.h"

namespace executorch::backends::coreai::testing {

struct StorageFaultScope {
  FaultPoint operation;
  int error = EIO;
  int match_index = 1;
  int matches = 0;
  int hits = 0;
  bool armed = true;
  void (^observe)(FaultPoint) = nil;

  explicit StorageFaultScope(FaultPoint op)
      : operation(op), previous_(current_), previous_callback_(fault_callback) {
    current_ = this;
    fault_callback = inject;
  }
  ~StorageFaultScope() {
    fault_callback = previous_callback_;
    current_ = previous_;
  }
  StorageFaultScope(const StorageFaultScope&) = delete;
  StorageFaultScope& operator=(const StorageFaultScope&) = delete;

 private:
  inline static thread_local StorageFaultScope* current_ = nullptr;
  StorageFaultScope* previous_;
  FaultCallback previous_callback_;

  static int inject(FaultPoint operation) {
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
