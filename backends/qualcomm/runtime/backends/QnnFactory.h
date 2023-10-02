/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <exception>
namespace torch {
namespace executor {
namespace qnn {
class QnnFactory {
 protected:
  class NotImplementedException : public std::exception {
   public:
    const char* what() const noexcept override {
      return "Backend not yet implemented";
    }
  };
};
} // namespace qnn
} // namespace executor
} // namespace torch
