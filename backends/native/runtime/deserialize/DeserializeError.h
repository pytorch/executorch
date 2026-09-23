// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>

namespace ptn {

class ResourceLimitError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class UnsupportedVersionError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

} // namespace ptn
