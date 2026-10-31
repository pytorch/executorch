// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <string>

namespace vkcompute::shader_profiling {

class GpuClock final {
  std::string path_;
  double units_per_mhz_ = 1;

 public:
  explicit GpuClock(const std::string& sysfs_root = "/sys");
  std::optional<double> read_mhz() const;
};

} // namespace vkcompute::shader_profiling
