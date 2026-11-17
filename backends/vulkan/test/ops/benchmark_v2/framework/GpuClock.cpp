// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/GpuClock.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

namespace vkcompute::shader_profiling {

GpuClock::GpuClock(const std::string& sysfs_root) {
  const std::filesystem::path root(sysfs_root);
  std::vector<std::pair<std::filesystem::path, double>> sources{
      {root / "class/kgsl/kgsl-3d0/gpuclk", 1e6}};
  std::error_code error;
  auto devices =
      std::filesystem::directory_iterator(root / "devices/platform", error);
  for (; !error && devices != std::filesystem::directory_iterator();
       devices.increment(error)) {
    const auto path = devices->path();
    if (path.extension() == ".mali" || path.filename() == "mali") {
      // Pixel Mali cur_freq uses kHz; legacy Mali clock uses MHz.
      sources.emplace_back(path / "cur_freq", 1000);
      sources.emplace_back(path / "clock", 1);
    }
  }
  std::optional<double> kernel_units;
  error.clear();
  devices = std::filesystem::directory_iterator(root / "class/devfreq", error);
  for (; !error && devices != std::filesystem::directory_iterator();
       devices.increment(error)) {
    const auto path = devices->path();
    const auto name = path.filename().string();
    const auto extension = path.extension();
    const bool kgsl = name.find("kgsl-3d0") != std::string::npos;
    const bool sgpu = extension == ".sgpu" || name == "sgpu";
    if (kgsl || sgpu || extension == ".gpu" || extension == ".mali" ||
        name == "gpu" || name == "mali" || name == "pvrsrvkm") {
      // Samsung SGPU devfreq uses kHz; standard devfreq and KGSL use Hz.
      sources.emplace_back(path / "cur_freq", sgpu ? 1000 : 1e6);
      if (kgsl)
        kernel_units = 1;
      else if (sgpu)
        kernel_units = 1000;
    }
  }
  if (kernel_units) {
    sources.emplace_back(root / "kernel/gpu/gpu_clock", *kernel_units);
  }
  for (const auto& source : sources) {
    path_ = source.first.string();
    units_per_mhz_ = source.second;
    if (read_mhz())
      return;
  }
  path_.clear();
}

std::optional<double> GpuClock::read_mhz() const {
  if (path_.empty())
    return std::nullopt;
  std::ifstream input(path_);
  double frequency;
  if (!(input >> frequency) || !std::isfinite(frequency) || frequency < 0)
    return std::nullopt;
  input >> std::ws;
  if (!input.eof())
    return std::nullopt;
  return frequency / units_per_mhz_;
}

} // namespace vkcompute::shader_profiling
