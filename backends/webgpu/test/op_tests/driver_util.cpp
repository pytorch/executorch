/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/test/op_tests/driver_util.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>

#include <nlohmann/json.hpp>

namespace executorch::backends::webgpu {

namespace {
std::string dir_of(const std::string& p) {
  const auto pos = p.find_last_of('/');
  return pos == std::string::npos ? std::string(".") : p.substr(0, pos);
}
std::string join(const std::string& a, const std::string& b) {
  return a + "/" + b;
}
} // namespace

std::vector<ManifestEntry> parse_manifest(const std::string& manifest_path) {
  std::ifstream in(manifest_path);
  if (!in.good()) {
    std::fprintf(
        stderr, "ERROR: cannot open manifest: %s\n", manifest_path.c_str());
    return {};
  }
  nlohmann::json j;
  in >> j;
  const std::string base = dir_of(manifest_path);

  std::vector<ManifestEntry> out;
  for (const auto& e : j) {
    ManifestEntry m;
    m.op = e.at("op").get<std::string>();
    m.name = e.at("case").get<std::string>();
    m.pte = join(base, e.at("pte").get<std::string>());
    for (const auto& ie : e.at("inputs")) {
      InputRef ir;
      ir.path = join(base, ie.at("path").get<std::string>());
      ir.shape = ie.at("shape").get<std::vector<int>>();
      m.inputs.push_back(std::move(ir));
    }
    const auto& g = e.at("golden");
    m.golden.path = join(base, g.at("path").get<std::string>());
    m.golden.shape = g.at("shape").get<std::vector<int>>();
    m.golden.output_index = g.value("output_index", 0);
    m.atol = e.value("atol", 1e-3f);
    m.rtol = e.value("rtol", 1e-3f);
    m.required = e.value("required", true);
    m.heavy = e.value("heavy", false);
    out.push_back(std::move(m));
  }
  return out;
}

std::vector<float> load_fp32_bin(const std::string& path, size_t numel) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return {};
  }
  std::vector<float> g(numel);
  const size_t n = std::fread(g.data(), sizeof(float), numel, f);
  std::fclose(f);
  if (n != numel) {
    return {};
  }
  return g;
}

bool within_tol(
    const float* out,
    const float* golden,
    int n,
    float atol,
    float rtol,
    float* max_abs,
    float* max_rel) {
  float ma = 0.0f, mr = 0.0f;
  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (std::isnan(out[i]) || std::isnan(golden[i])) {
      ok = false; // NaN never passes a tolerance check
      continue;
    }
    const float ae = std::abs(out[i] - golden[i]);
    const float re = ae / std::max(std::abs(golden[i]), 1e-6f);
    ma = std::max(ma, ae);
    mr = std::max(mr, re);
    if (ae > atol && re > rtol) {
      ok = false;
    }
  }
  *max_abs = ma;
  *max_rel = mr;
  return ok;
}

size_t numel(const std::vector<int>& shape) {
  size_t n = 1;
  for (int d : shape) {
    if (d <= 0) {
      return 0; // empty or malformed dim
    }
    const auto dd = static_cast<size_t>(d);
    if (n > SIZE_MAX / dd) {
      return 0; // overflow: malformed shape
    }
    n *= dd;
  }
  return n;
}

} // namespace executorch::backends::webgpu
