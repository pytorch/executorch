/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

struct InputRef {
  std::string path;
  std::vector<int> shape;
  std::string dtype = "float32";
};

struct GoldenRef {
  std::string path;
  std::vector<int> shape;
  int output_index = 0;
};

struct ManifestEntry {
  std::string op;
  std::string name;
  std::string pte;
  std::vector<InputRef> inputs;
  GoldenRef golden;
  float atol = 1e-3f;
  float rtol = 1e-3f;
  bool required = true; // a missing .pte is a FAIL, not a silent skip
  bool heavy = false; // export-gated behind WEBGPU_TEST_HEAVY (never required)
};

/// Parse manifest.json; resolve every relative path against the manifest's dir.
std::vector<ManifestEntry> parse_manifest(const std::string& manifest_path);

/// Load raw little-endian fp32; empty on size/IO mismatch.
std::vector<float> load_fp32_bin(const std::string& path, size_t numel);
std::vector<int32_t> load_int32_bin(const std::string& path, size_t numel);

/// Element OK if abs_err <= atol OR rel_err <= rtol (rel floored at
/// |golden|=1e-6). Sets the reported maxima; true iff all elements pass.
bool within_tol(
    const float* out,
    const float* golden,
    int n,
    float atol,
    float rtol,
    float* max_abs,
    float* max_rel);

size_t numel(const std::vector<int>& shape);

} // namespace executorch::backends::webgpu
