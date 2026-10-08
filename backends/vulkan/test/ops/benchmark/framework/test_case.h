// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "value_spec.h"

#include <functional>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

//
// ReferenceKey for caching reference computations
//

// Captures the identity of input conditions for test case grouping.
// Test cases with the same ReferenceKey should produce identical reference
// outputs, so reference computation can be cached and reused.
struct ReferenceKey {
  std::string key_string;

  static ReferenceKey from_test_case(const class TestCase& tc);

  bool operator==(const ReferenceKey& other) const {
    return key_string == other.key_string;
  }
};

struct ReferenceKeyHash {
  size_t operator()(const ReferenceKey& k) const {
    return std::hash<std::string>{}(k.key_string);
  }
};

//
// Shader filter defaults
//

// Default shader filter: excludes layout conversions and quantization overhead
// This is used for tests where quantize/dequantize are overhead operations
inline const std::vector<std::string> kDefaultShaderFilter = {
    "nchw_to",
    "to_nchw",
    "quantize_and_pack_4w4c",
    "unpack_4w4c_and_dequantize"};

// Layout-only filter: only excludes layout conversions
// Use this for tests where quantize/dequantize ARE the operations being tested
inline const std::vector<std::string> kLayoutOnlyShaderFilter = {
    "nchw_to",
    "to_nchw"};

//
// TestCase
//

// Default per-execute() wall-clock target used by the probe-then-scale logic
// in execute_test_cases(). Picked generously enough that even an under-sized
// chained_dispatches factor (the probe runs at governor-pinned clock and
// underestimates the boost-clock latency) still drives sustained GPU
// activity during measurement.
constexpr int kDefaultTargetExecuteTimeUs = 100000;

class TestCase {
 public:
  TestCase()
      : abs_tolerance_(2e-3f),
        rel_tolerance_(1e-3f),
        shader_filter_(kDefaultShaderFilter) {}
  explicit TestCase(const std::string& name)
      : name_(name),
        abs_tolerance_(2e-3f),
        rel_tolerance_(1e-3f),
        shader_filter_(kDefaultShaderFilter) {}

  void set_name(const std::string& name) {
    name_ = name;
  }
  const std::string& name() const {
    return name_;
  }

  void set_operator_name(const std::string& op_name) {
    operator_name_ = op_name;
  }
  const std::string& operator_name() const {
    return operator_name_;
  }

  // Tolerance settings
  void set_abs_tolerance(float abs_tolerance) {
    abs_tolerance_ = abs_tolerance;
  }
  float get_abs_tolerance() const {
    return abs_tolerance_;
  }

  void set_rel_tolerance(float rel_tolerance) {
    rel_tolerance_ = rel_tolerance;
  }
  float get_rel_tolerance() const {
    return rel_tolerance_;
  }

  // Shader filter settings - list of shader name patterns to exclude from
  // timing
  void set_shader_filter(const std::vector<std::string>& filter) {
    shader_filter_ = filter;
  }
  const std::vector<std::string>& get_shader_filter() const {
    return shader_filter_;
  }

  // Manual override for the number of chained dispatches per measurement
  // iteration (a.k.a. chained_dispatches). If > 0, the framework uses this
  // directly and skips the probe phase. 0 (the default) means adaptive
  // (probe-then-scale).
  void set_op_invocations_per_execute(int n) {
    op_invocations_per_execute_ = n;
  }
  int get_op_invocations_per_execute() const {
    return op_invocations_per_execute_;
  }

  // Target single-execute duration in microseconds. Used only when the
  // manual chained_dispatches override is not set. Default
  // kDefaultTargetExecuteTimeUs, picked generously enough to mitigate Adreno
  // DCVS governor pinning during the probe.
  void set_target_execute_time_us(int us) {
    target_execute_time_us_ = us;
  }
  int get_target_execute_time_us() const {
    return target_execute_time_us_;
  }

  // When true, the ComputeGraph built for this test case sets
  // GraphConfig::force_resize, so every DynamicDispatchNode runs its resize
  // function once during measurement setup (execute_test_case runs
  // propagate_resize() after prepack) even when no input shape changed.
  // Because the output is already allocated at the swept shape, the resize
  // must recompute the same shape from the current input — a wrong resize
  // formula resizes the output to a mismatched shape and surfaces as a test
  // failure. Default true (opt-out): every custom_ops test exercises its
  // resize formulas across the swept shapes. Call set_force_resize(false) for
  // the rare op whose resize fn is intentionally not shape-preserving under a
  // fixed output allocation.
  void set_force_resize(bool force_resize) {
    force_resize_ = force_resize;
  }
  bool get_force_resize() const {
    return force_resize_;
  }

  void add_input_spec(const ValueSpec& spec) {
    inputs_.push_back(spec);
  }

  const std::vector<ValueSpec>& inputs() const {
    return inputs_;
  }

  std::vector<ValueSpec>& inputs() {
    return inputs_;
  }

  size_t num_inputs() const {
    return inputs_.size();
  }

  void add_output_spec(const ValueSpec& spec) {
    outputs_.push_back(spec);
  }

  const std::vector<ValueSpec>& outputs() const {
    return outputs_;
  }

  std::vector<ValueSpec>& outputs() {
    return outputs_;
  }

  size_t num_outputs() const {
    return outputs_.size();
  }

  bool empty() const {
    return inputs_.empty() && outputs_.empty();
  }
  void clear() {
    inputs_.clear();
    outputs_.clear();
    name_.clear();
    operator_name_.clear();
    abs_tolerance_ = 2e-3f;
    rel_tolerance_ = 1e-3f;
    shader_filter_ = kDefaultShaderFilter;
    op_invocations_per_execute_ = 0;
    target_execute_time_us_ = kDefaultTargetExecuteTimeUs;
    force_resize_ = true;
  }

 private:
  std::string name_;
  std::string operator_name_;
  std::vector<ValueSpec> inputs_;
  std::vector<ValueSpec> outputs_;
  float abs_tolerance_;
  float rel_tolerance_;
  std::vector<std::string> shader_filter_;
  int op_invocations_per_execute_ = 0; // 0 = adaptive
  int target_execute_time_us_ = kDefaultTargetExecuteTimeUs;
  bool force_resize_ = true;
};

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
