// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <executorch/backends/native/runtime/deserialize/ZipReader.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// A recorded tensor, copied out of the package as contiguous little-endian
// bytes.
struct SampleTensor {
  ScalarType dtype = kFloat;
  std::vector<int64_t> sizes;
  std::vector<uint8_t> bytes;
};

struct Tolerance {
  double atol = 0.0;
  double rtol = 0.0;
};

struct SampleCall {
  std::string method;
  std::vector<SampleTensor> inputs;
  // Absent when the producer recorded no outputs.
  std::optional<std::vector<SampleTensor>> outputs;
  std::map<std::string, SampleTensor> intermediates;
  // The call's tolerance, else its sample's, else its set's.
  std::optional<Tolerance> tol;
};

// Calls run in order from the program's initial state, with no reset between.
struct Sample {
  std::vector<SampleCall> calls;
};

struct SampleSet {
  std::string description;
  std::string source;
  std::vector<Sample> samples;
};

// Reads samples/<set>/, specified in backends/native/README.md. Every in and
// out value must be a tensor; scalar and list values are not supported. Throws
// std::runtime_error if the set is absent or violates the format.
SampleSet read_sample_set(const ZipReader& zip, const std::string& set);

} // namespace ptn
