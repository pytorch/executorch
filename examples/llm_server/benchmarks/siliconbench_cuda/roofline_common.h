/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cuda_runtime_api.h>
#ifdef ROOFLINE_PROFILE
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace roofline {
constexpr int kHistory = 1024;
constexpr int kOwners = 32;
constexpr int kCapacity = 2048;
constexpr int kBatch = 1024;

struct Mix {
  const char* name;
  int decode;
  int prefill;
};
inline const std::vector<Mix> kMixes = {
    {"d1", 1, 0}, {"d8", 8, 0}, {"d32", 32, 0}, {"p64", 0, 64},
    {"p512", 0, 512}, {"d8p64", 8, 64}, {"d8p512", 8, 512}};

inline void require(bool value, const std::string& message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}

inline void sync() {
  auto status = cudaDeviceSynchronize();
  require(status == cudaSuccess, cudaGetErrorString(status));
}

// These are deterministic synthetic, valid MG token IDs. No shared prefix is
// reused between owners. Both drivers use this exact input definition.
inline std::int64_t token(int owner, int position) {
  return 256 + (static_cast<std::uint64_t>(owner + 1) * 7919 +
                static_cast<std::uint64_t>(position + 1) * 104729) % 190000;
}

struct Row {
  std::int64_t token;
  int position;
  int owner;
  bool output;
};

inline std::vector<Row> rows(const Mix& mix) {
  std::vector<Row> result;
  for (int owner = 0; owner < mix.decode; ++owner) {
    result.push_back({token(owner, kHistory), kHistory, owner, true});
  }
  for (int p = 0; p < mix.prefill; ++p) {
    result.push_back({token(mix.decode, kHistory + p),
                      kHistory + p, mix.decode, false});
  }
  return result;
}

inline void print_tokens(std::ostream& out, const std::vector<std::int64_t>& ids) {
  out << '[';
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i) out << ',';
    out << ids[i];
  }
  out << ']';
}

// Rewind and its synchronization are outside the timed interval. Every timed
// call extends the same 1024-token history, rather than growing it repeatedly.
// Report synchronized host/API latency, including selected greedy token IDs;
// this is deliberately not mislabeled as pure GPU kernel-event time.
template <class Forward, class Rewind, class Evidence>
void measure(std::ostream& out, int repetitions, Forward forward,
             Rewind rewind, Evidence evidence) {
#ifdef ROOFLINE_PROFILE
  // Diagnostic builds capture only this measurement section. Model loading
  // and the initial 32 histories are excluded from the profiler capture.
  require(cudaProfilerStart() == cudaSuccess, "cudaProfilerStart failed");
#endif
  out << "\"cases\":[";
  for (std::size_t c = 0; c < kMixes.size(); ++c) {
    const auto& mix = kMixes[c];
    auto input = rows(mix);
    std::vector<double> times;
    std::vector<std::int64_t> expected;
    int deterministic = 0;
    for (int i = -3; i < repetitions; ++i) {
      rewind();
      sync();
#ifdef ROOFLINE_PROFILE
      std::string label = std::string("mix=") + mix.name +
          ";phase=" + (i < 0 ? "warmup" : "timed") +
          ";iteration=" + std::to_string(i);
      nvtxRangePushA(label.c_str());
#endif
      auto start = std::chrono::steady_clock::now();
      auto ids = forward(input, mix.decode);
      sync();
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start).count();
#ifdef ROOFLINE_PROFILE
      nvtxRangePop();
#endif
      require(static_cast<int>(ids.size()) == mix.decode,
              "Selected output count differs from decode count");
      if (i == -3) expected = ids;
      deterministic += ids == expected;
      if (i >= 0) times.push_back(ms);
    }
    require(deterministic == repetitions + 3,
            "Repeated same-prefix greedy output is not deterministic");
    if (c) out << ',';
    out << "{\"mix\":\"" << mix.name << "\",\"decode_rows\":" << mix.decode
        << ",\"prefill_rows\":" << mix.prefill
        << ",\"batch_tokens\":" << input.size()
        << ",\"context_before_every_timed_step\":" << kHistory
        << ",\"warmups\":3,\"repetitions\":" << repetitions
        << ",\"same_prefix_greedy_repeats_equal\":true,\"latency_ms\":[";
    for (std::size_t i = 0; i < times.size(); ++i) {
      if (i) out << ',';
      out << std::setprecision(15) << times[i];
    }
    out << "],\"greedy_tokens\":";
    print_tokens(out, expected);
    out << ',';
    evidence(out, mix, repetitions + 3);
    out << '}';
    out.flush();
    std::cerr << "MEASURED " << mix.name << '\n';
  }
  out << ']';
#ifdef ROOFLINE_PROFILE
  require(cudaProfilerStop() == cudaSuccess, "cudaProfilerStop failed");
#endif
}
} // namespace roofline
