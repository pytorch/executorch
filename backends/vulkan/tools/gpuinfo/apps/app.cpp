/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/microsoft/ArchProbe/blob/main/apps/archprobe/app.cpp
// MIT-licensed by Rendong Liang.

#include <signal.h>
#include "args.h"
#include "env.h"
#include "stats.h"

using namespace gpuinfo;
using namespace gpuinfo::stats;

namespace {

void log_cb(log::LogLevel lv, const std::string& msg) {
  using log::LogLevel;
  switch (lv) {
    case LogLevel::L_LOG_LEVEL_DEBUG:
      printf("[\x1b[90mDEBUG\x1B[0m] %s\n", msg.c_str());
      break;
    case LogLevel::L_LOG_LEVEL_INFO:
      printf("[\x1B[32mINFO\x1B[0m] %s\n", msg.c_str());
      break;
    case LogLevel::L_LOG_LEVEL_WARNING:
      printf("[\x1B[33mWARN\x1B[0m] %s\n", msg.c_str());
      break;
    case LogLevel::L_LOG_LEVEL_ERROR:
      printf("[\x1B[31mERROR\x1B[0m] %s\n", msg.c_str());
      break;
  }
  std::fflush(stdout);
}

} // namespace

namespace gpuinfo {

std::string vec_name_by_ncomp(const char* scalar_name, uint32_t ncomp) {
  return scalar_name + (ncomp == 1 ? "" : std::to_string(ncomp));
}
std::string vec_name_by_ncomp_log2(
    const char* scalar_name,
    uint32_t ncomp_log2) {
  return vec_name_by_ncomp(scalar_name, 1 << ncomp_log2);
}

template <typename T>
bool is_pow2(T x) {
  return (x & (-x)) == x;
}
template <typename T>
T next_pow2(T x) {
  T y = 1;
  for (; y < x; y <<= 1) {
  }
  return y;
}

// Power of integer.
template <typename T>
T powi(T x, T p) {
  if (p == 0) {
    return 1;
  }
  if (p == 1) {
    return x;
  }
  return x * powi(x, p - 1);
}
// Log2 of integer. 0 is treated as two-to-the-zero.
template <typename T>
T log2i(T x) {
  T counter = 0;
  while (x >= 2) {
    x >>= 1;
    ++counter;
  }
  return counter;
}

using Aspect = std::function<void(Environment&)>;

class GPUInfo {
  Environment env_;

 public:
  GPUInfo(uint32_t idev) : env_(idev) {}

  GPUInfo& with_aspect(const Aspect& aspect) {
    aspect(env_);
    return *this;
  }

  void clear_aspect_report(const std::string& aspect) {
    env_.clear_aspect_report(aspect);
  }
};

template <uint32_t NTap>
struct DtJumpFinder {
 private:
  NTapAvgStats<double, NTap> time_avg_;
  AvgStats<double> dtime_avg_;
  double compensation_;
  double threshold_;

 public:
  // Compensation is a tiny additive to give on delta time so that the algorithm
  // works smoothly when a sequence of identical timing is ingested, which is
  // pretty common in our tests. Threshold is simply how many times the new
  // delta has to be to be recognized as a deviation.
  DtJumpFinder(double compensation = 0.01, double threshold = 10)
      : time_avg_(),
        dtime_avg_(),
        compensation_(compensation),
        threshold_(threshold) {}

  // Returns true if the delta time regarding to the last data point seems
  // normal; returns false if it seems the new data point is too much away from
  // the historical records.
  bool push(double time) {
    if (time_avg_.has_value()) {
      double dtime = std::abs(time - time_avg_) + (compensation_ * time_avg_);
      if (dtime_avg_.has_value()) {
        double ddtime = std::abs(dtime - dtime_avg_);
        if (ddtime > threshold_ * dtime_avg_) {
          return true;
        }
      }
      dtime_avg_.push(dtime);
    }
    time_avg_.push(time);
    return false;
  }

  double dtime_avg() const {
    return dtime_avg_;
  }
  double compensate_time() const {
    return compensation_ * time_avg_;
  }
};

namespace aspects {} // namespace aspects

static std::unique_ptr<GPUInfo> APP = nullptr;

void guarded_main(const std::string& clear_aspect) {
  // TODO: Transcribe to Vulkan
  /*
  cl_int err;

  gpuinfo::initialize();
  */

  APP = std::make_unique<GPUInfo>(0);
  APP->clear_aspect_report(clear_aspect);
  (*APP);

  APP.reset();
}

void sigproc(int sig) {
  const char* sig_name = "UNKNOWN SIGNAL";
#ifdef _WIN32
#define SIGHUP 1
#define SIGQUIT 3
#define SIGTRAP 5
#define SIGKILL 9
#endif
  switch (sig) {
    // When you interrupt adb, and adb kills GPUInfo in its SIGINT process.
    case SIGHUP:
      sig_name = "SIGHUP";
      break;
    // When you interrupt in an `adb shell` session.
    case SIGINT:
      sig_name = "SIGINT";
      break;
    // Other weird cases.
    case SIGQUIT:
      sig_name = "SIGQUIT";
      break;
    case SIGTRAP:
      sig_name = "SIGTRAP";
      break;
    case SIGABRT:
      sig_name = "SIGABRT";
      break;
    case SIGTERM:
      sig_name = "SIGTERM";
      break;
    case SIGKILL:
      sig_name = "SIGKILL";
      break;
  }
  log::error("captured ", sig_name, "! progress is saved");
  APP.reset();
  std::exit(1);
}

} // namespace gpuinfo

struct AppConfig {
  bool verbose;
  std::string clear_aspect;
};

AppConfig configurate(int argc, const char** argv) {
  using namespace gpuinfo::args;
  AppConfig cfg{};
  init_arg_parse("GPUInfo", "Discover hardware details by micro-benchmarks.");
  reg_arg<SwitchParser>(
      "-v", "--verbose", cfg.verbose, "Print more detail for debugging.");
  reg_arg<StringParser>(
      "-c",
      "--clear",
      cfg.clear_aspect,
      "Clear the results of specified aspect.");
  parse_args(argc, argv);
  return cfg;
}

int main(int argc, const char** argv) {
  log::set_log_callback(log_cb);
  AppConfig cfg = configurate(argc, argv);
  if (cfg.verbose) {
    log::set_log_filter_level(log::LogLevel::L_LOG_LEVEL_DEBUG);
  } else {
    log::set_log_filter_level(log::LogLevel::L_LOG_LEVEL_INFO);
  }

  signal(SIGHUP, gpuinfo::sigproc);
  signal(SIGINT, gpuinfo::sigproc);
  signal(SIGQUIT, gpuinfo::sigproc);
  signal(SIGTRAP, gpuinfo::sigproc);
  signal(SIGABRT, gpuinfo::sigproc);
  signal(SIGTERM, gpuinfo::sigproc);
  signal(SIGKILL, gpuinfo::sigproc);

  try {
    gpuinfo::guarded_main(cfg.clear_aspect);
  } catch (const std::exception& e) {
    log::error("application threw an exception");
    log::error(e.what());
    log::error("application cannot continue");
  } catch (...) {
    log::error("application threw an illiterate exception");
  }

  return 0;
}
