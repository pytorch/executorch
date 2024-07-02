/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/microsoft/ArchProbe/blob/main/apps/archprobe/env.hpp
// MIT-licensed by Rendong Liang.

#pragma once
#include <functional>
#include <map>
#include <memory>
#include <set>
#include "assert.h"
#include "json.h"
#include "log.h"
#include "table.h"
#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION CL_TARGET_OPENCL_VERSION

namespace gpuinfo {

std::string pretty_data_size(size_t size);

struct DeviceReport {
  bool has_page_size;
  size_t page_size;

  size_t buf_cacheline_size;
  size_t buf_size_max;
  size_t buf_cache_size;

  size_t const_mem_size_max;
  size_t local_mem_size_max;

  bool support_img;
  uint32_t img_width_max;
  uint32_t img_height_max;

  uint32_t nsm;
  uint32_t nthread_logic;
};
struct ProfiledReport {
  double timing_std;

  std::map<uint32_t, uint32_t> nthread_logic_for_nreg;

  double gflops_fp16;
  double gflops_fp32;
  double gflops_int32;
  uint32_t nmin_warp;
  uint32_t nwarp;
  uint32_t nthread_phys;
  uint32_t nthread_warp;
  uint32_t nthread_min_warp;

  uint32_t buf_vec_width;
  std::string buf_vec_ty;
  uint32_t buf_cacheline_size;
  std::vector<uint32_t> buf_cache_sizes;

  uint32_t img_cacheline_size;
  std::vector<uint32_t> img_cache_sizes;
  double img_bandwidth;
};
class Environment {
  // TODO: Transcribe to Vulkan
  // cl::Device dev_;
  // cl::Context ctxt_;
  // cl::CommandQueue cmd_queue_;
  std::set<std::string> aspects_started_;
  std::string cur_aspect_;
  std::unique_ptr<table::Table> cur_table_;
  std::string cfg_path_;
  std::string report_path_;
  json::JsonValue cfg_;
  json::JsonValue report_;

 public:
  const DeviceReport dev_report;
  ProfiledReport my_report;

  Environment(
      uint32_t idev,
      const char* cfg_path = "GPUInfo.json",
      const char* report_path = "GPUInfoReport.json");
  ~Environment();

  void report_started(const std::string& aspect_name);
  // Returns false if there is no existing report about the aspect to be started
  // or such report is not yet marked with '"Done": true'. It means that when
  // this method returns true, the aspect can return right a way.
  bool report_started_lazy(const std::string& aspect_name);
  void report_ready(bool done = false);
  void check_dep(const std::string& aspect_name);

  template <typename... TArgs>
  void init_table(TArgs&&... args) {
    gpuinfo::assert(
        !cur_aspect_.empty(),
        "table can only be initialized in scope of a report");
    log::info("initialized table for aspect '", cur_aspect_, "'");
    cur_table_ = std::make_unique<table::Table>(args...);
  }
  table::Table& table();

  inline json::JsonValue& get_aspect_cfg(const std::string& aspect) {
    auto it = cfg_.obj.find(aspect);
    if (it == cfg_.obj.end() || !it->second.is_obj()) {
      log::warn(
          "aspect configuration ('",
          cur_aspect_,
          "') is invalid, "
          "a new record is created");
      cfg_.obj[aspect] = json::JsonObject{};
    }
    return cfg_.obj[aspect];
  }
  inline json::JsonValue& get_cfg() {
    return get_aspect_cfg(cur_aspect_);
  }
  template <typename T>
  inline T cfg_num(const std::string& name, T default_value) {
    auto& cfg = get_cfg();
    if (cfg.obj.find(name) == cfg_.obj.end() || !cfg.obj[name].is_num()) {
      log::warn(
          "record entry ('",
          name,
          "') is invalid, a new record "
          "is created");
      cfg.obj[name] = json::JsonValue(default_value);
    }
    return (T)cfg[name];
  }

  inline json::JsonValue& get_report() {
    return get_aspect_report(cur_aspect_);
  }
  inline json::JsonValue& get_aspect_report(const std::string& aspect) {
    auto it = report_.obj.find(aspect);
    if (it == report_.obj.end() || !it->second.is_obj()) {
      log::warn(
          "aspect report ('",
          aspect,
          "') is invalid, a new record is "
          "created");
      report_.obj[aspect] = json::JsonObject{};
    }
    return report_.obj[aspect];
  }
  template <typename T>
  inline bool try_get_report(const std::string& name, T& out) {
    return try_get_aspect_report(cur_aspect_, name, out);
  }
  template <typename T>
  inline bool try_get_aspect_report(
      const std::string& aspect,
      const std::string& name,
      T& out) {
    const auto& report = get_aspect_report(aspect);
    auto it = report.obj.find(name);
    if (it == report.obj.end()) {
      return false;
    } else {
      out = (T)it->second;
      log::info(
          "already know that '", name, "' from aspect '", aspect, "' is ", out);
      return true;
    }
  }
  template <typename T>
  inline T must_get_aspect_report(
      const std::string& aspect,
      const std::string& name) {
    T out;
    gpuinfo::assert(
        try_get_aspect_report(aspect, name, out),
        "cannot get report '",
        name,
        "' from aspect '",
        aspect,
        "'");
    return out;
  }
  template <typename T>
  inline void report_value(const std::string& name, T value) {
    auto& report = get_report();
    log::info("reported '", name, "' = '", value, "'");
    report.obj[name] = json::JsonValue(value);
  }

  inline void clear_aspect_report(const std::string& aspect) {
    if (!aspect.empty()) {
      get_aspect_report(aspect) = json::JsonObject{};
      log::info("cleared report of aspect '", aspect, "'");
    }
  }

  void ensure_min_niter(
      double min_time_us,
      uint32_t& niter,
      std::function<double()> run);
};

} // namespace gpuinfo
