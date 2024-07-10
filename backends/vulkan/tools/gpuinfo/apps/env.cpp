/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/microsoft/ArchProbe/blob/main/apps/archprobe/env.cpp
// MIT-licensed by Rendong Liang.

#include "env.h"

namespace gpuinfo {

std::string pretty_data_size(size_t size) {
  const size_t K = 1024;
  if (size < K) {
    return util::format(size, "B");
  }
  size /= K;
  if (size < K) {
    return util::format(size, "KB");
  }
  size /= K;
  if (size < K) {
    return util::format(size, "MB");
  }
  size /= K;
  if (size < K) {
    return util::format(size, "GB");
  }
  size /= K;
  if (size < K) {
    return util::format(size, "TB");
  }
  size /= K;
  gpuinfo::panic("unsupported data size");
  return {};
}

json::JsonValue load_env_cfg(const char* path) {
  try {
    auto json_txt = util::load_text(path);
    log::debug("loaded configuration '", json_txt, "'");
    json::JsonValue out{};
    if (json::try_parse(json_txt, out)) {
      gpuinfo::assert(out.is_obj());
      return out;
    } else {
      log::warn(
          "failed to parse environment config from '",
          path,
          "', a default configuration will be created to overwrite it");
      return json::JsonObject{};
    }
  } catch (gpuinfo::AssertionFailedException) {
    log::warn(
        "configuration file cannot be opened at '",
        path,
        "', a default configuration will be created");
    return json::JsonObject{};
  }
}

json::JsonValue load_report(const char* path) {
  try {
    auto json_txt = util::load_text(path);
    log::debug("loaded report '", json_txt, "'");
    json::JsonValue out{};
    if (json::try_parse(json_txt, out)) {
      gpuinfo::assert(out.is_obj());
      return out;
    } else {
      log::warn(
          "failed to parse report from '",
          path,
          "', a new report "
          "will be created to overwrite it");
      return json::JsonObject{};
    }
  } catch (gpuinfo::AssertionFailedException) {
    log::warn(
        "report file cannot be opened at '",
        path,
        "', a new "
        "report will be created");
    return json::JsonObject{};
  }
}

void report_dev(Environment& env) {
  if (env.report_started_lazy("Device")) {
    return;
  }
  env.report_value("SmCount", env.dev_report.nsm);
  env.report_value("LogicThreadCount", env.dev_report.nthread_logic);
  env.report_value("MaxBufferSize", env.dev_report.buf_size_max);
  env.report_value("MaxConstMemSize", env.dev_report.const_mem_size_max);
  env.report_value("MaxLocalMemSize", env.dev_report.local_mem_size_max);
  env.report_value("CacheSize", env.dev_report.buf_cache_size);
  env.report_value("CachelineSize", env.dev_report.buf_cacheline_size);
  if (env.dev_report.support_img) {
    env.report_value("MaxImageWidth", env.dev_report.img_width_max);
    env.report_value("MaxImageHeight", env.dev_report.img_height_max);
  }
  if (env.dev_report.has_page_size) {
    env.report_value("PageSize_QCOM", env.dev_report.page_size);
  }
  env.report_ready(true);
}

Environment::Environment(
    uint32_t idev,
    const char* cfg_path,
    const char* report_path)
    : // TODO: Transcribe to Vulkan
      /*
        dev_(gpuinfo::select_dev(idev)),
        ctxt_(gpuinfo::create_ctxt(dev_)),
        cmd_queue_(gpuinfo::create_cmd_queue(ctxt_)),
      */
      aspects_started_(),
      cur_aspect_(),
      cur_table_(nullptr),
      cfg_path_(cfg_path),
      report_path_(report_path),
      cfg_(load_env_cfg(cfg_path)),
      report_(load_report(report_path)),
      dev_report(/*collect_dev_report(dev_)*/),
      my_report() {
  report_dev(*this);
}
Environment::~Environment() {
  util::save_text(cfg_path_.c_str(), json::print(cfg_));
  log::info("saved configuration to '", cfg_path_, "'");
  util::save_text(report_path_.c_str(), json::print(report_));
  log::info("saved report to '", report_path_, "'");
}

void Environment::report_started(const std::string& aspect_name) {
  gpuinfo::assert(!aspect_name.empty(), "aspect name cannot be empty");
  aspects_started_.insert(aspect_name);
  log::info("[", aspect_name, "]");
  log::push_indent();
  cur_aspect_ = aspect_name;
}
bool Environment::report_started_lazy(const std::string& aspect_name) {
  auto aspect_it = report_.obj.find(aspect_name);
  if (aspect_it == report_.obj.end() || !aspect_it->second.is_obj()) {
    report_started(aspect_name);
    return false;
  }
  auto done_it = aspect_it->second.obj.find("Done");
  if (done_it == aspect_it->second.obj.end() || !done_it->second.is_bool()) {
    report_started(aspect_name);
    return false;
  }
  if (done_it->second.b) {
    log::info("ignored aspect '", aspect_name, "' because it's done");
    return true;
  } else {
    report_started(aspect_name);
    return false;
  }
}
void Environment::report_ready(bool done) {
  gpuinfo::assert(
      !cur_aspect_.empty(),
      "announcing ready for an not-yet-started report is not allowed");
  gpuinfo::assert(
      aspects_started_.find(cur_aspect_) != aspects_started_.end(),
      "aspect has not report to start yet");
  report_value("Done", done);
  if (cur_table_ != nullptr) {
    auto csv = cur_table_->to_csv();
    auto fname = util::format(cur_aspect_, ".csv");
    util::save_text(fname.c_str(), csv);
    log::info("saved data table to '", fname, "'");
    cur_table_ = nullptr;
  }
  cur_aspect_ = {};
  log::pop_indent();
}
void Environment::check_dep(const std::string& aspect_name) {
  bool done = false;
  gpuinfo::assert(
      try_get_aspect_report(aspect_name, "Done", done) && done,
      "aspect '",
      aspect_name,
      "' is required but is not ready yet");
}

table::Table& Environment::table() {
  gpuinfo::assert(cur_table_ != nullptr, "requested table is not initialized");
  return *cur_table_;
}

// Find the minimal number of iterations that a kernel can run up to
// `min_time_us` microseconds.
void Environment::ensure_min_niter(
    double min_time_us,
    uint32_t& niter,
    std::function<double()> run) {
  const uint32_t DEFAULT_NITER = 100;
  niter = DEFAULT_NITER;
  for (uint32_t i = 0; i < 100; ++i) {
    double t = run();
    if (t > min_time_us * 0.99) {
      log::info("found minimal niter=", niter, " to take ", min_time_us, "us");
      return;
    }
    log::debug(
        "niter=",
        niter,
        " doesn't run long enough (",
        t,
        "us <= ",
        min_time_us,
        "us)");
    niter = uint32_t(niter * min_time_us / t);
  }
  gpuinfo::panic(
      "unable to find a minimal iteration number for ",
      cur_aspect_,
      "; is your code aggresively optimized by the compiler?");
}

} // namespace gpuinfo
