/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/0e3c1394b493db3e3d5b443c869545cac712827a/log.hpp;
// MIT-licensed by Rendong Liang.

// Logging infrastructure.
// @PENGUINLIONG
#pragma once
#include <cstdint>
#include <string>
#include "util.h"

namespace gpuinfo {

namespace log {
// Logging infrastructure.

enum class LogLevel {
  L_LOG_LEVEL_DEBUG,
  L_LOG_LEVEL_INFO,
  L_LOG_LEVEL_WARNING,
  L_LOG_LEVEL_ERROR,
};

namespace detail {

extern void (*log_callback)(LogLevel lv, const std::string& msg);
extern LogLevel filter_lv;
extern uint32_t indent;

} // namespace detail

void set_log_callback(decltype(detail::log_callback) cb);
void set_log_filter_level(LogLevel lv);
template <typename... TArgs>
void log(LogLevel lv, const TArgs&... msg) {
  if (detail::log_callback != nullptr && lv >= detail::filter_lv) {
    std::string indent(detail::indent, ' ');
    detail::log_callback(lv, util::format(indent, msg...));
  }
}

void push_indent();
void pop_indent();

template <typename... TArgs>
inline void debug(const TArgs&... msg) {
  log(LogLevel::L_LOG_LEVEL_DEBUG, msg...);
}
template <typename... TArgs>
inline void info(const TArgs&... msg) {
  log(LogLevel::L_LOG_LEVEL_INFO, msg...);
}
template <typename... TArgs>
inline void warn(const TArgs&... msg) {
  log(LogLevel::L_LOG_LEVEL_WARNING, msg...);
}
template <typename... TArgs>
inline void error(const TArgs&... msg) {
  log(LogLevel::L_LOG_LEVEL_ERROR, msg...);
}
} // namespace log

} // namespace gpuinfo
