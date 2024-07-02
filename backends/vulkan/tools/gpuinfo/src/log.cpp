/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/0e3c1394b493db3e3d5b443c869545cac712827a/src/log.cpp;
// MIT-licensed by Rendong Liang.
#include "log.h"

namespace gpuinfo {

namespace log {

namespace detail {

decltype(log_callback) log_callback = nullptr;
LogLevel filter_lv;
uint32_t indent;

} // namespace detail

void set_log_callback(decltype(detail::log_callback) cb) {
  detail::log_callback = cb;
}
void set_log_filter_level(LogLevel lv) {
  detail::filter_lv = lv;
}

void push_indent() {
  detail::indent += 4;
}
void pop_indent() {
  detail::indent -= 4;
}

} // namespace log

} // namespace gpuinfo
