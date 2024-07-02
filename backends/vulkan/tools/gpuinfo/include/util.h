/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified exerpt of
//  https://github.com/PENGUINLIONG/graphi-t/blob/da31ec530df07c9899e056eeced08a64062dcfce/util.hpp;
// MIT-licensed by Rendong Liang.

// HAL independent utilities.
// @PENGUINLIONG
#pragma once
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace gpuinfo {

namespace util {

namespace {

template <typename... TArgs>
struct format_impl_t;
template <>
struct format_impl_t<> {
  static inline void format_impl(std::stringstream& ss) {}
};
template <typename T>
struct format_impl_t<T> {
  static inline void format_impl(std::stringstream& ss, const T& x) {
    ss << x;
  }
};
template <typename T, typename... TArgs>
struct format_impl_t<T, TArgs...> {
  static inline void
  format_impl(std::stringstream& ss, const T& x, const TArgs&... others) {
    format_impl_t<T>::format_impl(ss, x);
    format_impl_t<TArgs...>::format_impl(ss, others...);
  }
};

} // namespace

template <typename... TArgs>
inline std::string format(const TArgs&... args) {
  std::stringstream ss{};
  format_impl_t<TArgs...>::format_impl(ss, args...);
  return ss.str();
}

extern std::vector<uint8_t> load_file(const char* path);
extern std::string load_text(const char* path);
extern void save_file(const char* path, const void* data, size_t size);
extern void save_text(const char* path, const std::string& txt);

} // namespace util

} // namespace gpuinfo
