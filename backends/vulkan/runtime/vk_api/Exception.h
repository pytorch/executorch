/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/backends/vulkan/runtime/utils/StringUtils.h>

#include <exception>
#include <ostream>
#include <string>
#include <vector>

#define VK_CHECK(function)                                                  \
  do {                                                                      \
    const VkResult result = (function);                                     \
    if (VK_SUCCESS != result) {                                             \
      throw ::vkcompute::vkapi::Error(                                      \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},            \
          ::vkcompute::utils::concat_str(#function, " returned ", result)); \
    }                                                                       \
  } while (false)

#define VK_CHECK_COND(cond, ...)                                 \
  do {                                                           \
    if (!(cond)) {                                               \
      throw ::vkcompute::vkapi::Error(                           \
          {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
          #cond,                                                 \
          ::vkcompute::utils::concat_str(__VA_ARGS__));          \
    }                                                            \
  } while (false)

#define VK_THROW(...)                                          \
  do {                                                         \
    throw ::vkcompute::vkapi::Error(                           \
        {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
        ::vkcompute::utils::concat_str(__VA_ARGS__));          \
  } while (false)

namespace vkcompute {
namespace vkapi {

std::ostream& operator<<(std::ostream& out, const VkResult loc);

struct SourceLocation {
  const char* function;
  const char* file;
  uint32_t line;
};

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc);

class Error : public std::exception {
 public:
  Error(SourceLocation source_location, std::string msg);
  Error(SourceLocation source_location, const char* cond, std::string msg);

 private:
  std::string msg_;
  SourceLocation source_location_;
  std::string what_;

 public:
  const std::string& msg() const {
    return msg_;
  }

  const char* what() const noexcept override {
    return what_.c_str();
  }
};

} // namespace vkapi
} // namespace vkcompute
