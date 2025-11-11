/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/Exception.h>

#include <sstream>

#ifdef ETVK_BOOST_STACKTRACE_AVAILABLE
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE
#include <boost/stacktrace.hpp>
#endif // ETVK_BOOST_STACKTRACE_AVAILABLE

namespace vkcompute {
namespace vkapi {

#define VK_RESULT_CASE(code) \
  case code:                 \
    out << #code;            \
    break;

std::ostream& operator<<(std::ostream& out, const VkResult result) {
  switch (result) {
    VK_RESULT_CASE(VK_SUCCESS)
    VK_RESULT_CASE(VK_NOT_READY)
    VK_RESULT_CASE(VK_TIMEOUT)
    VK_RESULT_CASE(VK_EVENT_SET)
    VK_RESULT_CASE(VK_EVENT_RESET)
    VK_RESULT_CASE(VK_INCOMPLETE)
    VK_RESULT_CASE(VK_ERROR_OUT_OF_HOST_MEMORY)
    VK_RESULT_CASE(VK_ERROR_OUT_OF_DEVICE_MEMORY)
    VK_RESULT_CASE(VK_ERROR_INITIALIZATION_FAILED)
    VK_RESULT_CASE(VK_ERROR_DEVICE_LOST)
    VK_RESULT_CASE(VK_ERROR_MEMORY_MAP_FAILED)
    VK_RESULT_CASE(VK_ERROR_LAYER_NOT_PRESENT)
    VK_RESULT_CASE(VK_ERROR_EXTENSION_NOT_PRESENT)
    VK_RESULT_CASE(VK_ERROR_FEATURE_NOT_PRESENT)
    VK_RESULT_CASE(VK_ERROR_INCOMPATIBLE_DRIVER)
    VK_RESULT_CASE(VK_ERROR_TOO_MANY_OBJECTS)
    VK_RESULT_CASE(VK_ERROR_FORMAT_NOT_SUPPORTED)
    VK_RESULT_CASE(VK_ERROR_FRAGMENTED_POOL)
    default:
      out << "VK_ERROR_UNKNOWN (VkResult " << result << ")";
      break;
  }
  return out;
}

#undef VK_RESULT_CASE

//
// SourceLocation
//

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;
  return out;
}

//
// Exception
//

Error::Error(SourceLocation source_location, std::string msg)
    : msg_(std::move(msg)), source_location_{source_location} {
  std::ostringstream oss;
  oss << "Exception raised from " << source_location_ << ": ";
  oss << msg_;
#ifdef ETVK_BOOST_STACKTRACE_AVAILABLE
  oss << "\n";
  oss << "Stack trace:\n";
  oss << boost::stacktrace::stacktrace();
#endif // ETVK_BOOST_STACKTRACE_AVAILABLE
  what_ = oss.str();
}

Error::Error(SourceLocation source_location, const char* cond, std::string msg)
    : msg_(std::move(msg)), source_location_{source_location} {
  std::ostringstream oss;
  oss << "Exception raised from " << source_location_ << ": ";
  oss << "(" << cond << ") is false! ";
  oss << msg_;
#ifdef ETVK_BOOST_STACKTRACE_AVAILABLE
  oss << "\n";
  oss << "Stack trace:\n";
  oss << boost::stacktrace::stacktrace();
#endif // ETVK_BOOST_STACKTRACE_AVAILABLE
  what_ = oss.str();
}

//
// ShaderNotSupportedError
//

std::ostream& operator<<(std::ostream& out, const VulkanExtension result) {
  switch (result) {
    case VulkanExtension::SHADER_INT16:
      out << "shaderInt16";
      break;
    case VulkanExtension::INT16_STORAGE:
      out << "VK_KHR_16bit_storage";
      break;
    case VulkanExtension::INT8_STORAGE:
      out << "VK_KHR_8bit_storage";
      break;
    case VulkanExtension::INTEGER_DOT_PRODUCT:
      out << "VK_KHR_shader_integer_dot_product";
      break;
    case VulkanExtension::SHADER_INT64:
      out << "shaderInt64";
      break;
    case VulkanExtension::SHADER_FLOAT64:
      out << "shaderFloat64";
      break;
  }
  return out;
}

ShaderNotSupportedError::ShaderNotSupportedError(
    std::string shader_name,
    VulkanExtension extension)
    : shader_name_(std::move(shader_name)), extension_{extension} {
  std::ostringstream oss;
  oss << "Shader " << shader_name_ << " ";
  oss << "not compatible with device. ";
  oss << "Missing support for extension or physical device feature: ";
  oss << extension_;
  what_ = oss.str();
}

} // namespace vkapi
} // namespace vkcompute
