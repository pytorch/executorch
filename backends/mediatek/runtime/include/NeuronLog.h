/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include <api/NeuronAdapter.h>

#include <android/log.h>
#include <sys/system_properties.h>

#include <cstdlib>
#include <string>

namespace torch {
namespace executor {
namespace neuron {

#define AndroidLog(priority, tag, format, ...) \
  __android_log_print(priority, tag, format, ##__VA_ARGS__)

#define LogError(tag, format, ...) \
  AndroidLog(ANDROID_LOG_ERROR, tag, format, ##__VA_ARGS__)

#define LogWarn(tag, format, ...) \
  AndroidLog(ANDROID_LOG_WARN, tag, format, ##__VA_ARGS__)

#define LogInfo(tag, format, ...) \
  AndroidLog(ANDROID_LOG_INFO, tag, format, ##__VA_ARGS__)

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CHECK_VALID_PTR(ptr)                                               \
  do {                                                                     \
    if (__builtin_expect(ptr == nullptr, 0)) {                             \
      LogError(                                                            \
          "NeuronBackend",                                                 \
          "Check fail: " #ptr                                              \
          " == nullptr at line " TOSTRING(__LINE__) " at file " __FILE__); \
      return NEURON_UNEXPECTED_NULL;                                       \
    }                                                                      \
  } while (0)

#define CHECK_NO_ERROR(value)                                            \
  do {                                                                   \
    if (__builtin_expect(value != NEURON_NO_ERROR, 0)) {                 \
      LogError(                                                          \
          "NeuronBackend",                                               \
          "Check fail: " #value " != NEURON_NO_ERROR at line " TOSTRING( \
              __LINE__) " at file " __FILE__);                           \
      return value;                                                      \
    }                                                                    \
  } while (0)

#define CHECK_TRUE(value)                                               \
  do {                                                                  \
    if (__builtin_expect(value != true, 0)) {                           \
      LogError(                                                         \
          "NeuronBackend",                                              \
          "Check fail: " #value                                         \
          " != true at line " TOSTRING(__LINE__) " at file " __FILE__); \
      return NEURON_BAD_STATE;                                          \
    }                                                                   \
  } while (0)

inline int ReadSystemProperty(std::string& property) {
  char property_value[PROP_VALUE_MAX];
  if (__system_property_get(property.c_str(), property_value)) {
    LogInfo("Get System Property  %s : %s", property.c_str(), property_value);
    try {
      int value = std::stoi(property_value);
      return value;
    } catch (...) {
      return -1;
    }
  }
  return -1;
}

} // namespace neuron
} // namespace executor
} // namespace torch
