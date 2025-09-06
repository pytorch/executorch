/*
 *  Copyright (c) 2025 Samsung Electronics Co. LTD
 *  All rights reserved
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree.
 *
 */
#pragma once

#include <stdint.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <include/common-types.h>

namespace torch {
namespace executor {
namespace enn {

template <class T>
struct ScalarTypeCast {
  constexpr static ScalarType value = ScalarType::UNKNOWN;
};

template <>
struct ScalarTypeCast<uint64_t> {
  constexpr static ScalarType value = ScalarType::UINT64;
};

template <>
struct ScalarTypeCast<int64_t> {
  constexpr static ScalarType value = ScalarType::INT64;
};

template <>
struct ScalarTypeCast<uint32_t> {
  constexpr static ScalarType value = ScalarType::UINT32;
};

template <>
struct ScalarTypeCast<int32_t> {
  constexpr static ScalarType value = ScalarType::INT32;
};

template <>
struct ScalarTypeCast<float> {
  constexpr static ScalarType value = ScalarType::FLOAT32;
};

template <>
struct ScalarTypeCast<double> {
  constexpr static ScalarType value = ScalarType::FLOAT64;
};

template <>
struct ScalarTypeCast<bool> {
  constexpr static ScalarType value = ScalarType::BOOL;
};

class OpParamWrapper {
 public:
  OpParamWrapper(std::string key) : key_name_(std::move(key)) {}

  ~OpParamWrapper() = default;

  std::string getKeyName() const {
    return key_name_;
  }

  template <typename T>
  void SetScalarValue(T value) {
    auto bytes = sizeof(T);
    storage_ = std::unique_ptr<uint8_t[]>(new uint8_t[bytes]);
    memcpy(storage_.get(), &value, bytes);
    size_ = 1;
    is_scalar_ = true;
    scalar_type_ = ScalarTypeCast<T>::value;
  }

  template <typename T>
  void SetVectorValue(const std::vector<T>& value) {
    auto bytes = sizeof(T) * value.size();
    storage_ = std::unique_ptr<uint8_t[]>(new uint8_t[bytes]);
    memcpy(storage_.get(), value.data(), bytes);
    size_ = value.size();
    is_scalar_ = false;
    scalar_type_ = ScalarTypeCast<T>::value;
  }

  void SetStringValue(const std::string& value) {
    auto bytes = sizeof(std::string::value_type) * value.size();
    storage_ = std::unique_ptr<uint8_t[]>(new uint8_t[bytes]);
    memcpy(storage_.get(), value.data(), bytes);
    size_ = value.size();
    is_scalar_ = false;
    scalar_type_ = ScalarType::CHAR;
  }

  ParamWrapper Dump() const {
    ParamWrapper param;
    param.data = storage_.get();
    param.size = size_;
    param.is_scalar = is_scalar_;
    param.type = scalar_type_;

    return param;
  }

 private:
  std::string key_name_;
  std::unique_ptr<uint8_t[]> storage_ = nullptr;
  uint32_t size_ = 0;
  bool is_scalar_ = false;
  ScalarType scalar_type_ = ScalarType::UNKNOWN;
};

} // namespace enn
} // namespace executor
} // namespace torch
