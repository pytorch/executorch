// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ptn {

// One member of a JSON object. Defined below JsonValue, which holds a vector of
// these (std::vector supports an incomplete element type, which is what makes
// the recursion work without indirection).
struct JsonMember;

// Minimal JSON reader for the index metadata inside a .ptn package: the
// safetensors header and the alias map.
//
// Not a general JSON implementation. Numbers are unsigned integers only,
// because every number in those two documents is a dimension, a byte offset, or
// a byte count; a fractional, negative, or exponent-bearing value means the
// index is malformed and throws rather than being silently truncated.
//
// The contents are a std::variant whose alternatives are listed in Kind order,
// so kind() is the variant's index. Objects keep their members in document
// order, since callers iterate them rather than probing for known keys.
class JsonValue {
 public:
  enum class Kind : int8_t {
    Null = 0,
    Bool = 1,
    Number = 2,
    String = 3,
    Array = 4,
    Object = 5,
  };

  using Array = std::vector<JsonValue>;
  using Object = std::vector<JsonMember>;

 private:
  std::variant<std::monostate, bool, uint64_t, std::string, Array, Object>
      value_;

  // The live payload, or a std::runtime_error naming the kind expected.
  template <typename T>
  const T& payload(const char* expected) const {
    const T* p = std::get_if<T>(&value_);
    if (p == nullptr) {
      throw_bad_kind(expected);
    }
    return *p;
  }

  [[noreturn]] static void throw_bad_kind(const char* expected);

 public:
  JsonValue() = default; // null
  /* implicit */ JsonValue(bool value) : value_(value) {}
  /* implicit */ JsonValue(uint64_t value) : value_(value) {}
  /* implicit */ JsonValue(std::string value) : value_(std::move(value)) {}
  /* implicit */ JsonValue(Array items) : value_(std::move(items)) {}
  /* implicit */ JsonValue(Object members) : value_(std::move(members)) {}

  Kind kind() const {
    return static_cast<Kind>(value_.index());
  }
  bool is_object() const {
    return std::holds_alternative<Object>(value_);
  }
  bool is_array() const {
    return std::holds_alternative<Array>(value_);
  }
  bool is_string() const {
    return std::holds_alternative<std::string>(value_);
  }
  bool is_number() const {
    return std::holds_alternative<uint64_t>(value_);
  }

  // Typed access. Each throws std::runtime_error if the value is another kind.
  bool as_bool() const {
    return payload<bool>("a bool");
  }
  uint64_t as_number() const {
    return payload<uint64_t>("a number");
  }
  const std::string& as_string() const {
    return payload<std::string>("a string");
  }
  const Array& as_array() const {
    return payload<Array>("an array");
  }
  const Object& as_object() const {
    return payload<Object>("an object");
  }

  // Object member by key, or nullptr when absent. Linear in member count; the
  // documents this parses are read by iteration, not by repeated probing.
  const JsonValue* find(std::string_view key) const;
};

struct JsonMember {
  std::string key;
  JsonValue value;
};

// Parse `text` as one complete JSON document. Throws std::runtime_error on
// anything malformed, including trailing non-whitespace after the value.
JsonValue json_parse(std::string_view text);

} // namespace ptn
