// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ptn {

// X-macro table of scalar types: (CPP_TYPE, NAME, ID).
//
// The ids are the serialized ScalarType values, so a deserializer maps a
// serialized byte straight to this enum. They are not sequential: the gaps are
// ids reserved for element types this header does not carry.
//
// Half and BFloat16 have no 16-bit float type in this dependency-free header;
// they map to uint16_t as a raw storage stand-in, correct for size and layout.
#define PTN_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte, 0)              \
  _(int8_t, Char, 1)               \
  _(int16_t, Short, 2)             \
  _(int32_t, Int, 3)               \
  _(int64_t, Long, 4)              \
  _(uint16_t, Half, 5)             \
  _(float, Float, 6)               \
  _(double, Double, 7)             \
  _(bool, Bool, 11)                \
  _(uint16_t, BFloat16, 15)        \
  _(uint16_t, UInt16, 16)          \
  _(uint32_t, UInt32, 17)          \
  _(uint64_t, UInt64, 18)

enum class ScalarType : int8_t {
#define PTN_DEFINE_ENUM(cpp_type, name, id) name = id,
  PTN_FORALL_SCALAR_TYPES(PTN_DEFINE_ENUM)
#undef PTN_DEFINE_ENUM
};

#define PTN_DEFINE_CONSTANT(cpp_type, name, id) \
  constexpr ScalarType k##name = ScalarType::name;
PTN_FORALL_SCALAR_TYPES(PTN_DEFINE_CONSTANT)
#undef PTN_DEFINE_CONSTANT

// Forward mapping only: a reverse C++-type -> ScalarType trait is omitted,
// since uint16_t would collide across Half / BFloat16 / UInt16.
template <ScalarType N>
struct ScalarTypeToCppType;
#define PTN_SPECIALIZE_S2C(cpp_type, name, id)   \
  template <>                                    \
  struct ScalarTypeToCppType<ScalarType::name> { \
    using type = cpp_type;                       \
  };
PTN_FORALL_SCALAR_TYPES(PTN_SPECIALIZE_S2C)
#undef PTN_SPECIALIZE_S2C

template <ScalarType N>
using cpp_type_t = typename ScalarTypeToCppType<N>::type;

// Throws std::runtime_error on a value outside the table, e.g. a bad cast from
// an out-of-range serialized byte.
inline size_t element_size(ScalarType t) {
  switch (t) {
#define PTN_CASE_ELEMSIZE(cpp_type, name, id) \
  case ScalarType::name:                      \
    return sizeof(cpp_type);
    PTN_FORALL_SCALAR_TYPES(PTN_CASE_ELEMSIZE)
#undef PTN_CASE_ELEMSIZE
  }
  throw std::runtime_error("element_size: unrecognized ScalarType");
}

// Enumerator name, e.g. "Float". Throws on a value outside the table.
inline const char* scalar_type_name(ScalarType t) {
  switch (t) {
#define PTN_CASE_NAME(cpp_type, name, id) \
  case ScalarType::name:                  \
    return #name;
    PTN_FORALL_SCALAR_TYPES(PTN_CASE_NAME)
#undef PTN_CASE_NAME
  }
  throw std::runtime_error("scalar_type_name: unrecognized ScalarType");
}

} // namespace ptn
