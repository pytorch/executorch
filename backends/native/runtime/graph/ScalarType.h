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

// X-macro table of scalar types: (CPP_TYPE, NAME, ID). One row per supported
// element type; the row drives the enum, the k<Name> constants, the
// ScalarType -> C++ type trait, element_size(), and scalar_type_name().
//
// IDs are pinned to ExecuTorch's ScalarType (runtime/core/portable_type/
// scalar_type.h) and the native_graph.fbs ScalarType enum, so a deserializer
// maps the serialized byte straight to this enum. The ids are therefore NOT
// sequential (complex / quantized-int ids are reserved and omitted).
//
// Half and BFloat16 have no standalone C++ 16-bit-float type in this
// dependency-free header; they map to uint16_t as a raw storage stand-in, which
// is correct for size / layout purposes.
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

// Shorthand constants: kFloat, kLong, ...
#define PTN_DEFINE_CONSTANT(cpp_type, name, id) \
  constexpr ScalarType k##name = ScalarType::name;
PTN_FORALL_SCALAR_TYPES(PTN_DEFINE_CONSTANT)
#undef PTN_DEFINE_CONSTANT

// ScalarType -> C++ type. Use as `ptn::cpp_type_t<ptn::kFloat>` (== float).
// Forward mapping only: a reverse C++-type -> ScalarType trait is
// intentionally omitted, since uint16_t would collide across Half / BFloat16 /
// UInt16.
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

// Size in bytes of one element. Throws std::runtime_error on an unrecognized
// value (e.g. a bad cast from an out-of-range serialized byte).
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

// Human-readable enumerator name (e.g. "Float"). Throws on unrecognized value.
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
