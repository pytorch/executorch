/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * Forked from
 * https://github.com/pytorch/pytorch/blob/master/c10/core/ScalarType.h
 *
 * See file comment in ../ScalarType.h.
 *
 * This file contains all of the non-critical parts of the original ScalarType.h
 * that are not required for the core ExecuTorch runtime, but may be helpful for
 * code that uses ScalarType.
 */

#pragma once

#include <cinttypes>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <executorch/runtime/platform/assert.h>
#ifdef USE_ATEN_LIB
// Note that a lot of the macros/functions defined in this ScalarTypeUtil.h file
// are also defined in c10/core/ScalarType.h, which is included via
// kernel_types.h when building in ATen mode. They tend to use different names
// and a different namespace, but if there are conflicts they should be resolved
// here.
#define ET_FORALL_SCALAR_TYPES AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS
#include <c10/core/ScalarType.h>
namespace exec_aten {
using ScalarType = at::ScalarType;
}
#else
#include <executorch/runtime/core/portable_type/scalar_type.h>
#include <executorch/runtime/core/portable_type/string_view.h>
namespace exec_aten {
using ScalarType = torch::executor::ScalarType;
using string_view = torch::executor::string_view;
} // namespace exec_aten
#endif

namespace torch {
namespace executor {
/// Maps ScalarTypes to C++ types.
template <exec_aten::ScalarType N>
struct ScalarTypeToCppType;

#define SPECIALIZE_ScalarTypeToCppType(cpp_type, scalar_type)      \
  template <>                                                      \
  struct ScalarTypeToCppType<exec_aten::ScalarType::scalar_type> { \
    using type = cpp_type;                                         \
  };

ET_FORALL_SCALAR_TYPES(SPECIALIZE_ScalarTypeToCppType)

#undef SPECIALIZE_ScalarTypeToCppType

/// Maps C++ types to ScalarTypes.
template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type) \
  template <>                                                 \
  struct CppTypeToScalarType<cpp_type>                        \
      : std::integral_constant<                               \
            exec_aten::ScalarType,                            \
            exec_aten::ScalarType::scalar_type> {};

ET_FORALL_SCALAR_TYPES(SPECIALIZE_CppTypeToScalarType)

#undef SPECIALIZE_CppTypeToScalarType

//
// Macros that iterate across different subsets of ScalarTypes.
//
// See ET_FORALL_SCALAR_TYPES in ScalarType.h to iterate across all ScalarType
// names and types.
//
// For all of these macros, the final `_` parameter is the name of another macro
// that takes two parameters: the name of a C type, and the name of the
// corresponding ScalarType enumerator.
//
// Note that these macros should use fully-qualified namespaces (starting with
// `::`) to ensure that they can be called safely in any arbitrary namespace.
//

// In this context, "INT" means integer C types, which is why the quantized
// integer types are not included.
#define ET_FORALL_INT_TYPES(_) \
  _(uint8_t, Byte)             \
  _(int8_t, Char)              \
  _(int16_t, Short)            \
  _(int32_t, Int)              \
  _(int64_t, Long)

// Here `ANOTHER_INPUT` should be another variable to be forwarded to a given
// function.
#define ET_FORALL_INT_TYPES_WITH(ANOTHER_INPUT, _) \
  _(ANOTHER_INPUT, uint8_t, Byte)                  \
  _(ANOTHER_INPUT, int8_t, Char)                   \
  _(ANOTHER_INPUT, int16_t, Short)                 \
  _(ANOTHER_INPUT, int32_t, Int)                   \
  _(ANOTHER_INPUT, int64_t, Long)

#define ET_FORALL_INT_TYPES_WITH2(ANOTHER_INPUT1, ANOTHER_INPUT2, _) \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, uint8_t, Byte)                   \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, int8_t, Char)                    \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, int16_t, Short)                  \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, int32_t, Int)                    \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, int64_t, Long)

#define ET_FORALL_INT_TYPES_AND(SCALARTYPE, _)      \
  _(uint8_t, Byte)                                  \
  _(int8_t, Char)                                   \
  _(int16_t, Short)                                 \
  _(int32_t, Int)                                   \
  _(int64_t, Long)                                  \
  _(::torch::executor::ScalarTypeToCppType<         \
        ::exec_aten::ScalarType::SCALARTYPE>::type, \
    SCALARTYPE)

// In this context, "FLOAT" means float C types, which is why BFloat16 is not
// included.
#define ET_FORALL_FLOAT_TYPES(_) \
  _(float, Float)                \
  _(double, Double)

// Here `ANOTHER_INPUT` should be another variable to be forwarded to a given
// function. Not to be confused with another scalar type as in
// `ET_FORALL_FLOAT_TYPES_AND`.
#define ET_FORALL_FLOAT_TYPES_WITH(ANOTHER_INPUT, _) \
  _(ANOTHER_INPUT, float, Float)                     \
  _(ANOTHER_INPUT, double, Double)

#define ET_FORALL_FLOAT_TYPES_WITH2(ANOTHER_INPUT1, ANOTHER_INPUT2, _) \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, float, Float)                      \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, double, Double)

// In this context, "REAL" means integer/float C types, which is why BFloat16
// and Half are not included.
#define ET_FORALL_REAL_TYPES(_) \
  _(uint8_t, Byte)              \
  _(int8_t, Char)               \
  _(int16_t, Short)             \
  _(int32_t, Int)               \
  _(int64_t, Long)              \
  _(float, Float)               \
  _(double, Double)

// Here `ANOTHER_INPUT` should be another variable to be forwarded to a given
// function. Not to be confused with another scalar type as in
// `ET_FORALL_REAL_TYPES_AND`.
#define ET_FORALL_REAL_TYPES_WITH(ANOTHER_INPUT, _) \
  _(ANOTHER_INPUT, uint8_t, Byte)                   \
  _(ANOTHER_INPUT, int8_t, Char)                    \
  _(ANOTHER_INPUT, int16_t, Short)                  \
  _(ANOTHER_INPUT, int32_t, Int)                    \
  _(ANOTHER_INPUT, int64_t, Long)                   \
  _(ANOTHER_INPUT, float, Float)                    \
  _(ANOTHER_INPUT, double, Double)

#define ET_FORALL_REAL_TYPES_WITH2(ANOTHER_INPUT1, ANOTHER_INPUT2, _) \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, uint8_t, Byte)                    \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, int8_t, Char)                     \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, int16_t, Short)                   \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, int32_t, Int)                     \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, int64_t, Long)                    \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, float, Float)                     \
  _(ANOTHER_INPUT1, ANOTHER_INPUT2, double, Double)

// For macros that take `SCALARTYPEn` parameters, those parameters should be
// an unquoted/unqualified enumerator name like `Int` or `Float`.
#define ET_FORALL_REAL_TYPES_AND(SCALARTYPE, _)     \
  _(uint8_t, Byte)                                  \
  _(int8_t, Char)                                   \
  _(int16_t, Short)                                 \
  _(int32_t, Int)                                   \
  _(int64_t, Long)                                  \
  _(float, Float)                                   \
  _(double, Double)                                 \
  _(::torch::executor::ScalarTypeToCppType<         \
        ::exec_aten::ScalarType::SCALARTYPE>::type, \
    SCALARTYPE)

#define ET_FORALL_REAL_TYPES_AND_WITH(SCALARTYPE, ANOTHER_INPUT, _) \
  _(ANOTHER_INPUT, uint8_t, Byte)                                   \
  _(ANOTHER_INPUT, int8_t, Char)                                    \
  _(ANOTHER_INPUT, int16_t, Short)                                  \
  _(ANOTHER_INPUT, int32_t, Int)                                    \
  _(ANOTHER_INPUT, int64_t, Long)                                   \
  _(ANOTHER_INPUT, float, Float)                                    \
  _(ANOTHER_INPUT, double, Double)                                  \
  _(ANOTHER_INPUT,                                                  \
    ::torch::executor::ScalarTypeToCppType<                         \
        ::exec_aten::ScalarType::SCALARTYPE>::type,                 \
    SCALARTYPE)

#define ET_FORALL_REAL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, _) \
  _(uint8_t, Byte)                                             \
  _(int8_t, Char)                                              \
  _(int16_t, Short)                                            \
  _(int32_t, Int)                                              \
  _(int64_t, Long)                                             \
  _(float, Float)                                              \
  _(double, Double)                                            \
  _(::torch::executor::ScalarTypeToCppType<                    \
        ::exec_aten::ScalarType::SCALARTYPE1>::type,           \
    SCALARTYPE1)                                               \
  _(::torch::executor::ScalarTypeToCppType<                    \
        ::exec_aten::ScalarType::SCALARTYPE2>::type,           \
    SCALARTYPE2)

#define ET_FORALL_REAL_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, _) \
  _(uint8_t, Byte)                                                          \
  _(int8_t, Char)                                                           \
  _(int16_t, Short)                                                         \
  _(int32_t, Int)                                                           \
  _(int64_t, Long)                                                          \
  _(float, Float)                                                           \
  _(double, Double)                                                         \
  _(::torch::executor::ScalarTypeToCppType<                                 \
        ::exec_aten::ScalarType::SCALARTYPE1>::type,                        \
    SCALARTYPE1)                                                            \
  _(::torch::executor::ScalarTypeToCppType<                                 \
        ::exec_aten::ScalarType::SCALARTYPE2>::type,                        \
    SCALARTYPE2)                                                            \
  _(::torch::executor::ScalarTypeToCppType<                                 \
        ::exec_aten::ScalarType::SCALARTYPE3>::type,                        \
    SCALARTYPE3)

#define ET_FORALL_QINT_TYPES(_)            \
  _(::torch::executor::qint8, QInt8)       \
  _(::torch::executor::quint8, QUInt8)     \
  _(::torch::executor::qint32, QInt32)     \
  _(::torch::executor::quint4x2, QUInt4x2) \
  _(::torch::executor::quint2x4, QUInt2x4)

// In this context, "COMPLEX" means complex types based on primitive C types,
// which is why ComplexHalf is not included.
#define ET_FORALL_COMPLEX_TYPES(_)                   \
  _(::torch::executor::complex<float>, ComplexFloat) \
  _(::torch::executor::complex<double>, ComplexDouble)

//
// Utility functions to retrieve metadata for a given ScalarType
//

/**
 * Returns true if the parameter is one of the values covered by
 * ET_FORALL_SCALAR_TYPES.
 */
inline bool isValid(exec_aten::ScalarType type) {
  return static_cast<int8_t>(type) >= 0 &&
      type < exec_aten::ScalarType::NumOptions &&
      type != exec_aten::ScalarType::Undefined;
}

/**
 * Returns the name of a ScalarType as a C string.
 *
 * @param[in] t The type to get the name of.
 * @return The name of the type, or "UNKNOWN_SCALAR" if the type is not known.
 */
inline const char* toString(exec_aten::ScalarType t) {
#define DEFINE_CASE(_, name)        \
  case exec_aten::ScalarType::name: \
    return #name;

  switch (t) {
    ET_FORALL_SCALAR_TYPES(DEFINE_CASE)
    case exec_aten::ScalarType::Undefined:
      return "Undefined";
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

/**
 * Returns the size in bytes of the C type associated with the ScalarType.
 *
 * Calls ET_CHECK_MSG() if the type is unknown or is ScalarType::Undefined.
 *
 * @param[in] t The type to get the underlying C type size of.
 * @return The size of the associated C type in bytes.
 */
inline size_t elementSize(exec_aten::ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name) \
  case exec_aten::ScalarType::name:        \
    return sizeof(ctype);

  switch (t) {
    ET_FORALL_SCALAR_TYPES(CASE_ELEMENTSIZE_CASE)
    default:
      ET_CHECK_MSG(false, "Unknown ScalarType %" PRId8, static_cast<int8_t>(t));
  }
#undef CASE_ELEMENTSIZE_CASE
}

inline bool isIntegralType(exec_aten::ScalarType t, bool includeBool) {
  bool isIntegral =
      (t == exec_aten::ScalarType::Byte || t == exec_aten::ScalarType::Char ||
       t == exec_aten::ScalarType::Int || t == exec_aten::ScalarType::Long ||
       t == exec_aten::ScalarType::Short);

  return includeBool ? isIntegral || (t == exec_aten::ScalarType::Bool)
                     : isIntegral;
}

inline bool isFloatingType(exec_aten::ScalarType t) {
  return (
      t == exec_aten::ScalarType::Double || t == exec_aten::ScalarType::Float ||
      t == exec_aten::ScalarType::Half || t == exec_aten::ScalarType::BFloat16);
}

inline bool isComplexType(exec_aten::ScalarType t) {
  return (
      t == exec_aten::ScalarType::ComplexHalf ||
      t == exec_aten::ScalarType::ComplexFloat ||
      t == exec_aten::ScalarType::ComplexDouble);
}

inline bool isBitsType(exec_aten::ScalarType t) {
  return t == exec_aten::ScalarType::Bits1x8 ||
      t == exec_aten::ScalarType::Bits2x4 ||
      t == exec_aten::ScalarType::Bits4x2 ||
      t == exec_aten::ScalarType::Bits8 || t == exec_aten::ScalarType::Bits16;
}

inline bool isQIntType(exec_aten::ScalarType t) {
  // Don't forget to extend this when adding new QInt types
  return t == exec_aten::ScalarType::QInt8 ||
      t == exec_aten::ScalarType::QUInt8 ||
      t == exec_aten::ScalarType::QInt32 ||
      t == exec_aten::ScalarType::QUInt4x2 ||
      t == exec_aten::ScalarType::QUInt2x4;
}

inline exec_aten::ScalarType toQIntType(exec_aten::ScalarType t) {
  switch (t) {
    case exec_aten::ScalarType::Byte:
      return exec_aten::ScalarType::QUInt8;
    case exec_aten::ScalarType::Char:
      return exec_aten::ScalarType::QInt8;
    case exec_aten::ScalarType::Int:
      return exec_aten::ScalarType::QInt32;
    default:
      return t;
  }
}

inline exec_aten::ScalarType toUnderlying(exec_aten::ScalarType t) {
  switch (t) {
    case exec_aten::ScalarType::QUInt8:
      return exec_aten::ScalarType::Byte;
    case exec_aten::ScalarType::QInt8:
      return exec_aten::ScalarType::Char;
    case exec_aten::ScalarType::QInt32:
      return exec_aten::ScalarType::Int;
    case exec_aten::ScalarType::QUInt4x2:
      return exec_aten::ScalarType::Byte;
    case exec_aten::ScalarType::QUInt2x4:
      return exec_aten::ScalarType::Byte;
    default:
      return t;
  }
}

inline bool isSignedType(exec_aten::ScalarType t) {
  ET_CHECK_MSG(
      !torch::executor::isQIntType(t),
      "isSignedType not supported for quantized types like %" PRId8,
      static_cast<int8_t>(t));
#define CASE_SIGNED(ctype, name)    \
  case exec_aten::ScalarType::name: \
    return std::numeric_limits<ctype>::is_signed;

  switch (t) {
    case exec_aten::ScalarType::ComplexHalf:
    case exec_aten::ScalarType::ComplexFloat:
    case exec_aten::ScalarType::ComplexDouble:
      return true;
      ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16, CASE_SIGNED)
    default:
      ET_CHECK_MSG(false, "Unknown ScalarType %" PRId8, static_cast<int8_t>(t));
  }
#undef CASE_SIGNED
}

inline bool isUnderlying(
    exec_aten::ScalarType type,
    exec_aten::ScalarType qtype) {
  return type == torch::executor::toUnderlying(qtype);
}

inline exec_aten::ScalarType toRealValueType(exec_aten::ScalarType t) {
  switch (t) {
    case exec_aten::ScalarType::ComplexHalf:
      return exec_aten::ScalarType::Half;
    case exec_aten::ScalarType::ComplexFloat:
      return exec_aten::ScalarType::Float;
    case exec_aten::ScalarType::ComplexDouble:
      return exec_aten::ScalarType::Double;
    default:
      return t;
  }
}

inline exec_aten::ScalarType toComplexType(exec_aten::ScalarType t) {
  switch (t) {
    case exec_aten::ScalarType::BFloat16:
      // BFloat16 has range equivalent to Float,
      // so we map it to ComplexFloat.
      return exec_aten::ScalarType::ComplexFloat;
    case exec_aten::ScalarType::Half:
      return exec_aten::ScalarType::ComplexHalf;
    case exec_aten::ScalarType::Float:
      return exec_aten::ScalarType::ComplexFloat;
    case exec_aten::ScalarType::Double:
      return exec_aten::ScalarType::ComplexDouble;
    case exec_aten::ScalarType::ComplexHalf:
      return exec_aten::ScalarType::ComplexHalf;
    case exec_aten::ScalarType::ComplexFloat:
      return exec_aten::ScalarType::ComplexFloat;
    case exec_aten::ScalarType::ComplexDouble:
      return exec_aten::ScalarType::ComplexDouble;
    default:
      ET_CHECK_MSG(
          false,
          "Unknown Complex ScalarType for %" PRId8,
          static_cast<int8_t>(t));
  }
}

/**
 * Encodes type casting rules that are consistent with ATen behaviour.
 */
inline bool canCast(
    const exec_aten::ScalarType from,
    const exec_aten::ScalarType to) {
  // Disallow complex -> non-complex
  if (torch::executor::isComplexType(from) &&
      !torch::executor::isComplexType(to)) {
    return false;
  }
  // Disallow float -> integral
  if (torch::executor::isFloatingType(from) &&
      torch::executor::isIntegralType(to, /*includeBool*/ false)) {
    return false;
  }

  // Treat bool as a special category. Disallow non-bool -> bool
  if (from != exec_aten::ScalarType::Bool &&
      to == exec_aten::ScalarType::Bool) {
    return false;
  }

  return true;
}

/**
 * When casting from floating point to integral type, if the floating value is
 * outside the integral type range, then an error is thrown if sanitization is
 * enabled. To circumvent this, we cast the floating point to int64_t first.
 */
template <
    typename To,
    typename From,
    typename std::enable_if<
        (std::is_floating_point<From>::value && std::is_integral<To>::value),
        int>::type = 0>
To convert(From val) {
  return static_cast<To>(static_cast<int64_t>(val));
}

template <
    typename To,
    typename From,
    typename std::enable_if<
        !(std::is_floating_point<From>::value && std::is_integral<To>::value),
        int>::type = 0>
To convert(From val) {
  return static_cast<To>(val);
}

/**
 * Implements type promotion rules that are consistent with ATen behaviour,
 * which in turn is consistent with NumPy's promote_types.
 * If half_to_float is set to true, then half will be promoted to float instead
 */
inline exec_aten::ScalarType promoteTypes(
    exec_aten::ScalarType a,
    exec_aten::ScalarType b,
    bool half_to_float = false) {
  // This is generated according to NumPy's promote_types
  constexpr auto u1 = exec_aten::ScalarType::Byte;
  constexpr auto i1 = exec_aten::ScalarType::Char;
  constexpr auto i2 = exec_aten::ScalarType::Short;
  constexpr auto i4 = exec_aten::ScalarType::Int;
  constexpr auto i8 = exec_aten::ScalarType::Long;
  constexpr auto f2 = exec_aten::ScalarType::Half;
  constexpr auto f4 = exec_aten::ScalarType::Float;
  constexpr auto f8 = exec_aten::ScalarType::Double;
  constexpr auto c2 = exec_aten::ScalarType::ComplexHalf;
  constexpr auto c4 = exec_aten::ScalarType::ComplexFloat;
  constexpr auto c8 = exec_aten::ScalarType::ComplexDouble;
  constexpr auto b1 = exec_aten::ScalarType::Bool;

  // For QInt types, only allow exact match
  if (torch::executor::isQIntType(a) && a == b) {
    return a;
  }
  if (torch::executor::isQIntType(a) || torch::executor::isQIntType(b)) {
    ET_CHECK_MSG(false, "promoteTypes not valid for quantized dtypes");
  }

  // For Bits types, only allow exact match
  if (torch::executor::isBitsType(a) && a == b) {
    return a;
  }
  if (torch::executor::isBitsType(a) || torch::executor::isBitsType(b)) {
    ET_CHECK_MSG(false, "promoteTypes not valid for bits dtypes");
  }

  // 12 types are handled by this function, see the constexpr definitions above
  const int NUM_PROMOTE_TYPES = 12;

  static constexpr exec_aten::ScalarType
      _promoteTypesLookup[NUM_PROMOTE_TYPES][NUM_PROMOTE_TYPES] = {
          /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  */
          /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, u1},
          /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, i1},
          /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, i2},
          /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c2, c4, c8, i4},
          /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c2, c4, c8, i8},
          /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c2, c4, c8, f2},
          /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, c8, f4},
          /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, c8, f8},
          /* c2 */ {c2, c2, c2, c2, c2, c2, c4, c8, c2, c4, c8, c2},
          /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4},
          /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
          /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1},
      };

  exec_aten::ScalarType promoted_type =
      _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];

  if (half_to_float && promoted_type == ScalarType::Half) {
    promoted_type = ScalarType::Float;
  }

  return promoted_type;
}

//
// Helper macros for switch case macros (see below)
//
// These macros are not meant to be used directly. They provide an easy way to
// generate a switch statement that can handle subsets of ScalarTypes supported
// by ExecuTorch.
//

#ifdef ET_INTERNAL_CHECK_SELECTIVE_BUILD
#define ET_INTERNAL_SWITCH_CASE(enum_type, CTYPE_ALIAS, ...)  \
  case enum_type: {                                           \
    ET_INTERNAL_CHECK_SELECTIVE_BUILD(enum_type);             \
    using CTYPE_ALIAS = ScalarTypeToCppType<enum_type>::type; \
    return __VA_ARGS__();                                     \
  }
#else
#define ET_INTERNAL_SWITCH_CASE(enum_type, CTYPE_ALIAS, ...)  \
  case enum_type: {                                           \
    using CTYPE_ALIAS = ScalarTypeToCppType<enum_type>::type; \
    return __VA_ARGS__();                                     \
  }
#endif

#define ET_INTERNAL_SWITCH(TYPE, CONTEXT, NAME, ...) \
  [&] {                                              \
    const auto& _st = TYPE;                          \
    constexpr const char* et_switch_name = NAME;     \
    switch (_st) {                                   \
      __VA_ARGS__                                    \
      default:                                       \
        ET_CHECK_MSG(                                \
            false,                                   \
            "Unhandled dtype %s for %s",             \
            torch::executor::toString(_st),          \
            et_switch_name);                         \
    }                                                \
  }()

#define ET_INTERNAL_SWITCH_CASE_ALL_TYPES(CTYPE_ALIAS, ...)           \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Byte, CTYPE_ALIAS, __VA_ARGS__)          \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Char, CTYPE_ALIAS, __VA_ARGS__)          \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Short, CTYPE_ALIAS, __VA_ARGS__)         \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Int, CTYPE_ALIAS, __VA_ARGS__)           \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Long, CTYPE_ALIAS, __VA_ARGS__)          \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Half, CTYPE_ALIAS, __VA_ARGS__)          \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Float, CTYPE_ALIAS, __VA_ARGS__)         \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Double, CTYPE_ALIAS, __VA_ARGS__)        \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::ComplexHalf, CTYPE_ALIAS, __VA_ARGS__)   \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::ComplexFloat, CTYPE_ALIAS, __VA_ARGS__)  \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::ComplexDouble, CTYPE_ALIAS, __VA_ARGS__) \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Bool, CTYPE_ALIAS, __VA_ARGS__)          \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::QInt8, CTYPE_ALIAS, __VA_ARGS__)         \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::QUInt8, CTYPE_ALIAS, __VA_ARGS__)        \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::QInt32, CTYPE_ALIAS, __VA_ARGS__)        \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::BFloat16, CTYPE_ALIAS, __VA_ARGS__)      \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::QUInt4x2, CTYPE_ALIAS, __VA_ARGS__)      \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::QUInt2x4, CTYPE_ALIAS, __VA_ARGS__)      \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Bits1x8, CTYPE_ALIAS, __VA_ARGS__)       \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Bits2x4, CTYPE_ALIAS, __VA_ARGS__)       \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Bits4x2, CTYPE_ALIAS, __VA_ARGS__)       \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Bits8, CTYPE_ALIAS, __VA_ARGS__)         \
  ET_INTERNAL_SWITCH_CASE(                                            \
      exec_aten::ScalarType::Bits16, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_REAL_TYPES(CTYPE_ALIAS, ...)  \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Byte, CTYPE_ALIAS, __VA_ARGS__)  \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Char, CTYPE_ALIAS, __VA_ARGS__)  \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Short, CTYPE_ALIAS, __VA_ARGS__) \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Int, CTYPE_ALIAS, __VA_ARGS__)   \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Long, CTYPE_ALIAS, __VA_ARGS__)  \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Float, CTYPE_ALIAS, __VA_ARGS__) \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Double, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_REAL_TYPES_AND(ADDITIONAL, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH_CASE_REAL_TYPES(CTYPE_ALIAS, __VA_ARGS__)               \
  ET_INTERNAL_SWITCH_CASE(                                                   \
      exec_aten::ScalarType::ADDITIONAL, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_REAL_TYPES_AND2(                    \
    ADDITIONAL1, ADDITIONAL2, CTYPE_ALIAS, ...)                     \
  ET_INTERNAL_SWITCH_CASE_REAL_TYPES(CTYPE_ALIAS, __VA_ARGS__)      \
  ET_INTERNAL_SWITCH_CASE(                                          \
      exec_aten::ScalarType::ADDITIONAL1, CTYPE_ALIAS, __VA_ARGS__) \
  ET_INTERNAL_SWITCH_CASE(                                          \
      exec_aten::ScalarType::ADDITIONAL2, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_INT_TYPES(CTYPE_ALIAS, ...)   \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Byte, CTYPE_ALIAS, __VA_ARGS__)  \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Char, CTYPE_ALIAS, __VA_ARGS__)  \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Short, CTYPE_ALIAS, __VA_ARGS__) \
  ET_INTERNAL_SWITCH_CASE(                                    \
      exec_aten::ScalarType::Int, CTYPE_ALIAS, __VA_ARGS__)   \
  ET_INTERNAL_SWITCH_CASE(exec_aten::ScalarType::Long, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_INT_TYPES_AND(ADDITIONAL, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH_CASE_INT_TYPES(CTYPE_ALIAS, __VA_ARGS__)               \
  ET_INTERNAL_SWITCH_CASE(                                                  \
      exec_aten::ScalarType::ADDITIONAL, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_FLOAT_TYPES(CTYPE_ALIAS, ...)  \
  ET_INTERNAL_SWITCH_CASE(                                     \
      exec_aten::ScalarType::Double, CTYPE_ALIAS, __VA_ARGS__) \
  ET_INTERNAL_SWITCH_CASE(                                     \
      exec_aten::ScalarType::Float, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_FLOAT_TYPES_AND(ADDITIONAL, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH_CASE_FLOAT_TYPES(CTYPE_ALIAS, __VA_ARGS__)               \
  ET_INTERNAL_SWITCH_CASE(                                                    \
      exec_aten::ScalarType::ADDITIONAL, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_QINT_TYPES(CTYPE_ALIAS, ...)     \
  ET_INTERNAL_SWITCH_CASE(                                       \
      exec_aten::ScalarType::QInt8, CTYPE_ALIAS, __VA_ARGS__)    \
  ET_INTERNAL_SWITCH_CASE(                                       \
      exec_aten::ScalarType::QUInt8, CTYPE_ALIAS, __VA_ARGS__)   \
  ET_INTERNAL_SWITCH_CASE(                                       \
      exec_aten::ScalarType::QInt32, CTYPE_ALIAS, __VA_ARGS__)   \
  ET_INTERNAL_SWITCH_CASE(                                       \
      exec_aten::ScalarType::QUInt4x2, CTYPE_ALIAS, __VA_ARGS__) \
  ET_INTERNAL_SWITCH_CASE(                                       \
      exec_aten::ScalarType::QUInt2x4, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_COMPLEX_TYPES(CTYPE_ALIAS, ...)      \
  ET_INTERNAL_SWITCH_CASE(                                           \
      exec_aten::ScalarType::ComplexFloat, CTYPE_ALIAS, __VA_ARGS__) \
  ET_INTERNAL_SWITCH_CASE(                                           \
      exec_aten::ScalarType::ComplexDouble, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_SCALAR_OBJ_TYPES(CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH_CASE(                                         \
      exec_aten::ScalarType::Bool, CTYPE_ALIAS, __VA_ARGS__)       \
  ET_INTERNAL_SWITCH_CASE(                                         \
      exec_aten::ScalarType::Long, CTYPE_ALIAS, __VA_ARGS__)       \
  ET_INTERNAL_SWITCH_CASE(                                         \
      exec_aten::ScalarType::Double, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_SCALAR_OBJ_REAL_TYPES(CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH_CASE(                                              \
      exec_aten::ScalarType::Long, CTYPE_ALIAS, __VA_ARGS__)            \
  ET_INTERNAL_SWITCH_CASE(                                              \
      exec_aten::ScalarType::Double, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_SCALAR_OBJ_INTB_TYPES(CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH_CASE(                                              \
      exec_aten::ScalarType::Bool, CTYPE_ALIAS, __VA_ARGS__)            \
  ET_INTERNAL_SWITCH_CASE(exec_aten::ScalarType::Long, CTYPE_ALIAS, __VA_ARGS__)

#define ET_INTERNAL_SWITCH_CASE_SCALAR_OBJ_FLOATB_TYPES(CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH_CASE(                                                \
      exec_aten::ScalarType::Bool, CTYPE_ALIAS, __VA_ARGS__)              \
  ET_INTERNAL_SWITCH_CASE(                                                \
      exec_aten::ScalarType::Double, CTYPE_ALIAS, __VA_ARGS__)

//
// Switch case macros
//
// These macros provide an easy way to generate switch statements that apply a
// common lambda function to subsets of ScalarTypes supported by ExecuTorch.
// The lambda function can type specialize to the ctype associated with the
// ScalarType being handled through an alias passed as the CTYPE_ALIAS argument.
//
// Arguments:
//   - ADDITIONAL: Additional ScalarType case to add
//   - TYPE: The ScalarType to handle through the switch statement
//   - CONTEXT: The RuntimeContext instance used for error handling, etc.
//   - NAME: A name for this operation which will be used in error messages
//   - CTYPE_ALIAS: A typedef for the ctype associated with the ScalarType.
//   - [&](){...}: A lambda function to be applied to each ScalarType case
//
// An example usage is:
//
// ET_SWITCH_REAL_TYPES(input.scalar_type(), "example", CTYPE, [&]() {
//   output.mutable_data_ptr<CTYPE>[0] = input.const_data_ptr<CTYPE>[0];
// });
//
// Note that these can be nested as well:
//
// ET_SWITCH_REAL_TYPES(input.scalar_type(), "example", CTYPE_IN, [&]() {
//   ET_SWITCH_REAL_TYPES(output.scalar_type(), "example", CTYPE_OUT, [&]() {
//     output.mutable_data_ptr<CTYPE_OUT>[0] =
//         input.const_data_ptr<CTYPE_IN>[0];
//   });
// });
//
// These macros are adapted from Dispatch.h in the ATen library. The primary
// difference is that the CTYPE_ALIAS argument is exposed to users, which is
// used to alias the ctype associated with the ScalarType that is being handled.
//

#define ET_SWITCH_ALL_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                              \
      TYPE,                                                        \
      CONTEXT,                                                     \
      NAME,                                                        \
      ET_INTERNAL_SWITCH_CASE_ALL_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_REAL_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                               \
      TYPE,                                                         \
      CONTEXT,                                                      \
      NAME,                                                         \
      ET_INTERNAL_SWITCH_CASE_REAL_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_REAL_TYPES_AND(                      \
    ADDITIONAL, TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                  \
      TYPE,                                            \
      CONTEXT,                                         \
      NAME,                                            \
      ET_INTERNAL_SWITCH_CASE_REAL_TYPES_AND(          \
          ADDITIONAL, CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_REAL_TYPES_AND2(                                   \
    ADDITIONAL1, ADDITIONAL2, TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                                \
      TYPE,                                                          \
      CONTEXT,                                                       \
      NAME,                                                          \
      ET_INTERNAL_SWITCH_CASE_REAL_TYPES_AND2(                       \
          ADDITIONAL1, ADDITIONAL2, CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_REALH_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_SWITCH_REAL_TYPES_AND(Half, TYPE, CONTEXT, NAME, CTYPE_ALIAS, __VA_ARGS__)

#define ET_SWITCH_REALB_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_SWITCH_REAL_TYPES_AND(Bool, TYPE, CONTEXT, NAME, CTYPE_ALIAS, __VA_ARGS__)

#define ET_SWITCH_REALHB_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_SWITCH_REAL_TYPES_AND2(                                          \
      Half, Bool, TYPE, CONTEXT, NAME, CTYPE_ALIAS, __VA_ARGS__)

#define ET_SWITCH_INT_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                              \
      TYPE,                                                        \
      CONTEXT,                                                     \
      NAME,                                                        \
      ET_INTERNAL_SWITCH_CASE_INT_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_INT_TYPES_AND(                       \
    ADDITIONAL, TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                  \
      TYPE,                                            \
      CONTEXT,                                         \
      NAME,                                            \
      ET_INTERNAL_SWITCH_CASE_INT_TYPES_AND(           \
          ADDITIONAL, CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_FLOAT_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                                \
      TYPE,                                                          \
      CONTEXT,                                                       \
      NAME,                                                          \
      ET_INTERNAL_SWITCH_CASE_FLOAT_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_FLOAT_TYPES_AND(                     \
    ADDITIONAL, TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                  \
      TYPE,                                            \
      CONTEXT,                                         \
      NAME,                                            \
      ET_INTERNAL_SWITCH_CASE_FLOAT_TYPES_AND(         \
          ADDITIONAL, CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_FLOATH_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_SWITCH_FLOAT_TYPES_AND(Half, TYPE, CONTEXT, NAME, CTYPE_ALIAS, __VA_ARGS__)

#define ET_SWITCH_QINT_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                               \
      TYPE,                                                         \
      CONTEXT,                                                      \
      NAME,                                                         \
      ET_INTERNAL_SWITCH_CASE_QINT_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_COMPLEX_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                                  \
      TYPE,                                                            \
      CONTEXT,                                                         \
      NAME,                                                            \
      ET_INTERNAL_SWITCH_CASE_COMPLEX_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_SCALAR_OBJ_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                                     \
      TYPE,                                                               \
      CONTEXT,                                                            \
      NAME,                                                               \
      ET_INTERNAL_SWITCH_CASE_SCALAR_OBJ_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_SCALAR_OBJ_REAL_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                                          \
      TYPE,                                                                    \
      CONTEXT,                                                                 \
      NAME,                                                                    \
      ET_INTERNAL_SWITCH_CASE_SCALAR_OBJ_REAL_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_SCALAR_OBJ_INTB_TYPES(TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                                          \
      TYPE,                                                                    \
      CONTEXT,                                                                 \
      NAME,                                                                    \
      ET_INTERNAL_SWITCH_CASE_SCALAR_OBJ_INTB_TYPES(CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_SCALAR_OBJ_FLOATB_TYPES(             \
    TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...)             \
  ET_INTERNAL_SWITCH(                                  \
      TYPE,                                            \
      CONTEXT,                                         \
      NAME,                                            \
      ET_INTERNAL_SWITCH_CASE_SCALAR_OBJ_FLOATB_TYPES( \
          CTYPE_ALIAS, __VA_ARGS__))

#define ET_SWITCH_TWO_TYPES(T1, T2, TYPE, CONTEXT, NAME, CTYPE_ALIAS, ...) \
  ET_INTERNAL_SWITCH(                                                      \
      TYPE,                                                                \
      CONTEXT,                                                             \
      NAME,                                                                \
      ET_INTERNAL_SWITCH_CASE(                                             \
          exec_aten::ScalarType::T1, CTYPE_ALIAS, __VA_ARGS__)             \
          ET_INTERNAL_SWITCH_CASE(                                         \
              exec_aten::ScalarType::T2, CTYPE_ALIAS, __VA_ARGS__))

} // namespace executor
} // namespace torch
