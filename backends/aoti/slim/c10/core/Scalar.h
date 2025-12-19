#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/c10/macros/Macros.h>
#include <executorch/backends/aoti/slim/c10/util/Half.h>
#include <executorch/backends/aoti/slim/c10/util/TypeCast.h>
#include <executorch/backends/aoti/slim/c10/util/complex.h>
#include <executorch/backends/aoti/slim/c10/util/overflows.h>
#include <executorch/runtime/platform/assert.h>

// Copy-pasted from c10/core/Scalar.h, but dropping SymScalar support

namespace executorch::backends::aoti::slim::c10 {

/**
 * Scalar represents a 0-dimensional tensor which contains a single element.
 * Unlike a tensor, numeric literals (in C++) are implicitly convertible to
 * Scalar (which is why, for example, we provide both add(Tensor) and
 * add(Scalar) overloads for many operations). It may also be used in
 * circumstances where you statically know a tensor is 0-dim and single size,
 * but don't know its type.
 */
class Scalar {
 public:
  Scalar() : Scalar(int64_t(0)) {}

#define DEFINE_IMPLICIT_CTOR(type, name) \
  Scalar(type vv) : Scalar(vv, true) {}

  AT_FORALL_SCALAR_TYPES_AND3(Half, BFloat16, ComplexHalf, DEFINE_IMPLICIT_CTOR)
  AT_FORALL_COMPLEX_TYPES(DEFINE_IMPLICIT_CTOR)
  AT_FORALL_FLOAT8_TYPES(DEFINE_IMPLICIT_CTOR)

  // Helper constructors to allow Scalar creation from long and long long types
  // As std::is_same_v<long, long long> is false(except Android), one needs to
  // provide a constructor from either long or long long in addition to one from
  // int64_t
#if defined(__APPLE__) || defined(__MACOSX)
  static_assert(
      std::is_same_v<long long, int64_t>,
      "int64_t is the same as long long on MacOS");
  Scalar(long vv) : Scalar(vv, true) {}
#endif
#if defined(_MSC_VER)
  static_assert(
      std::is_same_v<long long, int64_t>,
      "int64_t is the same as long long on Windows");
  Scalar(long vv) : Scalar(vv, true) {}
#endif
#if defined(__linux__) && !defined(__ANDROID__)
  static_assert(
      sizeof(void*) != 8 || std::is_same_v<long, int64_t>,
      "int64_t is the same as long on 64 bit Linux");
#if LONG_MAX != INT_MAX
  Scalar(long long vv) : Scalar(vv, true) {}
#endif /* not 32-bit system */
#endif

  Scalar(uint16_t vv) : Scalar(vv, true) {}
  Scalar(uint32_t vv) : Scalar(vv, true) {}
  Scalar(uint64_t vv) {
    if (vv > static_cast<uint64_t>(INT64_MAX)) {
      tag = Tag::HAS_u;
      v.u = vv;
    } else {
      tag = Tag::HAS_i;
      // NB: no need to use convert, we've already tested convertibility
      v.i = static_cast<int64_t>(vv);
    }
  }

#undef DEFINE_IMPLICIT_CTOR

  // Value* is both implicitly convertible to SymbolicVariable and bool which
  // causes ambiguity error. Specialized constructor for bool resolves this
  // problem.
  template <
      typename T,
      typename std::enable_if_t<std::is_same_v<T, bool>, bool>* = nullptr>
  Scalar(T vv) : tag(Tag::HAS_b) {
    v.i = convert<int64_t, bool>(vv);
  }

#define DEFINE_ACCESSOR(type, name)                                            \
  type to##name() const {                                                      \
    if (Tag::HAS_d == tag) {                                                   \
      return checked_convert<type, double>(v.d, #type);                        \
    } else if (Tag::HAS_z == tag) {                                            \
      return checked_convert<                                                  \
          type,                                                                \
          executorch::backends::aoti::slim::c10::complex<double>>(v.z, #type); \
    }                                                                          \
    if (Tag::HAS_b == tag) {                                                   \
      return checked_convert<type, bool>(v.i, #type);                          \
    } else if (Tag::HAS_i == tag) {                                            \
      return checked_convert<type, int64_t>(v.i, #type);                       \
    } else if (Tag::HAS_u == tag) {                                            \
      return checked_convert<type, uint64_t>(v.u, #type);                      \
    }                                                                          \
    ET_CHECK_MSG(false, "Unknown Scalar tag");                                 \
  }

  // TODO: Support ComplexHalf accessor
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ACCESSOR)
  DEFINE_ACCESSOR(uint16_t, UInt16)
  DEFINE_ACCESSOR(uint32_t, UInt32)
  DEFINE_ACCESSOR(uint64_t, UInt64)

#undef DEFINE_ACCESSOR

  // also support scalar.to<int64_t>();
  // Deleted for unsupported types, but specialized below for supported types
  template <typename T>
  T to() const = delete;

  // audit uses of data_ptr
  const void* data_ptr() const {
    return static_cast<const void*>(&v);
  }

  bool isFloatingPoint() const {
    return Tag::HAS_d == tag;
  }

  bool isIntegral(bool includeBool) const {
    return Tag::HAS_i == tag || Tag::HAS_u == tag ||
        (includeBool && isBoolean());
  }

  bool isComplex() const {
    return Tag::HAS_z == tag;
  }
  bool isBoolean() const {
    return Tag::HAS_b == tag;
  }

  STANDALONE_ALWAYS_INLINE Scalar& operator=(Scalar&& other) noexcept {
    if (&other == this) {
      return *this;
    }

    moveFrom(std::move(other));
    return *this;
  }

  STANDALONE_ALWAYS_INLINE Scalar& operator=(const Scalar& other) {
    if (&other == this) {
      return *this;
    }

    *this = Scalar(other);
    return *this;
  }

  Scalar operator-() const {
    ET_CHECK_MSG(
        !isBoolean(),
        "torch boolean negative, the `-` operator, is not supported");
    if (isFloatingPoint()) {
      return Scalar(-v.d);
    } else if (isComplex()) {
      return Scalar(-v.z);
    } else if (isIntegral(false)) {
      return Scalar(-v.i);
    }
    ET_CHECK_MSG(false, "unknown ivalue tag");
  }

  Scalar conj() const {
    if (isComplex()) {
      return Scalar(std::conj(v.z));
    } else {
      return *this;
    }
  }

  Scalar log() const {
    if (isComplex()) {
      return std::log(v.z);
    } else if (isFloatingPoint()) {
      return std::log(v.d);
    } else if (isIntegral(false)) {
      return std::log(v.i);
    }
    ET_CHECK_MSG(false, "unknown ivalue tag");
  }

  template <
      typename T,
      typename std::enable_if_t<
          !executorch::backends::aoti::slim::c10::is_complex<T>::value,
          int> = 0>
  bool equal(T num) const {
    if (isComplex()) {
      auto val = v.z;
      return (val.real() == num) && (val.imag() == T());
    } else if (isFloatingPoint()) {
      return toDouble() == num;
    } else if (tag == Tag::HAS_i) {
      if (overflows<T>(v.i, /* strict_unsigned */ true)) {
        return false;
      } else {
        return static_cast<T>(v.i) == num;
      }
    } else if (tag == Tag::HAS_u) {
      if (overflows<T>(v.u, /* strict_unsigned */ true)) {
        return false;
      } else {
        return static_cast<T>(v.u) == num;
      }
    } else if (isBoolean()) {
      // boolean scalar does not equal to a non boolean value
      return false;
    } else {
      ET_CHECK_MSG(false, "unexpected tag in equal");
    }
  }

  template <
      typename T,
      typename std::enable_if_t<
          executorch::backends::aoti::slim::c10::is_complex<T>::value,
          int> = 0>
  bool equal(T num) const {
    if (isComplex()) {
      return v.z == num;
    } else if (isFloatingPoint()) {
      return (toDouble() == num.real()) && (num.imag() == T());
    } else if (tag == Tag::HAS_i) {
      if (overflows<T>(v.i, /* strict_unsigned */ true)) {
        return false;
      } else {
        return static_cast<T>(v.i) == num.real() && num.imag() == T();
      }
    } else if (tag == Tag::HAS_u) {
      if (overflows<T>(v.u, /* strict_unsigned */ true)) {
        return false;
      } else {
        return static_cast<T>(v.u) == num.real() && num.imag() == T();
      }
    } else if (isBoolean()) {
      // boolean scalar does not equal to a non boolean value
      return false;
    } else {
      ET_CHECK_MSG(false, "unexpected tag in equal");
    }
  }

  bool equal(bool num) const {
    if (isBoolean()) {
      return static_cast<bool>(v.i) == num;
    } else {
      return false;
    }
  }

  executorch::backends::aoti::slim::c10::ScalarType type() const {
    if (isComplex()) {
      return executorch::backends::aoti::slim::c10::ScalarType::ComplexDouble;
    } else if (isFloatingPoint()) {
      return executorch::backends::aoti::slim::c10::ScalarType::Double;
    } else if (isIntegral(/*includeBool=*/false)) {
      // Represent all integers as long, UNLESS it is unsigned and therefore
      // unrepresentable as long
      if (Tag::HAS_u == tag) {
        return executorch::backends::aoti::slim::c10::ScalarType::UInt64;
      }
      return executorch::backends::aoti::slim::c10::ScalarType::Long;
    } else if (isBoolean()) {
      return executorch::backends::aoti::slim::c10::ScalarType::Bool;
    } else {
      ET_CHECK_MSG(false, "Unknown scalar type.");
    }
  }

  Scalar(Scalar&& rhs) noexcept : tag(rhs.tag) {
    moveFrom(std::move(rhs));
  }

  Scalar(const Scalar& rhs) : tag(rhs.tag), v(rhs.v) {}

  // We can't set v in the initializer list using the
  // syntax v{ .member = ... } because it doesn't work on MSVC
 private:
  enum class Tag { HAS_d, HAS_i, HAS_u, HAS_z, HAS_b };

  // Note [Meaning of HAS_u]
  // ~~~~~~~~~~~~~~~~~~~~~~~
  // HAS_u is a bit special.  On its face, it just means that we
  // are holding an unsigned integer.  However, we generally don't
  // distinguish between different bit sizes in Scalar (e.g., we represent
  // float as double), instead, it represents a mathematical notion
  // of some quantity (integral versus floating point).  So actually,
  // HAS_u is used solely to represent unsigned integers that could
  // not be represented as a signed integer.  That means only uint64_t
  // potentially can get this tag; smaller types like uint8_t fits into a
  // regular int and so for BC reasons we keep as an int.

  // NB: assumes that self has already been cleared
  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  STANDALONE_ALWAYS_INLINE void moveFrom(Scalar&& rhs) noexcept {
    v = rhs.v;
    tag = rhs.tag;
  }

  Tag tag;

  union v_t {
    double d{};
    int64_t i;
    // See Note [Meaning of HAS_u]
    uint64_t u;
    executorch::backends::aoti::slim::c10::complex<double> z;
    // NOLINTNEXTLINE(modernize-use-equals-default)
    v_t() {} // default constructor
  } v;

  template <
      typename T,
      typename std::enable_if_t<
          std::is_integral_v<T> && !std::is_same_v<T, bool>,
          bool>* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_i) {
    v.i = convert<decltype(v.i), T>(vv);
  }

  template <
      typename T,
      typename std::enable_if_t<
          !std::is_integral_v<T> &&
              !executorch::backends::aoti::slim::c10::is_complex<T>::value,
          bool>* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_d) {
    v.d = convert<decltype(v.d), T>(vv);
  }

  template <
      typename T,
      typename std::enable_if_t<
          executorch::backends::aoti::slim::c10::is_complex<T>::value,
          bool>* = nullptr>
  Scalar(T vv, bool) : tag(Tag::HAS_z) {
    v.z = convert<decltype(v.z), T>(vv);
  }
};

// define the scalar.to<int64_t>() specializations
#define DEFINE_TO(T, name)         \
  template <>                      \
  inline T Scalar::to<T>() const { \
    return to##name();             \
  }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_TO)
DEFINE_TO(uint16_t, UInt16)
DEFINE_TO(uint32_t, UInt32)
DEFINE_TO(uint64_t, UInt64)
#undef DEFINE_TO

} // namespace executorch::backends::aoti::slim::c10
