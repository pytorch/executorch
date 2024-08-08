/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/backends/vulkan/runtime/vk_api/Exception.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>

namespace vkcompute {
namespace utils {

//
// Hashing
//

/**
 * hash_combine is taken from c10/util/hash.h, which in turn is based on
 * implementation from Boost
 */
inline size_t hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}

//
// Alignment
//

template <typename Type>
inline constexpr Type align_down(const Type& number, const Type& multiple) {
  return (number / multiple) * multiple;
}

template <typename Type>
inline constexpr Type align_up(const Type& number, const Type& multiple) {
  return align_down(number + multiple - 1, multiple);
}

template <typename Type>
inline constexpr Type align_up_4(const Type& numerator) {
  return (numerator + 3) & -4;
}

template <typename Type>
inline constexpr Type div_up(const Type& numerator, const Type& denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename Type>
inline constexpr Type div_up_4(const Type& numerator) {
  return (numerator + 3) / 4;
}

//
// Casting Utilities
//

namespace detail {

/*
 * x cannot be less than 0 if x is unsigned
 */
template <typename T>
static inline constexpr bool is_negative(
    const T& /*x*/,
    std::true_type /*is_unsigned*/) {
  return false;
}

/*
 * check if x is less than 0 if x is signed
 */
template <typename T>
static inline constexpr bool is_negative(
    const T& x,
    std::false_type /*is_unsigned*/) {
  return x < T(0);
}

/*
 * Returns true if x < 0
 */
template <typename T>
inline constexpr bool is_negative(const T& x) {
  return is_negative(x, std::is_unsigned<T>());
}

/*
 * Returns true if x < lowest(Limit); standard comparison
 */
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& x,
    std::false_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  return x < std::numeric_limits<Limit>::lowest();
}

/*
 * Limit can contained negative values, but x cannot; return false
 */
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& /*x*/,
    std::false_type /*limit_is_unsigned*/,
    std::true_type /*x_is_unsigned*/) {
  return false;
}

/*
 * Limit cannot contained negative values, but x can; check if x is negative
 */
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& x,
    std::true_type /*limit_is_unsigned*/,
    std::false_type /*x_is_unsigned*/) {
  return x < T(0);
}

/*
 * Both x and Limit cannot be negative; return false
 */
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(
    const T& /*x*/,
    std::true_type /*limit_is_unsigned*/,
    std::true_type /*x_is_unsigned*/) {
  return false;
}

/*
 * Returns true if x is less than the lowest value of type T
 */
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(const T& x) {
  return less_than_lowest<Limit>(
      x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

// Suppress sign compare warning when compiling with GCC
// as later does not account for short-circuit rule before
// raising the warning, see https://godbolt.org/z/Tr3Msnz99
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

/*
 * Returns true if x is greater than the greatest value of the type Limit
 */
template <typename Limit, typename T>
inline constexpr bool greater_than_max(const T& x) {
  constexpr bool can_overflow =
      std::numeric_limits<T>::digits > std::numeric_limits<Limit>::digits;
  return can_overflow && x > std::numeric_limits<Limit>::max();
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

template <typename To, typename From>
std::enable_if_t<
    std::is_integral<From>::value && !std::is_same<From, bool>::value,
    bool>
overflows(From f) {
  using limit = std::numeric_limits<To>;
  // Casting from signed to unsigned; allow for negative numbers to wrap using
  // two's complement arithmetic.
  if (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    return greater_than_max<To>(f) ||
        (is_negative(f) && -static_cast<uint64_t>(f) > limit::max());
  }
  // standard case, check if f is outside the range of type To
  else {
    return less_than_lowest<To>(f) || greater_than_max<To>(f);
  }
}

template <typename To, typename From>
std::enable_if_t<std::is_floating_point<From>::value, bool> overflows(From f) {
  using limit = std::numeric_limits<To>;
  if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
    return false;
  }
  return f < limit::lowest() || f > limit::max();
}

template <typename To, typename From>
inline constexpr To safe_downcast(const From& v) {
  VK_CHECK_COND(!overflows<To>(v), "Cast failed: out of range!");
  return static_cast<To>(v);
}

template <typename To, typename From>
inline constexpr bool is_signed_to_unsigned() {
  return std::is_signed<From>::value && std::is_unsigned<To>::value;
}

} // namespace detail

template <
    typename To,
    typename From,
    std::enable_if_t<detail::is_signed_to_unsigned<To, From>(), bool> = true>
inline constexpr To safe_downcast(const From& v) {
  VK_CHECK_COND(v >= From{}, "Cast failed: negative signed to unsigned!");
  return detail::safe_downcast<To, From>(v);
}

template <
    typename To,
    typename From,
    std::enable_if_t<!detail::is_signed_to_unsigned<To, From>(), bool> = true>
inline constexpr To safe_downcast(const From& v) {
  return detail::safe_downcast<To, From>(v);
}

//
// Vector Types
//

namespace detail {

template <typename Type, uint32_t N>
struct vec final {
  // NOLINTNEXTLINE
  Type data[N];

  const Type& operator[](const uint32_t& i) const {
    VK_CHECK_COND(i >= 0 && i < N, "Index out of bounds!");
    return data[i];
  }

  Type& operator[](const uint32_t& i) {
    VK_CHECK_COND(i >= 0 && i < N, "Index out of bounds!");
    return data[i];
  }
};

} // namespace detail

template <uint32_t N>
using ivec = detail::vec<int32_t, N>;
using ivec2 = ivec<2u>;
using ivec3 = ivec<3u>;
using ivec4 = ivec<4u>;

template <uint32_t N>
using uvec = detail::vec<uint32_t, N>;
using uvec2 = uvec<2u>;
using uvec3 = uvec<3u>;
using uvec4 = uvec<4u>;

template <uint32_t N>
using vec = detail::vec<float, N>;
using vec2 = vec<2u>;
using vec3 = vec<3u>;
using vec4 = vec<4u>;

// uvec3 is the type representing tensor extents. Useful for debugging.
inline std::ostream& operator<<(std::ostream& os, const uvec3& v) {
  os << "(" << v[0u] << ", " << v[1u] << ", " << v[2u] << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const ivec3& v) {
  os << "(" << v[0u] << ", " << v[1u] << ", " << v[2u] << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const uvec4& v) {
  os << "(" << v[0u] << ", " << v[1u] << ", " << v[2u] << ", " << v[3u] << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const ivec4& v) {
  os << "(" << v[0u] << ", " << v[1u] << ", " << v[2u] << ", " << v[3u] << ")";
  return os;
}

template <typename T, uint32_t N>
inline detail::vec<T, N> divup_vec(
    const detail::vec<T, N>& a,
    const detail::vec<T, N>& b) {
  detail::vec<T, N> result;
  for (uint32_t i = 0; i < N; ++i) {
    result[i] = utils::div_up(a[i], b[i]);
  }
  return result;
}

//
// std::vector<T> Handling
//

/*
 * Utility function to perform indexing on an std::vector<T>. Negative indexing
 * is allowed. For instance, passing an index of -1 will retrieve the last
 * element. If the requested index is out of bounds, then 1u will be returned.
 */
template <typename T>
inline T val_at(const int64_t index, const std::vector<T>& sizes) {
  const int64_t ndim = static_cast<int64_t>(sizes.size());
  if (index >= 0) {
    return index >= ndim ? 1 : sizes[index];
  } else {
    return ndim + index < 0 ? 1 : sizes[ndim + index];
  }
}

inline ivec2 make_ivec2(
    const std::vector<int64_t>& ints,
    bool reverse = false) {
  VK_CHECK_COND(ints.size() == 2);
  if (reverse) {
    return {safe_downcast<int32_t>(ints[1]), safe_downcast<int32_t>(ints[0])};
  } else {
    return {safe_downcast<int32_t>(ints[0]), safe_downcast<int32_t>(ints[1])};
  }
}

inline ivec3 make_ivec3(
    const std::vector<int64_t>& ints,
    bool reverse = false) {
  VK_CHECK_COND(ints.size() == 3);
  if (reverse) {
    return {
        safe_downcast<int32_t>(ints[2]),
        safe_downcast<int32_t>(ints[1]),
        safe_downcast<int32_t>(ints[0]),
    };
  } else {
    return {
        safe_downcast<int32_t>(ints[0]),
        safe_downcast<int32_t>(ints[1]),
        safe_downcast<int32_t>(ints[2]),
    };
  }
}

inline ivec4 make_ivec4(
    const std::vector<int64_t>& ints,
    bool reverse = false) {
  VK_CHECK_COND(ints.size() == 4);
  if (reverse) {
    return {
        safe_downcast<int32_t>(ints[3]),
        safe_downcast<int32_t>(ints[2]),
        safe_downcast<int32_t>(ints[1]),
        safe_downcast<int32_t>(ints[0]),
    };
  } else {
    return {
        safe_downcast<int32_t>(ints[0]),
        safe_downcast<int32_t>(ints[1]),
        safe_downcast<int32_t>(ints[2]),
        safe_downcast<int32_t>(ints[3]),
    };
  }
}

inline ivec4 make_ivec4_prepadded1(const std::vector<int64_t>& ints) {
  VK_CHECK_COND(ints.size() <= 4);

  ivec4 result = {1, 1, 1, 1};
  size_t base = 4 - ints.size();
  for (size_t i = 0; i < ints.size(); ++i) {
    result[i + base] = safe_downcast<int32_t>(ints[i]);
  }

  return result;
}

inline ivec3 make_ivec3(uvec3 ints) {
  return {
      safe_downcast<int32_t>(ints[0u]),
      safe_downcast<int32_t>(ints[1u]),
      safe_downcast<int32_t>(ints[2u])};
}

inline uvec3 make_uvec3(ivec3 ints) {
  return {
      safe_downcast<uint32_t>(ints[0u]),
      safe_downcast<uint32_t>(ints[1u]),
      safe_downcast<uint32_t>(ints[2u])};
}

/*
 * Given an vector of up to 4 uint64_t representing the sizes of a tensor,
 * constructs a uvec4 containing those elements in reverse order.
 */
inline uvec4 make_whcn_uvec4(const std::vector<int64_t>& arr) {
  uint32_t w = safe_downcast<uint32_t>(val_at(-1, arr));
  uint32_t h = safe_downcast<uint32_t>(val_at(-2, arr));
  uint32_t c = safe_downcast<uint32_t>(val_at(-3, arr));
  uint32_t n = safe_downcast<uint32_t>(val_at(-4, arr));

  return {w, h, c, n};
}

/*
 * Given an vector of up to 4 int64_t representing the sizes of a tensor,
 * constructs an ivec4 containing those elements in reverse order.
 */
inline ivec4 make_whcn_ivec4(const std::vector<int64_t>& arr) {
  int32_t w = val_at(-1, arr);
  int32_t h = val_at(-2, arr);
  int32_t c = val_at(-3, arr);
  int32_t n = val_at(-4, arr);

  return {w, h, c, n};
}

/*
 * Wrapper around std::accumulate that accumulates values of a container of
 * integral types into int64_t. Taken from `multiply_integers` in
 * <c10/util/accumulate.h>
 */
template <
    typename C,
    std::enable_if_t<std::is_integral<typename C::value_type>::value, int> = 0>
inline int64_t multiply_integers(const C& container) {
  return std::accumulate(
      container.begin(),
      container.end(),
      static_cast<int64_t>(1),
      std::multiplies<>());
}

/*
 * Product of integer elements referred to by iterators; accumulates into the
 * int64_t datatype. Taken from `multiply_integers` in <c10/util/accumulate.h>
 */
template <
    typename Iter,
    std::enable_if_t<
        std::is_integral<
            typename std::iterator_traits<Iter>::value_type>::value,
        int> = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
  // std::accumulate infers return type from `init` type, so if the `init` type
  // is not large enough to hold the result, computation can overflow. We use
  // `int64_t` here to avoid this.
  return std::accumulate(
      begin, end, static_cast<int64_t>(1), std::multiplies<>());
}

} // namespace utils
} // namespace vkcompute
