#pragma once

// @nolint PATTERNLINT <functional> is required for std::equal_to, etc.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <cmath>
#include <type_traits>
#include <bitset>
#include <climits>

// These macros helped us unify vec_base.h
#ifdef CPU_CAPABILITY_AVX512
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(64)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(64))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 64
#define int_vector __m512i
#else // CPU_CAPABILITY_AVX512
#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(32)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(32))
#else
#define __at_align__
#endif
#define VECTOR_WIDTH 32
#define int_vector __m256i
#endif // CPU_CAPABILITY_AVX512

namespace executorch {
namespace vec {

// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

template<size_t n> struct int_of_size;

#define DEFINE_INT_OF_SIZE(int_t) \
template<> struct int_of_size<sizeof(int_t)> { using type = int_t; }

DEFINE_INT_OF_SIZE(int64_t);
DEFINE_INT_OF_SIZE(int32_t);
DEFINE_INT_OF_SIZE(int16_t);
DEFINE_INT_OF_SIZE(int8_t);

#undef DEFINE_INT_OF_SIZE

template <typename T>
using int_same_size_t = typename int_of_size<sizeof(T)>::type;

// NOTE: If you specialize on a type, you must define all operations!

// emulates Vectorized types
#if defined(__s390x__)
template <class T, class TEMP=void>
#else
template <class T>
#endif
struct Vectorized {
private:
  __at_align__ T values[VECTOR_WIDTH / sizeof(T)];
public:
  using value_type = T;
  using size_type = int;
  // Note [constexpr static function to avoid odr-usage compiler bug]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Why, you might ask, is size defined to be a static constexpr function,
  // rather than a more ordinary 'static constexpr int size;' variable?
  // The problem lies within ODR rules for static constexpr members versus
  // static constexpr functions.  First, recall that this class (along with all
  // of its derivations) live in an anonymous namespace: they are intended to be
  // *completely* inlined at their use-sites, because we need to compile it
  // multiple times for different instruction sets.
  //
  // Because of this constraint, we CANNOT provide a single definition for
  // any static members in this class; since we want to compile the class
  // multiple times, there wouldn't actually be any good place to put the
  // definition.  Now here is the problem: if we ODR-use a static constexpr
  // member, we are *obligated* to provide a definition.  Without the
  // definition, you get a compile error like:
  //
  //    relocation R_X86_64_PC32 against undefined symbol
  //    `_ZN2at6vec25612_GLOBAL__N_16VectorizedIdE4sizeE' can not be used when making
  //    a shared object; recompile with -fPIC
  //
  // If this were C++17, we could replace a static constexpr variable with
  // an inline variable which doesn't require one definition. But we are not
  // C++17.  So the next best thing is to replace the member with a static
  // constexpr (and therefore inline) function, which does not require ODR
  // either.
  //
  // Also, technically according to the C++ standard, we don't have to define
  // a constexpr variable if we never odr-use it.  But it seems that some
  // versions GCC/Clang have buggy determinations on whether or not an
  // identifier is odr-used or not, and in any case it's hard to tell if
  // a variable is odr-used or not.  So best to just cut the problem at the root.
  static constexpr size_type size_T = sizeof(T);  // Workaround to compile with VS2022.
  static constexpr size_type size() {
    return VECTOR_WIDTH / size_T;
  }
  Vectorized() : values{static_cast<T>(0)} {}
  Vectorized(T val) {
    for (size_t i = 0; i != size(); i++) {
      values[i] = val;
    }
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) : values{vals...}{
  }
  // This also implies const T& operator[](int idx) const
  inline operator const T*() const {
    return values;
  }
  // This also implies T& operator[](int idx)
  inline operator T*() {
    return values;
  }
  // Return the values as char* for type punning
  auto as_bytes() const -> const char* {
    return reinterpret_cast<const char*>(values);
  }
  template <int64_t mask_>
  static Vectorized<T> blend(const Vectorized<T>& a, const Vectorized<T>& b) {
    int64_t mask = mask_;
    Vectorized vector;
    for (size_t i = 0; i < size(); ++i) {
      if (mask & 0x01) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
      mask = mask >> 1;
    }
    return vector;
  }
  static Vectorized<T> blendv(const Vectorized<T>& a, const Vectorized<T>& b,
                          const Vectorized<T>& mask) {
    Vectorized vector;
    int_same_size_t<T> buffer[size()];
    mask.store(buffer);
    for (size_t i = 0; i < size(); ++i) {
      if (buffer[i] & 0x01)
       {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
    }
    return vector;
  }
  template<typename step_t>  // step sometimes requires a higher precision type (e.g., T=int, step_t=double)
  static Vectorized<T> arange(T base = static_cast<T>(0), step_t step = static_cast<step_t>(1)) {
    Vectorized vector;
    for (size_t i = 0; i < size(); ++i) {
      vector.values[i] = base + i * step;
    }
    return vector;
  }
  static Vectorized<T> set(const Vectorized<T>& a, const Vectorized<T>& b, int64_t count = size()) {
    Vectorized vector;
    for (size_t i = 0; i < size(); ++i) {
      if (i < count) {
        vector[i] = b[i];
      } else {
        vector[i] = a[i];
      }
    }
    return vector;
  }
  static Vectorized<T> loadu(const void* ptr) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, VECTOR_WIDTH);
    return vector;
  }
  static Vectorized<T> loadu(const void* ptr, int64_t count) {
    Vectorized vector;
    std::memcpy(vector.values, ptr, count * sizeof(T));
    return vector;
  }
  void store(void* ptr, int count = size()) const {
    std::memcpy(ptr, values, count * sizeof(T));
  }
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int mask = 0;
    for (size_t i = 0; i < size(); ++ i) {
      if (values[i] == static_cast<T>(0)) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vectorized<T> isnan() const {
    Vectorized<T> vector;
    for (size_t i = 0; i != size(); i++) {
      if (std::isnan(values[i])) {
        std::memset(static_cast<void*>(vector.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void*>(vector.values + i), 0, sizeof(T));
      }
    }
    return vector;
  }
  Vectorized<T> map(T (*const f)(T)) const {
    Vectorized<T> ret;
    for (size_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  Vectorized<T> map(T (*const f)(const T &)) const {
    Vectorized<T> ret;
    for (size_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
  template <typename other_t_abs = T,
            typename std::enable_if<!std::is_floating_point<other_t_abs>::value, int>::type = 0>
  Vectorized<T> abs() const {
    // other_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<other_t_abs, T>::value, "other_t_abs must be T");
    return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
  }
  template <typename float_t_abs = T,
            typename std::enable_if<std::is_floating_point<float_t_abs>::value, int>::type = 0>
  Vectorized<T> abs() const {
    // float_t_abs is for SFINAE and clarity. Make sure it is not changed.
    static_assert(std::is_same<float_t_abs, T>::value, "float_t_abs must be T");
    // Specifically deal with floating-point because the generic code above won't handle -0.0 (which should result in
    // 0.0) properly.
    return map([](T x) -> T { return std::abs(x); });
  }

  Vectorized<T> acos() const {
    return map(std::acos);
  }
  Vectorized<T> asin() const {
    return map(std::asin);
  }
  Vectorized<T> atan() const {
    return map(std::atan);
  }
  Vectorized<T> atan2(const Vectorized<T> &exp) const {
    Vectorized<T> ret;
    for (size_t i = 0; i < size(); ++i) {
      ret[i] = std::atan2(values[i], exp[i]);
    }
    return ret;
  }
  template <
    typename U = T,
    typename std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> copysign(const Vectorized<T> &sign) const {
    Vectorized<T> ret;
    for (size_t i = 0; i < size(); i++) {
      ret[i] = std::copysign(values[i], sign[i]);
    }
    return ret;
  }
  Vectorized<T> erf() const {
    return map(std::erf);
  }
  Vectorized<T> erfc() const {
    return map(std::erfc);
  }
  Vectorized<T> exp() const {
    return map(std::exp);
  }
  Vectorized<T> exp2() const {
    return map(std::exp2);
  }
  Vectorized<T> expm1() const {
    return map(std::expm1);
  }
  Vectorized<T> frac() const {
    return *this - this->trunc();
  }
  template <
    typename U = T,
    typename std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  Vectorized<T> fmod(const Vectorized<T>& q) const {
    // U is for SFINAE purposes only. Make sure it is not changed.
    static_assert(std::is_same<U, T>::value, "U must be T");
    Vectorized<T> ret;
    for (size_t i = 0; i < size(); ++i) {
      ret[i] = std::fmod(values[i], q[i]);
    }
    return ret;
  }
  Vectorized<T> log() const {
    return map(std::log);
  }
  Vectorized<T> log10() const {
    return map(std::log10);
  }
  Vectorized<T> log1p() const {
    return map(std::log1p);
  }
  Vectorized<T> log2() const {
    return map(std::log2);
  }
  Vectorized<T> ceil() const {
    return map(std::ceil);
  }
  Vectorized<T> cos() const {
    return map(std::cos);
  }
  Vectorized<T> cosh() const {
    return map(std::cosh);
  }
  Vectorized<T> floor() const {
    return map(std::floor);
  }
  Vectorized<T> hypot(const Vectorized<T> &b) const {
    Vectorized<T> ret;
    for (size_t i = 0; i < size(); ++i) {
      ret[i] = std::hypot(values[i], b[i]);
    }
    return ret;
  }
  Vectorized<T> neg() const {
    // NB: the trailing return type is needed because we need to coerce the
    // return value back to T in the case of unary operator- incuring a
    // promotion
    return map([](T x) -> T { return -x; });
  }
  Vectorized<T> nextafter(const Vectorized<T> &b) const {
    Vectorized<T> ret;
    for (size_t i = 0; i < size(); ++i) {
      ret[i] = std::nextafter(values[i], b[i]);
    }
    return ret;
  }
  Vectorized<T> round() const {
    // TODO(T149257433): implement custom round that rounds midway numbers to
    // the nearest even integer.
    return map(std::round);
  }
  Vectorized<T> sin() const {
    return map(std::sin);
  }
  Vectorized<T> sinh() const {
    return map(std::sinh);
  }
  Vectorized<T> tan() const {
    return map(std::tan);
  }
  Vectorized<T> tanh() const {
    return map(std::tanh);
  }
  Vectorized<T> trunc() const {
    return map(std::trunc);
  }
  Vectorized<T> lgamma() const {
    return map(std::lgamma);
  }
  Vectorized<T> sqrt() const {
    return map(std::sqrt);
  }
  Vectorized<T> reciprocal() const {
    return map([](T x) { return (T)(1) / x; });
  }
  Vectorized<T> rsqrt() const {
    return map([](T x) { return (T)1 / std::sqrt(x); });
  }
  Vectorized<T> pow(const Vectorized<T> &exp) const {
    Vectorized<T> ret;
    for (size_t i = 0; i < size(); ++i) {
      ret[i] = std::pow(values[i], exp[i]);
    }
    return ret;
  }
private:
  template <typename Op>
  inline Vectorized<T> binary_pred(const Vectorized<T>& other, Op op) const {
    // All bits are set to 1 if the pred is true, otherwise 0.
    Vectorized<T> vector;
    for (size_t i = 0; i != size(); i++) {
      if (op(values[i], other.values[i])) {
        std::memset(static_cast<void*>(vector.values + i), 0xFF, sizeof(T));
      } else {
        std::memset(static_cast<void*>(vector.values + i), 0, sizeof(T));
      }
    }
    return vector;
  }

public:
  Vectorized<T> operator==(const Vectorized<T>& other) const { return binary_pred(other, std::equal_to<T>()); }
  Vectorized<T> operator!=(const Vectorized<T>& other) const { return binary_pred(other, std::not_equal_to<T>()); }
  Vectorized<T> operator>=(const Vectorized<T>& other) const { return binary_pred(other, std::greater_equal<T>()); }
  Vectorized<T> operator<=(const Vectorized<T>& other) const { return binary_pred(other, std::less_equal<T>()); }
  Vectorized<T> operator>(const Vectorized<T>& other) const { return binary_pred(other, std::greater<T>()); }
  Vectorized<T> operator<(const Vectorized<T>& other) const { return binary_pred(other, std::less<T>()); }

private:
  template <typename Op>
  inline Vectorized<T> binary_pred_bool(const Vectorized<T>& other, Op op) const {
    // 1 if the pred is true, otherwise 0.
    Vectorized<T> vector;
    for (size_t i = 0; i != size(); ++ i) {
      vector[i] = static_cast<T>(op(values[i], other.values[i]));
    }
    return vector;
  }

public:
  Vectorized<T> eq(const Vectorized<T>& other) const { return binary_pred_bool(other, std::equal_to<T>()); }
  Vectorized<T> ne(const Vectorized<T>& other) const { return binary_pred_bool(other, std::not_equal_to<T>()); }
  Vectorized<T> gt(const Vectorized<T>& other) const { return binary_pred_bool(other, std::greater<T>()); }
  Vectorized<T> ge(const Vectorized<T>& other) const { return binary_pred_bool(other, std::greater_equal<T>()); }
  Vectorized<T> lt(const Vectorized<T>& other) const { return binary_pred_bool(other, std::less<T>()); }
  Vectorized<T> le(const Vectorized<T>& other) const { return binary_pred_bool(other, std::less_equal<T>()); }
};

template <class T> Vectorized<T> inline operator+(const Vectorized<T> &a, const Vectorized<T> &b) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] + b[i];
  }
  return c;
}

template <class T> Vectorized<T> inline operator-(const Vectorized<T> &a, const Vectorized<T> &b) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] - b[i];
  }
  return c;
}

template <class T> Vectorized<T> inline operator*(const Vectorized<T> &a, const Vectorized<T> &b) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] * b[i];
  }
  return c;
}

template <class T> Vectorized<T> inline operator/(const Vectorized<T> &a, const Vectorized<T> &b) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] / b[i];
  }
  return c;
}

template <class T> Vectorized<T> inline operator||(
    const Vectorized<T> &a, const Vectorized<T> &b) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] || b[i];
  }
  return c;
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <class T>
Vectorized<T> inline maximum(const Vectorized<T> &a, const Vectorized<T> &b) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] > b[i]) ? a[i] : b[i];
    if (std::isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <class T>
Vectorized<T> inline minimum(const Vectorized<T> &a, const Vectorized<T> &b) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (std::isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <class T>
Vectorized<T> inline clamp(const Vectorized<T> &a, const Vectorized<T> &min_vec, const Vectorized<T> &max_vec) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = std::min(std::max(a[i], min_vec[i]), max_vec[i]);
  }
  return c;
}

template <class T>
Vectorized<T> inline clamp_max(const Vectorized<T> &a, const Vectorized<T> &max_vec) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] > max_vec[i] ? max_vec[i] : a[i];
  }
  return c;
}

template <class T>
Vectorized<T> inline clamp_min(const Vectorized<T> &a, const Vectorized<T> &min_vec) {
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
  }
  return c;
}

struct Vectorizedi;

#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
template <class T, typename Op>
static inline Vectorized<T> bitwise_binary_op(const Vectorized<T> &a, const Vectorized<T> &b, Op op) {
  int_vector buffer;
#if defined(CPU_CAPABILITY_AVX2)
  int_vector a_buffer = _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer = _mm256_load_si256(reinterpret_cast<const int_vector*>((const T*)b));
#elif defined(CPU_CAPABILITY_AVX512)
  int_vector a_buffer = _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)a));
  int_vector b_buffer = _mm512_load_si512(reinterpret_cast<const int_vector*>((const T*)b));
#endif
  buffer = op(a_buffer, b_buffer);
  __at_align__ T results[Vectorized<T>::size()];

#if defined(CPU_CAPABILITY_AVX2)
  _mm256_store_si256(reinterpret_cast<int_vector*>(results), buffer);
#elif defined(CPU_CAPABILITY_AVX512)
  _mm512_store_si512(reinterpret_cast<int_vector*>(results), buffer);
#endif
  return Vectorized<T>::loadu(results);
}

template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_and_si512 or _mm256_and_si256 with lambda because it is always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm256_and_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm512_and_si512(a, b); });
#endif
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_or_si512 or _mm256_or_si256 with lambda because it is always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm256_or_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm512_or_si512(a, b); });
#endif
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  // We enclose _mm512_xor_si512 or _mm256_xor_si256 with lambda because it is always_inline
#if defined(CPU_CAPABILITY_AVX2)
  return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm256_xor_si256(a, b); });
#elif defined(CPU_CAPABILITY_AVX512)
  return bitwise_binary_op(a, b, [](int_vector a, int_vector b) { return _mm512_xor_si512(a, b); });
#endif
}

#else

template <typename T>
auto load(char const* data) -> T {
  T ret;
  std::memcpy(&ret, data, sizeof(ret));
  return ret;
}

template<class T, typename Op>
static inline Vectorized<T> bitwise_binary_op(const Vectorized<T> &a, const Vectorized<T> &b, Op op) {
  static constexpr uint32_t element_no = VECTOR_WIDTH / sizeof(intmax_t);
  __at_align__ intmax_t buffer[element_no];
  static_assert(VECTOR_WIDTH % sizeof(intmax_t) == 0, "VECTOR_WIDTH not a multiple of sizeof(intmax_t)");
  static_assert(sizeof(buffer) == sizeof(Vectorized<T>), "sizeof(buffer) must match sizeof(Vectorized<T>)");
  // We should be using memcpy in order to respect the strict aliasing rule
  // see: https://github.com/pytorch/pytorch/issues/66119
  // Using char* is defined in the C11 standard 6.5 Expression paragraph 7
  // (http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
  const auto* a_data = a.as_bytes();
  const auto* b_data = b.as_bytes();
  // load each intmax_t chunk and process; increase pointers by sizeof(intmax_t)
  for (auto& out : buffer) {
    out = op(load<intmax_t>(a_data), load<intmax_t>(b_data));
    a_data += sizeof(intmax_t);
    b_data += sizeof(intmax_t);
  }
  assert(a_data == a.as_bytes() + sizeof(a));
  assert(b_data == b.as_bytes() + sizeof(b));
  return Vectorized<T>::loadu(buffer);
}

template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
  return bitwise_binary_op(a, b, std::bit_and<intmax_t>());
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
  return bitwise_binary_op(a, b, std::bit_or<intmax_t>());
}
template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
  return bitwise_binary_op(a, b, std::bit_xor<intmax_t>());
}

#endif // defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)

template<class T, typename std::enable_if_t<!std::is_base_of<Vectorizedi, Vectorized<T>>::value, int> = 0>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
  Vectorized<T> ones;  // All bits are 1
  memset((T*) ones, 0xFF, VECTOR_WIDTH);
  return a ^ ones;
}

template <class T> Vectorized<T> inline operator<<(const Vectorized<T> &a, const Vectorized<T> &b) {
  constexpr T max_shift = sizeof(T) * CHAR_BIT;
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    T shift = b[i];
    if ((static_cast<std::make_signed_t<T>>(shift) < 0) || (shift >= max_shift)) {
      c[i] = 0;
    } else {
      c[i] = static_cast<std::make_unsigned_t<T>>(a[i]) << shift;
    }
  }
  return c;
}

template <class T> Vectorized<T> inline operator>>(const Vectorized<T> &a, const Vectorized<T> &b) {
  // right shift value to retain sign bit for signed and no bits for unsigned
  constexpr T max_shift = sizeof(T) * CHAR_BIT - std::is_signed_v<T>;
  Vectorized<T> c;
  for (size_t i = 0; i != Vectorized<T>::size(); i++) {
    T shift = b[i];
    if ((static_cast<std::make_signed_t<T>>(shift) < 0) || (shift >= max_shift)) {
      c[i] = a[i] >> max_shift;
    } else {
      c[i] = a[i] >> shift;
    }
  }
  return c;
}

template <typename T>
inline Vectorized<T>& operator += (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a + b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator -= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a - b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator /= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a / b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator %= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a % b;
  return a;
}
template <typename T>
inline Vectorized<T>& operator *= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a * b;
  return a;
}

template <typename T>
inline Vectorized<T>& operator <<= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a << b;
  return a;
}

template <typename T>
inline Vectorized<T>& operator >>= (Vectorized<T>& a, const Vectorized<T>& b) {
  a = a >> b;
  return a;
}

template <typename T>
inline Vectorized<T> fmadd(const Vectorized<T>& a, const Vectorized<T>& b, const Vectorized<T>& c) {
  return a * b + c;
}

template <typename T>
inline Vectorized<T> fmsub(const Vectorized<T>& a, const Vectorized<T>& b, const Vectorized<T>& c) {
  return a * b - c;
}

template <int64_t scale = 1, typename T = void>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<T>>
inline gather(T const* base_addr, const Vectorized<int_same_size_t<T>>& vindex) {
  static constexpr int size = Vectorized<T>::size();
  int_same_size_t<T> index_arr[size];
  vindex.store(static_cast<void*>(index_arr));
  T buffer[size];
  for (size_t i = 0; i < size; ++i) {
    buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
  }
  return Vectorized<T>::loadu(static_cast<void*>(buffer));
}

template <int64_t scale = 1, typename T = void>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<T>>
inline mask_gather(const Vectorized<T>& src, T const* base_addr,
                   const Vectorized<int_same_size_t<T>>& vindex, Vectorized<T>& mask) {
  static constexpr int size = Vectorized<T>::size();
  T src_arr[size];
  int_same_size_t<T> mask_arr[size];  // use int type so we can logical and
  int_same_size_t<T> index_arr[size];
  src.store(static_cast<void*>(src_arr));
  mask.store(static_cast<void*>(mask_arr));
  vindex.store(static_cast<void*>(index_arr));
  T buffer[size];
  for (size_t i = 0; i < size; ++i) {
    if (mask_arr[i] & 0x01) {  // check highest bit
      buffer[i] = base_addr[index_arr[i] * scale / sizeof(T)];
    } else {
      buffer[i] = src_arr[i];
    }
  }
  mask = Vectorized<T>();  // "zero out" mask
  return Vectorized<T>::loadu(static_cast<void*>(buffer));
}

// Cast a given vector to another type without changing the bits representation.
// So a Vectorized<double> of 512 bits containing all ones can be cast to a
// Vectorized<int64_t> of 512 bits containing all ones (i.e., eight negative 1s).
// A Vec<double> of 256 bits containing all ones can be cast to a
// Vec<int64_t> of 256 bits containing all ones (i.e., four negative 1s).
// There is a struct here because we don't have static_if and I can't
// partially specialize a templated function.
template<typename dst_t, typename src_t>
struct CastImpl {
  static inline Vectorized<dst_t> apply(const Vectorized<src_t>& src) {
    src_t src_arr[Vectorized<src_t>::size()];
    src.store(static_cast<void*>(src_arr));
    return Vectorized<dst_t>::loadu(static_cast<const void*>(src_arr));
  }
};

template<typename scalar_t>
struct CastImpl<scalar_t, scalar_t> {
  static inline Vectorized<scalar_t> apply(const Vectorized<scalar_t>& src) {
    return src;
  }
};

template<typename dst_t, typename src_t>
inline Vectorized<dst_t> cast(const Vectorized<src_t>& src) {
  return CastImpl<dst_t, src_t>::apply(src);
}

template <typename T>
inline Vectorized<int_same_size_t<T>> convert_to_int_of_same_size(const Vectorized<T>& src) {
  static constexpr int size = Vectorized<T>::size();
  T src_arr[size];
  src.store(static_cast<void*>(src_arr));
  int_same_size_t<T> buffer[size];
  for (size_t i = 0; i < size; ++i) {
    buffer[i] = static_cast<int_same_size_t<T>>(src_arr[i]);
  }
  return Vectorized<int_same_size_t<T>>::loadu(static_cast<void*>(buffer));
}

// Example inputs for AVX512:
// a   Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7}
// b   Vectorized<float>   = {a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15}
// returns:
//           Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15}
//           Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
// Example inputs for AVX2: a           Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3}
//               b                      Vectorized<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
//       returns:                       Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7}
//                                      Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7}
template <typename T>
inline std::enable_if_t<Vectorized<T>::size() % 2 == 0, std::pair<Vectorized<T>, Vectorized<T>>>
deinterleave2(const Vectorized<T>& a, const Vectorized<T>& b) {
  static constexpr int size = Vectorized<T>::size();
  static constexpr int half_size = size / 2;
  T a_arr[size];
  T b_arr[size];
  T buffer1[size];
  T buffer2[size];
  a.store(static_cast<void*>(a_arr));
  b.store(static_cast<void*>(b_arr));
  for (size_t i = 0; i < half_size; ++i) {
    buffer1[i] = a_arr[i * 2];
    buffer1[half_size + i] = b_arr[i * 2];
    buffer2[i] = a_arr[i * 2 + 1];
    buffer2[half_size + i] = b_arr[i * 2 + 1];
  }
  return std::make_pair(Vectorized<T>::loadu(static_cast<void*>(buffer1)),
                        Vectorized<T>::loadu(static_cast<void*>(buffer2)));
}

// inverse operation of deinterleave2
// Example inputs for AVX512:
//  a       Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15}
//  b       Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
// returns, for AVX512:
//          Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7}
//          Vectorized<float>   = {a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15}
// Example inputs for AVX2 : a           Vectorized<float>   = {a0, a1, a2, a3, a4, a5, a6, a7}
//                   b                   Vectorized<float>   = {b0, b1, b2, b3, b4, b5, b6, b7}
//       returns:            Vectorized<float>   = {a0, b0, a1, b1, a2, b2, a3, b3}
//                           Vectorized<float>   = {a4, b4, a5, b5, a6, b6, a7, b7}
template <typename T>
inline std::enable_if_t<Vectorized<T>::size() % 2 == 0, std::pair<Vectorized<T>, Vectorized<T>>>
interleave2(const Vectorized<T>& a, const Vectorized<T>& b) {
  static constexpr int size = Vectorized<T>::size();
  static constexpr int half_size = size / 2;
  T a_arr[size];
  T b_arr[size];
  T buffer1[size];
  T buffer2[size];
  a.store(static_cast<void*>(a_arr));
  b.store(static_cast<void*>(b_arr));
  for (size_t i = 0; i < half_size; ++i) {
    buffer1[i * 2] = a_arr[i];
    buffer1[i * 2 + 1] = b_arr[i];
    buffer2[i * 2] = a_arr[half_size + i];
    buffer2[i * 2 + 1] = b_arr[half_size + i];
  }
  return std::make_pair(Vectorized<T>::loadu(static_cast<void*>(buffer1)),
                        Vectorized<T>::loadu(static_cast<void*>(buffer2)));
}

template <typename src_T, typename dst_T>
inline void convert(const src_T *src, dst_T *dst, int64_t n) {
#ifndef _MSC_VER
# pragma unroll
#endif
  for (int64_t i = 0; i < n; ++i) {
    (void)i; //Suppress unused variable warning
    *dst = static_cast<dst_T>(*src);
    src++;
    dst++;
  }
}

template <typename T>
inline Vectorized<T> flip(const Vectorized<T> & data) {
  static constexpr int size = Vectorized<T>::size();
  T output[size];
  T buffer[size];
  data.store(static_cast<void*>(buffer));
  for (size_t i = 0; i < size; ++i) {
    output[i] = buffer[size - i - 1];
  }
  return Vectorized<T>::loadu(static_cast<void*>(output));
}

// Transpose the `src` buffer of type `T` and size (M,N) into the `dst` buffer. `ld_src` is the leading
// dimension of `src` and `ld_dst` is the leading dimension of `dst`.
template <typename T, int M, int N>
inline void transpose_mxn(const T* src, int64_t ld_src, T* dst, int64_t ld_dst) {
  for (size_t i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      dst[j*ld_dst + i] = src[i*ld_src + j];
    }
  }
}

} // namespace CPU_CAPABILITY

} // namespace vec
} // namespace executorch
