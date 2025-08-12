/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstring>
#include <ostream>

#include <c10/util/irange.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

using executorch::aten::BFloat16;
using executorch::aten::Half;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;

namespace executorch {
namespace runtime {
namespace testing {

namespace {

/**
 * Returns true if the two arrays are close according to the description on
 * `tensors_are_close()`.
 *
 * T must be a floating point type. Non-floating point data should be compared
 * directly.
 */
template <typename T>
bool data_is_close(
    const T* a,
    const T* b,
    size_t numel,
    double rtol,
    double atol) {
  ET_CHECK_MSG(
      numel == 0 || (a != nullptr && b != nullptr),
      "Pointers must not be null when numel > 0: numel %zu, a 0x%p, b 0x%p",
      numel,
      a,
      b);
  if (a == b) {
    return true;
  }
  for (const auto i : c10::irange(numel)) {
    const auto ai = a[i];
    const auto bi = b[i];

    if (std::isnan(ai) && std::isnan(bi)) {
      // NaN == NaN
    } else if (
        !std::isfinite(ai) && !std::isfinite(bi) && ((ai > 0) == (bi > 0))) {
      // -Inf == -Inf
      // +Inf == +Inf
    } else if (rtol == 0 && atol == 0) {
      // Exact comparison; avoid unnecessary math.
      if (ai != bi) {
        return false;
      }
    } else {
      auto allowed_error = atol + std::abs(rtol * bi);
      auto actual_error = std::abs(ai - bi);
      if (!std::isfinite(actual_error) || actual_error > allowed_error) {
        return false;
      }
    }
  }
  return true;
}

double default_atol_for_type(ScalarType t) {
  if (t == ScalarType::Half) {
    return internal::kDefaultHalfAtol;
  }
  if (t == ScalarType::BFloat16) {
    return internal::kDefaultBFloat16Atol;
  }
  return internal::kDefaultAtol;
}
} // namespace

bool tensors_are_close(
    const Tensor& a,
    const Tensor& b,
    double rtol,
    std::optional<double> opt_atol) {
  if (a.scalar_type() != b.scalar_type() || a.sizes() != b.sizes()) {
    return false;
  }

  // TODO(T132992348): support comparison between tensors of different strides
  ET_CHECK_MSG(
      a.strides() == b.strides(),
      "The two inputs of `tensors_are_close` function shall have same strides");

  // Since the two tensors have same shape and strides, any two elements that
  // share same index from underlying data perspective will also share same
  // index from tensor perspective, whatever the size and strides really are.
  // e.g. if a[i_1, i_2, ... i_n] = a.const_data_ptr()[m], we can assert
  // b[i_1, i_2, ... i_n] = b.const_data_ptr()[m])
  // So we can just compare the two underlying data sequentially to figure out
  // if the two tensors are same.

  double atol = opt_atol.value_or(default_atol_for_type(a.scalar_type()));

  if (a.nbytes() == 0) {
    // Note that this case is important. It's valid for a zero-size tensor to
    // have a null data pointer, but in some environments it's invalid to pass a
    // null pointer to memcmp() even when the size is zero.
    return true;
  } else if (a.scalar_type() == ScalarType::Float) {
    return data_is_close<float>(
        a.const_data_ptr<float>(),
        b.const_data_ptr<float>(),
        a.numel(),
        rtol,
        atol);
  } else if (a.scalar_type() == ScalarType::Double) {
    return data_is_close<double>(
        a.const_data_ptr<double>(),
        b.const_data_ptr<double>(),
        a.numel(),
        rtol,
        atol);
  } else if (a.scalar_type() == ScalarType::Half) {
    return data_is_close<Half>(
        a.const_data_ptr<Half>(),
        b.const_data_ptr<Half>(),
        a.numel(),
        rtol,
        atol);
  } else if (a.scalar_type() == ScalarType::BFloat16) {
    return data_is_close<BFloat16>(
        a.const_data_ptr<BFloat16>(),
        b.const_data_ptr<BFloat16>(),
        a.numel(),
        rtol,
        atol);
  } else {
    // Non-floating-point types can be compared bitwise.
    return memcmp(a.const_data_ptr(), b.const_data_ptr(), a.nbytes()) == 0;
  }
}

/**
 * Asserts that the provided tensors have the same sequence of close
 * underlying data elements and same numel. Note that this function is mainly
 * about comparing underlying data between two tensors, not relevant with how
 * tensor interpret the underlying data.
 */
bool tensor_data_is_close(
    const Tensor& a,
    const Tensor& b,
    double rtol,
    std::optional<double> opt_atol) {
  if (a.scalar_type() != b.scalar_type() || a.numel() != b.numel()) {
    return false;
  }

  double atol = opt_atol.value_or(default_atol_for_type(a.scalar_type()));
  if (a.nbytes() == 0) {
    // Note that this case is important. It's valid for a zero-size tensor to
    // have a null data pointer, but in some environments it's invalid to pass a
    // null pointer to memcmp() even when the size is zero.
    return true;
  } else if (a.scalar_type() == ScalarType::Float) {
    return data_is_close<float>(
        a.const_data_ptr<float>(),
        b.const_data_ptr<float>(),
        a.numel(),
        rtol,
        atol);
  } else if (a.scalar_type() == ScalarType::Double) {
    return data_is_close<double>(
        a.const_data_ptr<double>(),
        b.const_data_ptr<double>(),
        a.numel(),
        rtol,
        atol);
  } else {
    // Non-floating-point types can be compared bitwise.
    return memcmp(a.const_data_ptr(), b.const_data_ptr(), a.nbytes()) == 0;
  }
}

bool tensor_lists_are_close(
    const executorch::aten::Tensor* tensors_a,
    size_t num_tensors_a,
    const executorch::aten::Tensor* tensors_b,
    size_t num_tensors_b,
    double rtol,
    std::optional<double> opt_atol) {
  if (num_tensors_a != num_tensors_b) {
    return false;
  }
  for (const auto i : c10::irange(num_tensors_a)) {
    if (!tensors_are_close(tensors_a[i], tensors_b[i], rtol, opt_atol)) {
      return false;
    }
  }
  return true;
}

} // namespace testing
} // namespace runtime
} // namespace executorch

// ATen already defines operator<<() for Tensor and ScalarType.
#ifndef USE_ATEN_LIB

/*
 * These functions must be declared in the original namespaces of their
 * associated types so that C++ can find them.
 */
namespace executorch {
namespace runtime {
namespace etensor {

/**
 * Prints the ScalarType to the stream as a human-readable string.
 */
std::ostream& operator<<(std::ostream& os, const ScalarType& t) {
  const char* s = torch::executor::toString(t);
  if (std::strcmp(s, "UNKNOWN_SCALAR") == 0) {
    return os << "Unknown(" << static_cast<int32_t>(t) << ")";
  } else {
    return os << s;
  }
}

namespace {

/**
 * Prints the elements of `data` to the stream as comma-separated strings.
 */
template <typename T>
std::ostream& print_data(std::ostream& os, const T* data, size_t numel) {
  // TODO(dbort): Make this smarter: show dimensions, listen to strides,
  // break up or truncate data when it's huge
  for (const auto i : c10::irange(numel)) {
    os << data[i];
    if (i < numel - 1) {
      os << ", ";
    }
  }
  return os;
}

template <typename T>
std::ostream&
print_data(std::ostream& os, const etensor::complex<T>* data, size_t numel) {
  for (const auto i : c10::irange(numel)) {
    os << data[i].real_ << " + " << data[i].imag_ << "j";
    if (i < numel - 1) {
      os << ", ";
    }
  }
  return os;
}
/**
 * Prints the elements of `data` to the stream as comma-separated strings.
 *
 * Specialization for byte tensors as c++ default prints them as chars where as
 * debugging is typically easier with numbers here (tensors dont store string
 * data)
 */
template <>
std::ostream& print_data(std::ostream& os, const uint8_t* data, size_t numel) {
  // TODO(dbort): Make this smarter: show dimensions, listen to strides,
  // break up or truncate data when it's huge
  for (const auto i : c10::irange(numel)) {
    os << (uint64_t)data[i];
    if (i < numel - 1) {
      os << ", ";
    }
  }
  return os;
}

} // namespace

/**
 * Prints the Tensor to the stream as a human-readable string.
 */
std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  os << "ETensor(sizes={";
  for (const auto dim : c10::irange(t.dim())) {
    os << t.size(dim);
    if (dim < t.dim() - 1) {
      os << ", ";
    }
  }
  os << "}, dtype=" << t.scalar_type() << ", data={";

  // Map from the ScalarType to the C type.
#define PRINT_CASE(ctype, stype)                          \
  case ScalarType::stype:                                 \
    print_data(os, t.const_data_ptr<ctype>(), t.numel()); \
    break;

  switch (t.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND3(Half, Bool, BFloat16, PRINT_CASE)
    ET_FORALL_COMPLEX_TYPES(PRINT_CASE)
    default:
      ET_CHECK_MSG(
          false,
          "Unhandled dtype %s",
          torch::executor::toString(t.scalar_type()));
  }

#undef PRINT_CASE

  os << "})";

  return os;
}

} // namespace etensor
} // namespace runtime
} // namespace executorch

#endif // !USE_ATEN_LIB
