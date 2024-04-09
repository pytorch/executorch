/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/tag.h>
#include <executorch/runtime/platform/assert.h>

#include <variant>

namespace torch {
namespace executor {

struct EValue;

// Tensor gets proper reference treatment because its expensive to copy in aten
// mode, all other types are just copied.
template <typename T>
struct evalue_to_const_ref_overload_return {
  using type = T;
};

template <>
struct evalue_to_const_ref_overload_return<exec_aten::Tensor> {
  using type = const exec_aten::Tensor&;
};

template <typename T>
struct evalue_to_ref_overload_return {
  using type = T;
};

template <>
struct evalue_to_ref_overload_return<exec_aten::Tensor> {
  using type = exec_aten::Tensor&;
};

/*
 * Helper class used to correlate EValues in the executor table, with the
 * unwrapped list of the proper type. Because values in the runtime's values
 * table can change during execution, we cannot statically allocate list of
 * objects at deserialization. Imagine the serialized list says index 0 in the
 * value table is element 2 in the list, but during execution the value in
 * element 2 changes (in the case of tensor this means the TensorImpl* stored in
 * the tensor changes). To solve this instead they must be created dynamically
 * whenever they are used.
 */
template <typename T>
class BoxedEvalueList {
 public:
  BoxedEvalueList() = default;
  /*
   * Wrapped_vals is a list of pointers into the values table of the runtime
   * whose destinations correlate with the elements of the list, unwrapped_vals
   * is a container of the same size whose serves as memory to construct the
   * unwrapped vals.
   */
  BoxedEvalueList(EValue** wrapped_vals, T* unwrapped_vals, int size)
      : wrapped_vals_(wrapped_vals, size), unwrapped_vals_(unwrapped_vals) {}
  /*
   * Constructs and returns the list of T specified by the EValue pointers
   */
  exec_aten::ArrayRef<T> get() const;

 private:
  // Source of truth for the list
  exec_aten::ArrayRef<EValue*> wrapped_vals_;
  // Same size as wrapped_vals
  mutable T* unwrapped_vals_;
};

template <>
exec_aten::ArrayRef<exec_aten::optional<exec_aten::Tensor>>
BoxedEvalueList<exec_aten::optional<exec_aten::Tensor>>::get() const;

// Aggregate typing system similar to IValue only slimmed down with less
// functionality, no dependencies on atomic, and fewer supported types to better
// suit embedded systems (ie no intrusive ptr)
struct EValue {
 private:
  struct None {};
  std::variant<
      None,
      int64_t,
      double,
      bool,
      // TODO(jakeszwe): convert back to pointers to optimize size of this
      // struct
      exec_aten::ArrayRef<char>,
      exec_aten::ArrayRef<double>,
      exec_aten::ArrayRef<bool>,
      BoxedEvalueList<int64_t>,
      BoxedEvalueList<exec_aten::Tensor>,
      BoxedEvalueList<exec_aten::optional<exec_aten::Tensor>>,
      exec_aten::Tensor>
      repr_;

 public:
  // Basic ctors and assignments
  EValue(const EValue& rhs) = default;

  EValue(EValue&& rhs) noexcept = default;

  EValue& operator=(EValue&& rhs) noexcept = default;

  EValue& operator=(EValue const& rhs) = default;

  ~EValue() = default;

  /****** None Type ******/
  EValue() = default;

  bool isNone() const {
    return std::holds_alternative<None>(repr_);
  }

  /****** Int Type ******/
  /*implicit*/ EValue(int64_t i) : repr_(i) {}

  bool isInt() const {
    return std::holds_alternative<int64_t>(repr_);
  }

  int64_t toInt() const {
    ET_CHECK_MSG(isInt(), "EValue is not an int.");
    return std::get<int64_t>(repr_);
  }

  /****** Double Type ******/
  /*implicit*/ EValue(double d) : repr_(d) {}

  bool isDouble() const {
    return std::holds_alternative<double>(repr_);
  }

  double toDouble() const {
    ET_CHECK_MSG(isDouble(), "EValue is not a Double.");
    return std::get<double>(repr_);
  }

  /****** Bool Type ******/
  /*implicit*/ EValue(bool b) : repr_(b) {}

  bool isBool() const {
    return std::holds_alternative<bool>(repr_);
  }

  bool toBool() const {
    ET_CHECK_MSG(isBool(), "EValue is not a Bool.");
    return std::get<bool>(repr_);
  }

  /****** Scalar Type ******/
  /// Construct an EValue using the implicit value of a Scalar.
  /*implicit*/ EValue(exec_aten::Scalar s) {
    if (s.isIntegral(false)) {
      repr_ = s.to<int64_t>();
    } else if (s.isFloatingPoint()) {
      repr_ = s.to<double>();
    } else if (s.isBoolean()) {
      repr_ = s.to<bool>();
    } else {
      ET_CHECK_MSG(false, "Scalar passed to EValue is not initialized.");
    }
  }

  bool isScalar() const {
    return isInt() || isDouble() || isBool();
  }

  exec_aten::Scalar toScalar() const {
    return std::visit(
        [](auto&& val) {
          using T = std::decay_t<decltype(val)>;
          if constexpr (
              std::is_same_v<T, double> || std::is_same_v<T, int> ||
              std::is_same_v<T, bool>) {
            return exec_aten::Scalar(val);
          }
          ET_CHECK_MSG(false, "Evalue is not a Scalar.");
          return exec_aten::Scalar(0);
        },
        repr_);
  }

  /****** Tensor Type ******/
  /*implicit*/ EValue(exec_aten::Tensor t) : repr_(std::move(t)) {}

  bool isTensor() const {
    return std::holds_alternative<exec_aten::Tensor>(repr_);
  }

  exec_aten::Tensor toTensor() && {
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    auto result = std::get<exec_aten::Tensor>(std::move(repr_));
    repr_ = None{};
    return result;
  }

  exec_aten::Tensor& toTensor() & {
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    return std::get<exec_aten::Tensor>(repr_);
  }

  const exec_aten::Tensor& toTensor() const& {
    ET_CHECK_MSG(isTensor(), "EValue is not a Tensor.");
    return std::get<exec_aten::Tensor>(repr_);
  }

  /****** String Type ******/
  /*implicit*/ EValue(const char* s, size_t size)
      : repr_(exec_aten::ArrayRef<char>(s, size)) {}

  bool isString() const {
    return std::holds_alternative<exec_aten::ArrayRef<char>>(repr_);
  }

  exec_aten::string_view toString() const {
    ET_CHECK_MSG(isString(), "EValue is not a String.");
    const auto& str = std::get<exec_aten::ArrayRef<char>>(repr_);
    return exec_aten::string_view(str.data(), str.size());
  }

  /****** Int List Type ******/
  /*implicit*/ EValue(BoxedEvalueList<int64_t> i) : repr_(std::move(i)) {}

  bool isIntList() const {
    return std::holds_alternative<BoxedEvalueList<int64_t>>(repr_);
  }

  exec_aten::ArrayRef<int64_t> toIntList() const {
    ET_CHECK_MSG(isIntList(), "EValue is not an Int List.");
    return std::get<BoxedEvalueList<int64_t>>(repr_).get();
  }

  /****** Bool List Type ******/
  /*implicit*/ EValue(exec_aten::ArrayRef<bool> b) : repr_(b) {}

  bool isBoolList() const {
    return std::holds_alternative<exec_aten::ArrayRef<bool>>(repr_);
  }

  exec_aten::ArrayRef<bool> toBoolList() const {
    ET_CHECK_MSG(isBoolList(), "EValue is not a Bool List.");
    return std::get<exec_aten::ArrayRef<bool>>(repr_);
  }

  /****** Double List Type ******/
  /*implicit*/ EValue(exec_aten::ArrayRef<double> d) : repr_(d) {}

  bool isDoubleList() const {
    return std::holds_alternative<exec_aten::ArrayRef<double>>(repr_);
  }

  exec_aten::ArrayRef<double> toDoubleList() const {
    ET_CHECK_MSG(isDoubleList(), "EValue is not a Double List.");
    return std::get<exec_aten::ArrayRef<double>>(repr_);
  }

  /****** Tensor List Type ******/
  /*implicit*/ EValue(BoxedEvalueList<exec_aten::Tensor> t)
      : repr_(std::move(t)) {}

  bool isTensorList() const {
    return std::holds_alternative<BoxedEvalueList<exec_aten::Tensor>>(repr_);
  }

  exec_aten::ArrayRef<exec_aten::Tensor> toTensorList() const {
    ET_CHECK_MSG(isTensorList(), "EValue is not a Tensor List.");
    return std::get<BoxedEvalueList<exec_aten::Tensor>>(repr_).get();
  }

  /****** List Optional Tensor Type ******/
  /*implicit*/ EValue(BoxedEvalueList<exec_aten::optional<exec_aten::Tensor>> t)
      : repr_(std::move(t)) {}

  bool isListOptionalTensor() const {
    return std::holds_alternative<
        BoxedEvalueList<exec_aten::optional<exec_aten::Tensor>>>(repr_);
  }

  exec_aten::ArrayRef<exec_aten::optional<exec_aten::Tensor>>
  toListOptionalTensor() const {
    return std::get<BoxedEvalueList<exec_aten::optional<exec_aten::Tensor>>>(
               repr_)
        .get();
  }

  /****** ScalarType Type ******/
  exec_aten::ScalarType toScalarType() const {
    ET_CHECK_MSG(isInt(), "EValue is not a ScalarType.");
    return static_cast<exec_aten::ScalarType>(toInt());
  }

  /****** MemoryFormat Type ******/
  exec_aten::MemoryFormat toMemoryFormat() const {
    ET_CHECK_MSG(isInt(), "EValue is not a MemoryFormat.");
    return static_cast<exec_aten::MemoryFormat>(toInt());
  }

  /****** Layout Type ******/
  exec_aten::Layout toLayout() const {
    ET_CHECK_MSG(isInt(), "EValue is not a Layout.");
    return static_cast<exec_aten::Layout>(toInt());
  }

  /****** Device Type ******/
  exec_aten::Device toDevice() const {
    ET_CHECK_MSG(isInt(), "EValue is not a Device.");
    return exec_aten::Device(static_cast<exec_aten::DeviceType>(toInt()), -1);
  }

  template <typename T>
  T to() &&;
  template <typename T>
  typename evalue_to_const_ref_overload_return<T>::type to() const&;
  template <typename T>
  typename evalue_to_ref_overload_return<T>::type to() &;

  /**
   * Converts the EValue to an optional object that can represent both T and
   * an uninitialized state.
   */
  template <typename T>
  inline exec_aten::optional<T> toOptional() const {
    if (this->isNone()) {
      return exec_aten::nullopt;
    }
    return this->to<T>();
  }
};

#define EVALUE_DEFINE_TO(T, method_name)                                       \
  template <>                                                                  \
  inline T EValue::to<T>()&& {                                                 \
    return static_cast<T>(std::move(*this).method_name());                     \
  }                                                                            \
  template <>                                                                  \
  inline evalue_to_const_ref_overload_return<T>::type EValue::to<T>() const& { \
    typedef evalue_to_const_ref_overload_return<T>::type return_type;          \
    return static_cast<return_type>(this->method_name());                      \
  }                                                                            \
  template <>                                                                  \
  inline evalue_to_ref_overload_return<T>::type EValue::to<T>()& {             \
    typedef evalue_to_ref_overload_return<T>::type return_type;                \
    return static_cast<return_type>(this->method_name());                      \
  }

EVALUE_DEFINE_TO(exec_aten::Scalar, toScalar)
EVALUE_DEFINE_TO(int64_t, toInt)
EVALUE_DEFINE_TO(bool, toBool)
EVALUE_DEFINE_TO(double, toDouble)
EVALUE_DEFINE_TO(exec_aten::string_view, toString)
EVALUE_DEFINE_TO(exec_aten::ScalarType, toScalarType)
EVALUE_DEFINE_TO(exec_aten::MemoryFormat, toMemoryFormat)
EVALUE_DEFINE_TO(exec_aten::Layout, toLayout)
EVALUE_DEFINE_TO(exec_aten::Device, toDevice)
// Tensor and Optional Tensor
EVALUE_DEFINE_TO(
    exec_aten::optional<exec_aten::Tensor>,
    toOptional<exec_aten::Tensor>)
EVALUE_DEFINE_TO(exec_aten::Tensor, toTensor)

// IntList and Optional IntList
EVALUE_DEFINE_TO(exec_aten::ArrayRef<int64_t>, toIntList)
EVALUE_DEFINE_TO(
    exec_aten::optional<exec_aten::ArrayRef<int64_t>>,
    toOptional<exec_aten::ArrayRef<int64_t>>)

// DoubleList and Optional DoubleList
EVALUE_DEFINE_TO(exec_aten::ArrayRef<double>, toDoubleList)
EVALUE_DEFINE_TO(
    exec_aten::optional<exec_aten::ArrayRef<double>>,
    toOptional<exec_aten::ArrayRef<double>>)

// BoolList and Optional BoolList
EVALUE_DEFINE_TO(exec_aten::ArrayRef<bool>, toBoolList)
EVALUE_DEFINE_TO(
    exec_aten::optional<exec_aten::ArrayRef<bool>>,
    toOptional<exec_aten::ArrayRef<bool>>)

// TensorList and Optional TensorList
EVALUE_DEFINE_TO(exec_aten::ArrayRef<exec_aten::Tensor>, toTensorList)
EVALUE_DEFINE_TO(
    exec_aten::optional<exec_aten::ArrayRef<exec_aten::Tensor>>,
    toOptional<exec_aten::ArrayRef<exec_aten::Tensor>>)

// List of Optional Tensor
EVALUE_DEFINE_TO(
    exec_aten::ArrayRef<exec_aten::optional<exec_aten::Tensor>>,
    toListOptionalTensor)
#undef EVALUE_DEFINE_TO

template <typename T>
exec_aten::ArrayRef<T> BoxedEvalueList<T>::get() const {
  for (typename exec_aten::ArrayRef<T>::size_type i = 0;
       i < wrapped_vals_.size();
       i++) {
    ET_CHECK(wrapped_vals_[i] != nullptr);
    unwrapped_vals_[i] = wrapped_vals_[i]->template to<T>();
  }
  return exec_aten::ArrayRef<T>{unwrapped_vals_, wrapped_vals_.size()};
}

} // namespace executor
} // namespace torch
