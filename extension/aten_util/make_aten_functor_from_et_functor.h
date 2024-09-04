/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
/// \file runtime/kernel/make_aten_functor_from_et_functor.h
/// Defines a template that can be used to create a ATen version of an unboxed
/// ExecuTorch kernel.
//===----------------------------------------------------------------------===//

#pragma once
#include <type_traits>
#include <vector>
#if __cplusplus < 201703L
#error "This header requires C++17"
#endif
#include <ATen/native/Resize.h>
#include <executorch/extension/kernel_util/type_list.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <torch/torch.h>

namespace executorch {
namespace extension {
namespace internal {

// Map types from ETen to ATen.
// This is used to convert ETen arguments into ATen.
template <typename T>
struct type_map final {
  using type = T;
};

// Const.
template <typename T>
struct type_map<const T> final {
  using type = const typename type_map<T>::type;
};

// Ref.
template <typename T>
struct type_map<T&> final {
  using type = typename type_map<T>::type&;
};

// Const ref.
template <typename T>
struct type_map<const T&> final {
  using type = const typename type_map<T>::type&;
};

// Tensor.
template <>
struct type_map<torch::executor::Tensor> final {
  using type = at::Tensor;
};

// Optional.
template <class T>
struct type_map<torch::executor::optional<T>> final {
  using type = c10::optional<typename type_map<T>::type>;
};

template <class T>
struct type_map<torch::executor::optional<T>&> final {
  using type = c10::optional<typename type_map<T>::type>&;
};

// ArrayRef.
template <class T>
struct type_map<torch::executor::ArrayRef<T>> final {
  using type = at::ArrayRef<typename type_map<T>::type>;
};

template <typename T>
struct remove_const_ref final {
  using type = std::remove_const_t<std::remove_reference_t<T>>;
};

// Convert ATen->ETen: input args.
// Convert ETen->ATen: output args.
// Default argument conversions between ATen and ETen (scalars).
template <typename F, typename T, typename Enable = void>
struct type_convert final {
 public:
  F val;
  explicit type_convert(F value) : val(value) {}
  T call() {
    return static_cast<T>(val);
  }
};

// Tensors: ATen to ETen.
template <class ATensor, class ETensor>
struct type_convert<
    ATensor,
    ETensor,
    std::enable_if_t<
        std::is_same_v<typename remove_const_ref<ATensor>::type, at::Tensor> &&
        std::is_same_v<
            typename remove_const_ref<ETensor>::type,
            torch::executor::Tensor>>>
    final {
  explicit type_convert(ATensor value)
      : value_(value),
        converted_(from_blob(
            value_.mutable_data_ptr(),
            {value_.sizes().begin(), value_.sizes().end()},
            ::torch::executor::ScalarType(value_.scalar_type()))) {}

  ETensor call() {
    return *converted_;
  }

 private:
  ATensor value_;
  TensorPtr converted_;
};

// Tensors: ETen to ATen.
template <class ETensor, class ATensor>
struct type_convert<
    ETensor,
    ATensor,
    std::enable_if_t<
        std::is_same_v<typename remove_const_ref<ATensor>::type, at::Tensor> &&
        std::is_same_v<
            typename remove_const_ref<ETensor>::type,
            ::torch::executor::Tensor>>>
    final {
  explicit type_convert(ETensor value)
      : value_(value),
        converted_(at::from_blob(
            value_.mutable_data_ptr(),
            std::vector<int64_t>{value_.sizes().begin(), value_.sizes().end()},
            c10::ScalarType(value_.scalar_type()))) {}

  ATensor call() {
    return converted_;
  }

 private:
  ETensor value_;
  at::Tensor converted_;
};

// Optionals: ATen to ETen.
template <class F, class T>
struct type_convert<c10::optional<F>, torch::executor::optional<T>> final {
 public:
  c10::optional<F> val;
  std::unique_ptr<struct type_convert<F, T>> convert_struct;
  explicit type_convert(c10::optional<F> value) : val(value) {}
  torch::executor::optional<T> call() {
    if (val.has_value()) {
      convert_struct = std::make_unique<struct type_convert<F, T>>(
          type_convert<F, T>(val.value()));
      return torch::executor::optional<T>(convert_struct->call());
    } else {
      return torch::executor::optional<T>();
    }
  }
};

// Optionals: ETen to ATen.
template <class F, class T>
struct type_convert<torch::executor::optional<F>, c10::optional<T>> final {
 public:
  torch::executor::optional<F> val;
  std::unique_ptr<struct type_convert<F, T>> convert_struct;
  explicit type_convert(torch::executor::optional<F> value) : val(value) {}
  c10::optional<T> call() {
    if (val.has_value()) {
      convert_struct = std::make_unique<struct type_convert<F, T>>(
          type_convert<F, T>(val.value()));
      return c10::optional<T>(convert_struct->call());
    } else {
      return c10::optional<T>();
    }
  }
};

// ArrayRefs: ATen to ETen.
template <class F, class T>
struct type_convert<c10::ArrayRef<F>, torch::executor::ArrayRef<T>> final {
 public:
  c10::ArrayRef<F> val;
  std::vector<T> converted;
  std::vector<type_convert<F, T>> converters;
  explicit type_convert(c10::ArrayRef<F> value) : val(value) {
    for (int i = 0; i < val.size(); i++) {
      converters.push_back(type_convert<F, T>(val[i]));
    }
  }
  torch::executor::ArrayRef<T> call() {
    for (int i = 0; i < val.size(); i++) {
      converted.push_back(converters[i].call());
    }
    return torch::executor::ArrayRef<T>(converted.data(), converted.size());
  }
};

// ArrayRefs: ETen to ATen.
template <class F, class T>
struct type_convert<torch::executor::ArrayRef<F>, c10::ArrayRef<T>> final {
 public:
  torch::executor::ArrayRef<F> val;
  std::vector<T> converted;
  std::vector<type_convert<F, T>> converters;
  explicit type_convert(torch::executor::ArrayRef<F> value) : val(value) {
    for (int i = 0; i < val.size(); i++) {
      converters.push_back(type_convert<F, T>(val[i]));
    }
  }
  c10::ArrayRef<T> call() {
    for (int i = 0; i < val.size(); i++) {
      converted.push_back(converters[i].call());
    }
    return c10::ArrayRef<T>(converted);
  }
};

template <class F, F f, typename N = int, N index = N(-1)>
struct wrapper_impl;

template <class R, class... Args, R (*f)(Args...), int N>
struct wrapper_impl<R (*)(Args...), f, int, N> {
  static_assert(
      !(std::is_same<R, at::Tensor&>::value && N == -1),
      "Can't wrap a kernel with 'Tensor &' return type without specifying an index to the out tensor");
  using ReturnType = typename type_map<R>::type;
  using TupleConvertsType =
      std::tuple<type_convert<typename type_map<Args>::type, Args>...>;
  using TupleArgsType = std::tuple<typename type_map<Args>::type...>;
  static constexpr size_t num_args = sizeof...(Args);
  static_assert(
      (N < num_args &&
       std::is_same_v<
           executorch::extension::kernel_util_internal::element_t<
               N,
               executorch::extension::kernel_util_internal::typelist<Args...>>,
           R>) ||
          N == -1,
      "The index of the out tensor can't be greater or equal to num_args and "
      "the Nth argument type has to be the same as the return type.");

  static ReturnType wrap(typename type_map<Args>::type... args) {
    // The wrapped function that takes ATen argument types, convert them into
    // ExecuTorch equivalent, call `f` then return the result converted back to
    // ATen.
    TupleArgsType args_tuple = std::forward_as_tuple(args...);
    TupleConvertsType converts = std::forward_as_tuple(
        type_convert<typename type_map<Args>::type, Args>(args)...);
    R result =
        call_functor_with_args(converts, std::make_index_sequence<num_args>());
    typename std::remove_reference<ReturnType>::type converted_result =
        type_convert<R, ReturnType>(result).call();
    if constexpr (N == -1) {
      return converted_result;
    } else {
      static_assert(
          std::is_same_v<
              typename std::remove_reference<ReturnType>::type,
              at::Tensor>,
          "Only support at::Tensor-like return");
      ReturnType out = std::get<N>(args_tuple);
      at::native::resize_output(out, converted_result.sizes());
      out.copy_(converted_result);
      return out;
    }
  }

 private:
  template <size_t... indices>
  static R call_functor_with_args(
      TupleConvertsType& converts,
      std::index_sequence<indices...>) {
    return f(std::get<indices>(converts).call()...);
  }
};

} // namespace internal
} // namespace extension
} // namespace executorch

// Wrapper macro for out variant function. N is the index of the out tensor.
// We need N to know how to preserve the semantics of modifying out tensor and
// return the reference without allocating a new memory buffer for out tensor.
#define _WRAP_2(func, N)              \
  ::executorch::extension::internal:: \
      wrapper_impl<decltype(&func), func, decltype(N), N>::wrap
#define _WRAP_1(func) \
  ::executorch::extension::internal::wrapper_impl<decltype(&func), func>::wrap

#define _GET_MACRO(_1, _2, NAME, ...) NAME
#define WRAP_TO_ATEN(...) _GET_MACRO(__VA_ARGS__, _WRAP_2, _WRAP_1)(__VA_ARGS__)
