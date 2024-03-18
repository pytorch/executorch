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
#if __cplusplus < 201703L
#error "This header requires C++17"
#endif
#include <ATen/native/Resize.h>
#include <executorch/extension/kernel_util/type_list.h>
#include <executorch/extension/runner_util/managed_tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <torch/torch.h>

namespace torch {
namespace executor {

class KernelRuntimeContext; // Forward declaration
using RuntimeContext = KernelRuntimeContext; // TODO(T147221312): Remove

template <typename T>
struct type_map final {
  using type = T;
};

template <>
struct type_map<torch::executor::Tensor&> final {
  using type = at::Tensor&;
};

template <>
struct type_map<const torch::executor::Tensor&> final {
  using type = const at::Tensor&;
};

template <typename F, typename T, typename Enable = void>
struct type_convert final {
 public:
  F val;
  explicit type_convert(F value) : val(value) {}
  T call() {
    return static_cast<T>(val);
  }
};

template <typename T>
struct remove_const_ref final {
  using type = std::remove_const_t<std::remove_reference_t<T>>;
};

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
 public:
  ATensor val;
  std::unique_ptr<ManagedTensor> managed_tensor;
  torch::executor::Tensor converted;
  std::vector<exec_aten::SizesType> sizes;
  explicit type_convert(ATensor value)
      : val(value), converted(torch::executor::Tensor(nullptr)) {
    for (auto size : val.sizes()) {
      sizes.push_back(size);
    }
    torch::executor::ScalarType scalar_type =
        static_cast<torch::executor::ScalarType>(val.scalar_type());
    managed_tensor = std::make_unique<ManagedTensor>(
        val.mutable_data_ptr(), val.numel(), sizes, scalar_type);
    converted = managed_tensor->get_aliasing_tensor();
  }
  ETensor call() {
    return converted;
  }
};

template <>
struct type_convert<torch::executor::Tensor&, at::Tensor&> final {
 public:
  torch::executor::Tensor& val;
  at::Tensor converted;
  std::vector<int64_t> sizes;
  explicit type_convert(torch::executor::Tensor& value) : val(value) {
    for (auto size : val.sizes()) {
      sizes.push_back(size);
    }
    c10::ScalarType scalar_type =
        static_cast<c10::ScalarType>(val.scalar_type());
    converted =
        at::from_blob(val.mutable_data_ptr(), val.numel(), sizes, scalar_type);
  }
  at::Tensor& call() {
    return converted;
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
      (N < num_args && std::is_same_v<element_t<N, typelist<Args...>>, R>) ||
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

} // namespace executor
} // namespace torch

// Wrapper macro for out variant function. N is the index of the out tensor.
// We need N to know how to preserve the semantics of modifying out tensor and
// return the reference without allocating a new memory buffer for out tensor.
#define _WRAP_2(func, N) \
  ::torch::executor::wrapper_impl<decltype(&func), func, decltype(N), N>::wrap
#define _WRAP_1(func) \
  ::torch::executor::wrapper_impl<decltype(&func), func>::wrap

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define WRAP_TO_ATEN(...) GET_MACRO(__VA_ARGS__, _WRAP_2, _WRAP_1)(__VA_ARGS__)
