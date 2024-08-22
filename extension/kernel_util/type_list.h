/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

///
/// \file runtime/kernel/type_list.h
/// Forked from pytorch/c10/util/TypeList.h
/// \brief Utilities for working with type lists.
#pragma once
#if __cplusplus < 201703L
#error "This header requires C++17"
#endif

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace executorch {
namespace extension {
// This extension has a lot of generic internal names like "size"; use a unique
// internal namespace to avoid conflicts with other extensions.
namespace kernel_util_internal {

/**
 * Type holding a list of types for compile time type computations
 *     constexpr size_t num = size<typelist<int, double>>::value;
 *     static_assert(num == 2, "");
 */
template <class... T>
struct false_t : std::false_type {};

template <class... Items>
struct typelist final {
 public:
  typelist() = delete; // not for instantiation
};
template <class TypeList>
struct size final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::size<T>, T must be typelist<...>.");
};
template <class... Types>
struct size<typelist<Types...>> final {
  static constexpr size_t value = sizeof...(Types);
};

/**
 * is_instantiation_of<T, I> is true_type iff I is a template instantiation of T
 * (e.g. vector<int> is an instantiation of vector) Example:
 *    is_instantiation_of_t<vector, vector<int>> // true
 *    is_instantiation_of_t<pair, pair<int, string>> // true
 *    is_instantiation_of_t<vector, pair<int, string>> // false
 */
template <template <class...> class Template, class T>
struct is_instantiation_of : std::false_type {};
template <template <class...> class Template, class... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};
template <template <class...> class Template, class T>
using is_instantiation_of_t = typename is_instantiation_of<Template, T>::type;

/// Base template.
template <size_t Index, class TypeList>
struct element final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::element<T>, the T argument must be typelist<...>.");
};
/// Successful case, we have reached the zero index and can "return" the head
/// type.
template <class Head, class... Tail>
struct element<0, typelist<Head, Tail...>> {
  using type = Head;
};
/// Error case, we have an index but ran out of types! It will only be selected
/// if `Ts...` is actually empty!
template <size_t Index, class... Ts>
struct element<Index, typelist<Ts...>> {
  static_assert(
      Index < sizeof...(Ts),
      "Index is out of bounds in typelist::element");
};
/// Shave off types until we hit the <0, Head, Tail...> or <Index> case.
template <size_t Index, class Head, class... Tail>
struct element<Index, typelist<Head, Tail...>>
    : element<Index - 1, typelist<Tail...>> {};
/// Convenience alias.
template <size_t Index, class TypeList>
using element_t = typename element<Index, TypeList>::type;

/**
 * Returns the first element of a type list, or the specified default if the
 * type list is empty. Example: int  ==  head_t<bool, typelist<int, string>>
 *   bool  ==  head_t<bool, typelist<>>
 */
template <class Default, class TypeList>
struct head_with_default final {
  using type = Default;
};
template <class Default, class Head, class... Tail>
struct head_with_default<Default, typelist<Head, Tail...>> final {
  using type = Head;
};
template <class Default, class TypeList>
using head_with_default_t = typename head_with_default<Default, TypeList>::type;

/**
 * Take/drop a number of arguments from a typelist.
 * Example:
 *   typelist<int, string> == take_t<typelist<int, string, bool>, 2>
 *   typelist<bool> == drop_t<typelist<int, string, bool>, 2>
 */
template <class TypeList, size_t offset, class IndexSequence>
struct take_elements final {};
template <class TypeList, size_t offset, size_t... Indices>
struct take_elements<TypeList, offset, std::index_sequence<Indices...>> final {
  using type = typelist<typename element<offset + Indices, TypeList>::type...>;
};

/**
 * Like drop, but returns an empty list rather than an assertion error if `num`
 * is larger than the size of the TypeList.
 * Example:
 *   typelist<> == drop_if_nonempty_t<typelist<string, bool>, 2>
 *   typelist<> == drop_if_nonempty_t<typelist<int, string, bool>, 3>
 */
template <class TypeList, size_t num>
struct drop_if_nonempty final {
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::drop<T, num>, the T argument must be typelist<...>.");
  using type = typename take_elements<
      TypeList,
      std::min(num, size<TypeList>::value),
      std::make_index_sequence<
          size<TypeList>::value - std::min(num, size<TypeList>::value)>>::type;
};
template <class TypeList, size_t num>
using drop_if_nonempty_t = typename drop_if_nonempty<TypeList, num>::type;

} // namespace kernel_util_internal
} // namespace extension
} // namespace executorch
