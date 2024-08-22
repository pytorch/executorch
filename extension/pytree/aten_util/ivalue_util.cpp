/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/pytree/aten_util/ivalue_util.h>

#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace extension {

using namespace c10;
using namespace at;
using namespace executorch::extension::pytree;

ContainerHandle<IValue> getContainerHandle(const IValue& data) {
  if (data.isList()) {
    const auto& values = data.toList();
    auto c = ContainerHandle<IValue>(Kind::List, values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      c[i] = getContainerHandle(values[i]);
    }
    return c;
  }

  if (data.isTuple()) {
    const auto& values = data.toTupleRef().elements();
    auto c = ContainerHandle<IValue>(Kind::Tuple, values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      c[i] = getContainerHandle(values[i]);
    }
    return c;
  }

  if (data.isGenericDict()) {
    const auto& dict = data.toGenericDict();
    auto c = ContainerHandle<IValue>(Kind::Dict, dict.size());

    size_t i = 0;
    for (const auto& entry : dict) {
      const auto& key = entry.key().toStringRef();
      const auto& value = entry.value();

      c.key(i) = Key(key);
      c[i] = getContainerHandle(value);
      ++i;
    }
    return c;
  }

  return const_cast<IValue*>(&data);
}

template <std::size_t... Is>
auto create_tuple_impl(
    std::index_sequence<Is...>,
    const std::vector<IValue>& arguments) {
  return std::make_tuple(arguments[Is]...);
}

template <std::size_t N>
auto create_tuple(const std::vector<IValue>& arguments) {
  return create_tuple_impl(std::make_index_sequence<N>{}, arguments);
}

IValue constructTuple(const std::vector<IValue>& ivalues) {
  switch (ivalues.size()) {
    case 1:
      return create_tuple<1>(ivalues);
    case 2:
      return create_tuple<2>(ivalues);
    case 3:
      return create_tuple<3>(ivalues);
    case 4:
      return create_tuple<4>(ivalues);
    case 5:
      return create_tuple<5>(ivalues);
    case 6:
      return create_tuple<6>(ivalues);
    case 7:
      return create_tuple<7>(ivalues);
    case 8:
      return create_tuple<8>(ivalues);
    case 9:
      return create_tuple<9>(ivalues);
    case 10:
      return create_tuple<10>(ivalues);
  }
  ET_ASSERT_UNREACHABLE_MSG("Supports at most 10 inputs");
  return {};
}

IValue toIValue(const ContainerHandle<IValue>& c) {
  if (c.isList()) {
    auto ivalues = c10::impl::GenericList(c10::AnyType::get());
    for (size_t i = 0; i < c.size(); ++i) {
      ivalues.emplace_back(toIValue(c[i]));
    }
    return ivalues;
  }

  if (c.isTuple()) {
    std::vector<IValue> ivalues;
    for (size_t i = 0; i < c.size(); ++i) {
      ivalues.emplace_back(toIValue(c[i]));
    }
    return constructTuple(ivalues);
  }

  if (c.isDict()) {
    auto dict =
        c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get());
    for (size_t i = 0; i < c.size(); ++i) {
      dict.insert(std::string(c.key(i)), toIValue(c[i]));
    }
    return dict;
  }

  ET_CHECK(c.isLeaf());
  return {*c.leaf_ptr()};
}

std::pair<std::vector<at::Tensor>, std::unique_ptr<TreeSpec<Empty>>> flatten(
    const IValue& data) {
  auto c = getContainerHandle(data);

  auto p = flatten(c);

  std::vector<at::Tensor> tensors;
  for (int i = 0; i < p.first.size(); ++i) {
    tensors.emplace_back(p.first[i]->toTensor());
  }

  return {tensors, std::move(p.second)};
}

IValue unflatten(
    const std::vector<at::Tensor>& tensors,
    const std::unique_ptr<TreeSpec<Empty>>& tree_spec) {
  std::vector<IValue> ivalues;
  for (const auto& tensor : tensors) {
    ivalues.emplace_back(tensor);
  }
  ContainerHandle<IValue> c = unflatten(*tree_spec, ivalues.data());
  return toIValue(c);
}

bool is_same(
    const std::vector<at::Tensor>& a,
    const std::vector<at::Tensor>& b) {
  for (int i = 0; i < a.size(); ++i) {
    if (!at::all(a[i] == b[i]).item<bool>()) {
      return false;
    }
  }
  return true;
}

bool is_same(const IValue& lhs, const IValue& rhs) {
  if (lhs.isList() && rhs.isList()) {
    const auto& l = lhs.toList();
    const auto& r = rhs.toList();
    if (l.size() != r.size()) {
      return false;
    }
    for (size_t i = 0; i < l.size(); ++i) {
      if (!is_same(l[i], r[i])) {
        return false;
      }
    }
    return true;
  }

  if (lhs.isTuple() && rhs.isTuple()) {
    const auto& l = lhs.toTupleRef().elements();
    const auto& r = rhs.toTupleRef().elements();
    if (l.size() != r.size()) {
      return false;
    }
    for (size_t i = 0; i < l.size(); ++i) {
      if (!is_same(l[i], r[i])) {
        return false;
      }
    }
    return true;
  }

  if (lhs.isGenericDict() && rhs.isGenericDict()) {
    const auto& lhs_dict = lhs.toGenericDict();
    const auto& rhs_dict = rhs.toGenericDict();
    if (lhs_dict.size() != rhs_dict.size()) {
      return false;
    }

    for (const auto& entry : lhs_dict) {
      if (!rhs_dict.contains(entry.key())) {
        return false;
      }
      if (!is_same(entry.value(), rhs_dict.at(entry.key()))) {
        return false;
      }
    }
    return true;
  }

  ET_CHECK(lhs.isTensor() && rhs.isTensor());
  const auto& l = lhs.toTensor();
  const auto& r = rhs.toTensor();
  return at::all(l == r).item<bool>();
}

} // namespace extension
} // namespace executorch
