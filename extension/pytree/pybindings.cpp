/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <stack>

#include "executorch/extension/pytree/pytree.h"

namespace py = pybind11;

namespace executorch {
namespace extension {
namespace pytree {

namespace {

struct PyAux {
  py::object custom_type_context;
};
using PyTreeSpec = TreeSpec<PyAux>;

class PyTypeRegistry {
 public:
  struct PyTypeReg {
    explicit PyTypeReg(Kind k) : kind(k) {}

    Kind kind;

    // for custom types
    py::object type;
    // function type: object -> (children, spec_data)
    py::function flatten;
    // function type: (children, spec_data) -> object
    py::function unflatten;
  };

  static const PyTypeReg* get_by_str(const std::string& pytype) {
    auto* registry = instance();
    auto it = registry->regs_.find(pytype);
    return it == registry->regs_.end() ? nullptr : it->second.get();
  }

  static const PyTypeReg* get_by_type(py::handle pytype) {
    return get_by_str(py::str(pytype));
  }

  static void register_custom_type(
      py::object type,
      py::function flatten,
      py::function unflatten) {
    auto* registry = instance();
    auto reg = std::make_unique<PyTypeReg>(Kind::Custom);
    reg->type = type;
    reg->flatten = std::move(flatten);
    reg->unflatten = std::move(unflatten);
    std::string pytype_str = py::str(type);
    auto it = registry->regs_.emplace(pytype_str, std::move(reg));
    if (!it.second) {
      assert(false);
    }
  }

 private:
  static PyTypeRegistry* instance() {
    static auto* registry_instance = []() -> PyTypeRegistry* {
      auto* registry = new PyTypeRegistry;

      auto add_pytype_reg = [&](const std::string& pytype, Kind kind) {
        registry->regs_.emplace(pytype, std::make_unique<PyTypeReg>(kind));
      };

      add_pytype_reg("<class 'tuple'>", Kind::Tuple);
      add_pytype_reg("<class 'list'>", Kind::List);
      add_pytype_reg("<class 'dict'>", Kind::Dict);

      return registry;
    }();

    return registry_instance;
  }
  std::unordered_map<std::string, std::unique_ptr<PyTypeReg>> regs_;
};

class PyTree {
  PyTreeSpec spec_;

  static void flatten_internal(
      py::handle x,
      std::vector<py::object>& leaves,
      PyTreeSpec& s) {
    const auto* reg = PyTypeRegistry::get_by_type(x.get_type());
    const auto kind = [&reg, &x]() {
      if (reg) {
        return reg->kind;
      }
      if (py::isinstance<py::tuple>(x) && py::hasattr(x, "_fields")) {
        return Kind::NamedTuple;
      }
      return Kind::Leaf;
    }();
    switch (kind) {
      case Kind::List: {
        const size_t n = PyList_GET_SIZE(x.ptr());
        s = PyTreeSpec(Kind::List, n);
        for (size_t i = 0; i < n; ++i) {
          flatten_internal(PyList_GET_ITEM(x.ptr(), i), leaves, s[i]);
        }
        break;
      }
      case Kind::Tuple: {
        const size_t n = PyTuple_GET_SIZE(x.ptr());
        s = PyTreeSpec(Kind::Tuple, n);
        for (size_t i = 0; i < n; ++i) {
          flatten_internal(PyTuple_GET_ITEM(x.ptr(), i), leaves, s[i]);
        }
        break;
      }
      case Kind::NamedTuple: {
        py::tuple tuple = py::reinterpret_borrow<py::tuple>(x);
        const size_t n = tuple.size();
        s = PyTreeSpec(Kind::NamedTuple, n);
        size_t i = 0;
        for (py::handle entry : tuple) {
          flatten_internal(entry, leaves, s[i++]);
        }
        break;
      }
      case Kind::Dict: {
        py::dict dict = py::reinterpret_borrow<py::dict>(x);
        py::list keys =
            py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
        const auto n = PyList_GET_SIZE(keys.ptr());
        s = PyTreeSpec(Kind::Dict, n);
        size_t i = 0;
        for (py::handle key : keys) {
          if (py::isinstance<py::str>(key)) {
            s.key(i) = py::cast<std::string>(key);
          } else if (py::isinstance<py::int_>(key)) {
            s.key(i) = py::cast<int32_t>(key);
          } else {
            pytree_assert(false);
          }

          flatten_internal(dict[key], leaves, s[i]);
          i++;
        }
        break;
      }
      case Kind::Custom: {
        py::tuple out = py::cast<py::tuple>(reg->flatten(x));
        if (out.size() != 2) {
          assert(false);
        }
        py::list children = py::cast<py::list>(out[0]);
        const size_t n = children.size();
        s = PyTreeSpec(Kind::Custom, n);
        s.handle->custom_type = py::str(x.get_type());
        s.handle->custom_type_context = out[1];
        size_t i = 0;
        for (py::handle pychild : children) {
          flatten_internal(pychild, leaves, s[i++]);
        }
        break;
      }
      case Kind::Leaf: {
        s = PyTreeSpec(Kind::Leaf);
        leaves.push_back(py::reinterpret_borrow<py::object>(x));
        break;
      }
      case Kind::None:
        pytree_assert(false);
    }
  }

  template <typename T>
  py::object unflatten_internal(const PyTreeSpec& spec, T&& leaves_it) const {
    switch (spec.kind()) {
      case Kind::NamedTuple:
      case Kind::Tuple: {
        const size_t size = spec.size();
        py::tuple tuple(size);
        for (size_t i = 0; i < size; ++i) {
          tuple[i] = unflatten_internal(spec[i], leaves_it);
        }
        return std::move(tuple);
      }
      case Kind::List: {
        const size_t size = spec.size();
        py::list list(size);
        for (size_t i = 0; i < size; ++i) {
          list[i] = unflatten_internal(spec[i], leaves_it);
        }
        return std::move(list);
      }
      case Kind::Custom: {
        const auto& pytype_str = spec.handle->custom_type;
        const auto* reg = PyTypeRegistry::get_by_str(pytype_str);
        const size_t size = spec.size();
        py::list list(size);
        for (size_t i = 0; i < size; ++i) {
          list[i] = unflatten_internal(spec[i], leaves_it);
        }
        py::object o = reg->unflatten(list, spec.handle->custom_type_context);
        return o;
      }
      case Kind::Dict: {
        const size_t size = spec.size();
        py::dict dict;
        for (size_t i = 0; i < size; ++i) {
          auto& key = spec.key(i);
          auto py_key = [&key]() -> py::handle {
            switch (key.kind()) {
              case Key::Kind::Int:
                return py::cast(key.as_int()).release();
              case Key::Kind::Str:
                return py::cast(key.as_str()).release();
              case Key::Kind::None:
                pytree_assert(false);
            }
            pytree_assert(false);
            return py::none();
          }();
          dict[py_key] = unflatten_internal(spec[i], leaves_it);
        }
        return std::move(dict);
      }
      case Kind::Leaf: {
        py::object o =
            py::reinterpret_borrow<py::object>(*std::forward<T>(leaves_it));
        leaves_it++;
        return o;
      }
      case Kind::None: {
        return py::none();
      }
    }
    pytree_assert(false);
  }

 public:
  explicit PyTree(PyTreeSpec spec) : spec_(std::move(spec)) {}

  const PyTreeSpec& spec() const {
    return spec_;
  }

  static PyTree py_from_str(std::string spec) {
    return PyTree(from_str<PyAux>(spec));
  }

  StrTreeSpec py_to_str() const {
    return to_str(spec_);
  }

  static std::pair<std::vector<py::object>, std::unique_ptr<PyTree>>
  tree_flatten(py::handle x) {
    std::vector<py::object> leaves{};
    PyTreeSpec spec{};
    flatten_internal(x, leaves, spec);
    refresh_leaves_num(spec);
    return {std::move(leaves), std::make_unique<PyTree>(std::move(spec))};
  }

  static py::object tree_unflatten(py::iterable leaves, py::object o) {
    return o.cast<PyTree*>()->tree_unflatten(leaves);
  }

  template <typename T>
  py::object tree_unflatten(T leaves) const {
    return unflatten_internal(spec_, leaves.begin());
  }

  bool operator==(const PyTree& rhs) {
    return spec_ == rhs.spec_;
  }

  size_t leaves_num() const {
    return refresh_leaves_num(spec_);
  }
};

inline std::pair<std::vector<py::object>, std::unique_ptr<PyTree>> tree_flatten(
    py::handle x) {
  return PyTree::tree_flatten(x);
}

inline py::object tree_unflatten(py::iterable leaves, py::object o) {
  return PyTree::tree_unflatten(leaves, o);
}

static py::object tree_map(py::function& fn, py::handle x) {
  auto p = tree_flatten(x);
  const auto& leaves = p.first;
  const auto& pytree = p.second;
  std::vector<py::handle> vec;
  for (const py::handle& h : leaves) {
    vec.push_back(fn(h));
  }
  return pytree->tree_unflatten(vec);
}

static std::unique_ptr<PyTree> py_from_str(std::string spec) {
  return std::make_unique<PyTree>(from_str<PyAux>(spec));
}

static py::object broadcast_to_and_flatten(
    py::object x,
    py::object py_tree_spec) {
  auto p = tree_flatten(x);
  const auto& x_leaves = p.first;
  const auto& x_spec = p.second->spec();

  PyTree* tree_spec = py_tree_spec.cast<PyTree*>();

  py::list ret;
  struct StackItem {
    const PyTreeSpec* tree_spec_node;
    const PyTreeSpec* x_spec_node;
    const size_t x_leaves_offset;
  };
  std::stack<StackItem> stack;
  stack.push({&tree_spec->spec(), &x_spec, 0u});
  while (!stack.empty()) {
    const auto top = stack.top();
    stack.pop();
    if (top.x_spec_node->isLeaf()) {
      for (size_t i = 0; i < top.tree_spec_node->leaves_num(); ++i) {
        ret.append(x_leaves[top.x_leaves_offset]);
      }
    } else {
      const auto kind = top.tree_spec_node->kind();
      if (kind != top.x_spec_node->kind()) {
        return py::none();
      }
      pytree_assert(top.tree_spec_node->kind() == top.x_spec_node->kind());
      const size_t child_num = top.tree_spec_node->size();
      if (child_num != top.x_spec_node->size()) {
        return py::none();
      }
      pytree_assert(child_num == top.x_spec_node->size());

      size_t x_leaves_offset =
          top.x_leaves_offset + top.x_spec_node->leaves_num();
      auto fn_i = [&](size_t i) {
        x_leaves_offset -= (*top.x_spec_node)[i].leaves_num();
        stack.push(
            {&(*top.tree_spec_node)[i],
             &(*top.x_spec_node)[i],
             x_leaves_offset});
      };
      if (Kind::Dict == kind) {
        for (size_t i = child_num - 1; i < child_num; --i) {
          if (top.tree_spec_node->key(i) != top.x_spec_node->key(i)) {
            return py::none();
          }
          fn_i(i);
        }
      } else {
        for (size_t i = child_num - 1; i < child_num; --i) {
          fn_i(i);
        }
      }
    }
  }
  return std::move(ret);
}

} // namespace

PYBIND11_MODULE(pybindings, m) {
  m.def("tree_flatten", &tree_flatten, py::arg("tree"));
  m.def("tree_unflatten", &tree_unflatten, py::arg("leaves"), py::arg("tree"));
  m.def("tree_map", &tree_map);
  m.def("from_str", &py_from_str);
  m.def("broadcast_to_and_flatten", &broadcast_to_and_flatten);
  m.def("register_custom", &PyTypeRegistry::register_custom_type);

  py::class_<PyTree>(m, "TreeSpec")
      .def("from_str", &PyTree::py_from_str)
      .def(
          "tree_unflatten",
          static_cast<py::object (PyTree::*)(py::iterable leaves) const>(
              &PyTree::tree_unflatten))
      .def("__repr__", &PyTree::py_to_str)
      .def("__eq__", &PyTree::operator==)
      .def("to_str", &PyTree::py_to_str)
      .def("num_leaves", &PyTree::leaves_num);
}

} // namespace pytree
} // namespace extension
} // namespace executorch
