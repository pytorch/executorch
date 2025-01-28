/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/evalue_util/print_evalue.h>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>

using exec_aten::ScalarType;

namespace executorch {
namespace extension {

namespace {

/// Number of list items on a line before wrapping.
constexpr size_t kItemsPerLine = 10;

/// The default number of first/last list items to print before eliding.
constexpr size_t kDefaultEdgeItems = 3;

/// Returns a globally unique "iword" index that we can use to store the current
/// "edge items" count on arbitrary streams.
int get_edge_items_xalloc() {
  // Wrapping this in a function avoids a -Wglobal-constructors warning.
  static const int xalloc = std::ios_base::xalloc();
  return xalloc;
}

/// Returns the number of "edge items" to print at the beginning and end of
/// lists when using the provided stream.
long get_stream_edge_items(std::ostream& os) {
  long edge_items = os.iword(get_edge_items_xalloc());
  return edge_items <= 0 ? kDefaultEdgeItems : edge_items;
}

void print_double(std::ostream& os, double value) {
  if (std::isfinite(value)) {
    // Mimic PyTorch by printing a trailing dot when the float value is
    // integral, to distinguish from actual integers.
    bool add_dot = false;
    if (value == -0.0) {
      // Special case that won't be detected by a comparison with int.
      add_dot = true;
    } else {
      std::ostringstream oss_float;
      oss_float << value;
      std::ostringstream oss_int;
      oss_int << static_cast<int64_t>(value);
      if (oss_float.str() == oss_int.str()) {
        add_dot = true;
      }
    }
    if (add_dot) {
      os << value << ".";
    } else {
      os << value;
    }
  } else {
    // Infinity or NaN.
    os << value;
  }
}

template <class T>
void print_scalar_list(
    std::ostream& os,
    exec_aten::ArrayRef<T> list,
    bool print_length = true,
    bool elide_inner_items = true) {
  long edge_items = elide_inner_items ? get_stream_edge_items(os)
                                      : std::numeric_limits<long>::max();
  if (print_length) {
    os << "(len=" << list.size() << ")";
  }

  // See if we'll be printing enough elements to cause us to wrap.
  bool wrapping = false;
  {
    long num_printed_items;
    if (elide_inner_items) {
      num_printed_items =
          std::min(static_cast<long>(list.size()), edge_items * 2);
    } else {
      num_printed_items = static_cast<long>(list.size());
    }
    wrapping = num_printed_items > kItemsPerLine;
  }

  os << "[";
  size_t num_printed = 0;
  for (size_t i = 0; i < list.size(); ++i) {
    if (wrapping && num_printed % kItemsPerLine == 0) {
      // We've printed a full line, so wrap and begin a new one.
      os << "\n  ";
    }
    os << executorch::runtime::EValue(exec_aten::Scalar(list[i]));
    if (wrapping || i < list.size() - 1) {
      // No trailing comma when not wrapping. Always a trailing comma when
      // wrapping. This will leave a trailing space at the end of every wrapped
      // line, but it simplifies the logic here.
      os << ", ";
    }
    ++num_printed;
    if (i + 1 == edge_items && i + edge_items + 1 < list.size()) {
      if (wrapping) {
        os << "\n  ...,";
        // Make the first line after the elision be the ragged line, letting us
        // always end on a full line.
        num_printed = kItemsPerLine - edge_items % kItemsPerLine;
        if (num_printed % kItemsPerLine != 0) {
          // If the line ended exactly when the elision happened, the next
          // iteration of the loop will add this line break.
          os << "\n  ";
        }
      } else {
        // Non-wrapping elision.
        os << "..., ";
      }
      i = list.size() - edge_items - 1;
    }
  }
  if (wrapping) {
    // End the current line.
    os << "\n";
  }
  os << "]";
}

void print_tensor(std::ostream& os, exec_aten::Tensor tensor) {
  os << "tensor(sizes=";
  // Always print every element of the sizes list.
  print_scalar_list(
      os, tensor.sizes(), /*print_length=*/false, /*elide_inner_items=*/false);
  os << ", ";

  // Print the data as a one-dimensional list.
  //
  // TODO(T159700776): Print dim_order and strides when they have non-default
  // values.
  //
  // TODO(T159700776): Format multidimensional data like numpy/PyTorch does.
  // https://github.com/pytorch/pytorch/blob/main/torch/_tensor_str.py
#define PRINT_TENSOR_DATA(ctype, dtype)                      \
  case ScalarType::dtype:                                    \
    print_scalar_list(                                       \
        os,                                                  \
        exec_aten::ArrayRef<ctype>(                          \
            tensor.const_data_ptr<ctype>(), tensor.numel()), \
        /*print_length=*/false);                             \
    break;

  switch (tensor.scalar_type()) {
    ET_FORALL_REAL_TYPES_AND2(Bool, Half, PRINT_TENSOR_DATA)
    default:
      os << "[<unhandled scalar type " << (int)tensor.scalar_type() << ">]";
  }
  os << ")";

#undef PRINT_TENSOR_DATA
}

void print_tensor_list(
    std::ostream& os,
    exec_aten::ArrayRef<exec_aten::Tensor> list) {
  os << "(len=" << list.size() << ")[";
  for (size_t i = 0; i < list.size(); ++i) {
    if (list.size() > 1) {
      os << "\n  [" << i << "]: ";
    }
    print_tensor(os, list[i]);
    if (list.size() > 1) {
      os << ",";
    }
  }
  if (list.size() > 1) {
    os << "\n";
  }
  os << "]";
}

void print_list_optional_tensor(
    std::ostream& os,
    exec_aten::ArrayRef<exec_aten::optional<exec_aten::Tensor>> list) {
  os << "(len=" << list.size() << ")[";
  for (size_t i = 0; i < list.size(); ++i) {
    if (list.size() > 1) {
      os << "\n  [" << i << "]: ";
    }
    if (list[i].has_value()) {
      print_tensor(os, list[i].value());
    } else {
      os << "None";
    }
    if (list.size() > 1) {
      os << ",";
    }
  }
  if (list.size() > 1) {
    os << "\n";
  }
  os << "]";
}

} // namespace

void evalue_edge_items::set_edge_items(std::ostream& os, long edge_items) {
  os.iword(get_edge_items_xalloc()) = edge_items;
}

} // namespace extension
} // namespace executorch

namespace executorch {
namespace runtime {

// This needs to live in the same namespace as EValue.
std::ostream& operator<<(std::ostream& os, const EValue& value) {
  using namespace executorch::extension;

  switch (value.tag) {
    case Tag::None:
      os << "None";
      break;
    case Tag::Bool:
      if (value.toBool()) {
        os << "True";
      } else {
        os << "False";
      }
      break;
    case Tag::Int:
      os << value.toInt();
      break;
    case Tag::Double:
      print_double(os, value.toDouble());
      break;
    case Tag::String: {
      auto str = value.toString();
      os << std::quoted(std::string(str.data(), str.size()));
    } break;
    case Tag::Tensor:
      print_tensor(os, value.toTensor());
      break;
    case Tag::ListBool:
      print_scalar_list(os, value.toBoolList());
      break;
    case Tag::ListInt:
      print_scalar_list(os, value.toIntList());
      break;
    case Tag::ListDouble:
      print_scalar_list(os, value.toDoubleList());
      break;
    case Tag::ListTensor:
      print_tensor_list(os, value.toTensorList());
      break;
    case Tag::ListOptionalTensor:
      print_list_optional_tensor(os, value.toListOptionalTensor());
      break;
    default:
      os << "<Unknown EValue tag " << static_cast<int>(value.tag) << ">";
      break;
  }
  return os;
}

} // namespace runtime
} // namespace executorch
