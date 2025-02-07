/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ostream>

#include <executorch/runtime/core/evalue.h>

namespace executorch {
namespace runtime {
/**
 * Prints an Evalue to a stream.
 */
std::ostream& operator<<(std::ostream& os, const EValue& value);
// Note that this must be declared in the same namespace as EValue.
} // namespace runtime
} // namespace executorch

namespace executorch {
namespace extension {

/**
 * Sets the number of "edge items" when printing EValue lists to a stream.
 *
 * The edge item count is used to elide inner elements from large lists, and
 * like core PyTorch defaults to 3.
 *
 * For example,
 * ```
 * os << torch::executor::util::evalue_edge_items(3) << evalue_int_list << "\n";
 * os << torch::executor::util::evalue_edge_items(1) << evalue_int_list << "\n";
 * ```
 * will print the same list with three edge items, then with only one edge item:
 * ```
 * [0, 1, 2,  ..., 6, 7, 8]
 * [0,  ..., 8]
 * ```
 * This setting is sticky, and will affect all subsequent evalues printed to the
 * affected stream until the value is changed again.
 *
 * @param[in] os The stream to modify.
 * @param[in] edge_items The number of "edge items" to print at the beginning
 *     and end of a list before eliding inner elements. If zero or negative,
 *     uses the default number of edge items.
 */
class evalue_edge_items final {
  // See https://stackoverflow.com/a/29337924 for other examples of stream
  // manipulators like this.
 public:
  explicit evalue_edge_items(long edge_items)
      : edge_items_(edge_items < 0 ? 0 : edge_items) {}

  friend std::ostream& operator<<(
      std::ostream& os,
      const evalue_edge_items& e) {
    set_edge_items(os, e.edge_items_);
    return os;
  }

 private:
  static void set_edge_items(std::ostream& os, long edge_items);

  const long edge_items_;
};

} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
namespace util {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::evalue_edge_items;
} // namespace util
} // namespace executor
} // namespace torch
