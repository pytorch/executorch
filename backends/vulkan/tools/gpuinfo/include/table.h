/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/microsoft/ArchProbe/blob/main/include/table.hpp
// MIT-licensed by Rendong Liang.

// Numeric data table.
// @PENGUINLIONG
#include <string>
#include <vector>
#include "assert.h"

namespace gpuinfo {
namespace table {

struct Table {
  std::vector<std::string> headers;
  std::vector<std::vector<double>> rows;

  template <typename... THeaders>
  Table(THeaders&&... headers)
      : Table(std::vector<std::string>{std::string(headers)...}) {}
  Table(std::vector<std::string>&& headers);
  Table(
      std::vector<std::string>&& headers,
      std::vector<std::vector<double>>&& rows);

  template <typename... TArgs>
  void push(TArgs&&... values) {
    std::vector<double> row{(double)values...};
    gpuinfo::assert(
        row.size() == headers.size(), "row length mismatches header length");
    rows.emplace_back(std::move(row));
  }

  std::string to_csv(uint32_t nsig_digit = 6) const;
  static Table from_csv(std::string csv);
};

} // namespace table
} // namespace gpuinfo
