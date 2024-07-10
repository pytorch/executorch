/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NOTE: This is a modified excerpt of
//  https://github.com/microsoft/ArchProbe/blob/main/src/table.cpp
// MIT-licensed by Rendong Liang.

#include <iomanip>

#include "table.h"

namespace gpuinfo {
namespace table {

Table::Table(std::vector<std::string>&& headers) : headers(headers), rows() {}
Table::Table(
    std::vector<std::string>&& headers,
    std::vector<std::vector<double>>&& rows)
    : headers(headers), rows(rows) {}

std::string Table::to_csv(uint32_t nsig_digit) const {
  std::stringstream ss;
  ss << std::setprecision(nsig_digit);

  {
    bool first_col = true;
    for (const auto& header : headers) {
      if (first_col) {
        first_col = false;
      } else {
        ss << ",";
      }
      ss << header;
    }
    // Enforce newline character to be `\n`.
    ss << '\n';
  }

  {
    for (const auto& row : rows) {
      bool first_col = true;
      for (const auto& cell : row) {
        if (first_col) {
          first_col = false;
        } else {
          ss << ",";
        }
        ss << cell;
      }
      ss << '\n';
    }
  }

  return ss.str();
}

std::vector<std::string> parse_header_row(std::istringstream& ss) {
  std::vector<std::string> out;
  std::string buf;
  while (ss.peek() != EOF) {
    buf.clear();
    std::getline(ss, buf, ',');
    out.emplace_back(std::move(buf));
  }
  return out;
}
std::vector<double> parse_data_row(std::istringstream& ss) {
  std::vector<double> out;
  std::string buf;
  while (ss.peek() != EOF) {
    buf.clear();
    std::getline(ss, buf, ',');
    out.emplace_back(std::atof(buf.c_str()));
  }
  return out;
}
Table Table::from_csv(std::string csv) {
  std::istringstream ss;
  ss.str(csv);
  std::string line;

  // Capture the header.
  std::vector<std::string> headers;
  if (ss.peek() != EOF) {
    line.clear();
    std::getline(ss, line, '\n');

    std::istringstream sss;
    sss.str(line);
    headers = parse_header_row(sss);
  }

  std::vector<std::vector<double>> data_rows;
  while (ss.peek() != EOF) {
    line.clear();
    std::getline(ss, line, '\n');

    std::istringstream sss;
    sss.str(line);
    data_rows.emplace_back(parse_data_row(sss));
  }

  return Table(std::move(headers), std::move(data_rows));
}

} // namespace table
} // namespace gpuinfo
