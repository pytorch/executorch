// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <charconv>
#include <string>

namespace ptn {

// Render a double for a debug dump. Not std::to_string: its fixed six-decimal
// format prints 1e-8 as "0.000000" and 1e300 as 312 digits. to_chars emits the
// shortest form that round-trips. The longest such form is 24 characters
// ("-1.7976931348623157e+308"), so the buffer cannot overflow and the result
// needs no error check.
inline std::string format_double(double value) {
  std::array<char, 32> buf{};
  const std::to_chars_result out =
      std::to_chars(buf.data(), buf.data() + buf.size(), value);
  std::string text(buf.data(), out.ptr);
  // to_chars renders 6.0 as "6", which in a dump reads as an int argument.
  // Put the point back; exponent, "inf" and "nan" forms are already
  // unambiguous, and each carries one of these characters.
  if (text.find_first_of(".eni") == std::string::npos) {
    text += ".0";
  }
  return text;
}

} // namespace ptn
