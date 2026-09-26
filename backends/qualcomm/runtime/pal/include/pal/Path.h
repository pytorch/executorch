/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//---------------------------------------------------------------------------
/// @file
///   This file includes APIs for path related operations on supported platforms
//---------------------------------------------------------------------------

#pragma once

#include <string>

namespace pal {
namespace path {

//---------------------------------------------------------------------------
/// @brief
///   Returns the platform-specific shared library filename for the given
///   base name. For example, "QnnHtp" becomes "libQnnHtp.so" on Linux
///   or "QnnHtp.dll" on Windows.
/// @param baseName
///   The base name of the library without prefix or extension
/// @return
///   The platform-specific library filename
//---------------------------------------------------------------------------
std::string GetLibraryName(const std::string& baseName);

} // namespace path
} // namespace pal
