/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//---------------------------------------------------------------------------
/// @file
///   This file includes APIs for dynamic loading on supported platforms
//---------------------------------------------------------------------------

#pragma once

namespace pal {
namespace dynamic_loading {

enum {
  DL_NOW = 0x0001,
  DL_LOCAL = 0x0002,
  DL_GLOBAL = 0x0004,
  DL_NOLOAD = 0x0008
};

//---------------------------------------------------------------------------
/// @brief
///   Loads the dynamic shared object
/// @param filename
///   If contains path separators, treat it as relative or absolute pathname
///   or search it for the rule of dynamic linker
/// @param flags
///   - DL_NOW: resolve undefined symbols before return. MUST be specified.
///   - DL_LOCAL: symbols defined in this shared object are not made available
///     to resolve references in subsequently loaded shared objects
///   - DL_GLOBAL: resolve symbol globally
///   - DL_NOLOAD: only test if the library is loaded, return handle if so
/// @return
///   On success, a non-NULL handle for the loaded library.
///   On error, NULL
//---------------------------------------------------------------------------
void* DlOpen(const char* filename, int flags);

//---------------------------------------------------------------------------
/// @brief
///   Obtain address of a symbol in a shared object or executable
/// @param handle
///   A handle of a dynamic loaded shared object returned by DlOpen
/// @param symbol
///   A null-terminated symbol name
/// @return
///   On success, return the address associated with symbol
///   On error, NULL
//---------------------------------------------------------------------------
void* DlSym(void* handle, const char* symbol);

//---------------------------------------------------------------------------
/// @brief
///   Decrements the reference count on the dynamically loaded shared object
///   referred to by handle.
/// @param handle
///   A handle of a dynamic loaded shared object returned by DlOpen
/// @return
///   On success, 0; on error, a nonzero value
//---------------------------------------------------------------------------
int DlClose(void* handle);

//---------------------------------------------------------------------------
/// @brief
///   Obtain error diagnostic for functions in the dl-family APIs.
/// @return
///   Returns a human-readable, null-terminated string describing the most
///   recent error that occurred from a call to one of the functions in the
///   dl-family APIs.
//---------------------------------------------------------------------------
const char* DlError();

} // namespace dynamic_loading
} // namespace pal
