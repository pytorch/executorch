/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <windows.h>

#include <cstring>

#include <pal/DynamicLoading.h>

static thread_local const char* sg_lastErrMsg = "";

void* pal::dynamic_loading::DlOpen(const char* filename, int flags) {
  if (!filename || std::strlen(filename) == 0) {
    sg_lastErrMsg = "DlOpen: filename argument is null or empty";
    return nullptr;
  }

  if (flags & DL_NOLOAD) {
    HMODULE mod = GetModuleHandleA(filename);
    if (!mod) {
      sg_lastErrMsg = "DlOpen: library is not already loaded in the process";
    }
    return static_cast<void*>(mod);
  }

  if (!(flags & DL_NOW)) {
    sg_lastErrMsg =
        "DlOpen: flags must include DL_NOW for immediate resolution";
    return nullptr;
  }

  HMODULE mod =
      LoadLibraryExA(filename, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!mod) {
    sg_lastErrMsg =
        "DlOpen: LoadLibraryExA failed to load the specified library";
    return nullptr;
  }

  return static_cast<void*>(mod);
}

void* pal::dynamic_loading::DlSym(void* handle, const char* symbol) {
  FARPROC sym_addr = nullptr;

  if ((!handle) || (!symbol)) {
    sg_lastErrMsg = "DlSym: handle or symbol argument is null";
    return nullptr;
  }

  HMODULE mod = static_cast<HMODULE>(handle);
  sym_addr = GetProcAddress(mod, symbol);
  if (!sym_addr) {
    sg_lastErrMsg =
        "DlSym: GetProcAddress failed to resolve the specified symbol";
    return nullptr;
  }

  return reinterpret_cast<void*>(sym_addr);
}

int pal::dynamic_loading::DlClose(void* handle) {
  if (!handle) {
    return 0;
  }

  HMODULE mod = static_cast<HMODULE>(handle);

  if (FreeLibrary(mod) == 0) {
    sg_lastErrMsg = "DlClose: FreeLibrary failed to release the module handle";
    return -1;
  }

  return 0;
}

const char* pal::dynamic_loading::DlError() {
  const char* retStr = sg_lastErrMsg;
  sg_lastErrMsg = "";
  return retStr;
}
