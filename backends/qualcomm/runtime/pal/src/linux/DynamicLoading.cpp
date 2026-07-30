/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <dlfcn.h>

#include <pal/DynamicLoading.h>

void* pal::dynamic_loading::DlOpen(const char* filename, int flags) {
  int realFlags = 0;

  if (flags & DL_NOW) {
    realFlags |= RTLD_NOW;
  }

  if (flags & DL_LOCAL) {
    realFlags |= RTLD_LOCAL;
  }

  if (flags & DL_GLOBAL) {
    realFlags |= RTLD_GLOBAL;
  }

  if (flags & DL_NOLOAD) {
#ifndef __hexagon__
    realFlags |= RTLD_NOLOAD;
#else
    return nullptr;
#endif
  }

  return ::dlopen(filename, realFlags);
}

void* pal::dynamic_loading::DlSym(void* handle, const char* symbol) {
  return ::dlsym(handle, symbol);
}

int pal::dynamic_loading::DlClose(void* handle) {
  if (!handle) {
    return 0;
  }
  return ::dlclose(handle);
}

const char* pal::dynamic_loading::DlError() {
  return ::dlerror();
}
