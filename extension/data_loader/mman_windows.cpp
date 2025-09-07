/*
 * Copyright (c) Google Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license.
 */

/*
 * Adapted from: https://code.google.com/archive/p/mman-win32/
 *
 * mman-win32
 * mman library for Windows
 *
 * A light implementation of the mmap functions for MinGW.
 *
 * The mmap-win32 library implements a wrapper for mmap functions around the
 * memory mapping Windows API.
 */

#include <executorch/extension/data_loader/mman_windows.h>

#include <errno.h>
#include <io.h>
#include <windows.h>

#ifndef STATUS_SECTION_TOO_BIG
#define STATUS_SECTION_TOO_BIG 0xC0000040L
#endif

#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE 0x0020
#endif /* FILE_MAP_EXECUTE */

#define RETURN_IF_FAILED(hr) \
  do {                       \
    if (FAILED((hr))) {      \
      return hr;             \
    }                        \
  } while (false)

namespace {

HRESULT try_grow_process_memory_working_set(DWORD dwSizeRequired) {
  // Get current working set
  size_t minWorkingSetInitial;
  size_t maxWorkingSet;
  if (!GetProcessWorkingSetSize(
          GetCurrentProcess(), &minWorkingSetInitial, &maxWorkingSet)) {
    return GetLastError();
  }

  // Calculate new sizes
  size_t minWorkingSet = minWorkingSetInitial + dwSizeRequired;
  if (minWorkingSet < minWorkingSetInitial) {
    return HRESULT_FROM_WIN32(ERROR_ARITHMETIC_OVERFLOW);
  }

  if (maxWorkingSet < minWorkingSet) {
    maxWorkingSet = minWorkingSet;
  }

  // Grow working set
  if (!SetProcessWorkingSetSize(
          GetCurrentProcess(), minWorkingSet, maxWorkingSet)) {
    return GetLastError();
  }
  return S_OK;
}

HRESULT virtual_lock(void* pMem, DWORD dwSize) {
  if (!VirtualLock(pMem, dwSize)) {
    return GetLastError();
  }
  return S_OK;
}

HRESULT virtual_lock_allowing_working_set_growth(void* pMem, DWORD dwSize) {
  HRESULT hr = virtual_lock(pMem, dwSize);

  if (hr == HRESULT_FROM_WIN32(STATUS_SECTION_TOO_BIG)) {
    // Attempt to grow the process working set and try again
    RETURN_IF_FAILED(try_grow_process_memory_working_set(dwSize));
    RETURN_IF_FAILED(virtual_lock(pMem, dwSize));
  }

  return hr;
}

static int __map_mman_error(const DWORD err, const int deferr) {
  if (err == 0) {
    return 0;
  }
  // TODO: implement
  return err;
}

static DWORD __map_mmap_prot_page(const int prot) {
  DWORD protect = 0;

  if (prot == PROT_NONE) {
    return protect;
  }
  if ((prot & PROT_EXEC) != 0) {
    protect =
        ((prot & PROT_WRITE) != 0) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
  } else {
    protect = ((prot & PROT_WRITE) != 0) ? PAGE_READWRITE : PAGE_READONLY;
  }
  return protect;
}

static DWORD __map_mmap_prot_file(const int prot) {
  DWORD desiredAccess = 0;

  if (prot == PROT_NONE) {
    return desiredAccess;
  }
  if ((prot & PROT_READ) != 0) {
    desiredAccess |= FILE_MAP_READ;
  }
  if ((prot & PROT_WRITE) != 0) {
    desiredAccess |= FILE_MAP_WRITE;
  }
  if ((prot & PROT_EXEC) != 0) {
    desiredAccess |= FILE_MAP_EXECUTE;
  }
  return desiredAccess;
}

} // namespace

void* mmap(void* addr, size_t len, int prot, int flags, int fildes, off_t off) {
  HANDLE fm, h;

  void* map = MAP_FAILED;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4293)
#endif

  const DWORD dwFileOffsetLow = (sizeof(off_t) <= sizeof(DWORD))
      ? (DWORD)off
      : (DWORD)(off & 0xFFFFFFFFL);
  const DWORD dwFileOffsetHigh = (sizeof(off_t) <= sizeof(DWORD))
      ? (DWORD)0
      : (DWORD)((off >> 32) & 0xFFFFFFFFL);
  const DWORD protect = __map_mmap_prot_page(prot);
  const DWORD desiredAccess = __map_mmap_prot_file(prot);

  const off_t maxSize = off + (off_t)len;

  const DWORD dwMaxSizeLow = (sizeof(off_t) <= sizeof(DWORD))
      ? (DWORD)maxSize
      : (DWORD)(maxSize & 0xFFFFFFFFL);
  const DWORD dwMaxSizeHigh = (sizeof(off_t) <= sizeof(DWORD))
      ? (DWORD)0
      : (DWORD)((maxSize >> 32) & 0xFFFFFFFFL);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

  errno = 0;

  if (len == 0
      /* Unsupported flag combinations */
      || (flags & MAP_FIXED) != 0
      /* Usupported protection combinations */
      || prot == PROT_EXEC) {
    errno = EINVAL;
    return MAP_FAILED;
  }

  h = ((flags & MAP_ANONYMOUS) == 0) ? (HANDLE)_get_osfhandle(fildes)
                                     : INVALID_HANDLE_VALUE;

  if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return MAP_FAILED;
  }

  fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow, NULL);

  if (fm == NULL) {
    errno = __map_mman_error(GetLastError(), EPERM);
    return MAP_FAILED;
  }

  map =
      MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len);

  CloseHandle(fm);

  if (map == NULL) {
    errno = __map_mman_error(GetLastError(), EPERM);
    return MAP_FAILED;
  }

  return map;
}

int munmap(void* addr, size_t len) {
  if (UnmapViewOfFile(addr))
    return 0;

  errno = __map_mman_error(GetLastError(), EPERM);

  return -1;
}

int mprotect(void* addr, size_t len, int prot) {
  DWORD newProtect = __map_mmap_prot_page(prot);
  DWORD oldProtect = 0;

  if (VirtualProtect(addr, len, newProtect, &oldProtect))
    return 0;

  errno = __map_mman_error(GetLastError(), EPERM);

  return -1;
}

int msync(void* addr, size_t len, int flags) {
  if (FlushViewOfFile(addr, len))
    return 0;

  errno = __map_mman_error(GetLastError(), EPERM);

  return -1;
}

int mlock(const void* addr, size_t len) {
  HRESULT hr = virtual_lock_allowing_working_set_growth((LPVOID)addr, len);
  if (SUCCEEDED(hr)) {
    return 0;
  }

  errno = __map_mman_error(hr, EPERM);

  return -1;
}

int munlock(const void* addr, size_t len) {
  if (VirtualUnlock((LPVOID)addr, len))
    return 0;

  errno = __map_mman_error(GetLastError(), EPERM);

  return -1;
}
