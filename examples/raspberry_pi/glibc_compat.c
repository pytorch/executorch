/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Compiled by setup.sh against the Debian sysroot selected with --os and linked
// into every binary. The ARM GNU toolchain's static libstdc++ was built against
// glibc 2.38, so it references symbols that older Debian releases lack. Each
// block below provides one of them only when the sysroot's glibc is too old.

#include <features.h>
#include <stdlib.h>

#if !__GLIBC_PREREQ(2, 38)
// Pulled in by libstdc++'s eh_alloc.o to parse GLIBCXX_TUNABLES. The C23
// version only differs in accepting a 0b prefix, which that parser never sees.
unsigned long __isoc23_strtoul(const char* s, char** end, int base) {
  return strtoul(s, end, base);
}
#endif
