// This include statement should get the C99 standard header
// complex.h. At one point we messed up our c10 include setup such
// that it instead included runtime/core/portable_type/complex.h. This
// is a regression test for that issue.
#include <complex.h>

#ifndef complex
#warning "complex.h does not define complex"
#endif
