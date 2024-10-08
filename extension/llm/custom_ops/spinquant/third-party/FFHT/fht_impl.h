#ifndef _FHT_IMPL_H__
#define _FHT_IMPL_H__

#include "fast_copy.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __aarch64__
#include "fht_neon.c"
#define VECTOR_WIDTH (16u)
#else
#ifdef __AVX__
#include "fht_avx.c"
#define VECTOR_WIDTH (32u)
#else
#include "fht_sse.c"
#define VECTOR_WIDTH (16u)
#endif
#endif

int fht_float_oop(float* in, float* out, int log_n) {
  fast_copy(out, in, sizeof(float) << log_n);
  return fht_float(out, log_n);
}

#ifndef __aarch64__
int fht_double_oop(double* in, double* out, int log_n) {
  fast_copy(out, in, sizeof(double) << log_n);
  return fht_double(out, log_n);
}
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ifndef _FHT_IMPL_H__
