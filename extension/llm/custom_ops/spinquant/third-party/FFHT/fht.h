#ifndef _FHT_H_
#define _FHT_H_
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int fht_float(float* buf, int log_n);
#ifndef __aarch64__
int fht_double(double* buf, int log_n);
#endif
int fht_float_oop(float* in, float* out, int log_n);
#ifndef __aarch64__
int fht_double_oop(double* in, double* out, int log_n);
#endif

#ifdef __cplusplus

} // extern "C"

static inline int fht(float* buf, int log_n) {
  return fht_float(buf, log_n);
}

#ifndef __aarch64__
static inline int fht(double* buf, int log_n) {
  return fht_double(buf, log_n);
}
#endif

static inline int fht(float* buf, float* out, int log_n) {
  return fht_float_oop(buf, out, log_n);
}

#ifndef __aarch64__
static inline int fht(double* buf, double* out, int log_n) {
  return fht_double_oop(buf, out, log_n);
}
#endif

#endif

#endif
