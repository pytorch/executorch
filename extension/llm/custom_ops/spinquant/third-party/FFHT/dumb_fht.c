#include "dumb_fht.h"

void dumb_fht(float* buf, int log_n) {
  int n = 1 << log_n;
  for (int i = 0; i < log_n; ++i) {
    int s1 = 1 << i;
    int s2 = s1 << 1;
    for (int j = 0; j < n; j += s2) {
      for (int k = 0; k < s1; ++k) {
        float u = buf[j + k];
        float v = buf[j + k + s1];
        buf[j + k] = u + v;
        buf[j + k + s1] = u - v;
      }
    }
  }
}
