#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "fht.h"

void dumb_fht(float *buf, int log_n);
void dumb_fht(float *buf, int log_n) {
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

int main(void) {
    srand(4057218);
    for (int log_n = 1; log_n <= 30; ++log_n) {
        printf("%d ", log_n);
        int n = 1 << log_n;
        void *buf = malloc(sizeof(float) * n + 32);
        char *start = buf;
        while ((size_t)start % 32 != 0) start = start + 1;
        float *a = (float*)start;
        float *aux = (float*)malloc(sizeof(double) * n);
        for (int i = 0; i < n; ++i) {
            a[i] = 1.0 - 2.0 * (rand() & 1);
            aux[i] = a[i];
        }
        fht_float(a, log_n);
        dumb_fht(aux, log_n);
        double max_error = 0.0;
        for (int i = 0; i < n; ++i) {
            double error = fabs(a[i] - aux[i]);
            if (error > max_error) {
                max_error = error;
            }
        }
        if (max_error > 1e-5) {
            printf("ERROR: %.10lf\n", max_error);
            return 1;
        }
        for (int num_it = 10;; num_it *= 2) {
            clock_t tt1 = clock();
            for (int it = 0; it < num_it; ++it) {
                fht_float(a, log_n);
            }
            clock_t tt2 = clock();
            double sec = (tt2 - tt1) / (CLOCKS_PER_SEC + 0.0);
            if (sec >= 1.0) {
                printf("%.10e\n", sec / (num_it + 0.0));
                break;
            }
        }
        free(buf);
        free(aux);
    }
    return 0;
}
