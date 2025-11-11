#include "to_run.h"

#include <benchmark/benchmark.h>

#include <chrono>
#include <stdexcept>

#include <cstdlib>

static void benchmark_fht(benchmark::State &state) {
  double *buf;
  if (posix_memalign((void**)&buf, 32, sizeof(double) * (1 << log_n))) {
    throw std::runtime_error("posix_memalign failed");
  }
  while (state.KeepRunning()) {
    auto start = std::chrono::high_resolution_clock::now();
    run(buf);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
  free(buf);
}

BENCHMARK(benchmark_fht);

BENCHMARK_MAIN();
