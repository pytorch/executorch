#include <stdint.h>
#include <time.h>

#include <executorch/core/Assert.h>
#include <executorch/profiler/hooks.h>

namespace torch {
namespace executor {

#define NSEC_PER_USEC 1000UL
#define USEC_IN_SEC 1000000UL
#define NSEC_IN_USEC 1000UL
#define NSEC_IN_SEC (NSEC_IN_USEC * USEC_IN_SEC)

uint64_t get_curr_time(void) {
  struct timespec ts;
  auto ret = clock_gettime(CLOCK_REALTIME, &ts);
  ET_CHECK_MSG(ret == 0, "Failed to get time.");

  return ((ts.tv_sec * NSEC_IN_SEC) + (ts.tv_nsec));
}

} // namespace executor
} // namespace torch
