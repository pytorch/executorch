#include <sys/times.h>
#include <xtensa/sim.h>

#include <executorch/profiler/hooks.h>

namespace torch {
namespace executor {

static bool init = false;

uint64_t get_curr_time(void) {
  if (!init) {
    xt_iss_client_command("all", "enable");
    init = true;
  }

  struct tms curr_time;

  times(&curr_time);
  return curr_time.tms_utime;
}

} // namespace executor
} // namespace torch
