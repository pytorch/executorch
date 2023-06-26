#include <executorch/backends/backend.h>
#include <executorch/backends/test/demos/rpc/ExecutorBackend.h>
#include <executorch/core/Error.h>

namespace torch {
namespace executor {
namespace {
static Error register_success = registerExecutorBackend();
} // namespace
} // namespace executor
} // namespace torch
