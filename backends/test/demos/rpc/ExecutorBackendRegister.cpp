#include <executorch/backends/test/demos/rpc/ExecutorBackend.h>
#include <executorch/core/Error.h>
#include <executorch/runtime/backend/backend_registry.h>

namespace torch {
namespace executor {
namespace {
static Error register_success = registerExecutorBackend();
} // namespace
} // namespace executor
} // namespace torch
