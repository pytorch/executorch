#include <executorch/backends/test/demos/rpc/ExecutorBackend.h>
#include <executorch/runtime/backend/backend_registry.h>
#include <executorch/runtime/core/error.h>

namespace torch {
namespace executor {
namespace {
static Error register_success = registerExecutorBackend();
} // namespace
} // namespace executor
} // namespace torch
