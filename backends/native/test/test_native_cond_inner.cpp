/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Cond runtime BYPASS test.
 *
 * Today the partitioner can't claim HOPs end-to-end (see TODO in
 * native_partitioner.py). This test bypasses the partitioner: it feeds
 * a raw flatbuffer Program containing torch.cond directly to
 * NativeBackend::init() as if it were the delegate's `processed`
 * payload. That exercises the runtime side of cond:
 *
 *   - Graph adapter parses cond instructions (JumpFalseCall + branch
 *     ops + MoveCall).
 *   - GreedyRouter builds a plan with JumpFalseStep + ComputeSteps for
 *     each branch.
 *   - materialize_buffers + upload_constants succeed.
 *
 * Full execute is not driven here (would require building EValue arrays
 * + TensorImpls outside the Module API). Init succeeding is enough to
 * prove the cond runtime path doesn't regress when the AOT side
 * eventually grows partitioner support.
 *
 * Inner program path: NATIVE_COND_INNER_PATH (default
 * /tmp/native_cond_inner.fbb), produced by export_cond_inner.py.
 */

#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::BackendInterface;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::get_backend_class;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::Span;

int main() {
  ::executorch::runtime::runtime_init();

  const char* env_path = std::getenv("NATIVE_COND_INNER_PATH");
  std::string path = env_path ? std::string(env_path)
                              : std::string("/tmp/native_cond_inner.fbb");

  printf("=== test_native_cond_inner ===\n");
  printf("  Inner program: %s\n", path.c_str());

  // Read raw flatbuffer Program bytes.
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
    return 1;
  }
  std::streamsize size = f.tellg();
  if (size <= 0) {
    fprintf(stderr, "ERROR: empty file %s\n", path.c_str());
    return 1;
  }
  f.seekg(0);
  void* data = std::malloc(static_cast<size_t>(size));
  if (!data) {
    fprintf(stderr, "ERROR: oom\n");
    return 1;
  }
  f.read(static_cast<char*>(data), size);
  printf("  Loaded %lld bytes\n", static_cast<long long>(size));

  // Wrap in FreeableBuffer (transfers ownership; will free on destroy).
  FreeableBuffer processed(data, static_cast<size_t>(size), nullptr);

  // Get NativeBackend.
  BackendInterface* backend = get_backend_class("NativeBackend");
  if (!backend) {
    fprintf(stderr, "ERROR: NativeBackend not registered\n");
    std::free(data);
    return 2;
  }
  printf("  NativeBackend resolved OK\n");

  // BackendInitContext needs a MemoryAllocator. Use a 1 MiB scratch.
  std::vector<uint8_t> scratch(1 * 1024 * 1024);
  MemoryAllocator allocator(
      static_cast<uint32_t>(scratch.size()), scratch.data());
  BackendInitContext ctx(&allocator);

  // Empty CompileSpec list (init takes ArrayRef, not Span).
  ::executorch::runtime::ArrayRef<CompileSpec> compile_specs;

  // Init.
  auto handle_result = backend->init(ctx, &processed, compile_specs);
  if (!handle_result.ok()) {
    fprintf(
        stderr,
        "ERROR: NativeBackend::init() failed: %d\n",
        static_cast<int>(handle_result.error()));
    std::free(data);
    return 3;
  }
  DelegateHandle* handle = handle_result.get();
  printf("  init() OK — Graph parsed, route() succeeded, plan built\n");
  printf("  (Look above for the [router] partition log showing\n");
  printf("   JumpFalseStep + per-branch ComputeSteps.)\n");

  // Destroy.
  backend->destroy(handle);
  printf("  destroy() OK\n");

  printf("=== PASS ===\n");
  return 0;
}
