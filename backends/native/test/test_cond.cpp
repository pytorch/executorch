/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Integration test for NativeBackend control-flow execution.
 *
 * Drives the BackendInterface directly (no Module/Method indirection):
 * the .pte produced by ET's standard pipeline IS a complete
 * `executorch_flatbuffer::Program`, which is exactly what
 * NativeBackend's `init()` expects in its `processed` payload.
 * This lets us exercise the full router + executor path on a control-
 * flow program without going through the partitioner (which currently
 * has a pre-existing SpecViolationError when wrapping HOP submodules).
 *
 * Loads /tmp/cond_pred_v2.pte (CondPred: torch.cond(pred, x+x, x*x)
 * where pred is a Bool tensor input) and runs it twice:
 *   1. pred=True,  x=full(2,3, 1.0)  → x+x = full(2,3, 2.0)
 *   2. pred=False, x=full(2,3, 1.0)  → x*x = full(2,3, 1.0)
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using ::executorch::aten::ScalarType;
using ::executorch::aten::SizesType;
using ::executorch::extension::FileDataLoader;
using ::executorch::extension::from_blob;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::Backend;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::BackendInterface;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::get_backend_class;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

namespace {

constexpr const char* kDefaultPath = "/tmp/cond_pred_v2.pte";

bool run_case(
    BackendInterface* backend,
    DelegateHandle* handle,
    bool pred_val,
    float x_val,
    float expected_val,
    const char* label) {
  // Inputs: (pred: Bool[]; x: Float[2,3]). Output: Float[2,3].
  bool pred_data = pred_val;
  std::vector<float> x_data(6, x_val);
  std::vector<float> out_data(6, 0.0f);
  std::vector<SizesType> pred_sizes; // 0-d tensor
  std::vector<SizesType> tensor_sizes = {2, 3};

  auto pred_t = from_blob(&pred_data, pred_sizes, ScalarType::Bool);
  auto x_t = from_blob(x_data.data(), tensor_sizes, ScalarType::Float);
  auto out_t = from_blob(out_data.data(), tensor_sizes, ScalarType::Float);

  EValue pred_ev(pred_t);
  EValue x_ev(x_t);
  EValue out_ev(out_t);
  EValue* args[] = {&pred_ev, &x_ev, &out_ev};

  BackendExecutionContext exec_ctx;
  Error err = backend->execute(exec_ctx, handle, Span<EValue*>(args, 3));
  if (err != Error::Ok) {
    std::fprintf(
        stderr,
        "  [%s] execute returned error 0x%x\n",
        label,
        static_cast<unsigned>(err));
    return false;
  }

  bool ok = true;
  for (size_t i = 0; i < 6; ++i) {
    float diff = out_data[i] - expected_val;
    if (diff < 0) diff = -diff;
    if (diff > 1e-5f) ok = false;
  }
  std::printf(
      "  [%s] pred=%s x=%.2f → out=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
      "(expected %.3f) %s\n",
      label,
      pred_val ? "True" : "False",
      x_val,
      out_data[0],
      out_data[1],
      out_data[2],
      out_data[3],
      out_data[4],
      out_data[5],
      expected_val,
      ok ? "PASS" : "FAIL");
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  ::executorch::runtime::runtime_init();

  const char* path = (argc > 1) ? argv[1] : kDefaultPath;
  std::printf("Loading model: %s\n", path);

  // Load .pte bytes via FileDataLoader, then expose them as a
  // FreeableBuffer (the BackendInterface::init() input).
  auto loader_r = FileDataLoader::from(path);
  if (!loader_r.ok()) {
    std::fprintf(
        stderr,
        "FileDataLoader::from('%s') failed: 0x%x\n",
        path,
        static_cast<unsigned>(loader_r.error()));
    return 1;
  }
  FileDataLoader loader = std::move(loader_r.get());

  auto size_r = loader.size();
  if (!size_r.ok()) return 2;
  size_t bytes = size_r.get();

  auto seg_r = loader.load(
      0, bytes, ::executorch::runtime::DataLoader::SegmentInfo());
  if (!seg_r.ok()) {
    std::fprintf(
        stderr,
        "loader.load failed: 0x%x\n",
        static_cast<unsigned>(seg_r.error()));
    return 3;
  }
  FreeableBuffer processed = std::move(seg_r.get());

  // Look up the registered backend (registered at static init time by
  // NativeBackend.cpp).
  BackendInterface* backend = get_backend_class("NativeBackend");
  if (!backend) {
    std::fprintf(
        stderr, "NativeBackend not registered — link error?\n");
    return 4;
  }

  // Provide a runtime allocator (BackendInitContext requires one).
  // 1 MiB is enough for a tiny cond model.
  static constexpr size_t kArenaBytes = 1 * 1024 * 1024;
  std::vector<uint8_t> arena(kArenaBytes);
  MemoryAllocator runtime_alloc(kArenaBytes, arena.data());

  // No NamedDataMap for this test (no external constants).
  BackendInitContext init_ctx(&runtime_alloc, /*temp=*/nullptr);

  std::vector<CompileSpec> specs;
  auto handle_r = backend->init(
      init_ctx, &processed, ArrayRef<CompileSpec>(specs.data(), specs.size()));
  if (!handle_r.ok()) {
    std::fprintf(
        stderr,
        "backend->init failed: 0x%x\n",
        static_cast<unsigned>(handle_r.error()));
    return 5;
  }
  DelegateHandle* handle = handle_r.get();

  std::printf("\nControl-flow integration test:\n");
  bool ok = true;
  ok &= run_case(backend, handle, /*pred=*/true,  /*x=*/1.0f, /*expected=*/2.0f, "true_branch  (x+x)");
  ok &= run_case(backend, handle, /*pred=*/false, /*x=*/1.0f, /*expected=*/1.0f, "false_branch (x*x)");

  backend->destroy(handle);

  if (ok) {
    std::printf("\nPASS — control flow correct on both branches\n");
    return 0;
  }
  std::printf("\nFAIL\n");
  return 6;
}
