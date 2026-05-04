/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Integration test for in-place op dispatch through NativeBackend.
 *
 * Each test case loads a .pte (produced by test_export_inplace.py) and
 * runs it through NativeBackend's BackendInterface directly (same
 * pattern as test_cond.cpp). The .pte was exported with reinplace_pass
 * applied at top level, so the in-place op (e.g. aten::add_.Tensor)
 * appears directly in the program's instruction chain.
 *
 * For each case:
 *   1. Allocate input + output Float[2,3] tensors (output is the same
 *      tensor as the in-place op's `self` input — caller's buffer).
 *   2. Drive NativeBackend::execute with the .pte's bytes as processed.
 *   3. Verify each output element matches an eager-computed expected.
 *
 * Verifies the full_name dispatch path: routing the IR's
 * "aten::X_.<overload>" full identity to the registered handler and
 * invoking the corresponding "aten::X.out" kernel.
 */

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

using ::executorch::aten::ScalarType;
using ::executorch::aten::SizesType;
using ::executorch::extension::FileDataLoader;
using ::executorch::extension::from_blob;
using ::executorch::runtime::ArrayRef;
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
using ::executorch::runtime::Span;

namespace {

constexpr size_t kArenaBytes = 1 * 1024 * 1024;

// Compare two float buffers element-wise within `eps`.
bool nearly_equal(const float* a, const float* b, size_t n, float eps = 1e-5f) {
  for (size_t i = 0; i < n; ++i) {
    float d = a[i] - b[i];
    if (d < 0) d = -d;
    if (d > eps) return false;
  }
  return true;
}

void print_buf(const char* label, const float* p, size_t n) {
  std::printf("    %s = [", label);
  for (size_t i = 0; i < n; ++i) {
    std::printf("%s%.4f", i == 0 ? "" : ", ", p[i]);
  }
  std::printf("]\n");
}

struct TestCase {
  const char* name;
  std::string pte_path;
  std::function<void(std::vector<float>& self_data,
                     std::vector<std::vector<float>>& other_inputs,
                     std::vector<float>& expected)> setup;
  // Number of additional Tensor inputs beyond `self` (e.g. add_.Tensor
  // has 1 extra: `other`; clamp_ has 0 — its min/max are scalars baked
  // into the IR).
  int n_extra_inputs;
  // Number of program-level outputs (most in-place ops have 2 because
  // emit treats the in-place result both as the function's return AND
  // as the value-id-deduped formal_out — both pointing at the same
  // value; the C++ harness must supply EValue slots for each).
  int n_outputs = 2;
};

bool run_case(const TestCase& tc) {
  std::printf("\n--- %s ---\n", tc.name);

  std::vector<float> self_data(6);
  std::vector<std::vector<float>> other_inputs;
  std::vector<float> expected(6);
  tc.setup(self_data, other_inputs, expected);

  std::vector<SizesType> sizes = {2, 3};
  auto self_t = from_blob(self_data.data(), sizes, ScalarType::Float);
  // Keep additional TensorPtrs alive for the duration of execute —
  // the EValues we pass reference the underlying Tensors via these
  // smart pointers, so they must outlive the execute call.
  std::vector<::executorch::extension::TensorPtr> other_tensors;
  std::vector<EValue> evalues;
  evalues.reserve(1 + tc.n_extra_inputs + tc.n_outputs);
  evalues.emplace_back(self_t);
  for (auto& other : other_inputs) {
    other_tensors.push_back(from_blob(other.data(), sizes, ScalarType::Float));
    evalues.emplace_back(other_tensors.back());
  }
  for (int i = 0; i < tc.n_outputs; ++i) {
    evalues.emplace_back(self_t);
  }

  std::vector<EValue*> arg_ptrs;
  arg_ptrs.reserve(evalues.size());
  for (auto& e : evalues) arg_ptrs.push_back(&e);

  // Load the .pte bytes directly as the backend's processed payload.
  auto loader_r = FileDataLoader::from(tc.pte_path.c_str());
  if (!loader_r.ok()) {
    std::fprintf(stderr, "  FAIL: FileDataLoader::from('%s') 0x%x\n",
                 tc.pte_path.c_str(),
                 static_cast<unsigned>(loader_r.error()));
    return false;
  }
  FileDataLoader loader = std::move(loader_r.get());
  auto size_r = loader.size();
  if (!size_r.ok()) return false;
  auto seg_r = loader.load(0, size_r.get(),
                           ::executorch::runtime::DataLoader::SegmentInfo());
  if (!seg_r.ok()) return false;
  FreeableBuffer processed = std::move(seg_r.get());

  BackendInterface* backend = get_backend_class("NativeBackend");
  if (!backend) {
    std::fprintf(stderr, "  FAIL: NativeBackend not registered\n");
    return false;
  }

  std::vector<uint8_t> arena(kArenaBytes);
  MemoryAllocator runtime_alloc(kArenaBytes, arena.data());
  BackendInitContext init_ctx(&runtime_alloc, /*temp=*/nullptr);

  std::vector<CompileSpec> specs;
  auto handle_r = backend->init(
      init_ctx, &processed,
      ArrayRef<CompileSpec>(specs.data(), specs.size()));
  if (!handle_r.ok()) {
    std::fprintf(stderr, "  FAIL: init returned 0x%x\n",
                 static_cast<unsigned>(handle_r.error()));
    return false;
  }
  DelegateHandle* handle = handle_r.get();

  BackendExecutionContext exec_ctx;
  Error err = backend->execute(
      exec_ctx, handle,
      Span<EValue*>(arg_ptrs.data(), arg_ptrs.size()));
  backend->destroy(handle);

  if (err != Error::Ok) {
    std::fprintf(stderr, "  FAIL: execute returned 0x%x\n",
                 static_cast<unsigned>(err));
    return false;
  }

  // After execute, `self_data` holds the in-place result.
  print_buf("got     ", self_data.data(), 6);
  print_buf("expected", expected.data(), 6);
  if (!nearly_equal(self_data.data(), expected.data(), 6)) {
    std::printf("  FAIL: output mismatch\n");
    return false;
  }
  std::printf("  PASS\n");
  return true;
}

}  // namespace

namespace {

/// Generic input descriptor for heterogeneous in-place tests.
struct GenericInput {
  void* data;            // pointer to caller's storage
  std::vector<SizesType> sizes;
  ScalarType dtype;
};

/// Generic runner for in-place ops with heterogeneous input shapes/dtypes.
/// Caller supplies a list of GenericInputs (program inputs in order) and
/// the expected post-execute contents of `self_data` (the output buffer
/// that aliases the in-place op's mutated arg).
///
/// `inputs[0]` is `self` — same tensor used for ALL output slots
/// (the in-place op's output value_id == its self value_id, repeated
/// per program-output entry).
bool run_heterogeneous_case(
    const char* test_name,
    const std::string& pte_path,
    std::vector<GenericInput> inputs,
    int n_outputs,
    const float* expected,
    size_t expected_n) {
  std::printf("\n--- %s ---\n", test_name);

  // Build one TensorPtr per input, kept alive for the duration of execute.
  std::vector<::executorch::extension::TensorPtr> tensors;
  tensors.reserve(inputs.size());
  std::vector<EValue> evalues;
  evalues.reserve(inputs.size() + n_outputs);
  for (auto& inp : inputs) {
    tensors.push_back(from_blob(inp.data, inp.sizes, inp.dtype));
    evalues.emplace_back(tensors.back());
  }
  // Output slots — re-use inputs[0]'s tensor (the self alias).
  for (int i = 0; i < n_outputs; ++i) {
    evalues.emplace_back(tensors[0]);
  }

  std::vector<EValue*> arg_ptrs;
  for (auto& e : evalues) arg_ptrs.push_back(&e);

  auto loader_r = FileDataLoader::from(pte_path.c_str());
  if (!loader_r.ok()) {
    std::fprintf(stderr, "  FAIL: FileDataLoader 0x%x\n",
                 static_cast<unsigned>(loader_r.error()));
    return false;
  }
  FileDataLoader loader = std::move(loader_r.get());
  auto size_r = loader.size();
  if (!size_r.ok()) return false;
  auto seg_r = loader.load(0, size_r.get(),
                           ::executorch::runtime::DataLoader::SegmentInfo());
  if (!seg_r.ok()) return false;
  FreeableBuffer processed = std::move(seg_r.get());

  BackendInterface* backend = get_backend_class("NativeBackend");
  if (!backend) return false;

  std::vector<uint8_t> arena(kArenaBytes);
  MemoryAllocator runtime_alloc(kArenaBytes, arena.data());
  BackendInitContext init_ctx(&runtime_alloc, /*temp=*/nullptr);

  std::vector<CompileSpec> specs;
  auto handle_r = backend->init(
      init_ctx, &processed,
      ArrayRef<CompileSpec>(specs.data(), specs.size()));
  if (!handle_r.ok()) {
    std::fprintf(stderr, "  FAIL: init returned 0x%x\n",
                 static_cast<unsigned>(handle_r.error()));
    return false;
  }
  DelegateHandle* handle = handle_r.get();

  BackendExecutionContext exec_ctx;
  Error err = backend->execute(
      exec_ctx, handle,
      Span<EValue*>(arg_ptrs.data(), arg_ptrs.size()));
  backend->destroy(handle);

  if (err != Error::Ok) {
    std::fprintf(stderr, "  FAIL: execute returned 0x%x\n",
                 static_cast<unsigned>(err));
    return false;
  }

  // Read back self_data (inputs[0].data, treated as float).
  const float* got = static_cast<const float*>(inputs[0].data);
  print_buf("got     ", got, expected_n);
  print_buf("expected", expected, expected_n);
  if (!nearly_equal(got, expected, expected_n)) {
    std::printf("  FAIL: output mismatch\n");
    return false;
  }
  std::printf("  PASS\n");
  return true;
}

bool run_index_put_case() {
  std::vector<float> self_data(6, 0.0f);
  std::vector<int64_t> idx_data = {0};
  std::vector<float> values_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> expected = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f};
  return run_heterogeneous_case(
      "index_put_  (self[idx] = values; OptionalTensorList indices)",
      "/tmp/inplace_index_put_.pte",
      {
          {self_data.data(), {2, 3}, ScalarType::Float},
          {idx_data.data(), {1}, ScalarType::Long},
          {values_data.data(), {1, 3}, ScalarType::Float},
      },
      /*n_outputs=*/2,
      expected.data(), expected.size());
}

bool run_index_add_case() {
  // index_add_(self, dim=0, idx, src) — IR decomposed to index_put_.
  std::vector<float> self_data(6, 0.0f);
  std::vector<int64_t> idx_data = {0, 1};
  std::vector<float> src_data = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
  std::vector<float> expected = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
  return run_heterogeneous_case(
      "index_add_  (decomposes to index_put_ in IR)",
      "/tmp/inplace_index_add_.pte",
      {
          {self_data.data(), {2, 3}, ScalarType::Float},
          {idx_data.data(), {2}, ScalarType::Long},
          {src_data.data(), {2, 3}, ScalarType::Float},
      },
      /*n_outputs=*/2,
      expected.data(), expected.size());
}

bool run_scatter_add_case() {
  // scatter_add_(self, dim=0, idx, src):
  // self[idx[i,j], j] += src[i,j].
  std::vector<float> self_data(6, 0.0f);
  std::vector<int64_t> idx_data = {0, 1, 0, 1, 0, 1};  // [[0,1,0],[1,0,1]]
  std::vector<float> src_data = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};
  std::vector<float> expected = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  return run_heterogeneous_case(
      "scatter_add_  (self[idx[i,j], j] += src[i,j])",
      "/tmp/inplace_scatter_add_.pte",
      {
          {self_data.data(), {2, 3}, ScalarType::Float},
          {idx_data.data(), {2, 3}, ScalarType::Long},
          {src_data.data(), {2, 3}, ScalarType::Float},
      },
      /*n_outputs=*/2,
      expected.data(), expected.size());
}

bool run_masked_scatter_case() {
  // masked_scatter_(self, mask, src): write next-src-elem at each True pos.
  std::vector<float> self_data(6, 0.0f);
  std::vector<bool> mask_storage = {true, false, true, false, true, false};
  // std::vector<bool> is bit-packed; copy to a uint8_t buffer for the
  // Tensor's data_ptr (Bool tensors are stored as 1 byte per element).
  std::vector<uint8_t> mask_bytes(6);
  for (size_t i = 0; i < 6; ++i) mask_bytes[i] = mask_storage[i] ? 1 : 0;
  std::vector<float> src_data = {10.0f, 20.0f, 30.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> expected = {10.0f, 0.0f, 20.0f, 0.0f, 30.0f, 0.0f};
  return run_heterogeneous_case(
      "masked_scatter_  (mask Bool[2,3] picks positions, src Float[6])",
      "/tmp/inplace_masked_scatter_.pte",
      {
          {self_data.data(), {2, 3}, ScalarType::Float},
          {mask_bytes.data(), {2, 3}, ScalarType::Bool},
          {src_data.data(), {6}, ScalarType::Float},
      },
      /*n_outputs=*/2,
      expected.data(), expected.size());
}

/// HF-style KV-cache update test.
///
/// Model:
///   self.k_cache = zeros(1, n_heads=2, max_len=8, head_dim=4)   ← buffer
///   forward(k_val: [1,2,2,4], cache_position: [2]) →
///     self.k_cache.index_copy_(2, cache_position, k_val)
///     return self.k_cache + 0
///
/// Inputs:
///   k_val = arange(16) reshaped to [1,2,2,4]
///   cache_position = [3, 4]   (write at seq positions 3 and 4)
///
/// Expected: 64-element output buffer with non-zeros at positions
///   [12..15], [16..19], [44..47], [48..51] holding 0..15.
bool run_kvcache_index_copy_case() {
  std::printf("\n--- kvcache_index_copy  (HF-style index_copy_ on registered buffer) ---\n");

  // Inputs.
  std::vector<float> k_val_data(16);
  for (int i = 0; i < 16; ++i) k_val_data[i] = static_cast<float>(i);
  std::vector<int64_t> cache_pos_data = {3, 4};

  // Output buffer for k_cache (the program output is the post-update buffer).
  // Shape [1, 2, 8, 4] = 64 floats. Pre-fill with sentinel so we can detect
  // unwritten regions. The InitializedMutableBufferPass embedded zeros into
  // the .pte for k_cache's initial state, so the kernel sees zeros, writes
  // the new positions, and the output should be zero-valued except at the
  // 16 positions index_copy_ touched.
  std::vector<float> k_cache_out(64, -999.0f);

  // Build expected: 64 zeros except at 16 positions matching k_val.
  std::vector<float> expected(64, 0.0f);
  // Indexing: [b=0, h, s, d] → flat = h*8*4 + s*4 + d = h*32 + s*4 + d
  // For h=0..1, s ∈ {3, 4} (cache_position[0..1]), d=0..3:
  //   expected[h*32 + 3*4 + d] = k_val[h*2*4 + 0*4 + d] = h*8 + d
  //   expected[h*32 + 4*4 + d] = k_val[h*2*4 + 1*4 + d] = h*8 + 4 + d
  for (int h = 0; h < 2; ++h) {
    for (int d = 0; d < 4; ++d) {
      expected[h * 32 + 3 * 4 + d] = static_cast<float>(h * 8 + d);
      expected[h * 32 + 4 * 4 + d] = static_cast<float>(h * 8 + 4 + d);
    }
  }

  // Build the EValues. inputs = [k_val, cache_position]; output = k_cache.
  std::vector<::executorch::extension::TensorPtr> tensors;
  std::vector<EValue> evalues;
  tensors.push_back(from_blob(k_val_data.data(), {1, 2, 2, 4},
                              ScalarType::Float));
  evalues.emplace_back(tensors.back());
  tensors.push_back(from_blob(cache_pos_data.data(), {2}, ScalarType::Long));
  evalues.emplace_back(tensors.back());
  // Output: k_cache shape [1, 2, 8, 4].
  tensors.push_back(from_blob(k_cache_out.data(), {1, 2, 8, 4},
                              ScalarType::Float));
  evalues.emplace_back(tensors.back());

  std::vector<EValue*> arg_ptrs;
  for (auto& e : evalues) arg_ptrs.push_back(&e);

  auto loader_r = FileDataLoader::from("/tmp/inplace_kvcache_index_copy.pte");
  if (!loader_r.ok()) {
    std::fprintf(stderr, "  FAIL: FileDataLoader 0x%x\n",
                 static_cast<unsigned>(loader_r.error()));
    return false;
  }
  FileDataLoader loader = std::move(loader_r.get());
  auto size_r = loader.size();
  if (!size_r.ok()) return false;
  auto seg_r = loader.load(0, size_r.get(),
                           ::executorch::runtime::DataLoader::SegmentInfo());
  if (!seg_r.ok()) return false;
  FreeableBuffer processed = std::move(seg_r.get());

  BackendInterface* backend = get_backend_class("NativeBackend");
  if (!backend) return false;

  std::vector<uint8_t> arena(kArenaBytes);
  MemoryAllocator runtime_alloc(kArenaBytes, arena.data());
  BackendInitContext init_ctx(&runtime_alloc, /*temp=*/nullptr);

  std::vector<CompileSpec> specs;
  auto handle_r = backend->init(
      init_ctx, &processed,
      ArrayRef<CompileSpec>(specs.data(), specs.size()));
  if (!handle_r.ok()) {
    std::fprintf(stderr, "  FAIL: init returned 0x%x\n",
                 static_cast<unsigned>(handle_r.error()));
    return false;
  }
  DelegateHandle* handle = handle_r.get();

  BackendExecutionContext exec_ctx;
  Error err = backend->execute(
      exec_ctx, handle,
      Span<EValue*>(arg_ptrs.data(), arg_ptrs.size()));
  backend->destroy(handle);

  if (err != Error::Ok) {
    std::fprintf(stderr, "  FAIL: execute returned 0x%x\n",
                 static_cast<unsigned>(err));
    return false;
  }

  // Verify by element-wise comparison + sum check.
  float got_sum = 0.0f, exp_sum = 0.0f;
  for (size_t i = 0; i < 64; ++i) {
    got_sum += k_cache_out[i];
    exp_sum += expected[i];
  }
  std::printf("  got sum=%.1f  expected sum=%.1f\n", got_sum, exp_sum);
  // Print the head-0 slice for visual inspection.
  std::printf("  got[head 0, all 8 positions]:\n");
  for (int s = 0; s < 8; ++s) {
    std::printf("    s=%d: [%.1f, %.1f, %.1f, %.1f]\n", s,
                k_cache_out[s * 4 + 0], k_cache_out[s * 4 + 1],
                k_cache_out[s * 4 + 2], k_cache_out[s * 4 + 3]);
  }
  if (!nearly_equal(k_cache_out.data(), expected.data(), 64)) {
    std::printf("  FAIL: output mismatch\n");
    return false;
  }
  std::printf("  PASS\n");
  return true;
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
  ::executorch::runtime::runtime_init();

  std::vector<TestCase> cases;

  // -------------------------------------------------------------
  // Unary chain — exercises ~13 unary in-place ops in one program.
  // The chain is constructed to produce a clean final value:
  //   x=[4,9,16,25,36,49] → ... → [0,0,0,0,0,0]
  // See test/test_export_inplace.py UnaryChain for the full sequence.
  // Covers: sqrt_, neg_, abs_, pow (square_), relu_, ceil_, floor_,
  //         round_, trunc_, hardtanh_, exp_, log_, copy_.
  // -------------------------------------------------------------
  cases.push_back({
      "unary_chain  (13 unary in-place ops chained)",
      "/tmp/inplace_unary_chain.pte",
      [](auto& self, auto& others, auto& expected) {
        self = {4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f};
        expected = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        (void)others;
      },
      0,
  });

  // -------------------------------------------------------------
  // Binary in-place ops — separate .pte per op.
  // -------------------------------------------------------------

  // add_.Tensor: self += other
  cases.push_back({
      "add_.Tensor  (self += other)",
      "/tmp/inplace_add_Tensor.pte",
      [](auto& self, auto& others, auto& expected) {
        self = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        others.push_back({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
        expected = {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f};
      },
      1,
  });

  // sub_.Tensor: self -= other
  cases.push_back({
      "sub_.Tensor  (self -= other)",
      "/tmp/inplace_sub_Tensor.pte",
      [](auto& self, auto& others, auto& expected) {
        self = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
        others.push_back({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        expected = {9.0f, 18.0f, 27.0f, 36.0f, 45.0f, 54.0f};
      },
      1,
  });

  // mul_.Tensor: self *= other
  cases.push_back({
      "mul_.Tensor  (self *= other)",
      "/tmp/inplace_mul_Tensor.pte",
      [](auto& self, auto& others, auto& expected) {
        self = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        others.push_back({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
        expected = {10.0f, 40.0f, 90.0f, 160.0f, 250.0f, 360.0f};
      },
      1,
  });

  // div_.Tensor: self /= other
  cases.push_back({
      "div_.Tensor  (self /= other)",
      "/tmp/inplace_div_Tensor.pte",
      [](auto& self, auto& others, auto& expected) {
        self = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
        others.push_back({2.0f, 4.0f, 5.0f, 8.0f, 10.0f, 12.0f});
        expected = {5.0f, 5.0f, 6.0f, 5.0f, 5.0f, 5.0f};
      },
      1,
  });

  // -------------------------------------------------------------
  // Misc unary with baked-in scalar args.
  // -------------------------------------------------------------

  // clamp_(-1, 1): bake-in scalar bounds. Inputs out of range are clipped.
  cases.push_back({
      "clamp_  (unary, scalar bounds baked in)",
      "/tmp/inplace_clamp_.pte",
      [](auto& self, auto& others, auto& expected) {
        self = {-3.0f, -0.5f, 0.0f, 0.5f, 2.5f, 5.0f};
        expected = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.0f};
        (void)others;
      },
      0,
  });

  std::printf("\n=== In-place op integration tests ===\n");
  size_t n_pass = 0;
  size_t n_total = cases.size();
  for (const auto& tc : cases) {
    if (run_case(tc)) ++n_pass;
  }

  // Heterogeneous-input tests (separate from the homogeneous TestCase loop).
  ++n_total;
  if (run_index_put_case()) ++n_pass;
  ++n_total;
  if (run_index_add_case()) ++n_pass;
  ++n_total;
  if (run_scatter_add_case()) ++n_pass;
  ++n_total;
  if (run_masked_scatter_case()) ++n_pass;
  ++n_total;
  if (run_kvcache_index_copy_case()) ++n_pass;

  std::printf("\n=== Summary: %zu / %zu passed ===\n", n_pass, n_total);
  return n_pass == n_total ? 0 : 1;
}
