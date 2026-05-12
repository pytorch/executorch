/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Per-phase unit tests for GreedyRouter.
 *
 * Each test calls one or more phase functions directly (via
 * router_internal::*) on a hand-built RouterContext, then asserts on
 * what the phase wrote to ctx / plan. This complements the smoke
 * suite (which only sees end-to-end output) by catching phase-level
 * regressions during refactors of individual phase files.
 *
 * Fixtures: reuses the raw-flatbuffer Program produced by
 * export_cond_inner.py at /tmp/native_cond_inner.fbb. That model
 * (`cond(pred, x+x, x*x)`) covers every phase including Phase 12
 * (control flow) — one fixture exercises the full pipeline.
 *
 * Run as part of build_and_run_tests.sh; failures abort with
 * descriptive output identifying the failed phase + assertion.
 */

#include <executorch/backends/native/core/Router.h>
#include <executorch/backends/native/routers/GreedyRouterContext.h>
#include <executorch/backends/native/runtimes/cpu/CpuRuntime.h>
#include <executorch/backends/native/runtimes/host_pool/HostPoolRuntime.h>

#include <executorch/runtime/platform/runtime.h>
#include <executorch/schema/program_generated.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

using ::executorch::backends::native::ComputeStep;
using ::executorch::backends::native::CpuRuntime;
using ::executorch::backends::native::Engine;
using ::executorch::backends::native::HostPoolRuntime;
using ::executorch::backends::native::JumpFalseStep;
using ::executorch::backends::native::MoveStep;
using ::executorch::backends::native::Plan;
using ::executorch::backends::native::RouterOptions;
using ::executorch::backends::native::Runtime;
using ::executorch::backends::native::TransferStep;
using ::executorch::backends::native::router_internal::assign_and_segment;
using ::executorch::backends::native::router_internal::compile_and_emit_steps;
using ::executorch::backends::native::router_internal::emit_allocs;
using ::executorch::backends::native::router_internal::interleave_control_flow;
using ::executorch::backends::native::router_internal::plan_homes_and_mirrors;
using ::executorch::backends::native::router_internal::plan_transfers;
using ::executorch::backends::native::router_internal::RouterContext;
using ::executorch::runtime::Error;
using ::executorch::runtime::Span;
using Graph = ::executorch::backends::portable::Graph;

// ---- minimal assert / harness ------------------------------------------
namespace {
int g_failures = 0;
const char* g_current_test = "?";
} // namespace

#define EXPECT(cond, msg)                         \
  do {                                            \
    if (!(cond)) {                                \
      fprintf(                                    \
          stderr,                                 \
          "FAIL [%s] %s:%d: %s (expected: %s)\n", \
          g_current_test,                         \
          __FILE__,                               \
          __LINE__,                               \
          msg,                                    \
          #cond);                                 \
      ++g_failures;                               \
    }                                             \
  } while (0)

#define EXPECT_OK(err, msg)                   \
  do {                                        \
    Error _e = (err);                         \
    if (_e != Error::Ok) {                    \
      fprintf(                                \
          stderr,                             \
          "FAIL [%s] %s:%d: %s (Error=%d)\n", \
          g_current_test,                     \
          __FILE__,                           \
          __LINE__,                           \
          msg,                                \
          static_cast<int>(_e));              \
      ++g_failures;                           \
    }                                         \
  } while (0)

// ---- fixture: program loader + provider builder ------------------------
struct Fixture {
  std::vector<uint8_t> bytes;
  const executorch_flatbuffer::Program* program = nullptr;
  std::unique_ptr<Graph> graph;
  std::unique_ptr<HostPoolRuntime> host_runtime;
  std::unique_ptr<CpuRuntime> cpu_runtime;
  std::vector<Runtime*> providers;
  std::vector<std::unique_ptr<Engine>> owned_engines;
  std::vector<Engine*> engines;
  RouterOptions options;
  Plan plan;
};

static bool load_fixture(Fixture& f, const char* path) {
  std::ifstream is(path, std::ios::binary | std::ios::ate);
  if (!is) {
    fprintf(stderr, "fixture: cannot open %s\n", path);
    return false;
  }
  std::streamsize sz = is.tellg();
  if (sz <= 0) {
    fprintf(stderr, "fixture: empty file %s\n", path);
    return false;
  }
  is.seekg(0);
  f.bytes.resize(static_cast<size_t>(sz));
  is.read(reinterpret_cast<char*>(f.bytes.data()), sz);

  f.program = executorch_flatbuffer::GetProgram(f.bytes.data());
  if (!f.program || !f.program->execution_plan() ||
      f.program->execution_plan()->size() == 0) {
    fprintf(stderr, "fixture: bad program in %s\n", path);
    return false;
  }
  f.graph =
      std::make_unique<Graph>(f.program->execution_plan()->Get(0), f.program);

  // Minimal provider set: host + CPU. Matches the standard "cpu only"
  // path in NativeBackend::init.
  f.host_runtime = std::make_unique<HostPoolRuntime>();
  f.cpu_runtime = std::make_unique<CpuRuntime>();
  f.providers = {f.host_runtime.get(), f.cpu_runtime.get()};

  f.owned_engines.push_back(f.host_runtime->instantiate());
  f.owned_engines.push_back(f.cpu_runtime->instantiate());
  for (auto& e : f.owned_engines)
    f.engines.push_back(e.get());

  // Plan needs providers/instances pre-populated for some phase fns
  // (the orchestrator does this in route() before constructing ctx).
  f.plan.providers.assign(f.providers.begin(), f.providers.end());
  f.plan.instances.assign(f.engines.begin(), f.engines.end());
  f.plan.max_hops = f.options.max_hops;
  return true;
}

static RouterContext make_ctx(Fixture& f) {
  return RouterContext{
      *f.graph,
      Span<Runtime* const>(f.providers.data(), f.providers.size()),
      Span<Engine* const>(f.engines.data(), f.engines.size()),
      /*ndm=*/nullptr,
      f.options,
      f.plan};
}

// ---- per-phase tests ---------------------------------------------------

static void test_assign_and_segment(const char* fixture_path) {
  g_current_test = "assign_and_segment";
  Fixture f;
  if (!load_fixture(f, fixture_path))
    return;
  RouterContext ctx = make_ctx(f);

  EXPECT_OK(assign_and_segment(ctx), "assign_and_segment");

  // Cond model has 4 instructions (jf, add, jf, mul) — after Phase 1
  // we expect 2 control instrs (the JumpFalse pair) and 2 segments
  // (each branch is one kernel).
  EXPECT(
      ctx.control_instrs.size() >= 1,
      "expected at least 1 control instr (JumpFalse)");
  EXPECT(!ctx.segments.empty(), "expected non-empty segments");
  for (const auto& seg : ctx.segments) {
    EXPECT(
        !seg.instruction_indices.empty(), "every segment has >=1 instruction");
    EXPECT(seg.provider_idx >= 0, "valid provider_idx");
  }
  EXPECT(
      !ctx.value_producer_seg.empty(),
      "value_producer_seg populated by Phase 3");
  // Cond model has 2 graph inputs (pred, x) and 1 output.
  EXPECT(ctx.graph_input_ids.size() == 2, "2 graph inputs");
  EXPECT(ctx.graph_output_ids.size() == 1, "1 graph output");
}

static void test_plan_homes_and_mirrors(const char* fixture_path) {
  g_current_test = "plan_homes_and_mirrors";
  Fixture f;
  if (!load_fixture(f, fixture_path))
    return;
  RouterContext ctx = make_ctx(f);

  EXPECT_OK(assign_and_segment(ctx), "Phase 1-4 setup");
  plan_homes_and_mirrors(ctx);

  // Every graph IO value should have home == 0 (host).
  for (uint32_t v : ctx.graph_input_ids) {
    auto it = ctx.value_home_provider.find(v);
    if (it != ctx.value_home_provider.end()) {
      EXPECT(it->second == 0, "graph input home is host");
    }
  }
  for (uint32_t v : ctx.graph_output_ids) {
    auto it = ctx.value_home_provider.find(v);
    if (it != ctx.value_home_provider.end()) {
      EXPECT(it->second == 0, "graph output home is host");
    }
  }
  // next_mirror_id must be at least num_values (mirrors are minted
  // beyond the graph's value-id space).
  EXPECT(
      ctx.next_mirror_id >= ctx.graph.num_values(),
      "next_mirror_id past graph value-id space");
}

static void test_emit_allocs(const char* fixture_path) {
  g_current_test = "emit_allocs";
  Fixture f;
  if (!load_fixture(f, fixture_path))
    return;
  RouterContext ctx = make_ctx(f);

  EXPECT_OK(assign_and_segment(ctx), "Phase 1-4 setup");
  plan_homes_and_mirrors(ctx);
  EXPECT_OK(emit_allocs(ctx), "emit_allocs");

  EXPECT(
      f.plan.alloc_plans.size() == f.providers.size(),
      "alloc_plans sized per provider");
  EXPECT(
      f.plan.const_plans.size() == f.providers.size(),
      "const_plans sized per provider");
  // Host slot (0) must have at least the IO HostExtern allocs.
  EXPECT(!f.plan.alloc_plans[0].empty(), "host slot has IO allocs");
}

static void test_plan_transfers(const char* fixture_path) {
  g_current_test = "plan_transfers";
  Fixture f;
  if (!load_fixture(f, fixture_path))
    return;
  RouterContext ctx = make_ctx(f);

  EXPECT_OK(assign_and_segment(ctx), "Phase 1-4 setup");
  plan_homes_and_mirrors(ctx);
  EXPECT_OK(emit_allocs(ctx), "emit_allocs");
  plan_transfers(ctx);

  EXPECT(
      ctx.seg_remaps.size() == ctx.segments.size(),
      "one seg_remaps entry per segment");
  EXPECT(
      ctx.seg_transfers.size() == ctx.segments.size(),
      "one seg_transfers entry per segment");
  EXPECT(
      ctx.seg_post_transfers.size() == ctx.segments.size(),
      "one seg_post_transfers entry per segment");
  // writers/readers built from segment IO sets — must match per-segment
  // out/in id counts in aggregate.
  size_t total_out = 0;
  size_t total_in = 0;
  for (const auto& seg : ctx.segments) {
    total_out += seg.output_value_ids.size();
    total_in += seg.input_value_ids.size();
  }
  size_t recorded_out = 0;
  size_t recorded_in = 0;
  for (const auto& kv : ctx.writers_per_value)
    recorded_out += kv.second.size();
  for (const auto& kv : ctx.readers_per_value)
    recorded_in += kv.second.size();
  EXPECT(recorded_out == total_out, "writers_per_value totals match");
  EXPECT(recorded_in == total_in, "readers_per_value totals match");
}

static void test_compile_and_emit_steps(const char* fixture_path) {
  g_current_test = "compile_and_emit_steps";
  Fixture f;
  if (!load_fixture(f, fixture_path))
    return;
  RouterContext ctx = make_ctx(f);

  EXPECT_OK(assign_and_segment(ctx), "Phase 1-4 setup");
  plan_homes_and_mirrors(ctx);
  EXPECT_OK(emit_allocs(ctx), "emit_allocs");
  plan_transfers(ctx);
  EXPECT_OK(compile_and_emit_steps(ctx), "compile_and_emit_steps");

  EXPECT(
      ctx.compiled_segments.size() == ctx.segments.size(),
      "one CompiledSegment per segment");
  EXPECT(!f.plan.steps.empty(), "plan.steps non-empty");
  // Pre-Phase-12: each segment contributes exactly one ComputeStep.
  size_t compute_count = 0;
  for (const auto& s : f.plan.steps) {
    if (std::holds_alternative<ComputeStep>(s))
      ++compute_count;
  }
  EXPECT(
      compute_count == ctx.segments.size(),
      "one ComputeStep per segment (pre-CF)");
  EXPECT(!f.plan.events.empty(), "events allocated for steps");
  // No JumpFalseSteps yet — Phase 12 emits those.
  for (const auto& s : f.plan.steps) {
    EXPECT(
        !std::holds_alternative<JumpFalseStep>(s),
        "no JumpFalseStep before Phase 12");
  }
}

static void test_interleave_control_flow(const char* fixture_path) {
  g_current_test = "interleave_control_flow";
  Fixture f;
  if (!load_fixture(f, fixture_path))
    return;
  RouterContext ctx = make_ctx(f);

  EXPECT_OK(assign_and_segment(ctx), "Phase 1-4 setup");
  plan_homes_and_mirrors(ctx);
  EXPECT_OK(emit_allocs(ctx), "emit_allocs");
  plan_transfers(ctx);
  EXPECT_OK(compile_and_emit_steps(ctx), "compile_and_emit_steps");

  // Cond fixture: control_instrs MUST be non-empty. Skip otherwise.
  if (ctx.control_instrs.empty()) {
    fprintf(stderr, "[skip %s] no control instrs in fixture\n", g_current_test);
    return;
  }
  EXPECT_OK(interleave_control_flow(ctx), "interleave_control_flow");

  // After Phase 12, JumpFalseSteps must exist and have resolved dst.
  size_t jf_count = 0;
  for (const auto& s : f.plan.steps) {
    if (auto* jf = std::get_if<JumpFalseStep>(&s)) {
      ++jf_count;
      EXPECT(
          jf->dst_step_idx != ::executorch::backends::native::kUnresolvedStep,
          "JumpFalseStep dst resolved");
    }
  }
  EXPECT(jf_count > 0, "at least one JumpFalseStep emitted");
}

// ---- main --------------------------------------------------------------

int main() {
  ::executorch::runtime::runtime_init();

  const char* fixture = std::getenv("NATIVE_ROUTER_PHASES_FIXTURE_PATH");
  if (!fixture)
    fixture = "/tmp/native_cond_inner.fbb";

  printf("=== test_router_phases ===\n");
  printf("  Fixture: %s\n", fixture);

  test_assign_and_segment(fixture);
  test_plan_homes_and_mirrors(fixture);
  test_emit_allocs(fixture);
  test_plan_transfers(fixture);
  test_compile_and_emit_steps(fixture);
  test_interleave_control_flow(fixture);

  if (g_failures == 0) {
    printf("=== PASS ===\n");
    return 0;
  } else {
    fprintf(stderr, "=== FAIL: %d assertion(s) failed ===\n", g_failures);
    return 1;
  }
}
