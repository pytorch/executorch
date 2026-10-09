// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/cpu/runtime/CPUPlan.h>
#include <executorch/backends/cpu/runtime/providers/executorch/ETProvider.h>
#include <executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>

namespace executorch::backends::cpu {
namespace {
using namespace executorch::runtime;

class CPUPlanTest : public ::testing::Test {
 protected:
  void value(const char* name, bool input = false) {
    graph.values.emplace_back(
        name, ptn::ScalarType::Float, std::vector<int64_t>{11});
    if (input) {
      graph.input_ids.push_back(
          static_cast<ptn::ValueId>(graph.values.size() - 1));
      graph.values.back().role = ptn::ValueRole::UserInput;
    }
  }
  void add(const char* name, ptn::ValueId a, ptn::ValueId b, ptn::ValueId out) {
    ptn::Node node;
    node.name = name;
    node.target = "torch.ops.aten.add.Tensor";
    node.inputs = {
        {"self", ptn::TensorArg{a}},
        {"other", ptn::TensorArg{b}},
        {"alpha", ptn::IntArg{1}}};
    node.outputs = {{ptn::OutputValueKind::Tensor, out, {}}};
    graph.nodes.push_back(std::move(node));
  }
  void gelu(const char* name, ptn::ValueId in, ptn::ValueId out) {
    ptn::Node node;
    node.name = name;
    node.target = "torch.ops.aten.gelu.default";
    node.inputs = {
        {"self", ptn::TensorArg{in}}, {"approximate", ptn::StringArg{"tanh"}}};
    node.outputs = {{ptn::OutputValueKind::Tensor, out, {}}};
    graph.nodes.push_back(std::move(node));
  }
  void finish(std::vector<ptn::ValueId> outputs) {
    graph.output_ids = std::move(outputs);
    graph.initialize_schedule();
    graph.rebuild_def_use();
    buffers.resize(graph.values.size());
  }
  static float gelu(float x) {
    return x * 0.5f *
        (1 + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
  }
  template <size_t Capacity>
  BackendInitContext context_for(BackendOptions<Capacity>& options) {
    const auto values = options.view();
    return BackendInitContext(
        &allocator, nullptr, nullptr, nullptr, {values.data(), values.size()});
  }
  alignas(128) std::array<uint8_t, 65536> storage{};
  MemoryAllocator allocator{
      static_cast<uint32_t>(storage.size()),
      storage.data()};
  ptn::Graph graph;
  std::vector<Buffer> buffers;
  ExecutionContext execution;
  const std::array<ProviderFactory, 2> factories{
      create_xnnpack_provider,
      create_et_provider};
  RuntimeConfiguration configuration{
      {factories.data(), factories.size()},
      {},
      false};
};

// Cppcheck cannot expand GTest fixture macros without build headers.
// cppcheck-suppress-begin syntaxError
TEST_F(
    CPUPlanTest,
    MixedRegionsPreserveFanoutOutputsAndReuseArenaAcrossInvocations) {
  value("x", true);
  value("y", true);
  for (const auto* name : {"sum", "gelu", "skip", "second_gelu", "out"}) {
    value(name);
  }
  add("sum", 0, 1, 2);
  gelu("gelu", 2, 3);
  add("skip", 2, 3, 4);
  gelu("second_gelu", 4, 5);
  add("out", 3, 5, 6);
  finish({2, 6});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 5);
  EXPECT_EQ(plan.steps()[0].provider->name(), "XNNPACK");
  EXPECT_EQ(plan.steps()[1].provider->name(), "ET");
  EXPECT_EQ(plan.steps()[2].region.inputs, (std::vector<ptn::ValueId>{2, 3}));
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  EXPECT_LT(plan.arena_bytes(), 7 * 128);
  EXPECT_NE(buffers[0].data, buffers[2].data);
  EXPECT_NE(buffers[1].data, buffers[2].data);
  EXPECT_NE(buffers[3].data, buffers[4].data);
  EXPECT_NE(buffers[4].data, buffers[5].data);
  EXPECT_NE(buffers[2].data, buffers[6].data);
  for (float input : {1.0f, -0.25f, 2.0f}) {
    std::fill_n(static_cast<float*>(buffers[0].data), 11, input);
    std::fill_n(static_cast<float*>(buffers[1].data), 11, input);
    ASSERT_EQ(plan.execute(execution), Error::Ok);
    EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[10], 2 * input);
    EXPECT_NEAR(
        static_cast<float*>(buffers[6].data)[10],
        gelu(2 * input) + gelu(2 * input + gelu(2 * input)),
        1e-5);
  }
  const float before = static_cast<float*>(buffers[6].data)[0];
  buffers[5].readable_bytes = 0;
  EXPECT_EQ(plan.execute(execution), Error::InvalidArgument);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[6].data)[0], before);
}

TEST_F(CPUPlanTest, ConnectedRegionsExposeIntermediateModelOutputs) {
  value("x", true);
  value("y", true);
  value("sum");
  value("out");
  add("sum", 0, 1, 2);
  add("out", 2, 1, 3);
  finish({2, 3});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].region.outputs, (std::vector<ptn::ValueId>{2, 3}));
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  std::fill_n(static_cast<float*>(buffers[0].data), 11, 1);
  std::fill_n(static_cast<float*>(buffers[1].data), 11, 2);
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[0], 3);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[3].data)[0], 5);
}

TEST_F(CPUPlanTest, BatchedMatrixMultiplyUsesXNNPACKForDynamicOperands) {
  graph.values.emplace_back(
      "lhs", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 3});
  graph.values.emplace_back(
      "rhs", ptn::ScalarType::Float, std::vector<int64_t>{2, 3, 2});
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 2});
  graph.values[0].role = ptn::ValueRole::UserInput;
  graph.values[1].role = ptn::ValueRole::UserInput;
  graph.input_ids = {0, 1};
  ptn::Node bmm;
  bmm.name = "bmm";
  bmm.target = "torch.ops.aten.bmm.default";
  bmm.inputs = {{"self", ptn::TensorArg{0}}, {"mat2", ptn::TensorArg{1}}};
  bmm.outputs = {{ptn::OutputValueKind::Tensor, 2, {}}};
  graph.nodes.push_back(std::move(bmm));
  finish({2});

  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].provider->name(), "XNNPACK");
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  const std::array<float, 12> lhs{1, 2, 3, 4, 5, 6, 1, 0, 2, 0, 1, 3};
  const std::array<float, 12> rhs{1, 2, 3, 4, 5, 6, 2, 0, 0, 3, 4, 5};
  std::copy(lhs.begin(), lhs.end(), static_cast<float*>(buffers[0].data));
  std::copy(rhs.begin(), rhs.end(), static_cast<float*>(buffers[1].data));
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  const auto* result = static_cast<const float*>(buffers[2].data);
  EXPECT_EQ(
      (std::vector<float>(result, result + 8)),
      (std::vector<float>{22, 28, 49, 64, 10, 10, 12, 18}));
}

TEST_F(CPUPlanTest, LastDimensionSoftmaxUsesXNNPACKForDynamicInput) {
  graph.values.emplace_back(
      "input", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 3});
  graph.values.back().role = ptn::ValueRole::UserInput;
  graph.values.emplace_back(
      "output", ptn::ScalarType::Float, std::vector<int64_t>{2, 2, 3});
  graph.input_ids = {0};
  ptn::Node softmax;
  softmax.name = "softmax";
  softmax.target = "torch.ops.aten._softmax.default";
  softmax.inputs = {
      {"self", ptn::TensorArg{0}},
      {"dim", ptn::IntArg{-1}},
      {"half_to_float", ptn::BoolArg{false}}};
  softmax.outputs = {{ptn::OutputValueKind::Tensor, 1}};
  graph.nodes.push_back(std::move(softmax));
  finish({1});

  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].provider->name(), "XNNPACK");
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  const std::array<float, 12> input{
      0, 0, 0, 0, 1, 2, 1000, 1000, 1000, -1000, -999, -998};
  std::copy(input.begin(), input.end(), static_cast<float*>(buffers[0].data));
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  const float denominator = 1 + std::exp(1.0f) + std::exp(2.0f);
  const std::array<float, 12> expected{
      1.0f / 3,
      1.0f / 3,
      1.0f / 3,
      1 / denominator,
      std::exp(1.0f) / denominator,
      std::exp(2.0f) / denominator,
      1.0f / 3,
      1.0f / 3,
      1.0f / 3,
      1 / denominator,
      std::exp(1.0f) / denominator,
      std::exp(2.0f) / denominator};
  const auto* result = static_cast<const float*>(buffers[1].data);
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(result[index], expected[index], 1e-5) << index;
  }
}

TEST_F(
    CPUPlanTest,
    DisconnectedBranchesRemainSeparateAndRejectNonTopologicalSchedule) {
  value("x", true);
  value("y", true);
  value("a");
  value("b");
  value("out");
  add("a", 0, 1, 2);
  add("b", 0, 1, 3);
  add("out", 2, 3, 4);
  finish({4});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 2);
  EXPECT_EQ(plan.steps()[0].region.nodes, (std::vector<KernelId>{0}));
  EXPECT_EQ(plan.steps()[1].region.nodes, (std::vector<KernelId>{1, 2}));
  graph.schedule = {2, 0, 1};
  CPUPlan invalid(graph, buffers, configuration, execution);
  EXPECT_EQ(invalid.select(), Error::InvalidProgram);
}

TEST_F(CPUPlanTest, PreferenceForceAndExecutionSignatureAreChecked) {
  value("x", true);
  value("y", true);
  value("out");
  add("out", 0, 1, 2);
  finish({2});
  configuration.preferences = {{"ET", ""}};
  configuration.force = true;
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  EXPECT_EQ(plan.steps()[0].provider->name(), "ET");
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  auto changed = execution;
  changed.threads = 2;
  EXPECT_EQ(plan.execute(changed), Error::NotSupported);
  configuration.preferences = {{"unlinked", ""}};
  CPUPlan missing(graph, buffers, configuration, execution);
  EXPECT_EQ(missing.select(), Error::InvalidArgument);
  configuration.preferences = {{"XNNPACK", ""}};
  graph.nodes[0].inputs[2].arg = ptn::IntArg{2};
  CPUPlan unmatched(graph, buffers, configuration, execution);
  EXPECT_EQ(unmatched.select(), Error::NotSupported);
}

class SelectionImplementation final : public KernelImplementation {
 public:
  SelectionImplementation(const char* name, const char* target, int priority)
      : name_(name), target_(target), priority_(priority) {}
  std::string_view name() const override {
    return name_;
  }
  int baseline_priority() const override {
    return priority_;
  }
  Support supports(
      const Kernel& kernel,
      const ptn::Graph&,
      const ExecutionContext&) const override {
    return {
        target_.empty() || kernel.target == target_, "test selection route"};
  }
  Result<std::unique_ptr<Executable>> compile(
      const KernelRegion&,
      PreparationContext&) override {
    return Error::NotSupported;
  }

 private:
  std::string_view name_;
  std::string_view target_;
  int priority_;
};

class SelectionProvider final : public KernelProvider {
 public:
  std::string_view name() const override {
    return "selection_test";
  }
  std::vector<KernelImplementation*> implementations() override {
    return {&general_, &add_};
  }

 private:
  SelectionImplementation general_{"general", "", 1000};
  SelectionImplementation add_{"add", "torch.ops.aten.add.Tensor", -1000};
};

std::unique_ptr<KernelProvider> create_selection_provider() {
  return std::make_unique<SelectionProvider>();
}

class TiedProvider final : public KernelProvider {
 public:
  std::string_view name() const override {
    return "tied_test";
  }
  std::vector<KernelImplementation*> implementations() override {
    return {&first_, &second_};
  }

 private:
  SelectionImplementation first_{"first", "", 5};
  SelectionImplementation second_{"second", "", 5};
};

std::unique_ptr<KernelProvider> create_tied_provider() {
  return std::make_unique<TiedProvider>();
}

TEST_F(CPUPlanTest, EqualEligibleCandidatesRequireAPreference) {
  value("x", true);
  value("out");
  gelu("out", 0, 1);
  finish({1});
  const std::array<ProviderFactory, 1> providers{create_tied_provider};
  configuration.providers = {providers.data(), providers.size()};
  CPUPlan tied(graph, buffers, configuration, execution);
  EXPECT_EQ(tied.select(), Error::InvalidArgument);
  configuration.preferences = {{"tied_test", "second"}};
  CPUPlan preferred(graph, buffers, configuration, execution);
  ASSERT_EQ(preferred.select(), Error::Ok);
  EXPECT_EQ(preferred.steps()[0].implementation->name(), "second");
}

TEST_F(CPUPlanTest, SpecificImplementationsPrecedeAProviderWideFallback) {
  value("x", true);
  value("y", true);
  for (const auto* name : {"sum", "exact", "approximate"}) {
    value(name);
  }
  add("sum", 0, 1, 2);
  gelu("exact", 2, 3);
  graph.nodes.back().inputs[1].arg = ptn::StringArg{"none"};
  gelu("approximate", 3, 4);
  finish({4});
  const std::array<ProviderFactory, 3> providers{
      create_selection_provider, create_xnnpack_provider, create_et_provider};
  configuration.providers = {providers.data(), providers.size()};
  configuration.preferences = {
      {"selection_test", "add"}, {"XNNPACK", "subgraph"}, {"ET", ""}};
  configuration.force = true;
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 3);
  EXPECT_EQ(plan.steps()[0].implementation->name(), "add");
  EXPECT_EQ(plan.steps()[1].provider->name(), "XNNPACK");
  EXPECT_EQ(plan.steps()[2].provider->name(), "ET");
}

TEST_F(CPUPlanTest, UnbindableETNodesFallBackDuringSelection) {
  value("x", true);
  value("out");
  add("out", 0, 0, 1);
  finish({1});
  const std::array<ProviderFactory, 2> providers{
      create_et_provider, create_selection_provider};
  configuration.providers = {providers.data(), providers.size()};
  configuration.preferences = {{"ET", ""}};
  CPUPlan supported(graph, buffers, configuration, execution);
  ASSERT_EQ(supported.select(), Error::Ok);
  ASSERT_EQ(supported.steps().size(), 1);
  EXPECT_EQ(supported.steps()[0].provider->name(), "ET");

  auto& node = graph.nodes[0];
  node.inputs[2].arg = ptn::FloatArg{1.0, 0};
  CPUPlan dynamic_argument(graph, buffers, configuration, execution);
  ASSERT_EQ(dynamic_argument.select(), Error::Ok);
  ASSERT_EQ(dynamic_argument.steps().size(), 1);
  EXPECT_EQ(dynamic_argument.steps()[0].provider->name(), "selection_test");

  graph.values[1] = ptn::Value("out", ptn::ScalarType::Float, {5});
  node.target = "torch.ops.aten.split_with_sizes_copy.default";
  node.inputs = {
      {"self", ptn::TensorArg{0}},
      {"split_sizes", ptn::IntListArg{{5, 6}, {}}},
      {"dim", ptn::IntArg{0}}};
  node.outputs = {
      {ptn::OutputValueKind::TensorList, ptn::kInvalid, {1, ptn::kInvalid}}};
  graph.rebuild_def_use();
  CPUPlan sparse_outputs(graph, buffers, configuration, execution);
  ASSERT_EQ(sparse_outputs.select(), Error::Ok);
  ASSERT_EQ(sparse_outputs.steps().size(), 1);
  EXPECT_EQ(sparse_outputs.steps()[0].provider->name(), "selection_test");
}

TEST_F(CPUPlanTest, FirstMatchingPreferenceWinsBeforeNumericPriority) {
  value("x", true);
  value("y", true);
  value("out");
  add("out", 0, 1, 2);
  finish({2});
  const std::array<ProviderFactory, 2> providers{
      create_selection_provider, create_et_provider};
  configuration.providers = {providers.data(), providers.size()};
  configuration.preferences = {{"ET", ""}, {"selection_test", "add"}};
  CPUPlan et_first(graph, buffers, configuration, execution);
  ASSERT_EQ(et_first.select(), Error::Ok);
  EXPECT_EQ(et_first.steps()[0].provider->name(), "ET");
  configuration.preferences = {{"selection_test", "add"}, {"ET", ""}};
  CPUPlan specific_first(graph, buffers, configuration, execution);
  ASSERT_EQ(specific_first.select(), Error::Ok);
  EXPECT_EQ(specific_first.steps()[0].implementation->name(), "add");
  configuration.preferences = {
      {"selection_test", ""}, {"selection_test", "add"}};
  CPUPlan wildcard_first(graph, buffers, configuration, execution);
  ASSERT_EQ(wildcard_first.select(), Error::Ok);
  EXPECT_EQ(wildcard_first.steps()[0].implementation->name(), "general");
  configuration.preferences = {
      {"selection_test", "add"}, {"selection_test", ""}};
  CPUPlan wildcard_last(graph, buffers, configuration, execution);
  ASSERT_EQ(wildcard_last.select(), Error::Ok);
  EXPECT_EQ(wildcard_last.steps()[0].implementation->name(), "add");
}

TEST_F(CPUPlanTest, UnmatchedPreferencesFallBackAndInvalidNamesAreRejected) {
  value("x", true);
  value("out");
  gelu("out", 0, 1);
  finish({1});
  const std::array<ProviderFactory, 2> providers{
      create_selection_provider, create_et_provider};
  configuration.providers = {providers.data(), providers.size()};
  configuration.preferences = {{"selection_test", "add"}};
  CPUPlan fallback(graph, buffers, configuration, execution);
  ASSERT_EQ(fallback.select(), Error::Ok);
  EXPECT_EQ(fallback.steps()[0].implementation->name(), "general");
  configuration.force = true;
  CPUPlan forced(graph, buffers, configuration, execution);
  EXPECT_EQ(forced.select(), Error::NotSupported);
  configuration.preferences.push_back({"ET", ""});
  CPUPlan later_match(graph, buffers, configuration, execution);
  ASSERT_EQ(later_match.select(), Error::Ok);
  EXPECT_EQ(later_match.steps()[0].provider->name(), "ET");
  configuration.preferences.push_back({"selection_test", "missing"});
  CPUPlan unknown(graph, buffers, configuration, execution);
  EXPECT_EQ(unknown.select(), Error::InvalidArgument);
  configuration.preferences = {{"", "add"}};
  CPUPlan empty_provider(graph, buffers, configuration, execution);
  EXPECT_EQ(empty_provider.select(), Error::InvalidArgument);
  configuration.preferences.clear();
  CPUPlan empty_forced(graph, buffers, configuration, execution);
  EXPECT_EQ(empty_forced.select(), Error::InvalidArgument);
}

TEST_F(CPUPlanTest, IndexedLoadPreferencesReplaceDefaultsAndOwnTheirStrings) {
  value("x", true);
  value("y", true);
  value("out");
  add("out", 0, 1, 2);
  finish({2});
  configuration.preferences = {{"XNNPACK", ""}};
  BackendOptions<5> options;
  ASSERT_EQ(options.set_option("preference_count", 2), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_provider_0", "ET"), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_provider_1", "XNNPACK"), Error::Ok);
  ASSERT_EQ(
      options.set_option("preferred_implementation_1", "subgraph"), Error::Ok);
  ASSERT_EQ(options.set_option("force", true), Error::Ok);
  auto context = context_for(options);
  ASSERT_EQ(configuration.apply_options(context), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_provider_0", "unlinked"), Error::Ok);
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  EXPECT_EQ(plan.steps()[0].provider->name(), "ET");
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  std::fill_n(static_cast<float*>(buffers[0].data), 11, 1);
  std::fill_n(static_cast<float*>(buffers[1].data), 11, 2);
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  EXPECT_FLOAT_EQ(static_cast<float*>(buffers[2].data)[0], 3);
  BackendOptions<2> clear;
  ASSERT_EQ(clear.set_option("preference_count", 0), Error::Ok);
  ASSERT_EQ(clear.set_option("force", false), Error::Ok);
  auto clear_context = context_for(clear);
  ASSERT_EQ(configuration.apply_options(clear_context), Error::Ok);
  CPUPlan baseline(graph, buffers, configuration, execution);
  ASSERT_EQ(baseline.select(), Error::Ok);
  EXPECT_EQ(baseline.steps()[0].provider->name(), "XNNPACK");
}

TEST_F(CPUPlanTest, SinglePairLoadOptionsRemainAShorthand) {
  value("x", true);
  value("y", true);
  value("out");
  add("out", 0, 1, 2);
  finish({2});
  configuration.preferences = {{"XNNPACK", ""}, {"ET", ""}};
  BackendInitContext absent(&allocator);
  ASSERT_EQ(configuration.apply_options(absent), Error::Ok);
  CPUPlan defaults(graph, buffers, configuration, execution);
  ASSERT_EQ(defaults.select(), Error::Ok);
  EXPECT_EQ(defaults.steps()[0].provider->name(), "XNNPACK");
  BackendOptions<1> options;
  ASSERT_EQ(options.set_option("preferred_provider", "ET"), Error::Ok);
  auto context = context_for(options);
  ASSERT_EQ(configuration.apply_options(context), Error::Ok);
  CPUPlan loaded(graph, buffers, configuration, execution);
  ASSERT_EQ(loaded.select(), Error::Ok);
  EXPECT_EQ(loaded.steps()[0].provider->name(), "ET");
  ASSERT_EQ(configuration.preferences.size(), 1);
}

TEST_F(CPUPlanTest, LoadPreferencesRejectMalformedListsAndMixedForms) {
  BackendOptions<4> options;
  ASSERT_EQ(options.set_option("preference_count", "1"), Error::Ok);
  auto context = context_for(options);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preference_count", -1), Error::Ok);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preference_count", 1), Error::Ok);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preferred_provider_0", ""), Error::Ok);
  context = context_for(options);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preferred_provider_0", "ET"), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_implementation_0", true), Error::Ok);
  context = context_for(options);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
  ASSERT_EQ(options.set_option("preferred_implementation_0", ""), Error::Ok);
  ASSERT_EQ(options.set_option("preferred_provider", "XNNPACK"), Error::Ok);
  context = context_for(options);
  EXPECT_EQ(configuration.apply_options(context), Error::InvalidArgument);
}

struct ProbeState {
  int enumerations = 0;
  int compiled = 0;
  int destroyed = 0;
  bool fail_second = false;
  std::vector<void*> scratch;
};

class ProbeExecutable final : public Executable {
 public:
  explicit ProbeExecutable(std::shared_ptr<ProbeState> state)
      : state_(std::move(state)) {}
  ~ProbeExecutable() override {
    ++state_->destroyed;
  }
  Error reshape() override {
    return Error::Ok;
  }
  Error bind(const Buffer& scratch) override {
    state_->scratch.push_back(scratch.data);
    return scratch.accepts({640, 512, 128}) ? Error::Ok
                                            : Error::InvalidArgument;
  }
  Error run(const ExecutionContext&) override {
    return Error::Ok;
  }

 private:
  std::shared_ptr<ProbeState> state_;
};

class ProbeProvider final : public KernelProvider, public KernelImplementation {
 public:
  std::string_view name() const override {
    return "probe";
  }
  std::vector<KernelImplementation*> implementations() override {
    ++state_->enumerations;
    return {this};
  }
  int baseline_priority() const override {
    return 100;
  }
  Support supports(const Kernel&, const ptn::Graph&, const ExecutionContext&)
      const override {
    return {true, "test route"};
  }
  Result<StorageRequirements> requirements(
      const KernelRegion& region,
      const ptn::Graph& graph,
      const ExecutionContext& execution) const override {
    auto storage = KernelImplementation::requirements(region, graph, execution);
    if (!storage.ok()) {
      return storage.error();
    }
    for (auto& value : storage->values) {
      value.buffer.alignment = 128;
      value.buffer.readable_bytes += 128;
    }
    storage->scratch = {640, 512, 128};
    return storage;
  }
  Result<std::unique_ptr<Executable>> compile(
      const KernelRegion&,
      PreparationContext&) override {
    ++state_->compiled;
    if (state_->fail_second && state_->compiled == 2) {
      return Error::MemoryAllocationFailed;
    }
    return std::unique_ptr<Executable>(new ProbeExecutable(state_));
  }
  std::shared_ptr<ProbeState> state() const {
    return state_;
  }

 private:
  std::shared_ptr<ProbeState> state_ = std::make_shared<ProbeState>();
};
std::unique_ptr<KernelProvider> create_probe() {
  return std::make_unique<ProbeProvider>();
}

TEST_F(CPUPlanTest, DeclaredStoragePrecedesCompilationAndScratchIsShared) {
  value("x", true);
  value("y", true);
  value("a");
  value("b");
  add("a", 0, 1, 2);
  add("b", 2, 1, 3);
  finish({3});
  const ProviderFactory factory = create_probe;
  CPUPlan plan(graph, buffers, {{&factory, 1}, {}, false}, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  const auto probe =
      static_cast<ProbeProvider*>(plan.steps().at(0).provider)->state();
  EXPECT_EQ(probe->enumerations, 1);
  EXPECT_EQ(probe->compiled, 0);
  ASSERT_EQ(plan.prepare(allocator), Error::Ok);
  EXPECT_EQ(plan.scratch_bytes(), 640);
  for (const auto& buffer : buffers) {
    EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data) % 128, 0);
    EXPECT_GE(buffer.readable_bytes, 44 + 64 + 128);
  }
  ASSERT_EQ(plan.execute(execution), Error::Ok);
  ASSERT_EQ(probe->scratch.size(), 2);
  EXPECT_EQ(probe->scratch[0], probe->scratch[1]);
}

TEST_F(CPUPlanTest, PreparationFailureCleansUpAndCannotExecute) {
  std::shared_ptr<ProbeState> probe;
  value("x", true);
  value("y", true);
  value("a");
  value("b");
  add("a", 0, 1, 2);
  add("b", 2, 1, 3);
  finish({3});
  const ProviderFactory factory = create_probe;
  {
    CPUPlan plan(graph, buffers, {{&factory, 1}, {}, false}, execution);
    ASSERT_EQ(plan.select(), Error::Ok);
    probe = static_cast<ProbeProvider*>(plan.steps().at(0).provider)->state();
    probe->fail_second = true;
    EXPECT_EQ(plan.prepare(allocator), Error::MemoryAllocationFailed);
    EXPECT_EQ(plan.execute(execution), Error::InvalidState);
  }
  EXPECT_EQ(probe->destroyed, 1);
  buffers.assign(graph.values.size(), {});
  CPUPlan plan(graph, buffers, {{&factory, 1}, {}, false}, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  probe = static_cast<ProbeProvider*>(plan.steps().at(0).provider)->state();
  MemoryAllocator empty(0, nullptr);
  EXPECT_EQ(plan.prepare(empty), Error::MemoryAllocationFailed);
  EXPECT_EQ(probe->compiled, 0);
  graph.values[1].role = ptn::ValueRole::Parameter;
  buffers[1] = {storage.data(), 44 + kReadableTail, 0, 128};
  CPUPlan short_constant(graph, buffers, {{&factory, 1}, {}, false}, execution);
  ASSERT_EQ(short_constant.select(), Error::Ok);
  probe = static_cast<ProbeProvider*>(short_constant.steps().at(0).provider)
              ->state();
  EXPECT_EQ(short_constant.prepare(allocator), Error::InvalidArgument);
  EXPECT_EQ(probe->compiled, 0);
}

TEST_F(CPUPlanTest, TensorListOutputsPlanAgainstETRegistry) {
  value("x", true);
  value("first");
  value("second");
  ptn::Node split;
  split.name = "split";
  split.target = "torch.ops.aten.split_with_sizes_copy.default";
  split.inputs = {
      {"self", ptn::TensorArg{0}},
      {"split_sizes", ptn::IntListArg{{5, 6}, {}}},
      {"dim", ptn::IntArg{0}}};
  split.outputs = {{ptn::OutputValueKind::TensorList, ptn::kInvalid, {1, 2}}};
  graph.nodes.push_back(std::move(split));
  finish({1, 2});
  CPUPlan plan(graph, buffers, configuration, execution);
  ASSERT_EQ(plan.select(), Error::Ok);
  ASSERT_EQ(plan.steps().size(), 1);
  EXPECT_EQ(plan.steps()[0].provider->name(), "ET");
  EXPECT_EQ(plan.steps()[0].implementation->name(), "registry");
}
// cppcheck-suppress-end syntaxError
} // namespace
} // namespace executorch::backends::cpu
