// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/utils/Aliases.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace ptn {
namespace {

constexpr const char* kAdd = "torch.ops.aten.add.Tensor";
constexpr const char* kView = "torch.ops.aten.view.default";
constexpr const char* kTranspose = "torch.ops.aten.transpose.int";
constexpr const char* kAsStrided = "torch.ops.aten.as_strided.default";
constexpr const char* kRelu = "torch.ops.aten.relu_.default";
constexpr const char* kSigmoid = "torch.ops.aten.sigmoid_.default";

NamedArgument tensor(ValueId id, bool mutated = false) {
  return {.name = "self", .arg = TensorArg{id}, .mutated = mutated};
}

NamedArgument ints(std::vector<int64_t> values) {
  return {.arg = IntListArg{.values = std::move(values)}};
}

class GraphBuilder {
 public:
  ValueId input(
      const char* name,
      std::vector<int64_t> sizes,
      ValueRole role = ValueRole::UserInput) {
    const auto id = value(name, std::move(sizes), kInvalid);
    graph_.value(id).role = role;
    graph_.nodes.push_back(Node{
        .name = name,
        .op_kind = OpKind::Placeholder,
        .outputs = {{.value_id = id}},
    });
    graph_.input_ids.push_back(id);
    return id;
  }

  ValueId call(
      const char* name,
      const char* target,
      std::vector<NamedArgument> inputs,
      std::vector<int64_t> sizes,
      ValueId alias = kInvalid) {
    const auto id = value(name, std::move(sizes), alias);
    graph_.nodes.push_back(Node{
        .name = name,
        .target = target,
        .inputs = std::move(inputs),
        .outputs = {{.value_id = id}},
    });
    return id;
  }

  ValueId add(const char* name, ValueId left, ValueId right) {
    return call(
        name,
        kAdd,
        {tensor(left), tensor(right)},
        graph_.value(left).tensor_meta().sizes);
  }

  ValueId
  view(const char* name, ValueId source, const std::vector<int64_t>& sizes) {
    return call(name, kView, {tensor(source), ints(sizes)}, sizes, source);
  }

  ValueId transpose(const char* name, ValueId source) {
    auto sizes = graph_.value(source).tensor_meta().sizes;
    std::swap(sizes[0], sizes[1]);
    return call(
        name,
        kTranspose,
        {tensor(source),
         {.arg = IntArg{.value = 0}},
         {.arg = IntArg{.value = 1}}},
        sizes,
        source);
  }

  ValueId in_place(const char* name, const char* target, ValueId source) {
    return call(
        name,
        target,
        {tensor(source, true)},
        graph_.value(source).tensor_meta().sizes,
        source);
  }

  Graph finish(std::vector<ValueId> outputs) {
    Node node{.name = "output", .op_kind = OpKind::Output};
    for (const auto id : outputs) {
      node.inputs.push_back({.arg = TensorArg{id}});
    }
    graph_.nodes.push_back(std::move(node));
    graph_.output_ids = std::move(outputs);
    graph_.initialize_schedule();
    graph_.rebuild_def_use();
    return std::move(graph_);
  }

  Graph& graph() {
    return graph_;
  }

 private:
  ValueId value(const char* name, std::vector<int64_t> sizes, ValueId alias) {
    const auto id =
        graph_.append_value(Value(name, ScalarType::Float, std::move(sizes)));
    graph_.value(id).alias_id = alias;
    return id;
  }

  Graph graph_;
};

const Node& producer(const Graph& graph, ValueId id) {
  return graph.node(graph.value(id).producer_id);
}

void expect_functional(const Graph& graph) {
  for (const auto& value : graph.values) {
    EXPECT_FALSE(valid(value.alias_id)) << value.name;
  }
  for (const auto& node : graph.nodes) {
    for (const auto& input : node.inputs) {
      EXPECT_FALSE(input.mutated) << node.name;
    }
  }
  EXPECT_NO_THROW(validate_graph(graph));
}

void expect_failure(Graph graph, const std::string& reason) {
  try {
    functionalize_aliases(graph);
    ADD_FAILURE() << "expected failure: " << reason;
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(reason), std::string::npos)
        << error.what();
  }
}

// cppcheck-suppress-begin syntaxError
TEST(AliasesTest, AliasRootFollowsChainsAndRejectsCycles) {
  GraphBuilder builder;
  const auto x = builder.input("x", {2, 2});
  const auto flat = builder.view("flat", x, {4});
  const auto again = builder.view("again", flat, {4});
  const auto sum = builder.add("sum", again, again);
  auto graph = builder.finish({sum});

  EXPECT_EQ(alias_root(graph, x), x);
  EXPECT_EQ(alias_root(graph, again), x);
  EXPECT_EQ(alias_root(graph, sum), sum);

  graph.value(x).alias_id = again;
  EXPECT_EQ(alias_root(graph, again), kInvalid);
}

TEST(AliasesTest, DetectsMutationAnywhereInAliasGroup) {
  GraphBuilder builder;
  const auto x = builder.input("x", {4});
  const auto sum = builder.add("sum", x, x);
  const auto flat = builder.view("flat", sum, {4});
  const auto other = builder.add("other", x, x);
  const auto relu = builder.in_place("relu", kRelu, flat);
  auto graph = builder.finish({relu, other});

  EXPECT_TRUE(has_mutated_alias_use(graph, sum));
  EXPECT_TRUE(has_mutated_alias_use(graph, relu));
  EXPECT_FALSE(has_mutated_alias_use(graph, other));
  EXPECT_FALSE(has_mutated_alias_use(graph, x));

  graph.schedule.erase(graph.schedule.end() - 2);
  EXPECT_FALSE(has_mutated_alias_use(graph, sum));
}

TEST(AliasesTest, MaterializesViewsAsCopies) {
  GraphBuilder builder;
  const auto x = builder.input("x", {2, 2});
  const auto flat = builder.view("flat", x, {4});
  const auto transposed = builder.transpose("transposed", x);
  auto graph = builder.finish({flat, transposed});

  functionalize_aliases(graph);

  EXPECT_EQ(producer(graph, flat).target, "torch.ops.aten.view_copy.default");
  EXPECT_EQ(
      producer(graph, transposed).target, "torch.ops.aten.transpose_copy.int");
  EXPECT_EQ(graph.output_ids, std::vector<ValueId>({flat, transposed}));
  expect_functional(graph);
}

TEST(AliasesTest, FunctionalizesMutationAndRedirectsLaterReads) {
  GraphBuilder builder;
  const auto x = builder.input("x", {4});
  const auto sum = builder.add("sum", x, x);
  const auto relu = builder.in_place("relu", kRelu, sum);
  const auto sigmoid = builder.in_place("sigmoid", kSigmoid, sum);
  const auto reader = builder.add("reader", sum, x);
  auto graph = builder.finish({sum, reader});

  functionalize_aliases(graph);

  EXPECT_EQ(producer(graph, relu).target, "torch.ops.aten.relu.default");
  EXPECT_EQ(producer(graph, sigmoid).target, "torch.ops.aten.sigmoid.default");
  EXPECT_EQ(producer(graph, sigmoid).inputs[0].arg.as_tensor().id, relu);
  EXPECT_EQ(producer(graph, reader).inputs[0].arg.as_tensor().id, sigmoid);
  EXPECT_EQ(producer(graph, reader).inputs[1].arg.as_tensor().id, x);
  EXPECT_EQ(graph.output_ids, std::vector<ValueId>({sigmoid, reader}));
  EXPECT_EQ(
      graph.value(sum).consumer_ids,
      std::vector<NodeId>({graph.value(relu).producer_id}));
  expect_functional(graph);
}

TEST(AliasesTest, RedirectsTensorListReadsOfMutatedValue) {
  GraphBuilder builder;
  const auto x = builder.input("x", {4});
  const auto sum = builder.add("sum", x, x);
  const auto relu = builder.in_place("relu", kRelu, sum);
  const auto cat = builder.call(
      "cat",
      "torch.ops.aten.cat.default",
      {{.name = "tensors", .arg = TensorListArg{{sum, x}}}},
      {8});
  auto graph = builder.finish({cat});

  functionalize_aliases(graph);

  EXPECT_EQ(
      producer(graph, cat).inputs[0].arg.as_tensor_list().ids,
      std::vector<ValueId>({relu, x}));
  expect_functional(graph);
}

TEST(AliasesTest, RedirectsLiveViewWithSameIndexing) {
  GraphBuilder builder;
  const auto x = builder.input("x", {4});
  const auto sum = builder.add("sum", x, x);
  const auto flat = builder.view("flat", sum, {4});
  const auto relu = builder.in_place("relu", kRelu, sum);
  const auto reader = builder.add("reader", flat, flat);
  auto graph = builder.finish({reader, flat});

  functionalize_aliases(graph);

  EXPECT_EQ(producer(graph, reader).inputs[0].arg.as_tensor().id, relu);
  EXPECT_EQ(producer(graph, reader).inputs[1].arg.as_tensor().id, relu);
  EXPECT_EQ(graph.output_ids, std::vector<ValueId>({reader, relu}));
  expect_functional(graph);
}

TEST(AliasesTest, IgnoresAliasesWithoutLaterReaders) {
  GraphBuilder builder;
  const auto x = builder.input("x", {2, 2});
  const auto sum = builder.add("sum", x, x);
  const auto transposed = builder.transpose("transposed", sum);
  const auto reader = builder.add("reader", transposed, transposed);
  const auto relu = builder.in_place("relu", kRelu, sum);
  auto graph = builder.finish({reader, relu});

  functionalize_aliases(graph);

  EXPECT_EQ(producer(graph, reader).inputs[0].arg.as_tensor().id, transposed);
  EXPECT_EQ(producer(graph, relu).target, "torch.ops.aten.relu.default");
  expect_functional(graph);
}

TEST(AliasesTest, RejectsMutationThroughLiveViewWithDifferentIndexing) {
  for (const std::string scenario : {"reshape", "transpose"}) {
    SCOPED_TRACE(scenario);
    GraphBuilder builder;
    const auto x = builder.input("x", {2, 2});
    const auto sum = builder.add("sum", x, x);
    const auto alias = scenario == "reshape" ? builder.view("alias", sum, {4})
                                             : builder.transpose("alias", sum);
    builder.in_place("relu", kRelu, sum);
    const auto reader = builder.add("reader", alias, alias);
    expect_failure(
        builder.finish({reader}),
        "mutation through a live view is not supported");
  }
}

TEST(AliasesTest, RejectsMutationOfNonIntermediateStorage) {
  for (const auto role :
       {ValueRole::UserInput,
        ValueRole::Parameter,
        ValueRole::Buffer,
        ValueRole::ConstantTensor}) {
    for (const bool through_view : {false, true}) {
      SCOPED_TRACE(static_cast<int>(role));
      SCOPED_TRACE(through_view);
      GraphBuilder builder;
      const auto x = builder.input("x", {4}, role);
      const auto target = through_view ? builder.view("flat", x, {4}) : x;
      const auto relu = builder.in_place("relu", kRelu, target);
      expect_failure(builder.finish({relu}), "unsupported Native mutation");
    }
  }
}

TEST(AliasesTest, RejectsUnsupportedMutations) {
  {
    GraphBuilder builder;
    const auto x = builder.input("x", {4});
    const auto sum = builder.add("sum", x, x);
    const auto add = builder.call(
        "add_",
        "torch.ops.aten.add_.Tensor",
        {tensor(sum, true), tensor(x)},
        {4},
        sum);
    expect_failure(builder.finish({add}), "unsupported Native mutation");
  }
  {
    GraphBuilder builder;
    const auto x = builder.input("x", {4});
    const auto sum = builder.add("sum", x, x);
    const auto other = builder.add("other", x, x);
    const auto relu = builder.call(
        "relu", kRelu, {tensor(sum, true), tensor(other, true)}, {4}, sum);
    expect_failure(builder.finish({relu}), "unsupported Native mutation");
  }
  {
    GraphBuilder builder;
    const auto x = builder.input("x", {4});
    const auto sum = builder.add("sum", x, x);
    const auto relu =
        builder.call("relu", kRelu, {tensor(sum, true)}, {2, 2}, sum);
    expect_failure(builder.finish({relu}), "unsupported Native mutation");
  }
}

TEST(AliasesTest, RejectsUnsupportedAliases) {
  {
    GraphBuilder builder;
    const auto x = builder.input("x", {4});
    const auto expanded = builder.call(
        "expanded",
        "torch.ops.aten.expand.default",
        {tensor(x), ints({2, 4})},
        {2, 4},
        x);
    expect_failure(builder.finish({expanded}), "unsupported Native view");
  }
  {
    GraphBuilder builder;
    const auto x = builder.input("x", {4});
    const auto y = builder.input("y", {4});
    const auto flat =
        builder.call("flat", kView, {tensor(x), ints({4})}, {4}, y);
    expect_failure(
        builder.finish({flat}), "Native alias does not match its operand");
  }
  {
    GraphBuilder builder;
    const auto x = builder.input("x", {4});
    const auto alias = builder.call(
        "alias", kView, {{.arg = IntArg{.value = 4}}, tensor(x)}, {4}, x);
    expect_failure(
        builder.finish({alias}), "unsupported Native alias or mutation");
  }
}

TEST(AliasesTest, MaterializesAsStridedReshapeOnlyInStorageOrder) {
  struct Case {
    std::vector<int64_t> sizes;
    std::vector<int32_t> dim_order;
    bool accepted;
  };
  for (const auto& [sizes, dim_order, accepted] :
       {Case{{2, 2}, {}, true},
        Case{{2, 2}, {0, 1}, true},
        Case{{2, 2}, {1, 0}, false},
        Case{{1, 4}, {1, 0}, true}}) {
    SCOPED_TRACE(::testing::PrintToString(dim_order));
    GraphBuilder builder;
    const auto x = builder.input("x", sizes);
    builder.graph().value(x).tensor_meta().dim_order_hint = dim_order;
    const auto flat = builder.call(
        "flat",
        kAsStrided,
        {tensor(x), ints({4}), ints({1}), {.arg = NoneArg{}}},
        {4},
        x);
    auto graph = builder.finish({flat});
    if (accepted) {
      functionalize_aliases(graph);
      EXPECT_EQ(
          producer(graph, flat).target,
          "torch.ops.aten.as_strided_copy.default");
      expect_functional(graph);
    } else {
      expect_failure(
          std::move(graph),
          "as_strided requires a contiguous full-span reshape");
    }
  }
}

TEST(AliasesTest, MaterializesAsStridedReshapeOfFunctionalizedMutation) {
  GraphBuilder builder;
  const auto x = builder.input("x", {2, 2});
  const auto sum = builder.add("sum", x, x);
  const auto relu = builder.in_place("relu", kRelu, sum);
  const auto flat = builder.call(
      "flat",
      kAsStrided,
      {tensor(sum), ints({4}), ints({1}), {.arg = IntArg{.value = 0}}},
      {4},
      sum);
  auto graph = builder.finish({flat});

  functionalize_aliases(graph);

  EXPECT_EQ(producer(graph, flat).inputs[0].arg.as_tensor().id, relu);
  EXPECT_EQ(
      producer(graph, flat).target, "torch.ops.aten.as_strided_copy.default");
  expect_functional(graph);
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
