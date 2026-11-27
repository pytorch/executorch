// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// A gtest main that runs the bundled samples of .ptn packages on the engine the
// binary links (see PtnRunner.h): one test per package and method, grouping
// samples by their first call. Each sample runs from a fresh context, its calls
// in order on one executable per method, resizing dynamic inputs between them.
// Each method output is judged over all its calls by OutputCheck.
//
// Usage:
//   <test> --packages=a.ptn,dir [--rmse_ratio --max_ratio --rtol --sigmas]
//          [--allow_activation_quant_for_weight_only]
//          [--gpu_fp16_block_scales] [engine host flags] [gtest flags]

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/Samples.h>
#include <executorch/backends/native/runtime/deserialize/ZipReader.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/test/runtime/PtnRunner.h>
#include <executorch/backends/native/test/samples/OutputCheck.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

DEFINE_string(
    packages,
    "",
    "Comma-separated .ptn files or directories of them.");
DEFINE_double(
    rmse_ratio,
    ptn::testing::Policy{}.rmse_ratio,
    "Allowed engine RMSE and biases, in multiples of eager's.");
DEFINE_double(
    max_ratio,
    ptn::testing::Policy{}.max_ratio,
    "Allowed engine max abs error, in multiples of eager's.");
DEFINE_double(
    rtol,
    ptn::testing::Policy{}.rtol,
    "Error allowed on top, relative to the precise outputs' magnitude.");
DEFINE_double(
    sigmas,
    ptn::testing::Policy{}.sigmas,
    "Standard errors the fitted biases may deviate by.");
DEFINE_bool(
    allow_activation_quant_for_weight_only,
    false,
    "Let the engine quantize the activations of weight-only quantized linears, "
    "judging it against golden-int8_act samples where a package has them.");
DEFINE_bool(
    gpu_fp16_block_scales,
    false,
    "Allow, in outputs that Q4_1 or Q5_1 linears compute from the inputs, "
    "directly or chained, the error of ggml's GPU fp16 block scale products "
    "(see block_scale_linear).");

namespace ptn::testing {
namespace {

struct Bundle {
  std::shared_ptr<const Package> package;
  std::shared_ptr<const Program> program;
  std::string golden_set;
  SampleSet golden;
  SampleSet precise;
};

// Created on first use, so listing tests touches no device.
EngineHost& host() {
  static const std::unique_ptr<EngineHost> instance = [] {
    std::unique_ptr<EngineHost> made = make_engine_host(
        {.allow_activation_quant_for_weight_only =
             FLAGS_allow_activation_quant_for_weight_only});
    std::printf(
        "engine %s on %s\n", made->name().c_str(), made->device_name().c_str());
    return made;
  }();
  return *instance;
}

// Keyed by method and output index.
using Checks = std::map<std::pair<std::string, size_t>, OutputCheck>;

// A value that a method computes from its inputs through Q4_1 and Q5_1 linears
// and views alone, with a bound on its error from the GPU's fp16 products.
// Empty `sizes` when a view's are not known.
struct Traced {
  std::vector<double> values;
  std::vector<double> bounds;
  std::vector<int64_t> sizes;
};

// Whether ggml's GPU may take its fp16-scale kernels for `x`: up to 8 rows in
// the second-to-last dim, or in all, where a view's sizes are not known.
bool vector_kernel(const Traced& x, int64_t in) {
  constexpr int64_t kMaxRows = 8;
  const size_t dims = x.sizes.size();
  return dims < 2 || x.sizes[dims - 2] <= kMaxRows ||
      static_cast<int64_t>(x.values.size()) / in <= kMaxRows;
}

std::string_view op_name(std::string_view target) {
  constexpr std::string_view kPrefix = "torch.ops.";
  return target.starts_with(kPrefix) ? target.substr(kPrefix.size()) : target;
}

const Argument* argument(const Node& node, std::string_view name, size_t at) {
  const auto named = std::find_if(
      node.inputs.begin(), node.inputs.end(), [name](const NamedArgument& arg) {
        return arg.name == name;
      });
  if (named != node.inputs.end()) {
    return &named->arg;
  }
  return at < node.inputs.size() ? &node.inputs[at].arg : nullptr;
}

std::vector<double> to_double(const SampleTensor& t) {
  const std::vector<float> values = to_float(t);
  return {values.begin(), values.end()};
}

bool is_float(ScalarType dtype) {
  return dtype == kFloat || dtype == kHalf || dtype == kBFloat16;
}

class SampleTest : public ::testing::Test {
 public:
  SampleTest(std::shared_ptr<const Bundle> bundle, std::string method)
      : bundle_(std::move(bundle)), method_(std::move(method)) {}

  void TestBody() override {
    const std::vector<Sample>& golden = bundle_->golden.samples;
    const std::vector<Sample>& precise = bundle_->precise.samples;
    ASSERT_EQ(golden.size(), precise.size());
    Checks checks;
    for (size_t s = 0; s < golden.size(); ++s) {
      if (golden[s].calls.front().method != method_) {
        continue;
      }
      SCOPED_TRACE("sample " + std::to_string(s));
      try {
        run(golden[s], precise[s], checks);
      } catch (const std::exception& e) {
        FAIL() << e.what();
      }
      if (HasFatalFailure()) {
        return;
      }
    }
    const Policy policy{
        .rmse_ratio = FLAGS_rmse_ratio,
        .max_ratio = FLAGS_max_ratio,
        .rtol = FLAGS_rtol,
        .sigmas = FLAGS_sigmas};
    for (const auto& [key, check] : checks) {
      const std::string name = key.first + " out " +
          std::to_string(key.second) + " vs " + bundle_->golden_set;
      std::printf("%s: %s\n", name.c_str(), check.summary().c_str());
      for (const std::string& failure : check.failures(policy)) {
        ADD_FAILURE() << name << ": " << failure;
      }
    }
  }

 private:
  void run(const Sample& golden, const Sample& precise, Checks& checks) {
    ASSERT_EQ(golden.calls.size(), precise.calls.size());
    const std::unique_ptr<EngineContext> context =
        host().create_context(bundle_->program, bundle_->package);
    std::map<std::string, std::unique_ptr<EngineExecutable>> executables;
    for (size_t c = 0; c < golden.calls.size(); ++c) {
      const SampleCall& call = golden.calls[c];
      ASSERT_EQ(call.method, precise.calls[c].method);
      ASSERT_TRUE(call.inputs == precise.calls[c].inputs)
          << call.method << " has " << bundle_->golden_set
          << " and precise inputs that differ";
      ASSERT_TRUE(call.outputs && precise.calls[c].outputs)
          << call.method << " has no recorded outputs";
      const std::vector<SampleTensor>& want = *call.outputs;
      const std::vector<SampleTensor>& ref = *precise.calls[c].outputs;
      std::unique_ptr<EngineExecutable>& executable = executables[call.method];
      if (!executable) {
        executable = context->compile(call.method);
      }
      ASSERT_EQ(executable->num_inputs(), call.inputs.size());
      ASSERT_EQ(executable->num_outputs(), want.size());
      ASSERT_EQ(ref.size(), want.size());

      for (size_t i = 0; i < call.inputs.size(); ++i) {
        const SampleTensor& in = call.inputs[i];
        ASSERT_STREQ(
            scalar_type_name(executable->input_dtype(i)),
            scalar_type_name(in.dtype))
            << "input " << i;
        if (executable->input_sizes(i) != in.sizes) {
          executable->resize_input(i, in.sizes);
        }
        executable->set_input(
            i,
            in.bytes.data(),
            in.bytes.size() / element_size(in.dtype),
            in.dtype);
      }
      executable->execute();
      const std::vector<std::vector<double>> allowances =
          FLAGS_gpu_fp16_block_scales ? block_scale_allowances(call)
                                      : std::vector<std::vector<double>>();
      for (size_t o = 0; o < want.size(); ++o) {
        ASSERT_EQ(executable->output_sizes(o), want[o].sizes) << "output " << o;
        ASSERT_EQ(ref[o].sizes, want[o].sizes) << "precise output " << o;
        ASSERT_STREQ(
            scalar_type_name(executable->output_dtype(o)),
            scalar_type_name(want[o].dtype))
            << "output " << o;
        std::vector<float> got(
            want[o].bytes.size() / element_size(want[o].dtype));
        executable->get_output(o, got.data(), got.size(), kFloat);
        const bool allowed =
            o < allowances.size() && allowances[o].size() == got.size();
        checks[{call.method, o}].add(
            got,
            want[o],
            ref[o],
            allowed ? std::span<const double>(allowances[o])
                    : std::span<const double>());
      }
    }
  }

  const std::vector<uint8_t>& constant(const Method& method, ValueId id) {
    const auto binding = std::find_if(
        method.data_bindings.begin(),
        method.data_bindings.end(),
        [id](const DataBinding& b) { return b.value_id == id && b.has_data; });
    if (binding == method.data_bindings.end()) {
      throw std::runtime_error("no data for constant " + std::to_string(id));
    }
    auto [it, added] = constants_.try_emplace(binding->key);
    if (added) {
      const OwnedBytes bytes =
          bundle_->package->acquire_constant(binding->key).value();
      const ByteSpan span = bytes.span();
      it->second.assign(span.data(), span.data() + span.size());
    }
    return it->second;
  }

  // The linear's output if it reads a traced input with Q4_1 or Q5_1 weights.
  std::optional<Traced> trace_linear(
      const Method& method,
      const Node& node,
      const std::map<ValueId, Traced>& traced) {
    const Argument* input = argument(node, "input", 0);
    const Argument* weight = argument(node, "weight", 1);
    const Argument* bias = argument(node, "bias", 2);
    if (input == nullptr || input->kind() != ArgKind::Tensor ||
        !traced.contains(input->as_tensor().id) || weight == nullptr ||
        weight->kind() != ArgKind::Tensor) {
      return std::nullopt;
    }
    const TensorMeta& meta =
        method.graph.value(weight->as_tensor().id).tensor_meta();
    const PackedQuant* packed =
        meta.quant ? std::get_if<PackedQuant>(&*meta.quant) : nullptr;
    if (packed == nullptr ||
        (packed->codec != "gguf:q4_1" && packed->codec != "gguf:q5_1") ||
        meta.sizes.size() != 2) {
      return std::nullopt;
    }
    const Traced& x = traced.at(input->as_tensor().id);
    const bool exact =
        std::ranges::all_of(x.bounds, [](double b) { return b == 0.0; });
    BlockScaleLinear linear = block_scale_linear(
        packed->codec,
        constant(method, weight->as_tensor().id),
        meta.sizes[0],
        meta.sizes[1],
        x.values,
        exact ? std::span<const double>() : std::span<const double>(x.bounds),
        vector_kernel(x, meta.sizes[1]));
    if (bias != nullptr && bias->kind() == ArgKind::Tensor) {
      const ValueId id = bias->as_tensor().id;
      const std::vector<double> b = to_double(SampleTensor{
          .dtype = method.graph.value(id).tensor_meta().dtype,
          .bytes = constant(method, id)});
      for (size_t i = 0; i < linear.outputs.size(); ++i) {
        linear.outputs[i] += b.at(i % b.size());
      }
    }
    std::vector<int64_t> sizes = x.sizes;
    if (!sizes.empty()) {
      sizes.back() = meta.sizes[0];
    }
    return Traced{
        std::move(linear.outputs), std::move(linear.bounds), std::move(sizes)};
  }

  // Per user output, the GPU fp16 block scale error to allow in it, empty for
  // outputs that no traced chain computes.
  std::vector<std::vector<double>> block_scale_allowances(
      const SampleCall& call) {
    const Method& method = bundle_->program->get_method(call.method);
    const Graph& graph = method.graph;
    std::map<ValueId, Traced> traced;
    size_t next = 0;
    for (const ValueId id : graph.input_ids) {
      if (graph.value(id).role != ValueRole::UserInput) {
        continue;
      }
      const SampleTensor& in = call.inputs.at(next++);
      if (is_float(in.dtype)) {
        std::vector<double> values = to_double(in);
        std::vector<double> bounds(values.size());
        traced[id] = {std::move(values), std::move(bounds), in.sizes};
      }
    }
    for (const NodeId node_id : graph.schedule) {
      const Node& node = graph.node(node_id);
      if (!node.is_call() || node.outputs.size() != 1) {
        continue;
      }
      const ValueId out = node.outputs.front().value_id;
      const std::string_view op = op_name(node.target);
      const Argument* self = argument(node, "self", 0);
      if (op == "aten.linear.default") {
        if (std::optional<Traced> y = trace_linear(method, node, traced)) {
          traced[out] = std::move(*y);
        }
      } else if (
          (op == "aten.view_copy.default" || op == "aten.view.default" ||
           op == "aten.reshape.default" || op == "aten._unsafe_view.default") &&
          self != nullptr && self->kind() == ArgKind::Tensor &&
          traced.contains(self->as_tensor().id)) {
        traced[out] = traced.at(self->as_tensor().id);
        traced[out].sizes.clear();
      }
    }
    std::vector<std::vector<double>> allowances;
    for (size_t o = 0; o < graph.output_ids.size(); ++o) {
      if (method.output_specs.at(o).kind != OutputKind::UserOutput) {
        continue;
      }
      const auto it = traced.find(graph.output_ids[o]);
      allowances.push_back(
          it == traced.end() ? std::vector<double>() : it->second.bounds);
    }
    return allowances;
  }

  std::shared_ptr<const Bundle> bundle_;
  std::string method_;
  std::map<std::string, std::vector<uint8_t>> constants_;
};

std::vector<std::filesystem::path> package_paths() {
  std::vector<std::filesystem::path> paths;
  std::stringstream list(FLAGS_packages);
  for (std::string entry; std::getline(list, entry, ',');) {
    if (entry.empty()) {
      continue;
    }
    if (!std::filesystem::is_directory(entry)) {
      paths.emplace_back(entry);
      continue;
    }
    std::vector<std::filesystem::path> found;
    for (const auto& file : std::filesystem::directory_iterator(entry)) {
      if (file.path().extension() == ".ptn") {
        found.push_back(file.path());
      }
    }
    std::ranges::sort(found);
    paths.insert(paths.end(), found.begin(), found.end());
  }
  return paths;
}

void register_package(const std::filesystem::path& path) {
  auto bundle = std::make_shared<Bundle>();
  bundle->package =
      std::make_shared<const Package>(Package::load(path.string()));
  const ByteSpan program = bundle->package->program_bytes();
  bundle->program = std::make_shared<const Program>(
      Program::load(program.data(), program.size()));
  const ZipReader zip = ZipReader::open(path.string());
  // Eager outputs with int8 per-token dynamic activations, which packages with
  // weight-only quantized linears also carry.
  const bool act_quant_golden = FLAGS_allow_activation_quant_for_weight_only &&
      zip.member_size("samples/golden-int8_act/index.json").has_value();
  bundle->golden_set = act_quant_golden ? "golden-int8_act" : "golden";
  bundle->golden = read_sample_set(zip, bundle->golden_set);
  bundle->precise = read_sample_set(zip, "precise");

  std::set<std::string> methods;
  for (const Sample& sample : bundle->golden.samples) {
    methods.insert(sample.calls.front().method);
  }
  const std::string suite = path.stem().string();
  for (const std::string& method : methods) {
    ::testing::RegisterTest(
        suite.c_str(),
        method.c_str(),
        nullptr,
        nullptr,
        __FILE__,
        __LINE__,
        [bundle, method]() { return new SampleTest(bundle, method); });
  }
}

} // namespace
} // namespace ptn::testing

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    const std::vector<std::filesystem::path> paths =
        ptn::testing::package_paths();
    if (paths.empty()) {
      throw std::invalid_argument("--packages names no .ptn");
    }
    for (const std::filesystem::path& path : paths) {
      ptn::testing::register_package(path);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return RUN_ALL_TESTS();
}
