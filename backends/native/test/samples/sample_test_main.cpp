// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// A gtest main that runs the bundled samples of .ptn packages on the engine the
// binary links (see PtnRunner.h): one test per package and method, grouping
// samples by their first call. Each sample runs from a fresh context, and each
// method output is judged over all its samples by OutputCheck.
//
// Usage:
//   <test> --packages=a.ptn,dir [--rmse_ratio --max_ratio --rtol --sigmas]
//          [--allow_activation_quant_for_weight_only] [engine host flags]
//          [gtest flags]

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
        checks[{call.method, o}].add(got, want[o], ref[o]);
      }
    }
  }

  std::shared_ptr<const Bundle> bundle_;
  std::string method_;
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
