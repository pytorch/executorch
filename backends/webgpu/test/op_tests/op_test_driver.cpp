/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic, manifest-driven WebGPU op-test driver: registers one gtest
// per manifest case, runs forward() on the device, and compares vs golden.

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/test/op_tests/driver_util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace executorch::backends::webgpu {

using executorch::extension::make_tensor_ptr;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

class OpCase : public ::testing::Test {
 public:
  explicit OpCase(ManifestEntry entry) : e_(std::move(entry)) {}

  void TestBody() override {
    // required/heavy gate: a heavy case whose artifacts weren't exported is a
    // SKIP; a missing required .pte is a FAIL (never a silent skip).
    {
      std::ifstream pf(e_.pte);
      if (!pf.good()) {
        const bool heavy_enabled = std::getenv("WEBGPU_TEST_HEAVY") != nullptr;
        if (e_.heavy && !heavy_enabled) {
          GTEST_SKIP() << "heavy case, artifacts not exported: " << e_.pte;
        }
        if (e_.required) {
          FAIL() << "required .pte missing: " << e_.pte;
        }
        GTEST_SKIP() << "optional .pte missing: " << e_.pte;
      }
    }
    Module module(e_.pte);
    ASSERT_EQ(module.load_forward(), Error::Ok)
        << "load_forward failed: " << e_.pte;

    std::vector<TensorPtr> tensors;
    for (const auto& in : e_.inputs) {
      const size_t n = numel(in.shape);
      std::vector<executorch::aten::SizesType> sizes(
          in.shape.begin(), in.shape.end());
      if (in.dtype == "int32") {
        auto data = load_int32_bin(in.path, n);
        ASSERT_FALSE(data.empty()) << "missing/short input: " << in.path;
        tensors.push_back(make_tensor_ptr(std::move(sizes), std::move(data)));
      } else {
        auto data = load_fp32_bin(in.path, n);
        ASSERT_FALSE(data.empty()) << "missing/short input: " << in.path;
        tensors.push_back(make_tensor_ptr(std::move(sizes), std::move(data)));
      }
    }
    std::vector<EValue> inputs;
    for (auto& t : tensors) {
      inputs.emplace_back(t);
    }

    auto result = module.forward(inputs);
    ASSERT_TRUE(result.ok()) << "forward failed: error " << (int)result.error();
    const auto& outs = result.get();
    ASSERT_GT(static_cast<int>(outs.size()), e_.golden.output_index);
    const auto& out_tensor = outs[e_.golden.output_index].toTensor();

    const size_t gn = numel(e_.golden.shape);
    ASSERT_EQ(static_cast<size_t>(out_tensor.numel()), gn)
        << "output numel mismatch";
    // Shape check (not just numel): a wrong-shape output that shares numel
    // (plausible for permute/view/squeeze) must still FAIL.
    std::vector<int> out_shape(
        out_tensor.sizes().begin(), out_tensor.sizes().end());
    EXPECT_EQ(out_shape, e_.golden.shape)
        << "output shape != golden shape (numel matched but dims differ)";
    auto golden = load_fp32_bin(e_.golden.path, gn);
    ASSERT_FALSE(golden.empty()) << "missing/short golden: " << e_.golden.path;

    float max_abs = 0.0f, max_rel = 0.0f;
    EXPECT_TRUE(within_tol(
        out_tensor.const_data_ptr<float>(),
        golden.data(),
        static_cast<int>(gn),
        e_.atol,
        e_.rtol,
        &max_abs,
        &max_rel))
        << "max_abs=" << max_abs << " max_rel=" << max_rel;
  }

 private:
  ManifestEntry e_;
};

// Reconciliation: a manifest with zero entries means nothing ran — FAIL, don't
// pass vacuously.
class ReconcileTest : public ::testing::Test {
 public:
  explicit ReconcileTest(size_t n) : n_(n) {}

  void TestBody() override {
    EXPECT_GT(n_, 0u) << "manifest had no entries — nothing ran";
  }

 private:
  size_t n_;
};

} // namespace executorch::backends::webgpu

namespace {
std::string parse_manifest_arg(int argc, char** argv) {
  const std::string flag = "--manifest";
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    if (a.rfind(flag + "=", 0) == 0) {
      return a.substr(flag.size() + 1);
    }
    if (a == flag && i + 1 < argc) {
      return argv[i + 1];
    }
  }
  return {};
}
} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const std::string manifest = parse_manifest_arg(argc, argv);
  if (manifest.empty()) {
    std::fprintf(stderr, "usage: webgpu_op_test --manifest <manifest.json>\n");
    return 2;
  }

  using namespace executorch::backends::webgpu;
  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& ex) {
    std::printf("SKIP: no WebGPU device (%s)\n", ex.what());
    return 0;
  }
  set_default_webgpu_context(&ctx);

  auto entries = parse_manifest(manifest);
  for (auto& entry : entries) {
    const std::string suite = "OpTest_" + entry.op;
    const std::string name = entry.name;
    ::testing::RegisterTest(
        suite.c_str(),
        name.c_str(),
        nullptr,
        nullptr,
        __FILE__,
        __LINE__,
        [entry = std::move(entry)]() -> ::testing::Test* {
          return new OpCase(entry);
        });
  }
  const size_t n_entries = entries.size();
  ::testing::RegisterTest(
      "OpTest_manifest",
      "reconciliation",
      nullptr,
      nullptr,
      __FILE__,
      __LINE__,
      [n_entries]() -> ::testing::Test* {
        return new ReconcileTest(n_entries);
      });

  const int rc = RUN_ALL_TESTS();
  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);
  return rc;
}
