/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Device-free unit tests for the manifest/golden/tolerance helpers.

#include <executorch/backends/webgpu/test/op_tests/driver_util.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace executorch::backends::webgpu;

TEST(DriverUtil, WithinTolAbsOrRel) {
  // elem0: abs ok; elem1: abs fails AND rel fails -> overall fail.
  const float out[2] = {1.0f, 1.0f};
  const float gold[2] = {1.00005f, 2.0f};
  float ma = 0, mr = 0;
  EXPECT_FALSE(within_tol(out, gold, 2, 1e-4f, 1e-3f, &ma, &mr));

  // large magnitude: abs fails but rel passes -> overall pass.
  const float out2[1] = {1000.0f};
  const float gold2[1] = {1000.5f};
  EXPECT_TRUE(within_tol(out2, gold2, 1, 1e-4f, 1e-3f, &ma, &mr));

  // near-zero golden: rel floored, abs governs -> pass within abs.
  const float out3[1] = {1e-5f};
  const float gold3[1] = {0.0f};
  EXPECT_TRUE(within_tol(out3, gold3, 1, 1e-4f, 1e-3f, &ma, &mr));
}

TEST(DriverUtil, ParseManifestResolvesRelativePaths) {
  const std::string dir = std::string(::testing::TempDir()) + "/wgpu_optest_ut";
  std::filesystem::create_directories(dir);
  const std::string mpath = dir + "/manifest.json";
  std::ofstream(mpath) << R"([
    {"op":"add","case":"c0","pte":"c0.pte",
     "inputs":[{"path":"c0.in0.bin","shape":[2,3],"dtype":"float32"}],
     "golden":{"path":"c0.golden.bin","shape":[2,3],"dtype":"float32","output_index":0},
     "atol":0.001,"rtol":0.001}
  ])";

  auto entries = parse_manifest(mpath);
  ASSERT_EQ(entries.size(), 1u);
  const auto& e = entries[0];
  EXPECT_EQ(e.op, "add");
  EXPECT_EQ(e.name, "c0");
  EXPECT_EQ(e.pte, dir + "/c0.pte"); // resolved against manifest dir
  ASSERT_EQ(e.inputs.size(), 1u);
  EXPECT_EQ(e.inputs[0].path, dir + "/c0.in0.bin");
  EXPECT_EQ(numel(e.inputs[0].shape), 6u);
  EXPECT_EQ(e.golden.output_index, 0);
  EXPECT_FLOAT_EQ(e.atol, 0.001f);
  // required defaults True, heavy defaults False when the keys are absent.
  EXPECT_TRUE(e.required);
  EXPECT_FALSE(e.heavy);
}

TEST(DriverUtil, ParseManifestRequiredHeavyExplicit) {
  const std::string dir =
      std::string(::testing::TempDir()) + "/wgpu_optest_ut2";
  std::filesystem::create_directories(dir);
  const std::string mpath = dir + "/manifest.json";
  std::ofstream(mpath) << R"([
    {"op":"b","case":"heavy","pte":"b.pte","inputs":[],
     "golden":{"path":"b.g","shape":[1],"output_index":0},
     "atol":0.001,"rtol":0.001,"required":false,"heavy":true}
  ])";
  auto entries = parse_manifest(mpath);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_FALSE(entries[0].required);
  EXPECT_TRUE(entries[0].heavy);
}
