/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/bundled_module.h>

#include <gtest/gtest.h>

#include <executorch/extension/data_loader/file_data_loader.h>

using namespace ::executorch::extension;
using namespace ::executorch::runtime;

class BundledModuleTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    std::string resources_path;
    if (const char* env = std::getenv("RESOURCES_PATH")) {
      resources_path = env;
    }
    bpte_path_ = resources_path + "/bundled_program.bpte";
  }

  static inline std::string bpte_path_;
};

#include <fstream>

std::vector<uint8_t> load_file_or_die(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const size_t nbytes = file.tellg();
  file.seekg(0, std::ios::beg);
  auto file_data = std::vector<uint8_t>(nbytes);
  ET_CHECK_MSG(
      file.read(reinterpret_cast<char*>(file_data.data()), nbytes),
      "Could not load contents of file '%s'",
      path);
  return file_data;
}

TEST_F(BundledModuleTest, TestExecute) {
  std::vector<uint8_t> file_data = load_file_or_die(bpte_path_.c_str());
  BundledModule bundled_module(reinterpret_cast<void*>(file_data.data()));

  auto outputs = bundled_module.execute("forward", 0);
  EXPECT_EQ(outputs.error(), Error::Ok);
  auto status = bundled_module.verify_method_outputs(
      "forward", 0, 1e-3, 1e-5);
  EXPECT_EQ(status, Error::Ok);
}
