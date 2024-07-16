/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/exir/backend/test/demos/rpc/ExecutorBackend.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <gtest/gtest.h>
#include <cstring>
using torch::executor::ArrayRef;
using torch::executor::Error;
using torch::executor::PyTorchBackendInterface;
using torch::executor::RuntimeInfo;
using torch::executor::SizedBuffer;

TEST(BackendCompatibility, GetRuntimeInfo) {
  torch::executor::registerExecutorBackend();
  PyTorchBackendInterface* executor_backend =
      torch::executor::get_backend_class("ExecutorBackend");

  const char* kKey = "version";
  const char* kET00 = "ET_00";
  const char* kET12 = "ET_12";

  std::array<char, 6> buffer;
  ASSERT_EQ(buffer.size(), strlen(kET00) + 1);
  size_t nbytes = strlen(kET00) + 1;
  memcpy(buffer.data(), kET00, nbytes);
  // Check the default value is ET_00
  EXPECT_STREQ(reinterpret_cast<const char*>(buffer.data()), kET00);
  SizedBuffer runtime_info_value = {buffer.data(), nbytes};
  RuntimeInfo runtime_info = {kKey, runtime_info_value};

  ArrayRef<RuntimeInfo> runtime_info_list(runtime_info);
  Error err = executor_backend->runtime_info(runtime_info_list);
  EXPECT_EQ(err, Error::Ok);
  // Check the runtime version is updated to ET_12
  EXPECT_STREQ(
      reinterpret_cast<const char*>(runtime_info_list.at(0).value.buffer),
      kET12);
}
