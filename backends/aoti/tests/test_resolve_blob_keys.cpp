/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/aoti_delegate_handle.h>

#include <gtest/gtest.h>
#include <string>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/freeable_buffer.h>

using executorch::backends::aoti::resolve_blob_keys;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;

TEST(ResolveBlobKeysTest, ParsesKeysFromPayload) {
  const std::string payload = "aaa_so_blob\nbbb_weights_blob";
  FreeableBuffer processed(payload.data(), payload.size(), nullptr);
  std::string so_key;
  std::string weights_key;

  ASSERT_EQ(
      resolve_blob_keys(&processed, "forward", so_key, weights_key), Error::Ok);
  EXPECT_EQ(so_key, "aaa_so_blob");
  EXPECT_EQ(weights_key, "bbb_weights_blob");
}

TEST(ResolveBlobKeysTest, FallsBackToMethodNameKeysWhenEmpty) {
  FreeableBuffer processed; // size 0: a pre-this-change artifact
  std::string so_key;
  std::string weights_key;

  ASSERT_EQ(
      resolve_blob_keys(&processed, "forward", so_key, weights_key), Error::Ok);
  EXPECT_EQ(so_key, "forward_so_blob");
  EXPECT_EQ(weights_key, "forward_weights_blob");
}

TEST(ResolveBlobKeysTest, FailsOnMalformedPayload) {
  const std::string payload = "missing_the_newline_separator";
  FreeableBuffer processed(payload.data(), payload.size(), nullptr);
  std::string so_key;
  std::string weights_key;

  EXPECT_EQ(
      resolve_blob_keys(&processed, "forward", so_key, weights_key),
      Error::Internal);
}
