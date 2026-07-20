/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/cuda/caller_stream.h>

namespace executorch::extension::cuda {

namespace {
thread_local std::optional<cudaStream_t> caller_stream_;
} // namespace

std::optional<cudaStream_t> getCallerStream() {
  return caller_stream_;
}

CallerStreamGuard::CallerStreamGuard(cudaStream_t stream)
    : previous_(caller_stream_) {
  caller_stream_ = stream;
}

CallerStreamGuard::~CallerStreamGuard() {
  caller_stream_ = previous_;
}

} // namespace executorch::extension::cuda
