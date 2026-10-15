// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>

#include <executorch/backends/native/runtime/Method.h>

namespace ptn::vulkan {

// Rewrites q4 embeddings into et_vk.embedding_q4gsw: either
// quantized_decomposed.embedding_4bit, or aten.embedding over a constant
// carrying AffineGroupQuant read directly.
size_t fuse_quantized_embeddings(Method& method);

} // namespace ptn::vulkan
