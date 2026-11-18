// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <functional>
#include <memory>

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn {

// Greedy generation from a decoder-only LM PTN taking (input_ids [1, T],
// positions): the start position [1] of ExecuTorch Llama exports or one cache
// position per token [T] of HuggingFace exports. Only Llama 3 TikToken
// tokenizers are supported, and the default EOS IDs are Llama 3's. Reads the
// runner's command line flags, which the caller has parsed, and creates the
// host of the one engine the binary links. Returns the process exit status.
int run_llm(const std::function<std::unique_ptr<EngineHost>()>& make_host);

} // namespace ptn
