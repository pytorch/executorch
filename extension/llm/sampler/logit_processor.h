/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * In-place logit transform applied between the model forward pass and the
 * sampler. Examples: grammar masks, logit bias, repetition penalty.
 *
 * `TextTokenGenerator` runs registered processors in order; each sees
 * prior processors' edits. Called once per decoded token — keep it cheap.
 *
 * Tensor contract:
 *   rank 2 [batch, vocab]      — operate on the full last dim
 *   rank 3 [batch, seq, vocab] — operate on the LAST sequence position
 *   other ranks                 — undefined behavior
 *
 * Implementations dispatch their own dtype (the chain runner neither casts
 * nor copies the tensor). Return non-Ok to abort the chain.
 */
class ET_EXPERIMENTAL LogitProcessor {
 public:
  virtual ~LogitProcessor() = default;

  virtual ::executorch::runtime::Error process(
      ::executorch::aten::Tensor logits) = 0;
};

} // namespace llm
} // namespace extension
} // namespace executorch
