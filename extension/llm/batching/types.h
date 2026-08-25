/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The vocabulary shared by the runner, the scheduler, and the executor.
//
// An Input is one slice of work for one session, either a decode token or one
// chunk of a prompt, never a whole generation. A Task is an Input plus the
// scheduling identity used to order and cancel it.

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

using Token = std::int64_t;
using SessionId = std::int64_t;
using Position = std::int32_t;
using TaskId = std::int32_t;

struct SamplingParams { // per step, not per generation
  float temperature = 0.0f;
  float top_p = 1.0f;
  std::int32_t top_k = 0;
};

struct Input {
  SessionId sid;
  bool produce_output;

  // The selected slice is tokens[offset : offset + size]. It starts at the
  // absolute logical position `position + offset`; `position` is the base of
  // the complete backing vector, not of the slice.
  size_t offset;
  size_t size;

  std::shared_ptr<const std::vector<Token>> tokens;
  Position position;

  SamplingParams sampling_params;
};

struct Output {
  SessionId sid;
  std::shared_ptr<const std::vector<Token>> tokens;
  struct Continuation {
    std::shared_ptr<const std::vector<Token>> tokens;
    Position position;
  };
  std::optional<Continuation> next;
};

struct Task {
  TaskId tid;
  bool cancelled;
  Input input;
  bool is_decode;
};

struct BatchInput {
  std::vector<Input> inputs;
  size_t size() const {
    size_t sz = 0;
    for (const auto& i : inputs) {
      sz += i.size;
    }
    return sz;
  }
};

// The executor's view of a batch: the Inputs, without the tid, cancelled flag,
// and is_decode that only the runner and scheduler use.
//
// Moves each Input out of its Task, preserving task order, so outputs[i]
// answers batch.inputs[i].
inline BatchInput to_batch_input(std::vector<Task>& tasks) {
  BatchInput batch;
  batch.inputs.reserve(tasks.size());
  for (Task& t : tasks) {
    batch.inputs.push_back(std::move(t.input));
  }
  return batch;
}

struct BatchOutput {
  std::vector<std::optional<Output>> outputs;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
