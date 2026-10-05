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
// chunk of a prompt, never a whole generation. Tasks distinguish context-free
// preparation from session execution and carry scheduling/cancellation
// identity.

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

using Token = std::uint64_t;
using SessionId = std::int64_t;
using Position = std::int32_t;
// Wide enough that a monotonically issued id cannot wrap in any realistic
// lifetime, so ids never have to be recycled.
using TaskId = std::int64_t;

// Tokenized text and CPU-preprocessed images in prompt order. Source text and
// encoded images must be resolved before preparation.
struct ET_EXPERIMENTAL PreparationInput {
  std::vector<MultimodalInput> segments;
};
using CancellationToken = std::shared_ptr<const std::atomic<bool>>;

// Immutable executor-private owned backing, executable over every valid range.
// retained_bytes() accounts for all owned backing, not merely a borrowing view.
// Metadata queries must be thread-safe and independent of Executor lifetime;
// admission may inspect them on the caller's thread without model/device work.
// Tags identify payload contracts by stable addresses, without requiring RTTI.
class ET_EXPERIMENTAL PreparedInput {
 public:
  virtual ~PreparedInput() = default;
  virtual std::size_t position_count() const = 0;
  virtual std::size_t retained_bytes() const = 0;
  virtual const void* compatibility_tag() const = 0;
};
using PreparedInputPtr = std::shared_ptr<const PreparedInput>;

// Optional text implementation; generic scheduling never depends on this type.
class ET_EXPERIMENTAL TokenPreparedInput final : public PreparedInput {
 public:
  explicit TokenPreparedInput(std::shared_ptr<const std::vector<Token>> tokens)
      : tokens_(std::move(tokens)) {}
  std::size_t position_count() const override {
    return tokens_ ? tokens_->size() : 0;
  }
  std::size_t retained_bytes() const override {
    return sizeof(*this) + sizeof(std::vector<Token>) +
        (tokens_ ? tokens_->capacity() * sizeof(Token) : 0);
  }
  static const void* tag() {
    static const char identity = 0;
    return &identity;
  }
  const void* compatibility_tag() const override {
    return tag();
  }
  const std::vector<Token>& tokens() const {
    return *tokens_;
  }

 private:
  std::shared_ptr<const std::vector<Token>> tokens_;
};

// Sampling policy for a generation. Installed on the session before its tasks
// are submitted, so it does not ride on every Input.
struct ET_EXPERIMENTAL SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  std::int32_t top_k = 0;
};

struct ET_EXPERIMENTAL Input {
  SessionId sid;
  bool produce_output;

  // The selected slice is prepared[offset : offset + size]. It starts at the
  // absolute logical position `position + offset`; `position` is the base of
  // the complete backing, not of the slice. The base may be negative.
  size_t offset;
  size_t size;

  PreparedInputPtr prepared;
  Position position;
};

struct ET_EXPERIMENTAL Output {
  SessionId sid;

  // What this input produced: usually one token, can be more than one for
  // speculative decoding
  std::vector<Token> tokens;
};

struct ET_EXPERIMENTAL ExecutionTask {
  TaskId tid;
  bool cancelled;
  Input input;
  bool is_decode;
};

struct ET_EXPERIMENTAL PrepareTask {
  TaskId tid;
  std::shared_ptr<const PreparationInput> input;
};

using Task = std::variant<PrepareTask, ExecutionTask>;
struct ET_EXPERIMENTAL ExecutionBatch {
  std::vector<ExecutionTask> tasks;
};
using Work = std::variant<PrepareTask, ExecutionBatch>;

struct ET_EXPERIMENTAL BatchInput {
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
ET_EXPERIMENTAL inline BatchInput to_batch_input(
    std::vector<ExecutionTask>& tasks) {
  BatchInput batch;
  batch.inputs.reserve(tasks.size());
  for (ExecutionTask& t : tasks) {
    batch.inputs.push_back(std::move(t.input));
  }
  return batch;
}

struct ET_EXPERIMENTAL BatchOutput {
  std::vector<std::optional<Output>> outputs;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
