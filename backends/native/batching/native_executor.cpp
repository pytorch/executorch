// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/batching/native_executor.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/OffGraphCache.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/extension/llm/batching/util/session_table.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>

namespace ptn {

namespace batching = executorch::extension::llm::batching;
namespace cache = executorch::extension::llm::cache;
namespace llm = executorch::extension::llm;
using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {
// get_logits_to_keep_mode's value for selected logits.
constexpr int64_t kSelectedLogits = 2;
} // namespace

struct NativeExecutor::State {
  std::string method;
  std::shared_ptr<const Package> package;
  std::shared_ptr<const Program> program;
  std::unique_ptr<EngineHost> host;
  std::shared_ptr<cache::Cache> cache;
  std::unique_ptr<cache::InstallGuard> guard;
  cache::BatchControl* control = nullptr;
  std::unique_ptr<batching::util::SessionTable> sessions;
  // Declared after the host and cache, so they outlive it.
  std::unique_ptr<EngineContext> context;
  std::unique_ptr<EngineExecutable> executable;
  size_t logits_output = 0;
  int max_step_tokens = 0;
  int max_session_tokens = 0;
  int32_t vocab_size = 0;
};

NativeExecutor::NativeExecutor(std::unique_ptr<State> state)
    : state_(std::move(state)) {}

NativeExecutor::~NativeExecutor() = default;

Result<std::unique_ptr<NativeExecutor>> NativeExecutor::create(
    NativeExecutorConfig config) {
  if (!config.make_host || config.max_sessions <= 0 ||
      config.max_session_tokens < 0) {
    ET_LOG(Error, "NativeExecutor: invalid engine or session limits");
    return Error::InvalidArgument;
  }
  try {
    auto state = std::make_unique<State>();
    state->method = config.method;
    state->package =
        std::make_shared<const Package>(Package::load(config.model_path));
    const ByteSpan bytes = state->package->program_bytes();
    state->program = std::make_shared<const Program>(
        Program::load(bytes.data(), bytes.size()));
    const Method& method = state->program->get_method(config.method);
    const auto geometry = offgraph_cache_geometry(method, *state->package);
    if (!geometry || method.graph.input_ids.size() != 3 ||
        state->package->metadata_int(llm::kLogitsToKeepMode) !=
            kSelectedLogits) {
      ET_LOG(
          Error,
          "NativeExecutor: %s needs an off-graph KV cache and selected "
          "logits (export_hf_llm.py --offgraph_cache --logits_to_keep "
          "selected)",
          config.model_path.c_str());
      return Error::InvalidProgram;
    }
    const std::optional<int64_t> context =
        state->package->metadata_int(llm::kMaxContextLen);
    if (!context || *context <= 0 ||
        *context > std::numeric_limits<int32_t>::max() ||
        config.max_session_tokens > *context) {
      ET_LOG(
          Error,
          "NativeExecutor: the session token limit must fit the package's "
          "context length");
      return Error::InvalidArgument;
    }
    state->max_session_tokens = config.max_session_tokens > 0
        ? config.max_session_tokens
        : static_cast<int>(*context);
    if (config.max_sessions >
        std::numeric_limits<int>::max() / state->max_session_tokens) {
      ET_LOG(Error, "NativeExecutor: the cell pool exceeds int range");
      return Error::InvalidArgument;
    }
    state->max_step_tokens = max_step_tokens(method);
    for (size_t i = 0; i < method.output_specs.size(); ++i) {
      if (method.output_specs[i].kind == OutputKind::UserOutput) {
        state->logits_output = i;
        const auto& sizes =
            method.graph.value(method.graph.output_ids[i]).tensor_meta().sizes;
        state->vocab_size =
            sizes.empty() ? 0 : static_cast<int32_t>(sizes.back());
        break;
      }
    }
    if (state->vocab_size <= 0) {
      ET_LOG(Error, "NativeExecutor: the method returns no logits");
      return Error::InvalidProgram;
    }

    // Every resident session may fill its context, so the pool holds them
    // all; storage grows with the cells in use.
    auto built = cache::CacheFactory::global().build(
        config.cache_backend_id,
        cache::kind::kBatched,
        *geometry,
        cache::CacheConfig{
            .capacity = config.max_sessions * state->max_session_tokens,
            .kv_dtype = static_cast<int>(config.kv_dtype)});
    if (!built.ok()) {
      ET_LOG(
          Error,
          "NativeExecutor: no batched cache for backend %s",
          config.cache_backend_id.c_str());
      return built.error();
    }
    state->cache = built.get();
    state->control = state->cache->as<cache::BatchControl>();
    if (state->control == nullptr) {
      return Error::InvalidType;
    }
    const std::optional<int> seq_limit = state->control->max_seqs();
    if (seq_limit && config.max_sessions > *seq_limit) {
      ET_LOG(
          Error,
          "NativeExecutor: %d resident sessions requested, but the cache "
          "holds %d",
          config.max_sessions,
          *seq_limit);
      return Error::InvalidArgument;
    }
    state->sessions = std::make_unique<batching::util::SessionTable>(
        *state->control,
        config.max_sessions,
        state->max_session_tokens,
        state->vocab_size);

    state->guard = std::make_unique<cache::InstallGuard>(state->cache);
    executorch::runtime::BackendOptions<1> options;
    const char* key = nullptr;
    if (state->guard->set_option(options) != Error::Ok ||
        options.get_option(cache::kCacheKeyOption, key) != Error::Ok) {
      return Error::Internal;
    }
    state->host = config.make_host();
    state->context = state->host->create_context_with_options(
        state->program, state->package, {{cache::kCacheKeyOption, key}});
    return std::unique_ptr<NativeExecutor>(
        new NativeExecutor(std::move(state)));
  } catch (const std::exception& error) {
    ET_LOG(Error, "NativeExecutor: creation failed: %s", error.what());
    return Error::InvalidProgram;
  }
}

int32_t NativeExecutor::vocab_size() const {
  return state_->vocab_size;
}

int NativeExecutor::max_session_tokens() const {
  return state_->max_session_tokens;
}

bool NativeExecutor::initialize() {
  try {
    state_->executable = state_->context->compile(state_->method);
    return true;
  } catch (const std::exception& error) {
    ET_LOG(Error, "NativeExecutor: compilation failed: %s", error.what());
    return false;
  }
}

size_t NativeExecutor::preferred_batch_tokens() const {
  return static_cast<size_t>(state_->max_step_tokens);
}

std::optional<batching::SessionId> NativeExecutor::open_session() {
  return state_->sessions->open();
}

void NativeExecutor::close_session(batching::SessionId session) {
  state_->sessions->close(session);
}

std::optional<batching::SessionId> NativeExecutor::clone(
    batching::SessionId source,
    batching::Position upto) {
  return state_->sessions->clone(source, upto);
}

void NativeExecutor::set_sampling(
    batching::SessionId session,
    const batching::SamplingParams& params,
    std::optional<uint64_t> seed) {
  state_->sessions->set_sampling(session, params, seed);
}

bool NativeExecutor::execute(
    const batching::BatchInput& batch,
    batching::BatchOutput& out) {
  out.outputs.clear();
  out.outputs.resize(batch.inputs.size());
  try {
    const auto step = state_->sessions->pack(batch);
    if (!step.ok()) {
      return false;
    }
    EngineExecutable& executable = *state_->executable;
    // A batch wider than the method runs as several forwards, in order, so a
    // slice attends the cells its predecessors wrote.
    const int total = static_cast<int>(step->tokens.size());
    std::vector<float> logits;
    for (int off = 0; off < total; off += state_->max_step_tokens) {
      const int n = std::min(state_->max_step_tokens, total - off);
      if (!state_->control->declare_step(std::vector<int32_t>(
              step->seq_ids.begin() + off, step->seq_ids.begin() + off + n))) {
        ET_LOG(Error, "NativeExecutor: the cache refused %d tokens", n);
        return false;
      }
      const batching::util::SliceRows rows =
          batching::util::select_rows(*step, off, n);
      const auto k = static_cast<int64_t>(rows.selector.size());
      executable.resize_input(0, {1, n});
      executable.set_input(0, step->tokens.data() + off, n, ScalarType::Long);
      executable.resize_input(1, {n});
      executable.set_input(
          1, step->positions.data() + off, n, ScalarType::Long);
      executable.resize_input(2, {k});
      executable.set_input(
          2, rows.selector.data(), rows.selector.size(), ScalarType::Long);
      executable.execute();
      if (rows.inputs.empty()) {
        continue;
      }
      logits.resize(static_cast<size_t>(k * state_->vocab_size));
      executable.get_output(
          state_->logits_output,
          logits.data(),
          logits.size(),
          ScalarType::Float);
      // Non-const: the sampler reduces each row in place.
      auto selected = executorch::extension::make_tensor_ptr(
          {static_cast<int32_t>(k), state_->vocab_size}, logits);
      for (size_t row = 0; row < rows.inputs.size(); ++row) {
        const size_t input = rows.inputs[row];
        const batching::SessionId session = batch.inputs[input].sid;
        const auto token =
            state_->sessions->sample(session, *selected, static_cast<int>(row));
        if (!token) {
          return false;
        }
        out.outputs[input] = batching::Output{session, {*token}};
      }
    }
    return true;
  } catch (const std::exception& error) {
    ET_LOG(Error, "NativeExecutor: execution failed: %s", error.what());
    return false;
  }
}

} // namespace ptn
