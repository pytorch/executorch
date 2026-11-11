// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/backends/native/examples/hf_llm/runtime/LlmRunner.h>
#include <executorch/backends/native/examples/hf_llm/runtime/LlmRunnerUtils.h>
#include <executorch/backends/native/examples/runner_utils/runner_utils.h>
#include <executorch/backends/native/runtime/Program.h>
#include <executorch/backends/native/runtime/deserialize/Package.h>
#include <executorch/backends/native/runtime/deserialize/Samples.h>
#include <executorch/backends/native/runtime/deserialize/ZipReader.h>
#include <executorch/backends/native/runtime/engine/Engine.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/runner/constants.h>
#include <executorch/runtime/backend/options.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/error.h>

DEFINE_string(package, "", "Path to the *.ptn model. Required.");
DEFINE_string(tokenizer, "", "Path to the Llama TikToken model. Required.");
DEFINE_string(prompt, "The capital of France is", "Prompt text.");
DEFINE_string(method, "", "Method to run. Defaults to the first one.");
DEFINE_string(eos_ids, "128001,128009", "Comma-separated EOS token IDs.");
DEFINE_int32(max_new_tokens, 32, "Maximum generated token count.");
DEFINE_int32(max_seq_length, 2048, "Maximum prompt plus generated tokens.");
DEFINE_int32(num_bos, 1, "Number of BOS tokens prepended to the prompt.");
DEFINE_bool(
    print_token_ids,
    false,
    "Print the generated token IDs for exact comparisons.");
DEFINE_bool(
    print_top_logits,
    false,
    "Print top-five raw logits for numerical diagnostics.");
DEFINE_string(
    dump_first_logits,
    "",
    "Write the initial FP32 logits row; refuse to overwrite.");
DEFINE_bool(
    warmup_prefill,
    false,
    "Replay the identical prompt once before timing prefill (benchmark only).");
DEFINE_string(
    verify_samples,
    "",
    "Replay this sample set, whose calls must all run --method, instead of "
    "--prompt, comparing each call's logits and forcing the sample's next "
    "token; exits 1 on mismatch.");
DEFINE_int32(
    kv_capacity,
    0,
    "Off-graph KV cache capacity in tokens; 0 uses --max_seq_length. Only "
    "for packages exported with --offgraph_cache.");
DEFINE_string(
    kv_dtype,
    "fp32",
    "Off-graph KV cache storage: fp32 or fp16.");
DEFINE_int32(
    kv_initial_capacity,
    -1,
    "Off-graph KV cache rows allocated up front, doubling toward "
    "--kv_capacity as the sequence grows; -1 keeps the cache default.");
DEFINE_bool(
    interactive,
    false,
    "Chat on stdin with the Llama 3 chat template, keeping the conversation "
    "in the off-graph KV cache. Needs a package exported with "
    "--offgraph_cache.");
namespace {

namespace cache = executorch::extension::llm::cache;

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// Returns the shortest cache's sequence length, or 0 if the method has none.
size_t kv_cache_length(const ptn::Method& method) {
  size_t length = 0;
  for (const ptn::Node& node : method.graph.nodes) {
    for (const ptn::NamedArgument& input : node.inputs) {
      // An absent optional tensor has an invalid id.
      if (input.arg.kind() != ptn::ArgKind::Tensor ||
          !ptn::valid(input.arg.as_tensor().id)) {
        continue;
      }
      const ptn::Value& value = method.graph.value(input.arg.as_tensor().id);
      if (!value.is_tensor()) {
        continue;
      }
      const size_t sequence = ptn::kv_cache_sequence_length(
          node.target, input.name, value.tensor_meta().sizes);
      if (sequence > 0) {
        length = length == 0 ? sequence : std::min(length, sequence);
      }
    }
  }
  return length;
}

constexpr std::string_view kUpdateAndAttend =
    "torch.ops.kvcache.update_and_attend.default";

// The off-graph KV cache a method's update_and_attend calls need: one layer per
// cache id, shaped like its K, with the sliding window (or none) the package's
// metadata gives it. nullopt if the method has none.
std::optional<cache::CacheGeometry> offgraph_cache_geometry(
    const ptn::Method& method,
    const ptn::Package& package) {
  std::map<int64_t, cache::LayerGeometry> layers;
  for (const ptn::Node& node : method.graph.nodes) {
    if (node.target != kUpdateAndAttend) {
      continue;
    }
    const auto arg = [&](std::string_view name,
                         size_t positional) -> const ptn::Argument& {
      for (const ptn::NamedArgument& input : node.inputs) {
        if (input.name == name) {
          return input.arg;
        }
      }
      if (positional >= node.inputs.size()) {
        throw std::runtime_error(
            "update_and_attend is missing argument " + std::string(name));
      }
      return node.inputs[positional].arg;
    };
    const ptn::Argument& layer_arg = arg("layer_id", 4);
    const ptn::Argument& k_arg = arg("k", 1);
    if (layer_arg.kind() != ptn::ArgKind::Int ||
        ptn::valid(layer_arg.as_int().id) ||
        k_arg.kind() != ptn::ArgKind::Tensor) {
      throw std::runtime_error(
          "update_and_attend needs a constant layer_id and a K tensor");
    }
    const std::vector<int64_t>& k_sizes =
        method.graph.value(k_arg.as_tensor().id).tensor_meta().sizes;
    if (k_sizes.size() != 4) {
      throw std::runtime_error("update_and_attend K must be [B, H, T, D]");
    }
    const cache::LayerGeometry layer{
        .n_kv_heads = static_cast<int>(k_sizes[1]),
        .head_dim = static_cast<int>(k_sizes[3])};
    const auto [it, inserted] =
        layers.try_emplace(layer_arg.as_int().value, layer);
    if (!inserted &&
        (it->second.n_kv_heads != layer.n_kv_heads ||
         it->second.head_dim != layer.head_dim)) {
      throw std::runtime_error(
          "update_and_attend calls disagree on cache layer " +
          std::to_string(it->first) + "'s shape");
    }
  }
  if (layers.empty()) {
    return std::nullopt;
  }
  cache::CacheGeometry geometry;
  for (const auto& [id, layer] : layers) {
    if (id != static_cast<int64_t>(geometry.layers.size())) {
      throw std::runtime_error("update_and_attend cache ids must be 0..n-1");
    }
    geometry.layers.push_back(layer);
  }

  namespace llm = executorch::extension::llm;
  const auto per_layer = [&](const char* name) {
    std::optional<std::vector<int64_t>> values = package.metadata_ints(name);
    if (values && values->size() != geometry.layers.size()) {
      throw std::runtime_error(
          std::string("package metadata ") + name + " has " +
          std::to_string(values->size()) + " entries for " +
          std::to_string(geometry.layers.size()) + " cache layers");
    }
    return values;
  };
  const auto kv_heads = per_layer(llm::kKVHeads);
  const auto head_dims = per_layer(llm::kHeadDims);
  const auto windows = per_layer(llm::kWindows);
  // Without them a sliding layer would silently attend its whole history.
  if (!windows) {
    throw std::runtime_error(
        std::string("package metadata has no ") + llm::kWindows +
        "; export the package with export_hf_llm.py --offgraph_cache");
  }
  for (size_t i = 0; i < geometry.layers.size(); ++i) {
    cache::LayerGeometry& layer = geometry.layers[i];
    if ((kv_heads && (*kv_heads)[i] != layer.n_kv_heads) ||
        (head_dims && (*head_dims)[i] != layer.head_dim)) {
      throw std::runtime_error(
          "package metadata disagrees with the graph on cache layer " +
          std::to_string(i) + "'s shape");
    }
    const int64_t window = (*windows)[i];
    if (window < 0 || window > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "package metadata gives cache layer " + std::to_string(i) +
          " an invalid window");
    }
    if (window > 0) {
      layer.policy = {cache::LayerPolicy::Kind::Ring, static_cast<int>(window)};
    }
  }
  return geometry;
}

// A runner-owned off-graph KV cache, published for the engine to bind.
struct OffGraphCache {
  std::shared_ptr<cache::Cache> cache;
  std::unique_ptr<cache::InstallGuard> guard;
  cache::SequenceControl* control = nullptr;
  ptn::EngineOptions options;
};

// `max_write` is the most tokens one call writes, which sizes a ring layer.
OffGraphCache install_offgraph_cache(
    const char* backend_id,
    const cache::CacheGeometry& geometry,
    int max_write) {
  if (backend_id == nullptr) {
    throw std::runtime_error(
        "the package keeps its KV cache off-graph, which this engine's "
        "runner does not support");
  }
  if (FLAGS_kv_dtype != "fp16" && FLAGS_kv_dtype != "fp32") {
    throw std::invalid_argument("--kv_dtype must be fp32 or fp16");
  }
  cache::CacheConfig cfg{
      .capacity =
          FLAGS_kv_capacity > 0 ? FLAGS_kv_capacity : FLAGS_max_seq_length,
      .kv_dtype = static_cast<int>(
          FLAGS_kv_dtype == "fp16" ? ptn::ScalarType::Half
                                   : ptn::ScalarType::Float),
      .max_write = max_write};
  if (FLAGS_kv_initial_capacity >= 0) {
    cfg.initial_capacity = FLAGS_kv_initial_capacity;
  }
  auto built = cache::CacheFactory::global().build(
      backend_id, cache::kind::kSingle, geometry, cfg);
  if (!built.ok()) {
    throw std::runtime_error(
        "failed to build the off-graph KV cache: error " +
        std::to_string(static_cast<int>(built.error())));
  }
  OffGraphCache out;
  out.cache = built.get();
  out.control = out.cache->as<cache::SequenceControl>();
  if (out.control == nullptr) {
    throw std::runtime_error("the KV cache has no single-sequence control");
  }
  out.guard = std::make_unique<cache::InstallGuard>(out.cache);
  executorch::runtime::BackendOptions<1> options;
  const char* key = nullptr;
  if (out.guard->set_option(options) != executorch::runtime::Error::Ok ||
      options.get_option(cache::kCacheKeyOption, key) !=
          executorch::runtime::Error::Ok) {
    throw std::runtime_error("failed to publish the KV cache key");
  }
  out.options.emplace(cache::kCacheKeyOption, key);
  const auto sliding = std::ranges::count_if(
      geometry.layers, [](const cache::LayerGeometry& layer) {
        return layer.policy.kind == cache::LayerPolicy::Kind::Ring;
      });
  std::printf(
      "kv cache:  off-graph, %zu layers (%td sliding), %d tokens (%d "
      "allocated up front), %s\n",
      geometry.layers.size(),
      sliding,
      cfg.capacity,
      std::min(std::max(cfg.initial_capacity, 1), cfg.capacity),
      FLAGS_kv_dtype.c_str());
  return out;
}

// The --kv_* flags size an off-graph cache, so they are mistakes elsewhere.
void reject_offgraph_cache_flags() {
  for (const char* flag : {"kv_capacity", "kv_dtype", "kv_initial_capacity"}) {
    if (!gflags::GetCommandLineFlagInfoOrDie(flag).is_default) {
      throw std::invalid_argument(
          std::string("--") + flag +
          " only applies to packages exported with --offgraph_cache");
    }
  }
}

// The values of an [n] or [1, n] Int or Long sample tensor.
std::vector<int64_t> integer_values(const ptn::SampleTensor& tensor) {
  int64_t n = -1;
  if (tensor.sizes.size() == 1) {
    n = tensor.sizes[0];
  } else if (tensor.sizes.size() == 2 && tensor.sizes[0] == 1) {
    n = tensor.sizes[1];
  }
  if ((tensor.dtype != ptn::kLong && tensor.dtype != ptn::kInt) || n < 0) {
    throw std::runtime_error(
        "sample token and position inputs must be [n] or [1, n] Int or Long");
  }
  std::vector<int64_t> values(static_cast<size_t>(n));
  for (size_t i = 0; i < values.size(); ++i) {
    if (tensor.dtype == ptn::kLong) {
      std::memcpy(&values[i], tensor.bytes.data() + 8 * i, 8);
    } else {
      int32_t value = 0;
      std::memcpy(&value, tensor.bytes.data() + 4 * i, 4);
      values[i] = value;
    }
  }
  return values;
}

// Chooses the token after a logits output whose last dimension is the
// vocabulary.
using NextToken = std::function<
    uint64_t(const std::vector<float>& logits, size_t vocabulary_size)>;

struct Generation {
  double prompt_ms = 0.0;
  double decode_ms = 0.0;
  size_t decode_steps = 0;
  // The last token chosen, which was never fed to the model.
  uint64_t unfed_token = 0;
  bool ended_on_eos = false;
};

// Prefills `prompt_tokens` from `start_position` in calls of at most
// `max_chunk` tokens, then decodes one token per call until `limit` tokens or
// an EOS, streaming the text to stdout. With --warmup_prefill it prefills the
// prompt once beforehand, then clears `offgraph_cache` if there is one.
Generation generate(
    ptn::EngineExecutable& executable,
    size_t logits_output,
    bool per_token_positions,
    size_t max_chunk,
    ::tokenizers::Tiktoken& tokenizer,
    const std::vector<uint64_t>& prompt_tokens,
    int64_t start_position,
    size_t limit,
    const std::unordered_set<uint64_t>& eos_ids,
    const NextToken& next_token_of,
    cache::CacheControl* offgraph_cache) {
  std::vector<float> logits;

  auto set_positions = [&](int64_t start, size_t count) {
    const size_t n = per_token_positions ? count : 1;
    if (per_token_positions) {
      executable.resize_input(1, {static_cast<int64_t>(n)});
    }
    std::vector<int64_t> values(n);
    std::iota(values.begin(), values.end(), start);
    if (executable.input_dtype(1) == ptn::kInt) {
      const std::vector<int32_t> narrowed(values.begin(), values.end());
      executable.set_input(1, narrowed.data(), narrowed.size(), ptn::kInt);
    } else {
      executable.set_input(1, values.data(), values.size(), ptn::kLong);
    }
  };
  auto set_tokens = [&](const std::vector<uint64_t>& tokens) {
    executable.resize_input(0, {1, static_cast<int64_t>(tokens.size())});
    if (executable.input_dtype(0) == ptn::kInt) {
      std::vector<int32_t> narrowed;
      narrowed.reserve(tokens.size());
      for (const uint64_t token : tokens) {
        if (token >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          throw std::runtime_error("token ID is outside Int range");
        }
        narrowed.push_back(static_cast<int32_t>(token));
      }
      executable.set_input(0, narrowed.data(), narrowed.size(), ptn::kInt);
      return;
    }
    std::vector<int64_t> values;
    values.reserve(tokens.size());
    for (const uint64_t token : tokens) {
      if (token > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("token ID is outside Long range");
      }
      values.push_back(static_cast<int64_t>(token));
    }
    executable.set_input(0, values.data(), values.size(), ptn::kLong);
  };
  // Runs one call and returns the vocabulary size of the logits it leaves in
  // `logits`.
  auto run_tokens = [&](const std::vector<uint64_t>& tokens, int64_t position) {
    set_tokens(tokens);
    set_positions(position, tokens.size());
    executable.execute();
    const std::vector<int64_t> output_shape =
        executable.output_sizes(logits_output);
    if (output_shape.empty() || output_shape.back() <= 0) {
      throw std::runtime_error("logits output has no vocabulary dimension");
    }
    const size_t logits_numel =
        static_cast<size_t>(ptn::runner_utils::numel_of(output_shape));
    const size_t vocabulary_size = static_cast<size_t>(output_shape.back());
    logits.resize(logits_numel);
    executable.get_output(
        logits_output, logits.data(), logits.size(), ptn::kFloat);
    if (FLAGS_print_top_logits) {
      const float* row = logits.data() + logits_numel - vocabulary_size;
      const std::vector<size_t> ranks =
          ptn::top_logit_indices(row, vocabulary_size, 5);
      std::fprintf(stderr, "top_logits=%lld", static_cast<long long>(position));
      for (const size_t rank : ranks) {
        std::fprintf(stderr, ",%zu:%.8g", rank, row[rank]);
      }
      std::fprintf(stderr, "\n");
    }
    return vocabulary_size;
  };
  auto prefill = [&](int64_t start) {
    size_t vocabulary_size = 0;
    for (size_t offset = 0; offset < prompt_tokens.size();
         offset += max_chunk) {
      const auto first = prompt_tokens.begin() + static_cast<ptrdiff_t>(offset);
      vocabulary_size = run_tokens(
          {first,
           first +
               static_cast<ptrdiff_t>(
                   std::min(max_chunk, prompt_tokens.size() - offset))},
          start + static_cast<int64_t>(offset));
    }
    if (start == 0 && !FLAGS_dump_first_logits.empty()) {
      FILE* output = std::fopen(FLAGS_dump_first_logits.c_str(), "wbx");
      if (output == nullptr) {
        throw std::runtime_error(
            "cannot create new logits dump: " + FLAGS_dump_first_logits);
      }
      const float* row = logits.data() + logits.size() - vocabulary_size;
      const size_t written =
          std::fwrite(row, sizeof(float), vocabulary_size, output);
      const int close_status = std::fclose(output);
      if (written != vocabulary_size || close_status != 0) {
        std::remove(FLAGS_dump_first_logits.c_str());
        throw std::runtime_error(
            "failed to write complete initial logits dump");
      }
    }
    return vocabulary_size;
  };

  // The same prompt and positions overwrite the same cache rows. Keep this
  // explicitly opt-in so the normal timing still includes first-use GPU
  // warmup and can be compared with standalone benchmarks either way.
  if (FLAGS_warmup_prefill) {
    if (!FLAGS_dump_first_logits.empty() || FLAGS_print_top_logits) {
      throw std::invalid_argument(
          "warmup_prefill cannot be combined with logits diagnostics");
    }
    prefill(start_position);
    // A sliding layer cannot rewind to the start of a prompt longer than its
    // window, so the real prefill starts from an empty cache.
    if (offgraph_cache != nullptr) {
      offgraph_cache->clear();
    }
  }

  Generation generation;
  const Clock::time_point prompt_start = Clock::now();
  uint64_t next_token = next_token_of(logits, prefill(start_position));
  generation.prompt_ms = elapsed_ms(prompt_start, Clock::now());

  size_t generated = 0;
  std::vector<uint64_t> generated_ids;
  uint64_t previous_token = prompt_tokens.back();
  if (!FLAGS_interactive) {
    std::printf("text:      ");
  }
  std::fflush(stdout);
  while (generated < limit && !eos_ids.contains(next_token)) {
    auto decoded = tokenizer.decode(previous_token, next_token);
    if (!decoded.ok()) {
      throw std::runtime_error(
          "failed to decode token: tokenizer error " +
          std::to_string(static_cast<uint32_t>(decoded.error())));
    }
    std::printf("%s", decoded.get().c_str());
    std::fflush(stdout);
    previous_token = next_token;
    if (FLAGS_print_token_ids) {
      generated_ids.push_back(next_token);
    }
    ++generated;
    if (generated < limit) {
      const Clock::time_point step_start = Clock::now();
      const size_t vocabulary_size = run_tokens(
          {previous_token},
          start_position +
              static_cast<int64_t>(prompt_tokens.size() + generated - 1));
      next_token = next_token_of(logits, vocabulary_size);
      generation.decode_ms += elapsed_ms(step_start, Clock::now());
      ++generation.decode_steps;
    }
  }
  std::printf("\n");
  generation.unfed_token = next_token;
  generation.ended_on_eos = eos_ids.contains(next_token);
  if (FLAGS_print_token_ids) {
    std::printf("token_ids: ");
    for (size_t i = 0; i < generated_ids.size(); ++i) {
      std::printf(
          "%s%llu",
          i == 0 ? "" : ",",
          static_cast<unsigned long long>(generated_ids[i]));
    }
    std::printf("\n");
  }
  return generation;
}

} // namespace

namespace ptn {

// cppcheck-suppress unusedFunction
int run_llm(
    const std::function<std::unique_ptr<EngineHost>()>& make_host,
    const char* cache_backend_id) {
  if (FLAGS_package.empty() || FLAGS_tokenizer.empty()) {
    std::fprintf(stderr, "error: --package and --tokenizer are required\n");
    return 1;
  }
  // The tokenizer takes num_bos as an int8_t.
  if (FLAGS_max_new_tokens <= 0 || FLAGS_max_seq_length <= 0 ||
      FLAGS_num_bos < 0 || FLAGS_num_bos > 127) {
    std::fprintf(stderr, "error: invalid generation limits\n");
    return 1;
  }

  try {
    const auto eos_ids = ptn::parse_token_ids(FLAGS_eos_ids);

    std::unique_ptr<::tokenizers::Tiktoken> tokenizer =
        example::get_tiktoken_for_llama();
    if (tokenizer->load(FLAGS_tokenizer) != ::tokenizers::Error::Ok) {
      throw std::runtime_error(
          "failed to load --tokenizer as a Llama 3 TikToken model");
    }
    auto encoded =
        tokenizer->encode(FLAGS_prompt, static_cast<int8_t>(FLAGS_num_bos), 0);
    if (!encoded.ok()) {
      throw std::runtime_error(
          "failed to encode prompt: tokenizer error " +
          std::to_string(static_cast<uint32_t>(encoded.error())));
    }
    std::vector<uint64_t> prompt_tokens = std::move(encoded.get());
    if (prompt_tokens.empty()) {
      throw std::runtime_error("prompt encoded to no tokens");
    }

    const Clock::time_point load_start = Clock::now();
    const auto package =
        std::make_shared<const ptn::Package>(ptn::Package::load(FLAGS_package));
    const ptn::ByteSpan program_bytes = package->program_bytes();
    const auto program = std::make_shared<const ptn::Program>(
        ptn::Program::load(program_bytes.data(), program_bytes.size()));
    const std::vector<std::string> method_names = program->method_names();
    if (method_names.empty()) {
      throw std::runtime_error("package holds no methods");
    }
    const std::string method_name =
        FLAGS_method.empty() ? method_names.front() : FLAGS_method;
    const ptn::Method& method = program->get_method(method_name);

    std::vector<bool> user_outputs;
    user_outputs.reserve(method.output_specs.size());
    // cppcheck-suppress-begin useStlAlgorithm
    for (const ptn::OutputSpec& spec : method.output_specs) {
      user_outputs.push_back(spec.kind == ptn::OutputKind::UserOutput);
    }
    // cppcheck-suppress-end useStlAlgorithm
    const size_t logits_output = ptn::resolve_logits_output(
        method.graph.output_ids.size(), user_outputs);

    const std::unique_ptr<ptn::EngineHost> host = make_host();
    // Declared before the context, which may hold the cache, so it outlives it.
    std::optional<OffGraphCache> kv;
    if (const auto geometry = offgraph_cache_geometry(method, *package)) {
      // Serialized input sizes are upper bounds; the tokens are [1, T].
      const std::vector<int64_t>& token_bound =
          method.graph.value(method.graph.input_ids.at(0)).tensor_meta().sizes;
      if (token_bound.empty() || token_bound.back() <= 0 ||
          token_bound.back() > std::numeric_limits<int>::max()) {
        throw std::runtime_error("token input must have shape [1, sequence]");
      }
      kv = install_offgraph_cache(
          cache_backend_id, *geometry, static_cast<int>(token_bound.back()));
    } else if (FLAGS_interactive) {
      throw std::invalid_argument(
          "--interactive needs a package exported with --offgraph_cache");
    }
    const size_t cache_length = kv
        ? static_cast<size_t>(kv->control->capacity())
        : kv_cache_length(method);
    if (!kv) {
      reject_offgraph_cache_flags();
    }
    std::unique_ptr<ptn::EngineContext> context =
        host->create_context_with_options(
            program, package, kv ? kv->options : ptn::EngineOptions{});
    const Clock::time_point load_end = Clock::now();
    const Clock::time_point compile_start = Clock::now();
    std::unique_ptr<ptn::EngineExecutable> executable =
        context->compile(method_name);
    const Clock::time_point compile_end = Clock::now();

    std::printf(
        "engine:    %s on %s\n",
        host->name().c_str(),
        host->device_name().c_str());
    std::printf(
        "method:    %s (%zu inputs, %zu outputs)\n",
        method_name.c_str(),
        executable->num_inputs(),
        executable->num_outputs());
    std::printf("compile:   %.3f ms\n", elapsed_ms(compile_start, compile_end));
    std::printf(
        "load:      %.3f ms (package, program, engine host, and context)\n",
        elapsed_ms(load_start, load_end));
    std::printf("constants: %zu package bytes\n", package->constant_bytes());

    if (executable->num_inputs() != 2) {
      throw std::runtime_error("runner requires token and position inputs");
    }
    if (logits_output >= executable->num_outputs()) {
      throw std::runtime_error("logits output is outside engine outputs");
    }
    // Before any resize, input sizes are the serialized upper bounds.
    const std::vector<int64_t> token_shape = executable->input_sizes(0);
    if (token_shape.size() != 2 || token_shape[0] != 1 || token_shape[1] <= 0) {
      throw std::runtime_error("token input must have shape [1, sequence]");
    }
    // ExecuTorch Llama exports take the start position; HuggingFace exports
    // take one cache position per token.
    const std::vector<int64_t> position_shape = executable->input_sizes(1);
    const bool per_token_positions =
        position_shape.size() == 1 && position_shape[0] == token_shape[1];
    if (!per_token_positions && runner_utils::numel_of(position_shape) != 1) {
      throw std::runtime_error(
          "position input must have one element or one per token");
    }
    for (size_t i = 0; i < 2; ++i) {
      const ptn::ScalarType dtype = executable->input_dtype(i);
      if (dtype != ptn::kInt && dtype != ptn::kLong) {
        throw std::runtime_error(
            "token and position inputs must be Int or Long, got " +
            std::string(ptn::scalar_type_name(dtype)));
      }
    }
    const ptn::ScalarType output_dtype =
        executable->output_dtype(logits_output);
    if (output_dtype != ptn::kFloat && output_dtype != ptn::kHalf) {
      throw std::runtime_error(
          "logits must be Float or Half, got " +
          std::string(ptn::scalar_type_name(output_dtype)));
    }
    if (!FLAGS_verify_samples.empty()) {
      if (FLAGS_warmup_prefill || !FLAGS_dump_first_logits.empty()) {
        throw std::invalid_argument(
            "verify_samples cannot be combined with warmup_prefill or "
            "dump_first_logits");
      }
      const ptn::SampleSet set = ptn::read_sample_set(
          ptn::ZipReader::open(FLAGS_package), FLAGS_verify_samples);
      size_t failed_calls = 0;
      for (size_t i = 0; i < set.samples.size(); ++i) {
        const std::vector<ptn::SampleCall>& calls = set.samples[i].calls;
        // Mutable state lives in the context, so a fresh one is the program's
        // initial state; an off-graph cache only needs clearing.
        if (i > 0 && kv) {
          kv->control->clear();
        } else if (i > 0) {
          executable.reset();
          context.reset();
          context = host->create_context(program, package);
          executable = context->compile(method_name);
        }
        std::vector<uint64_t> prompt;
        std::vector<uint64_t> forced;
        for (size_t j = 0; j < calls.size(); ++j) {
          if (calls[j].method != method_name) {
            throw std::runtime_error(
                "sample " + std::to_string(i) + " call " + std::to_string(j) +
                " runs method " + calls[j].method +
                ", but the runner replays only " + method_name);
          }
          if (calls[j].inputs.size() != 2 || !calls[j].outputs ||
              calls[j].outputs->size() <= logits_output ||
              (*calls[j].outputs)[logits_output].dtype != ptn::kFloat ||
              (*calls[j].outputs)[logits_output].sizes.empty() ||
              calls[j].inputs[0].sizes.size() != 2 ||
              calls[j].inputs[1].sizes.size() != 1) {
            throw std::runtime_error(
                "sample calls need [1, n] tokens, [n] positions and Float logits");
          }
          const std::vector<int64_t> ids = integer_values(calls[j].inputs[0]);
          const int64_t vocabulary_size =
              (*calls[j].outputs)[logits_output].sizes.back();
          if (std::any_of(ids.begin(), ids.end(), [&](int64_t id) {
                return id < 0 || id >= vocabulary_size;
              })) {
            throw std::runtime_error(
                "sample " + std::to_string(i) + " call " + std::to_string(j) +
                " has a token outside the vocabulary");
          }
          // The runner sends what generate() would: the prompt from 0, then one
          // token per call.
          const size_t count = j == 0 ? ids.size() : 1;
          const int64_t start =
              static_cast<int64_t>(j == 0 ? 0 : prompt.size() + j - 1);
          std::vector<int64_t> positions(per_token_positions ? count : 1);
          std::iota(positions.begin(), positions.end(), start);
          if (ids.empty() || (j > 0 && ids.size() != 1) ||
              integer_values(calls[j].inputs[1]) != positions) {
            throw std::runtime_error(
                "sample " + std::to_string(i) + " call " + std::to_string(j) +
                " is not a prefill or the next decode step");
          }
          if (ids.size() > static_cast<size_t>(token_shape[1])) {
            throw std::runtime_error(
                "sample " + std::to_string(i) +
                " prefill exceeds the method's " +
                std::to_string(token_shape[1]) + " token input");
          }
          if (j == 0) {
            prompt.assign(ids.begin(), ids.end());
          } else {
            forced.push_back(static_cast<uint64_t>(ids[0]));
          }
        }
        // generate() runs one call per generated token.
        if (ptn::generation_limit(
                prompt.size(),
                calls.size(),
                static_cast<size_t>(FLAGS_max_seq_length),
                cache_length,
                static_cast<size_t>(token_shape[1])) < calls.size()) {
          throw std::runtime_error(
              "sample " + std::to_string(i) + " does not fit the method");
        }
        size_t call = 0;
        std::string report;
        const NextToken check_then_force = [&](const std::vector<float>& logits,
                                               size_t vocabulary_size) {
          const ptn::SampleTensor& want = (*calls[call].outputs)[logits_output];
          if (want.bytes.size() != logits.size() * sizeof(float)) {
            throw std::runtime_error("sample logits have a different size");
          }
          std::vector<float> want_logits(logits.size());
          std::memcpy(want_logits.data(), want.bytes.data(), want.bytes.size());
          const ptn::Tolerance tol = calls[call].tol.value_or(ptn::Tolerance{});
          const ptn::Closeness result = ptn::compare_close(
              logits.data(),
              want_logits.data(),
              logits.size(),
              tol.atol,
              tol.rtol);
          failed_calls += result.close ? 0 : 1;
          std::array<char, 160> line{};
          std::snprintf(
              line.data(),
              line.size(),
              "sample %zu call %zu: max_abs_error=%.6g nan=%zu (atol %g, rtol %g) "
              "%s\n",
              i,
              call,
              result.max_abs_error,
              result.nan_count,
              tol.atol,
              tol.rtol,
              result.close ? "ok" : "FAIL");
          report += line.data();
          ++call;
          return call < calls.size()
              ? forced[call - 1]
              : ptn::greedy_argmax_last_row(logits, vocabulary_size);
        };
        generate(
            *executable,
            logits_output,
            per_token_positions,
            static_cast<size_t>(token_shape[1]),
            *tokenizer,
            prompt,
            0,
            calls.size(),
            {},
            check_then_force,
            nullptr);
        std::printf("%s", report.c_str());
      }
      std::printf(
          "verify:    %s, %zu calls failed\n",
          FLAGS_verify_samples.c_str(),
          failed_calls);
      return failed_calls == 0 ? 0 : 1;
    }

    if (FLAGS_interactive) {
      // Every conversation prefills from position 0, which would rewrite the
      // logits dump.
      if (FLAGS_warmup_prefill || !FLAGS_dump_first_logits.empty()) {
        throw std::invalid_argument(
            "--warmup_prefill and --dump_first_logits cannot be combined with "
            "--interactive");
      }
      const int64_t capacity = kv->control->capacity();
      std::printf("chat:      Ctrl-D quits, /reset starts over\n");
      int64_t position = 0;
      // The last reply's final token, never fed, unless it was an EOS; and
      // whether a reply precedes the next turn, which closes it with
      // <|eot_id|> whichever way it ended.
      std::optional<uint64_t> unfed;
      bool after_reply = false;
      std::string line;
      while (true) {
        std::printf("\n> ");
        std::fflush(stdout);
        if (!std::getline(std::cin, line)) {
          std::printf("\n");
          return 0;
        }
        if (line == "/reset" || line.empty()) {
          if (line == "/reset") {
            kv->control->clear();
            position = 0;
            unfed.reset();
            after_reply = false;
          }
          continue;
        }
        const auto encode_turn = [&] {
          const std::string text =
              std::string(after_reply ? "<|eot_id|>" : "") +
              "<|start_header_id|>user<|end_header_id|>\n\n" + line +
              "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
          auto turn = tokenizer->encode(
              text, static_cast<int8_t>(position == 0 ? FLAGS_num_bos : 0), 0);
          if (!turn.ok()) {
            throw std::runtime_error(
                "failed to encode message: tokenizer error " +
                std::to_string(static_cast<uint32_t>(turn.error())));
          }
          std::vector<uint64_t> tokens = std::move(turn.get());
          if (unfed) {
            tokens.insert(tokens.begin(), *unfed);
          }
          return tokens;
        };
        std::vector<uint64_t> tokens = encode_turn();
        // Room for the turn and at least one reply token.
        if (position + static_cast<int64_t>(tokens.size()) >= capacity) {
          std::printf("[context full, starting a new conversation]\n");
          kv->control->clear();
          position = 0;
          unfed.reset();
          after_reply = false;
          tokens = encode_turn();
          if (static_cast<int64_t>(tokens.size()) >= capacity) {
            std::printf("[message exceeds the KV cache capacity]\n");
            continue;
          }
        }
        const size_t limit = std::min(
            static_cast<size_t>(FLAGS_max_new_tokens),
            static_cast<size_t>(
                capacity + 1 - position - static_cast<int64_t>(tokens.size())));
        const Generation reply = generate(
            *executable,
            logits_output,
            per_token_positions,
            static_cast<size_t>(token_shape[1]),
            *tokenizer,
            tokens,
            position,
            limit,
            eos_ids,
            ptn::greedy_argmax_last_row,
            nullptr);
        position += static_cast<int64_t>(tokens.size() + reply.decode_steps);
        unfed = reply.ended_on_eos ? std::nullopt
                                   : std::optional(reply.unfed_token);
        after_reply = true;
      }
    }

    const size_t limit = ptn::generation_limit(
        prompt_tokens.size(),
        static_cast<size_t>(FLAGS_max_new_tokens),
        static_cast<size_t>(FLAGS_max_seq_length),
        cache_length,
        static_cast<size_t>(token_shape[1]));

    const Generation generation = generate(
        *executable,
        logits_output,
        per_token_positions,
        static_cast<size_t>(token_shape[1]),
        *tokenizer,
        prompt_tokens,
        0,
        limit,
        eos_ids,
        ptn::greedy_argmax_last_row,
        kv ? kv->control : nullptr);

    const double prompt_tokens_per_second = generation.prompt_ms > 0.0
        ? 1000.0 * static_cast<double>(prompt_tokens.size()) /
            generation.prompt_ms
        : 0.0;
    const double tokens_per_second = generation.decode_ms > 0.0 ? 1000.0 *
            static_cast<double>(generation.decode_steps) / generation.decode_ms
                                                                : 0.0;
    std::printf(
        "prompt:    %zu tokens, %.3f ms (%.3f tokens/s)\n",
        prompt_tokens.size(),
        generation.prompt_ms,
        prompt_tokens_per_second);
    std::printf(
        "decode:    %zu steps, %.3f ms (%.3f steps/s)\n",
        generation.decode_steps,
        generation.decode_ms,
        tokens_per_second);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}

} // namespace ptn
