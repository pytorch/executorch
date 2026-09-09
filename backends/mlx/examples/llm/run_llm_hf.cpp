/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// C++ runner for HuggingFace LLMs on the MLX backend. Unlike the pybindings
// run_llm_hf.py, it can bind the off-graph KV cache: with --kv-max-capacity it
// builds an MLXSequenceCache, installs it in the process-global registry, and
// passes its cache_key as a load-time backend option (the rendezvous init()
// reads). Its shape comes from constant methods the export publishes; the flags
// below only choose policy. Without --kv-max-capacity it runs an in-graph model
// unchanged -- so the same binary compares both cache paths. Greedy decode
// unless --temperature is set.
//
// Usage:
//   run_llm_hf --pte <model.pte> --tokenizer <tokenizer file> [flags]
//
// --help lists every flag with its default.

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#pragma clang diagnostic pop

#include <executorch/backends/mlx/runtime/MLXSequenceCache.h>
#include <executorch/backends/mlx/runtime/backend_options.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_stream.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>

#include <pytorch/tokenizers/tokenizer.h>

#include <gflags/gflags.h>
#include <mlx/memory.h>

#include <executorch/backends/mlx/examples/llm/runner_utils.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

DEFINE_string(pte, "", "Model .pte file.");
DEFINE_string(
    tokenizer,
    "",
    "Tokenizer file; any format the shared loader accepts (tokenizer.json, "
    "tiktoken, sentencepiece).");
DEFINE_string(prompt, "The quick brown fox", "Prompt to generate from.");
DEFINE_int32(max_new_tokens, 50, "Tokens to generate, excluding the prompt.");
DEFINE_double(
    temperature,
    0.0,
    "Sampling temperature. 0 is greedy argmax, which is what makes two .pte "
    "files comparable; above 0 samples and the run stops being reproducible.");
DEFINE_string(
    chat,
    "llama3",
    "Instruct chat template to wrap the prompt in: llama3, gemma, gemma4, or 0 "
    "to disable. Raw text confuses an instruct model into emitting turn "
    "markers.");
DEFINE_int32(
    kv_max_capacity,
    0,
    "Off-graph: how much history the cache may hold. Setting it selects the "
    "off-graph path; the cache's shape comes from the .pte, so the kv_ flags "
    "only choose policy.");
DEFINE_string(
    kv_storage_dtype,
    "",
    "Off-graph: override KV storage dtype with bf16|fp16|fp32. Defaults to "
    "the PTE activation dtype, or bf16 when metadata is absent.");
DEFINE_int32(
    kv_initial_capacity,
    -1,
    "Off-graph: the cache pool's starting size; it grows (doubling) up to "
    "capacity. -1 keeps the CacheConfig default. Small values force growth.");
DEFINE_string(
    kv_windows,
    "",
    "Off-graph: impose an attention pattern other than the model's own, e.g. "
    "\"512\" to make every layer sliding.");
DEFINE_bool(
    interactive,
    false,
    "Multi-turn chat on stdin instead of a single prompt; off-graph only.");
DEFINE_bool(
    warmup,
    false,
    "Run once before measuring, to absorb JIT and pool growth.");

using ::executorch::backends::mlx::examples::llm::resolve_kv_storage_dtype;
using ::executorch::backends::mlx::examples::llm::resolve_stop_tokens;
using ::executorch::backends::mlx::examples::llm::StopTokens;
using ::executorch::backends::mlx::examples::llm::wrap_turn;
using ::executorch::extension::make_tensor_ptr;
using ::executorch::extension::Module;
using ::executorch::extension::llm::check_vocab_size;
using ::executorch::extension::llm::LogitsToKeepMode;
using ::executorch::extension::llm::read_activation_dtype;
using ::executorch::extension::llm::read_logits_to_keep_mode;
using ::executorch::extension::llm::read_max_seq_len;
using ::executorch::extension::llm::read_vocab_size;
using ::executorch::extension::llm::TextStream;
using ::executorch::runtime::Error;

namespace cache = ::executorch::extension::llm::cache;

namespace {

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t d = s.find(delim, pos);
    out.push_back(
        s.substr(pos, d == std::string::npos ? std::string::npos : d - pos));
    if (d == std::string::npos) {
      break;
    }
    pos = d + 1;
  }
  return out;
}

bool parse_int_list(
    const std::string& spec,
    char delim,
    std::vector<int>& out) {
  for (const std::string& field : split(spec, delim)) {
    if (field.empty()) {
      return false;
    }
    try {
      out.push_back(std::stoi(field));
    } catch (const std::exception&) {
      return false;
    }
  }
  return true;
}

// Constant methods the export publishes (get_n_caches and friends). They carry
// no delegate, so reading them only needs the program loaded -- which is what
// lets the cache be built before forward's backend init consumes its key.
std::optional<int64_t> const_int(Module& module, const char* name) {
  const auto r = module.execute(name);
  if (!r.ok() || r->empty() || !r->at(0).isInt()) {
    return std::nullopt;
  }
  return r->at(0).toInt();
}

bool is_supported_logits_type(::executorch::aten::ScalarType type) {
  using ScalarType = ::executorch::aten::ScalarType;
  return type == ScalarType::Float || type == ScalarType::Half ||
      type == ScalarType::BFloat16 || type == ScalarType::UInt16;
}

bool validate_forward_abi(
    Module& module,
    LogitsToKeepMode logits_to_keep_mode,
    std::int64_t& vocab_size) {
  const auto meta = module.method_meta("forward");
  if (!meta.ok()) {
    std::cerr << "Forward metadata is unavailable" << std::endl;
    return false;
  }
  const std::size_t expected_inputs =
      logits_to_keep_mode == LogitsToKeepMode::Selected ? 3 : 2;
  if (meta->num_inputs() != expected_inputs) {
    std::cerr << "Forward must take " << expected_inputs
              << " inputs for its logits-to-keep mode, got "
              << meta->num_inputs() << std::endl;
    return false;
  }

  const auto tokens = meta->input_tensor_meta(0);
  const auto positions = meta->input_tensor_meta(1);
  if (!tokens.ok() || !positions.ok()) {
    std::cerr << "Forward token and position inputs must be tensors"
              << std::endl;
    return false;
  }
  const auto token_sizes = tokens->sizes();
  const auto position_sizes = positions->sizes();
  if (tokens->scalar_type() != ::executorch::aten::ScalarType::Long ||
      token_sizes.size() != 2 || token_sizes[0] != 1 || token_sizes[1] <= 0 ||
      positions->scalar_type() != ::executorch::aten::ScalarType::Long ||
      position_sizes.size() != 1 || position_sizes[0] != token_sizes[1]) {
    std::cerr << "Forward must take Long[1, T] tokens and Long[T] positions"
              << std::endl;
    return false;
  }
  if (logits_to_keep_mode == LogitsToKeepMode::Selected) {
    const auto selector = meta->input_tensor_meta(2);
    if (!selector.ok() ||
        selector->scalar_type() != ::executorch::aten::ScalarType::Long ||
        selector->sizes().size() != 1) {
      std::cerr << "Selected logits selector must be rank-one Long"
                << std::endl;
      return false;
    }
  }

  if (meta->num_outputs() == 0) {
    std::cerr << "Forward publishes no logits output" << std::endl;
    return false;
  }
  const auto logits = meta->output_tensor_meta(0);
  if (!logits.ok() || logits->sizes().size() < 2 ||
      logits->sizes()[logits->sizes().size() - 1] <= 0 ||
      !is_supported_logits_type(logits->scalar_type())) {
    std::cerr
        << "Forward logits must have supported dtype and shape [..., vocab]"
        << std::endl;
    return false;
  }
  vocab_size = logits->sizes()[logits->sizes().size() - 1];
  return true;
}

std::optional<std::vector<int>> const_ints(Module& module, const char* name) {
  const auto r = module.execute(name);
  if (!r.ok() || r->empty() || !r->at(0).isTensor()) {
    return std::nullopt;
  }
  const auto t = r->at(0).toTensor();
  if (t.scalar_type() != ::executorch::aten::ScalarType::Int) {
    return std::nullopt;
  }
  const int32_t* p = t.const_data_ptr<int32_t>();
  return std::vector<int>(p, p + t.numel());
}

// Fill in the cache geometry the export published: get_n_caches, then one
// entry per cache in get_kv_heads / get_head_dims / get_windows (0 = flat).
// Capacity and dtype stay with the flags. False means this is not an off-graph
// model.
bool read_kv_layout(
    Module& module,
    int prefill_chunk,
    cache::CacheConfig& cfg) {
  const auto n_caches = const_int(module, "get_n_caches");
  const auto kv_heads = const_ints(module, "get_kv_heads");
  const auto head_dims = const_ints(module, "get_head_dims");
  const auto windows = const_ints(module, "get_windows");
  if (!n_caches || !kv_heads || !head_dims || !windows) {
    return false;
  }
  cfg.max_write = prefill_chunk;
  const size_t n = static_cast<size_t>(*n_caches);
  if (kv_heads->size() != n || head_dims->size() != n || windows->size() != n) {
    return false;
  }
  cfg.n_layers = static_cast<int>(n);
  cfg.layers.clear();
  cfg.layers.reserve(n);
  for (size_t l = 0; l < n; ++l) {
    cache::LayerConfig lc{};
    lc.n_kv_heads = (*kv_heads)[l];
    lc.head_dim = (*head_dims)[l];
    lc.policy = (*windows)[l] > 0
        ? cache::LayerPolicy{cache::LayerPolicy::Kind::Ring, (*windows)[l]}
        : cache::LayerPolicy{cache::LayerPolicy::Kind::Flat, 0};
    cfg.layers.push_back(lc);
  }
  return true;
}

// Replace the model's own attention pattern with `spec`, a comma-separated list
// of windows repeating over the caches (0 = flat). One entry makes every layer
// sliding. Only the policy changes; each cache keeps the geometry the .pte
// declared, so this cannot desync from the graph.
//
// The export sizes the chunk to the model's own window; narrowing the window
// here would leave the ring (window + max_write - 1) sized by the chunk
// instead, so the chunk follows the window down.
bool apply_window_override(const std::string& spec, cache::CacheConfig& cfg) {
  std::vector<int> pattern;
  if (!parse_int_list(spec, ',', pattern) || pattern.empty()) {
    return false;
  }
  for (size_t l = 0; l < cfg.layers.size(); ++l) {
    const int w = pattern[l % pattern.size()];
    cfg.layers[l].policy = w > 0
        ? cache::LayerPolicy{cache::LayerPolicy::Kind::Ring, w}
        : cache::LayerPolicy{cache::LayerPolicy::Kind::Flat, 0};
  }
  int narrowest = 0; // smallest ring window in the pattern; 0 if all flat
  for (int w : pattern) {
    if (w > 0 && (narrowest == 0 || w < narrowest)) {
      narrowest = w;
    }
  }
  if (cfg.max_write && narrowest > 0 && narrowest < *cfg.max_write) {
    cfg.max_write = narrowest;
  }
  return cache::valid(cfg);
}

// Human-readable name for a kv_dtype (an ET ScalarType int). Only the
// storage dtypes the pool uses are named; anything else prints its raw value.
std::string dtype_name(int st) {
  using S = ::executorch::runtime::etensor::ScalarType;
  switch (static_cast<S>(st)) {
    case S::Half:
      return "Half(fp16)";
    case S::Float:
      return "Float(fp32)";
    case S::BFloat16:
      return "BFloat16";
    default:
      return "scalar_type_" + std::to_string(st);
  }
}

// Announce the cache shape: the same .pte runs under whatever config this
// invocation asks for -- capacity, storage dtype, flat/ring layers -- with no
// re-export. The footprint lines printed later then show it growing at runtime.
void print_cache_summary(const cache::CacheConfig& cfg) {
  // Ring layers grouped by window: --kv-windows can give each layer its own,
  // and the pools are sized per layer, so a single number would misreport them.
  std::map<int, int> ring;
  int flat = 0;
  for (int l = 0; l < cfg.n_layers; ++l) {
    const cache::LayerConfig& lc =
        cfg.layers.size() == 1 ? cfg.layers.front() : cfg.layers[l];
    if (lc.policy.kind == cache::LayerPolicy::Kind::Ring) {
      ++ring[lc.policy.window];
    } else {
      ++flat;
    }
  }
  std::cout << "\n[cache] off-graph seq | capacity=" << cfg.capacity
            << " initial=" << cfg.initial_capacity
            << " kv_dtype=" << dtype_name(cfg.kv_dtype);
  if (cfg.max_write) {
    std::cout << " max_write=" << *cfg.max_write;
  }
  std::cout << "\n        " << cfg.n_layers << " layers: " << flat << " flat";
  for (const auto& [window, n] : ring) {
    std::cout << " + " << n << " ring(window " << window << ")";
  }
  std::cout << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string& pte = FLAGS_pte;
  const std::string& tok_path = FLAGS_tokenizer;
  const std::string& kv_dtype = FLAGS_kv_storage_dtype;
  const std::string& kv_windows = FLAGS_kv_windows;
  const std::string& prompt = FLAGS_prompt;
  const std::string& chat = FLAGS_chat;
  const int kv_capacity = FLAGS_kv_max_capacity;
  const int max_new = FLAGS_max_new_tokens;
  const float temperature = static_cast<float>(FLAGS_temperature);
  const int initial_capacity = FLAGS_kv_initial_capacity;
  const bool interactive = FLAGS_interactive;
  const bool warmup = FLAGS_warmup;
  if (pte.empty() || tok_path.empty()) {
    std::cerr << "Required: --pte <file> --tokenizer <file>  "
                 "[--kv-max-capacity N for off-graph models]\n";
    return 1;
  }
  if (warmup && kv_capacity <= 0) {
    std::cerr << "--warmup requires an off-graph cache selected with "
                 "--kv_max_capacity"
              << std::endl;
    return 1;
  }

  try {
    // The shared loader sniffs the format, so --tokenizer takes any of the
    // files the other runners accept, not just tokenizer.json.
    auto tokenizer = ::executorch::extension::llm::load_tokenizer(tok_path);
    if (!tokenizer) {
      std::cerr << "Failed to load tokenizer: " << tok_path << std::endl;
      return 1;
    }

    // Outer-scoped because mlx_opts must outlive load_method(): the map holds
    // a view into it.
    ::executorch::runtime::BackendOptions<1> mlx_opts;
    ::executorch::runtime::LoadBackendOptionsMap options_map;
    // Load the program but not forward: the cache must exist before forward's
    // backend init reads its key, and the layout it needs is published by
    // constant methods in the same file.
    Module module(pte);
    const long load_start_ms = ::executorch::extension::llm::time_in_ms();
    if (module.load() != Error::Ok) {
      std::cerr << "Failed to load " << pte << std::endl;
      return 1;
    }
    const auto logits_to_keep_mode_result = read_logits_to_keep_mode(module);
    if (!logits_to_keep_mode_result.ok()) {
      std::cerr << "Invalid model metadata in " << pte << std::endl;
      return 1;
    }
    const LogitsToKeepMode logits_to_keep_mode = *logits_to_keep_mode_result;
    std::int64_t output_vocab_size = 0;
    if (!validate_forward_abi(module, logits_to_keep_mode, output_vocab_size)) {
      return 1;
    }
    const auto published_vocab_size = read_vocab_size(module);
    if (!published_vocab_size.ok()) {
      std::cerr << "Invalid get_vocab_size in " << pte << std::endl;
      return 1;
    }
    const auto vocab_size_result =
        check_vocab_size(*published_vocab_size, output_vocab_size);
    if (!vocab_size_result.ok()) {
      std::cerr << "Invalid get_vocab_size for the forward output in " << pte
                << std::endl;
      return 1;
    }
    const std::int32_t vocab_size = *vocab_size_result;
    const auto max_seq_len = read_max_seq_len(module);
    if (!max_seq_len.ok()) {
      std::cerr << "Invalid or missing get_max_seq_len in " << pte << std::endl;
      return 1;
    }
    const int prefill_chunk = static_cast<int>(*max_seq_len);
    StopTokens stop_tokens;
    if (!resolve_stop_tokens(*tokenizer, module, chat, stop_tokens)) {
      std::cerr << "Could not resolve stop tokens for --chat=" << chat
                << std::endl;
      return 1;
    }
    auto write_text = [](const std::string& text) {
      std::cout << text << std::flush;
    };

    // Everything past load_method is identical for both model kinds; only
    // setup differs. ctl is null for an in-graph model, which owns its cache
    // inside the graph and exposes no control face.
    auto run =
        [&](cache::SequenceControl* ctl,
            const ::executorch::runtime::LoadBackendOptionsMap* load_opts,
            int run_prefill_chunk) -> int {
      if (module.load_method(
              "forward",
              /*planned_memory=*/nullptr,
              /*event_tracer=*/nullptr,
              load_opts) != Error::Ok) {
        std::cerr << "Failed to load forward" << std::endl;
        return 1;
      }
      // Timings reported at the end, in the shared runner's format.
      ::executorch::extension::llm::Stats stats;
      stats.model_load_start_ms = load_start_ms;
      stats.model_load_end_ms = ::executorch::extension::llm::time_in_ms();

      // Weights-only baseline, so the deltas below isolate the cache.
      const double mem_at_load = ::mlx::core::get_active_memory() / 1048576.0;
      std::cout << "[mem]   after load  : " << mem_at_load << " MiB"
                << std::endl;

      auto is_stop = [&](int64_t token) {
        return stop_tokens.ids.count(static_cast<uint64_t>(token)) != 0;
      };

      // One Sampler for the whole run, as the shared runner does: constructing
      // one per token would reseed its RNG from the wall clock every time.
      // Built on first use because the vocab size comes from the logits -- this
      // export publishes no get_vocab_size.
      std::optional<::executorch::extension::llm::Sampler> sampler;

      auto step = [&](const std::vector<int64_t>& ids,
                      const std::vector<int64_t>& pos) {
        auto in =
            make_tensor_ptr({1, (int)ids.size()}, std::vector<int64_t>(ids));
        auto cp = make_tensor_ptr({(int)pos.size()}, std::vector<int64_t>(pos));
        auto out = [&]() -> ::executorch::runtime::Result<
                             std::vector<::executorch::runtime::EValue>> {
          if (logits_to_keep_mode == LogitsToKeepMode::Selected) {
            auto selector = make_tensor_ptr(
                {1},
                std::vector<int64_t>{static_cast<int64_t>(ids.size() - 1)});
            return module.execute("forward", {in, cp, selector});
          }
          return module.execute("forward", {in, cp});
        }();
        if (!out.ok()) {
          throw std::runtime_error("execute failed");
        }
        if (out->empty() || !out->at(0).isTensor()) {
          throw std::runtime_error("forward returned no logits");
        }
        const auto& logits = out->at(0).toTensor();
        const int64_t actual_vocab_size = logits.dim() == 0
            ? 0
            : static_cast<int64_t>(logits.size(logits.dim() - 1));
        const int64_t expected_rows =
            logits_to_keep_mode == LogitsToKeepMode::Full
            ? static_cast<int64_t>(ids.size())
            : 1;
        if (logits.dim() != 3 || logits.size(0) != 1 ||
            logits.size(1) != expected_rows ||
            actual_vocab_size != vocab_size) {
          throw std::runtime_error("forward returned an invalid logits shape");
        }
        if (!sampler) {
          sampler.emplace(vocab_size, temperature);
        }
        stats.on_sampling_begin();
        const int32_t tok =
            ::executorch::extension::llm::sample_from_logits(logits, *sampler);
        stats.on_sampling_end();
        return static_cast<int64_t>(tok);
      };

      // Prefill in chunks, so a ring layer holds window + chunk - 1 slots
      // rather than growing with the prompt. Only the last chunk's token is
      // kept; the earlier ones exist to place their K/V in the cache.
      auto prefill = [&](const std::vector<int64_t>& ids,
                         const std::vector<int64_t>& pos) {
        const size_t step_size = static_cast<size_t>(run_prefill_chunk);
        int64_t next = 0;
        for (size_t off = 0; off < ids.size(); off += step_size) {
          const size_t n = std::min(step_size, ids.size() - off);
          next = step(
              {ids.begin() + off, ids.begin() + off + n},
              {pos.begin() + off, pos.begin() + off + n});
        }
        return next;
      };

      // Multi-turn: history stays in the cache, so each turn only prefills its
      // own tokens at the running position. /reset and /undo drive the cache's
      // control face directly -- off-graph only, since an in-graph cache gives
      // the runner no handle to its state.
      if (interactive) {
        if (ctl == nullptr) {
          std::cerr << "--interactive requires --kv-max-capacity\n";
          return 1;
        }
        std::cout
            << "Multi-turn chat. /reset clears, /undo drops the last turn, "
               "/undo N drops N tokens, /quit exits.\n";
        int64_t position = 0;
        int64_t turn_start = 0; // position this turn began at, for /undo
        std::string line;
        while (std::cout << "\n> " && std::getline(std::cin, line)) {
          if (line == "/quit") {
            break;
          }
          if (line == "/reset") {
            ctl->clear();
            position = turn_start = 0;
            std::cout << "[cleared]\n";
            continue;
          }
          if (line == "/undo" || line.rfind("/undo ", 0) == 0) {
            // Bare /undo drops the last turn; /undo N drops N tokens.
            int64_t target = turn_start;
            if (line.size() > 6) {
              try {
                const int64_t n = std::stoll(line.substr(6));
                target = n >= position ? 0 : position - n;
              } catch (const std::exception&) {
                std::cout << "[usage: /undo [n_tokens]]\n";
                continue;
              }
            }
            if (ctl->rewind(static_cast<int>(target))) {
              position = target;
              turn_start = std::min(turn_start, position);
              std::cout << "[rewound to " << position << "]\n";
            } else {
              // A sliding-window layer has physically dropped those cells.
              std::cout << "[cannot rewind to " << target << "]\n";
            }
            continue;
          }
          if (line.empty()) {
            continue;
          }

          std::string turn;
          wrap_turn(chat, line, /*with_bos=*/position == 0, turn);
          auto te = tokenizer->encode(turn, /*bos=*/chat == "0" ? 1 : 0, 0);
          if (!te.ok() || te->empty()) {
            std::cerr << "Encode failed or produced no tokens\n";
            continue;
          }
          const int n = static_cast<int>(te->size());
          // Admit the turn if its prompt plus one token fits; reserving the
          // whole max_new budget up front would report "full" with most of the
          // cache still free. Generation is then clamped to the room that
          // remains.
          if (!ctl->can_extend(n + 1)) {
            std::cout << "[cache full: " << position << "/" << ctl->capacity()
                      << ", turn " << n << " tokens"
                      << (ctl->can_extend(1) ? "" : ", length at capacity")
                      << ", use /reset]\n";
            continue;
          }
          const int budget = std::min(
              max_new, ctl->capacity() - static_cast<int>(position) - n);

          turn_start = position;
          std::vector<int64_t> tin(te->begin(), te->end()), tpos;
          for (int i = 0; i < n; ++i) {
            tpos.push_back(position + i);
          }
          int64_t next = prefill(tin, tpos);
          position += n;

          TextStream text_stream(*tokenizer, write_text, te->back());
          for (int i = 0; i < budget && !is_stop(next); ++i) {
            if (text_stream.append(static_cast<uint64_t>(next)) != Error::Ok) {
              text_stream.flush();
              std::cerr << "Failed to decode generated token" << std::endl;
              return 1;
            }
            next = step({next}, {position});
            ++position;
          }
          text_stream.flush();
          // The turn-end token stops generation, so it is neither printed nor
          // fed back -- but the next turn opens without closing this one, and
          // an unterminated assistant turn compounds over a session. Commit it,
          // at the cost of one extra step per turn.
          if (stop_tokens.turn_end_id &&
              static_cast<uint64_t>(next) == *stop_tokens.turn_end_id &&
              ctl->can_extend(1)) {
            step({next}, {position});
            ++position;
          }
          std::cout << "\n[" << position << "/" << ctl->capacity() << " tokens"
                    << (budget < max_new ? ", generation capped by capacity"
                                         : "")
                    << "]\n";
        }
        return 0;
      }

      std::string enc_input;
      if (!wrap_turn(chat, prompt, /*with_bos=*/true, enc_input)) {
        std::cerr << "Unknown --chat template: " << chat
                  << " (expected llama3, gemma, gemma4, or 0)" << std::endl;
        return 1;
      }
      const int8_t bos = chat == "0" ? 1 : 0;
      auto enc = tokenizer->encode(enc_input, bos, /*eos=*/0);
      if (!enc.ok() || enc->empty()) {
        std::cerr << "Encode failed or produced no tokens" << std::endl;
        return 1;
      }
      std::vector<uint64_t> tokens = std::move(*enc);
      const int prompt_len = static_cast<int>(tokens.size());
      std::vector<int64_t> ids(tokens.begin(), tokens.end()), prefill_pos;
      for (int i = 0; i < prompt_len; ++i) {
        prefill_pos.push_back(i);
      }
      // Sequence length against the configured ceiling, with what MLX actually
      // holds for it. Pools start at initial_capacity and grow by doubling, so
      // the bytes lag the token count in steps; bf16 storage (kv_dtype 15)
      // halves them vs fp32 (6).
      auto print_footprint = [&](const char* when, int len) {
        if (ctl == nullptr) {
          return;
        }
        const int cap = ctl->capacity();
        const double pct = cap > 0 ? 100.0 * len / cap : 0.0;
        std::cout << "[cache] " << when << ": " << len << " / " << cap
                  << " tokens (" << pct << "%)" << std::endl;
        const double mem = ::mlx::core::get_active_memory() / 1048576.0;
        std::cout << "[mem]   " << when << ": " << mem << " MiB (+"
                  << (mem - mem_at_load) << " MiB since load)" << std::endl;
      };

      // One optional warmup run to absorb JIT and pool growth, then one
      // measured run, as the shared LLM runners do. Repeats belong in a harness
      // that restarts the process: clear() rewinds the sequence but leaves the
      // pools at their grown size, so an in-process repeat cannot see
      // reallocation.
      for (int iter = 0; iter < (warmup ? 2 : 1); ++iter) {
        const bool measured = !warmup || iter == 1;
        if (iter > 0 && ctl != nullptr) {
          ctl->clear();
        }
        stats.inference_start_ms = ::executorch::extension::llm::time_in_ms();
        int64_t next = prefill(ids, prefill_pos);
        stats.prompt_eval_end_ms = ::executorch::extension::llm::time_in_ms();
        // prefill returns the first generated token, so TTFT ends with prefill
        stats.first_token_ms = stats.prompt_eval_end_ms;
        if (measured) {
          std::cout << "\n";
          print_footprint("after prefill", prompt_len);
          std::cout << "\n"; // blank line before the streamed generation
        }

        TextStream::Sink sink;
        if (measured) {
          sink = write_text;
        }
        TextStream text_stream(*tokenizer, std::move(sink), tokens.back());
        int generated = 0;
        for (int i = 0; i < max_new; ++i) {
          if (is_stop(next)) {
            break;
          }
          if (text_stream.append(static_cast<uint64_t>(next)) != Error::Ok) {
            text_stream.flush();
            std::cerr << "Failed to decode generated token" << std::endl;
            return 1;
          }
          ++generated;
          next = step({next}, {prompt_len + i});
        }
        text_stream.flush();
        stats.inference_end_ms = ::executorch::extension::llm::time_in_ms();
        if (measured) {
          std::cout << "\n\n"; // close the generation line + blank separator
          // trailing space aligns the colon with the "after prefill" line above
          print_footprint("after decode ", prompt_len + generated);
          stats.num_prompt_tokens = prompt_len;
          stats.num_generated_tokens = generated;
        }
      }
      std::cout << std::endl;
      ::executorch::extension::llm::print_report(stats);
      return 0;
    };

    // An in-graph model (mlx::kv_cache_update) binds no cache: nothing to
    // build, no key to hand the delegate, and so no registry entry to guard.
    if (kv_capacity <= 0) {
      return run(
          /*ctl=*/nullptr,
          /*load_opts=*/nullptr,
          /*run_prefill_chunk=*/prefill_chunk);
    }

    const auto activation_dtype = read_activation_dtype(module);
    if (!activation_dtype.ok()) {
      std::cerr << "Invalid get_activation_dtype in " << pte << std::endl;
      return 1;
    }
    cache::CacheConfig cfg{};
    cfg.capacity = kv_capacity;
    cfg.kv_dtype = resolve_kv_storage_dtype(kv_dtype, *activation_dtype);
    if (cfg.kv_dtype < 0) {
      std::cerr << "Invalid --kv-storage-dtype override: " << kv_dtype
                << " (bf16|fp16|fp32)" << std::endl;
      return 1;
    }
    if (!read_kv_layout(module, prefill_chunk, cfg)) {
      std::cerr << "No KV cache layout in " << pte
                << "; re-export with --use-offgraph-cache" << std::endl;
      return 1;
    }
    if (!kv_windows.empty() && !apply_window_override(kv_windows, cfg)) {
      std::cerr << "Invalid --kv-windows: " << kv_windows << std::endl;
      return 1;
    }
    if (!cache::valid(cfg)) {
      std::cerr << "Invalid cache config" << std::endl;
      return 1;
    }
    if (initial_capacity >= 0) {
      cfg.initial_capacity = initial_capacity;
    }

    const char* const cache_kind = cache::kind::kSingle;
    auto built = cache::CacheFactory::global().build(
        ::executorch::backends::mlx::kMLXBackendId, cache_kind, cfg);
    if (!built.ok()) {
      std::cerr << "Failed to build cache: " << static_cast<int>(built.error())
                << std::endl;
      return 1;
    }
    const std::shared_ptr<cache::Cache> kv = built.get();

    // Published for the delegate to find by key, and erased when this scope
    // exits. That is after run() returns, so the entry is still there for the
    // load_method() inside it.
    const cache::InstallGuard guard{kv};

    print_cache_summary(cfg);
    if (guard.set_option(mlx_opts) != Error::Ok ||
        options_map.set_options(
            ::executorch::backends::mlx::kMLXBackendId, mlx_opts.view()) !=
            Error::Ok) {
      std::cerr << "Failed to set cache_key option" << std::endl;
      return 1;
    }

    // Checked here so a null ctl inside run() can only mean "in-graph model".
    // A cache kind that offers BatchControl instead would otherwise be run as
    // if it had no cache at all, with a key published and options set.
    auto* ctl = kv->as<cache::SequenceControl>();
    if (ctl == nullptr) {
      std::cerr << "Cache kind '" << cache_kind
                << "' offers no single-sequence control face" << std::endl;
      return 1;
    }

    return run(ctl, &options_map, *cfg.max_write);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
