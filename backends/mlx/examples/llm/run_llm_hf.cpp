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
// unchanged -- so the same binary compares both cache paths. Greedy decode.
//
// Usage:
//   run_llm_hf --pte <model.pte> --tokenizer <tokenizer.json> [flags]
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
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>

#include <pytorch/tokenizers/hf_tokenizer.h>

#include <gflags/gflags.h>
#include <mlx/memory.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

DEFINE_string(pte, "", "Model .pte file.");
DEFINE_string(tokenizer, "", "tokenizer.json for the model.");
DEFINE_string(prompt, "The quick brown fox", "Prompt to generate from.");
DEFINE_int32(max_new_tokens, 50, "Tokens to generate, excluding the prompt.");
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
    "bf16",
    "Off-graph: KV storage dtype, bf16|fp16|fp32.");
DEFINE_int32(
    kv_initial_capacity,
    -1,
    "Off-graph: the cache pool's starting size; it grows (doubling) up to "
    "capacity. -1 keeps the CacheConfig default. Small values force growth.");
DEFINE_int32(
    prefill_chunk_size,
    512,
    "Tokens per prefill step. This is the largest single write, so it also "
    "sizes a ring layer to window + chunk - 1 -- without it a long prompt "
    "would need a ring as large as itself, defeating the window. Must not "
    "exceed the sequence length the .pte was exported with.");
DEFINE_string(
    kv_windows,
    "",
    "Off-graph: impose an attention pattern other than the model's own, e.g. "
    "\"512\" to make every layer sliding.");
DEFINE_bool(
    interactive,
    false,
    "Multi-turn chat on stdin instead of a single prompt; off-graph only.");
DEFINE_int32(
    warmup,
    0,
    "Throwaway iterations before measuring, to absorb JIT and pool growth.");
DEFINE_int32(
    iters,
    1,
    "Measured iterations; reports per-iter tok/s and a mean +/- stddev "
    "summary.");

using ::executorch::extension::make_tensor_ptr;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

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

int storage_dtype(const std::string& name) {
  using S = ::executorch::runtime::etensor::ScalarType;
  if (name == "bf16") {
    return static_cast<int>(S::BFloat16);
  }
  if (name == "fp16") {
    return static_cast<int>(S::Half);
  }
  if (name == "fp32") {
    return static_cast<int>(S::Float);
  }
  return -1;
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
// Only the shape comes from the .pte -- capacity, dtype and any override stay
// with the flags. False means this is not an off-graph model.
bool read_kv_layout(Module& module, cache::CacheConfig& cfg) {
  const auto n_caches = const_int(module, "get_n_caches");
  const auto kv_heads = const_ints(module, "get_kv_heads");
  const auto head_dims = const_ints(module, "get_head_dims");
  const auto windows = const_ints(module, "get_windows");
  if (!n_caches || !kv_heads || !head_dims || !windows) {
    return false;
  }
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
  return cache::valid(cfg);
}

// argmax over the last position's vocab row of a [1, T, vocab] logits tensor,
// reading whatever float dtype the op emitted.
int64_t argmax_last(const ::executorch::aten::Tensor& logits) {
  const auto dim = logits.dim();
  const int64_t vocab = logits.size(dim - 1);
  const int64_t t = dim >= 2 ? logits.size(dim - 2) : 1;
  const int64_t offset = (t - 1) * vocab; // start of the last row
  auto scan = [&](auto* data) {
    int64_t best = 0;
    float best_v = -1e30f;
    for (int64_t i = 0; i < vocab; ++i) {
      const float v = static_cast<float>(data[offset + i]);
      if (v > best_v) {
        best_v = v;
        best = i;
      }
    }
    return best;
  };
  switch (logits.scalar_type()) {
    case ::executorch::aten::ScalarType::Float:
      return scan(logits.const_data_ptr<float>());
    case ::executorch::aten::ScalarType::Half:
      return scan(logits.const_data_ptr<::executorch::aten::Half>());
    case ::executorch::aten::ScalarType::BFloat16:
      return scan(logits.const_data_ptr<::executorch::aten::BFloat16>());
    default:
      throw std::runtime_error("argmax_last: unsupported logits dtype");
  }
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

// One user turn wrapped in the model's instruct template. Returns false for an
// unknown template name. The leading BOS belongs to the first turn only, so a
// continuing conversation passes with_bos=false.
bool wrap_turn(
    const std::string& chat,
    const std::string& prompt,
    bool with_bos,
    std::string& out) {
  if (chat == "0") {
    out = prompt;
  } else if (chat == "llama3") {
    out = std::string(with_bos ? "<|begin_of_text|>" : "") +
        "<|start_header_id|>user<|end_header_id|>\n\n" + prompt +
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
  } else if (chat == "gemma") {
    out = std::string(with_bos ? "<bos>" : "") + "<start_of_turn>user\n" +
        prompt + "<end_of_turn>\n<start_of_turn>model\n";
  } else if (chat == "gemma4") {
    // Gemma 4 renamed the turn markers; its own <turn|> is also the eos.
    out = std::string(with_bos ? "<bos>" : "") + "<|turn>user\n" + prompt +
        "<turn|>\n<|turn>model\n";
  } else {
    return false;
  }
  return true;
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
  const int initial_capacity = FLAGS_kv_initial_capacity;
  const int chunk = FLAGS_prefill_chunk_size;
  const bool interactive = FLAGS_interactive;
  const int warmup = FLAGS_warmup;
  const int iters = FLAGS_iters;
  if (pte.empty() || tok_path.empty()) {
    std::cerr << "Required: --pte <file> --tokenizer <file>  "
                 "[--kv-max-capacity N for off-graph models]\n";
    return 1;
  }

  try {
    // Tokenizer.
    ::tokenizers::HFTokenizer tokenizer;
    if (tokenizer.load(tok_path) != ::tokenizers::Error::Ok) {
      std::cerr << "Failed to load tokenizer: " << tok_path << std::endl;
      return 1;
    }

    // Off-graph models (update_and_attend) need a cache bound via cache_key;
    // in-graph models (mlx::kv_cache_update) don't -- omit --kv-max-capacity
    // for those. session/options are outer-scoped: session must outlive the
    // Module (it keeps the cache in the registry) and mlx_opts must outlive
    // load_method() (the map holds a view into it).
    std::optional<cache::CacheSession> session;
    ::executorch::runtime::BackendOptions<1> mlx_opts;
    ::executorch::runtime::LoadBackendOptionsMap options_map;
    const bool off_graph = kv_capacity > 0;

    // Load the program but not forward: the cache must exist before forward's
    // backend init reads its key, and the layout it needs is published by
    // constant methods in the same file.
    Module module(pte);
    if (module.load() != Error::Ok) {
      std::cerr << "Failed to load " << pte << std::endl;
      return 1;
    }

    if (off_graph) {
      cache::CacheConfig cfg{};
      cfg.capacity = kv_capacity;
      cfg.kv_dtype = storage_dtype(kv_dtype);
      if (cfg.kv_dtype < 0) {
        std::cerr << "Invalid --kv-storage-dtype: " << kv_dtype
                  << " (bf16|fp16|fp32)" << std::endl;
        return 1;
      }
      if (!read_kv_layout(module, cfg)) {
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
      // Prefill is chunked, so the chunk is the largest step the cache sees.
      cfg.max_write = chunk;
      auto built = cache::CacheBuilderRegistry::global().build(
          ::executorch::backends::mlx::kMLXBackendId, "seq", cfg);
      if (!built.ok()) {
        std::cerr << "Failed to build cache: "
                  << static_cast<int>(built.error()) << std::endl;
        return 1;
      }
      session.emplace(cache::make_unique_key(), built.get());

      print_cache_summary(cfg);
      if (mlx_opts.set_option(
              ::executorch::backends::mlx::kCacheKeyKey,
              session->key().c_str()) != Error::Ok ||
          options_map.set_options(
              ::executorch::backends::mlx::kMLXBackendId, mlx_opts.view()) !=
              Error::Ok) {
        std::cerr << "Failed to set cache_key option" << std::endl;
        return 1;
      }
    }

    if (module.load_method(
            "forward",
            /*planned_memory=*/nullptr,
            /*event_tracer=*/nullptr,
            off_graph ? &options_map : nullptr) != Error::Ok) {
      std::cerr << "Failed to load forward" << std::endl;
      return 1;
    }
    // Weights-only baseline, so the deltas below isolate the cache.
    const double mem_at_load = ::mlx::core::get_active_memory() / 1048576.0;
    std::cout << "[mem]   after load  : " << mem_at_load << " MiB" << std::endl;

    // Encode. HFTokenizer maps special-token markers in the string to their
    // ids, so the template's <|...|> tokens encode correctly; it already
    // carries <|begin_of_text|>, so pass bos=0 to avoid a doubled BOS.
    std::string enc_input;
    if (!wrap_turn(chat, prompt, /*with_bos=*/true, enc_input)) {
      std::cerr << "Unknown --chat template: " << chat
                << " (expected llama3, gemma, gemma4, or 0)" << std::endl;
      return 1;
    }
    // The template carries its own BOS, so only a raw prompt asks for one.
    const int8_t bos = chat == "0" ? 1 : 0;
    auto enc = tokenizer.encode(enc_input, bos, /*eos=*/0);
    if (!enc.ok()) {
      std::cerr << "Encode failed" << std::endl;
      return 1;
    }
    std::vector<uint64_t> tokens = std::move(*enc);
    const int prompt_len = static_cast<int>(tokens.size());

    // Stop on end-of-text and (for chat) the turn-end token <|eot_id|>.
    std::vector<uint64_t> stop_ids = {tokenizer.eos_tok()};
    std::optional<int64_t> turn_end_id;
    if (chat != "0") {
      const char* turn_end = chat == "llama3" ? "<|eot_id|>"
          : chat == "gemma4"                  ? "<turn|>"
                                              : "<end_of_turn>";
      if (auto eot = tokenizer.piece_to_id(turn_end); eot.ok()) {
        turn_end_id = static_cast<int64_t>(*eot);
        stop_ids.push_back(*eot);
      }
    }
    auto is_stop = [&](int64_t t) {
      for (uint64_t s : stop_ids) {
        if (t == static_cast<int64_t>(s)) {
          return true;
        }
      }
      return false;
    };

    auto step = [&](const std::vector<int64_t>& ids,
                    const std::vector<int64_t>& pos) {
      auto in =
          make_tensor_ptr({1, (int)ids.size()}, std::vector<int64_t>(ids));
      auto cp = make_tensor_ptr({(int)pos.size()}, std::vector<int64_t>(pos));
      auto out = module.execute("forward", {in, cp});
      if (!out.ok()) {
        throw std::runtime_error("execute failed");
      }
      return argmax_last(out->at(0).toTensor());
    };

    // Prefill in chunks, so a ring layer holds window + chunk - 1 slots rather
    // than growing with the prompt. Only the last chunk's token is kept; the
    // earlier ones exist to place their K/V in the cache.
    auto prefill = [&](const std::vector<int64_t>& ids,
                       const std::vector<int64_t>& pos) {
      int64_t next = 0;
      for (size_t off = 0; off < ids.size(); off += chunk) {
        const size_t n = std::min(static_cast<size_t>(chunk), ids.size() - off);
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
      if (!off_graph) {
        std::cerr << "--interactive requires --kv-max-capacity\n";
        return 1;
      }
      auto* control = session->control();
      std::cout << "Multi-turn chat. /reset clears, /undo drops the last turn, "
                   "/undo N drops N tokens, /quit exits.\n";
      int64_t position = 0;
      int64_t turn_start = 0; // position this turn began at, for /undo
      std::string line;
      while (std::cout << "\n> " && std::getline(std::cin, line)) {
        if (line == "/quit") {
          break;
        }
        if (line == "/reset") {
          control->clear();
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
          if (control->rewind(static_cast<int>(target))) {
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
        auto te = tokenizer.encode(turn, /*bos=*/chat == "0" ? 1 : 0, 0);
        if (!te.ok()) {
          std::cerr << "Encode failed\n";
          continue;
        }
        const int n = static_cast<int>(te->size());
        // Admit the turn if its prompt plus one token fits; reserving the whole
        // max_new budget up front would report "full" with most of the cache
        // still free. Generation is then clamped to the room that remains.
        if (!control->can_extend(n + 1)) {
          std::cout << "[cache full: " << position << "/" << control->capacity()
                    << ", turn " << n << " tokens"
                    << (control->can_extend(1) ? "" : ", length at capacity")
                    << ", use /reset]\n";
          continue;
        }
        const int budget = std::min(
            max_new, control->capacity() - static_cast<int>(position) - n);

        turn_start = position;
        std::vector<int64_t> tin(te->begin(), te->end()), tpos;
        for (int i = 0; i < n; ++i) {
          tpos.push_back(position + i);
        }
        int64_t next = prefill(tin, tpos);
        position += n;

        uint64_t prev = te->back();
        for (int i = 0; i < budget && !is_stop(next); ++i) {
          if (auto piece = tokenizer.decode(prev, static_cast<uint64_t>(next));
              piece.ok()) {
            std::cout << *piece << std::flush;
          }
          prev = static_cast<uint64_t>(next);
          next = step({next}, {position});
          ++position;
        }
        // The turn-end token stops generation, so it is neither printed nor
        // fed back -- but the next turn opens without closing this one, and an
        // unterminated assistant turn compounds over a session. Commit it, at
        // the cost of one extra step per turn.
        if (turn_end_id && next == *turn_end_id && control->can_extend(1)) {
          step({next}, {position});
          ++position;
        }
        std::cout << "\n[" << position << "/" << control->capacity()
                  << " tokens"
                  << (budget < max_new ? ", generation capped by capacity" : "")
                  << "]\n";
      }
      return 0;
    }

    std::vector<int64_t> ids(tokens.begin(), tokens.end()), prefill_pos;
    for (int i = 0; i < prompt_len; ++i) {
      prefill_pos.push_back(i);
    }
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    // Sequence length against the configured ceiling, with what MLX actually
    // holds for it. Pools start at initial_capacity and grow by doubling, so
    // the bytes lag the token count in steps; bf16 storage (kv_dtype 15) halves
    // them vs fp32 (6).
    auto print_footprint = [&](const char* when, int len) {
      if (!session) {
        return;
      }
      const int cap = session->control()->capacity();
      const double pct = cap > 0 ? 100.0 * len / cap : 0.0;
      std::cout << "[cache] " << when << ": " << len << " / " << cap
                << " tokens (" << pct << "%)" << std::endl;
      const double mem = ::mlx::core::get_active_memory() / 1048576.0;
      std::cout << "[mem]   " << when << ": " << mem << " MiB (+"
                << (mem - mem_at_load) << " MiB since load)" << std::endl;
    };

    // warmup iters absorb JIT + pool growth; measured iters record tok/s. The
    // off-graph cache is cleared between iters so each prefill starts at length
    // 0 (in-graph overwrites by position, so no reset needed there).
    const int total_iters = warmup + std::max(1, iters);
    std::vector<double> pf_tps, dc_tps;
    for (int iter = 0; iter < total_iters; ++iter) {
      if (iter > 0 && off_graph) {
        session->control()->clear();
      }
      const bool measured = iter >= warmup;
      const bool print_text = (iter == 0);
      // Report the footprint from the last iteration: a measured one, and in
      // steady state once any pool growth has settled.
      const bool print_mem = (iter == total_iters - 1);

      const auto t0 = std::chrono::steady_clock::now();
      int64_t next = prefill(ids, prefill_pos);
      const auto t1 = std::chrono::steady_clock::now();
      if (print_text || print_mem) {
        std::cout << "\n"; // blank line separating this section from the banner
      }
      if (print_mem) {
        print_footprint("after prefill", prompt_len);
      }
      if (print_text) {
        std::cout << "\n"; // blank line before the streamed generation
      }

      uint64_t prev = tokens.back();
      int generated = 0;
      for (int i = 0; i < max_new; ++i) {
        if (is_stop(next)) {
          break;
        }
        if (print_text) {
          if (auto piece = tokenizer.decode(prev, static_cast<uint64_t>(next));
              piece.ok()) {
            std::cout << *piece << std::flush;
          }
        }
        prev = static_cast<uint64_t>(next);
        ++generated;
        next = step({next}, {prompt_len + i});
      }
      const auto t2 = std::chrono::steady_clock::now();
      if (print_text) {
        std::cout << "\n\n"; // close the generation line + blank separator
      }
      if (print_mem) {
        // trailing space aligns the colon with the "after prefill" line above
        print_footprint("after decode ", prompt_len + generated);
      }

      const double pf = ms(t0, t1), dc = ms(t1, t2);
      const double pf_t = prompt_len / (pf / 1000.0);
      const double dc_t = dc > 0 ? generated / (dc / 1000.0) : 0.0;
      std::cout << "\n[iter " << iter << (measured ? "" : " warmup")
                << "] prefill " << pf_t << " tok/s (" << prompt_len << " tok, "
                << pf << " ms) | decode " << dc_t << " tok/s (" << generated
                << " tok, " << dc << " ms)\n";
      if (measured) {
        pf_tps.push_back(pf_t);
        dc_tps.push_back(dc_t);
      }
    }

    auto summarize = [](const char* label, const std::vector<double>& v) {
      double m = 0.0;
      for (double x : v) {
        m += x;
      }
      m /= static_cast<double>(v.size());
      double var = 0.0;
      for (double x : v) {
        var += (x - m) * (x - m);
      }
      const double sd = v.size() > 1
          ? std::sqrt(var / static_cast<double>(v.size() - 1))
          : 0.0;
      std::cout << label << ": " << m << " +/- " << sd
                << " tok/s (n=" << v.size() << ")\n";
    };
    std::cout << "\n";
    summarize("prefill", pf_tps);
    summarize("decode ", dc_tps);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
