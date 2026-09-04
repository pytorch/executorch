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
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>

#include <pytorch/tokenizers/tokenizer.h>

#include <gflags/gflags.h>
#include <mlx/memory.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
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
    "bf16",
    "Off-graph: KV storage dtype, bf16|fp16|fp32.");
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
// entry per cache in get_kv_heads / get_head_dims / get_windows (0 = flat),
// plus get_prefill_chunk_size, which the export validates against the sliding
// window and which becomes max_write -- the largest step the cache may see.
// Capacity and dtype stay with the flags. False means this is not an off-graph
// model.
bool read_kv_layout(Module& module, cache::CacheConfig& cfg) {
  const auto n_caches = const_int(module, "get_n_caches");
  const auto kv_heads = const_ints(module, "get_kv_heads");
  const auto head_dims = const_ints(module, "get_head_dims");
  const auto windows = const_ints(module, "get_windows");
  const auto chunk = const_int(module, "get_prefill_chunk_size");
  if (!n_caches || !kv_heads || !head_dims || !windows || !chunk) {
    return false;
  }
  cfg.max_write = static_cast<int>(*chunk);
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
  const float temperature = static_cast<float>(FLAGS_temperature);
  const int initial_capacity = FLAGS_kv_initial_capacity;
  const bool interactive = FLAGS_interactive;
  const bool warmup = FLAGS_warmup;
  if (pte.empty() || tok_path.empty()) {
    std::cerr << "Required: --pte <file> --tokenizer <file>  "
                 "[--kv-max-capacity N for off-graph models]\n";
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
    // Tokens per prefill step, from the .pte. 0 means one step: an
    // in-graph model publishes no chunk and has no ring to bound.
    int prefill_chunk = 0;

    // Load the program but not forward: the cache must exist before forward's
    // backend init reads its key, and the layout it needs is published by
    // constant methods in the same file.
    Module module(pte);
    const long load_start_ms = ::executorch::extension::llm::time_in_ms();
    if (module.load() != Error::Ok) {
      std::cerr << "Failed to load " << pte << std::endl;
      return 1;
    }

    // Everything past load_method is identical for both model kinds; only
    // setup differs. ctl is null for an in-graph model, which owns its cache
    // inside the graph and exposes no control face.
    auto run =
        [&](cache::SequenceControl* ctl,
            const ::executorch::runtime::LoadBackendOptionsMap* load_opts)
        -> int {
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
      auto enc = tokenizer->encode(enc_input, bos, /*eos=*/0);
      if (!enc.ok()) {
        std::cerr << "Encode failed" << std::endl;
        return 1;
      }
      std::vector<uint64_t> tokens = std::move(*enc);
      const int prompt_len = static_cast<int>(tokens.size());

      // End-of-text from the model's metadata when it publishes any, else the
      // tokenizer's. The turn-end token is ours: it depends on --chat, which
      // the .pte knows nothing about.
      std::unordered_set<uint64_t> stop_ids =
          ::executorch::extension::llm::get_eos_ids(tokenizer.get(), &module);
      std::optional<int64_t> turn_end_id;
      if (chat != "0") {
        const char* turn_end = chat == "llama3" ? "<|eot_id|>"
            : chat == "gemma4"                  ? "<turn|>"
                                                : "<end_of_turn>";
        if (auto eot = tokenizer->piece_to_id(turn_end); eot.ok()) {
          turn_end_id = static_cast<int64_t>(*eot);
          stop_ids.insert(*eot);
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
        auto out = module.execute("forward", {in, cp});
        if (!out.ok()) {
          throw std::runtime_error("execute failed");
        }
        const auto& logits = out->at(0).toTensor();
        if (!sampler) {
          sampler.emplace(
              static_cast<int32_t>(logits.size(logits.dim() - 1)), temperature);
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
        const size_t step_size =
            prefill_chunk > 0 ? static_cast<size_t>(prefill_chunk) : ids.size();
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
        auto* control = ctl;
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
          auto te = tokenizer->encode(turn, /*bos=*/chat == "0" ? 1 : 0, 0);
          if (!te.ok()) {
            std::cerr << "Encode failed\n";
            continue;
          }
          const int n = static_cast<int>(te->size());
          // Admit the turn if its prompt plus one token fits; reserving the
          // whole max_new budget up front would report "full" with most of the
          // cache still free. Generation is then clamped to the room that
          // remains.
          if (!control->can_extend(n + 1)) {
            std::cout << "[cache full: " << position << "/"
                      << control->capacity() << ", turn " << n << " tokens"
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
            if (auto piece =
                    tokenizer->decode(prev, static_cast<uint64_t>(next));
                piece.ok()) {
              std::cout << *piece << std::flush;
            }
            prev = static_cast<uint64_t>(next);
            next = step({next}, {position});
            ++position;
          }
          // The turn-end token stops generation, so it is neither printed nor
          // fed back -- but the next turn opens without closing this one, and
          // an unterminated assistant turn compounds over a session. Commit it,
          // at the cost of one extra step per turn.
          if (turn_end_id && next == *turn_end_id && control->can_extend(1)) {
            step({next}, {position});
            ++position;
          }
          std::cout << "\n[" << position << "/" << control->capacity()
                    << " tokens"
                    << (budget < max_new ? ", generation capped by capacity"
                                         : "")
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

        uint64_t prev = tokens.back();
        int generated = 0;
        for (int i = 0; i < max_new; ++i) {
          if (is_stop(next)) {
            break;
          }
          if (measured) {
            if (auto piece =
                    tokenizer->decode(prev, static_cast<uint64_t>(next));
                piece.ok()) {
              ::executorch::extension::llm::safe_printf(piece->c_str());
              fflush(stdout);
            }
          }
          prev = static_cast<uint64_t>(next);
          ++generated;
          next = step({next}, {prompt_len + i});
        }
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
      return run(/*ctl=*/nullptr, /*load_opts=*/nullptr);
    }

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

    const char* const cache_kind = cache::kind::kSingle;
    auto built = cache::CacheFactory::global().build(
        ::executorch::backends::mlx::kMLXBackendId, cache_kind, cfg);
    if (!built.ok()) {
      std::cerr << "Failed to build cache: " << static_cast<int>(built.error())
                << std::endl;
      return 1;
    }
    const std::shared_ptr<cache::Cache> kv = built.get();
    prefill_chunk = cfg.max_write ? *cfg.max_write : 0;

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

    return run(ctl, &options_map);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
