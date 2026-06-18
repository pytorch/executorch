/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// EAGLE-3 speculative-decoding runner for ExecuTorch (CUDA/AOTI or MLX
// backend).
//
// Loads the speculator .pte (examples/models/eagle3/export.py). The CUDA export
// exposes three methods that share the target / draft KV caches:
//   prefill(tokens[1,T], pos[T])        -> (next_token[1,1], feat[1,T,H])
//   target_verify(tokens[1,C], pos[C])  -> (greedy_ids[1,C], feat[1,C,H])
//   draft_decode(tokens[1,T], feat[1,T,H], pos[T]) -> (target_ids[1,T],
//   g[1,T,H])
// MLX has no cross-method KV sharing, so prefill+verify are one dynamic-seq
// method and the methods return logits instead of ids:
//   target_forward(tokens[1,T], pos[T]) -> (logits[1,T,V], feat[1,T,H])
//   draft_decode(...) -> (draft_logits[1,T,Vd], g[1,T,H])
// This runner argmaxes those host-side and maps draft ids via d2t (get_d2t).
// feat is the fused (hidden-size) draft feature and H is the draft hidden size.
// Verification is greedy (argmax), so emitted tokens equal greedy target
// decoding (lossless) by construction.
//
// Scheme: the shifted EAGLE convention (vllm/v1/spec_decode/eagle.py,
// set_inputs_first_pass: "Shift the input ids by one token" with unshifted
// hidden_states). The draft pairs target hidden_state_t with token_{t+1}, so a
// new draft chain seeds from the hidden states target_verify already produced
// for the just-confirmed positions plus the corrected token's embedding -- the
// corrected/bonus token never needs its own target forward, giving one target
// forward per round (speedup ~= acceptance length tau).
//
// Features round-trip through the host between method calls (D2H copy + re-feed
// as host tensors). They are small (<= max_prefill x H bf16), so the cost is
// negligible next to the INT4 31B target forward, and it keeps device-tensor
// lifetimes simple.
//
// Run (CUDA: export model.pte + aoti_cuda_blob.ptd and source the CUDA env;
// MLX: a single model.pte, no --data_path):
//   eagle3_speculator_runner --model_path <dir>/model.pte \
//     [--data_path <dir>/aoti_cuda_blob.ptd] --tokenizer_path <tokenizer.json>
//     \
//     --prompt "..." --max_new_tokens 128
// The chat template and stop tokens default to Gemma 4 IT; override
// --chat_prefix/--chat_suffix/--stop_ids/--stop_token (and --bos_id -1) for
// other target/tokenizer pairs. Per-run timing counters (tau, verify/draft ms)
// print at the end.
//
// Scope: a single-sequence, fixed-shape demo runner -- not a generic EAGLE
// serving path. CUDA is greedy; MLX adds --temperature (modified rejection
// sampling) but no top-k/p. No batching, grammar/tool constraints, streaming
// API, or integration with the standard ExecuTorch LLM runner. The host feature
// round-trip above is a first-implementation choice (the target forward
// dominates here); a device-resident handoff is future work. The target, draft,
// and tokenizer must be a matched, co-trained set -- a mismatch can pass export
// and silently degrade acceptance/output.

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/types.h>
extern "C" void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  if (level == 'D' || level == 'I') {
    return;
  }
  fprintf(stderr, "%c [%s:%zu] %s\n", (char)level, filename, line, message);
}

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#endif

DEFINE_string(model_path, "", "Speculator model.pte path.");
DEFINE_string(data_path, "", "Tensor data (.ptd) path for the CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Explain why the sky is blue.", "Prompt text.");
DEFINE_bool(raw_prompt, false, "Skip the Gemma 4 IT chat template.");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");
DEFINE_int32(bos_id, 2, "BOS token id (-1 to skip; Gemma convention: 2).");
DEFINE_int32(eos_id, 1, "EOS token id (Gemma convention: 1).");
DEFINE_bool(
    cuda_graph,
    true,
    "Capture target_verify as a CUDA graph (CUDA only).");
// Chat template + stop tokens default to Gemma 4 IT; override for other models.
DEFINE_string(
    chat_prefix,
    "<|turn>user\n",
    "Chat-template text before the prompt.");
DEFINE_string(
    chat_suffix,
    "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    "Chat-template text after the prompt.");
DEFINE_string(
    stop_ids,
    "1,50,106",
    "Comma-separated extra stop token ids (empty to add none).");
DEFINE_string(
    stop_token,
    "<turn|>",
    "A stop-delimiter string to encode and add to EOS (empty to skip).");
DEFINE_double(
    temperature,
    0.0,
    "Sampling temperature (0 = greedy/argmax; >0 = modified rejection "
    "sampling; MLX only).");
DEFINE_int64(seed, 0, "RNG seed for --temperature sampling.");
DEFINE_int64(
    chain,
    0,
    "Override draft chain length K (MLX only; 0 = use the exported "
    "get_chain_len). Ignored on CUDA, whose verify shape is static.");

using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
namespace llm = executorch::extension::llm;
using SizesType = executorch::aten::SizesType;

namespace {

// D2H-copy a tensor's raw bytes into a host buffer (the AOTI backend returns
// device tensors). Works for any dtype; caller reinterprets.
std::vector<uint8_t> to_host_bytes(const executorch::aten::Tensor& t) {
  std::vector<uint8_t> out(t.nbytes());
  const void* ptr = t.const_data_ptr();
#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attrs{};
  if (cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice) {
    cudaError_t err =
        cudaMemcpy(out.data(), ptr, out.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      ET_LOG(Error, "D2H copy failed: %s", cudaGetErrorString(err));
      exit(1);
    }
    return out;
  }
#endif
  memcpy(out.data(), ptr, out.size());
  return out;
}

// Read an int64 (1, N) tensor to a host vector.
std::vector<int64_t> read_ids(const executorch::aten::Tensor& t) {
  auto bytes = to_host_bytes(t);
  size_t n = bytes.size() / sizeof(int64_t);
  std::vector<int64_t> ids(n);
  memcpy(ids.data(), bytes.data(), bytes.size());
  return ids;
}

// A draft feature held on the host as raw bf16 (uint16) so it can be re-fed.
struct HostFeature {
  std::vector<uint16_t> data; // row-major (T, H)
  int64_t T = 0;
  int64_t H = 0;
};

HostFeature read_feature(const executorch::aten::Tensor& t) {
  // t is (1, T, H) bf16.
  HostFeature f;
  f.T = t.size(1);
  f.H = t.size(2);
  auto bytes = to_host_bytes(t);
  f.data.resize(bytes.size() / sizeof(uint16_t));
  memcpy(f.data.data(), bytes.data(), bytes.size());
  return f;
}

#ifndef EXECUTORCH_BUILD_CUDA
// MLX returns logits; read one element of a (1, T, V) Float or BFloat16 row.
inline float
logit_at(const uint8_t* row, int64_t v, executorch::aten::ScalarType st) {
  if (st == executorch::aten::ScalarType::BFloat16) {
    uint32_t u =
        static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(row)[v]) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
  }
  return reinterpret_cast<const float*>(row)[v];
}

// Per-position argmax over a (1, T, V) logits tensor -> T ids.
std::vector<int64_t> argmax_rows(const executorch::aten::Tensor& t) {
  const int64_t T = t.size(1), V = t.size(2);
  const auto st = t.scalar_type();
  const int64_t esz = t.numel() > 0 ? (int64_t)(t.nbytes() / t.numel()) : 0;
  auto bytes = to_host_bytes(t);
  std::vector<int64_t> ids(T);
  for (int64_t r = 0; r < T; r++) {
    const uint8_t* row = bytes.data() + (size_t)r * V * esz;
    int64_t best = 0;
    float bv = logit_at(row, 0, st);
    for (int64_t v = 1; v < V; v++) {
      float x = logit_at(row, v, st);
      if (x > bv) {
        bv = x;
        best = v;
      }
    }
    ids[r] = best;
  }
  return ids;
}

// Numerically-stable softmax in place (same max-subtract approach as the static
// softmax in extension/llm/sampler/sampler.cpp). Apply temperature by scaling
// logits first (see temperature_softmax).
void softmax_inplace(std::vector<float>& x) {
  if (x.empty()) {
    return;
  }
  float max_val = *std::max_element(x.begin(), x.end());
  float sum = 0.0f;
  for (float& v : x) {
    v = std::exp(v - max_val);
    sum += v;
  }
  for (float& v : x) {
    v /= sum;
  }
}

// Temperature-scale logits then softmax (mirrors Sampler::sample: logits *=
// inv_temperature; softmax(...)).
void temperature_softmax(std::vector<float>& x, float inv_temp) {
  for (float& v : x) {
    v *= inv_temp;
  }
  softmax_inplace(x);
}

// Multinomial sample from a probability vector (mirrors Sampler::sample_mult:
// walk the CDF and return the first index past the coin).
int64_t sample_mult(const std::vector<float>& p, std::mt19937_64& rng) {
  float coin = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
  float cdf = 0.0f;
  for (size_t i = 0; i < p.size(); i++) {
    cdf += p[i];
    if (coin < cdf) {
      return static_cast<int64_t>(i);
    }
  }
  return static_cast<int64_t>(p.size()) - 1;
}

// Copy row r of a (1, T, V) Float/BFloat16 logits tensor to float[V].
std::vector<float> row_to_floats(const executorch::aten::Tensor& t, int64_t r) {
  const int64_t V = t.size(2);
  const auto st = t.scalar_type();
  const int64_t esz = t.numel() > 0 ? (int64_t)(t.nbytes() / t.numel()) : 0;
  auto bytes = to_host_bytes(t);
  const uint8_t* row = bytes.data() + (size_t)r * V * esz;
  std::vector<float> out(V);
  for (int64_t v = 0; v < V; v++) {
    out[v] = logit_at(row, v, st);
  }
  return out;
}
#endif

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_path.empty() || FLAGS_tokenizer_path.empty()) {
    ET_LOG(Error, "Must specify --model_path and --tokenizer_path");
    return 1;
  }

  llm::Stats stats;
  stats.model_load_start_ms = llm::time_in_ms();

  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (tokenizer->load(FLAGS_tokenizer_path) != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }
  auto module = std::make_unique<Module>(
      FLAGS_model_path,
      data_files,
      Module::LoadMode::MmapUseMlockIgnoreErrors,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/true);

  // Weight sharing across methods (prefill and target_verify share the target).
#ifdef EXECUTORCH_BUILD_CUDA
  {
    executorch::runtime::BackendOptions<1> backend_options;
    backend_options.set_option("weight_sharing_across_methods", true);
    executorch::runtime::set_option("CudaBackend", backend_options.view());
  }
  if (FLAGS_cuda_graph) {
    // target_verify is the one target forward per round and has a static shape
    // (chain+1 tokens), so capture it as a CUDA graph to avoid paying the
    // 60-layer per-kernel launch overhead every round (the dominant cost
    // otherwise). Its input tensors must wrap stable host buffers (below).
    executorch::runtime::BackendOptions<1> g;
    g.set_option("enable_cuda_graph_for_method", "target_verify");
    executorch::runtime::set_option("CudaBackend", g.view());
  }
#endif

#ifdef EXECUTORCH_BUILD_CUDA
  for (const char* m : {"prefill", "target_verify", "draft_decode"}) {
#else
  for (const char* m : {"target_forward", "draft_decode"}) {
#endif
    if (module->load_method(m) != Error::Ok) {
      ET_LOG(Error, "Failed to load method %s", m);
      return 1;
    }
  }

  if (FLAGS_max_new_tokens <= 0) {
    ET_LOG(Error, "--max_new_tokens must be >= 1");
    return 1;
  }

  // Metadata baked in by export.py (required: a missing key means a
  // mismatched/old .pte, so fail loudly instead of guessing).
  auto meta = [&](const char* name) -> int64_t {
    auto r = module->get(name);
    if (!r.ok()) {
      ET_LOG(Error, "missing required .pte metadata: %s", name);
      exit(1);
    }
    return r->toScalar().to<int64_t>();
  };
  const int64_t chain_len = meta("get_chain_len");
  const int64_t max_prefill = meta("get_max_prefill_chunk");
  const int64_t min_prefill = meta("get_min_prefill_chunk");
  const int64_t max_seq_len = meta("get_max_seq_len");
  int64_t K = chain_len;
#ifndef EXECUTORCH_BUILD_CUDA
  // MLX methods are dynamic-seq, so the chain can be set at runtime as long as
  // the verify window (K+1) fits the exported prefill range.
  if (FLAGS_chain > 0) {
    K = FLAGS_chain;
  }
  if (K + 1 > max_prefill) {
    ET_LOG(
        Error,
        "--chain %" PRId64 " (verify window %" PRId64
        ") exceeds exported max_prefill %" PRId64,
        K,
        K + 1,
        max_prefill);
    return 1;
  }
#endif

#ifndef EXECUTORCH_BUILD_CUDA
  // MLX draft_decode returns draft-vocab logits; load the draft->target map
  // (target_id = draft_id + d2t[draft_id]) baked in by export.py.
  std::vector<int64_t> d2t;
  {
    auto r = module->get("get_d2t");
    if (!r.ok()) {
      ET_LOG(Error, "missing get_d2t metadata in .pte");
      return 1;
    }
    d2t = read_ids(r->toTensor());
  }
  const float temp =
      FLAGS_temperature > 0.0 ? static_cast<float>(FLAGS_temperature) : 0.0f;
  const float inv_temp = temp > 0.0f ? 1.0f / temp : 0.0f;
  std::mt19937_64 rng(static_cast<uint64_t>(FLAGS_seed));
#endif

  // EOS: tokenizer/metadata ids, the configured eos, any --stop_ids, and the
  // encoded --stop_token delimiter (all default to the Gemma 4 IT conventions).
  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());
  eos_ids.insert(static_cast<uint64_t>(FLAGS_eos_id));
  for (size_t b = 0, e; b <= FLAGS_stop_ids.size(); b = e + 1) {
    e = FLAGS_stop_ids.find(',', b);
    if (e == std::string::npos) {
      e = FLAGS_stop_ids.size();
    }
    std::string tok = FLAGS_stop_ids.substr(b, e - b);
    if (!tok.empty()) {
      eos_ids.insert(static_cast<uint64_t>(std::stoll(tok)));
    }
  }
  if (!FLAGS_stop_token.empty()) {
    if (auto t = tokenizer->encode(FLAGS_stop_token, /*bos=*/0, /*eos=*/0);
        t.ok() && t->size() == 1) {
      eos_ids.insert(t.get()[0]);
    }
  }

  std::string prompt_text = FLAGS_prompt;
  if (!FLAGS_raw_prompt) {
    prompt_text = FLAGS_chat_prefix + prompt_text + FLAGS_chat_suffix;
  }
  auto enc = tokenizer->encode(prompt_text);
  if (!enc.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  std::vector<int64_t> prompt(enc->begin(), enc->end());
  if (FLAGS_bos_id >= 0) {
    prompt.insert(prompt.begin(), static_cast<int64_t>(FLAGS_bos_id));
  }
  const int64_t L = static_cast<int64_t>(prompt.size());
  // The runner does not chunk: the whole prompt must fit one prefill, and its
  // length must be within the exported prefill range [min_prefill,
  // max_prefill].
  if (L > max_prefill) {
    ET_LOG(
        Error,
        "Prompt (%" PRId64 " tokens) exceeds max_prefill %" PRId64
        "; this runner does not chunk prefill.",
        L,
        max_prefill);
    return 1;
  }
  if (L < min_prefill) {
    ET_LOG(
        Error,
        "Prompt (%" PRId64
        " tokens) is below the exported prefill "
        "minimum %" PRId64 "; use a longer prompt.",
        L,
        min_prefill);
    return 1;
  }
  // The prefill bonus token is always emittable (no KV write past the prompt).
  // Each speculative round, however, writes a K-token verify window, so it
  // needs anchor_pos + K <= max_seq_len - 1 (enforced in the loop below). Cap
  // the total at the positions available; max_new >= 1 since L <= max_prefill <
  // max_seq_len.
  int64_t max_new = std::min<int64_t>(FLAGS_max_new_tokens, max_seq_len - L);
  printf(
      "Prompt tokens: %" PRId64 ", chain K=%" PRId64 ", max_new=%" PRId64 "\n",
      L,
      K,
      max_new);

  auto S = [](int64_t v) { return static_cast<SizesType>(v); };

  // Persistent host buffers backing the tensors handed to each execute() call.
  std::vector<int64_t> tok_buf, pos_buf;
  std::vector<uint16_t> feat_buf;

  auto long_tensor = [&](std::vector<int64_t>& buf) {
    return from_blob(
        buf.data(),
        {1, S((int64_t)buf.size())},
        executorch::aten::ScalarType::Long);
  };
  auto pos_tensor = [&](std::vector<int64_t>& buf) {
    return from_blob(
        buf.data(),
        {S((int64_t)buf.size())},
        executorch::aten::ScalarType::Long);
  };

  // draft_decode over (tokens, feat rows, positions); returns proposals + the
  // last row of g (the recurrent feature for the next chain step).
  auto draft_decode = [&](const std::vector<int64_t>& tokens,
                          const uint16_t* feat_rows,
                          int64_t feat_T,
                          int64_t H,
                          int64_t start_pos,
                          std::vector<int64_t>& out_ids,
                          HostFeature& out_last_g) {
    tok_buf.assign(tokens.begin(), tokens.end());
    pos_buf.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
      pos_buf[i] = start_pos + static_cast<int64_t>(i);
    }
    feat_buf.assign(feat_rows, feat_rows + feat_T * H);
    auto t_tok = long_tensor(tok_buf);
    auto t_feat = from_blob(
        feat_buf.data(),
        {1, S(feat_T), S(H)},
        executorch::aten::ScalarType::BFloat16);
    auto t_pos = pos_tensor(pos_buf);
    auto r = module->execute(
        "draft_decode", {EValue(t_tok), EValue(t_feat), EValue(t_pos)});
    if (r.error() != Error::Ok) {
      ET_LOG(Error, "draft_decode failed");
      exit(1);
    }
#ifdef EXECUTORCH_BUILD_CUDA
    out_ids = read_ids(r->at(0).toTensor());
#else
    // draft-vocab argmax -> target ids via d2t.
    out_ids = argmax_rows(r->at(0).toTensor());
    for (auto& id : out_ids) {
      id += d2t[id];
    }
#endif
    HostFeature g = read_feature(r->at(1).toTensor());
    out_last_g.T = 1;
    out_last_g.H = g.H;
    out_last_g.data.assign(g.data.end() - g.H, g.data.end()); // last row of g
  };

  // Run a draft chain seeded by (seed_tokens, seed_feat) at seed positions; the
  // last seeded slot predicts proposal 0, then K-1 recurrent steps.
  auto chain = [&](const std::vector<int64_t>& seed_tokens,
                   const HostFeature& seed_feat,
                   int64_t seed_start_pos) {
    std::vector<int64_t> proposals;
    std::vector<int64_t> ids;
    HostFeature last_g;
    draft_decode(
        seed_tokens,
        seed_feat.data.data(),
        seed_feat.T,
        seed_feat.H,
        seed_start_pos,
        ids,
        last_g);
    proposals.push_back(ids.back());
    int64_t last_pos = seed_start_pos + seed_feat.T - 1;
    for (int64_t k = 1; k < K; k++) {
      std::vector<int64_t> step_ids;
      HostFeature step_g;
      draft_decode(
          {proposals.back()},
          last_g.data.data(),
          1,
          last_g.H,
          last_pos + k,
          step_ids,
          step_g);
      proposals.push_back(step_ids[0]);
      last_g = step_g;
    }
    return proposals;
  };

#ifndef EXECUTORCH_BUILD_CUDA
  // Run draft_decode and return the draft-vocab probabilities for the last
  // (predicting) row plus the recurrent feature; the sampling counterpart of
  // draft_decode that keeps the distribution for rejection sampling.
  auto draft_logits_step = [&](const std::vector<int64_t>& tokens,
                               const uint16_t* feat_rows,
                               int64_t feat_T,
                               int64_t Hh,
                               int64_t start_pos,
                               HostFeature& out_last_g) -> std::vector<float> {
    tok_buf.assign(tokens.begin(), tokens.end());
    pos_buf.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
      pos_buf[i] = start_pos + static_cast<int64_t>(i);
    }
    feat_buf.assign(feat_rows, feat_rows + feat_T * Hh);
    auto t_tok = long_tensor(tok_buf);
    auto t_feat = from_blob(
        feat_buf.data(),
        {1, S(feat_T), S(Hh)},
        executorch::aten::ScalarType::BFloat16);
    auto t_pos = pos_tensor(pos_buf);
    auto r = module->execute(
        "draft_decode", {EValue(t_tok), EValue(t_feat), EValue(t_pos)});
    if (r.error() != Error::Ok) {
      ET_LOG(Error, "draft_decode failed");
      exit(1);
    }
    auto dl = r->at(0).toTensor();
    std::vector<float> q = row_to_floats(dl, dl.size(1) - 1);
    temperature_softmax(q, inv_temp);
    HostFeature g = read_feature(r->at(1).toTensor());
    out_last_g.T = 1;
    out_last_g.H = g.H;
    out_last_g.data.assign(g.data.end() - g.H, g.data.end());
    return q;
  };

  // Sampling counterpart of `chain`: sample each proposal from the draft and
  // record (draft_id, q) for the rejection test.
  auto chain_sample = [&](const std::vector<int64_t>& seed_tokens,
                          const HostFeature& seed_feat,
                          int64_t seed_start_pos,
                          std::vector<int64_t>& draft_ids,
                          std::vector<std::vector<float>>& q_rows) {
    std::vector<int64_t> proposals;
    HostFeature last_g;
    std::vector<float> q = draft_logits_step(
        seed_tokens,
        seed_feat.data.data(),
        seed_feat.T,
        seed_feat.H,
        seed_start_pos,
        last_g);
    int64_t d = sample_mult(q, rng);
    draft_ids.push_back(d);
    proposals.push_back(d + d2t[d]);
    q_rows.push_back(std::move(q));
    int64_t last_pos = seed_start_pos + seed_feat.T - 1;
    for (int64_t k = 1; k < K; k++) {
      HostFeature step_g;
      std::vector<float> qk = draft_logits_step(
          {proposals.back()},
          last_g.data.data(),
          1,
          last_g.H,
          last_pos + k,
          step_g);
      int64_t dk = sample_mult(qk, rng);
      draft_ids.push_back(dk);
      proposals.push_back(dk + d2t[dk]);
      q_rows.push_back(std::move(qk));
      last_g = step_g;
    }
    return proposals;
  };
#endif

  stats.model_load_end_ms = llm::time_in_ms();
  stats.inference_start_ms = stats.model_load_end_ms;

  // --- Prefill: target over the prompt -> bonus token + per-position feature.
  // ---
  tok_buf = prompt;
  pos_buf.resize(L);
  for (int64_t i = 0; i < L; i++) {
    pos_buf[i] = i;
  }
#ifdef EXECUTORCH_BUILD_CUDA
  auto pf = module->execute(
      "prefill", {EValue(long_tensor(tok_buf)), EValue(pos_tensor(pos_buf))});
#else
  auto pf = module->execute(
      "target_forward",
      {EValue(long_tensor(tok_buf)), EValue(pos_tensor(pos_buf))});
#endif
  if (pf.error() != Error::Ok) {
    ET_LOG(Error, "prefill failed");
    return 1;
  }
#ifdef EXECUTORCH_BUILD_CUDA
  int64_t anchor =
      read_ids(pf->at(0).toTensor())[0]; // bonus token at position L
#else
  // target_forward returns per-position logits; the bonus is the last argmax.
  int64_t anchor = argmax_rows(pf->at(0).toTensor()).back();
#endif
  HostFeature feat_prompt = read_feature(pf->at(1).toTensor());
  const int64_t H = feat_prompt.H;
  int64_t anchor_pos = L;

  stats.prompt_eval_end_ms = llm::time_in_ms();
  stats.first_token_ms = stats.prompt_eval_end_ms;

  std::vector<int64_t> emitted = {anchor};
  uint64_t prev = static_cast<uint64_t>(prompt.back());
  {
    auto s = tokenizer->decode(prev, static_cast<uint64_t>(anchor));
    if (s.ok()) {
      printf("%s", s->c_str());
      fflush(stdout);
    }
    prev = static_cast<uint64_t>(anchor);
  }

  // We only run the speculative loop if more than the (already emitted) prefill
  // bonus is wanted, the bonus wasn't EOS, and there is room for a K-token
  // verify window. Otherwise we are done -- no draft seeding needed.
  bool hit_eos = eos_ids.count(static_cast<uint64_t>(anchor)) > 0;
  bool speculate = max_new > 1 && !hit_eos && anchor_pos + K <= max_seq_len - 1;
  std::vector<int64_t> proposals;
#ifndef EXECUTORCH_BUILD_CUDA
  // Per-round draft distributions kept for --temperature rejection sampling.
  std::vector<int64_t> draft_ids;
  std::vector<std::vector<float>> q_rows;
#endif
  if (speculate) {
    // Seed the first chain (shifted): draft slot p pairs feat_prompt[p] with
    // token_{p+1}; the last slot pairs feat_prompt[L-1] with the bonus and
    // predicts position L+1.
    std::vector<int64_t> seed_tokens(prompt.begin() + 1, prompt.end());
    seed_tokens.push_back(anchor);
#ifdef EXECUTORCH_BUILD_CUDA
    proposals = chain(seed_tokens, feat_prompt, 0);
#else
    proposals = (temp == 0.0f)
        ? chain(seed_tokens, feat_prompt, 0)
        : chain_sample(seed_tokens, feat_prompt, 0, draft_ids, q_rows);
#endif
  }

  // Stable buffers for target_verify (fixed length K+1) so the CUDA graph
  // replays against the same input addresses; we mutate the contents in place.
  std::vector<int64_t> vtok_buf(K + 1), vpos_buf(K + 1);
  auto vtok_t = from_blob(
      vtok_buf.data(), {1, S(K + 1)}, executorch::aten::ScalarType::Long);
  auto vpos_t = from_blob(
      vpos_buf.data(), {S(K + 1)}, executorch::aten::ScalarType::Long);

  // --- Speculative rounds: one target forward (target_verify) per round. ---
  int64_t rounds = 0;
  int64_t verify_ms = 0, draft_ms = 0; // instrumentation
  while (speculate && (int64_t)emitted.size() < max_new && !hit_eos &&
         anchor_pos + K <= max_seq_len - 1) {
    rounds++;
    // Verify [anchor, p0..p_{K-1}] at positions [anchor_pos .. anchor_pos+K].
    vtok_buf[0] = anchor;
    for (int64_t j = 0; j < K; j++) {
      vtok_buf[j + 1] = proposals[j];
    }
    for (int64_t i = 0; i <= K; i++) {
      vpos_buf[i] = anchor_pos + i;
    }
    int64_t t_v = llm::time_in_ms();
#ifdef EXECUTORCH_BUILD_CUDA
    auto vr =
        module->execute("target_verify", {EValue(vtok_t), EValue(vpos_t)});
#else
    auto vr =
        module->execute("target_forward", {EValue(vtok_t), EValue(vpos_t)});
#endif
    if (vr.error() != Error::Ok) {
      ET_LOG(Error, "target forward failed");
      return 1;
    }
    HostFeature verify_feat = read_feature(vr->at(1).toTensor());
    verify_ms += llm::time_in_ms() - t_v;

    // Acceptance: count the leading proposals that pass, then a corrected
    // token. verify slot j is the target distribution after token j (proposal j
    // sits at position anchor_pos+1+j).
    int64_t a = 0;
    int64_t corrected = 0;
#ifdef EXECUTORCH_BUILD_CUDA
    std::vector<int64_t> verify_ids = read_ids(vr->at(0).toTensor());
    for (int64_t j = 0; j < K; j++) {
      if (proposals[j] == verify_ids[j]) {
        a++;
      } else {
        break;
      }
    }
    corrected = verify_ids[a];
#else
    auto verify_logits = vr->at(0).toTensor();
    if (temp == 0.0f) {
      // Greedy acceptance against the per-position argmax.
      std::vector<int64_t> verify_ids = argmax_rows(verify_logits);
      for (int64_t j = 0; j < K; j++) {
        if (proposals[j] == verify_ids[j]) {
          a++;
        } else {
          break;
        }
      }
      corrected = verify_ids[a];
    } else {
      // Modified rejection sampling (lossless w.r.t. target sampling): accept
      // proposal j with prob min(1, p_j[x]/q_j[d]); on reject resample from the
      // residual (p - q)_+ over the target vocab; the all-accepted bonus is a
      // sample from p_K.
      std::uniform_real_distribution<double> u(0.0, 1.0);
      bool rejected = false;
      for (int64_t j = 0; j < K; j++) {
        std::vector<float> p = row_to_floats(verify_logits, j);
        temperature_softmax(p, inv_temp);
        const int64_t x = proposals[j], d = draft_ids[j];
        const float qx = q_rows[j][d];
        const float ratio = qx > 0.0f ? std::min(1.0f, p[x] / qx) : 0.0f;
        if (u(rng) <= ratio) {
          a++;
          continue;
        }
        // residual: subtract q mapped to the target vocab, clamp, renormalize.
        const std::vector<float>& q = q_rows[j];
        for (int64_t dd = 0; dd < (int64_t)q.size(); dd++) {
          p[dd + d2t[dd]] -= q[dd];
        }
        double sum = 0.0;
        for (float& pv : p) {
          if (pv < 0.0f) {
            pv = 0.0f;
          }
          sum += pv;
        }
        for (float& pv : p) {
          pv = static_cast<float>(pv / sum);
        }
        corrected = sample_mult(p, rng);
        rejected = true;
        break;
      }
      if (!rejected) {
        std::vector<float> p = row_to_floats(verify_logits, K);
        temperature_softmax(p, inv_temp);
        corrected = sample_mult(p, rng);
      }
    }
#endif

    std::vector<int64_t> newly(proposals.begin(), proposals.begin() + a);
    newly.push_back(corrected);
    for (int64_t t : newly) {
      if ((int64_t)emitted.size() >= max_new)
        break;
      emitted.push_back(t);
      auto s = tokenizer->decode(prev, static_cast<uint64_t>(t));
      if (s.ok()) {
        printf("%s", s->c_str());
        fflush(stdout);
      }
      prev = static_cast<uint64_t>(t);
      if (eos_ids.count(static_cast<uint64_t>(t)) > 0) {
        // Stop at the first accepted EOS; do not emit the rest of this batch.
        // An accepted proposal (not just the corrected/bonus token) can be EOS,
        // so this truncates newly at the first stop token, matching the eager
        // reference.
        hit_eos = true;
        break;
      }
    }
    if (hit_eos || (int64_t)emitted.size() >= max_new)
      break;

    // Reseed the draft (shifted): slot anchor_pos+i holds (verify_feat[i],
    // token_{anchor_pos+i+1}) where token = p_i (i<a) / corrected (i=a). The
    // last slot predicts the next chain's proposal 0. verify_feat already holds
    // the target hidden states for these positions -- no extra target forward.
    std::vector<int64_t> reseed_tokens(
        proposals.begin(), proposals.begin() + a);
    reseed_tokens.push_back(corrected);
    HostFeature reseed_feat;
    reseed_feat.T = a + 1;
    reseed_feat.H = H;
    reseed_feat.data.assign(
        verify_feat.data.begin(), verify_feat.data.begin() + (a + 1) * H);
    int64_t t_d = llm::time_in_ms();
#ifdef EXECUTORCH_BUILD_CUDA
    proposals = chain(reseed_tokens, reseed_feat, anchor_pos);
#else
    if (temp == 0.0f) {
      proposals = chain(reseed_tokens, reseed_feat, anchor_pos);
    } else {
      draft_ids.clear();
      q_rows.clear();
      proposals = chain_sample(
          reseed_tokens, reseed_feat, anchor_pos, draft_ids, q_rows);
    }
#endif
    draft_ms += llm::time_in_ms() - t_d;
    anchor = corrected;
    anchor_pos = anchor_pos + 1 + a;
  }
  printf("\n");
  printf(
      "[timing] verify=%" PRId64 "ms draft=%" PRId64 "ms over %" PRId64
      " rounds (%.1f / %.1f ms per round)\n",
      verify_ms,
      draft_ms,
      rounds,
      rounds ? (double)verify_ms / rounds : 0.0,
      rounds ? (double)draft_ms / rounds : 0.0);

  stats.inference_end_ms = llm::time_in_ms();
  stats.num_prompt_tokens = L;
  stats.num_generated_tokens = static_cast<int64_t>(emitted.size());

#ifdef EXECUTORCH_BUILD_CUDA
  cudaDeviceSynchronize();
#endif

  // tau = mean tokens emitted per verify round; emitted[0] is the free prefill
  // bonus (not produced by a round), so exclude it.
  double tau = rounds ? static_cast<double>(emitted.size() - 1) / rounds : 0.0;
  double gen_s = (stats.inference_end_ms - stats.prompt_eval_end_ms) / 1000.0;
  printf("\n--- EAGLE-3 speculative decode ---\n");
  printf(
      "generated %zu tokens in %" PRId64 " rounds (tau=%.3f)\n",
      emitted.size(),
      rounds,
      tau);
  if (gen_s > 0) {
    printf("decode: %.2f tok/s (%.3f s)\n", emitted.size() / gen_s, gen_s);
  }
  llm::print_report(stats);
  return 0;
}
