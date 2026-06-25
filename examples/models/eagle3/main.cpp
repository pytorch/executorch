/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// EAGLE-3 speculative-decoding runner for ExecuTorch (CUDA/AOTI backend).
//
// Loads the speculator .pte (examples/models/eagle3/export.py) exposing three
// methods that share the target / draft KV caches:
//   prefill(tokens[1,T], pos[T])        -> (next_token[1,1], feat[1,T,H])
//   target_verify(tokens[1,C], pos[C], kv_window[V]) -> (greedy_ids[1,C],
//   feat[1,C,H])  -- kv_window's dynamic length V (= valid KV positions =
//   anchor_pos+C) bounds the mid-M SDPA key loop (ignored if mid-M is off).
//   Its growing per-round shape means target_verify can't be a CUDA graph when
//   mid-M is on, so pass --cuda_graph=false there.
//   draft_decode(tokens[1,T], feat[1,T,H], pos[T]) -> (target_ids[1,T],
//   g[1,T,H])
// where feat is the fused (hidden-size) draft feature and H is the draft hidden
// size. Verification is greedy (argmax), so emitted tokens equal greedy target
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
// as host tensors), which keeps device-tensor lifetimes simple. Chunked prefill
// concatenates per-position features for the whole prompt before draft seeding,
// so the host buffer is prompt_len x H bf16 (~672 MiB at 64K context, H=5376),
// scaling with prompt_len rather than max_prefill. That is negligible next to
// the INT4 31B target forward at today's context lengths; stream draft seeding
// as each prefill chunk completes if it becomes a memory/perf concern at larger
// contexts or hidden sizes.
//
// Run (after exporting model.pte + aoti_cuda_blob.ptd via export.py, sourcing
// the CUDA env, and building the eagle3-cuda preset):
//   eagle3_speculator_runner --model_path <dir>/model.pte \
//     --data_path <dir>/aoti_cuda_blob.ptd --tokenizer_path <tokenizer.json> \
//     --prompt "..." --max_new_tokens 128
// The chat template and stop tokens default to Gemma 4 IT; override
// --chat_prefix/--chat_suffix/--stop_ids/--stop_token (and --bos_id -1) for
// other target/tokenizer pairs. Per-run timing counters (tau, verify/draft ms)
// print at the end.
//
// Scope: a single-sequence, greedy, fixed-shape demo runner -- not a generic
// EAGLE serving path. No batching, sampler stack (top-k/p/temperature),
// grammar/ tool constraints, streaming API, or integration with the standard
// ExecuTorch LLM runner. The host feature round-trip above is a
// first-implementation choice (the target forward dominates here); a
// device-resident handoff is future work. The target, draft, and tokenizer must
// be a matched, co-trained set -- a mismatch can pass export and silently
// degrade acceptance/output.

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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
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
DEFINE_string(
    prompt_file,
    "",
    "Read the prompt text from this file instead of --prompt.");
DEFINE_bool(raw_prompt, false, "Skip the Gemma 4 IT chat template.");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");
DEFINE_int32(bos_id, 2, "BOS token id (-1 to skip; Gemma convention: 2).");
DEFINE_int32(eos_id, 1, "EOS token id (Gemma convention: 1).");
DEFINE_bool(
    cuda_graph,
    false,
    "Capture target_verify as a CUDA graph (CUDA only). Off by default: the "
    "current export feeds target_verify a kv_window whose length changes every "
    "round, so capture is unsafe (stale-shape replay). Only enable for an "
    "export whose target_verify inputs all have stable shapes.");
DEFINE_int32(
    chain,
    -1,
    "Override chain length K at runtime (<=0 uses the .pte's get_chain_len). "
    "Requires a dynamic-T verify export; clamped to [1, 7] (verify M=K+1<=8).");
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
      /*temp_allocator=*/nullptr);

  // Weight sharing across methods (prefill and target_verify share the target).
#ifdef EXECUTORCH_BUILD_CUDA
  {
    executorch::runtime::BackendOptions<1> backend_options;
    backend_options.set_option("weight_sharing_across_methods", true);
    executorch::runtime::set_option("CudaBackend", backend_options.view());
  }
  if (FLAGS_cuda_graph) {
    // Opt-in only (default off): capturing target_verify avoids the 60-layer
    // per-kernel launch overhead every round, but it is only sound when every
    // target_verify input has a stable shape across rounds. This export does
    // not satisfy that -- kv_window's length is the per-round valid-KV count
    // (see the kvwin_buf NOTE below) -- so enabling capture here risks stale-
    // shape replay. The flag is kept for a future fixed-shape verify export.
    executorch::runtime::BackendOptions<1> g;
    g.set_option("enable_cuda_graph_for_method", "target_verify");
    executorch::runtime::set_option("CudaBackend", g.view());
  }
#endif

  for (const char* m : {"prefill", "target_verify", "draft_decode"}) {
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
  // Prefill chunks must not exceed the sliding window: a chunk larger than the
  // window overflows the 2*window ring cache across chunk boundaries,
  // truncating sliding attention for the first ~(chunk-window) queries of each
  // chunk (the global flat-cache layers stay exact). Prefer get_sliding_window
  // when the export provides it, else fall back to max_prefill/2.
  int64_t prefill_chunk = max_prefill / 2;
  {
    auto sw = module->get("get_sliding_window");
    if (sw.ok()) {
      prefill_chunk = sw->toScalar().to<int64_t>();
    }
  }
  // Also bound by the exported prefill range: get_max_prefill_chunk is the
  // largest T the prefill kernels were compiled for, which need not be
  // 2*sliding_window (small --max-prefill with a larger window), so cap here.
  prefill_chunk = std::min(prefill_chunk, max_prefill);
  const int64_t min_prefill = meta("get_min_prefill_chunk");
  const int64_t max_seq_len = meta("get_max_seq_len");
  const int64_t K_req = (FLAGS_chain > 0) ? FLAGS_chain : chain_len;
  const int64_t K = (K_req < 1) ? 1 : (K_req > 7 ? 7 : K_req);

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
  if (!FLAGS_prompt_file.empty()) {
    std::ifstream f(FLAGS_prompt_file);
    if (!f) {
      ET_LOG(
          Error, "Failed to open prompt file: %s", FLAGS_prompt_file.c_str());
      return 1;
    }
    prompt_text = std::string(
        std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  }
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
  // A single prefill forward caps at max_prefill (the sliding-ring 2*window
  // limit), so prompts beyond that are looped in <= max_prefill chunks below;
  // the flat global KV cache accumulates across chunks. The prompt only has to
  // fit the exported context (its features then seed the speculative loop).
  if (L >= max_seq_len) {
    ET_LOG(
        Error,
        "Prompt (%" PRId64 " tokens) does not fit max_seq_len %" PRId64,
        L,
        max_seq_len);
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
  // the total at the positions available; max_new >= 1 since L < max_seq_len
  // (L may exceed max_prefill -- the prompt is fed as chunks; L >= max_seq_len
  // is rejected above).
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
    out_ids = read_ids(r->at(0).toTensor());
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

  stats.model_load_end_ms = llm::time_in_ms();
  stats.inference_start_ms = stats.model_load_end_ms;

  // The exported prefill forward accepts T in [min_prefill, max_prefill]; pick
  // the next chunk so the running tail never drops below min_prefill (it would
  // be an out-of-range shape). All but the last one or two chunks are
  // max_prefill.
  auto next_chunk = [&](int64_t done) {
    int64_t remaining = L - done;
    int64_t len = std::min(remaining, prefill_chunk);
    if (remaining - len > 0 && remaining - len < min_prefill) {
      len = remaining - min_prefill;
    }
    return len;
  };

  // --- Prefill: target over the prompt (chunked to respect the prefill cap) ->
  // bonus token + per-position feature. The flat target KV cache accumulates
  // across chunks; the bonus token is the last chunk's output, and the
  // per-position features of every chunk are concatenated to seed the draft.
  HostFeature feat_prompt;
  int64_t anchor = 0;
  int64_t prefill_pos = 0;
  while (prefill_pos < L) {
    int64_t chunk_len = next_chunk(prefill_pos);
    tok_buf.assign(
        prompt.begin() + prefill_pos, prompt.begin() + prefill_pos + chunk_len);
    pos_buf.resize(chunk_len);
    for (int64_t i = 0; i < chunk_len; i++) {
      pos_buf[i] = prefill_pos + i;
    }
    auto pf = module->execute(
        "prefill", {EValue(long_tensor(tok_buf)), EValue(pos_tensor(pos_buf))});
    if (pf.error() != Error::Ok) {
      ET_LOG(Error, "prefill failed at pos %" PRId64, prefill_pos);
      return 1;
    }
    anchor = read_ids(pf->at(0).toTensor())[0]; // bonus token after the prompt
    HostFeature chunk_feat = read_feature(pf->at(1).toTensor());
    feat_prompt.H = chunk_feat.H;
    feat_prompt.T += chunk_feat.T;
    feat_prompt.data.insert(
        feat_prompt.data.end(), chunk_feat.data.begin(), chunk_feat.data.end());
    prefill_pos += chunk_len;
  }
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
  if (speculate) {
    // Seed the first chain (shifted): draft slot p pairs feat_prompt[p] with
    // token_{p+1}; the last slot pairs feat_prompt[L-1] with the bonus and
    // predicts position L+1. Seed in <= max_prefill chunks (draft_decode shares
    // the prefill shape range), each contiguous from the previous so the draft
    // KV cache fills; the last chunk's last row predicts proposal 0 and carries
    // the recurrent feature, then K-1 recurrent steps follow (mirroring chain).
    std::vector<int64_t> seed_tokens(prompt.begin() + 1, prompt.end());
    seed_tokens.push_back(anchor);
    std::vector<int64_t> ids;
    HostFeature last_g;
    for (int64_t seed_pos = 0; seed_pos < L;) {
      int64_t chunk_len = next_chunk(seed_pos);
      std::vector<int64_t> chunk_tokens(
          seed_tokens.begin() + seed_pos,
          seed_tokens.begin() + seed_pos + chunk_len);
      draft_decode(
          chunk_tokens,
          feat_prompt.data.data() + seed_pos * H,
          chunk_len,
          H,
          seed_pos,
          ids,
          last_g);
      seed_pos += chunk_len;
    }
    proposals.push_back(ids.back());
    int64_t last_pos = L - 1;
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
  }

  // Stable buffers for target_verify (fixed length K+1) so the CUDA graph
  // replays against the same input addresses; we mutate the contents in place.
  std::vector<int64_t> vtok_buf(K + 1), vpos_buf(K + 1);
  auto vtok_t = from_blob(
      vtok_buf.data(), {1, S(K + 1)}, executorch::aten::ScalarType::Long);
  auto vpos_t = from_blob(
      vpos_buf.data(), {S(K + 1)}, executorch::aten::ScalarType::Long);
  // kv_window: its dynamic length (= valid KV positions this round) is the
  // mid-M SDPA key bound (ignored if the export has mid-M off). Contents are
  // unused -- only the shape matters -- so one max-size buffer is reused and
  // viewed at the per-round length. NOTE: this per-round shape change is why
  // target_verify can't be captured as a CUDA graph for this export -- hence
  // --cuda_graph defaults to false.
  std::vector<int32_t> kvwin_buf(max_seq_len, 0);

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
    // Valid KV positions after writing this round = [0, anchor_pos+K].
    int64_t valid_len = anchor_pos + K + 1;
    auto kvwin_t = from_blob(
        kvwin_buf.data(), {S(valid_len)}, executorch::aten::ScalarType::Int);
    int64_t t_v = llm::time_in_ms();
    auto vr = module->execute(
        "target_verify", {EValue(vtok_t), EValue(vpos_t), EValue(kvwin_t)});
    if (vr.error() != Error::Ok) {
      ET_LOG(Error, "target_verify failed");
      return 1;
    }
    std::vector<int64_t> verify_ids = read_ids(vr->at(0).toTensor());
    HostFeature verify_feat = read_feature(vr->at(1).toTensor());
    verify_ms += llm::time_in_ms() - t_v;

    // Greedy acceptance: verify_ids[j] is the greedy token after token j, so it
    // checks proposal j (which sits at position anchor_pos+1+j).
    int64_t a = 0;
    for (int64_t j = 0; j < K; j++) {
      if (proposals[j] == verify_ids[j]) {
        a++;
      } else {
        break;
      }
    }
    int64_t corrected = verify_ids[a];

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
    proposals = chain(reseed_tokens, reseed_feat, anchor_pos);
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
