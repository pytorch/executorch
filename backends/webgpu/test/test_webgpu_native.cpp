/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDelegateHeader.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

static bool test_single_add(const std::string& model_path) {
  printf("\n--- Test: single add (1024x1024) ---\n");

  Module module(model_path);
  auto err = module.load_forward();
  if (err != Error::Ok) {
    printf("FAIL: could not load forward method (error %d)\n", (int)err);
    return false;
  }
  printf("Model loaded: %s\n", model_path.c_str());

  constexpr int dim = 1024;
  constexpr int size = dim * dim;

  std::vector<float> a_data(size);
  std::vector<float> b_data(size);
  for (int i = 0; i < size; i++) {
    a_data[i] = static_cast<float>(i) * 1.0f;
    b_data[i] = static_cast<float>(i) * 2.0f;
  }

  auto a = make_tensor_ptr({dim, dim}, std::vector<float>(a_data));
  auto b = make_tensor_ptr({dim, dim}, std::vector<float>(b_data));

  auto result = module.forward({EValue(a), EValue(b)});
  if (!result.ok()) {
    printf("FAIL: forward failed (error %d)\n", (int)result.error());
    return false;
  }

  const auto& outputs = result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    printf("FAIL: no tensor output\n");
    return false;
  }

  const auto& out_tensor = outputs[0].toTensor();
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_error = 0.0f;
  int check_count = std::min(size, 1024);
  for (int i = 0; i < check_count; i++) {
    float expected = a_data[i] + b_data[i];
    float error = std::abs(out_data[i] - expected);
    max_error = std::max(max_error, error);
  }

  printf("Max error: %e (checked %d elements)\n", max_error, check_count);
  if (max_error > 1e-3f) {
    printf("FAIL: max error exceeds tolerance 1e-3\n");
    return false;
  }
  printf("PASS: single add test\n");
  return true;
}

static bool test_chained_add(const std::string& model_path) {
  printf("\n--- Test: chained add (1024x1024, 5 ops) ---\n");

  Module module(model_path);
  auto err = module.load_forward();
  if (err != Error::Ok) {
    printf("FAIL: could not load forward method (error %d)\n", (int)err);
    return false;
  }
  printf("Model loaded: %s\n", model_path.c_str());

  constexpr int dim = 1024;
  constexpr int size = dim * dim;

  std::vector<float> x_data(size);
  std::vector<float> y_data(size);
  for (int i = 0; i < size; i++) {
    x_data[i] = static_cast<float>(i % 100) * 0.01f;
    y_data[i] = static_cast<float>(i % 50) * 0.02f;
  }

  auto x = make_tensor_ptr({dim, dim}, std::vector<float>(x_data));
  auto y = make_tensor_ptr({dim, dim}, std::vector<float>(y_data));

  auto result = module.forward({EValue(x), EValue(y)});
  if (!result.ok()) {
    printf("FAIL: forward failed (error %d)\n", (int)result.error());
    return false;
  }

  const auto& outputs = result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    printf("FAIL: no tensor output\n");
    return false;
  }

  // z=x+y; z=z+x=2x+y; z=z+y=2x+2y; z=z+x=3x+2y; z=z+y=3x+3y
  const auto& out_tensor = outputs[0].toTensor();
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_error = 0.0f;
  for (int i = 0; i < size; i++) {
    float expected = 3.0f * x_data[i] + 3.0f * y_data[i];
    float error = std::abs(out_data[i] - expected);
    max_error = std::max(max_error, error);
  }

  printf("Max error: %e (checked %d elements)\n", max_error, size);
  if (max_error > 1e-3f) {
    printf("FAIL: max error exceeds tolerance 1e-3\n");
    return false;
  }
  printf("PASS: chained add test\n");
  return true;
}

static bool test_update_cache(const std::string& model_path) {
  // update_cache: value [1,2,2,4] scattered into cache [1,8,2,4] at
  // input_pos=0.
  printf(
      "\n--- Test: update_cache (value[1,2,2,4] -> cache[1,8,2,4], pos=0) ---\n");

  Module module(model_path);
  auto err = module.load_forward();
  if (err != Error::Ok) {
    printf("FAIL: could not load forward method (error %d)\n", (int)err);
    return false;
  }
  printf("Model loaded: %s\n", model_path.c_str());

  constexpr int S = 2, H = 2, D = 4, Cmax = 8;
  constexpr int vnumel = S * H * D; // 16
  constexpr int cnumel = Cmax * H * D; // 64
  constexpr int input_pos = 0;

  std::vector<float> value(vnumel);
  std::vector<float> cache(cnumel);
  for (int i = 0; i < vnumel; i++) {
    value[i] = static_cast<float>(i) * 0.5f;
  }
  for (int i = 0; i < cnumel; i++) {
    cache[i] = static_cast<float>(i) + 100.0f;
  }

  // Reference: input_pos=0 overwrites the [0,S) seq slice of the cache with
  // value; the rest is preserved. Trivial scatter -- no library math involved.
  std::vector<float> ref(cache);
  for (int i = 0; i < vnumel; i++) {
    ref[input_pos * H * D + i] = value[i];
  }

  auto v = make_tensor_ptr({1, S, H, D}, std::vector<float>(value));
  auto c = make_tensor_ptr({1, Cmax, H, D}, std::vector<float>(cache));
  auto result = module.forward({EValue(v), EValue(c)});
  if (!result.ok()) {
    printf("FAIL: forward failed (error %d)\n", (int)result.error());
    return false;
  }

  const auto& outputs = result.get();
  if (outputs.empty() || !outputs[0].isTensor()) {
    printf("FAIL: no tensor output\n");
    return false;
  }
  const auto& out_tensor = outputs[0].toTensor();
  if (out_tensor.numel() != cnumel) {
    printf(
        "FAIL: output numel %zu != expected %d\n",
        (size_t)out_tensor.numel(),
        cnumel);
    return false;
  }
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f;
  for (int i = 0; i < cnumel; i++) {
    max_abs_err = std::max(max_abs_err, std::abs(out_data[i] - ref[i]));
  }
  printf("Max abs error: %e (checked %d elements)\n", max_abs_err, cnumel);
  if (max_abs_err > 1e-3f) {
    printf("FAIL: max error exceeds tolerance 1e-3\n");
    return false;
  }
  printf("PASS: update_cache test\n");
  return true;
}

static std::vector<float> load_golden(const std::string& path, size_t numel) {
  // Load a raw little-endian fp32 golden written by the export .py (the native
  // binary has no ATen/torch, so the reference is computed offline).
  std::vector<float> g(numel);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return {};
  }
  size_t n = std::fread(g.data(), sizeof(float), numel, f);
  std::fclose(f);
  if (n != numel) {
    return {};
  }
  return g;
}

// Per-element dual tolerance mirroring at::allclose's combined gate: an element
// is OK if within abs (1e-4) OR within rel (1e-3) tol, so a near-zero golden
// value can't blow up the rel metric (the kernel's ~1e-8 abs error is the real
// signal at llama3 scale). Sets the reported maxima; true iff all elements
// pass.
static bool sdpa_within_tol(
    const float* out,
    const float* golden,
    int n,
    float* ma,
    float* mr) {
  float max_abs = 0.0f, max_rel = 0.0f;
  bool ok = true;
  for (int i = 0; i < n; i++) {
    const float ae = std::abs(out[i] - golden[i]);
    const float re = ae / std::max(std::abs(golden[i]), 1e-6f);
    max_abs = std::max(max_abs, ae);
    max_rel = std::max(max_rel, re);
    if (ae > 1e-4f && re > 1e-3f) {
      ok = false;
    }
  }
  *ma = max_abs;
  *mr = max_rel;
  return ok;
}

// Fused sdpa_with_kv_cache sweep config. Mirrors the Python CONFIGS table in
// test_sdpa.py exactly (name, Hq, Hkv, D, S, Cmax, input_pos).
struct SdpaConfig {
  const char* name;
  int hq; // query heads
  int hkv; // key/value heads (GQA groups when hq != hkv)
  int d; // head dim
  int s; // new tokens this step
  int cmax; // kv-cache capacity
  int input_pos; // prior tokens already in the cache (decode)
  float denom; // ramp divisor (mirrors Python); small -> large logits
};

static const SdpaConfig kSdpaConfigs[] = {
    // name             Hq Hkv  D  S Cmax pos denom
    {"gqa31_prefill", 6, 2, 8, 4, 16, 0, 16.0f}, // GQA 3:1 (original case)
    {"mha_ctxodd", 4, 4, 16, 3, 8, 0, 16.0f}, // MHA; context_len=3 (odd)
    {"gqa21_prefill", 8, 4, 4, 5, 16, 0, 16.0f}, // GQA 2:1; multi-token S=5
    {"gqa31_decode", 6, 2, 8, 2, 16, 2, 16.0f}, // decode: 2 prior tokens
    // llama3-ish GQA, D=128, S=128.
    {"llama3_prefill", 24, 8, 128, 128, 256, 0, 16.0f},
    // Adversarial: denom=0.5 -> peak logit ~177 (>88) overflows naive fp32 exp.
    {"mha_biglogit", 4, 4, 32, 4, 16, 0, 0.5f},
    // Llama 3.2 1B shape (Hq=32,Hkv=8,D=64): decode at 4k/8k ctx.
    {"llama1b_decode_4k", 32, 8, 64, 1, 4096, 4095, 16.0f},
    {"llama1b_decode_8k", 32, 8, 64, 1, 8192, 8191, 16.0f},
    // Llama 3.2 1B shape: realistic prefill (S=128 at pos 0) + decode (S=1 at pos 127).
    {"llama1b_prefill", 32, 8, 64, 128, 512, 0, 16.0f},
    {"llama1b_decode", 32, 8, 64, 1, 512, 127, 16.0f},
};

// /denom ramp: ((i % mod) - off) / denom, exact in fp32 (power-of-two denom).
// Mirrors test_sdpa.py::_ramp.
static float sdpa_ramp(int i, int mod, int off, float denom = 16.0f) {
  return static_cast<float>((i % mod) - off) / denom;
}

// Step-indexed ramp; mirrors test_sdpa.py::_ramp_t bit-for-bit (integer
// modulo).
static float sdpa_ramp_t(int i, int mod, int off, int t) {
  return static_cast<float>(((i + 31 * t) % mod) - off) / 16.0f;
}

// Multi-step replay sequences. Mirror the Python REPLAY_SEQS / Vulkan param
// sets (sdpa_test.cpp:856/867/875). Each seq_lens entry is one step replayed on
// a host-threaded KV cache (big=prefill, mid=multi-token, 1=decode).
struct SdpaSequence {
  const char* name;
  int hq;
  int hkv;
  int d;
  int cmax;
  std::vector<int> seq_lens;
};

static const SdpaSequence kSdpaSequences[] = {
    {"small", 8, 4, 4, 16, {3, 1, 1, 5, 1, 1, 2}},
    {"small_d", 6, 2, 8, 16, {3, 1, 1, 5, 1, 1}},
    {"llama3", 24, 8, 128, 256, {111, 1, 1, 1, 57, 1, 1}},
};

static bool test_sdpa_config(
    const SdpaConfig& cfg,
    const std::string& model_path,
    const std::string& golden_path) {
  // Inputs reconstruct test_sdpa.py::_det_inputs bit-for-bit (/16 exact fp32).
  printf(
      "\n--- Test: sdpa_with_kv_cache (%s: Hq=%d,Hkv=%d,D=%d,S=%d,Cmax=%d,pos=%d) ---\n",
      cfg.name,
      cfg.hq,
      cfg.hkv,
      cfg.d,
      cfg.s,
      cfg.cmax,
      cfg.input_pos);

  Module module(model_path);
  auto err = module.load_forward();
  if (err != Error::Ok) {
    printf("FAIL: could not load forward method (error %d)\n", (int)err);
    return false;
  }
  printf("Model loaded: %s\n", model_path.c_str());

  const int qn = cfg.s * cfg.hq * cfg.d;
  const int kn = cfg.s * cfg.hkv * cfg.d;
  const int cn = cfg.cmax * cfg.hkv * cfg.d;
  const int on = cfg.s * cfg.hq * cfg.d;

  std::vector<float> q(qn), k(kn), v(kn), kc(cn, 0.0f), vc(cn, 0.0f);
  for (int i = 0; i < qn; i++) {
    q[i] = sdpa_ramp(i, 17, 8, cfg.denom);
  }
  for (int i = 0; i < kn; i++) {
    k[i] = sdpa_ramp(i, 13, 6, cfg.denom);
    v[i] = sdpa_ramp(i, 11, 5, cfg.denom);
  }
  // Decode: seed cache rows [0, input_pos) with prior_k/prior_v (flat over
  // input_pos*Hkv*D elements); all other rows stay zero.
  const int prior_n = cfg.input_pos * cfg.hkv * cfg.d;
  for (int i = 0; i < prior_n; i++) {
    kc[i] = sdpa_ramp(i, 7, 3);
    vc[i] = sdpa_ramp(i, 5, 2);
  }

  auto qt = make_tensor_ptr({1, cfg.s, cfg.hq, cfg.d}, std::vector<float>(q));
  auto kt = make_tensor_ptr({1, cfg.s, cfg.hkv, cfg.d}, std::vector<float>(k));
  auto vt = make_tensor_ptr({1, cfg.s, cfg.hkv, cfg.d}, std::vector<float>(v));
  auto kct =
      make_tensor_ptr({1, cfg.cmax, cfg.hkv, cfg.d}, std::vector<float>(kc));
  auto vct =
      make_tensor_ptr({1, cfg.cmax, cfg.hkv, cfg.d}, std::vector<float>(vc));

  auto result = module.forward(
      {EValue(qt), EValue(kt), EValue(vt), EValue(kct), EValue(vct)});
  if (!result.ok()) {
    printf("FAIL: forward failed (error %d)\n", (int)result.error());
    return false;
  }

  const auto& outputs = result.get();
  // The mutating op returns [k_cache, v_cache, attn_output]; select the
  // attention output (numel == S*Hq*D), not a mutated cache (numel Cmax*Hkv*D).
  int attn_idx = -1;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i].isTensor() && outputs[i].toTensor().numel() == on) {
      attn_idx = static_cast<int>(i);
      break;
    }
  }
  if (attn_idx < 0) {
    printf(
        "FAIL: no attention output (numel %d) among %zu outputs\n",
        on,
        outputs.size());
    return false;
  }
  const auto& out_tensor = outputs[attn_idx].toTensor();
  const float* out_data = out_tensor.const_data_ptr<float>();

  std::vector<float> golden = load_golden(golden_path, on);
  if (golden.empty()) {
    printf("FAIL: could not load golden %s\n", golden_path.c_str());
    return false;
  }

  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  const bool pass =
      sdpa_within_tol(out_data, golden.data(), on, &max_abs_err, &max_rel_err);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      on);
  if (!pass) {
    printf(
        "FAIL: %s exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)\n",
        cfg.name);
    return false;
  }
  printf("PASS: sdpa test (%s)\n", cfg.name);
  return true;
}

// Run the full SDPA sweep. Each config self-discovers its embedded/on-disk
// sdpa_<name>.pte; a config is skipped silently when its .pte is absent, so the
// same binary works whether one or all configs are embedded. Returns false only
// if a discovered config actually fails. Sets *ran true if any config ran.
static bool test_sdpa_sweep(const std::string& dir, bool* ran) {
  bool ok = true;
  for (const auto& cfg : kSdpaConfigs) {
    const std::string pte = dir + "sdpa_" + cfg.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      continue; // config not embedded in this binary
    }
    std::fclose(f);
    const std::string golden = dir + "sdpa_" + cfg.name + ".golden.bin";
    *ran = true;
    ok = test_sdpa_config(cfg, pte, golden) && ok;
  }
  return ok;
}

// Replay one sequence: thread the op's returned (mutated) KV cache across
// steps, comparing each step's attention output to its accumulated-context
// golden.
static bool test_sdpa_replay(const SdpaSequence& seq, const std::string& dir) {
  printf(
      "\n--- Test: sdpa replay (%s: Hq=%d,Hkv=%d,D=%d,Cmax=%d, %zu steps) ---\n",
      seq.name,
      seq.hq,
      seq.hkv,
      seq.d,
      seq.cmax,
      seq.seq_lens.size());

  const int cn = seq.cmax * seq.hkv * seq.d;
  std::vector<float> kc(cn, 0.0f), vc(cn, 0.0f);
  int input_pos = 0;
  int k_idx = -1,
      v_idx = -1; // pinned at step 0 by content (caches share numel)
  bool ok = true;

  for (size_t t = 0; t < seq.seq_lens.size(); t++) {
    const int s = seq.seq_lens[t];
    const std::string base = dir + "sdpa_" + seq.name + "_step" +
        std::to_string(t) + "_S" + std::to_string(s) + "_pos" +
        std::to_string(input_pos);
    Module module(base + ".pte");
    if (module.load_forward() != Error::Ok) {
      printf("FAIL: could not load %s.pte\n", base.c_str());
      return false;
    }

    const int qn = s * seq.hq * seq.d;
    const int kvn = s * seq.hkv * seq.d;
    std::vector<float> q(qn), k(kvn), v(kvn);
    for (int i = 0; i < qn; i++) {
      q[i] = sdpa_ramp_t(i, 17, 8, static_cast<int>(t));
    }
    for (int i = 0; i < kvn; i++) {
      k[i] = sdpa_ramp_t(i, 13, 6, static_cast<int>(t));
      v[i] = sdpa_ramp_t(i, 11, 5, static_cast<int>(t));
    }

    auto qt = make_tensor_ptr({1, s, seq.hq, seq.d}, std::vector<float>(q));
    auto kt = make_tensor_ptr({1, s, seq.hkv, seq.d}, std::vector<float>(k));
    auto vt = make_tensor_ptr({1, s, seq.hkv, seq.d}, std::vector<float>(v));
    auto kct =
        make_tensor_ptr({1, seq.cmax, seq.hkv, seq.d}, std::vector<float>(kc));
    auto vct =
        make_tensor_ptr({1, seq.cmax, seq.hkv, seq.d}, std::vector<float>(vc));

    auto result = module.forward(
        {EValue(qt), EValue(kt), EValue(vt), EValue(kct), EValue(vct)});
    if (!result.ok()) {
      printf(
          "FAIL: forward %s.pte (error %d)\n",
          base.c_str(),
          (int)result.error());
      return false;
    }
    const auto& outs = result.get();

    // The op returns [k_cache, v_cache, attn_output]: attn has a unique numel;
    // the two caches share numel cn, so identify them by content at step 0.
    int attn_idx = -1;
    std::vector<int> cache_idxs;
    for (size_t i = 0; i < outs.size(); i++) {
      if (!outs[i].isTensor()) {
        continue;
      }
      const int ne = static_cast<int>(outs[i].toTensor().numel());
      if (ne == qn) {
        attn_idx = static_cast<int>(i);
      } else if (ne == cn) {
        cache_idxs.push_back(static_cast<int>(i));
      }
    }
    if (attn_idx < 0 || cache_idxs.size() != 2) {
      printf("FAIL: %s step%zu: expected 1 attn + 2 caches\n", seq.name, t);
      return false;
    }

    if (t == 0) {
      const float* c0 = outs[cache_idxs[0]].toTensor().const_data_ptr<float>();
      const float* c1 = outs[cache_idxs[1]].toTensor().const_data_ptr<float>();
      auto rows_match = [&](const float* c, const std::vector<float>& src) {
        for (int i = 0; i < kvn; i++) {
          if (std::abs(c[i] - src[i]) > 1e-6f) {
            return false;
          }
        }
        return true;
      };
      if (rows_match(c0, k) && rows_match(c1, v)) {
        k_idx = cache_idxs[0];
        v_idx = cache_idxs[1];
      } else if (rows_match(c1, k) && rows_match(c0, v)) {
        k_idx = cache_idxs[1];
        v_idx = cache_idxs[0];
      } else {
        printf(
            "FAIL: %s step0 cannot identify k/v cache by content\n", seq.name);
        return false;
      }
      printf("  k/v cache outputs: k_idx=%d v_idx=%d\n", k_idx, v_idx);
    }

    std::vector<float> golden = load_golden(base + ".golden.bin", qn);
    if (golden.empty()) {
      printf("FAIL: could not load %s.golden.bin\n", base.c_str());
      return false;
    }
    const float* ad = outs[attn_idx].toTensor().const_data_ptr<float>();
    float ma = 0.0f, mr = 0.0f;
    const bool step_ok = sdpa_within_tol(ad, golden.data(), qn, &ma, &mr);
    printf(
        "  step%zu (S=%d pos=%d ctx=%d): max abs %e  rel %e\n",
        t,
        s,
        input_pos,
        input_pos + s,
        ma,
        mr);
    if (!step_ok) {
      printf(
          "FAIL: %s step%zu exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)\n",
          seq.name,
          t);
      ok = false;
    }

    // Thread the device-written caches into the next step (K->K, V->V).
    const float* kd = outs[k_idx].toTensor().const_data_ptr<float>();
    const float* vd = outs[v_idx].toTensor().const_data_ptr<float>();
    kc.assign(kd, kd + cn);
    vc.assign(vd, vd + cn);
    input_pos += s;
  }

  if (ok) {
    printf("PASS: sdpa replay (%s)\n", seq.name);
  }
  return ok;
}

// Run all replay sequences whose step0 .pte is present (self-skip otherwise).
static bool test_sdpa_replay_sweep(const std::string& dir, bool* ran) {
  bool ok = true;
  for (const auto& seq : kSdpaSequences) {
    const std::string step0 = dir + "sdpa_" + seq.name + "_step0_S" +
        std::to_string(seq.seq_lens[0]) + "_pos0.pte";
    FILE* f = std::fopen(step0.c_str(), "rb");
    if (!f) {
      continue; // sequence not embedded in this binary
    }
    std::fclose(f);
    *ran = true;
    ok = test_sdpa_replay(seq, dir) && ok;
  }
  return ok;
}

// Dynamic input_pos decode: ONE .pte (S=1, runtime SymInt input_pos) reused
// across decode steps. Each forward() supplies input_pos as a [1] int64 tensor;
// the backend reads it (update_symints_from_inputs) and recomputes dispatch
// state (propagate_resize) before replaying. The cache is threaded host-side
// (the Module re-copies inputs each call), so correctness hinges on the
// per-step input_pos actually being read + applied. negative=true pins
// input_pos at 0 every step (stale context_len) and asserts the run DIVERGES,
// proving the runtime input_pos + resize hook are load-bearing (no false-pass).
static bool test_sdpa_dynamic_decode(
    const SdpaSequence& seq,
    const std::string& dir,
    bool negative) {
  constexpr int kSteps = 6; // mirrors DYN_DECODE_STEPS in test_sdpa.py
  printf(
      "\n--- Test: sdpa dynamic decode%s (%s: Hq=%d,Hkv=%d,D=%d,Cmax=%d, %d steps) ---\n",
      negative ? " [NEGATIVE]" : "",
      seq.name,
      seq.hq,
      seq.hkv,
      seq.d,
      seq.cmax,
      kSteps);

  const std::string pte = dir + "sdpa_dyn_" + seq.name + ".pte";
  Module module(pte);
  if (module.load_forward() != Error::Ok) {
    printf("FAIL: could not load %s\n", pte.c_str());
    return false;
  }

  const int cn = seq.cmax * seq.hkv * seq.d;
  std::vector<float> kc(cn, 0.0f), vc(cn, 0.0f);
  int k_idx = -1,
      v_idx = -1; // pinned at step 0 by content (caches share numel)
  bool ok = true;
  bool any_mismatch = false;

  for (int t = 0; t < kSteps; t++) {
    const int qn = seq.hq * seq.d; // S=1
    const int kvn = seq.hkv * seq.d; // S=1
    std::vector<float> q(qn), k(kvn), v(kvn);
    for (int i = 0; i < qn; i++) {
      q[i] = sdpa_ramp_t(i, 17, 8, t);
    }
    for (int i = 0; i < kvn; i++) {
      k[i] = sdpa_ramp_t(i, 13, 6, t);
      v[i] = sdpa_ramp_t(i, 11, 5, t);
    }
    auto qt = make_tensor_ptr({1, 1, seq.hq, seq.d}, std::vector<float>(q));
    auto kt = make_tensor_ptr({1, 1, seq.hkv, seq.d}, std::vector<float>(k));
    auto vt = make_tensor_ptr({1, 1, seq.hkv, seq.d}, std::vector<float>(v));
    auto kct =
        make_tensor_ptr({1, seq.cmax, seq.hkv, seq.d}, std::vector<float>(kc));
    auto vct =
        make_tensor_ptr({1, seq.cmax, seq.hkv, seq.d}, std::vector<float>(vc));
    const int64_t pos = negative ? 0 : t;
    auto ipt = make_tensor_ptr({1}, std::vector<int64_t>{pos});

    auto result = module.forward(
        {EValue(qt),
         EValue(kt),
         EValue(vt),
         EValue(kct),
         EValue(vct),
         EValue(ipt)});
    if (!result.ok()) {
      printf("FAIL: forward step%d (error %d)\n", t, (int)result.error());
      return false;
    }
    const auto& outs = result.get();

    int attn_idx = -1;
    std::vector<int> cache_idxs;
    for (size_t i = 0; i < outs.size(); i++) {
      if (!outs[i].isTensor()) {
        continue;
      }
      const int ne = static_cast<int>(outs[i].toTensor().numel());
      if (ne == qn) {
        attn_idx = static_cast<int>(i);
      } else if (ne == cn) {
        cache_idxs.push_back(static_cast<int>(i));
      }
    }
    if (attn_idx < 0 || cache_idxs.size() != 2) {
      printf("FAIL: %s step%d: expected 1 attn + 2 caches\n", seq.name, t);
      return false;
    }
    if (t == 0) {
      const float* c0 = outs[cache_idxs[0]].toTensor().const_data_ptr<float>();
      const float* c1 = outs[cache_idxs[1]].toTensor().const_data_ptr<float>();
      auto rows_match = [&](const float* c, const std::vector<float>& src) {
        for (int i = 0; i < kvn; i++) {
          if (std::abs(c[i] - src[i]) > 1e-6f) {
            return false;
          }
        }
        return true;
      };
      if (rows_match(c0, k) && rows_match(c1, v)) {
        k_idx = cache_idxs[0];
        v_idx = cache_idxs[1];
      } else if (rows_match(c1, k) && rows_match(c0, v)) {
        k_idx = cache_idxs[1];
        v_idx = cache_idxs[0];
      } else {
        printf("FAIL: %s step0 cannot identify k/v cache\n", seq.name);
        return false;
      }
    }

    const std::string gpath = dir + "sdpa_dyn_" + seq.name + "_step" +
        std::to_string(t) + ".golden.bin";
    std::vector<float> golden = load_golden(gpath, qn);
    if (golden.empty()) {
      printf("FAIL: could not load %s\n", gpath.c_str());
      return false;
    }
    const float* ad = outs[attn_idx].toTensor().const_data_ptr<float>();
    float ma = 0.0f, mr = 0.0f;
    const bool step_ok = sdpa_within_tol(ad, golden.data(), qn, &ma, &mr);
    printf(
        "  step%d (pos=%d ctx=%d): max abs %e  rel %e%s\n",
        t,
        (int)pos,
        t + 1,
        ma,
        mr,
        step_ok ? "" : "  <-- mismatch");
    if (!step_ok) {
      any_mismatch = true;
    }

    const float* kd = outs[k_idx].toTensor().const_data_ptr<float>();
    const float* vd = outs[v_idx].toTensor().const_data_ptr<float>();
    kc.assign(kd, kd + cn);
    vc.assign(vd, vd + cn);
  }

  if (negative) {
    if (any_mismatch) {
      printf(
          "PASS: sdpa dynamic decode NEGATIVE (%s): stale input_pos diverges "
          "as expected\n",
          seq.name);
      return true;
    }
    printf(
        "FAIL: %s negative control matched the golden (oracle has no teeth)\n",
        seq.name);
    return false;
  }
  if (any_mismatch) {
    printf(
        "FAIL: %s exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)\n",
        seq.name);
    ok = false;
  }
  if (ok) {
    printf("PASS: sdpa dynamic decode (%s)\n", seq.name);
  }
  return ok;
}

// Run dynamic decode (positive + negative control) for each param set whose
// sdpa_dyn_<name>.pte is embedded (self-skip otherwise).
static bool test_sdpa_dynamic_decode_sweep(const std::string& dir, bool* ran) {
  bool ok = true;
  for (const auto& seq : kSdpaSequences) {
    const std::string pte = dir + "sdpa_dyn_" + seq.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      continue;
    }
    std::fclose(f);
    *ran = true;
    ok = test_sdpa_dynamic_decode(seq, dir, /*negative=*/false) && ok;
    ok = test_sdpa_dynamic_decode(seq, dir, /*negative=*/true) && ok;
  }
  return ok;
}

// In-graph mutable KV cache: ONE .pte whose k_cache/v_cache are mutable buffers
// (NOT forward inputs); the decode loop feeds only the new token (q/k/v, S=1) +
// runtime input_pos, and the cache accumulates in-graph across forward() calls
// (no host threading). fresh_per_step is the static control: reloading the
// Module each step re-seeds the cache to zeros, so it MUST diverge from the
// accumulating golden at step>=1. Persistent-matches + fresh-diverges = proof
// the pass comes from real accumulation, not a static artifact.
static bool test_sdpa_incache_decode(
    const SdpaSequence& seq,
    const std::string& dir,
    bool fresh_per_step) {
  constexpr int kSteps = 6; // mirrors DYN_DECODE_STEPS in test_sdpa.py
  printf(
      "\n--- Test: sdpa in-graph-cache decode%s (%s: Hq=%d,Hkv=%d,D=%d,Cmax=%d, %d steps) ---\n",
      fresh_per_step ? " [STATIC CONTROL: fresh Module/step]" : "",
      seq.name,
      seq.hq,
      seq.hkv,
      seq.d,
      seq.cmax,
      kSteps);

  const std::string pte = dir + "sdpa_incache_" + seq.name + ".pte";
  std::unique_ptr<Module> persistent;
  if (!fresh_per_step) {
    persistent = std::make_unique<Module>(pte);
    if (persistent->load_forward() != Error::Ok) {
      printf("FAIL: could not load %s\n", pte.c_str());
      return false;
    }
  }

  bool any_mismatch = false;
  for (int t = 0; t < kSteps; t++) {
    const int qn = seq.hq * seq.d; // S=1
    const int kvn = seq.hkv * seq.d; // S=1
    std::vector<float> q(qn), k(kvn), v(kvn);
    for (int i = 0; i < qn; i++) {
      q[i] = sdpa_ramp_t(i, 17, 8, t);
    }
    for (int i = 0; i < kvn; i++) {
      k[i] = sdpa_ramp_t(i, 13, 6, t);
      v[i] = sdpa_ramp_t(i, 11, 5, t);
    }
    auto qt = make_tensor_ptr({1, 1, seq.hq, seq.d}, std::vector<float>(q));
    auto kt = make_tensor_ptr({1, 1, seq.hkv, seq.d}, std::vector<float>(k));
    auto vt = make_tensor_ptr({1, 1, seq.hkv, seq.d}, std::vector<float>(v));
    auto ipt =
        make_tensor_ptr({1}, std::vector<int64_t>{static_cast<int64_t>(t)});

    // Persistent: reuse the one Module (cache accumulates). Fresh: a new Module
    // each step (cache re-seeded to zeros -> no history).
    std::unique_ptr<Module> fresh;
    Module* mod = persistent.get();
    if (fresh_per_step) {
      fresh = std::make_unique<Module>(pte);
      if (fresh->load_forward() != Error::Ok) {
        printf("FAIL: could not load %s\n", pte.c_str());
        return false;
      }
      mod = fresh.get();
    }

    // NOTE: only q/k/v + input_pos -- NO cache args (caches are mutable
    // buffers).
    auto result =
        mod->forward({EValue(qt), EValue(kt), EValue(vt), EValue(ipt)});
    if (!result.ok()) {
      printf("FAIL: forward step%d (error %d)\n", t, (int)result.error());
      return false;
    }
    const auto& outs = result.get();
    int attn_idx = -1;
    for (size_t i = 0; i < outs.size(); i++) {
      if (outs[i].isTensor() &&
          static_cast<int>(outs[i].toTensor().numel()) == qn) {
        attn_idx = static_cast<int>(i);
        break;
      }
    }
    if (attn_idx < 0) {
      printf("FAIL: %s step%d: no attn output (numel %d)\n", seq.name, t, qn);
      return false;
    }

    const std::string gpath = dir + "sdpa_incache_" + seq.name + "_step" +
        std::to_string(t) + ".golden.bin";
    std::vector<float> golden = load_golden(gpath, qn);
    if (golden.empty()) {
      printf("FAIL: could not load %s\n", gpath.c_str());
      return false;
    }
    const float* ad = outs[attn_idx].toTensor().const_data_ptr<float>();
    float ma = 0.0f, mr = 0.0f;
    const bool step_ok = sdpa_within_tol(ad, golden.data(), qn, &ma, &mr);
    printf(
        "  step%d (pos=%d ctx=%d): max abs %e  rel %e%s\n",
        t,
        t,
        t + 1,
        ma,
        mr,
        step_ok ? "" : "  <-- mismatch");
    if (!step_ok) {
      any_mismatch = true;
    }
  }

  if (fresh_per_step) {
    // The control must DIVERGE: a fresh Module per step has no accumulated
    // history, so it cannot match the accumulating golden at step>=1.
    if (any_mismatch) {
      printf(
          "PASS: in-graph-cache STATIC CONTROL (%s) diverges as expected -- "
          "persistence is load-bearing; the positive pass is real accumulation\n",
          seq.name);
      return true;
    }
    printf(
        "FAIL: %s static control matched the accumulating golden -- "
        "accumulation was not actually exercised (false-pass risk)\n",
        seq.name);
    return false;
  }
  if (!any_mismatch) {
    printf(
        "PASS: sdpa in-graph-cache decode (%s) -- cache accumulated in-graph "
        "with NO host threading\n",
        seq.name);
    return true;
  }
  printf("FAIL: %s in-graph-cache decode exceeds tolerance\n", seq.name);
  return false;
}

static bool test_sdpa_incache_decode_sweep(const std::string& dir, bool* ran) {
  bool ok = true;
  for (const auto& seq : kSdpaSequences) {
    const std::string pte = dir + "sdpa_incache_" + seq.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      continue;
    }
    std::fclose(f);
    *ran = true;
    ok = test_sdpa_incache_decode(seq, dir, /*fresh_per_step=*/false) && ok;
    ok = test_sdpa_incache_decode(seq, dir, /*fresh_per_step=*/true) && ok;
  }
  return ok;
}

// S1 SymInt round-trip: build a graph directly from a dynamic-input_pos SDPA
// blob; confirm input_pos deserializes as a live SymInt and set/read
// round-trips.
static bool test_symint_roundtrip(const std::string& blob_path) {
  printf("\n--- Test: symint round-trip (%s) ---\n", blob_path.c_str());
  FILE* f = std::fopen(blob_path.c_str(), "rb");
  if (!f) {
    printf("SKIP: %s not present\n", blob_path.c_str());
    return true;
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> blob(static_cast<size_t>(n));
  size_t rd = std::fread(blob.data(), 1, blob.size(), f);
  std::fclose(f);
  if (rd != blob.size()) {
    printf("FAIL: short read of %s\n", blob_path.c_str());
    return false;
  }

  auto header = WebGPUDelegateHeader::parse(blob.data());
  if (!header.ok()) {
    printf("FAIL: delegate header parse\n");
    return false;
  }
  const uint8_t* base = blob.data();
  WebGPUGraph graph;
  try {
    graph.build(
        base + header->flatbuffer_offset, base + header->bytes_offset, nullptr);
  } catch (const std::exception& e) {
    printf("FAIL: graph build: %s\n", e.what());
    return false;
  }

  int sid = -1;
  for (int i = 0; i < graph.num_values(); i++) {
    if (graph.get_value_type(i) == WebGPUGraph::ValueType::SymInt) {
      sid = i;
      break;
    }
  }
  if (sid < 0) {
    printf(
        "FAIL: no SymInt value deserialized (input_pos should be a SymInt)\n");
    return false;
  }
  if (graph.symint_buffer(sid) == nullptr) {
    printf("FAIL: SymInt %d has no live uniform buffer\n", sid);
    return false;
  }
  if (graph.read_symint(sid) != 0) {
    printf(
        "FAIL: SymInt %d placeholder != 0 (got %d)\n",
        sid,
        graph.read_symint(sid));
    return false;
  }
  graph.set_symint(sid, 7);
  if (graph.read_symint(sid) != 7) {
    printf("FAIL: set/read round-trip (got %d)\n", graph.read_symint(sid));
    return false;
  }

  // Execute-read: feed a fake input_pos=5 via the recorded select_as_symint
  // source and confirm update_symints_from_inputs populates the SymInt.
  const auto& srcs = graph.symint_sources();
  if (srcs.empty()) {
    printf("FAIL: no select_as_symint source recorded\n");
    return false;
  }
  const auto& in_ids = graph.input_ids();
  std::vector<std::pair<const void*, size_t>> fake_inputs(
      in_ids.size(), {nullptr, 0});
  int64_t fake_pos = 5;
  for (size_t i = 0; i < in_ids.size(); i++) {
    if (in_ids[i] == srcs[0].input_tensor_id) {
      fake_inputs[i] = {&fake_pos, sizeof(int64_t)};
    }
  }
  graph.update_symints_from_inputs(fake_inputs);
  if (graph.read_symint(srcs[0].symint_id) != 5) {
    printf(
        "FAIL: execute-read (got %d, want 5)\n",
        graph.read_symint(srcs[0].symint_id));
    return false;
  }

  printf(
      "PASS: symint round-trip (SymInt %d: deserialize, live buffer, "
      "set 0->7, execute-read input_pos->5)\n",
      sid);
  return true;
}

// Group 1: the resize-hook dirty-gating mechanism (no SDPA dependency).
// A hook keyed to a SymInt must run via propagate_resize() iff that SymInt
// changed since the last propagate_resize, and exactly once per change.
static bool test_resize_hook(const std::string& blob_path) {
  printf("\n--- Test: resize-hook dirty-gating (%s) ---\n", blob_path.c_str());
  FILE* f = std::fopen(blob_path.c_str(), "rb");
  if (!f) {
    printf("SKIP: %s not present\n", blob_path.c_str());
    return true;
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> blob(static_cast<size_t>(n));
  size_t rd = std::fread(blob.data(), 1, blob.size(), f);
  std::fclose(f);
  if (rd != blob.size()) {
    printf("FAIL: short read of %s\n", blob_path.c_str());
    return false;
  }
  auto header = WebGPUDelegateHeader::parse(blob.data());
  if (!header.ok()) {
    printf("FAIL: delegate header parse\n");
    return false;
  }
  const uint8_t* base = blob.data();
  WebGPUGraph graph;
  try {
    graph.build(
        base + header->flatbuffer_offset, base + header->bytes_offset, nullptr);
  } catch (const std::exception& e) {
    printf("FAIL: graph build: %s\n", e.what());
    return false;
  }

  int sid = -1;
  for (int i = 0; i < graph.num_values(); i++) {
    if (graph.get_value_type(i) == WebGPUGraph::ValueType::SymInt) {
      sid = i;
      break;
    }
  }
  if (sid < 0) {
    printf("FAIL: no SymInt value deserialized\n");
    return false;
  }

  int run_count = 0;
  int last_seen = -1;
  graph.add_resize_hook(sid, [&](WebGPUGraph& g) {
    run_count++;
    last_seen = g.read_symint(sid);
  });

  // 1: change 0->3 then propagate -> hook runs once, sees 3.
  graph.set_symint(sid, 3);
  graph.propagate_resize();
  if (run_count != 1 || last_seen != 3) {
    printf(
        "FAIL: after set(3)+propagate run_count=%d last_seen=%d (want 1,3)\n",
        run_count,
        last_seen);
    return false;
  }
  // 2: propagate again with no change -> hook does NOT run.
  graph.propagate_resize();
  if (run_count != 1) {
    printf(
        "FAIL: propagate with clean dirty-set ran the hook (run_count=%d)\n",
        run_count);
    return false;
  }
  // 3: set to the SAME value -> not dirty -> hook does NOT run.
  graph.set_symint(sid, 3);
  graph.propagate_resize();
  if (run_count != 1) {
    printf(
        "FAIL: set(same)+propagate ran the hook (run_count=%d)\n", run_count);
    return false;
  }
  // 4: change 3->8 then propagate -> hook runs again, sees 8.
  graph.set_symint(sid, 8);
  graph.propagate_resize();
  if (run_count != 2 || last_seen != 8) {
    printf(
        "FAIL: after set(8)+propagate run_count=%d last_seen=%d (want 2,8)\n",
        run_count,
        last_seen);
    return false;
  }

  printf(
      "PASS: resize-hook dirty-gating (SymInt %d: runs only on change, "
      "once per change; saw 3 then 8)\n",
      sid);
  return true;
}

// Capacity-overrun must throw; runs without a device or TimestampQuery.
static bool test_query_pool_overrun_throws() {
  printf("\n--- Test: WebGPUQueryPool capacity-overrun guard ---\n");
  WebGPUQueryPool qp;
  try {
    qp.reset(1);
  } catch (const std::exception&) {
    printf("PASS: reset beyond capacity throws\n");
    return true;
  }
  printf("FAIL: reset beyond capacity did not throw\n");
  return false;
}

// WebGPUQueryPool roundtrip: time a probe pass; assert non-zero GPU duration.
static bool test_query_pool_roundtrip(const WebGPUContext& ctx) {
  printf("\n--- Test: WebGPUQueryPool roundtrip ---\n");
  if (!ctx.timestamp_supported) {
    printf("SKIP: adapter lacks TimestampQuery feature\n");
    return true;
  }
  WGPUDevice device = ctx.device;

  // Probe loop iterates enough to burn a measurable, non-zero GPU duration.
  const char* kProbeWGSL =
      "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n"
      "@compute @workgroup_size(64)\n"
      "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
      "  var acc = 0.0;\n"
      "  for (var i = 0u; i < 8192u; i = i + 1u) {\n"
      "    acc = acc + f32(i) * 1.000001;\n"
      "  }\n"
      "  out[gid.x] = acc;\n"
      "}\n";

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kProbeWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry bgl_entry = {};
  bgl_entry.binding = 0;
  bgl_entry.visibility = WGPUShaderStage_Compute;
  bgl_entry.buffer.type = WGPUBufferBindingType_Storage;
  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 1;
  bgl_desc.entries = &bgl_entry;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUComputePipelineDescriptor pipe_desc = {};
  pipe_desc.layout = pl;
  pipe_desc.compute.module = shader;
  pipe_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipe =
      wgpuDeviceCreateComputePipeline(device, &pipe_desc);

  WGPUBufferDescriptor obd = {};
  obd.size = 64 * sizeof(float);
  obd.usage = WGPUBufferUsage_Storage;
  WGPUBuffer out_buf = wgpuDeviceCreateBuffer(device, &obd);

  WGPUBindGroupEntry bg_entry = {};
  bg_entry.binding = 0;
  bg_entry.buffer = out_buf;
  bg_entry.size = obd.size;
  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 1;
  bg_desc.entries = &bg_entry;
  WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &bg_desc);

  WebGPUQueryPool qp;
  qp.initialize(device, 1);
  qp.reset(1);

  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, nullptr);
  WGPUPassTimestampWrites tw = qp.writes_for(0);
  WGPUComputePassDescriptor pass_desc = {};
  pass_desc.timestampWrites = &tw;
  WGPUComputePassEncoder pass =
      wgpuCommandEncoderBeginComputePass(enc, &pass_desc);
  wgpuComputePassEncoderSetPipeline(pass, pipe);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
  qp.record(0, "probe", {1, 1, 1}, {64, 1, 1});
  qp.resolve(enc);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
  wgpuQueueSubmit(ctx.queue, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  qp.extract_results(ctx.instance);

  wgpuBufferRelease(out_buf);
  wgpuComputePipelineRelease(pipe);
  wgpuPipelineLayoutRelease(pl);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuBindGroupRelease(bg);
  wgpuShaderModuleRelease(shader);

  if (qp.results().size() != 1) {
    printf("FAIL: expected 1 duration, got %zu\n", qp.results().size());
    return false;
  }
  const uint64_t dur = qp.results()[0].execution_duration_ns;
  printf("  probe duration: %llu ns\n", (unsigned long long)dur);
  if (dur == 0) {
    printf("FAIL: probe duration is zero (expected monotonic non-zero)\n");
    return false;
  }
  printf("PASS: WebGPUQueryPool roundtrip -- non-zero GPU kernel duration\n");
  return true;
}

int main(int argc, char** argv) {
  std::string model_path = "webgpu_add_test.pte";
  if (argc > 1) {
    model_path = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_TEST_MODEL")) {
    model_path = env;
  }

  std::string chained_model_path;
  if (const char* env = std::getenv("WEBGPU_TEST_CHAINED_MODEL")) {
    chained_model_path = env;
  }

  std::string update_cache_model_path;
  if (const char* env = std::getenv("WEBGPU_TEST_UPDATE_CACHE_MODEL")) {
    update_cache_model_path = env;
  }

  // SDPA sweep: configs self-discover their sdpa_<name>.pte/.golden.bin under
  // this directory (default "" = the embedded-file root / cwd). Set
  // WEBGPU_TEST_SDPA_DIR to point at the exported .pte directory (e.g. /tmp/).
  std::string sdpa_dir;
  if (const char* env = std::getenv("WEBGPU_TEST_SDPA_DIR")) {
    sdpa_dir = env;
    if (!sdpa_dir.empty() && sdpa_dir.back() != '/') {
      sdpa_dir += '/';
    }
  }

  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& e) {
    printf("SKIP: %s\n", e.what());
    return 0;
  }

  set_default_webgpu_context(&ctx);
  printf("WebGPU device acquired (native)\n");

  bool ok = test_query_pool_overrun_throws();
  ok = test_query_pool_roundtrip(ctx) && ok;
  ok = test_single_add(model_path) && ok;

  if (!chained_model_path.empty()) {
    ok = test_chained_add(chained_model_path) && ok;
  }

  if (!update_cache_model_path.empty()) {
    ok = test_update_cache(update_cache_model_path) && ok;
  }

  bool sdpa_ran = false;
  bool sdpa_ok = test_sdpa_sweep(sdpa_dir, &sdpa_ran);
  if (sdpa_ran) {
    ok = sdpa_ok && ok;
  }

  // Guard python<->C++ ramp bit-identity (recorded: _ramp_t(0,17,8,2)=0.1875).
  if (std::abs(sdpa_ramp_t(0, 17, 8, 2) - 0.1875f) > 1e-12f) {
    printf("FAIL: sdpa_ramp_t bit-identity check\n");
    ok = false;
  }
  // Guard the adversarial denom path: sdpa_ramp(0,17,8,0.5)= -16.0 exactly.
  if (std::abs(sdpa_ramp(0, 17, 8, 0.5f) - (-16.0f)) > 1e-12f) {
    printf("FAIL: sdpa_ramp denom bit-identity check\n");
    ok = false;
  }

  bool replay_ran = false;
  bool replay_ok = test_sdpa_replay_sweep(sdpa_dir, &replay_ran);
  if (replay_ran) {
    ok = replay_ok && ok;
  }

  bool dyn_ran = false;
  bool dyn_ok = test_sdpa_dynamic_decode_sweep(sdpa_dir, &dyn_ran);
  if (dyn_ran) {
    ok = dyn_ok && ok;
  }

  bool incache_ran = false;
  bool incache_ok = test_sdpa_incache_decode_sweep(sdpa_dir, &incache_ran);
  if (incache_ran) {
    ok = incache_ok && ok;
  }

  // If an SDPA dir was given, the exports must have produced .ptes for every
  // family; a self-skip there means a silent export failure, not a pass.
  if (!sdpa_dir.empty() &&
      !(sdpa_ran && replay_ran && dyn_ran && incache_ran)) {
    printf("FAIL: WEBGPU_TEST_SDPA_DIR set but an SDPA family found no .pte\n");
    ok = false;
  }

  if (const char* env = std::getenv("WEBGPU_TEST_SYMINT_BLOB")) {
    ok = test_symint_roundtrip(env) && ok;
    ok = test_resize_hook(env) && ok;
  }

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    return 1;
  }
  printf("\nAll tests passed\n");
  return 0;
}
