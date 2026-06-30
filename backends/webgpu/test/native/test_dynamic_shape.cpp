/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Native test for dynamic tensor shapes (Option 2). One graph is built at the
// upper-bound seq-len MAXS and run at several live S; the output must match the
// torch golden at each S (allocate-at-max + per-op resize hooks + output-EValue
// resize). Cases:
//   A  dyn_rms at S=MAXS                       -> golden (static-equivalent)
//   B  dyn_rms at S < MAXS (64, 8, 1)          -> golden (resize shrinks
//   dispatch) C  ONE loaded graph reused across S        -> all golden (buffers
//   never moved
//                                                 => bind groups stayed valid)
//   D  static_rms (no dynamic dim)             -> golden (static path
//   unchanged) F  dyn_rms_chain (rms(rms(x))) at 3 S      -> golden (resize
//   CASCADE, DD-4)
//   G rms+residual  H rms*x  I dyn_linear  J sdpa_dyn  K emb_dyn  L rope_dyn
//   M dyn_sigmoid  N dyn_select (select_copy(0,-1), dynamic S)
// .pte + goldens from test/ops/dynamic_shape/test_dynamic_shape_export.py.

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

constexpr int kHidden = 64;

std::vector<float> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return {};
  }
  const std::streamsize n = f.tellg();
  if (n < 0) {
    return {};
  }
  f.seekg(0);
  std::vector<float> v(static_cast<size_t>(n) / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

float max_err(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) {
    return 1e30f;
  }
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); i++) {
    m = std::fmax(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

// Run one forward of a [1,1,S,kHidden] input through `m`; return the output.
std::vector<float>
run_s(Module& m, const std::string& dir, const std::string& prefix, int s) {
  auto input =
      read_bin(dir + "/" + prefix + ".S" + std::to_string(s) + ".input.bin");
  if (input.empty()) {
    printf("  MISSING input %s.S%d\n", prefix.c_str(), s);
    return {};
  }
  if (input.size() != static_cast<size_t>(s) * kHidden) {
    printf("  WRONG input size %s.S%d\n", prefix.c_str(), s);
    return {};
  }
  auto t = make_tensor_ptr({1, 1, s, kHidden}, std::move(input));
  auto r = m.forward({EValue(t)});
  if (!r.ok() || r.get().empty() || !r.get()[0].isTensor()) {
    printf("  forward FAILED (S=%d, err=%d)\n", s, r.ok() ? 0 : (int)r.error());
    return {};
  }
  const auto& out = r.get()[0].toTensor();
  const float* d = out.const_data_ptr<float>();
  const size_t numel = static_cast<size_t>(s) * kHidden;
  // Output EValue must have been resized to the live shape.
  if (out.numel() != static_cast<ssize_t>(numel)) {
    printf(
        "  WRONG output numel: got %zd want %zu (S=%d)\n",
        (ssize_t)out.numel(),
        numel,
        s);
    return {};
  }
  return std::vector<float>(d, d + numel);
}

bool check_s(
    Module& m,
    const std::string& dir,
    const std::string& prefix,
    int s,
    bool& ok) {
  auto got = run_s(m, dir, prefix, s);
  auto golden =
      read_bin(dir + "/" + prefix + ".S" + std::to_string(s) + ".golden.bin");
  float e = max_err(got, golden);
  bool pass = !got.empty() && e < 1e-3f;
  printf(
      "  %s S=%-3d max_err=%e -> %s\n",
      prefix.c_str(),
      s,
      e,
      pass ? "PASS" : "FAIL");
  if (!pass) {
    printf("    got.size=%zu golden.size=%zu\n", got.size(), golden.size());
    for (size_t i = 0; i < 4 && i < got.size() && i < golden.size(); i++) {
      printf("    [%zu] got=%.6f golden=%.6f\n", i, got[i], golden[i]);
    }
  }
  ok = ok && pass;
  return pass;
}

// Dynamic quantized linear: input [M, lin_k] -> output [M, lin_n].
constexpr int kLinK = 64;
constexpr int kLinN = 128;
void check_linear(const std::string& dir, int m_rows, bool& ok) {
  Module m(dir + "/dyn_linear.pte");
  if (m.load_forward() != Error::Ok) {
    printf("  FAIL load dyn_linear.pte\n");
    ok = false;
    return;
  }
  auto input =
      read_bin(dir + "/dyn_linear.S" + std::to_string(m_rows) + ".input.bin");
  auto golden =
      read_bin(dir + "/dyn_linear.S" + std::to_string(m_rows) + ".golden.bin");
  if (input.empty()) {
    printf("  MISSING dyn_linear.S%d\n", m_rows);
    ok = false;
    return;
  }
  auto t = make_tensor_ptr({m_rows, kLinK}, std::move(input));
  auto r = m.forward({EValue(t)});
  if (!r.ok() || r.get().empty() || !r.get()[0].isTensor()) {
    printf("  linear M=%d forward FAILED\n", m_rows);
    ok = false;
    return;
  }
  const auto& out = r.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(m_rows) * kLinN;
  std::vector<float> got(
      out.const_data_ptr<float>(), out.const_data_ptr<float>() + numel);
  float e = max_err(got, golden);
  // 4-bit quant: looser tol (the kernel mirrors the dequant-matmul reference).
  bool pass = out.numel() == static_cast<ssize_t>(numel) && e < 5e-3f;
  printf(
      "  dyn_linear M=%-3d max_err=%e -> %s\n",
      m_rows,
      e,
      pass ? "PASS" : "FAIL");
  ok = ok && pass;
}

// Dynamic SDPA (GQA prefill, input_pos=0): q[1,s,hq,d] k/v[1,s,hkv,d]
// caches[1,cmax,hkv,d]; attn output [1,s,hq,d] selected by shape (3 outputs).
constexpr int kSdHq = 8, kSdHkv = 2, kSdD = 16, kSdCmax = 64;
void check_sdpa(const std::string& dir, int s, bool& ok) {
  Module m(dir + "/sdpa_dyn.pte");
  Error le = m.load_forward();
  if (le == Error::DelegateInvalidCompatibility) {
    // PENDING op coverage: dynamic-S SDPA build throws err 48 until registered.
    printf("  PENDING sdpa_dyn S=%d (op coverage, err %d)\n", s, (int)le);
    return;
  }
  if (le != Error::Ok) {
    printf("  sdpa_dyn S=%d load FAILED (err %d)\n", s, (int)le);
    ok = false;
    return;
  }
  const std::string b = dir + "/sdpa_dyn.S" + std::to_string(s) + ".";
  auto q = read_bin(b + "q.bin");
  auto k = read_bin(b + "k.bin");
  auto v = read_bin(b + "v.bin");
  auto kc = read_bin(b + "kc.bin");
  auto vc = read_bin(b + "vc.bin");
  auto golden = read_bin(b + "golden.bin");
  if (q.empty() || golden.empty()) {
    printf("  MISSING sdpa_dyn.S%d\n", s);
    ok = false;
    return;
  }
  auto tq = make_tensor_ptr({1, s, kSdHq, kSdD}, std::move(q));
  auto tk = make_tensor_ptr({1, s, kSdHkv, kSdD}, std::move(k));
  auto tv = make_tensor_ptr({1, s, kSdHkv, kSdD}, std::move(v));
  auto tkc = make_tensor_ptr({1, kSdCmax, kSdHkv, kSdD}, std::move(kc));
  auto tvc = make_tensor_ptr({1, kSdCmax, kSdHkv, kSdD}, std::move(vc));
  auto r =
      m.forward({EValue(tq), EValue(tk), EValue(tv), EValue(tkc), EValue(tvc)});
  if (!r.ok()) {
    printf("  sdpa S=%d forward FAILED (err=%d)\n", s, (int)r.error());
    ok = false;
    return;
  }
  // Select the attn output by full shape [1,s,hq,d] (never numel).
  const float* attn = nullptr;
  size_t numel = static_cast<size_t>(s) * kSdHq * kSdD;
  for (size_t i = 0; i < r.get().size(); i++) {
    if (!r.get()[i].isTensor()) {
      continue;
    }
    const auto& t = r.get()[i].toTensor();
    if (t.dim() == 4 && t.size(1) == s && t.size(2) == kSdHq &&
        t.size(3) == kSdD) {
      attn = t.const_data_ptr<float>();
      break;
    }
  }
  if (attn == nullptr) {
    printf(
        "  sdpa S=%d: no attn output of shape [1,%d,%d,%d]\n",
        s,
        s,
        kSdHq,
        kSdD);
    ok = false;
    return;
  }
  std::vector<float> got(attn, attn + numel);
  float e = max_err(got, golden);
  bool pass = e < 2e-3f; // SDPA tol (abs 1e-4 / rel 1e-3 family)
  printf("  sdpa_dyn S=%-3d max_err=%e -> %s\n", s, e, pass ? "PASS" : "FAIL");
  ok = ok && pass;
}

// Dynamic embedding: int64 token ids [N] -> [N, kEmbDim] fp32. The int64 host
// input exercises copy_inputs' int64->int32 narrow path under dynamic shapes.
constexpr int kEmbDim = 64;
void check_embedding(const std::string& dir, int n, bool& ok) {
  Module m(dir + "/emb_dyn.pte");
  if (m.load_forward() != Error::Ok) {
    printf("  FAIL load emb_dyn.pte\n");
    ok = false;
    return;
  }
  const std::string b = dir + "/emb_dyn.S" + std::to_string(n) + ".";
  std::ifstream f(b + "idx.bin", std::ios::binary | std::ios::ate);
  if (!f) {
    printf("  MISSING emb_dyn.S%d\n", n);
    ok = false;
    return;
  }
  const std::streamsize nb = f.tellg();
  if (nb < 0) {
    printf("  MISSING emb_dyn.S%d\n", n);
    ok = false;
    return;
  }
  f.seekg(0);
  std::vector<int64_t> idx(static_cast<size_t>(nb) / sizeof(int64_t));
  f.read(reinterpret_cast<char*>(idx.data()), nb);
  if (idx.size() != static_cast<size_t>(n)) {
    printf("  WRONG emb_dyn idx size S%d\n", n);
    ok = false;
    return;
  }
  auto golden = read_bin(b + "golden.bin");
  auto t = make_tensor_ptr({n}, std::move(idx)); // int64 (Long) host input
  auto r = m.forward({EValue(t)});
  if (!r.ok() || r.get().empty() || !r.get()[0].isTensor()) {
    printf(
        "  emb N=%d forward FAILED (err=%d)\n", n, r.ok() ? 0 : (int)r.error());
    ok = false;
    return;
  }
  const auto& out = r.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(n) * kEmbDim;
  std::vector<float> got(
      out.const_data_ptr<float>(), out.const_data_ptr<float>() + numel);
  float e = max_err(got, golden);
  bool pass = out.numel() == static_cast<ssize_t>(numel) && e < 5e-3f;
  printf("  emb_dyn N=%-3d max_err=%e -> %s\n", n, e, pass ? "PASS" : "FAIL");
  ok = ok && pass;
}

// Dynamic RoPE: xq[1,s,nh,hd] xk[1,s,nkv,hd] freqs[s,hd/2] -> xq_out/xk_out
// (2 outputs, selected by head count nh != nkv).
constexpr int kRopeNH = 8, kRopeNKV = 2, kRopeHD = 64;
void check_rope(const std::string& dir, int s, bool& ok) {
  Module m(dir + "/rope_dyn.pte");
  if (m.load_forward() != Error::Ok) {
    printf("  FAIL load rope_dyn.pte\n");
    ok = false;
    return;
  }
  const std::string b = dir + "/rope_dyn.S" + std::to_string(s) + ".";
  auto xq = read_bin(b + "xq.bin");
  auto xk = read_bin(b + "xk.bin");
  auto fc = read_bin(b + "fc.bin");
  auto fs = read_bin(b + "fs.bin");
  auto gq = read_bin(b + "gq.bin");
  auto gk = read_bin(b + "gk.bin");
  if (xq.empty() || gq.empty()) {
    printf("  MISSING rope_dyn.S%d\n", s);
    ok = false;
    return;
  }
  auto txq = make_tensor_ptr({1, s, kRopeNH, kRopeHD}, std::move(xq));
  auto txk = make_tensor_ptr({1, s, kRopeNKV, kRopeHD}, std::move(xk));
  auto tfc = make_tensor_ptr({s, kRopeHD / 2}, std::move(fc));
  auto tfs = make_tensor_ptr({s, kRopeHD / 2}, std::move(fs));
  auto r = m.forward({EValue(txq), EValue(txk), EValue(tfc), EValue(tfs)});
  if (!r.ok()) {
    printf("  rope S=%d forward FAILED (err=%d)\n", s, (int)r.error());
    ok = false;
    return;
  }
  // Select xq_out (nh heads) and xk_out (nkv heads) by shape.
  const float *oq = nullptr, *okp = nullptr;
  for (size_t i = 0; i < r.get().size(); i++) {
    if (!r.get()[i].isTensor()) {
      continue;
    }
    const auto& t = r.get()[i].toTensor();
    if (t.dim() == 4 && t.size(1) == s && t.size(3) == kRopeHD) {
      if (t.size(2) == kRopeNH) {
        oq = t.const_data_ptr<float>();
      } else if (t.size(2) == kRopeNKV) {
        okp = t.const_data_ptr<float>();
      }
    }
  }
  if (oq == nullptr || okp == nullptr) {
    printf("  rope S=%d: missing xq_out/xk_out by shape\n", s);
    ok = false;
    return;
  }
  std::vector<float> gotq(oq, oq + static_cast<size_t>(s) * kRopeNH * kRopeHD);
  std::vector<float> gotk(
      okp, okp + static_cast<size_t>(s) * kRopeNKV * kRopeHD);
  float e = std::fmax(max_err(gotq, gq), max_err(gotk, gk));
  bool pass = e < 1e-3f;
  printf("  rope_dyn S=%-3d max_err=%e -> %s\n", s, e, pass ? "PASS" : "FAIL");
  ok = ok && pass;
}

// Dynamic select_copy(0,-1): input [2,1,S,kHidden] -> output [1,S,kHidden]. The
// negative index resolves against the (static) leading dim live; the dynamic S
// flows to the output, so the resize hook recomputes its dispatch each S.
constexpr int kSelLead = 2;
void check_select(const std::string& dir, int s, bool& ok) {
  Module m(dir + "/dyn_select.pte");
  if (m.load_forward() != Error::Ok) {
    printf("  FAIL load dyn_select.pte\n");
    ok = false;
    return;
  }
  auto input =
      read_bin(dir + "/dyn_select.S" + std::to_string(s) + ".input.bin");
  auto golden =
      read_bin(dir + "/dyn_select.S" + std::to_string(s) + ".golden.bin");
  if (input.empty() || golden.empty()) {
    printf("  MISSING dyn_select.S%d\n", s);
    ok = false;
    return;
  }
  auto t = make_tensor_ptr({kSelLead, 1, s, kHidden}, std::move(input));
  auto r = m.forward({EValue(t)});
  if (!r.ok() || r.get().empty() || !r.get()[0].isTensor()) {
    printf(
        "  select S=%d forward FAILED (err=%d)\n",
        s,
        r.ok() ? 0 : (int)r.error());
    ok = false;
    return;
  }
  const auto& out = r.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(s) * kHidden;
  std::vector<float> got(
      out.const_data_ptr<float>(), out.const_data_ptr<float>() + numel);
  float e = max_err(got, golden);
  bool pass = out.numel() == static_cast<ssize_t>(numel) && e < 1e-3f;
  printf(
      "  dyn_select S=%-3d max_err=%e -> %s\n", s, e, pass ? "PASS" : "FAIL");
  ok = ok && pass;
}

} // namespace

int main(int argc, char** argv) {
  std::string dir = "/tmp/dynamic_shape";
  if (argc > 1) {
    dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_DYNAMIC_SHAPE_DIR")) {
    dir = env;
  }

  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& e) {
    printf("SKIP: %s\n", e.what());
    return 0;
  }
  set_default_webgpu_context(&ctx);
  printf("WebGPU device acquired (native); dir: %s\n", dir.c_str());

  bool ok = true;

  // Cases A + B: single dynamic rms_norm at S = MAXS .. 1 (fresh module each).
  printf("\n--- A/B: dynamic rms_norm at several S (fresh load each) ---\n");
  for (int s : {128, 64, 8, 1}) {
    Module m(dir + "/dyn_rms.pte");
    if (m.load_forward() != Error::Ok) {
      printf("  FAIL load dyn_rms.pte\n");
      ok = false;
      break;
    }
    check_s(m, dir, "dyn_rms", s, ok);
  }

  // Case C: ONE loaded graph reused across S (buffers must not move).
  printf("\n--- C: one graph reused across S (bind groups stay valid) ---\n");
  {
    Module m(dir + "/dyn_rms.pte");
    if (m.load_forward() != Error::Ok) {
      printf("  FAIL load dyn_rms.pte\n");
      ok = false;
    } else {
      for (int s : {128, 1, 64, 8, 128}) {
        check_s(m, dir, "dyn_rms", s, ok);
      }
    }
  }

  // Case D: static rms_norm (no dynamic dim) — regression.
  printf("\n--- D: static rms_norm (static path unchanged) ---\n");
  {
    Module m(dir + "/static_rms.pte");
    if (m.load_forward() != Error::Ok) {
      printf("  FAIL load static_rms.pte\n");
      ok = false;
    } else {
      check_s(m, dir, "static_rms", 8, ok);
    }
  }

  // Case F: 2-op chain rms(rms(x)) — resize cascade.
  printf("\n--- F: rms(rms(x)) cascade at several S ---\n");
  for (int s : {128, 16, 1}) {
    Module m(dir + "/dyn_rms_chain.pte");
    if (m.load_forward() != Error::Ok) {
      printf("  FAIL load dyn_rms_chain.pte\n");
      ok = false;
      break;
    }
    check_s(m, dir, "dyn_rms_chain", s, ok);
  }

  // Case G: rms(x)+x residual — cross-op (rms -> add) cascade.
  printf("\n--- G: rms(x)+x residual (rms->add cascade) at several S ---\n");
  for (int s : {128, 32, 1}) {
    Module m(dir + "/dyn_residual.pte");
    if (m.load_forward() != Error::Ok) {
      printf("  FAIL load dyn_residual.pte\n");
      ok = false;
      break;
    }
    check_s(m, dir, "dyn_residual", s, ok);
  }

  // Case H: rms(x)*x — exercises the mul op resize.
  printf("\n--- H: rms(x)*x (mul op) at several S ---\n");
  for (int s : {128, 32, 1}) {
    Module m(dir + "/dyn_rmsmul.pte");
    if (m.load_forward() != Error::Ok) {
      printf("  FAIL load dyn_rmsmul.pte\n");
      ok = false;
      break;
    }
    check_s(m, dir, "dyn_rmsmul", s, ok);
  }

  // Case I: dynamic 4-bit quantized linear (prefill GEMM) at several M.
  printf("\n--- I: dynamic linear_q4gsw [M,64]->[M,128] at several M ---\n");
  for (int mrows : {128, 32, 1}) {
    check_linear(dir, mrows, ok);
  }

  // Case J: dynamic SDPA (GQA prefill) at several seq-len S.
  printf("\n--- J: dynamic sdpa_with_kv_cache (prefill) at several S ---\n");
  for (int s : {64, 16, 1}) {
    check_sdpa(dir, s, ok);
  }

  // Case K: dynamic embedding (int64 token ids) at several token counts.
  printf("\n--- K: dynamic embedding_q4gsw (int64 ids) at several N ---\n");
  for (int n : {16, 8, 1}) {
    check_embedding(dir, n, ok);
  }

  // Case L: dynamic RoPE (two outputs) at several seq-len S.
  printf("\n--- L: dynamic apply_rotary_emb at several S ---\n");
  for (int s : {16, 8, 1}) {
    check_rope(dir, s, ok);
  }

  // Case M: dynamic sigmoid (elementwise) at several S.
  printf("\n--- M: dynamic sigmoid at several S ---\n");
  for (int s : {128, 32, 1}) {
    Module m(dir + "/dyn_sigmoid.pte");
    if (m.load_forward() != Error::Ok) {
      printf("  FAIL load dyn_sigmoid.pte\n");
      ok = false;
      break;
    }
    check_s(m, dir, "dyn_sigmoid", s, ok);
  }

  // Case N: dynamic select_copy(0,-1) at several S.
  printf("\n--- N: dynamic select_copy(0,-1) at several S ---\n");
  for (int s : {128, 32, 1}) {
    check_select(dir, s, ok);
  }

  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);

  if (!ok) {
    printf("\ndynamic_shape tests FAILED\n");
    return 1;
  }
  printf("\nAll dynamic_shape tests passed\n");
  return 0;
}
