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
//   A  dyn_rms at S=MAXS                  -> golden (static-equivalent)
//   B  dyn_rms at S < MAXS (64, 8, 1)     -> golden (resize shrinks dispatch)
//   C  ONE loaded graph reused across S   -> all golden (buffers never moved =>
//                                            bind groups stayed valid)
//   D  static_rms (no dynamic dim)        -> golden (static path unchanged)
//   F  dyn_rms_chain (rms(rms(x))) at 3 S -> golden (resize CASCADE, DD-4)
//   G  rms+residual  H rms*x  I dyn_linear  J sdpa_dyn  K emb_dyn  L rope_dyn
//   M  dyn_sigmoid   N dyn_select (select_copy(0,-1), dynamic S)
// .pte + goldens from test/ops/dynamic_shape/test_dynamic_shape_export.py.
//
// Artifacts dir: $WEBGPU_DYNAMIC_SHAPE_DIR, else argv[1], else
// /tmp/dynamic_shape.

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUQueryPool.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

constexpr int kHidden = 64;

// Artifacts directory; set from env/argv in main() before RUN_ALL_TESTS().
std::string g_dir; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

#ifdef WGPU_BACKEND_ENABLE_PROFILING
std::vector<std::string> current_profile_names() {
  const auto* context = get_default_webgpu_context();
  if (context == nullptr || context->querypool == nullptr) {
    return {};
  }
  std::vector<std::string> names;
  for (const auto& duration : context->querypool->results()) {
    names.push_back(duration.kernel_name);
  }
  return names;
}

bool contains_name(
    const std::vector<std::string>& names,
    const std::string& expected) {
  return std::find(names.begin(), names.end(), expected) != names.end();
}
#endif

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

// Run a [1,1,S,kHidden] input through `m` and compare to the golden. Shared by
// every single-output rms-shaped case (A-H, M).
void check_s(Module& m, const std::string& prefix, int s) {
  const std::string base = g_dir + "/" + prefix + ".S" + std::to_string(s);
  auto input = read_bin(base + ".input.bin");
  ASSERT_FALSE(input.empty()) << "missing input: " << prefix << ".S" << s;
  ASSERT_EQ(input.size(), static_cast<size_t>(s) * kHidden)
      << "wrong input size: " << prefix << ".S" << s;
  auto t = make_tensor_ptr({1, 1, s, kHidden}, std::move(input));
  auto r = m.forward({EValue(t)});
  ASSERT_TRUE(r.ok() && !r.get().empty() && r.get()[0].isTensor())
      << prefix << " S=" << s
      << " forward failed (err=" << (r.ok() ? 0 : (int)r.error()) << ")";
  const auto& out = r.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(s) * kHidden;
  // Output EValue must have been resized to the live shape.
  ASSERT_EQ(static_cast<size_t>(out.numel()), numel)
      << prefix << " S=" << s << " output numel mismatch";
  const float* d = out.const_data_ptr<float>();
  std::vector<float> got(d, d + numel);
  auto golden = read_bin(base + ".golden.bin");
  const float e = max_err(got, golden);
  EXPECT_LT(e, 1e-3f) << prefix << " S=" << s << " max_err=" << e
                      << " (got.size=" << got.size()
                      << " golden.size=" << golden.size() << ")";
}

// Dynamic quantized linear: input [M, kLinK] -> output [M, n]. kLinN is the
// register-tiled/bicol config; kLinNShmem (N>=2048) routes to the shmem GEMM.
constexpr int kLinK = 64;
constexpr int kLinAltK = 72;
constexpr int kLinN = 128;
constexpr int kLinNShmem = 2048;
// Run <prefix> at [m_rows, kLinK] on an already-loaded module (so it can be
// reused across M without a fresh load), and compare to the golden.
void run_linear(
    Module& m,
    int m_rows,
    const char* prefix,
    int n,
    int k = kLinK) {
  const std::string base = g_dir + "/" + prefix + ".S" + std::to_string(m_rows);
  auto input = read_bin(base + ".input.bin");
  auto golden = read_bin(base + ".golden.bin");
  ASSERT_FALSE(input.empty()) << "missing " << prefix << ".S" << m_rows;
  auto t = make_tensor_ptr({m_rows, k}, std::move(input));
  auto r = m.forward({EValue(t)});
  ASSERT_TRUE(r.ok() && !r.get().empty() && r.get()[0].isTensor())
      << prefix << " M=" << m_rows << " forward failed";
  const auto& out = r.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(m_rows) * n;
  ASSERT_EQ(static_cast<size_t>(out.numel()), numel)
      << prefix << " M=" << m_rows << " output numel mismatch";
  std::vector<float> got(
      out.const_data_ptr<float>(), out.const_data_ptr<float>() + numel);
  const float e = max_err(got, golden);
  // 4-bit quant: looser tol (the kernel mirrors the dequant-matmul reference).
  EXPECT_LT(e, 5e-3f) << prefix << " M=" << m_rows << " max_err=" << e;
}

void check_linear(int m_rows) {
  Module m(g_dir + "/dyn_linear.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_linear.pte";
  run_linear(m, m_rows, "dyn_linear", kLinN);
}

void check_linear_shmem(int m_rows) {
  Module m(g_dir + "/dyn_linear_shmem.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_linear_shmem.pte";
  run_linear(m, m_rows, "dyn_linear_shmem", kLinNShmem, kLinAltK);
}

void check_linear_tiled(int m_rows) {
  Module m(g_dir + "/dyn_linear_tiled.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_linear_tiled.pte";
  run_linear(m, m_rows, "dyn_linear_tiled", kLinN, kLinAltK);
}

constexpr int kSwiGluWidth = 8192;
constexpr int kSwiGluSmallWidth = 64;
constexpr int kSwiGluK = 64;

void run_swiglu(
    Module& module,
    int m_rows,
    const char* prefix,
    int width,
    bool separate_inputs = false) {
  const std::string base =
      g_dir + "/" + prefix + ".S" + std::to_string(m_rows) + ".";
  auto input = read_bin(base + "input.bin");
  auto golden = read_bin(base + "golden.bin");
  ASSERT_FALSE(input.empty() || golden.empty())
      << "missing " << prefix << ".S" << m_rows;
  auto input_tensor = make_tensor_ptr({m_rows, kSwiGluK}, std::move(input));
  std::vector<EValue> inputs{EValue(input_tensor)};
  decltype(input_tensor) up_input_tensor;
  if (separate_inputs) {
    auto up_input = read_bin(base + "up_input.bin");
    ASSERT_FALSE(up_input.empty());
    up_input_tensor = make_tensor_ptr({m_rows, kSwiGluK}, std::move(up_input));
    inputs.emplace_back(up_input_tensor);
  }
  auto result = module.forward(inputs);
  ASSERT_TRUE(
      result.ok() && result.get().size() == 1 && result.get()[0].isTensor())
      << prefix << " M=" << m_rows << " forward failed";
  const auto& output = result.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(m_rows) * width;
  ASSERT_EQ(static_cast<size_t>(output.numel()), numel);
  std::vector<float> got(
      output.const_data_ptr<float>(), output.const_data_ptr<float>() + numel);
  EXPECT_LT(max_err(got, golden), 1e-2f) << prefix << " M=" << m_rows;
}

void run_swiglu_outputs(
    Module& module,
    int m_rows,
    const char* prefix,
    size_t output_count) {
  const std::string base =
      g_dir + "/" + prefix + ".S" + std::to_string(m_rows) + ".";
  auto input = read_bin(base + "input.bin");
  ASSERT_FALSE(input.empty());
  auto input_tensor = make_tensor_ptr({m_rows, kSwiGluK}, std::move(input));
  auto result = module.forward({EValue(input_tensor)});
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result.get().size(), output_count);
  const size_t numel = static_cast<size_t>(m_rows) * kSwiGluSmallWidth;
  for (size_t index = 0; index < result.get().size(); ++index) {
    ASSERT_TRUE(result.get()[index].isTensor());
    const auto& output = result.get()[index].toTensor();
    ASSERT_EQ(static_cast<size_t>(output.numel()), numel);
    std::vector<float> got(
        output.const_data_ptr<float>(), output.const_data_ptr<float>() + numel);
    const auto golden =
        read_bin(base + "golden" + std::to_string(index) + ".bin");
    EXPECT_LT(max_err(got, golden), 1e-2f) << "output " << index;
  }
}

void run_swiglu_graph_outputs(Module& module, int m_rows) {
  run_swiglu_outputs(module, m_rows, "dyn_swiglu_graph_outputs", 4);
}

// Dynamic SDPA (GQA prefill, input_pos=0): q[1,s,hq,d] k/v[1,s,hkv,d]
// caches[1,cmax,hkv,d]; attn output [1,s,hq,d] selected by shape (3 outputs).
constexpr int kSdHq = 8, kSdHkv = 2, kSdD = 16, kSdCmax = 64;
void run_sdpa_case(
    Module& m,
    int s,
    const char* prefix,
    int hq,
    int hkv,
    int d,
    int cmax) {
  const std::string b = g_dir + "/" + prefix + ".S" + std::to_string(s) + ".";
  auto q = read_bin(b + "q.bin");
  auto k = read_bin(b + "k.bin");
  auto v = read_bin(b + "v.bin");
  auto kc = read_bin(b + "kc.bin");
  auto vc = read_bin(b + "vc.bin");
  auto golden = read_bin(b + "golden.bin");
  ASSERT_FALSE(
      q.empty() || k.empty() || v.empty() || kc.empty() || vc.empty() ||
      golden.empty())
      << "missing sdpa_dyn.S" << s;
  auto tq = make_tensor_ptr({1, s, hq, d}, std::move(q));
  auto tk = make_tensor_ptr({1, s, hkv, d}, std::move(k));
  auto tv = make_tensor_ptr({1, s, hkv, d}, std::move(v));
  auto tkc = make_tensor_ptr({1, cmax, hkv, d}, std::move(kc));
  auto tvc = make_tensor_ptr({1, cmax, hkv, d}, std::move(vc));
  auto r =
      m.forward({EValue(tq), EValue(tk), EValue(tv), EValue(tkc), EValue(tvc)});
  ASSERT_TRUE(r.ok()) << "sdpa S=" << s
                      << " forward failed (err=" << (int)r.error() << ")";
  // Select the attn output by full shape [1,s,hq,d] (never numel).
  const float* attn = nullptr;
  const size_t numel = static_cast<size_t>(s) * hq * d;
  for (size_t i = 0; i < r.get().size(); i++) {
    if (!r.get()[i].isTensor()) {
      continue;
    }
    const auto& t = r.get()[i].toTensor();
    if (t.dim() == 4 && t.size(1) == s && t.size(2) == hq && t.size(3) == d) {
      attn = t.const_data_ptr<float>();
      break;
    }
  }
  ASSERT_NE(attn, nullptr) << "sdpa S=" << s << ": no attn output of shape [1,"
                           << s << "," << hq << "," << d << "]";
  std::vector<float> got(attn, attn + numel);
  const float e = max_err(got, golden);
  EXPECT_LT(e, 2e-3f) << "sdpa_dyn S=" << s << " max_err=" << e;
}

void run_sdpa(Module& m, int s) {
  run_sdpa_case(m, s, "sdpa_dyn", kSdHq, kSdHkv, kSdD, kSdCmax);
}

void check_sdpa(int s) {
  Module m(g_dir + "/sdpa_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "sdpa_dyn S=" << s << " load";
  run_sdpa(m, s);
}

void run_combined_routes(Module& m, int s) {
  const std::string b = g_dir + "/combined_routes.S" + std::to_string(s) + ".";
  auto x = read_bin(b + "x.bin");
  auto q = read_bin(b + "q.bin");
  auto k = read_bin(b + "k.bin");
  auto v = read_bin(b + "v.bin");
  auto kc = read_bin(b + "kc.bin");
  auto vc = read_bin(b + "vc.bin");
  auto golden = read_bin(b + "golden.bin");
  ASSERT_FALSE(
      x.empty() || q.empty() || k.empty() || v.empty() || kc.empty() ||
      vc.empty() || golden.empty())
      << "missing combined_routes.S" << s;
  auto tx = make_tensor_ptr({s, kLinK}, std::move(x));
  auto tq = make_tensor_ptr({1, s, kSdHq, kSdD}, std::move(q));
  auto tk = make_tensor_ptr({1, s, kSdHkv, kSdD}, std::move(k));
  auto tv = make_tensor_ptr({1, s, kSdHkv, kSdD}, std::move(v));
  auto tkc = make_tensor_ptr({1, kSdCmax, kSdHkv, kSdD}, std::move(kc));
  auto tvc = make_tensor_ptr({1, kSdCmax, kSdHkv, kSdD}, std::move(vc));
  auto result = m.forward(
      {EValue(tx),
       EValue(tq),
       EValue(tk),
       EValue(tv),
       EValue(tkc),
       EValue(tvc)});
  ASSERT_TRUE(result.ok()) << "combined routes S=" << s
                           << " forward failed (err=" << (int)result.error()
                           << ")";
  const float* attn = nullptr;
  const size_t numel = static_cast<size_t>(s) * kSdHq * kSdD;
  for (const auto& output : result.get()) {
    if (!output.isTensor()) {
      continue;
    }
    const auto& tensor = output.toTensor();
    if (tensor.dim() == 4 && tensor.size(1) == s && tensor.size(2) == kSdHq &&
        tensor.size(3) == kSdD) {
      attn = tensor.const_data_ptr<float>();
      break;
    }
  }
  ASSERT_NE(attn, nullptr);
  const std::vector<float> got(attn, attn + numel);
  EXPECT_LT(max_err(got, golden), 1e-2f) << "combined_routes S=" << s;
}

// Dynamic embedding: int64 token ids [N] -> [N, kEmbDim] fp32. The int64 host
// input exercises copy_inputs' int64->int32 narrow path under dynamic shapes.
constexpr int kEmbDim = 64;
// Run emb_dyn at N tokens on an already-loaded module (so it can be reused
// across N), and compare to the golden.
void run_embedding(Module& m, int n) {
  const std::string b = g_dir + "/emb_dyn.S" + std::to_string(n) + ".";
  std::ifstream f(b + "idx.bin", std::ios::binary | std::ios::ate);
  ASSERT_TRUE(f.good()) << "missing emb_dyn.S" << n;
  const std::streamsize nb = f.tellg();
  ASSERT_GE(nb, 0) << "missing emb_dyn.S" << n;
  f.seekg(0);
  std::vector<int64_t> idx(static_cast<size_t>(nb) / sizeof(int64_t));
  f.read(reinterpret_cast<char*>(idx.data()), nb);
  ASSERT_EQ(idx.size(), static_cast<size_t>(n))
      << "wrong emb_dyn idx size S" << n;
  auto golden = read_bin(b + "golden.bin");
  auto t = make_tensor_ptr({n}, std::move(idx)); // int64 (Long) host input
  auto r = m.forward({EValue(t)});
  ASSERT_TRUE(r.ok() && !r.get().empty() && r.get()[0].isTensor())
      << "emb N=" << n
      << " forward failed (err=" << (r.ok() ? 0 : (int)r.error()) << ")";
  const auto& out = r.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(n) * kEmbDim;
  ASSERT_EQ(static_cast<size_t>(out.numel()), numel)
      << "emb N=" << n << " output numel mismatch";
  std::vector<float> got(
      out.const_data_ptr<float>(), out.const_data_ptr<float>() + numel);
  const float e = max_err(got, golden);
  EXPECT_LT(e, 5e-3f) << "emb_dyn N=" << n << " max_err=" << e;
}

void check_embedding(int n) {
  Module m(g_dir + "/emb_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load emb_dyn.pte";
  run_embedding(m, n);
}

// Dynamic RoPE: xq[1,s,nh,hd] xk[1,s,nkv,hd] freqs[s,hd/2] -> xq_out/xk_out
// (2 outputs, selected by head count nh != nkv).
constexpr int kRopeNH = 8, kRopeNKV = 2, kRopeHD = 64;
// Run rope_dyn at seq-len s on an already-loaded module (so it can be reused
// across s), comparing xq_out/xk_out (selected by head count) to the goldens.
void run_rope(Module& m, int s) {
  const std::string b = g_dir + "/rope_dyn.S" + std::to_string(s) + ".";
  auto xq = read_bin(b + "xq.bin");
  auto xk = read_bin(b + "xk.bin");
  auto fc = read_bin(b + "fc.bin");
  auto fs = read_bin(b + "fs.bin");
  auto gq = read_bin(b + "gq.bin");
  auto gk = read_bin(b + "gk.bin");
  ASSERT_FALSE(
      xq.empty() || xk.empty() || fc.empty() || fs.empty() || gq.empty() ||
      gk.empty())
      << "missing rope_dyn.S" << s;
  auto txq = make_tensor_ptr({1, s, kRopeNH, kRopeHD}, std::move(xq));
  auto txk = make_tensor_ptr({1, s, kRopeNKV, kRopeHD}, std::move(xk));
  auto tfc = make_tensor_ptr({s, kRopeHD / 2}, std::move(fc));
  auto tfs = make_tensor_ptr({s, kRopeHD / 2}, std::move(fs));
  auto r = m.forward({EValue(txq), EValue(txk), EValue(tfc), EValue(tfs)});
  ASSERT_TRUE(r.ok()) << "rope S=" << s
                      << " forward failed (err=" << (int)r.error() << ")";
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
  ASSERT_TRUE(oq != nullptr && okp != nullptr)
      << "rope S=" << s << ": missing xq_out/xk_out by shape";
  std::vector<float> gotq(oq, oq + static_cast<size_t>(s) * kRopeNH * kRopeHD);
  std::vector<float> gotk(
      okp, okp + static_cast<size_t>(s) * kRopeNKV * kRopeHD);
  const float e = std::fmax(max_err(gotq, gq), max_err(gotk, gk));
  EXPECT_LT(e, 1e-3f) << "rope_dyn S=" << s << " max_err=" << e;
}

void check_rope(int s) {
  Module m(g_dir + "/rope_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load rope_dyn.pte";
  run_rope(m, s);
}

// Dynamic select_copy(0,-1): input [2,1,S,kHidden] -> output [1,S,kHidden]. The
// negative index resolves against the (static) leading dim live; the dynamic S
// flows to the output, so the resize hook recomputes its dispatch each S.
constexpr int kSelLead = 2;
// Run dyn_select at seq-len s on an already-loaded module (so it can be reused
// across s), and compare to the golden.
void run_select(Module& m, int s) {
  const std::string base = g_dir + "/dyn_select.S" + std::to_string(s);
  auto input = read_bin(base + ".input.bin");
  auto golden = read_bin(base + ".golden.bin");
  ASSERT_FALSE(input.empty() || golden.empty()) << "missing dyn_select.S" << s;
  auto t = make_tensor_ptr({kSelLead, 1, s, kHidden}, std::move(input));
  auto r = m.forward({EValue(t)});
  ASSERT_TRUE(r.ok() && !r.get().empty() && r.get()[0].isTensor())
      << "select S=" << s
      << " forward failed (err=" << (r.ok() ? 0 : (int)r.error()) << ")";
  const auto& out = r.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(s) * kHidden;
  ASSERT_EQ(static_cast<size_t>(out.numel()), numel)
      << "select S=" << s << " output numel mismatch";
  std::vector<float> got(
      out.const_data_ptr<float>(), out.const_data_ptr<float>() + numel);
  const float e = max_err(got, golden);
  EXPECT_LT(e, 1e-3f) << "dyn_select S=" << s << " max_err=" << e;
}

void check_select(int s) {
  Module m(g_dir + "/dyn_select.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_select.pte";
  run_select(m, s);
}

} // namespace

// A + B: single dynamic rms_norm at S = MAXS .. 1 (fresh module load each S).
TEST(DynamicShape, RmsNormFreshLoad) {
  for (int s : {128, 64, 8, 1}) {
    Module m(g_dir + "/dyn_rms.pte");
    ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_rms.pte";
    check_s(m, "dyn_rms", s);
  }
}

// C: ONE loaded graph reused across S (buffers must not move => bind groups
// stay valid).
TEST(DynamicShape, RmsNormReusedGraph) {
  Module m(g_dir + "/dyn_rms.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_rms.pte";
  for (int s : {128, 1, 64, 8, 128}) {
    check_s(m, "dyn_rms", s);
  }
}

// C2: grow-only reuse — one loaded rms graph run smallest -> largest, so the
// FIRST resize grows the dispatch (every other reuse test starts at MAXS and
// only shrinks; this catches a hook with a shrink-only short-circuit).
TEST(DynamicShape, RmsNormGrowReused) {
  Module m(g_dir + "/dyn_rms.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_rms.pte";
  for (int s : {1, 8, 64, 128}) {
    check_s(m, "dyn_rms", s);
  }
}

// D: static rms_norm (no dynamic dim) — regression that the static path is
// unchanged.
TEST(DynamicShape, StaticRmsNorm) {
  Module m(g_dir + "/static_rms.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load static_rms.pte";
  check_s(m, "static_rms", 8);
}

// F: 2-op chain rms(rms(x)) — resize cascade.
TEST(DynamicShape, RmsChainCascade) {
  for (int s : {128, 16, 1}) {
    Module m(g_dir + "/dyn_rms_chain.pte");
    ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_rms_chain.pte";
    check_s(m, "dyn_rms_chain", s);
  }
}

// F2: cascade graph REUSED across S — one loaded rms(rms(x)) graph run across S
// (op1's output-resize feeds op2's input-resize on a persistent graph; the
// single-op reuse tests don't cover an inter-op resize cascade under reuse).
TEST(DynamicShape, RmsChainCascadeReusedGraph) {
  Module m(g_dir + "/dyn_rms_chain.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_rms_chain.pte";
  for (int s : {128, 16, 1, 128}) {
    check_s(m, "dyn_rms_chain", s);
  }
}

// G: rms(x)+x residual — cross-op (rms -> add) cascade.
TEST(DynamicShape, RmsResidualCascade) {
  for (int s : {128, 32, 1}) {
    Module m(g_dir + "/dyn_residual.pte");
    ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_residual.pte";
    check_s(m, "dyn_residual", s);
  }
}

// H: rms(x)*x — exercises the mul op resize.
TEST(DynamicShape, RmsMul) {
  for (int s : {128, 32, 1}) {
    Module m(g_dir + "/dyn_rmsmul.pte");
    ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_rmsmul.pte";
    check_s(m, "dyn_rmsmul", s);
  }
}

// I: dynamic 4-bit quantized linear (prefill GEMM) at several M.
TEST(DynamicShape, QuantizedLinear) {
  for (int m_rows : {128, 32, 1}) {
    check_linear(m_rows);
  }
}

// I2: dynamic linear reusing ONE loaded graph across M (buffers must not move
// => bind groups stay valid; the resize hook recomputes dispatch each M).
TEST(DynamicShape, QuantizedLinearReusedGraph) {
  Module m(g_dir + "/dyn_linear.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_linear.pte";
  for (int m_rows : {128, 32, 1, 128}) {
    run_linear(m, m_rows, "dyn_linear", kLinN);
  }
}

// I3: K=72 disables Steel; N=2048 forces shmem for M>1 and bicol for M=1.
TEST(DynamicShape, QuantizedLinearShmem) {
  for (int m_rows : {128, 32, 1}) {
    check_linear_shmem(m_rows);
  }
}

// I4: shmem-routed linear reusing ONE loaded graph across M.
TEST(DynamicShape, QuantizedLinearShmemReusedGraph) {
  Module m(g_dir + "/dyn_linear_shmem.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_linear_shmem.pte";
  for (int m_rows : {128, 32, 1, 128}) {
    run_linear(m, m_rows, "dyn_linear_shmem", kLinNShmem, kLinAltK);
  }
}

// I5: K=72 disables Steel; N=128 keeps the tiled fallback for M>1.
TEST(DynamicShape, QuantizedLinearTiled) {
  for (int m_rows : {128, 32, 1}) {
    check_linear_tiled(m_rows);
  }
}

TEST(DynamicShape, QuantizedLinearTiledReusedGraph) {
  Module m(g_dir + "/dyn_linear_tiled.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_linear_tiled.pte";
  for (int m_rows : {128, 1, 32, 1, 128}) {
    run_linear(m, m_rows, "dyn_linear_tiled", kLinN, kLinAltK);
  }
}

#ifdef WGPU_BACKEND_ENABLE_PROFILING
TEST(DynamicShape, CombinedLiveRoutesProfile) {
  const auto* context = get_default_webgpu_context();
  if (std::getenv("WEBGPU_TIMESTAMP_QUERY") == nullptr || context == nullptr ||
      !context->timestamp_supported) {
    GTEST_SKIP() << "timestamp queries unavailable";
  }
  Module m(g_dir + "/combined_routes.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load combined_routes.pte";
  for (int s : {64, 1, 16, 1, 64}) {
    run_combined_routes(m, s);
    const auto names = current_profile_names();
    ASSERT_EQ(names.size(), s == 1 ? 6 : 7);
    EXPECT_EQ(std::count(names.begin(), names.end(), ""), 1);
    EXPECT_EQ(
        std::count(names.begin(), names.end(), "linear_q4gsw_coop4_bicol"),
        s == 1 ? 1 : 0);
    EXPECT_EQ(
        std::count(names.begin(), names.end(), "linear_q4gsw_steel"),
        s != 1 ? 1 : 0);
    EXPECT_FALSE(contains_name(names, "linear_q4gsw_shmem"));
    EXPECT_FALSE(contains_name(names, "linear_q4gsw_tiled"));
    EXPECT_EQ(std::count(names.begin(), names.end(), "update_cache"), 2);
    EXPECT_EQ(std::count(names.begin(), names.end(), "fd_split"), s == 1);
    EXPECT_EQ(std::count(names.begin(), names.end(), "fd_reduce"), s == 1);
    EXPECT_EQ(
        std::count(names.begin(), names.end(), "sdpa_compute_attn_weights"),
        s != 1);
    EXPECT_EQ(std::count(names.begin(), names.end(), "sdpa_softmax"), s != 1);
    EXPECT_EQ(
        std::count(names.begin(), names.end(), "sdpa_compute_out"), s != 1);
  }
}

TEST(DynamicShape, StaticRouteProfiles) {
  const auto* context = get_default_webgpu_context();
  if (std::getenv("WEBGPU_TIMESTAMP_QUERY") == nullptr || context == nullptr ||
      !context->timestamp_supported) {
    GTEST_SKIP() << "timestamp queries unavailable";
  }

  Module linear_m1(g_dir + "/static_linear_m1.pte");
  ASSERT_EQ(linear_m1.load_forward(), Error::Ok);
  run_linear(linear_m1, 1, "static_linear_m1", kLinN);
  auto names = current_profile_names();
  ASSERT_EQ(names.size(), 1);
  EXPECT_EQ(
      std::count(names.begin(), names.end(), "linear_q4gsw_coop4_bicol"), 1);

  Module linear_m32(g_dir + "/static_linear_m32.pte");
  ASSERT_EQ(linear_m32.load_forward(), Error::Ok);
  run_linear(linear_m32, 32, "static_linear_m32", kLinN);
  names = current_profile_names();
  ASSERT_EQ(names.size(), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "linear_q4gsw_steel"), 1);

  Module linear_shmem(g_dir + "/dyn_linear_shmem.pte");
  ASSERT_EQ(linear_shmem.load_forward(), Error::Ok);
  run_linear(linear_shmem, 128, "dyn_linear_shmem", kLinNShmem, kLinAltK);
  names = current_profile_names();
  ASSERT_EQ(names.size(), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "linear_q4gsw_shmem"), 1);

  Module linear_tiled(g_dir + "/dyn_linear_tiled.pte");
  ASSERT_EQ(linear_tiled.load_forward(), Error::Ok);
  run_linear(linear_tiled, 128, "dyn_linear_tiled", kLinN, kLinAltK);
  names = current_profile_names();
  ASSERT_EQ(names.size(), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "linear_q4gsw_tiled"), 1);

  Module sdpa_s1(g_dir + "/static_sdpa_s1.pte");
  ASSERT_EQ(sdpa_s1.load_forward(), Error::Ok);
  run_sdpa_case(sdpa_s1, 1, "static_sdpa_s1", kSdHq, kSdHkv, kSdD, kSdCmax);
  names = current_profile_names();
  ASSERT_EQ(names.size(), 4);
  EXPECT_EQ(std::count(names.begin(), names.end(), "fd_split"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "fd_reduce"), 1);

  Module sdpa_s16(g_dir + "/static_sdpa_s16.pte");
  ASSERT_EQ(sdpa_s16.load_forward(), Error::Ok);
  run_sdpa_case(sdpa_s16, 16, "static_sdpa_s16", kSdHq, kSdHkv, kSdD, kSdCmax);
  names = current_profile_names();
  ASSERT_EQ(names.size(), 5);
  EXPECT_EQ(
      std::count(names.begin(), names.end(), "sdpa_compute_attn_weights"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "sdpa_softmax"), 1);
  EXPECT_EQ(std::count(names.begin(), names.end(), "sdpa_compute_out"), 1);
}
#endif

// J: dynamic SDPA reuses one graph across prefill and FlashDecoding shapes.
TEST(DynamicShape, Sdpa) {
  for (int s : {64, 16, 1}) {
    check_sdpa(s);
  }
}

TEST(DynamicShape, SdpaReusedGraph) {
  Module m(g_dir + "/sdpa_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load sdpa_dyn.pte";
  for (int s : {64, 1, 16, 1, 64}) {
    run_sdpa(m, s);
  }
}

TEST(DynamicShape, CombinedLiveRoutes) {
  Module m(g_dir + "/combined_routes.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load combined_routes.pte";
  for (int s : {64, 1, 16, 1, 64}) {
    run_combined_routes(m, s);
  }
}

TEST(DynamicShape, SdpaWideMaterializedOnly) {
  Module m(g_dir + "/sdpa_wide.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load sdpa_wide.pte";
  for (int s : {16, 1, 16}) {
    run_sdpa_case(m, s, "sdpa_wide", 8, 2, 132, 16);
  }
}

#ifdef WGPU_BACKEND_ENABLE_PROFILING
TEST(DynamicShape, SdpaLiveRoutesProfile) {
  const auto* context = get_default_webgpu_context();
  if (std::getenv("WEBGPU_TIMESTAMP_QUERY") == nullptr || context == nullptr ||
      !context->timestamp_supported) {
    GTEST_SKIP() << "timestamp queries unavailable";
  }
  Module m(g_dir + "/sdpa_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load sdpa_dyn.pte";
  for (int s : {64, 1, 16, 1, 64}) {
    run_sdpa(m, s);
    const auto names = current_profile_names();
    ASSERT_EQ(names.size(), s == 1 ? 4 : 5);
    EXPECT_EQ(std::count(names.begin(), names.end(), "update_cache"), 2);
    EXPECT_EQ(std::count(names.begin(), names.end(), "fd_split"), s == 1);
    EXPECT_EQ(std::count(names.begin(), names.end(), "fd_reduce"), s == 1);
    EXPECT_EQ(
        std::count(names.begin(), names.end(), "sdpa_compute_attn_weights"),
        s != 1);
    EXPECT_EQ(std::count(names.begin(), names.end(), "sdpa_softmax"), s != 1);
    EXPECT_EQ(
        std::count(names.begin(), names.end(), "sdpa_compute_out"), s != 1);
  }
}
#endif

// K: dynamic embedding (int64 token ids) at several token counts.
TEST(DynamicShape, Embedding) {
  for (int n : {16, 8, 1}) {
    check_embedding(n);
  }
}

// K2: dynamic embedding reusing ONE loaded graph across N (buffers must not
// move
// => the multi-buffer bind group stays valid; resize recomputes blocks each N).
TEST(DynamicShape, EmbeddingReusedGraph) {
  Module m(g_dir + "/emb_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load emb_dyn.pte";
  for (int n : {16, 8, 1, 16}) {
    run_embedding(m, n);
  }
}

// L: dynamic RoPE (two outputs) at several seq-len S.
TEST(DynamicShape, Rope) {
  for (int s : {16, 8, 1}) {
    check_rope(s);
  }
}

// L2: dynamic RoPE reusing ONE loaded graph across S (buffers must not move =>
// bind groups stay valid; the resize hook recomputes both outputs each S).
TEST(DynamicShape, RopeReusedGraph) {
  Module m(g_dir + "/rope_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load rope_dyn.pte";
  for (int s : {16, 8, 1, 16}) {
    run_rope(m, s);
  }
}

// M: dynamic sigmoid (elementwise) at several S.
TEST(DynamicShape, Sigmoid) {
  for (int s : {128, 32, 1}) {
    Module m(g_dir + "/dyn_sigmoid.pte");
    ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_sigmoid.pte";
    check_s(m, "dyn_sigmoid", s);
  }
}

// M2: dynamic sigmoid reusing ONE loaded graph across S.
TEST(DynamicShape, SigmoidReusedGraph) {
  Module m(g_dir + "/dyn_sigmoid.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_sigmoid.pte";
  for (int s : {128, 32, 1, 128}) {
    check_s(m, "dyn_sigmoid", s);
  }
}

TEST(DynamicShape, SwiGluReusedGraph) {
  Module module(g_dir + "/dyn_swiglu.pte");
  ASSERT_EQ(module.load_forward(), Error::Ok) << "load dyn_swiglu.pte";
  for (int m_rows : {512, 128, 1, 512}) {
    run_swiglu(module, m_rows, "dyn_swiglu", kSwiGluWidth);
  }
}

TEST(DynamicShape, SwiGluCommutativeAndOwnership) {
  for (const char* prefix :
       {"dyn_swiglu_inner_reversed", "dyn_swiglu_outer_reversed"}) {
    Module module(g_dir + "/" + prefix + ".pte");
    ASSERT_EQ(module.load_forward(), Error::Ok) << "load " << prefix;
    run_swiglu(module, 128, prefix, kSwiGluSmallWidth);
  }

  Module negative(g_dir + "/dyn_swiglu_extra_gate_consumer.pte");
  ASSERT_EQ(negative.load_forward(), Error::Ok)
      << "load dyn_swiglu_extra_gate_consumer";
  run_swiglu(
      negative, 128, "dyn_swiglu_extra_gate_consumer", kSwiGluSmallWidth);

  Module graph_outputs(g_dir + "/dyn_swiglu_graph_outputs.pte");
  ASSERT_EQ(graph_outputs.load_forward(), Error::Ok)
      << "load dyn_swiglu_graph_outputs";
  run_swiglu_graph_outputs(graph_outputs, 128);

  for (const char* prefix :
       {"dyn_swiglu_extra_sigmoid_consumer",
        "dyn_swiglu_extra_silu_consumer"}) {
    Module module(g_dir + "/" + prefix + ".pte");
    ASSERT_EQ(module.load_forward(), Error::Ok) << "load " << prefix;
    run_swiglu(module, 128, prefix, kSwiGluSmallWidth);
  }

  for (const char* prefix :
       {"dyn_swiglu_gate_graph_output",
        "dyn_swiglu_sigmoid_graph_output",
        "dyn_swiglu_silu_graph_output"}) {
    Module module(g_dir + "/" + prefix + ".pte");
    ASSERT_EQ(module.load_forward(), Error::Ok) << "load " << prefix;
    run_swiglu_outputs(module, 128, prefix, 2);
  }

  Module different_inputs(g_dir + "/dyn_swiglu_different_inputs.pte");
  ASSERT_EQ(different_inputs.load_forward(), Error::Ok);
  run_swiglu(
      different_inputs,
      128,
      "dyn_swiglu_different_inputs",
      kSwiGluSmallWidth,
      true);

  Module interleaved(g_dir + "/dyn_swiglu_interleaved_q4.pte");
  ASSERT_EQ(interleaved.load_forward(), Error::Ok);
  run_swiglu(interleaved, 128, "dyn_swiglu_interleaved_q4", kSwiGluSmallWidth);
}

#ifdef WGPU_BACKEND_ENABLE_PROFILING
void expect_swiglu_profile(
    Module& module,
    int m_rows,
    const char* prefix,
    int width,
    bool expect_2d) {
  run_swiglu(module, m_rows, prefix, width);
  const auto* context = get_default_webgpu_context();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(context->querypool, nullptr);
  const auto& profile = context->querypool->results();
  ASSERT_EQ(profile.size(), 3)
      << "two q4 projections plus one fused SwiGLU dispatch expected";
  EXPECT_EQ(
      std::count_if(
          profile.begin(),
          profile.end(),
          [](const auto& duration) {
            return duration.kernel_name == "silu_mul_fused";
          }),
      1);
  EXPECT_EQ(
      std::count_if(
          profile.begin(),
          profile.end(),
          [](const auto& duration) {
            return duration.kernel_name == "mul" ||
                duration.kernel_name == "sigmoid";
          }),
      0);
  const auto fused =
      std::find_if(profile.begin(), profile.end(), [](const auto& duration) {
        return duration.kernel_name == "silu_mul_fused";
      });
  ASSERT_NE(fused, profile.end());
  EXPECT_EQ(fused->global_wg[1] > 1, expect_2d);
}

TEST(DynamicShape, SwiGluFusionProfile) {
  const auto* context = get_default_webgpu_context();
  if (std::getenv("WEBGPU_TIMESTAMP_QUERY") == nullptr || context == nullptr ||
      !context->timestamp_supported) {
    GTEST_SKIP() << "timestamp queries unavailable";
  }

  for (const char* prefix :
       {"dyn_swiglu_inner_reversed", "dyn_swiglu_outer_reversed"}) {
    Module module(g_dir + "/" + prefix + ".pte");
    ASSERT_EQ(module.load_forward(), Error::Ok) << prefix;
    expect_swiglu_profile(module, 128, prefix, kSwiGluSmallWidth, false);
  }

  Module canonical(g_dir + "/dyn_swiglu.pte");
  ASSERT_EQ(canonical.load_forward(), Error::Ok);
  for (int m_rows : {1, 128, 512}) {
    expect_swiglu_profile(
        canonical, m_rows, "dyn_swiglu", kSwiGluWidth, m_rows == 512);
  }

  Module negative(g_dir + "/dyn_swiglu_extra_gate_consumer.pte");
  ASSERT_EQ(negative.load_forward(), Error::Ok);
  run_swiglu(
      negative, 128, "dyn_swiglu_extra_gate_consumer", kSwiGluSmallWidth);
  const auto names = current_profile_names();
  EXPECT_FALSE(contains_name(names, "silu_mul_fused"));
  EXPECT_EQ(std::count(names.begin(), names.end(), "mul"), 2);

  Module graph_outputs(g_dir + "/dyn_swiglu_graph_outputs.pte");
  ASSERT_EQ(graph_outputs.load_forward(), Error::Ok);
  run_swiglu_graph_outputs(graph_outputs, 128);
  const auto graph_output_names = current_profile_names();
  EXPECT_FALSE(contains_name(graph_output_names, "silu_mul_fused"));
  EXPECT_EQ(
      std::count(graph_output_names.begin(), graph_output_names.end(), "mul"),
      2);

  for (const char* prefix :
       {"dyn_swiglu_extra_sigmoid_consumer",
        "dyn_swiglu_extra_silu_consumer"}) {
    Module module(g_dir + "/" + prefix + ".pte");
    ASSERT_EQ(module.load_forward(), Error::Ok) << prefix;
    run_swiglu(module, 128, prefix, kSwiGluSmallWidth);
    const auto consumer_names = current_profile_names();
    EXPECT_FALSE(contains_name(consumer_names, "silu_mul_fused")) << prefix;
    EXPECT_EQ(
        std::count(consumer_names.begin(), consumer_names.end(), "mul"), 2)
        << prefix;
  }

  for (const char* prefix :
       {"dyn_swiglu_gate_graph_output",
        "dyn_swiglu_sigmoid_graph_output",
        "dyn_swiglu_silu_graph_output"}) {
    Module module(g_dir + "/" + prefix + ".pte");
    ASSERT_EQ(module.load_forward(), Error::Ok) << prefix;
    run_swiglu_outputs(module, 128, prefix, 2);
    const auto output_names = current_profile_names();
    EXPECT_FALSE(contains_name(output_names, "silu_mul_fused")) << prefix;
    EXPECT_EQ(std::count(output_names.begin(), output_names.end(), "mul"), 2)
        << prefix;
  }

  Module different_inputs(g_dir + "/dyn_swiglu_different_inputs.pte");
  ASSERT_EQ(different_inputs.load_forward(), Error::Ok);
  run_swiglu(
      different_inputs,
      128,
      "dyn_swiglu_different_inputs",
      kSwiGluSmallWidth,
      true);
  const auto different_input_names = current_profile_names();
  EXPECT_FALSE(contains_name(different_input_names, "silu_mul_fused"));
  EXPECT_EQ(
      std::count(
          different_input_names.begin(), different_input_names.end(), "mul"),
      2);

  Module interleaved(g_dir + "/dyn_swiglu_interleaved_q4.pte");
  ASSERT_EQ(interleaved.load_forward(), Error::Ok);
  run_swiglu(interleaved, 128, "dyn_swiglu_interleaved_q4", kSwiGluSmallWidth);
  const auto interleaved_names = current_profile_names();
  EXPECT_EQ(interleaved_names.size(), 5);
  EXPECT_EQ(
      std::count(
          interleaved_names.begin(), interleaved_names.end(), "silu_mul_fused"),
      1);
  EXPECT_EQ(
      std::count(interleaved_names.begin(), interleaved_names.end(), "mul"), 0);
}
#endif

// N: dynamic select_copy(0,-1) at several S.
TEST(DynamicShape, Select) {
  for (int s : {128, 32, 1}) {
    check_select(s);
  }
}

// N2: dynamic select_copy reusing ONE loaded graph across S (the resize hook
// re-resolves the negative index against the LIVE leading dim each call).
TEST(DynamicShape, SelectReusedGraph) {
  Module m(g_dir + "/dyn_select.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_select.pte";
  for (int s : {128, 32, 1, 128}) {
    run_select(m, s);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Artifacts dir: env wins, else first positional arg, else default (gtest
  // flags were already stripped by InitGoogleTest above).
  g_dir = "/tmp/dynamic_shape";
  if (argc > 1) {
    g_dir = argv[1];
  }
  if (const char* env = std::getenv("WEBGPU_DYNAMIC_SHAPE_DIR")) {
    g_dir = env;
  }

  WebGPUContext ctx;
  try {
    ctx = create_webgpu_context();
  } catch (const std::exception& e) {
    std::printf("SKIP: no WebGPU device (%s)\n", e.what());
    return 0;
  }
  set_default_webgpu_context(&ctx);

  const int rc = RUN_ALL_TESTS();
  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);
  return rc;
}
