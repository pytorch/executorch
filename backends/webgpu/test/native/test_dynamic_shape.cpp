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
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <gtest/gtest.h>

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

// Dynamic quantized linear: input [M, kLinK] -> output [M, kLinN].
constexpr int kLinK = 64;
constexpr int kLinN = 128;
void check_linear(int m_rows) {
  Module m(g_dir + "/dyn_linear.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_linear.pte";
  const std::string base = g_dir + "/dyn_linear.S" + std::to_string(m_rows);
  auto input = read_bin(base + ".input.bin");
  auto golden = read_bin(base + ".golden.bin");
  ASSERT_FALSE(input.empty()) << "missing dyn_linear.S" << m_rows;
  auto t = make_tensor_ptr({m_rows, kLinK}, std::move(input));
  auto r = m.forward({EValue(t)});
  ASSERT_TRUE(r.ok() && !r.get().empty() && r.get()[0].isTensor())
      << "dyn_linear M=" << m_rows << " forward failed";
  const auto& out = r.get()[0].toTensor();
  const size_t numel = static_cast<size_t>(m_rows) * kLinN;
  ASSERT_EQ(static_cast<size_t>(out.numel()), numel)
      << "dyn_linear M=" << m_rows << " output numel mismatch";
  std::vector<float> got(
      out.const_data_ptr<float>(), out.const_data_ptr<float>() + numel);
  const float e = max_err(got, golden);
  // 4-bit quant: looser tol (the kernel mirrors the dequant-matmul reference).
  EXPECT_LT(e, 5e-3f) << "dyn_linear M=" << m_rows << " max_err=" << e;
}

// Dynamic SDPA (GQA prefill, input_pos=0): q[1,s,hq,d] k/v[1,s,hkv,d]
// caches[1,cmax,hkv,d]; attn output [1,s,hq,d] selected by shape (3 outputs).
constexpr int kSdHq = 8, kSdHkv = 2, kSdD = 16, kSdCmax = 64;
void check_sdpa(int s) {
  Module m(g_dir + "/sdpa_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "sdpa_dyn S=" << s << " load";
  const std::string b = g_dir + "/sdpa_dyn.S" + std::to_string(s) + ".";
  auto q = read_bin(b + "q.bin");
  auto k = read_bin(b + "k.bin");
  auto v = read_bin(b + "v.bin");
  auto kc = read_bin(b + "kc.bin");
  auto vc = read_bin(b + "vc.bin");
  auto golden = read_bin(b + "golden.bin");
  ASSERT_FALSE(q.empty() || golden.empty()) << "missing sdpa_dyn.S" << s;
  auto tq = make_tensor_ptr({1, s, kSdHq, kSdD}, std::move(q));
  auto tk = make_tensor_ptr({1, s, kSdHkv, kSdD}, std::move(k));
  auto tv = make_tensor_ptr({1, s, kSdHkv, kSdD}, std::move(v));
  auto tkc = make_tensor_ptr({1, kSdCmax, kSdHkv, kSdD}, std::move(kc));
  auto tvc = make_tensor_ptr({1, kSdCmax, kSdHkv, kSdD}, std::move(vc));
  auto r =
      m.forward({EValue(tq), EValue(tk), EValue(tv), EValue(tkc), EValue(tvc)});
  ASSERT_TRUE(r.ok()) << "sdpa S=" << s
                      << " forward failed (err=" << (int)r.error() << ")";
  // Select the attn output by full shape [1,s,hq,d] (never numel).
  const float* attn = nullptr;
  const size_t numel = static_cast<size_t>(s) * kSdHq * kSdD;
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
  ASSERT_NE(attn, nullptr) << "sdpa S=" << s << ": no attn output of shape [1,"
                           << s << "," << kSdHq << "," << kSdD << "]";
  std::vector<float> got(attn, attn + numel);
  const float e = max_err(got, golden);
  EXPECT_LT(e, 2e-3f) << "sdpa_dyn S=" << s << " max_err=" << e;
}

// Dynamic embedding: int64 token ids [N] -> [N, kEmbDim] fp32. The int64 host
// input exercises copy_inputs' int64->int32 narrow path under dynamic shapes.
constexpr int kEmbDim = 64;
void check_embedding(int n) {
  Module m(g_dir + "/emb_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load emb_dyn.pte";
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

// Dynamic RoPE: xq[1,s,nh,hd] xk[1,s,nkv,hd] freqs[s,hd/2] -> xq_out/xk_out
// (2 outputs, selected by head count nh != nkv).
constexpr int kRopeNH = 8, kRopeNKV = 2, kRopeHD = 64;
void check_rope(int s) {
  Module m(g_dir + "/rope_dyn.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load rope_dyn.pte";
  const std::string b = g_dir + "/rope_dyn.S" + std::to_string(s) + ".";
  auto xq = read_bin(b + "xq.bin");
  auto xk = read_bin(b + "xk.bin");
  auto fc = read_bin(b + "fc.bin");
  auto fs = read_bin(b + "fs.bin");
  auto gq = read_bin(b + "gq.bin");
  auto gk = read_bin(b + "gk.bin");
  ASSERT_FALSE(xq.empty() || gq.empty()) << "missing rope_dyn.S" << s;
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

// Dynamic select_copy(0,-1): input [2,1,S,kHidden] -> output [1,S,kHidden]. The
// negative index resolves against the (static) leading dim live; the dynamic S
// flows to the output, so the resize hook recomputes its dispatch each S.
constexpr int kSelLead = 2;
void check_select(int s) {
  Module m(g_dir + "/dyn_select.pte");
  ASSERT_EQ(m.load_forward(), Error::Ok) << "load dyn_select.pte";
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

// J: dynamic SDPA (GQA prefill) at several seq-len S. The whole case skips
// while op coverage is pending (the dynamic-S build throws err 48 until
// registered).
TEST(DynamicShape, Sdpa) {
  {
    Module probe(g_dir + "/sdpa_dyn.pte");
    if (probe.load_forward() == Error::DelegateInvalidCompatibility) {
      GTEST_SKIP() << "sdpa_dyn pending op coverage (err "
                   << (int)Error::DelegateInvalidCompatibility << ")";
    }
  }
  for (int s : {64, 16, 1}) {
    check_sdpa(s);
  }
}

// K: dynamic embedding (int64 token ids) at several token counts.
TEST(DynamicShape, Embedding) {
  for (int n : {16, 8, 1}) {
    check_embedding(n);
  }
}

// L: dynamic RoPE (two outputs) at several seq-len S.
TEST(DynamicShape, Rope) {
  for (int s : {16, 8, 1}) {
    check_rope(s);
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

// N: dynamic select_copy(0,-1) at several S.
TEST(DynamicShape, Select) {
  for (int s : {128, 32, 1}) {
    check_select(s);
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
