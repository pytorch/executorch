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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace executorch::backends::webgpu;
using namespace executorch::extension;
using namespace executorch::runtime;

namespace {

// Environment-derived config; captured in main() before RUN_ALL_TESTS().
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::string g_update_cache_model_path;
std::string g_qlinear_dir;
std::string g_prepack_model_path, g_prepack_golden_path;
std::string g_prepack2_model_path, g_prepack2_golden_path;
std::string g_prepack_tied_model_path, g_prepack_tied_golden_path;
std::string g_sdpa_dir;
std::string g_symint_blob;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

#ifdef WGPU_BACKEND_ENABLE_PROFILING
// Capacity-overrun must throw; runs without a device or TimestampQuery.
void test_query_pool_overrun_throws() {
  WebGPUQueryPool qp;
  EXPECT_THROW(qp.reset(1), std::exception)
      << "reset beyond capacity did not throw";
}

// WebGPUQueryPool roundtrip: time a probe pass; assert non-zero GPU duration.
void test_query_pool_roundtrip(const WebGPUContext& ctx) {
  if (!ctx.timestamp_supported) {
    GTEST_SKIP() << "adapter lacks TimestampQuery feature";
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

  ASSERT_EQ(qp.results().size(), 1u)
      << "expected 1 duration, got " << qp.results().size();
  const uint64_t dur = qp.results()[0].execution_duration_ns;
  printf("  probe duration: %llu ns\n", (unsigned long long)dur);
  EXPECT_NE(dur, 0u) << "probe duration is zero (expected monotonic non-zero)";
}
#endif // WGPU_BACKEND_ENABLE_PROFILING

void test_update_cache(const std::string& model_path) {
  // update_cache: value [1,2,2,4] scattered into cache [1,8,2,4] at
  // input_pos=0.
  Module module(model_path);
  auto err = module.load_forward();
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
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
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";

  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ((int)out_tensor.numel(), cnumel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << cnumel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f;
  for (int i = 0; i < cnumel; i++) {
    max_abs_err = std::max(max_abs_err, std::abs(out_data[i] - ref[i]));
  }
  printf("Max abs error: %e (checked %d elements)\n", max_abs_err, cnumel);
  EXPECT_LE(max_abs_err, 1e-3f) << "max error exceeds tolerance 1e-3";
}

std::vector<float> load_golden(const std::string& path, size_t numel) {
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
bool sdpa_within_tol(
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

// linear_q4gsw sweep config; mirrors CONFIGS in test_quantized_linear.py.
struct Q4gswConfig {
  const char* name;
  int m; // rows (tokens)
  int k; // in_features (reduction dim)
  int n; // out_features
  float tol_abs; // per-element abs gate
  float tol_rel; // per-element rel gate
  bool required; // dir set + .pte absent => FAIL (not skip)
  bool heavy; // huge/slow: export-gated; runs only if WEBGPU_TEST_HEAVY
};

// Llama-3.2-1B linear shapes (q/o/k/v/gate/up/down + lm_head) + 4k/8k prefill.
// tol scales with K (fp32 accum depth), not M; down_proj (K=8192) is looser.
const Q4gswConfig kQ4gswConfigs[] = {
    // name         M     K     N      tol_abs tol_rel req    heavy
    {"q_proj", 1, 2048, 2048, 1e-4f, 1e-3f, true, false},
    {"kv_proj", 1, 2048, 512, 1e-4f, 1e-3f, true, false},
    {"gate_proj", 1, 2048, 8192, 1e-4f, 1e-3f, true, false},
    {"down_proj", 1, 8192, 2048, 1e-3f, 1e-2f, true, false}, // big-K accum
    {"lm_head", 1, 2048, 128256, 1e-4f, 1e-3f, false, true},
    {"q_proj_4k", 4096, 2048, 2048, 1e-4f, 1e-3f, true, false},
    {"kv_proj_4k", 4096, 2048, 512, 1e-4f, 1e-3f, true, false},
    {"q_proj_8k", 8192, 2048, 2048, 1e-4f, 1e-3f, false, true},
    {"kv_proj_8k", 8192, 2048, 512, 1e-4f, 1e-3f, false, true},
    // The M==1 configs above (q/kv/gate/down_proj) exercise the bicol 2-col
    // decode GEMV (handler routes M==1 -> bicol; each reads its own per-column
    // scale over 64-256 K-groups). q4gsw requires N % 8 == 0, so odd-N is not
    // exportable; bicol's has1 odd-N guard is defensive (mirrors coop4
    // general-N robustness).
    {"gate_proj_pf", 128, 2048, 8192, 1e-4f, 1e-3f, true, false}, // shmem via N
    {"down_proj_pf", 128, 8192, 2048, 1e-3f, 1e-2f, true, false}, // shmem via K
    {"shmem_edge", 130, 4096, 2056, 1e-4f, 1e-3f, true, false}, // partial tiles
};

// /16 ramp over the flat index; mirrors test_quantized_linear.py _ramp_input.
float q4gsw_ramp(int i) {
  return static_cast<float>((i % 17) - 8) / 16.0f;
}

// Per-element abs-OR-rel tolerance helper.
bool quant_within_tol(
    const float* out,
    const float* golden,
    int n,
    float atol,
    float rtol,
    float* ma,
    float* mr) {
  float max_abs = 0.0f, max_rel = 0.0f;
  bool ok = true;
  for (int i = 0; i < n; i++) {
    const float ae = std::abs(out[i] - golden[i]);
    const float re = ae / std::max(std::abs(golden[i]), 1e-6f);
    max_abs = std::max(max_abs, ae);
    max_rel = std::max(max_rel, re);
    if (ae > atol && re > rtol) {
      ok = false;
    }
  }
  *ma = max_abs;
  *mr = max_rel;
  return ok;
}

std::vector<int32_t> load_indices(const std::string& path, size_t numel) {
  // Load raw little-endian int32 indices written by the export .py.
  std::vector<int32_t> g(numel);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return {};
  }
  size_t n = std::fread(g.data(), sizeof(int32_t), numel, f);
  std::fclose(f);
  if (n != numel) {
    return {};
  }
  return g;
}

void test_embedding_q4gsw(
    const std::string& model_path,
    const std::string& indices_path,
    const std::string& golden_path,
    int num_indices,
    int embed,
    const char* label) {
  // q4gsw embedding-gather vs torch golden; shapes per test_embedding_q4gsw.py.
  const int out_numel = num_indices * embed;
  printf(
      "\n--- Test: embedding_q4gsw (%s: indices=%d, embed=%d) ---\n",
      label,
      num_indices,
      embed);

  Module module(model_path);
  auto err = module.load_forward();
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
  printf("Model loaded: %s\n", model_path.c_str());

  std::vector<int32_t> idx32 = load_indices(indices_path, num_indices);
  std::vector<float> golden = load_golden(golden_path, out_numel);
  ASSERT_FALSE(idx32.empty() || golden.empty())
      << "could not load indices " << indices_path << " / golden "
      << golden_path;

  // int64 at the program boundary; copy_inputs narrows to the int32 buffer.
  std::vector<int64_t> idx64(idx32.begin(), idx32.end());
  auto idx = make_tensor_ptr({num_indices}, std::move(idx64));

  auto result = module.forward({EValue(idx)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ((int)out_tensor.numel(), out_numel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << out_numel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  const bool pass = quant_within_tol(
      out_data,
      golden.data(),
      out_numel,
      1e-3f,
      1e-3f,
      &max_abs_err,
      &max_rel_err);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      out_numel);
  EXPECT_TRUE(pass) << "embedding_q4gsw exceeds tolerance 1e-3 (abs AND rel)";
}

void test_rope(
    const std::string& model_path,
    const std::string& xq_golden_path,
    const std::string& xk_golden_path,
    int S,
    int NH,
    int NKV,
    int HD,
    const char* label) {
  // Llama interleaved RoPE vs torch goldens; shapes/ramps per test_rope.py.
  const int xq_numel = S * NH * HD;
  const int xk_numel = S * NKV * HD;
  const int freqs_numel = S * (HD / 2);
  printf(
      "\n--- Test: apply_rotary_emb (%s: S=%d,NH=%d,NKV=%d,HD=%d) ---\n",
      label,
      S,
      NH,
      NKV,
      HD);

  Module module(model_path);
  auto err = module.load_forward();
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
  printf("Model loaded: %s\n", model_path.c_str());

  // ((i % mod) - off) / 16: exact in fp32, matches test_rope.py::_ramp.
  auto ramp = [](int i, int mod, int off) {
    return static_cast<float>((i % mod) - off) / 16.0f;
  };
  std::vector<float> xq(xq_numel), xk(xk_numel), fc(freqs_numel),
      fs(freqs_numel);
  for (int i = 0; i < xq_numel; i++) {
    xq[i] = ramp(i, 17, 8);
  }
  for (int i = 0; i < xk_numel; i++) {
    xk[i] = ramp(i, 13, 6);
  }
  for (int i = 0; i < freqs_numel; i++) {
    fc[i] = ramp(i, 11, 5);
    fs[i] = ramp(i, 7, 3);
  }

  auto xqt = make_tensor_ptr({1, S, NH, HD}, std::vector<float>(xq));
  auto xkt = make_tensor_ptr({1, S, NKV, HD}, std::vector<float>(xk));
  auto fct = make_tensor_ptr({S, HD / 2}, std::vector<float>(fc));
  auto fst = make_tensor_ptr({S, HD / 2}, std::vector<float>(fs));

  auto result =
      module.forward({EValue(xqt), EValue(xkt), EValue(fct), EValue(fst)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();

  // Outputs in graph order [0]=xq_out, [1]=xk_out (positional; the numel check
  // below guards a swap, since NH != NKV under GQA).
  ASSERT_TRUE(
      outputs.size() >= 2 && outputs[0].isTensor() && outputs[1].isTensor())
      << "expected 2 tensor outputs, got " << outputs.size();
  const auto& xq_t = outputs[0].toTensor();
  const auto& xk_t = outputs[1].toTensor();
  ASSERT_TRUE(xq_t.numel() == xq_numel && xk_t.numel() == xk_numel)
      << "output shapes [" << (size_t)xq_t.numel() << ","
      << (size_t)xk_t.numel() << "] != expected [" << xq_numel << ","
      << xk_numel << "]";
  const float* xq_out = xq_t.const_data_ptr<float>();
  const float* xk_out = xk_t.const_data_ptr<float>();

  std::vector<float> gq = load_golden(xq_golden_path, xq_numel);
  std::vector<float> gk = load_golden(xk_golden_path, xk_numel);
  ASSERT_FALSE(gq.empty() || gk.empty())
      << "could not load goldens " << xq_golden_path << " / " << xk_golden_path;

  // Per-element abs-OR-rel on xq and xk (shared helper).
  float maq = 0.0f, mrq = 0.0f, mak = 0.0f, mrk = 0.0f;
  const bool pass_q =
      quant_within_tol(xq_out, gq.data(), xq_numel, 1e-3f, 1e-3f, &maq, &mrq);
  const bool pass_k =
      quant_within_tol(xk_out, gk.data(), xk_numel, 1e-3f, 1e-3f, &mak, &mrk);
  const float max_abs_err = std::max(maq, mak);
  const float max_rel_err = std::max(mrq, mrk);

  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      xq_numel + xk_numel);
  EXPECT_TRUE(pass_q && pass_k)
      << "apply_rotary_emb exceeds tolerance 1e-3 (abs AND rel)";
}

void test_prepack(
    const std::string& model_path,
    const std::string& golden_path,
    const std::string& label = "x + const w") {
  // et_vk.prepack copy vs golden; unrun copy leaves zeros. See test_prepack.py.
  constexpr int n = 4;
  constexpr int numel = n * n;
  printf("\n--- Test: prepack (%s, %dx%d) ---\n", label.c_str(), n, n);

  Module module(model_path);
  auto err = module.load_forward();
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
  printf("Model loaded: %s\n", model_path.c_str());

  std::vector<float> golden = load_golden(golden_path, numel);
  ASSERT_FALSE(golden.empty()) << "could not load golden " << golden_path;

  // ((i % 13) - 6) / 16: exact in fp32, matches test_prepack.py::_inputs.
  std::vector<float> x_data(numel);
  for (int i = 0; i < numel; i++) {
    x_data[i] = static_cast<float>((i % 13) - 6) / 16.0f;
  }
  auto x = make_tensor_ptr({n, n}, std::vector<float>(x_data));

  auto result = module.forward({EValue(x)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ((int)out_tensor.numel(), numel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << numel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  // Per-element abs-OR-rel (quant_within_tol): a global rel gate spuriously
  // fails near-zero outputs where rel error explodes.
  const bool within = quant_within_tol(
      out_data, golden.data(), numel, 1e-3f, 1e-3f, &max_abs_err, &max_rel_err);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      numel);
  EXPECT_TRUE(within) << "prepack exceeds tolerance 1e-3";
}

// Reconstruct _ramp_input bit-for-bit, run the op, compare to the fp64 golden.
void test_q4gsw_config(
    const Q4gswConfig& cfg,
    const std::string& pte,
    const std::string& golden_path) {
  printf(
      "\n--- Test: linear_q4gsw (%s: M=%d,K=%d,N=%d) ---\n",
      cfg.name,
      cfg.m,
      cfg.k,
      cfg.n);

  Module module(pte);
  ASSERT_EQ(module.load_forward(), Error::Ok) << "could not load " << pte;

  const int in_numel = cfg.m * cfg.k;
  const int out_numel = cfg.m * cfg.n;
  std::vector<float> input(in_numel);
  for (int i = 0; i < in_numel; i++) {
    input[i] = q4gsw_ramp(i);
  }

  auto x = make_tensor_ptr({cfg.m, cfg.k}, std::vector<float>(input));
  auto result = module.forward({EValue(x)});
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";
  const auto& outputs = result.get();
  ASSERT_TRUE(!outputs.empty() && outputs[0].isTensor()) << "no tensor output";
  const auto& out_tensor = outputs[0].toTensor();
  ASSERT_EQ((int)out_tensor.numel(), out_numel)
      << "output numel " << (size_t)out_tensor.numel() << " != expected "
      << out_numel;
  const float* out_data = out_tensor.const_data_ptr<float>();

  std::vector<float> golden = load_golden(golden_path, out_numel);
  ASSERT_FALSE(golden.empty()) << "could not load golden " << golden_path;

  float ma = 0.0f, mr = 0.0f;
  const bool pass = quant_within_tol(
      out_data, golden.data(), out_numel, cfg.tol_abs, cfg.tol_rel, &ma, &mr);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      ma,
      mr,
      out_numel);
  EXPECT_TRUE(pass) << "linear_q4gsw " << cfg.name << " exceeds tolerance (abs "
                    << cfg.tol_abs << " OR rel " << cfg.tol_rel << ")";
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
  bool required = false; // CI (SDPA dir set): absent .pte = FAIL, not skip
  bool expect_reject = false; // load MUST fail (e.g. D%4 guard), no golden
};

const SdpaConfig kSdpaConfigs[] = {
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
    {"llama1b_decode_4k", 32, 8, 64, 1, 4096, 4095, 16.0f, /*required=*/true},
    {"llama1b_decode_8k", 32, 8, 64, 1, 8192, 8191, 16.0f, /*required=*/true},
    // Llama 3.2 1B shape: realistic prefill (S=128 at pos 0) + decode (S=1 at
    // pos 127).
    {"llama1b_prefill", 32, 8, 64, 128, 512, 0, 16.0f},
    {"llama1b_decode", 32, 8, 64, 1, 512, 127, 16.0f},
    // D=6 is not a multiple of 4: the head_dim%4 guard must reject it at load.
    {"reject_d6",
     4,
     4,
     6,
     4,
     16,
     0,
     16.0f,
     /*required=*/false,
     /*expect_reject=*/true},
    // 2D-dispatch cap (>65535 wg): S=512 folds QK; S=2048 folds QK+softmax+AV
    // (cap+1).
    {"llama1b_prefill_512", 32, 8, 64, 512, 512, 0, 16.0f, /*required=*/true},
    {"llama1b_prefill_2048",
     32,
     8,
     64,
     2048,
     2048,
     0,
     16.0f,
     /*required=*/true},
};

// Ramp denominator; mirror of test_sdpa.py::_RAMP_DENOM (keep in sync).
constexpr float kSdpaRampDenom = 16.0f;

// /denom ramp: ((i % mod) - off) / denom, exact in fp32 (power-of-two denom).
// Mirrors test_sdpa.py::_ramp.
float sdpa_ramp(int i, int mod, int off, float denom = kSdpaRampDenom) {
  return static_cast<float>((i % mod) - off) / denom;
}

// Step-indexed ramp; mirrors test_sdpa.py::_ramp_t bit-for-bit. denom defaults
// to kSdpaRampDenom and must match the Python denom for bit-identity.
float sdpa_ramp_t(
    int i,
    int mod,
    int off,
    int t,
    float denom = kSdpaRampDenom) {
  return static_cast<float>(((i + 31 * t) % mod) - off) / denom;
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

const SdpaSequence kSdpaSequences[] = {
    {"small", 8, 4, 4, 16, {3, 1, 1, 5, 1, 1, 2}},
    {"small_d", 6, 2, 8, 16, {3, 1, 1, 5, 1, 1}},
    {"llama3", 24, 8, 128, 256, {111, 1, 1, 1, 57, 1, 1}},
};

void test_sdpa_config(
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
  if (cfg.expect_reject) {
    // D not a multiple of 4 must be rejected at load by the head_dim guard.
    ASSERT_NE(err, Error::Ok)
        << cfg.name << " loaded OK; head_dim%4 guard did not fire";
    printf("PASS: %s rejected at load (error %d)\n", cfg.name, (int)err);
    return;
  }
  ASSERT_EQ(err, Error::Ok)
      << "could not load forward method (error " << (int)err << ")";
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
  ASSERT_TRUE(result.ok()) << "forward failed (error " << (int)result.error()
                           << ")";

  const auto& outputs = result.get();
  // Select the attention output [1,S,Hq,D] by shape; the op returns
  // [k_cache, v_cache, attn_output] and a cache [1,Cmax,Hkv,D] can share numel.
  int attn_idx = -1;
  int attn_matches = 0;
  for (size_t i = 0; i < outputs.size(); i++) {
    if (!outputs[i].isTensor()) {
      continue;
    }
    const auto& t = outputs[i].toTensor();
    if (t.dim() == 4 && static_cast<int>(t.size(1)) == cfg.s &&
        static_cast<int>(t.size(2)) == cfg.hq &&
        static_cast<int>(t.size(3)) == cfg.d) {
      attn_idx = static_cast<int>(i);
      attn_matches++;
    }
  }
  ASSERT_GE(attn_idx, 0) << "no attention output [1," << cfg.s << "," << cfg.hq
                         << "," << cfg.d << "] among " << outputs.size()
                         << " outputs";
  ASSERT_LE(attn_matches, 1) << "ambiguous attention output: " << attn_matches
                             << " tensors match shape [1," << cfg.s << ","
                             << cfg.hq << "," << cfg.d << "]";
  const auto& out_tensor = outputs[attn_idx].toTensor();
  const float* out_data = out_tensor.const_data_ptr<float>();

  std::vector<float> golden = load_golden(golden_path, on);
  ASSERT_FALSE(golden.empty()) << "could not load golden " << golden_path;

  float max_abs_err = 0.0f, max_rel_err = 0.0f;
  const bool pass =
      sdpa_within_tol(out_data, golden.data(), on, &max_abs_err, &max_rel_err);
  printf(
      "Max abs error: %e   Max rel error: %e (checked %d elements)\n",
      max_abs_err,
      max_rel_err,
      on);
  EXPECT_TRUE(pass) << cfg.name
                    << " exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)";
}

// Replay one sequence: thread the op's returned (mutated) KV cache across
// steps, comparing each step's attention output to its accumulated-context
// golden.
void test_sdpa_replay(const SdpaSequence& seq, const std::string& dir) {
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

  for (size_t t = 0; t < seq.seq_lens.size(); t++) {
    const int s = seq.seq_lens[t];
    const std::string base = dir + "sdpa_" + seq.name + "_step" +
        std::to_string(t) + "_S" + std::to_string(s) + "_pos" +
        std::to_string(input_pos);
    Module module(base + ".pte");
    ASSERT_EQ(module.load_forward(), Error::Ok)
        << "could not load " << base << ".pte";

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
    ASSERT_TRUE(result.ok())
        << "forward " << base << ".pte (error " << (int)result.error() << ")";
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
    ASSERT_TRUE(attn_idx >= 0 && cache_idxs.size() == 2)
        << seq.name << " step" << t << ": expected 1 attn + 2 caches";

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
        FAIL() << seq.name << " step0 cannot identify k/v cache by content";
      }
      printf("  k/v cache outputs: k_idx=%d v_idx=%d\n", k_idx, v_idx);
    }

    std::vector<float> golden = load_golden(base + ".golden.bin", qn);
    ASSERT_FALSE(golden.empty()) << "could not load " << base << ".golden.bin";
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
    EXPECT_TRUE(step_ok)
        << seq.name << " step" << t
        << " exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)";

    // Thread the device-written caches into the next step (K->K, V->V).
    const float* kd = outs[k_idx].toTensor().const_data_ptr<float>();
    const float* vd = outs[v_idx].toTensor().const_data_ptr<float>();
    kc.assign(kd, kd + cn);
    vc.assign(vd, vd + cn);
    input_pos += s;
  }
}

// Dynamic input_pos decode: ONE .pte (S=1, runtime SymInt input_pos) reused
// across decode steps. Each forward() supplies input_pos as a [1] int64 tensor;
// the backend reads it (update_symints_from_inputs) and recomputes dispatch
// state (propagate_resize) before replaying. The cache is threaded host-side
// (the Module re-copies inputs each call), so correctness hinges on the
// per-step input_pos actually being read + applied. negative=true pins
// input_pos at 0 every step (stale context_len) and asserts the run DIVERGES,
// proving the runtime input_pos + resize hook are load-bearing (no false-pass).
void test_sdpa_dynamic_decode(
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
  ASSERT_EQ(module.load_forward(), Error::Ok) << "could not load " << pte;

  const int cn = seq.cmax * seq.hkv * seq.d;
  std::vector<float> kc(cn, 0.0f), vc(cn, 0.0f);
  int k_idx = -1,
      v_idx = -1; // pinned at step 0 by content (caches share numel)
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
    ASSERT_TRUE(result.ok())
        << "forward step" << t << " (error " << (int)result.error() << ")";
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
    ASSERT_TRUE(attn_idx >= 0 && cache_idxs.size() == 2)
        << seq.name << " step" << t << ": expected 1 attn + 2 caches";
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
        FAIL() << seq.name << " step0 cannot identify k/v cache";
      }
    }

    const std::string gpath = dir + "sdpa_dyn_" + seq.name + "_step" +
        std::to_string(t) + ".golden.bin";
    std::vector<float> golden = load_golden(gpath, qn);
    ASSERT_FALSE(golden.empty()) << "could not load " << gpath;
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
    // The negative control must DIVERGE: a stale input_pos=0 every step cannot
    // match the accumulating golden -- otherwise the oracle has no teeth.
    EXPECT_TRUE(any_mismatch)
        << seq.name
        << " negative control matched the golden (oracle has no teeth)";
    if (any_mismatch) {
      printf(
          "PASS: sdpa dynamic decode NEGATIVE (%s): stale input_pos diverges "
          "as expected\n",
          seq.name);
    }
    return;
  }
  EXPECT_FALSE(any_mismatch)
      << seq.name << " exceeds tolerance (per-element abs 1e-4 OR rel 1e-3)";
  if (!any_mismatch) {
    printf("PASS: sdpa dynamic decode (%s)\n", seq.name);
  }
}

// In-graph mutable KV cache: ONE .pte whose k_cache/v_cache are mutable buffers
// (NOT forward inputs); the decode loop feeds only the new token (q/k/v, S=1) +
// runtime input_pos, and the cache accumulates in-graph across forward() calls
// (no host threading). fresh_per_step is the static control: reloading the
// Module each step re-seeds the cache to zeros, so it MUST diverge from the
// accumulating golden at step>=1. Persistent-matches + fresh-diverges = proof
// the pass comes from real accumulation, not a static artifact.
void test_sdpa_incache_decode(
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
    ASSERT_EQ(persistent->load_forward(), Error::Ok)
        << "could not load " << pte;
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
      ASSERT_EQ(fresh->load_forward(), Error::Ok) << "could not load " << pte;
      mod = fresh.get();
    }

    // NOTE: only q/k/v + input_pos -- NO cache args (caches are mutable
    // buffers).
    auto result =
        mod->forward({EValue(qt), EValue(kt), EValue(vt), EValue(ipt)});
    ASSERT_TRUE(result.ok())
        << "forward step" << t << " (error " << (int)result.error() << ")";
    const auto& outs = result.get();
    int attn_idx = -1;
    for (size_t i = 0; i < outs.size(); i++) {
      if (outs[i].isTensor() &&
          static_cast<int>(outs[i].toTensor().numel()) == qn) {
        attn_idx = static_cast<int>(i);
        break;
      }
    }
    ASSERT_GE(attn_idx, 0) << seq.name << " step" << t
                           << ": no attn output (numel " << qn << ")";

    const std::string gpath = dir + "sdpa_incache_" + seq.name + "_step" +
        std::to_string(t) + ".golden.bin";
    std::vector<float> golden = load_golden(gpath, qn);
    ASSERT_FALSE(golden.empty()) << "could not load " << gpath;
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
    EXPECT_TRUE(any_mismatch)
        << seq.name
        << " static control matched the accumulating golden -- "
           "accumulation was not actually exercised (false-pass risk)";
    if (any_mismatch) {
      printf(
          "PASS: in-graph-cache STATIC CONTROL (%s) diverges as expected -- "
          "persistence is load-bearing; the positive pass is real accumulation\n",
          seq.name);
    }
    return;
  }
  EXPECT_FALSE(any_mismatch)
      << seq.name << " in-graph-cache decode exceeds tolerance";
  if (!any_mismatch) {
    printf(
        "PASS: sdpa in-graph-cache decode (%s) -- cache accumulated in-graph "
        "with NO host threading\n",
        seq.name);
  }
}

// S1 SymInt round-trip: build a graph directly from a dynamic-input_pos SDPA
// blob; confirm input_pos deserializes as a live SymInt and set/read
// round-trips.
void test_symint_roundtrip(const std::string& blob_path) {
  printf("\n--- Test: symint round-trip (%s) ---\n", blob_path.c_str());
  FILE* f = std::fopen(blob_path.c_str(), "rb");
  if (!f) {
    GTEST_SKIP() << blob_path << " not present";
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> blob(static_cast<size_t>(n));
  size_t rd = std::fread(blob.data(), 1, blob.size(), f);
  std::fclose(f);
  ASSERT_EQ(rd, blob.size()) << "short read of " << blob_path;

  auto header = WebGPUDelegateHeader::parse(blob.data());
  ASSERT_TRUE(header.ok()) << "delegate header parse";
  const uint8_t* base = blob.data();
  WebGPUGraph graph;
  try {
    graph.build(
        base + header->flatbuffer_offset, base + header->bytes_offset, nullptr);
  } catch (const std::exception& e) {
    FAIL() << "graph build: " << e.what();
  }

  int sid = -1;
  for (int i = 0; i < graph.num_values(); i++) {
    if (graph.get_value_type(i) == WebGPUGraph::ValueType::SymInt) {
      sid = i;
      break;
    }
  }
  ASSERT_GE(sid, 0)
      << "no SymInt value deserialized (input_pos should be a SymInt)";
  ASSERT_NE(graph.symint_buffer(sid), nullptr)
      << "SymInt " << sid << " has no live uniform buffer";
  ASSERT_EQ(graph.read_symint(sid), 0)
      << "SymInt " << sid << " placeholder != 0 (got " << graph.read_symint(sid)
      << ")";
  graph.set_symint(sid, 7);
  ASSERT_EQ(graph.read_symint(sid), 7)
      << "set/read round-trip (got " << graph.read_symint(sid) << ")";

  // Execute-read: feed a fake input_pos=5 via the recorded select_as_symint
  // source and confirm update_symints_from_inputs populates the SymInt.
  const auto& srcs = graph.symint_sources();
  ASSERT_FALSE(srcs.empty()) << "no select_as_symint source recorded";
  const auto& in_ids = graph.input_ids();
  std::vector<InputData> fake_inputs(in_ids.size());
  int64_t fake_pos = 5;
  for (size_t i = 0; i < in_ids.size(); i++) {
    if (in_ids[i] == srcs[0].input_tensor_id) {
      fake_inputs[i] = {&fake_pos, sizeof(int64_t), true};
    }
  }
  graph.update_symints_from_inputs(fake_inputs);
  ASSERT_EQ(graph.read_symint(srcs[0].symint_id), 5)
      << "execute-read (got " << graph.read_symint(srcs[0].symint_id)
      << ", want 5)";

  printf(
      "PASS: symint round-trip (SymInt %d: deserialize, live buffer, "
      "set 0->7, execute-read input_pos->5)\n",
      sid);
}

// Group 1: the resize-hook dirty-gating mechanism (no SDPA dependency).
// A hook keyed to a SymInt must run via propagate_resize() iff that SymInt
// changed since the last propagate_resize, and exactly once per change.
void test_resize_hook(const std::string& blob_path) {
  printf("\n--- Test: resize-hook dirty-gating (%s) ---\n", blob_path.c_str());
  FILE* f = std::fopen(blob_path.c_str(), "rb");
  if (!f) {
    GTEST_SKIP() << blob_path << " not present";
  }
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> blob(static_cast<size_t>(n));
  size_t rd = std::fread(blob.data(), 1, blob.size(), f);
  std::fclose(f);
  ASSERT_EQ(rd, blob.size()) << "short read of " << blob_path;
  auto header = WebGPUDelegateHeader::parse(blob.data());
  ASSERT_TRUE(header.ok()) << "delegate header parse";
  const uint8_t* base = blob.data();
  WebGPUGraph graph;
  try {
    graph.build(
        base + header->flatbuffer_offset, base + header->bytes_offset, nullptr);
  } catch (const std::exception& e) {
    FAIL() << "graph build: " << e.what();
  }

  int sid = -1;
  for (int i = 0; i < graph.num_values(); i++) {
    if (graph.get_value_type(i) == WebGPUGraph::ValueType::SymInt) {
      sid = i;
      break;
    }
  }
  ASSERT_GE(sid, 0) << "no SymInt value deserialized";

  int run_count = 0;
  int last_seen = -1;
  graph.add_resize_hook(sid, [&](WebGPUGraph& g) {
    run_count++;
    last_seen = g.read_symint(sid);
  });

  // 1: change 0->3 then propagate -> hook runs once, sees 3.
  graph.set_symint(sid, 3);
  graph.propagate_resize();
  ASSERT_TRUE(run_count == 1 && last_seen == 3)
      << "after set(3)+propagate run_count=" << run_count
      << " last_seen=" << last_seen << " (want 1,3)";
  // 2: propagate again with no change -> hook does NOT run.
  graph.propagate_resize();
  ASSERT_EQ(run_count, 1)
      << "propagate with clean dirty-set ran the hook (run_count=" << run_count
      << ")";
  // 3: set to the SAME value -> not dirty -> hook does NOT run.
  graph.set_symint(sid, 3);
  graph.propagate_resize();
  ASSERT_EQ(run_count, 1) << "set(same)+propagate ran the hook (run_count="
                          << run_count << ")";
  // 4: change 3->8 then propagate -> hook runs again, sees 8.
  graph.set_symint(sid, 8);
  graph.propagate_resize();
  ASSERT_TRUE(run_count == 2 && last_seen == 8)
      << "after set(8)+propagate run_count=" << run_count
      << " last_seen=" << last_seen << " (want 2,8)";

  printf(
      "PASS: resize-hook dirty-gating (SymInt %d: runs only on change, "
      "once per change; saw 3 then 8)\n",
      sid);
}

// q4gsw embedding_q4gsw on-GPU configs: small + llama1b (env-gated,
// run-if-present).
struct EmbConfig {
  const char* name;
  const char* model_env;
  const char* indices_env;
  const char* golden_env;
  int num_indices;
  int embed;
};
const EmbConfig kEmbConfigs[] = {
    {"small",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_MODEL",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_INDICES",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_GOLDEN",
     4,
     64},
    {"llama1b",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_MODEL",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_INDICES",
     "WEBGPU_TEST_EMBEDDING_Q4GSW_LLAMA1B_GOLDEN",
     4,
     2048},
};

// apply_rotary_emb on-GPU configs: multi + decode (env-gated, run-if-present).
struct RopeConfig {
  const char* name;
  const char* model_env;
  const char* xq_env;
  const char* xk_env;
  int S;
  int NH;
  int NKV;
  int HD;
};
const RopeConfig kRopeConfigs[] = {
    {"multi",
     "WEBGPU_TEST_ROPE_MODEL",
     "WEBGPU_TEST_ROPE_XQ_GOLDEN",
     "WEBGPU_TEST_ROPE_XK_GOLDEN",
     5,
     8,
     2,
     64},
    {"decode",
     "WEBGPU_TEST_ROPE_DECODE_MODEL",
     "WEBGPU_TEST_ROPE_DECODE_XQ_GOLDEN",
     "WEBGPU_TEST_ROPE_DECODE_XK_GOLDEN",
     1,
     32,
     8,
     64},
};

} // namespace

#ifdef WGPU_BACKEND_ENABLE_PROFILING
TEST(WebGPUNative, QueryPoolOverrunThrows) {
  test_query_pool_overrun_throws();
}

TEST(WebGPUNative, QueryPoolRoundtrip) {
  test_query_pool_roundtrip(*get_default_webgpu_context());
}
#endif // WGPU_BACKEND_ENABLE_PROFILING

TEST(WebGPUNative, UpdateCache) {
  if (g_update_cache_model_path.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_UPDATE_CACHE_MODEL not set";
  }
  test_update_cache(g_update_cache_model_path);
}

// Guard python<->C++ ramp bit-identity: q4gsw_ramp(0) = -0.5 exactly.
TEST(WebGPUNative, Q4gswRampBitIdentity) {
  EXPECT_LT(std::abs(q4gsw_ramp(0) - (-0.5f)), 1e-12f)
      << "q4gsw_ramp bit-identity check";
}

// q4gsw sweep: self-discover q4gsw_<name>.pte; required=FAIL, heavy=gate.
TEST(WebGPUNative, QuantizedLinearSweep) {
  const std::string& dir = g_qlinear_dir;
  const bool heavy_run = std::getenv("WEBGPU_TEST_HEAVY") != nullptr;
  bool ran = false;
  for (const auto& cfg : kQ4gswConfigs) {
    const std::string pte = dir + "q4gsw_" + cfg.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      if (cfg.required && !dir.empty()) {
        ADD_FAILURE() << "required q4gsw config " << cfg.name
                      << " has no .pte in " << dir;
      }
      continue;
    }
    std::fclose(f);
    if (cfg.heavy && !heavy_run) {
      printf(
          "SKIP: heavy q4gsw config %s (set WEBGPU_TEST_HEAVY=1 on a real GPU)\n",
          cfg.name);
      continue;
    }
    const std::string golden = dir + "q4gsw_" + cfg.name + ".golden.bin";
    ran = true;
    test_q4gsw_config(cfg, pte, golden);
  }
  if (!dir.empty() && !ran) {
    ADD_FAILURE()
        << "WEBGPU_TEST_QUANTIZED_LINEAR_DIR set but no q4gsw config ran";
  }
}

TEST(WebGPUNative, EmbeddingQ4gsw) {
  bool any = false;
  for (const auto& c : kEmbConfigs) {
    const char* m = std::getenv(c.model_env);
    const char* ip = std::getenv(c.indices_env);
    const char* g = std::getenv(c.golden_env);
    if (m && ip && g && *m && *ip && *g) {
      any = true;
      test_embedding_q4gsw(m, ip, g, c.num_indices, c.embed, c.name);
    }
  }
  if (!any) {
    GTEST_SKIP() << "no embedding_q4gsw config env set";
  }
}

TEST(WebGPUNative, Rope) {
  bool any = false;
  for (const auto& c : kRopeConfigs) {
    const char* m = std::getenv(c.model_env);
    const char* xq = std::getenv(c.xq_env);
    const char* xk = std::getenv(c.xk_env);
    if (m && xq && xk && *m && *xq && *xk) {
      any = true;
      test_rope(m, xq, xk, c.S, c.NH, c.NKV, c.HD, c.name);
    }
  }
  if (!any) {
    GTEST_SKIP() << "no apply_rotary_emb config env set";
  }
}

TEST(WebGPUNative, Prepack) {
  if (g_prepack_model_path.empty() || g_prepack_golden_path.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_PREPACK_MODEL/GOLDEN not set";
  }
  test_prepack(g_prepack_model_path, g_prepack_golden_path);
}

TEST(WebGPUNative, Prepack2) {
  if (g_prepack2_model_path.empty() || g_prepack2_golden_path.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_PREPACK2_MODEL/GOLDEN not set";
  }
  test_prepack(g_prepack2_model_path, g_prepack2_golden_path, "x + w1 + w2");
}

TEST(WebGPUNative, PrepackTied) {
  if (g_prepack_tied_model_path.empty() || g_prepack_tied_golden_path.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_PREPACK_TIED_MODEL/GOLDEN not set";
  }
  test_prepack(
      g_prepack_tied_model_path,
      g_prepack_tied_golden_path,
      "x + w + w (tied weights, shared key)");
}

// SDPA sweep: configs self-discover sdpa_<name>.pte; required=FAIL else skip.
TEST(WebGPUNative, SdpaSweep) {
  const std::string& dir = g_sdpa_dir;
  bool ran = false;
  for (const auto& cfg : kSdpaConfigs) {
    const std::string pte = dir + "sdpa_" + cfg.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      // required config absent (dir set) = FAIL; otherwise skip silently.
      if (cfg.required && !dir.empty()) {
        ADD_FAILURE() << "required sdpa config " << cfg.name
                      << " has no .pte in " << dir;
      }
      continue; // not embedded in this binary
    }
    std::fclose(f);
    const std::string golden = dir + "sdpa_" + cfg.name + ".golden.bin";
    ran = true;
    test_sdpa_config(cfg, pte, golden);
  }
  if (!dir.empty() && !ran) {
    ADD_FAILURE() << "WEBGPU_TEST_SDPA_DIR set but no sdpa config found a .pte";
  }
}

// Guard python<->C++ ramp bit-identity (recorded: _ramp_t(0,17,8,2)=0.1875).
TEST(WebGPUNative, SdpaRampTBitIdentity) {
  EXPECT_LT(std::abs(sdpa_ramp_t(0, 17, 8, 2) - 0.1875f), 1e-12f)
      << "sdpa_ramp_t bit-identity check";
}

// Guard the adversarial denom path: sdpa_ramp(0,17,8,0.5)= -16.0 exactly.
TEST(WebGPUNative, SdpaRampDenomBitIdentity) {
  EXPECT_LT(std::abs(sdpa_ramp(0, 17, 8, 0.5f) - (-16.0f)), 1e-12f)
      << "sdpa_ramp denom bit-identity check";
}

// Replay sweep: run every sequence whose step0 .pte is present.
TEST(WebGPUNative, SdpaReplaySweep) {
  const std::string& dir = g_sdpa_dir;
  for (const auto& seq : kSdpaSequences) {
    const std::string step0 = dir + "sdpa_" + seq.name + "_step0_S" +
        std::to_string(seq.seq_lens[0]) + "_pos0.pte";
    FILE* f = std::fopen(step0.c_str(), "rb");
    if (!f) {
      continue; // sequence not embedded in this binary
    }
    std::fclose(f);
    test_sdpa_replay(seq, dir);
  }
}

// Dynamic decode sweep: positive + negative control per embedded param set.
TEST(WebGPUNative, SdpaDynamicDecodeSweep) {
  const std::string& dir = g_sdpa_dir;
  for (const auto& seq : kSdpaSequences) {
    const std::string pte = dir + "sdpa_dyn_" + seq.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      continue;
    }
    std::fclose(f);
    test_sdpa_dynamic_decode(seq, dir, /*negative=*/false);
    test_sdpa_dynamic_decode(seq, dir, /*negative=*/true);
  }
}

// In-graph-cache decode sweep: persistent + fresh (static control) per set.
TEST(WebGPUNative, SdpaIncacheDecodeSweep) {
  const std::string& dir = g_sdpa_dir;
  for (const auto& seq : kSdpaSequences) {
    const std::string pte = dir + "sdpa_incache_" + seq.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (!f) {
      continue;
    }
    std::fclose(f);
    test_sdpa_incache_decode(seq, dir, /*fresh_per_step=*/false);
    test_sdpa_incache_decode(seq, dir, /*fresh_per_step=*/true);
  }
}

// If an SDPA dir was given, the exports must have produced .ptes for every
// family; a self-skip there means a silent export failure, not a pass.
TEST(WebGPUNative, SdpaAllFamiliesRanWhenDirSet) {
  const std::string& dir = g_sdpa_dir;
  if (dir.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_SDPA_DIR not set";
  }
  auto has_glob = [&](const std::string& prefix, const std::string& suffix) {
    for (const auto& seq : kSdpaSequences) {
      const std::string p = dir + prefix + seq.name + suffix;
      FILE* f = std::fopen(p.c_str(), "rb");
      if (f) {
        std::fclose(f);
        return true;
      }
    }
    return false;
  };
  bool sdpa_ran = false;
  for (const auto& cfg : kSdpaConfigs) {
    const std::string pte = dir + "sdpa_" + cfg.name + ".pte";
    FILE* f = std::fopen(pte.c_str(), "rb");
    if (f) {
      std::fclose(f);
      sdpa_ran = true;
      break;
    }
  }
  const bool replay_ran = [&] {
    for (const auto& seq : kSdpaSequences) {
      const std::string step0 = dir + "sdpa_" + seq.name + "_step0_S" +
          std::to_string(seq.seq_lens[0]) + "_pos0.pte";
      FILE* f = std::fopen(step0.c_str(), "rb");
      if (f) {
        std::fclose(f);
        return true;
      }
    }
    return false;
  }();
  const bool dyn_ran = has_glob("sdpa_dyn_", ".pte");
  const bool incache_ran = has_glob("sdpa_incache_", ".pte");
  EXPECT_TRUE(sdpa_ran && replay_ran && dyn_ran && incache_ran)
      << "WEBGPU_TEST_SDPA_DIR set but an SDPA family found no .pte";
}

TEST(WebGPUNative, SymintRoundtrip) {
  if (g_symint_blob.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_SYMINT_BLOB not set";
  }
  test_symint_roundtrip(g_symint_blob);
}

TEST(WebGPUNative, ResizeHook) {
  if (g_symint_blob.empty()) {
    GTEST_SKIP() << "WEBGPU_TEST_SYMINT_BLOB not set";
  }
  test_resize_hook(g_symint_blob);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  if (const char* env = std::getenv("WEBGPU_TEST_UPDATE_CACHE_MODEL")) {
    g_update_cache_model_path = env;
  }

  // Quantized-linear sweep dir (mirrors WEBGPU_TEST_SDPA_DIR).
  if (const char* env = std::getenv("WEBGPU_TEST_QUANTIZED_LINEAR_DIR")) {
    g_qlinear_dir = env;
    if (!g_qlinear_dir.empty() && g_qlinear_dir.back() != '/') {
      g_qlinear_dir += '/';
    }
  }

  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK_MODEL")) {
    g_prepack_model_path = env;
  }
  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK_GOLDEN")) {
    g_prepack_golden_path = env;
  }

  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK2_MODEL")) {
    g_prepack2_model_path = env;
  }
  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK2_GOLDEN")) {
    g_prepack2_golden_path = env;
  }

  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK_TIED_MODEL")) {
    g_prepack_tied_model_path = env;
  }
  if (const char* env = std::getenv("WEBGPU_TEST_PREPACK_TIED_GOLDEN")) {
    g_prepack_tied_golden_path = env;
  }

  // SDPA sweep: configs self-discover their sdpa_<name>.pte/.golden.bin under
  // this directory (default "" = the embedded-file root / cwd). Set
  // WEBGPU_TEST_SDPA_DIR to point at the exported .pte directory (e.g. /tmp/).
  if (const char* env = std::getenv("WEBGPU_TEST_SDPA_DIR")) {
    g_sdpa_dir = env;
    if (!g_sdpa_dir.empty() && g_sdpa_dir.back() != '/') {
      g_sdpa_dir += '/';
    }
  }

  if (const char* env = std::getenv("WEBGPU_TEST_SYMINT_BLOB")) {
    g_symint_blob = env;
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

  const int rc = RUN_ALL_TESTS();
  set_default_webgpu_context(nullptr);
  destroy_webgpu_context(ctx);
  return rc;
}
