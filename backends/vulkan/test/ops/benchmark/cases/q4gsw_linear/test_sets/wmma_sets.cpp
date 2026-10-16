// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Correctness cases for the cooperative-matrix family kernels of linear_q4gsw
// and linear_dq8ca_q4gsw (impl/gemm/linear/qw/CoopmatFamily.cpp). The shapes
// are aligned to every family tile (M a multiple of 256, N of 128, K of 64,
// group size 128) and include the shapes that select each row: the Xe2 rows
// (any aligned shape; K * N >= 24e6 for the blocked 8da4w order) and the
// NVIDIA rows (the validated Llama projections, K > 8192 for fp32
// accumulation). On a device without a family the same cases exercise the
// existing kernels.

#include <executorch/backends/vulkan/test/ops/benchmark/cases/q4gsw_linear/q4gsw_linear.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {
namespace q4gsw_linear {

namespace {

std::vector<float> to_float(const ValueSpec& s) {
  if (s.dtype == vkapi::kFloat) {
    return s.get_float_data();
  }
  const auto& h = s.get_half_data();
  std::vector<float> o(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    o[i] = half_to_float(h[i]);
  }
  return o;
}

// Zero-mean data with exactly representable activations: multiples of 1/16
// in [-1, 1) (not periodic in K) and the random int4 weights and scales of
// make_case. Partial sums stay small, so the fp16-accumulating kernels (family
// and existing) stay within the tolerance; outputs are of order 1 to 5, so a
// structural error (a missing or misplaced tile, a wrong scale group) does
// not.
// For dq8ca the per-row activation scale is 1/16 so the dynamic int8
// quantization is exact; the zero points vary per row to exercise the
// zero-point correction (quantized values stay within int8).
void set_family_data(TestCase& t, const CoopmatLinearConfig& cfg) {
  auto& hin = t.inputs()[0].get_half_data();
  for (size_t i = 0; i < hin.size(); ++i) {
    hin[i] = float_to_half(float(int((i * 7 + i / 61) % 32) - 16) / 16.0f);
  }
  if (is_dq8ca(cfg.op_name)) {
    auto& hs = t.inputs()[1].get_half_data();
    std::fill(hs.begin(), hs.end(), float_to_half(0.0625f));
    auto& zp = t.inputs()[2].get_int8_data();
    for (size_t i = 0; i < zp.size(); ++i) {
      zp[i] = int8_t(int(i % 17) - 8);
    }
  }
  // fp16 accumulation error grows with K: at K = 14336 it reaches about 0.2
  // for the fp16 family kernels and 0.7 for the existing fp16 q4gsw kernel
  // (llvmpipe), against outputs of order 10.
  t.set_abs_tolerance(cfg.K > 8192 ? 1.0f : 0.25f);
  t.set_rel_tolerance(0.02f);
}

// Reference for both ops with 4-bit weights and no bias:
//   q4gsw: out[m][n] = SUM_k a[m][k] * w[n][k] * s[k / group][n]
//   dq8ca: a is first quantized per row (scale, zero point) and dequantized.
// Weights are dequantized once and the output columns are split across
// threads, so the larger shapes stay fast.
void family_reference(TestCase& tc) {
  const bool dq = tc.operator_name().find("dq8ca") != std::string::npos;
  const ValueSpec& in = tc.inputs()[0];
  ValueSpec& out = tc.outputs()[0];
  const auto in_sizes = in.get_tensor_sizes();
  const int64_t M = in_sizes[0];
  const int64_t K = in_sizes[1];
  const int64_t N = out.get_tensor_sizes()[1];
  const ValueSpec& w = tc.inputs()[dq ? 3 : 1];
  const std::vector<float> scales = to_float(tc.inputs()[dq ? 5 : 2]);
  const int64_t group = tc.inputs()[dq ? 6 : 3].get_int_value();
  const std::vector<uint8_t>& w4 = w.get_uint8_data(); // [N, K/2] nibbles

  std::vector<float> a = to_float(in);
  std::vector<float> row_scale(M, 1.0f);
  if (dq) {
    const std::vector<float> in_scale = to_float(tc.inputs()[1]);
    const std::vector<int8_t>& in_zp = tc.inputs()[2].get_int8_data();
    for (int64_t m = 0; m < M; ++m) {
      const float s = in_scale[m];
      const int zp = int(in_zp[m]);
      for (int64_t k = 0; k < K; ++k) {
        float q = std::round(a[m * K + k] / s) + float(zp);
        q = std::min(std::max(q, -128.0f), 127.0f);
        a[m * K + k] = q - float(zp);
      }
      row_scale[m] = s;
    }
  }

  auto& ref = out.get_ref_float_data();
  ref.assign(M * N, 0.0f);
  const int64_t num_threads = std::max<int64_t>(
      1, std::min<int64_t>(N, std::thread::hardware_concurrency()));
  auto work = [&](const int64_t t) {
    std::vector<float> wn(K);
    for (int64_t n = t; n < N; n += num_threads) {
      for (int64_t k = 0; k < K; ++k) {
        const uint8_t byte = w4[n * (K / 2) + k / 2];
        const int nib = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
        wn[k] = float(nib - 8) * scales[(k / group) * N + n];
      }
      for (int64_t m = 0; m < M; ++m) {
        const float* am = a.data() + m * K;
        float acc = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          acc += am[k] * wn[k];
        }
        ref[m * N + n] = acc * row_scale[m];
      }
    }
  };
  std::vector<std::thread> threads;
  for (int64_t t = 1; t < num_threads; ++t) {
    threads.emplace_back(work, t);
  }
  work(0);
  for (auto& th : threads) {
    th.join();
  }
}

} // namespace

REGISTER_TEST_CASE_SET("q4gsw_linear", "wmma_correctness") {
  // {M, K, N, group_size}
  const std::vector<CoopmatLinearConfig> shapes = {
      {256, 256, 384, 128, ""}, // small, multi-workgroup
      {2048, 2048, 512, 128, ""}, // long M
      {256, 3072, 1024, 128, ""}, // M = 256 with N % 2048 != 0
      {256, 2048, 8192, 128, ""}, // M = 256 with N % 2048 == 0
      {256, 4096, 8192, 128, ""}, // K * N >= 24e6
      {256, 14336, 4096, 128, ""}, // K > 8192
  };
  std::vector<TestCase> cases;
  for (const std::string op : {"linear_q4gsw", "linear_dq8ca_q4gsw"}) {
    for (const auto& shape : shapes) {
      const CoopmatLinearConfig cfg{
          shape.M, shape.K, shape.N, shape.group_size, op};
      for (const auto st : {utils::kTexture3D, utils::kBuffer}) {
        TestCase t = make_case(cfg, st);
        set_family_data(t, cfg);
        cases.push_back(std::move(t));
      }
    }
  }
  return {
      std::move(cases),
      family_reference,
      flop_calc,
      /*warmup_runs=*/1,
      /*benchmark_runs=*/3};
}

} // namespace q4gsw_linear
} // namespace prototyping
} // namespace vulkan
} // namespace executorch
