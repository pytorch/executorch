/* Copyright (c) Meta Platforms, Inc. and affiliates. */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <executorch/backends/cuda/runtime/shims/int5_plain_mm.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cuda_shims = executorch::backends::cuda;

namespace {

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(                                                           \
          stderr,                                                             \
          "CUDA error %s:%d: %s\n",                                          \
          __FILE__,                                                           \
          __LINE__,                                                           \
          cudaGetErrorString(err__));                                         \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

int32_t log2_pow2_host(int32_t v) {
  int32_t r = 0;
  while (v > 1) {
    v >>= 1;
    ++r;
  }
  return r;
}

uint16_t float_to_bf16(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

void fill_case(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t gs,
    uint32_t seed,
    std::vector<uint16_t>& A,
    std::vector<uint8_t>& ql,
    std::vector<uint8_t>& qh,
    std::vector<uint8_t>& scale,
    std::vector<uint16_t>& scale_step,
    std::vector<uint8_t>& zero,
    std::vector<uint16_t>& zero_step) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> adist(-2.0f, 2.0f);
  std::uniform_int_distribution<int> qdist(0, 31);
  std::uniform_int_distribution<int> codedist(1, 15);
  std::uniform_real_distribution<float> sdist(0.003f, 0.035f);
  std::uniform_real_distribution<float> zdist(0.25f, 1.25f);

  A.resize(M * K);
  ql.assign(N * (K / 2), 0);
  qh.assign(N * (K / 8), 0);
  scale.resize(N * (K / gs));
  scale_step.resize(N * (K / 256));
  zero.resize(N * (K / gs));
  zero_step.resize(N * (K / 256));

  for (auto& x : A) {
    x = float_to_bf16(adist(rng));
  }
  for (auto& x : scale) {
    x = static_cast<uint8_t>(codedist(rng));
  }
  for (auto& x : zero) {
    x = static_cast<uint8_t>(codedist(rng));
  }
  for (auto& x : scale_step) {
    x = __half_as_ushort(__float2half(sdist(rng)));
  }
  for (auto& x : zero_step) {
    x = __half_as_ushort(__float2half(zdist(rng)));
  }

  std::vector<uint8_t> u(K);
  for (int64_t n = 0; n < N; ++n) {
    uint8_t* qlrow = ql.data() + n * (K / 2);
    uint8_t* qhrow = qh.data() + n * (K / 8);
    for (int64_t k = 0; k < K; ++k) {
      u[k] = static_cast<uint8_t>(qdist(rng));
    }
    for (int64_t k = 0; k < K; k += 2) {
      qlrow[k / 2] = static_cast<uint8_t>((u[k] & 0xF) | ((u[k + 1] & 0xF) << 4));
    }
    for (int64_t k = 0; k < K; k += 32) {
      uint8_t* chunk = qhrow + k / 8;
      for (int w = 0; w < 4; ++w) {
        uint8_t hi_even = 0;
        uint8_t hi_odd = 0;
        for (int j = 0; j < 4; ++j) {
          hi_even |= static_cast<uint8_t>(((u[k + w * 8 + j * 2] >> 4) & 0x1) << j);
          hi_odd |= static_cast<uint8_t>(((u[k + w * 8 + j * 2 + 1] >> 4) & 0x1) << j);
        }
        chunk[w] = static_cast<uint8_t>(hi_even | (hi_odd << 4));
      }
    }
  }
}

template <typename Fn>
float time_ms(Fn fn, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / iterations;
}

void* device_alloc_copy(const void* src, size_t bytes) {
  void* dst = nullptr;
  CUDA_CHECK(cudaMalloc(&dst, bytes));
  CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
  return dst;
}

int64_t arg_value(int argc, char** argv, const char* name, int64_t fallback) {
  std::string key = std::string("--") + name + "=";
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind(key, 0) == 0) {
      return std::stoll(arg.substr(key.size()));
    }
  }
  return fallback;
}

} // namespace

int main(int argc, char** argv) {
  const int64_t M = arg_value(argc, argv, "M", 1);
  const int64_t K = arg_value(argc, argv, "K", 6656);
  const int64_t N = arg_value(argc, argv, "N", 202048);
  const int64_t gs = arg_value(argc, argv, "gs", 32);
  const int64_t warmup = arg_value(argc, argv, "warmup", 30);
  const int64_t iterations = arg_value(argc, argv, "iters", 200);
  const int64_t seeds = arg_value(argc, argv, "seeds", 3);

  if (M < 1 || M > 4 || K % 256 != 0 || K % 32 != 0 || gs < 32 ||
      (gs & (gs - 1))) {
    std::fprintf(stderr, "unsupported shape M=%lld N=%lld K=%lld gs=%lld\n", M, N, K, gs);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(0));
  const int32_t gs_shift = log2_pow2_host(static_cast<int32_t>(gs));
  const int32_t n_groups = static_cast<int32_t>(K / gs);
  const int32_t n_super = static_cast<int32_t>(K / 256);
  const int32_t n_q8_blocks = static_cast<int32_t>(K / cuda_shims::Q8_BLOCK_SIZE_I5);
  dim3 q8_grid((n_q8_blocks + cuda_shims::MV5_NWARPS - 1) / cuda_shims::MV5_NWARPS, M);
  dim3 block(cuda_shims::MV5_WARP_SIZE, cuda_shims::MV5_NWARPS);
  dim3 grid((N + cuda_shims::MV5_NWARPS - 1) / cuda_shims::MV5_NWARPS);

  std::printf(
      "shape M=%lld N=%lld K=%lld gs=%lld q8_blocks=%d warmup=%lld iters=%lld seeds=%lld\n",
      M,
      N,
      K,
      gs,
      n_q8_blocks,
      warmup,
      iterations,
      seeds);

  double ref_total = 0.0;
  double cand_total = 0.0;
  for (int64_t seed_idx = 0; seed_idx < seeds; ++seed_idx) {
    std::vector<uint16_t> hA;
    std::vector<uint8_t> hql;
    std::vector<uint8_t> hqh;
    std::vector<uint8_t> hscale;
    std::vector<uint16_t> hscale_step;
    std::vector<uint8_t> hzero;
    std::vector<uint16_t> hzero_step;
    fill_case(
        M,
        N,
        K,
        gs,
        3001 + seed_idx * 17,
        hA,
        hql,
        hqh,
        hscale,
        hscale_step,
        hzero,
        hzero_step);

    auto* dA = static_cast<__nv_bfloat16*>(device_alloc_copy(hA.data(), hA.size() * sizeof(uint16_t)));
    auto* dql = static_cast<uint8_t*>(device_alloc_copy(hql.data(), hql.size() * sizeof(uint8_t)));
    auto* dqh = static_cast<uint8_t*>(device_alloc_copy(hqh.data(), hqh.size() * sizeof(uint8_t)));
    auto* dscale = static_cast<uint8_t*>(device_alloc_copy(hscale.data(), hscale.size() * sizeof(uint8_t)));
    auto* dscale_step = static_cast<__half*>(device_alloc_copy(hscale_step.data(), hscale_step.size() * sizeof(uint16_t)));
    auto* dzero = static_cast<uint8_t*>(device_alloc_copy(hzero.data(), hzero.size() * sizeof(uint8_t)));
    auto* dzero_step = static_cast<__half*>(device_alloc_copy(hzero_step.data(), hzero_step.size() * sizeof(uint16_t)));
    cuda_shims::Q8Block_i5* dq8 = nullptr;
    __nv_bfloat16* dref = nullptr;
    __nv_bfloat16* dcand = nullptr;
    CUDA_CHECK(cudaMalloc(&dq8, M * n_q8_blocks * sizeof(cuda_shims::Q8Block_i5)));
    CUDA_CHECK(cudaMalloc(&dref, M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dcand, M * N * sizeof(__nv_bfloat16)));

    cuda_shims::quantize_activations_q8_i5_kernel<<<q8_grid, block>>>(dA, dq8, static_cast<int32_t>(K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cuda_shims::int5_w5a8_matvec_kernel<<<grid, block>>>(
        dql,
        dqh,
        dscale,
        dscale_step,
        dzero,
        dzero_step,
        dq8,
        dref,
        static_cast<int32_t>(N),
        static_cast<int32_t>(K),
        static_cast<int32_t>(M),
        gs_shift,
        n_groups,
        n_super);
    if (M == 1 && gs == 32) {
      cuda_shims::int5_w5a8_matvec_m1_gs32_kernel<<<grid, block>>>(
          dql,
          dqh,
          dscale,
          dscale_step,
          dzero,
          dzero_step,
          dq8,
          dcand,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          n_groups,
          n_super);
    } else if (M == 4 && gs == 32) {
      cuda_shims::int5_w5a8_matvec_m4_gs32_kernel<<<grid, block>>>(
          dql,
          dqh,
          dscale,
          dscale_step,
          dzero,
          dzero_step,
          dq8,
          dcand,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          n_groups,
          n_super);
    } else {
      cuda_shims::int5_w5a8_matvec_kernel<<<grid, block>>>(
          dql,
          dqh,
          dscale,
          dscale_step,
          dzero,
          dzero_step,
          dq8,
          dcand,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          static_cast<int32_t>(M),
          gs_shift,
          n_groups,
          n_super);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint16_t> href(M * N);
    std::vector<uint16_t> hcand(M * N);
    CUDA_CHECK(cudaMemcpy(href.data(), dref, href.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hcand.data(), dcand, hcand.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (size_t i = 0; i < href.size(); ++i) {
      if (href[i] != hcand[i]) {
        if (mismatches < 8) {
          std::printf("mismatch seed=%lld idx=%zu ref=0x%04x cand=0x%04x\n", seed_idx, i, href[i], hcand[i]);
        }
        ++mismatches;
      }
    }
    if (mismatches != 0) {
      std::printf("bitwise=FAIL mismatches=%zu/%zu\n", mismatches, href.size());
      return 1;
    }

    float q8_ms = time_ms([&]() {
      cuda_shims::quantize_activations_q8_i5_kernel<<<q8_grid, block>>>(dA, dq8, static_cast<int32_t>(K));
    }, warmup, iterations);
    float ref_ms = time_ms([&]() {
      cuda_shims::int5_w5a8_matvec_kernel<<<grid, block>>>(
          dql, dqh, dscale, dscale_step, dzero, dzero_step, dq8, dref, static_cast<int32_t>(N), static_cast<int32_t>(K), static_cast<int32_t>(M), gs_shift, n_groups, n_super);
    }, warmup, iterations);
    float cand_ms = time_ms([&]() {
      if (M == 1 && gs == 32) {
        cuda_shims::int5_w5a8_matvec_m1_gs32_kernel<<<grid, block>>>(
            dql, dqh, dscale, dscale_step, dzero, dzero_step, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), n_groups, n_super);
      } else if (M == 4 && gs == 32) {
        cuda_shims::int5_w5a8_matvec_m4_gs32_kernel<<<grid, block>>>(
            dql, dqh, dscale, dscale_step, dzero, dzero_step, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), n_groups, n_super);
      } else {
        cuda_shims::int5_w5a8_matvec_kernel<<<grid, block>>>(
            dql, dqh, dscale, dscale_step, dzero, dzero_step, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), static_cast<int32_t>(M), gs_shift, n_groups, n_super);
      }
    }, warmup, iterations);
    ref_total += ref_ms;
    cand_total += cand_ms;
    std::printf(
        "seed=%lld bitwise=OK q8_ms=%.6f ref_matvec_ms=%.6f cand_matvec_ms=%.6f speedup=%.3fx\n",
        seed_idx,
        q8_ms,
        ref_ms,
        cand_ms,
        ref_ms / cand_ms);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dql));
    CUDA_CHECK(cudaFree(dqh));
    CUDA_CHECK(cudaFree(dscale));
    CUDA_CHECK(cudaFree(dscale_step));
    CUDA_CHECK(cudaFree(dzero));
    CUDA_CHECK(cudaFree(dzero_step));
    CUDA_CHECK(cudaFree(dq8));
    CUDA_CHECK(cudaFree(dref));
    CUDA_CHECK(cudaFree(dcand));
  }

  std::printf(
      "avg ref_matvec_ms=%.6f cand_matvec_ms=%.6f speedup=%.3fx\n",
      ref_total / seeds,
      cand_total / seeds,
      ref_total / cand_total);
  return 0;
}
