/* Copyright (c) Meta Platforms, Inc. and affiliates. */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <executorch/backends/cuda/runtime/shims/int6_plain_mm.cuh>

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

__global__ void __launch_bounds__(cuda_shims::MV6_THREADS)
    int6_w6a8_matvec_reference_kernel(
        const uint8_t* __restrict__ ql,
        const uint8_t* __restrict__ qh,
        const int8_t* __restrict__ w_scale,
        const __half* __restrict__ w_scale_step,
        const cuda_shims::Q8Block_i6* __restrict__ q8,
        __nv_bfloat16* __restrict__ out,
        int32_t N,
        int32_t K,
        int32_t M,
        int32_t gs_shift,
        int32_t n_groups,
        int32_t n_super) {
  const int32_t n = blockIdx.x * cuda_shims::MV6_NWARPS + threadIdx.y;
  const int32_t m = blockIdx.y;
  if (n >= N || m >= M) {
    return;
  }

  const int32_t K_half = K / 2;
  const int32_t K_quarter = K / 4;
  const int32_t lane_id = threadIdx.x;
  const int32_t n_q8_blocks = K / cuda_shims::Q8_BLOCK_SIZE_I6;

  const uint8_t* qlrow = ql + static_cast<int64_t>(n) * K_half;
  const uint8_t* qhrow = qh + static_cast<int64_t>(n) * K_quarter;
  const int8_t* scale_row = w_scale + static_cast<int64_t>(n) * n_groups;
  const __half* scale_step_row =
      w_scale_step + static_cast<int64_t>(n) * n_super;
  const cuda_shims::Q8Block_i6* q8_row =
      q8 + static_cast<int64_t>(m) * n_q8_blocks;

  const uint4* qlrow16 = reinterpret_cast<const uint4*>(qlrow);
  const uint2* qhrow8 = reinterpret_cast<const uint2*>(qhrow);
  const int32_t K_half_16 = K_half / 16;
  const int32_t sb_shift = cuda_shims::SUPER_BLOCK_SHIFT_I6 - gs_shift;
  const int32_t wpg_shift = gs_shift - 3;

  float sum = 0.0f;
  for (int32_t i = lane_id; i < K_half_16; i += cuda_shims::MV6_WARP_SIZE) {
    uint4 packed16 = __ldg(&qlrow16[i]);
    uint2 qh_chunk = __ldg(&qhrow8[i]);
    int32_t k_base = i * 32;
    uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};
    uint32_t hi_even_word = qh_chunk.x;
    uint32_t hi_odd_word = qh_chunk.y;
    int32_t g_base = k_base >> gs_shift;
    float scale_step =
        __half2float(__ldg(&scale_step_row[g_base >> sb_shift]));
    float ws0 = static_cast<float>(__ldg(&scale_row[g_base])) * scale_step;
    float ws1 = static_cast<float>(__ldg(&scale_row[g_base + 1])) * scale_step;
    const cuda_shims::Q8Block_i6* qb = &q8_row[i];
    uint4 ae = *reinterpret_cast<const uint4*>(qb->qs_even);
    uint4 ao = *reinterpret_cast<const uint4*>(qb->qs_odd);
    float a_scale = qb->d;
    uint32_t a_even[4] = {ae.x, ae.y, ae.z, ae.w};
    uint32_t a_odd[4] = {ao.x, ao.y, ao.z, ao.w};

#pragma unroll
    for (int32_t w = 0; w < 4; ++w) {
      uint32_t packed = words[w];
      int32_t vi_lo = static_cast<int32_t>(packed & 0x0F0F0F0F);
      int32_t vi_hi = static_cast<int32_t>((packed >> 4) & 0x0F0F0F0F);
      uint32_t hi_even_byte = (hi_even_word >> (w * 8)) & 0xFF;
      uint32_t hi_odd_byte = (hi_odd_word >> (w * 8)) & 0xFF;
      int32_t vfull_even =
          vi_lo | static_cast<int32_t>(cuda_shims::spread2_i6(hi_even_byte) << 4);
      int32_t vfull_odd =
          vi_hi | static_cast<int32_t>(cuda_shims::spread2_i6(hi_odd_byte) << 4);
      int32_t dp = __dp4a(vfull_even, static_cast<int32_t>(a_even[w]), 0);
      dp = __dp4a(vfull_odd, static_cast<int32_t>(a_odd[w]), dp);
      int32_t a_sum =
          __dp4a(0x01010101, static_cast<int32_t>(a_even[w]), 0);
      a_sum = __dp4a(0x01010101, static_cast<int32_t>(a_odd[w]), a_sum);
      float ws = (w >> wpg_shift) ? ws1 : ws0;
      sum += ws * a_scale *
          (static_cast<float>(dp) - 32.0f * static_cast<float>(a_sum));
    }
  }

  for (int offset = cuda_shims::MV6_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  if (lane_id == 0) {
    out[static_cast<int64_t>(m) * N + n] = __float2bfloat16(sum);
  }
}

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
    std::vector<int8_t>& scale,
    std::vector<uint16_t>& steps) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> adist(-2.0f, 2.0f);
  std::uniform_int_distribution<int> qdist(0, 63);
  std::uniform_int_distribution<int> sdist(-8, 8);
  std::uniform_real_distribution<float> stepdist(0.003f, 0.035f);

  A.resize(M * K);
  ql.assign(N * (K / 2), 0);
  qh.assign(N * (K / 4), 0);
  scale.resize(N * (K / gs));
  steps.resize(N * (K / 256));

  for (auto& x : A) {
    x = float_to_bf16(adist(rng));
  }
  for (auto& x : scale) {
    int v = sdist(rng);
    x = static_cast<int8_t>(v == 0 ? 1 : v);
  }
  for (auto& x : steps) {
    x = __half_as_ushort(__float2half(stepdist(rng)));
  }

  std::vector<uint8_t> u(K);
  for (int64_t n = 0; n < N; ++n) {
    uint8_t* qlrow = ql.data() + n * (K / 2);
    uint8_t* qhrow = qh.data() + n * (K / 4);
    for (int64_t k = 0; k < K; ++k) {
      u[k] = static_cast<uint8_t>(qdist(rng));
    }
    for (int64_t k = 0; k < K; k += 2) {
      qlrow[k / 2] = static_cast<uint8_t>((u[k] & 0xF) | ((u[k + 1] & 0xF) << 4));
    }
    for (int64_t k = 0; k < K; k += 32) {
      uint8_t* chunk = qhrow + k / 4;
      for (int w = 0; w < 4; ++w) {
        uint8_t he = 0;
        uint8_t ho = 0;
        for (int j = 0; j < 4; ++j) {
          he |= static_cast<uint8_t>(((u[k + w * 8 + j * 2] >> 4) & 0x3) << (j * 2));
          ho |= static_cast<uint8_t>(((u[k + w * 8 + j * 2 + 1] >> 4) & 0x3) << (j * 2));
        }
        chunk[w] = he;
        chunk[4 + w] = ho;
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
  const int64_t M = arg_value(argc, argv, "M", 4);
  const int64_t K = arg_value(argc, argv, "K", 6656);
  const int64_t N = arg_value(argc, argv, "N", 6656);
  const int64_t gs = arg_value(argc, argv, "gs", 16);
  const int64_t warmup = arg_value(argc, argv, "warmup", 30);
  const int64_t iterations = arg_value(argc, argv, "iters", 200);
  const int64_t seeds = arg_value(argc, argv, "seeds", 3);

  if (M < 1 || M > 4 || K % 256 != 0 || K % 32 != 0 || gs < 8 || (gs & (gs - 1))) {
    std::fprintf(stderr, "unsupported shape M=%lld N=%lld K=%lld gs=%lld\n", M, N, K, gs);
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(0));
  int32_t gs_shift = log2_pow2_host(static_cast<int32_t>(gs));
  int32_t n_groups = static_cast<int32_t>(K / gs);
  int32_t n_super = static_cast<int32_t>(K / 256);
  int32_t n_q8_blocks = static_cast<int32_t>(K / cuda_shims::Q8_BLOCK_SIZE_I6);
  dim3 q8_grid((n_q8_blocks + cuda_shims::MV6_NWARPS - 1) / cuda_shims::MV6_NWARPS, M);
  dim3 block(cuda_shims::MV6_WARP_SIZE, cuda_shims::MV6_NWARPS);
  dim3 ref_grid((N + cuda_shims::MV6_NWARPS - 1) / cuda_shims::MV6_NWARPS, M);
  dim3 cand_grid((N + cuda_shims::MV6_NWARPS - 1) / cuda_shims::MV6_NWARPS);

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
    std::vector<int8_t> hscale;
    std::vector<uint16_t> hsteps;
    fill_case(M, N, K, gs, 1009 + seed_idx * 17, hA, hql, hqh, hscale, hsteps);

    auto* dA = static_cast<__nv_bfloat16*>(device_alloc_copy(hA.data(), hA.size() * sizeof(uint16_t)));
    auto* dql = static_cast<uint8_t*>(device_alloc_copy(hql.data(), hql.size() * sizeof(uint8_t)));
    auto* dqh = static_cast<uint8_t*>(device_alloc_copy(hqh.data(), hqh.size() * sizeof(uint8_t)));
    auto* dscale = static_cast<int8_t*>(device_alloc_copy(hscale.data(), hscale.size() * sizeof(int8_t)));
    auto* dsteps = static_cast<__half*>(device_alloc_copy(hsteps.data(), hsteps.size() * sizeof(uint16_t)));
    cuda_shims::Q8Block_i6* dq8 = nullptr;
    __nv_bfloat16* dref = nullptr;
    __nv_bfloat16* dcand = nullptr;
    CUDA_CHECK(cudaMalloc(&dq8, M * n_q8_blocks * sizeof(cuda_shims::Q8Block_i6)));
    CUDA_CHECK(cudaMalloc(&dref, M * N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dcand, M * N * sizeof(__nv_bfloat16)));

    cuda_shims::quantize_activations_q8_i6_kernel<<<q8_grid, block>>>(dA, dq8, static_cast<int32_t>(K));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (M == 3 && gs == 16) {
      cuda_shims::int6_w6a8_matvec_m3_gs16_kernel<<<cand_grid, block>>>(
          dql,
          dqh,
          dscale,
          dsteps,
          dq8,
          dref,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          n_groups,
          n_super);
    } else {
      int6_w6a8_matvec_reference_kernel<<<ref_grid, block>>>(
          dql,
          dqh,
          dscale,
          dsteps,
          dq8,
          dref,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          static_cast<int32_t>(M),
          gs_shift,
          n_groups,
          n_super);
    }
    if (M == 4 && gs == 16) {
      cuda_shims::int6_w6a8_matvec_m4_sum_gs16_kernel<<<cand_grid, block>>>(
          dql,
          dqh,
          dscale,
          dsteps,
          dq8,
          dcand,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          n_groups,
          n_super);
    } else if (M == 3 && gs == 16) {
      cuda_shims::int6_w6a8_matvec_m3_sum_gs16_kernel<<<cand_grid, block>>>(
          dql,
          dqh,
          dscale,
          dsteps,
          dq8,
          dcand,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          n_groups,
          n_super);
    } else if (M == 2 && gs == 16) {
      cuda_shims::int6_w6a8_matvec_m2_gs16_kernel<<<cand_grid, block>>>(
          dql,
          dqh,
          dscale,
          dsteps,
          dq8,
          dcand,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          n_groups,
          n_super);
    } else if (M == 1 && gs == 16) {
      cuda_shims::int6_w6a8_matvec_m1_gs16_kernel<<<cand_grid, block>>>(
          dql,
          dqh,
          dscale,
          dsteps,
          dq8,
          dcand,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          n_groups,
          n_super);
    } else if (M == 4) {
      cuda_shims::int6_w6a8_matvec_m4_kernel<<<cand_grid, block>>>(
          dql,
          dqh,
          dscale,
          dsteps,
          dq8,
          dcand,
          static_cast<int32_t>(N),
          static_cast<int32_t>(K),
          gs_shift,
          n_groups,
          n_super);
    } else {
      cuda_shims::int6_w6a8_matvec_kernel<<<cand_grid, block>>>(
          dql,
          dqh,
          dscale,
          dsteps,
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
      cuda_shims::quantize_activations_q8_i6_kernel<<<q8_grid, block>>>(dA, dq8, static_cast<int32_t>(K));
    }, warmup, iterations);
    float ref_ms = time_ms([&]() {
      if (M == 3 && gs == 16) {
        cuda_shims::int6_w6a8_matvec_m3_gs16_kernel<<<cand_grid, block>>>(
            dql, dqh, dscale, dsteps, dq8, dref, static_cast<int32_t>(N), static_cast<int32_t>(K), n_groups, n_super);
      } else {
        int6_w6a8_matvec_reference_kernel<<<ref_grid, block>>>(
            dql, dqh, dscale, dsteps, dq8, dref, static_cast<int32_t>(N), static_cast<int32_t>(K), static_cast<int32_t>(M), gs_shift, n_groups, n_super);
      }
    }, warmup, iterations);
    float cand_ms = time_ms([&]() {
      if (M == 4 && gs == 16) {
        cuda_shims::int6_w6a8_matvec_m4_sum_gs16_kernel<<<cand_grid, block>>>(
            dql, dqh, dscale, dsteps, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), n_groups, n_super);
      } else if (M == 3 && gs == 16) {
        cuda_shims::int6_w6a8_matvec_m3_sum_gs16_kernel<<<cand_grid, block>>>(
            dql, dqh, dscale, dsteps, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), n_groups, n_super);
      } else if (M == 2 && gs == 16) {
        cuda_shims::int6_w6a8_matvec_m2_gs16_kernel<<<cand_grid, block>>>(
            dql, dqh, dscale, dsteps, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), n_groups, n_super);
      } else if (M == 1 && gs == 16) {
        cuda_shims::int6_w6a8_matvec_m1_gs16_kernel<<<cand_grid, block>>>(
            dql, dqh, dscale, dsteps, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), n_groups, n_super);
      } else if (M == 4) {
        cuda_shims::int6_w6a8_matvec_m4_kernel<<<cand_grid, block>>>(
            dql, dqh, dscale, dsteps, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), gs_shift, n_groups, n_super);
      } else {
        cuda_shims::int6_w6a8_matvec_kernel<<<cand_grid, block>>>(
            dql, dqh, dscale, dsteps, dq8, dcand, static_cast<int32_t>(N), static_cast<int32_t>(K), static_cast<int32_t>(M), gs_shift, n_groups, n_super);
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
    CUDA_CHECK(cudaFree(dsteps));
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
