/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/mm.h>
#include <executorch/runtime/platform/platform.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace executorch::backends::cuda;
using executorch::runtime::Error;
namespace slim_c10 = executorch::backends::aoti::slim::c10;

namespace {

// Small integers, so every product and sum below is exact in float16 and
// bfloat16 too, and results can be compared exactly.
float pattern(int64_t i) {
  return static_cast<float>((i * 7) % 5 - 2);
}

class AOTITorchCudaMmTest
    : public ::testing::TestWithParam<slim_c10::ScalarType> {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  void TearDown() override {
    for (Tensor* t : tensors_) {
      aoti_torch_delete_tensor_object(t);
    }
  }

  slim_c10::ScalarType dtype() const {
    return GetParam();
  }

  // A CUDA buffer of `numel` elements holding pattern(i) at element i.
  Tensor* buffer(int64_t numel) {
    Tensor* t = nullptr;
    EXPECT_EQ(
        aoti_torch_empty_strided(
            1,
            &numel,
            nullptr,
            static_cast<int32_t>(dtype()),
            static_cast<int32_t>(slim_c10::DeviceType::CUDA),
            0,
            &t),
        Error::Ok);
    tensors_.push_back(t);
    std::vector<float> values(numel);
    for (int64_t i = 0; i < numel; ++i) {
      values[i] = pattern(i);
    }
    upload(t, values);
    return t;
  }

  // A view of `base` with the given sizes and strides.
  Tensor* view(
      Tensor* base,
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides) {
    Tensor* t = nullptr;
    EXPECT_EQ(
        aoti_torch__reinterpret_tensor(
            base, sizes.size(), sizes.data(), strides.data(), 0, &t),
        Error::Ok);
    tensors_.push_back(t);
    return t;
  }

  void upload(Tensor* t, const std::vector<float>& values) {
    std::vector<uint16_t> half(values.size());
    const void* src = values.data();
    if (dtype() == slim_c10::ScalarType::Half) {
      for (size_t i = 0; i < values.size(); ++i) {
        half[i] = __half_as_ushort(__float2half(values[i]));
      }
      src = half.data();
    } else if (dtype() == slim_c10::ScalarType::BFloat16) {
      for (size_t i = 0; i < values.size(); ++i) {
        half[i] = __bfloat16_as_ushort(__float2bfloat16(values[i]));
      }
      src = half.data();
    }
    ASSERT_EQ(
        cudaMemcpy(
            t->data_ptr(),
            src,
            values.size() * t->itemsize(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
  }

  // Every element of the contiguous buffer `base`, as floats.
  std::vector<float> download(Tensor* base) {
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const size_t numel = base->numel();
    std::vector<float> out(numel);
    if (dtype() == slim_c10::ScalarType::Float) {
      EXPECT_EQ(
          cudaMemcpy(
              out.data(),
              base->data_ptr(),
              numel * sizeof(float),
              cudaMemcpyDeviceToHost),
          cudaSuccess);
      return out;
    }
    std::vector<uint16_t> half(numel);
    EXPECT_EQ(
        cudaMemcpy(
            half.data(),
            base->data_ptr(),
            numel * sizeof(uint16_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t i = 0; i < numel; ++i) {
      out[i] = dtype() == slim_c10::ScalarType::Half
          ? __half2float(__ushort_as_half(half[i]))
          : __bfloat162float(__ushort_as_bfloat16(half[i]));
    }
    return out;
  }

  // Reads element (b, r, c) of a 2-D or 3-D view from the host copy of the
  // pattern its base was filled with.
  static float at(Tensor* t, int64_t b, int64_t r, int64_t c) {
    const int64_t d = static_cast<int64_t>(t->dim());
    const int64_t batch_stride = d == 3 ? t->stride(0) : 0;
    return pattern(
        b * batch_stride + r * t->stride(d - 2) + c * t->stride(d - 1));
  }

  // Checks out (contiguous base `out_base` viewed as `out`) against
  // beta * self_value(b, r, c) + alpha * a @ b.
  template <typename SelfValue>
  void expect_product(
      Tensor* out_base,
      Tensor* out,
      Tensor* a,
      Tensor* b,
      float alpha,
      float beta,
      SelfValue self_value) {
    const std::vector<float> got = download(out_base);
    const int64_t d = static_cast<int64_t>(out->dim());
    const int64_t batch = d == 3 ? out->size(0) : 1;
    const int64_t rows = out->size(d - 2);
    const int64_t cols = out->size(d - 1);
    const int64_t inner = a->size(d - 1);
    const int64_t out_batch_stride = d == 3 ? out->stride(0) : 0;
    for (int64_t n = 0; n < batch; ++n) {
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          float expected = 0;
          for (int64_t k = 0; k < inner; ++k) {
            expected += at(a, n, r, k) * at(b, n, k, c);
          }
          expected = alpha * expected + beta * self_value(n, r, c);
          const int64_t index = n * out_batch_stride + r * out->stride(d - 2) +
              c * out->stride(d - 1);
          ASSERT_EQ(got[index], expected)
              << "batch " << n << " row " << r << " col " << c;
        }
      }
    }
  }

  // Records an m x k by k x n mm into a CUDA graph, replays it, and checks
  // the result. Counts the graph's memory allocations into *allocations, which
  // tells whether cuBLASLt used a workspace.
  void expect_captured_mm(
      int64_t m,
      int64_t k,
      int64_t n,
      size_t* allocations = nullptr) {
    Tensor* a = view(buffer(m * k), {m, k}, {k, 1});
    Tensor* b = view(buffer(k * n), {k, n}, {1, k});
    Tensor* out_base = buffer(m * n);
    aoti_torch_zero_(out_base);
    Tensor* out = view(out_base, {m, n}, {n, 1});

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(setCurrentCUDAStream(stream, 0), Error::Ok);
    ASSERT_EQ(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed),
        cudaSuccess);
    const Error captured = aoti_torch_cuda_mm_out(out, a, b);
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_EQ(captured, Error::Ok);
    if (allocations != nullptr) {
      size_t count = 0;
      ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &count), cudaSuccess);
      std::vector<cudaGraphNode_t> nodes(count);
      ASSERT_EQ(cudaGraphGetNodes(graph, nodes.data(), &count), cudaSuccess);
      *allocations = 0;
      for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        ASSERT_EQ(cudaGraphNodeGetType(node, &type), cudaSuccess);
        *allocations += type == cudaGraphNodeTypeMemAlloc ? 1 : 0;
      }
    }
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(
        cudaGraphInstantiate(
            &exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch),
        cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    clearCurrentCUDAStream(0);
    expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }

  std::vector<Tensor*> tensors_;
};

constexpr int64_t M = 5;
constexpr int64_t K = 7;
constexpr int64_t N = 3;

TEST_P(AOTITorchCudaMmTest, MmContiguous) {
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

// A view that starts one element into its storage is not 16-byte aligned.
// Sizes are multiples of 16 so that only the pointer is misaligned.
TEST_P(AOTITorchCudaMmTest, MmMisalignedOperand) {
  constexpr int64_t S = 64;
  Tensor* a_base = buffer(S * S + 1);
  Tensor* a = nullptr;
  const std::vector<int64_t> sizes = {S, S};
  const std::vector<int64_t> strides = {S, 1};
  ASSERT_EQ(
      aoti_torch__reinterpret_tensor(
          a_base, 2, sizes.data(), strides.data(), 1, &a),
      Error::Ok);
  tensors_.push_back(a);
  Tensor* b = view(buffer(S * S), {S, S}, {S, 1});
  Tensor* out_base = buffer(S * S);
  Tensor* out = view(out_base, {S, S}, {S, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  const std::vector<float> got = download(out_base);
  for (int64_t r = 0; r < S; ++r) {
    for (int64_t c = 0; c < S; ++c) {
      float expected = 0;
      for (int64_t k = 0; k < S; ++k) {
        expected += pattern(1 + r * S + k) * pattern(k * S + c);
      }
      ASSERT_EQ(got[r * S + c], expected) << "row " << r << " col " << c;
    }
  }
}

// A linear layer's weight arrives transposed (column-major), and a slice keeps
// its parent's row stride.
TEST_P(AOTITorchCudaMmTest, MmTransposedAndSlicedOperands) {
  Tensor* a = view(buffer(M * K), {M, K}, {1, M});
  Tensor* b = view(buffer(K * (N + 4)), {K, N}, {N + 4, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, MmColumnMajorResult) {
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {1, M});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

// A dimension of size 1 can have any stride, so these are still matrices:
// a row without a unit stride, and a 1 x 1 matrix expanded from a scalar.
TEST_P(AOTITorchCudaMmTest, MmRowWithoutUnitStride) {
  Tensor* a = view(buffer(2 * K), {1, K}, {2 * K, 2});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(N);
  Tensor* out = view(out_base, {1, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, MmOneByOneWithZeroStrides) {
  Tensor* a = view(buffer(1), {1, 1}, {0, 0});
  Tensor* b = view(buffer(N), {1, N}, {N, 1});
  Tensor* out_base = buffer(N);
  Tensor* out = view(out_base, {1, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, AddmmBiasRow) {
  Tensor* bias = view(buffer(N), {N}, {1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {1, K});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 2.0), Error::Ok);
  expect_product(out_base, out, a, b, 2, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

// A linear layer at decode time has one row.
TEST_P(AOTITorchCudaMmTest, AddmmBiasRowSingleRow) {
  Tensor* bias = view(buffer(N), {N}, {1});
  Tensor* a = view(buffer(K), {1, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {1, K});
  Tensor* out_base = buffer(N);
  Tensor* out = view(out_base, {1, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmBiasRowSingleColumn) {
  Tensor* bias = view(buffer(1), {1}, {1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K), {K, 1}, {1, 1});
  Tensor* out_base = buffer(M);
  Tensor* out = view(out_base, {M, 1}, {1, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmExpandedBias) {
  Tensor* bias = view(buffer(N), {M, N}, {0, 1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmFullSelf) {
  Tensor* self = view(buffer(M * N), {M, N}, {N, 1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 0.5, 2.0), Error::Ok);
  expect_product(
      out_base, out, a, b, 2, 0.5, [](int64_t, int64_t r, int64_t c) {
        return pattern(r * N + c);
      });
}

TEST_P(AOTITorchCudaMmTest, AddmmScaledBiasRow) {
  Tensor* bias = view(buffer(N), {N}, {1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, bias, a, b, 0.5, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0.5, [](int64_t, int64_t, int64_t c) {
    return pattern(c);
  });
}

// Inductor passes a broadcast self already expanded to M x N, with zero
// strides.
TEST_P(AOTITorchCudaMmTest, AddmmExpandedColumn) {
  Tensor* column = view(buffer(M), {M, N}, {1, 0});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, column, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t r, int64_t) {
    return pattern(r);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmExpandedScalar) {
  Tensor* scalar = view(buffer(1), {M, N}, {0, 0});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, scalar, a, b, 2.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 2, [](auto...) { return pattern(0); });
}

TEST_P(AOTITorchCudaMmTest, AddmmTransposedSelf) {
  Tensor* self = view(buffer(M * N), {M, N}, {1, M});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 1.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 1, [](int64_t, int64_t r, int64_t c) {
    return pattern(r + c * M);
  });
}

TEST_P(AOTITorchCudaMmTest, AddmmZeroBetaIgnoresSelf) {
  Tensor* self_base = buffer(M * N);
  upload(self_base, std::vector<float>(M * N, NAN));
  Tensor* self = view(self_base, {M, N}, {N, 1});
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out_base = buffer(M * N);
  Tensor* out = view(out_base, {M, N}, {N, 1});
  ASSERT_EQ(aoti_torch_cuda_addmm_out(out, self, a, b, 0.0, 1.0), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, Bmm) {
  constexpr int64_t B = 3;
  Tensor* a = view(buffer(B * M * K), {B, M, K}, {M * K, K, 1});
  Tensor* b = view(buffer(B * K * N), {B, K, N}, {K * N, 1, K});
  Tensor* out_base = buffer(B * M * N);
  Tensor* out = view(out_base, {B, M, N}, {M * N, N, 1});
  ASSERT_EQ(aoti_torch_cuda_bmm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

TEST_P(AOTITorchCudaMmTest, BmmBroadcastOperand) {
  constexpr int64_t B = 3;
  Tensor* a = view(buffer(B * M * K), {B, M, K}, {M * K, K, 1});
  Tensor* b = view(buffer(K * N), {B, K, N}, {0, N, 1});
  Tensor* out_base = buffer(B * M * N);
  Tensor* out = view(out_base, {B, M, N}, {M * N, N, 1});
  ASSERT_EQ(aoti_torch_cuda_bmm_out(out, a, b), Error::Ok);
  expect_product(out_base, out, a, b, 1, 0, [](auto...) { return 0.0f; });
}

// The backend records calls into a CUDA graph after its warm-up run and
// replays them.
TEST_P(AOTITorchCudaMmTest, MmCapturesIntoCudaGraph) {
  expect_captured_mm(M, K, N);
}

// Large enough that cuBLASLt asks for a workspace on an H100, so the graph
// also allocates and frees it. A GPU whose algorithm needs none skips.
TEST_P(AOTITorchCudaMmTest, MmWithWorkspaceCapturesIntoCudaGraph) {
  if (dtype() != slim_c10::ScalarType::Float) {
    GTEST_SKIP() << "Sums over this K are exact only in float32";
  }
  size_t allocations = 0;
  expect_captured_mm(16, 8192, 16, &allocations);
  if (allocations == 0) {
    GTEST_SKIP() << "cuBLASLt used no workspace for this shape on this GPU";
  }
}

TEST_P(AOTITorchCudaMmTest, RejectsMismatchedShapes) {
  Tensor* a = view(buffer(M * K), {M, K}, {K, 1});
  Tensor* b = view(buffer((K + 1) * N), {K + 1, N}, {N, 1});
  Tensor* out = view(buffer(M * N), {M, N}, {N, 1});
  EXPECT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::InvalidArgument);
}

TEST_P(AOTITorchCudaMmTest, RejectsOperandThatIsNotAMatrixLayout) {
  // Neither dimension is unit-stride.
  Tensor* a = view(buffer(2 * M * K), {M, K}, {2 * K, 2});
  Tensor* b = view(buffer(K * N), {K, N}, {N, 1});
  Tensor* out = view(buffer(M * N), {M, N}, {N, 1});
  EXPECT_EQ(aoti_torch_cuda_mm_out(out, a, b), Error::NotSupported);
}

INSTANTIATE_TEST_SUITE_P(
    Dtypes,
    AOTITorchCudaMmTest,
    ::testing::Values(
        slim_c10::ScalarType::Float,
        slim_c10::ScalarType::Half,
        slim_c10::ScalarType::BFloat16));

} // namespace
