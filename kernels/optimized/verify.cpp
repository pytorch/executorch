// Standalone on-device verifier for the optimized_kernels implementations of
// grid_sampler_2d.out and sum.IntList_out, cross-checked against an fp32
// reference derived from the portable kernel.
//
// Build target: `verify_optimized_kernels` (opt-in via
// -DEXECUTORCH_BUILD_OPTIMIZED_VERIFY=ON).
//
// Usage:
//   adb push /path/to/verify_optimized_kernels /data/local/tmp/
//   adb shell /data/local/tmp/verify_optimized_kernels
//
// Reports max abs / max rel diff per test case. Non-zero exit on divergence
// beyond tolerance. For fp16 tests, reference is portable run on up-cast
// fp32 inputs, then down-cast — independent of portable's own fp16 path.

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <vector>

using executorch::aten::ArrayRef;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::etensor::TensorImpl;
using executorch::aten::DimOrderType;
using executorch::aten::SizesType;
using executorch::aten::StridesType;
using torch::executor::KernelRuntimeContext;

// ======================================================================
// Forward decls for the two implementations we want to compare.
// ======================================================================

namespace torch {
namespace executor {
namespace native {
// Portable reference implementations.
Tensor& sum_dim_out(
    KernelRuntimeContext&,
    const Tensor&,
    std::optional<ArrayRef<int64_t>>,
    bool,
    std::optional<ScalarType>,
    Tensor&);
Tensor& grid_sampler_2d_out(
    KernelRuntimeContext&,
    const Tensor&,
    const Tensor&,
    int64_t,
    int64_t,
    bool,
    Tensor&);
// Optimized implementations (this directory).
Tensor& opt_sum_dim_out(
    KernelRuntimeContext&,
    const Tensor&,
    std::optional<ArrayRef<int64_t>>,
    bool,
    std::optional<ScalarType>,
    Tensor&);
Tensor& opt_grid_sampler_2d_out(
    KernelRuntimeContext&,
    const Tensor&,
    const Tensor&,
    int64_t,
    int64_t,
    bool,
    Tensor&);
} // namespace native
} // namespace executor
} // namespace torch

// ======================================================================
// Tiny tensor builder: owns storage + metadata, hands out a Tensor view.
// ======================================================================

struct OwnedTensor {
  std::vector<SizesType> sizes;
  std::vector<DimOrderType> dim_order;
  std::vector<StridesType> strides;
  std::vector<uint8_t> storage;
  std::unique_ptr<TensorImpl> impl;

  OwnedTensor() = default;
  OwnedTensor(const OwnedTensor&) = delete;
  OwnedTensor& operator=(const OwnedTensor&) = delete;
  OwnedTensor(OwnedTensor&&) = default;
  OwnedTensor& operator=(OwnedTensor&&) = default;

  static OwnedTensor make(ScalarType dtype, std::vector<int32_t> shape) {
    OwnedTensor t;
    t.sizes.assign(shape.begin(), shape.end());
    t.dim_order.resize(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      t.dim_order[i] = static_cast<DimOrderType>(i);
    }
    t.strides.resize(shape.size());
    int32_t running = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      t.strides[i] = running;
      running *= shape[i];
    }
    size_t numel = 1;
    for (auto s : shape) {
      numel *= static_cast<size_t>(s);
    }
    size_t elem_size = (dtype == ScalarType::Float) ? 4
        : (dtype == ScalarType::Half)               ? 2
                                                    : 4;
    t.storage.assign(numel * elem_size, 0);
    t.impl = std::make_unique<TensorImpl>(
        dtype,
        static_cast<ssize_t>(shape.size()),
        t.sizes.data(),
        t.storage.data(),
        t.dim_order.data(),
        t.strides.data());
    return t;
  }

  Tensor view() {
    // Hold the Tensor as a member so callers can bind it to Tensor&
    // parameters (the kernel signatures take non-const Tensor& for the out
    // tensor). Refreshes each call because TensorImpl pointer may shift
    // if the OwnedTensor is moved (though we disallow move in practice).
    tensor_view_ = Tensor(impl.get());
    return tensor_view_;
  }
  Tensor& view_ref() {
    tensor_view_ = Tensor(impl.get());
    return tensor_view_;
  }

  template <typename T>
  T* data() {
    return reinterpret_cast<T*>(storage.data());
  }

  size_t numel() const {
    size_t n = 1;
    for (auto s : sizes) {
      n *= static_cast<size_t>(s);
    }
    return n;
  }

 private:
  Tensor tensor_view_{nullptr};
};

// ======================================================================
// Compare helpers.
// ======================================================================

template <typename T>
struct DiffStats {
  double max_abs = 0;
  double max_rel_nonzero = 0; // rel diff ignoring near-zero cells
  size_t violations = 0;      // elements failing combined tol check
  size_t count = 0;
  // How many elements were near-zero enough that rel is meaningless.
  size_t near_zero = 0;

  bool passes() const {
    return violations == 0;
  }
};

// numpy.testing.assert_allclose semantics:
//   |a - b| <= abs_tol + rel_tol * |b|
// Near-zero cells are bounded by abs_tol alone; away from zero, rel_tol
// dominates. Avoids the "relative error explodes at zero crossings" trap.
template <typename T>
DiffStats<T> diff(
    const T* a,
    const T* b,
    size_t n,
    double abs_tol,
    double rel_tol) {
  DiffStats<T> s;
  s.count = n;
  for (size_t i = 0; i < n; ++i) {
    double va = static_cast<double>(a[i]);
    double vb = static_cast<double>(b[i]);
    double abs_d = std::fabs(va - vb);
    double bound = abs_tol + rel_tol * std::fabs(vb);
    if (abs_d > bound) {
      ++s.violations;
    }
    s.max_abs = std::max(s.max_abs, abs_d);
    double mag = std::max(std::fabs(va), std::fabs(vb));
    if (mag < 10 * abs_tol) {
      ++s.near_zero;
    } else {
      s.max_rel_nonzero = std::max(s.max_rel_nonzero, abs_d / mag);
    }
  }
  return s;
}

// Half <-> float conversion via ARM fp16 type (aarch64-only). We already build
// with -march=armv8.2-a+fp16, so these are cheap.
#ifdef __aarch64__
#include <arm_neon.h>
static inline float half_to_float(uint16_t h) {
  __fp16 f;
  std::memcpy(&f, &h, sizeof(f));
  return static_cast<float>(f);
}
static inline uint16_t float_to_half(float f) {
  __fp16 h = static_cast<__fp16>(f);
  uint16_t u;
  std::memcpy(&u, &h, sizeof(u));
  return u;
}
#endif

// ======================================================================
// Test cases.
// ======================================================================

struct TestResult {
  const char* name;
  size_t n;
  double max_abs;
  double max_rel;
  bool passed;
};

static std::vector<TestResult> results;

template <typename T>
void report(
    const char* name,
    const DiffStats<T>& s,
    double abs_tol,
    double rel_tol) {
  bool ok = s.passes();
  results.push_back({name, s.count, s.max_abs, s.max_rel_nonzero, ok});
  std::printf(
      "  %-58s  n=%-7zu max_abs=%-10.3g max_rel(far)=%-10.3g near_zero=%-5zu viol=%-4zu  [%s]\n",
      name,
      s.count,
      s.max_abs,
      s.max_rel_nonzero,
      s.near_zero,
      s.violations,
      ok ? "PASS" : "FAIL");
}

// ---------- grid_sampler_2d tests ----------

template <ScalarType DTYPE, typename T>
static void test_grid_sampler(
    const char* label,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    bool align_corners) {
  auto input = OwnedTensor::make(DTYPE, {N, C, H_in, W_in});
  auto grid = OwnedTensor::make(DTYPE, {N, H_out, W_out, 2});
  auto out_neon = OwnedTensor::make(DTYPE, {N, C, H_out, W_out});

  // For fp16 / bf16 the portable kernel's own fp16 path is itself imprecise
  // (catastrophic cancellation on weight computation). We compute the
  // reference by up-casting inputs to fp32, running portable in fp32, and
  // down-casting the output — that's the "best achievable" fp16 output.
  // For fp32 this upcast is a no-op and we just run portable directly.
  auto input_f = OwnedTensor::make(ScalarType::Float, {N, C, H_in, W_in});
  auto grid_f = OwnedTensor::make(ScalarType::Float, {N, H_out, W_out, 2});
  auto out_ref_f =
      OwnedTensor::make(ScalarType::Float, {N, C, H_out, W_out});

  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

  auto* in_d = input.data<T>();
  auto* in_fd = input_f.data<float>();
  for (size_t i = 0; i < input.numel(); ++i) {
    float v = ud(rng);
#ifdef __aarch64__
    if constexpr (std::is_same_v<T, uint16_t>) {
      in_d[i] = float_to_half(v);
      in_fd[i] = half_to_float(in_d[i]); // round-trip to match fp16 input
    } else {
      in_d[i] = static_cast<T>(v);
      in_fd[i] = static_cast<float>(in_d[i]);
    }
#else
    in_d[i] = static_cast<T>(v);
    in_fd[i] = static_cast<float>(in_d[i]);
#endif
  }

  // Grid has spread of coordinates — mostly in [-1, 1] with some edges.
  auto* g_d = grid.data<T>();
  auto* g_fd = grid_f.data<float>();
  std::uniform_real_distribution<float> gd(-1.1f, 1.1f);
  for (size_t i = 0; i < grid.numel(); ++i) {
    float v = gd(rng);
#ifdef __aarch64__
    if constexpr (std::is_same_v<T, uint16_t>) {
      g_d[i] = float_to_half(v);
      g_fd[i] = half_to_float(g_d[i]);
    } else {
      g_d[i] = static_cast<T>(v);
      g_fd[i] = static_cast<float>(g_d[i]);
    }
#else
    g_d[i] = static_cast<T>(v);
    g_fd[i] = static_cast<float>(g_d[i]);
#endif
  }

  // Reference: portable kernel run on fp32 inputs.
  KernelRuntimeContext ctx_ref, ctx_neon;
  torch::executor::native::grid_sampler_2d_out(
      ctx_ref,
      input_f.view(),
      grid_f.view(),
      /*interpolation_mode=*/0,
      /*padding_mode=*/0,
      align_corners,
      out_ref_f.view_ref());

  // Optimized kernel on the native-dtype inputs.
  torch::executor::native::opt_grid_sampler_2d_out(
      ctx_neon,
      input.view(),
      grid.view(),
      /*interpolation_mode=*/0,
      /*padding_mode=*/0,
      align_corners,
      out_neon.view_ref());

  // Compare both to the fp32 reference. For fp16/bf16, down-cast the
  // reference to the optimized output's dtype before comparing — optimized can't
  // represent more precision than its output dtype allows.
  std::vector<float> ref_f(out_ref_f.numel());
  std::vector<float> neon_f(out_neon.numel());
  auto* ref_fd = out_ref_f.data<float>();
  auto* neon_d = out_neon.data<T>();
  for (size_t i = 0; i < ref_f.size(); ++i) {
#ifdef __aarch64__
    if constexpr (std::is_same_v<T, uint16_t>) {
      // Round reference through fp16 to match optimized output precision.
      uint16_t ref_h = float_to_half(ref_fd[i]);
      ref_f[i] = half_to_float(ref_h);
      neon_f[i] = half_to_float(neon_d[i]);
    } else {
      ref_f[i] = ref_fd[i];
      neon_f[i] = static_cast<float>(neon_d[i]);
    }
#else
    ref_f[i] = ref_fd[i];
    neon_f[i] = static_cast<float>(neon_d[i]);
#endif
  }
  // Portable and optimized both accumulate in fp32. For fp16 inputs the only
  // remaining difference is the final fp16 round-trip on store (half a ULP)
  // plus tiny FMA ordering noise.
  double abs_tol = (DTYPE == ScalarType::Float) ? 1e-5 : 1e-3;
  double rel_tol = (DTYPE == ScalarType::Float) ? 1e-4 : 2e-3;
  auto s = diff(ref_f.data(), neon_f.data(), ref_f.size(), abs_tol, rel_tol);
  report(label, s, abs_tol, rel_tol);
}

// ---------- sum.IntList_out tests ----------

template <ScalarType DTYPE, typename T>
static void test_sum(
    const char* label,
    std::vector<int32_t> input_shape,
    int64_t reduce_dim,
    bool keepdim) {
  auto input = OwnedTensor::make(DTYPE, input_shape);
  // Compute output shape.
  std::vector<int32_t> out_shape = input_shape;
  if (keepdim) {
    out_shape[reduce_dim] = 1;
  } else {
    out_shape.erase(out_shape.begin() + reduce_dim);
  }
  auto out_neon = OwnedTensor::make(DTYPE, out_shape);

  // Same strategy as the grid_sampler test: fp32 reference run on up-cast
  // inputs, then down-cast the output for comparison. Avoids depending on
  // portable's fp16 accumulator precision.
  auto input_f = OwnedTensor::make(ScalarType::Float, input_shape);
  auto out_ref_f = OwnedTensor::make(ScalarType::Float, out_shape);

  std::mt19937 rng(9999);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

  auto* in_d = input.data<T>();
  auto* in_fd = input_f.data<float>();
  for (size_t i = 0; i < input.numel(); ++i) {
    float v = ud(rng);
#ifdef __aarch64__
    if constexpr (std::is_same_v<T, uint16_t>) {
      in_d[i] = float_to_half(v);
      in_fd[i] = half_to_float(in_d[i]);
    } else {
      in_d[i] = static_cast<T>(v);
      in_fd[i] = static_cast<float>(in_d[i]);
    }
#else
    in_d[i] = static_cast<T>(v);
    in_fd[i] = static_cast<float>(in_d[i]);
#endif
  }

  std::array<int64_t, 1> dims = {reduce_dim};
  std::optional<ArrayRef<int64_t>> dim_list{ArrayRef<int64_t>(dims.data(), 1)};
  std::optional<ScalarType> dtype_opt = std::nullopt;

  KernelRuntimeContext ctx_ref, ctx_neon;
  torch::executor::native::sum_dim_out(
      ctx_ref,
      input_f.view(),
      dim_list,
      keepdim,
      dtype_opt,
      out_ref_f.view_ref());
  torch::executor::native::opt_sum_dim_out(
      ctx_neon, input.view(), dim_list, keepdim, dtype_opt, out_neon.view_ref());

  std::vector<float> ref_f(out_ref_f.numel());
  std::vector<float> neon_f(out_neon.numel());
  auto* ref_fd = out_ref_f.data<float>();
  auto* neon_d = out_neon.data<T>();
  for (size_t i = 0; i < ref_f.size(); ++i) {
#ifdef __aarch64__
    if constexpr (std::is_same_v<T, uint16_t>) {
      uint16_t ref_h = float_to_half(ref_fd[i]);
      ref_f[i] = half_to_float(ref_h);
      neon_f[i] = half_to_float(neon_d[i]);
    } else {
      ref_f[i] = ref_fd[i];
      neon_f[i] = static_cast<float>(neon_d[i]);
    }
#else
    ref_f[i] = ref_fd[i];
    neon_f[i] = static_cast<float>(neon_d[i]);
#endif
  }
  // Portable and optimized both accumulate in fp32. For fp16 inputs the only
  // remaining delta is the final fp16-cast on store and any FMA reordering.
  double abs_tol = (DTYPE == ScalarType::Float) ? 1e-4 : 1e-2;
  double rel_tol = (DTYPE == ScalarType::Float) ? 1e-4 : 2e-3;
  auto s = diff(ref_f.data(), neon_f.data(), ref_f.size(), abs_tol, rel_tol);
  report(label, s, abs_tol, rel_tol);
}

// ======================================================================
// Entry point.
// ======================================================================

int main() {
  std::printf(
      "=== grid_sampler_2d.out: optimized vs portable-fp32 reference ===\n"
      "(for fp16 tests, reference is portable run on up-cast fp32 inputs,\n"
      " then down-cast to fp16 — independent of portable's fp16 path)\n");
  test_grid_sampler<ScalarType::Float, float>(
      "grid_sampler fp32  N=1 C=16 in=32x48 out=24x32  align=0", 1, 16, 32, 48, 24, 32, false);
  test_grid_sampler<ScalarType::Float, float>(
      "grid_sampler fp32  N=1 C=32 in=72x96 out=72x96  align=1", 1, 32, 72, 96, 72, 96, true);
  test_grid_sampler<ScalarType::Float, float>(
      "grid_sampler fp32  N=1 C=7 (odd)  in=16x24 out=16x24  align=0", 1, 7, 16, 24, 16, 24, false);
  test_grid_sampler<ScalarType::Float, float>(
      "grid_sampler fp32  N=2 C=64 in=48x64 out=48x64  align=0", 2, 64, 48, 64, 48, 64, false);
#ifdef __aarch64__
  test_grid_sampler<ScalarType::Half, uint16_t>(
      "grid_sampler fp16  N=1 C=16 in=32x48 out=24x32  align=0", 1, 16, 32, 48, 24, 32, false);
  test_grid_sampler<ScalarType::Half, uint16_t>(
      "grid_sampler fp16  N=1 C=32 in=72x96 out=72x96  align=1", 1, 32, 72, 96, 72, 96, true);
#endif

  std::printf("\n=== sum.IntList_out: optimized vs portable-fp32 reference ===\n");
  // Innermost reduction.
  test_sum<ScalarType::Float, float>(
      "sum fp32  [1, 32, 192, 256] reduce=-1 keepdim=0",
      {1, 32, 192, 256},
      3,
      false);
  test_sum<ScalarType::Float, float>(
      "sum fp32  [2, 64, 128] reduce=2 keepdim=1",
      {2, 64, 128},
      2,
      true);
  // Middle (strided) reduction.
  test_sum<ScalarType::Float, float>(
      "sum fp32  [1, 32, 192, 256] reduce=1 keepdim=0",
      {1, 32, 192, 256},
      1,
      false);
  test_sum<ScalarType::Float, float>(
      "sum fp32  [4, 16, 64, 64] reduce=2 keepdim=0",
      {4, 16, 64, 64},
      2,
      false);
#ifdef __aarch64__
  test_sum<ScalarType::Half, uint16_t>(
      "sum fp16  [1, 32, 192, 256] reduce=-1 keepdim=0",
      {1, 32, 192, 256},
      3,
      false);
  test_sum<ScalarType::Half, uint16_t>(
      "sum fp16  [1, 32, 192, 256] reduce=1 keepdim=0",
      {1, 32, 192, 256},
      1,
      false);
#endif

  int failed = 0;
  for (auto& r : results) {
    if (!r.passed) ++failed;
  }
  std::printf(
      "\n=== %zu tests total, %d failed ===\n", results.size(), failed);
  return failed == 0 ? 0 : 1;
}
