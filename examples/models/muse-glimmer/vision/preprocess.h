/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// C++ mirror of vision/precompute.py. It converts one decoded image into the
// host-computed inputs expected by the exported vision encoder.

#pragma once

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>

#include <stb_image_resize.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch {
namespace examples {
namespace muse_glimmer_vision {

using ::executorch::extension::TensorPtr;

// Vision constants (mirror MuseGlimmerVisionConfig / mmproj clip metadata).
static constexpr int32_t kPatchSize = 14;
static constexpr int32_t kPatchTemporal = 2;
static constexpr int32_t kDownsample = 2;
static constexpr int32_t kSparseFactor = 4;
static constexpr int32_t kPosGrid = 32; // 32x32 positional table
static constexpr int32_t kLatent = 1536;
static constexpr int32_t kHeadDim = 96; // latent / n_heads(16)
static constexpr int32_t kPatchDim =
    kPatchTemporal * 3 * kPatchSize * kPatchSize; // 1176
// Cell = patch_size * downsample (image resized to a whole number of cells).
static constexpr int32_t kCell = kPatchSize * kDownsample; // 28
static constexpr int64_t kMaxImageTokens = 4096;

struct VisionInputs {
  TensorPtr patches; // (1, P, 1176) float32
  TensorPtr pos_emb; // (1, P, 1536) bfloat16
  TensorPtr cos_2d; // (P, 48) float32
  TensorPtr sin_2d; // (P, 48) float32
  TensorPtr sparse_perm; // (P,) int64
  TensorPtr inv_perm; // (P,) int64
  TensorPtr global_mask; // (1, 1, P, P) bool
  TensorPtr sparse_mask; // (1, 1, P, P) bool
  TensorPtr pixel_perm; // (P,) int64
  int64_t num_patches; // P
  int64_t num_soft_tokens; // P / downsample^2
};

// Reproduce vision_precompute.compute_grid_size, which picks the grid with
//
//     min(list(set(itertools.product([floor, ceil], [floor, ceil]))), key=cost)
//
// Equal-cost candidates are resolved by CPython set iteration order, not by
// grid size. The helpers below reproduce the deterministic small-integer tuple
// hashing and probing used by CPython 3.8 and later.
namespace detail {

constexpr int32_t kSetTableSize = 8; // PySet_MINSIZE; <=4 entries never resize
constexpr uint64_t kSetMask = kSetTableSize - 1;
constexpr int32_t kSetPerturbShift = 5; // CPython PERTURB_SHIFT

// hash() of a 2-tuple of small non-negative ints under CPython >= 3.8, whose
// tuple hash is xxHash-based. hash(int) is the int itself in this range.
inline uint64_t tuple2_hash(int32_t a, int32_t b) {
  constexpr uint64_t kPrime1 = 11400714785074694791ULL;
  constexpr uint64_t kPrime2 = 14029467366897019727ULL;
  constexpr uint64_t kPrime5 = 2870177450012600261ULL;
  uint64_t acc = kPrime5;
  for (const int32_t lane : {a, b}) {
    acc += static_cast<uint64_t>(lane) * kPrime2;
    acc = (acc << 31) | (acc >> 33);
    acc *= kPrime1;
  }
  acc += 2ULL ^ (kPrime5 ^ 3527539ULL);
  return acc == ~uint64_t{0} ? 1546275796ULL : acc;
}

struct GridCandidate {
  int32_t nph;
  int32_t npw;
};

// itertools.product([fh, ch], [fw, cw]) deduplicated and emitted in the order
// CPython walks the resulting set's table. At most 4 entries keep that table at
// PySet_MINSIZE, where `i + LINEAR_PROBES <= mask` never holds, so insertion is
// the plain perturbed probe below with no linear scan. Returns the count.
inline int32_t grid_candidates_in_set_order(
    int32_t fh,
    int32_t ch,
    int32_t fw,
    int32_t cw,
    GridCandidate* out) {
  GridCandidate table[kSetTableSize];
  bool filled[kSetTableSize] = {};
  for (const int32_t nph : {fh, ch}) {
    for (const int32_t npw : {fw, cw}) {
      uint64_t perturb = tuple2_hash(nph, npw);
      uint64_t i = perturb & kSetMask;
      while (filled[i] && (table[i].nph != nph || table[i].npw != npw)) {
        perturb >>= kSetPerturbShift;
        i = (i * 5 + 1 + perturb) & kSetMask;
      }
      table[i] = GridCandidate{nph, npw};
      filled[i] = true;
    }
  }
  int32_t n = 0;
  for (int32_t i = 0; i < kSetTableSize; ++i) {
    if (filled[i]) {
      out[n++] = table[i];
    }
  }
  return n;
}

// Python's round() for a non-negative double: halfway cases go to even.
inline int32_t round_half_even(double x) {
  const double lower = std::floor(x);
  int64_t n = static_cast<int64_t>(lower);
  const double frac = x - lower;
  if (frac > 0.5 || (frac == 0.5 && (n % 2) != 0)) {
    ++n;
  }
  return static_cast<int32_t>(n);
}

} // namespace detail

// Pick the image grid under the token cap, including the tie-break above.
inline void compute_grid_size(
    int32_t img_w,
    int32_t img_h,
    int32_t& target_h,
    int32_t& target_w,
    int64_t max_tokens = kMaxImageTokens) {
  const double ph = static_cast<double>(kCell);
  double i_nph = img_h / ph;
  double i_npw = img_w / ph;
  const double ratio = (i_nph > 0) ? (i_npw / i_nph) : 1.0;
  if (i_nph * i_npw > static_cast<double>(max_tokens)) {
    i_nph = std::sqrt(static_cast<double>(max_tokens) / ratio);
    i_npw = i_nph * ratio;
  }

  // Candidates are built UNCLAMPED (floor may be 0) and filtered afterwards,
  // as Python does. Clamping to >=1 up front would change which tuples enter
  // the set, and so both the candidate list and the order it comes back in.
  detail::GridCandidate candidates[detail::kSetTableSize];
  const int32_t n_candidates = detail::grid_candidates_in_set_order(
      static_cast<int32_t>(std::floor(i_nph)),
      static_cast<int32_t>(std::ceil(i_nph)),
      static_cast<int32_t>(std::floor(i_npw)),
      static_cast<int32_t>(std::ceil(i_npw)),
      candidates);

  const double aspect = static_cast<double>(img_h) / static_cast<double>(img_w);
  int32_t best_h = 0;
  int32_t best_w = 0;
  double best_cost = 0.0;
  bool found = false;
  for (int32_t i = 0; i < n_candidates; ++i) {
    const int32_t nph = candidates[i].nph;
    const int32_t npw = candidates[i].npw;
    if (nph < 1 || npw < 1 || static_cast<int64_t>(nph) * npw > max_tokens) {
      continue;
    }
    const double cost =
        std::abs(static_cast<double>(nph) / static_cast<double>(npw) - aspect);
    // Strict <, walking the set order: exactly what Python's min() does.
    if (!found || cost < best_cost) {
      found = true;
      best_cost = cost;
      best_h = nph;
      best_w = npw;
    }
  }
  if (!found) {
    best_h = std::max(1, detail::round_half_even(i_nph));
    best_w = std::max(1, detail::round_half_even(i_npw));
  }
  target_h = best_h * kCell;
  target_w = best_w * kCell;
}

inline std::vector<uint8_t> resize_rgb(
    const uint8_t* src,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h) {
  std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * dst_h * 3);
  stbir_resize_uint8_generic(
      src,
      src_w,
      src_h,
      0,
      dst.data(),
      dst_w,
      dst_h,
      0,
      3,
      STBIR_ALPHA_CHANNEL_NONE,
      0,
      STBIR_EDGE_CLAMP,
      STBIR_FILTER_CATMULLROM,
      STBIR_COLORSPACE_SRGB,
      nullptr);
  return dst;
}
// PIL-compatible Lanczos resize.
//
// Downscale-aware support, fixed-point coefficients and accumulation, and a
// clipped uint8 intermediate reproduce Pillow's Image.LANCZOS output.

static constexpr int32_t kLanczosA = 3;
// Pillow's PRECISION_BITS = 32 - 8 - 2.
static constexpr int32_t kResamplePrecisionBits = 22;

inline double lanczos_kernel(double x) {
  if (x == 0.0) {
    return 1.0;
  }
  if (x < 0.0) {
    x = -x;
  }
  if (x >= static_cast<double>(kLanczosA)) {
    return 0.0;
  }
  const double px = M_PI * x;
  return (std::sin(px) / px) * (std::sin(px / kLanczosA) / (px / kLanczosA));
}

// Per-output-pixel taps for one axis, in Pillow's fixed-point form.
struct ResampleAxis {
  int32_t ksize = 0;
  std::vector<int32_t> bounds; // 2 per output pixel: {start, count}
  std::vector<int32_t> kk; // out_size * ksize
};

inline ResampleAxis precompute_resample_axis(
    int32_t in_size,
    int32_t out_size) {
  const double scale = static_cast<double>(in_size) / out_size;
  const double filterscale = scale < 1.0 ? 1.0 : scale;
  const double support = kLanczosA * filterscale;

  ResampleAxis ax;
  ax.ksize = static_cast<int32_t>(std::ceil(support)) * 2 + 1;
  ax.bounds.assign(static_cast<size_t>(out_size) * 2, 0);
  ax.kk.assign(static_cast<size_t>(out_size) * ax.ksize, 0);

  std::vector<double> k(static_cast<size_t>(ax.ksize), 0.0);
  for (int32_t xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    int32_t xmin = static_cast<int32_t>(center - support + 0.5);
    if (xmin < 0) {
      xmin = 0;
    }
    int32_t xmax = static_cast<int32_t>(center + support + 0.5);
    if (xmax > in_size) {
      xmax = in_size;
    }
    xmax -= xmin;

    double ww = 0.0;
    for (int32_t x = 0; x < xmax; ++x) {
      const double w = lanczos_kernel((x + xmin - center + 0.5) / filterscale);
      k[static_cast<size_t>(x)] = w;
      ww += w;
    }
    if (ww != 0.0) {
      for (int32_t x = 0; x < xmax; ++x) {
        k[static_cast<size_t>(x)] /= ww;
      }
    }
    int32_t* out_k = ax.kk.data() + static_cast<size_t>(xx) * ax.ksize;
    for (int32_t x = 0; x < xmax; ++x) {
      const double v =
          k[static_cast<size_t>(x)] * (1 << kResamplePrecisionBits);
      out_k[x] = static_cast<int32_t>(v < 0 ? v - 0.5 : v + 0.5);
    }
    ax.bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
    ax.bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
  }
  return ax;
}

inline uint8_t resample_clip8(int64_t acc) {
  const int64_t v = acc >> kResamplePrecisionBits;
  if (v <= 0) {
    return 0;
  }
  if (v >= 255) {
    return 255;
  }
  return static_cast<uint8_t>(v);
}

// One separable pass along x over interleaved RGB24. The vertical pass reuses
// this by being handed a transposed view, exactly as Pillow runs two 1-D
// passes.
inline void resample_pass_x(
    const uint8_t* src,
    int32_t src_w,
    int32_t rows,
    const ResampleAxis& ax,
    int32_t out_w,
    uint8_t* dst) {
  constexpr int64_t kInit = int64_t{1} << (kResamplePrecisionBits - 1);
  (void)src_w;
  for (int32_t y = 0; y < rows; ++y) {
    const uint8_t* srow = src + static_cast<size_t>(y) * src_w * 3;
    uint8_t* drow = dst + static_cast<size_t>(y) * out_w * 3;
    for (int32_t xx = 0; xx < out_w; ++xx) {
      const int32_t xmin = ax.bounds[static_cast<size_t>(xx) * 2 + 0];
      const int32_t xcnt = ax.bounds[static_cast<size_t>(xx) * 2 + 1];
      const int32_t* k = ax.kk.data() + static_cast<size_t>(xx) * ax.ksize;
      int64_t s0 = kInit, s1 = kInit, s2 = kInit;
      for (int32_t x = 0; x < xcnt; ++x) {
        const uint8_t* p = srow + static_cast<size_t>(xmin + x) * 3;
        s0 += static_cast<int64_t>(p[0]) * k[x];
        s1 += static_cast<int64_t>(p[1]) * k[x];
        s2 += static_cast<int64_t>(p[2]) * k[x];
      }
      drow[xx * 3 + 0] = resample_clip8(s0);
      drow[xx * 3 + 1] = resample_clip8(s1);
      drow[xx * 3 + 2] = resample_clip8(s2);
    }
  }
}

inline void
transpose_rgb(const uint8_t* src, int32_t w, int32_t h, uint8_t* dst) {
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {
      const uint8_t* p = src + (static_cast<size_t>(y) * w + x) * 3;
      uint8_t* q = dst + (static_cast<size_t>(x) * h + y) * 3;
      q[0] = p[0];
      q[1] = p[1];
      q[2] = p[2];
    }
  }
}

// Resize interleaved RGB24 with Pillow's LANCZOS semantics.
inline std::vector<uint8_t> resize_rgb_lanczos(
    const uint8_t* src,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h) {
  const ResampleAxis hx = precompute_resample_axis(src_w, dst_w);
  std::vector<uint8_t> horiz(static_cast<size_t>(dst_w) * src_h * 3);
  resample_pass_x(src, src_w, src_h, hx, dst_w, horiz.data());

  // Vertical pass as a horizontal pass over the transpose, then transpose back.
  std::vector<uint8_t> t(static_cast<size_t>(src_h) * dst_w * 3);
  transpose_rgb(horiz.data(), dst_w, src_h, t.data());
  const ResampleAxis vx = precompute_resample_axis(src_h, dst_h);
  std::vector<uint8_t> tv(static_cast<size_t>(dst_h) * dst_w * 3);
  resample_pass_x(t.data(), src_h, dst_w, vx, dst_h, tv.data());

  std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * dst_h * 3);
  transpose_rgb(tv.data(), dst_h, dst_w, dst.data());
  return dst;
}

namespace detail {

// Bilinear grid_sample of the 32x32 pos table, matching the eager reference's
// meshgrid(ys, xs, indexing="xy") and align_corners=False semantics. The
// resulting tokens are flattened over [grid_w, grid_h], with grid_h fastest.
// ``table`` is [pos_tokens(=1024), latent] row-major (float).
inline void interpolate_pos_emb(
    const std::vector<float>& table,
    int32_t grid_h,
    int32_t grid_w,
    std::vector<float>& out /* [grid_h*grid_w, latent] */) {
  const int32_t gh = kPosGrid, gw = kPosGrid;
  out.assign(static_cast<size_t>(grid_h) * grid_w * kLatent, 0.0f);
  const double inv_h = 1.0 / grid_h;
  const double inv_w = 1.0 / grid_w;
  for (int32_t oy = 0; oy < grid_w; ++oy) {
    // meshgrid(ys, xs, indexing="xy") puts xs in grid_sample's y slot.
    const double ny = (grid_w == 1)
        ? 0.0
        : (-1.0 + inv_w + oy * (2.0 - 2.0 * inv_w) / (grid_w - 1));
    const double sy = (ny + 1.0) * 0.5 * gh - 0.5;
    const int32_t y0 = static_cast<int32_t>(std::floor(sy));
    const double wy = sy - y0;
    for (int32_t ox = 0; ox < grid_h; ++ox) {
      // meshgrid puts ys in grid_sample's x slot.
      const double nx = (grid_h == 1)
          ? 0.0
          : (-1.0 + inv_h + ox * (2.0 - 2.0 * inv_h) / (grid_h - 1));
      const double sx = (nx + 1.0) * 0.5 * gw - 0.5;
      const int32_t x0 = static_cast<int32_t>(std::floor(sx));
      const double wx = sx - x0;

      const int32_t y1 = y0 + 1;
      const int32_t x1 = x0 + 1;
      const auto in_bounds = [gh, gw](int32_t y, int32_t x) {
        return y >= 0 && y < gh && x >= 0 && x < gw;
      };
      // F.grid_sample defaults to zero padding. This matters whenever a
      // runtime grid dimension exceeds the 32x32 source table: the first and
      // last sample centers then straddle the source boundary.
      const float* p00 = in_bounds(y0, x0)
          ? table.data() + (static_cast<size_t>(y0) * gw + x0) * kLatent
          : nullptr;
      const float* p01 = in_bounds(y0, x1)
          ? table.data() + (static_cast<size_t>(y0) * gw + x1) * kLatent
          : nullptr;
      const float* p10 = in_bounds(y1, x0)
          ? table.data() + (static_cast<size_t>(y1) * gw + x0) * kLatent
          : nullptr;
      const float* p11 = in_bounds(y1, x1)
          ? table.data() + (static_cast<size_t>(y1) * gw + x1) * kLatent
          : nullptr;
      float* dst =
          out.data() + (static_cast<size_t>(oy) * grid_h + ox) * kLatent;
      for (int32_t c = 0; c < kLatent; ++c) {
        const double top = (p00 == nullptr ? 0.0 : p00[c]) * (1.0 - wx) +
            (p01 == nullptr ? 0.0 : p01[c]) * wx;
        const double bot = (p10 == nullptr ? 0.0 : p10[c]) * (1.0 - wx) +
            (p11 == nullptr ? 0.0 : p11[c]) * wx;
        dst[c] = static_cast<float>(top * (1.0 - wy) + bot * wy);
      }
    }
  }
}

// 2D-RoPE cos/sin, mirroring vision_precompute.make_2d_rope.
inline void make_2d_rope(
    int32_t grid_h,
    int32_t grid_w,
    std::vector<float>& cos_out /* [P, 48] */,
    std::vector<float>& sin_out) {
  const int32_t half = kHeadDim / 2; // 48
  const int32_t quarter = half / 2; // 24
  const double theta = 10000.0;
  std::vector<double> inv_freq(quarter);
  for (int32_t i = 0; i < quarter; ++i) {
    inv_freq[i] = 1.0 / std::pow(theta, static_cast<double>(2 * i) / half);
  }
  const int64_t p = static_cast<int64_t>(grid_h) * grid_w;
  cos_out.assign(static_cast<size_t>(p) * half, 0.0f);
  sin_out.assign(static_cast<size_t>(p) * half, 0.0f);
  for (int32_t r = 0; r < grid_h; ++r) {
    for (int32_t c = 0; c < grid_w; ++c) {
      const int64_t tok = static_cast<int64_t>(r) * grid_w + c;
      float* crow = cos_out.data() + tok * half;
      float* srow = sin_out.data() + tok * half;
      // freq = concat(freq_w, freq_h); each is quarter-wide.
      for (int32_t i = 0; i < quarter; ++i) {
        const double fw = (c + 1) * inv_freq[i];
        crow[i] = static_cast<float>(std::cos(fw));
        srow[i] = static_cast<float>(std::sin(fw));
      }
      for (int32_t i = 0; i < quarter; ++i) {
        const double fh = (r + 1) * inv_freq[i];
        crow[quarter + i] = static_cast<float>(std::cos(fh));
        srow[quarter + i] = static_cast<float>(std::sin(fh));
      }
    }
  }
}

// Sparse tiling permutation + per-group lengths, mirroring
// vision_precompute.sparse_perm_and_slens (32x32 window tiling).
inline void sparse_perm_and_slens(
    int32_t grid_h,
    int32_t grid_w,
    std::vector<int64_t>& perm,
    std::vector<int32_t>& slens) {
  const int32_t gh = kPosGrid, gw = kPosGrid;
  const int32_t pad_h = ((grid_h + gh - 1) / gh) * gh;
  const int32_t pad_w = ((grid_w + gw - 1) / gw) * gw;
  perm.clear();
  slens.clear();
  for (int32_t bh = 0; bh < pad_h / gh; ++bh) {
    for (int32_t bw = 0; bw < pad_w / gw; ++bw) {
      int32_t count = 0;
      for (int32_t iy = 0; iy < gh; ++iy) {
        for (int32_t ix = 0; ix < gw; ++ix) {
          const int32_t y = bh * gh + iy;
          const int32_t x = bw * gw + ix;
          if (y < grid_h && x < grid_w) {
            perm.push_back(static_cast<int64_t>(y) * grid_w + x);
            ++count;
          }
        }
      }
      if (count > 0) {
        slens.push_back(count);
      }
    }
  }
}

// Pixel-shuffle permutation, mirroring vision_precompute.pixel_shuffle_perm.
inline void
pixel_shuffle_perm(int32_t grid_h, int32_t grid_w, std::vector<int64_t>& perm) {
  const int32_t f = kDownsample;
  perm.assign(static_cast<size_t>(grid_h) * grid_w, 0);
  int64_t idx = 0;
  for (int32_t oh = 0; oh < grid_h / f; ++oh) {
    for (int32_t ow = 0; ow < grid_w / f; ++ow) {
      for (int32_t iy = 0; iy < f; ++iy) {
        for (int32_t ix = 0; ix < f; ++ix) {
          const int32_t y = oh * f + iy;
          const int32_t x = ow * f + ix;
          perm[idx++] = static_cast<int64_t>(y) * grid_w + x;
        }
      }
    }
  }
}

} // namespace detail

// Preprocess a decoded RGB image (HWC uint8) into the 9 vision_encoder inputs.
// ``pos_table`` is the [pos_tokens(=1024), latent(=1536)] float positional
// table loaded from pos_embed.bin (written by export.py).
inline VisionInputs preprocess_image(
    const uint8_t* rgb,
    int32_t width,
    int32_t height,
    const std::vector<float>& pos_table,
    int64_t max_image_tokens = kMaxImageTokens) {
  namespace ext = ::executorch::extension;
  using SizesType = ::executorch::aten::SizesType;

  int32_t target_h = 0, target_w = 0;
  compute_grid_size(width, height, target_h, target_w, max_image_tokens);

  std::vector<uint8_t> resized;
  const uint8_t* img = rgb;
  if (target_h != height || target_w != width) {
    resized = resize_rgb_lanczos(rgb, width, height, target_w, target_h);
    img = resized.data();
    height = target_h;
    width = target_w;
  }

  const int32_t grid_h = height / kPatchSize;
  const int32_t grid_w = width / kPatchSize;
  const int64_t P = static_cast<int64_t>(grid_h) * grid_w;
  const int64_t n_out = (grid_h / kDownsample) * (grid_w / kDownsample);
  const int32_t half = kHeadDim / 2;

  // --- patches [1, P, 1176] float32 (normalize to [-1,1], replicate temporal).
  auto patches = ext::zeros(
      {1, static_cast<SizesType>(P), kPatchDim},
      executorch::aten::ScalarType::Float);
  float* pv = patches->mutable_data_ptr<float>();
  const int32_t hp = kPatchSize * kPatchSize; // 196
  for (int32_t py = 0; py < grid_h; ++py) {
    for (int32_t px = 0; px < grid_w; ++px) {
      const int64_t tok = static_cast<int64_t>(py) * grid_w + px;
      float* prow = pv + tok * kPatchDim;
      // Layout mirrors eager: permute to [c, ps, ps] then temporal replicate.
      // eager patches reshape is [pt, 3, ps, ps] flattened -> here we fill the
      // first (3*ps*ps) block and copy it to the 2nd temporal block.
      for (int32_t c = 0; c < 3; ++c) {
        for (int32_t dy = 0; dy < kPatchSize; ++dy) {
          for (int32_t dx = 0; dx < kPatchSize; ++dx) {
            const int32_t iy = py * kPatchSize + dy;
            const int32_t ix = px * kPatchSize + dx;
            const int32_t src = (iy * width + ix) * 3 + c;
            const float val = (img[src] / 255.0f - 0.5f) / 0.5f;
            prow[(c * hp) + dy * kPatchSize + dx] = val;
          }
        }
      }
      // Temporal replicate: 2nd half == 1st half (3*ps*ps floats).
      const int32_t block = 3 * hp;
      for (int32_t i = 0; i < block; ++i) {
        prow[block + i] = prow[i];
      }
    }
  }

  // --- positional embedding [1, P, 1536] bf16 (host grid_sample interp).
  std::vector<float> pos_f;
  detail::interpolate_pos_emb(pos_table, grid_h, grid_w, pos_f);
  auto pos_emb = ext::zeros(
      {1, static_cast<SizesType>(P), kLatent},
      executorch::aten::ScalarType::BFloat16);
  auto* pe = pos_emb->mutable_data_ptr<executorch::aten::BFloat16>();
  for (int64_t i = 0; i < P * kLatent; ++i) {
    pe[i] = static_cast<executorch::aten::BFloat16>(pos_f[i]);
  }

  // --- 2D-RoPE cos/sin [P, 48] float32.
  std::vector<float> cos_f, sin_f;
  detail::make_2d_rope(grid_h, grid_w, cos_f, sin_f);
  auto cos_t = ext::zeros(
      {static_cast<SizesType>(P), half}, executorch::aten::ScalarType::Float);
  auto sin_t = ext::zeros(
      {static_cast<SizesType>(P), half}, executorch::aten::ScalarType::Float);
  std::memcpy(
      cos_t->mutable_data_ptr<float>(),
      cos_f.data(),
      cos_f.size() * sizeof(float));
  std::memcpy(
      sin_t->mutable_data_ptr<float>(),
      sin_f.data(),
      sin_f.size() * sizeof(float));

  // --- sparse perm / inv perm [P] int64 + block-diagonal masks [1,1,P,P] bool.
  std::vector<int64_t> perm;
  std::vector<int32_t> slens;
  detail::sparse_perm_and_slens(grid_h, grid_w, perm, slens);
  std::vector<int64_t> inv(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    inv[perm[i]] = static_cast<int64_t>(i);
  }
  auto perm_t = ext::zeros(
      {static_cast<SizesType>(P)}, executorch::aten::ScalarType::Long);
  auto inv_t = ext::zeros(
      {static_cast<SizesType>(P)}, executorch::aten::ScalarType::Long);
  std::memcpy(
      perm_t->mutable_data_ptr<int64_t>(), perm.data(), P * sizeof(int64_t));
  std::memcpy(
      inv_t->mutable_data_ptr<int64_t>(), inv.data(), P * sizeof(int64_t));

  // global_mask: all-True (single image); sparse_mask: block-diagonal by slens
  // in PERMUTED order.
  auto global_mask = ext::full(
      {1, 1, static_cast<SizesType>(P), static_cast<SizesType>(P)},
      true,
      executorch::aten::ScalarType::Bool);
  auto sparse_mask = ext::zeros(
      {1, 1, static_cast<SizesType>(P), static_cast<SizesType>(P)},
      executorch::aten::ScalarType::Bool);
  auto* sm = sparse_mask->mutable_data_ptr<bool>();
  int64_t offset = 0;
  for (int32_t s : slens) {
    for (int64_t r = 0; r < s; ++r) {
      for (int64_t c = 0; c < s; ++c) {
        sm[(offset + r) * P + (offset + c)] = true;
      }
    }
    offset += s;
  }

  // --- pixel-shuffle perm [P] int64.
  std::vector<int64_t> px_perm;
  detail::pixel_shuffle_perm(grid_h, grid_w, px_perm);
  auto pixel_t = ext::zeros(
      {static_cast<SizesType>(P)}, executorch::aten::ScalarType::Long);
  std::memcpy(
      pixel_t->mutable_data_ptr<int64_t>(),
      px_perm.data(),
      P * sizeof(int64_t));

  return VisionInputs{
      std::move(patches),
      std::move(pos_emb),
      std::move(cos_t),
      std::move(sin_t),
      std::move(perm_t),
      std::move(inv_t),
      std::move(global_mask),
      std::move(sparse_mask),
      std::move(pixel_t),
      P,
      n_out,
  };
}

// Read the pos_embed.bin table (float32 [pos_tokens, latent]) written by
// export.py. Returns the flat table; throws on size mismatch.
inline std::vector<float> load_pos_embed_table(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open pos_embed.bin: " + path);
  }
  f.seekg(0, std::ios::end);
  const int64_t nbytes = f.tellg();
  f.seekg(0, std::ios::beg);
  const int64_t n = nbytes / static_cast<int64_t>(sizeof(float));
  const int64_t expected = static_cast<int64_t>(kPosGrid) * kPosGrid * kLatent;
  if (n != expected) {
    throw std::runtime_error(
        "pos_embed.bin size mismatch: got " + std::to_string(n) +
        " floats, expected " + std::to_string(expected));
  }
  std::vector<float> table(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(table.data()), nbytes);
  return table;
}

} // namespace muse_glimmer_vision
} // namespace examples
} // namespace executorch
