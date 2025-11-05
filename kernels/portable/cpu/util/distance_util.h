/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

bool check_pdist_args(const Tensor& in, double p, const Tensor& out);

void get_pdist_out_target_size(
    const Tensor& in,
    Tensor::SizesType* out_sizes,
    size_t* out_ndim);

template <typename CTYPE, typename Norm>
void pdist(const Tensor& in, Tensor& out, double p) {
  const CTYPE* in_data = in.const_data_ptr<CTYPE>();
  CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

  size_t n = in.size(0);
  size_t m = in.size(1);

  size_t out_ix = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      const CTYPE* row_i = in_data + i * m;
      const CTYPE* row_j = in_data + j * m;
      CTYPE agg = 0;
      for (size_t k = 0; k < m; ++k) {
        CTYPE diff = std::abs(row_i[k] - row_j[k]);
        agg = Norm::reduce(agg, Norm::map(diff, p));
      }
      out_data[out_ix++] = Norm::finish(agg, p);
    }
  }
}

template <typename CTYPE>
struct L0 {
  static inline CTYPE map(const CTYPE& diff, const CTYPE&) {
    return diff == 0 ? 0 : 1;
  }
  static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up) {
    return agg + up;
  }
  static inline CTYPE finish(const CTYPE& agg, const CTYPE&) {
    return agg;
  }
};

template <typename CTYPE>
struct L1 {
  static inline CTYPE map(const CTYPE& diff, const CTYPE&) {
    return diff;
  }
  static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up) {
    return agg + up;
  }
  static inline CTYPE finish(const CTYPE& agg, const CTYPE&) {
    return agg;
  }
};

template <typename CTYPE>
struct L2 {
  static inline CTYPE map(const CTYPE& diff, const CTYPE&) {
    return diff * diff;
  }
  static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up) {
    return agg + up;
  }
  static inline CTYPE finish(const CTYPE& agg, const CTYPE&) {
    return std::sqrt(agg);
  }
};

template <typename CTYPE>
struct Lp {
  static inline CTYPE map(const CTYPE& diff, const CTYPE& p) {
    return std::pow(diff, p);
  }
  static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up) {
    return agg + up;
  }
  static inline CTYPE finish(const CTYPE& agg, const CTYPE& p) {
    return std::pow(agg, 1.0 / p);
  }
};

template <typename CTYPE>
struct Linf {
  static inline CTYPE map(const CTYPE& diff, const CTYPE&) {
    return diff;
  }
  static inline CTYPE reduce(const CTYPE& agg, const CTYPE& up) {
    return std::max(agg, up);
  }
  static inline CTYPE finish(const CTYPE& agg, const CTYPE&) {
    return agg;
  }
};

template <typename CTYPE>
void pdist(const Tensor& in, Tensor& out, double p) {
  if (p == 0.0) {
    pdist<CTYPE, L0<CTYPE>>(in, out, p);
  } else if (p == 1.0) {
    pdist<CTYPE, L1<CTYPE>>(in, out, p);
  } else if (p == 2.0) {
    pdist<CTYPE, L2<CTYPE>>(in, out, p);
  } else if (p == INFINITY) {
    pdist<CTYPE, Linf<CTYPE>>(in, out, p);
  } else {
    pdist<CTYPE, Lp<CTYPE>>(in, out, p);
  }
}

bool check_cdist_args(
    const Tensor& x1,
    const Tensor& x2,
    double p,
    optional<int64_t> compute_mode,
    const Tensor& out);

} // namespace executor
} // namespace torch
