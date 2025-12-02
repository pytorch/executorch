/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_linalg_svd.h>

#include <algorithm>
#include <cmath>
#include <tuple>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

const float EPSILON = 1e-10;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;

// A simple 3x3 matrix struct.
struct Matrix3x3 {
  float m[3][3];
};

// Returns the 3x3 identity matrix.
Matrix3x3 identityMatrix() {
  Matrix3x3 I{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      I.m[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
  return I;
}

// Transposes matrix A.
Matrix3x3 transpose(const Matrix3x3& A) {
  Matrix3x3 At{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      At.m[i][j] = A.m[j][i];
    }
  }
  return At;
}

// Multiplies matrices A and B.
Matrix3x3 multiply(const Matrix3x3& A, const Matrix3x3& B) {
  Matrix3x3 C{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      C.m[i][j] = 0.0;
      for (int k = 0; k < 3; k++) {
        C.m[i][j] += A.m[i][k] * B.m[k][j];
      }
    }
  }
  return C;
}

// Jacobi method to compute the eigen-decomposition of a symmetric 3x3 matrix A.
// It outputs the eigenvalues (in 'diag') and the eigenvectors as columns in V.
void jacobiEigenDecomposition(const Matrix3x3& A, float diag[3], Matrix3x3& V) {
  Matrix3x3 D = A; // Make a copy; D will be transformed into a diagonal matrix.
  V = identityMatrix();

  // Iterate until convergence (or max iterations)
  for (int iter = 0; iter < 100; iter++) {
    // Find the largest off-diagonal element in D.
    int p = 0, q = 1;
    float maxOff = std::fabs(D.m[0][1]);
    if (std::fabs(D.m[0][2]) > maxOff) {
      maxOff = std::fabs(D.m[0][2]);
      p = 0;
      q = 2;
    }
    if (std::fabs(D.m[1][2]) > maxOff) {
      maxOff = std::fabs(D.m[1][2]);
      p = 1;
      q = 2;
    }

    if (maxOff < EPSILON) {
      break;
    }

    // Compute the Jacobi rotation angle.
    float theta = 0.0;
    if (std::fabs(D.m[p][p] - D.m[q][q]) < EPSILON) {
      theta = M_PI / 4.0;
    } else {
      theta = 0.5 * std::atan2(2 * D.m[p][q], D.m[q][q] - D.m[p][p]);
    }

    float c = std::cos(theta);
    float s = std::sin(theta);

    // Update the diagonal elements.
    float D_pp = c * c * D.m[p][p] - 2 * s * c * D.m[p][q] + s * s * D.m[q][q];
    float D_qq = s * s * D.m[p][p] + 2 * s * c * D.m[p][q] + c * c * D.m[q][q];
    D.m[p][p] = D_pp;
    D.m[q][q] = D_qq;
    D.m[p][q] = D.m[q][p] = 0.0;

    // Update the remaining elements.
    for (int j = 0; j < 3; j++) {
      if (j != p && j != q) {
        float D_pj = c * D.m[p][j] - s * D.m[q][j];
        float D_qj = s * D.m[p][j] + c * D.m[q][j];
        D.m[p][j] = D.m[j][p] = D_pj;
        D.m[q][j] = D.m[j][q] = D_qj;
      }
    }

    // Update the eigenvector matrix V.
    for (int i = 0; i < 3; i++) {
      float V_ip = c * V.m[i][p] - s * V.m[i][q];
      float V_iq = s * V.m[i][p] + c * V.m[i][q];
      V.m[i][p] = V_ip;
      V.m[i][q] = V_iq;
    }
  }

  diag[0] = D.m[0][0];
  diag[1] = D.m[1][1];
  diag[2] = D.m[2][2];
}

// Sorts the eigenvalues (and the corresponding eigenvectors in V) in descending
// order.
void sortEigenDecomposition(float eigenvalues[3], Matrix3x3& V) {
  int indices[3] = {0, 1, 2};
  std::sort(indices, indices + 3, [&](int a, int b) {
    return eigenvalues[a] > eigenvalues[b];
  });

  float sortedEigenvalues[3];
  Matrix3x3 sortedV{};
  for (int i = 0; i < 3; i++) {
    sortedEigenvalues[i] = eigenvalues[indices[i]];
    for (int j = 0; j < 3; j++) {
      sortedV.m[j][i] = V.m[j][indices[i]];
    }
  }
  for (int i = 0; i < 3; i++) {
    eigenvalues[i] = sortedEigenvalues[i];
    for (int j = 0; j < 3; j++) {
      V.m[j][i] = sortedV.m[j][i];
    }
  }
}

// Computes the cross product of two 3D vectors.
void crossProduct(const float a[3], const float b[3], float result[3]) {
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}

// Normalizes a 3D vector.
void normalize(float v[3]) {
  float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (norm > EPSILON) {
    v[0] /= norm;
    v[1] /= norm;
    v[2] /= norm;
  }
}

// Computes the singular value decomposition of A such that A = U * S * Vt.
// U and Vt are orthogonal matrices and S is a diagonal matrix with singular
// values.
std::tuple<Matrix3x3, Matrix3x3, Matrix3x3> svd(const Matrix3x3& A) {
  // Compute A^T * A (which is symmetric).
  Matrix3x3 At = transpose(A);
  Matrix3x3 ATA = multiply(At, A);

  // Compute the eigen-decomposition of ATA.
  float eigenvalues[3];
  Matrix3x3 V{};
  jacobiEigenDecomposition(ATA, eigenvalues, V);
  sortEigenDecomposition(eigenvalues, V);

  // The singular values are the square roots of the eigenvalues.
  float sigma[3];
  for (int i = 0; i < 3; i++) {
    sigma[i] = (eigenvalues[i] > 0.0) ? std::sqrt(eigenvalues[i]) : 0.0;
  }

  // Compute U = A * V * S^{-1}.
  Matrix3x3 U{};
  for (int i = 0; i < 3; i++) {
    float av[3] = {0, 0, 0};
    // Multiply A by the i-th eigenvector (column of V).
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        av[r] += A.m[r][c] * V.m[c][i];
      }
    }
    if (sigma[i] > EPSILON) {
      for (int r = 0; r < 3; r++) {
        U.m[r][i] = av[r] / sigma[i];
      }
    } else {
      // If sigma[i] is nearly zero, choose a vector orthogonal to the previous
      // ones.
      float vec[3] = {0, 0, 0};
      if (i == 1) {
        float u0[3] = {U.m[0][0], U.m[1][0], U.m[2][0]};
        float tmp[3] = {1, 0, 0};
        float dot = u0[0] * tmp[0] + u0[1] * tmp[1] + u0[2] * tmp[2];
        if (std::fabs(dot) > 0.9) {
          tmp[0] = 0;
          tmp[1] = 1;
          tmp[2] = 0;
        }
        crossProduct(u0, tmp, vec);
      } else if (i == 2) {
        float u0[3] = {U.m[0][0], U.m[1][0], U.m[2][0]};
        float u1[3] = {U.m[0][1], U.m[1][1], U.m[2][1]};
        crossProduct(u0, u1, vec);
      }
      normalize(vec);
      for (int r = 0; r < 3; r++) {
        U.m[r][i] = vec[r];
      }
    }
  }

  // Construct the diagonal S matrix.
  Matrix3x3 S{};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      S.m[i][j] = (i == j) ? sigma[i] : 0.0;
    }
  }

  // Vt is the transpose of V.
  Matrix3x3 Vt = transpose(V);

  return std::make_tuple(U, S, Vt);
}
} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> linalg_svd_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& A,
    bool full_matrices,
    bool compute_uv,
    ::executorch::aten::optional<::executorch::aten::string_view> driver,
    Tensor& U,
    Tensor& S,
    Tensor& Vh) {
  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(U, S, Vh);

  ET_KERNEL_CHECK_MSG(
      ctx,
      A.scalar_type() == ScalarType::Float,
      InvalidArgument,
      ret_val,
      "input.dtype(): %s must be %s",
      ::torch::executor::toString(A.scalar_type()),
      ::torch::executor::toString(ScalarType::Float));

  ET_KERNEL_CHECK_MSG(
      ctx, A.numel() > 0, InvalidArgument, ret_val, "input.size() must be > 0");

  ET_KERNEL_CHECK_MSG(
      ctx,
      A.numel() % 9 == 0,
      InvalidArgument,
      ret_val,
      "SVD of only 3x3 matrix is supported! Expected the input to have (batch_size x 9) number of elements, but received %d elements instead",
      int(A.numel()));

  int batch_size = A.numel() / 9;

  ET_KERNEL_CHECK_MSG(
      ctx,
      U.numel() / 9 == batch_size,
      InvalidArgument,
      ret_val,
      "Output tensor U must have the same batch_size as input: %d, but got: %d instead",
      batch_size,
      int(U.numel() / 9));

  ET_KERNEL_CHECK_MSG(
      ctx,
      S.numel() / 3 == batch_size,
      InvalidArgument,
      ret_val,
      "Output tensor S must have the same batch_size as input: %d, but got: %d instead",
      batch_size,
      int(S.numel() / 3));

  ET_KERNEL_CHECK_MSG(
      ctx,
      Vh.numel() / 9 == batch_size,
      InvalidArgument,
      ret_val,
      "Output tensor Vh must have the same batch_size as input: %d, but got: %d instead",
      batch_size,
      int(Vh.numel() / 9));

  const float* A_data = A.const_data_ptr<float>();
  float* U_data = U.mutable_data_ptr<float>();
  float* S_data = S.mutable_data_ptr<float>();
  float* Vh_data = Vh.mutable_data_ptr<float>();

  for (int i = 0; i < batch_size; i++) {
    int offset = i * 9;
    Matrix3x3 A_mat = {{
        {A_data[offset + 0], A_data[offset + 1], A_data[offset + 2]},
        {A_data[offset + 3], A_data[offset + 4], A_data[offset + 5]},
        {A_data[offset + 6], A_data[offset + 7], A_data[offset + 8]},
    }};

    Matrix3x3 U_mat{}, S_mat{}, Vh_mat{};
    std::tie(U_mat, S_mat, Vh_mat) = svd(A_mat);

    U_data[offset + 0] = U_mat.m[0][0];
    U_data[offset + 1] = U_mat.m[0][1];
    U_data[offset + 2] = U_mat.m[0][2];
    U_data[offset + 3] = U_mat.m[1][0];
    U_data[offset + 4] = U_mat.m[1][1];
    U_data[offset + 5] = U_mat.m[1][2];
    U_data[offset + 6] = U_mat.m[2][0];
    U_data[offset + 7] = U_mat.m[2][1];
    U_data[offset + 8] = U_mat.m[2][2];

    S_data[offset + 0] = S_mat.m[0][0];
    S_data[offset + 1] = S_mat.m[1][1];
    S_data[offset + 2] = S_mat.m[2][2];

    Vh_data[offset + 0] = Vh_mat.m[0][0];
    Vh_data[offset + 1] = Vh_mat.m[0][1];
    Vh_data[offset + 2] = Vh_mat.m[0][2];
    Vh_data[offset + 3] = Vh_mat.m[1][0];
    Vh_data[offset + 4] = Vh_mat.m[1][1];
    Vh_data[offset + 5] = Vh_mat.m[1][2];
    Vh_data[offset + 6] = Vh_mat.m[2][0];
    Vh_data[offset + 7] = Vh_mat.m[2][1];
    Vh_data[offset + 8] = Vh_mat.m[2][2];
  }

  return ret_val;
}

} // namespace native
} // namespace generic
} // namespace impl
