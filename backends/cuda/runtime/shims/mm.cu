/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/shims/mm.h>

#include <executorch/backends/aoti/slim/c10/cuda/Exception.h>
#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/runtime/platform/log.h>

#include <cublasLt.h>
#include <cuda_runtime.h>
#ifdef ET_CUBLASLT_LIBRARY
#include <dlfcn.h>
#endif

#include <algorithm>
#include <utility>
#include <vector>

namespace executorch::backends::cuda {

namespace c10 = executorch::backends::aoti::slim::c10;
using executorch::extension::cuda::CudaAllocator;
using executorch::runtime::Error;

namespace {

#define ET_CUBLASLT_FUNCTIONS(X)  \
  X(Create)                       \
  X(Destroy)                      \
  X(GetStatusString)              \
  X(Matmul)                       \
  X(MatmulAlgoGetHeuristic)       \
  X(MatmulDescCreate)             \
  X(MatmulDescDestroy)            \
  X(MatmulDescSetAttribute)       \
  X(MatmulPreferenceCreate)       \
  X(MatmulPreferenceDestroy)      \
  X(MatmulPreferenceSetAttribute) \
  X(MatrixLayoutCreate)           \
  X(MatrixLayoutDestroy)          \
  X(MatrixLayoutSetAttribute)

struct CublasLt {
#define ET_CUBLASLT_MEMBER(name) decltype(&::cublasLt##name) name = nullptr;
  ET_CUBLASLT_FUNCTIONS(ET_CUBLASLT_MEMBER)
#undef ET_CUBLASLT_MEMBER
};

// CMake builds load cuBLASLt on first use so it is not a load-time dependency.
// Buck links a stub that already loads it lazily, so it uses the symbols as is.
const CublasLt* lt() {
  static const CublasLt* const table = []() -> const CublasLt* {
    static CublasLt functions;
#ifdef ET_CUBLASLT_LIBRARY
    void* library = dlopen(ET_CUBLASLT_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      ET_LOG(Error, "Cannot load %s: %s", ET_CUBLASLT_LIBRARY, dlerror());
      return nullptr;
    }
#define ET_CUBLASLT_RESOLVE(name)                                      \
  functions.name = reinterpret_cast<decltype(&::cublasLt##name)>(      \
      dlsym(library, "cublasLt" #name));                               \
  if (functions.name == nullptr) {                                     \
    ET_LOG(Error, "%s has no cublasLt%s", ET_CUBLASLT_LIBRARY, #name); \
    return nullptr;                                                    \
  }
    ET_CUBLASLT_FUNCTIONS(ET_CUBLASLT_RESOLVE)
#undef ET_CUBLASLT_RESOLVE
#else
#define ET_CUBLASLT_ASSIGN(name) functions.name = &::cublasLt##name;
    ET_CUBLASLT_FUNCTIONS(ET_CUBLASLT_ASSIGN)
#undef ET_CUBLASLT_ASSIGN
#endif
    return &functions;
  }();
  return table;
}

#define ET_CUBLAS_CHECK_OR_RETURN_ERROR(EXPR) \
  do {                                        \
    const cublasStatus_t __status = (EXPR);   \
    ET_CHECK_OR_RETURN_ERROR(                 \
        __status == CUBLAS_STATUS_SUCCESS,    \
        Internal,                             \
        "%s failed: %s",                      \
        #EXPR,                                \
        lt()->GetStatusString(__status));     \
  } while (0)

// Only what the chosen algorithm needs is allocated, so this cap only limits
// which algorithms the heuristic may pick.
constexpr size_t kMaxWorkspaceBytes = 32 * 1024 * 1024;

// One handle per thread and device, kept for the thread's lifetime, so calls
// reuse a handle instead of creating one, and need no lock.
class LtHandles {
 public:
  ~LtHandles() {
    for (cublasLtHandle_t handle : handles_) {
      if (handle != nullptr) {
        lt()->Destroy(handle);
      }
    }
  }

  cublasStatus_t get(c10::DeviceIndex device, cublasLtHandle_t* handle) {
    if (static_cast<size_t>(device) >= handles_.size()) {
      handles_.resize(device + 1, nullptr);
    }
    if (handles_[device] == nullptr) {
      const cublasStatus_t status = lt()->Create(&handles_[device]);
      if (status != CUBLAS_STATUS_SUCCESS) {
        handles_[device] = nullptr;
        return status;
      }
    }
    *handle = handles_[device];
    return CUBLAS_STATUS_SUCCESS;
  }

 private:
  std::vector<cublasLtHandle_t> handles_;
};

template <typename T, cublasStatus_t (*CublasLt::*Destroy)(T)>
struct LtDescriptor {
  T value = nullptr;
  ~LtDescriptor() {
    if (value != nullptr) {
      (lt()->*Destroy)(value);
    }
  }
};
using MatmulDesc =
    LtDescriptor<cublasLtMatmulDesc_t, &CublasLt::MatmulDescDestroy>;
using LayoutDesc =
    LtDescriptor<cublasLtMatrixLayout_t, &CublasLt::MatrixLayoutDestroy>;
using PreferenceDesc = LtDescriptor<
    cublasLtMatmulPreference_t,
    &CublasLt::MatmulPreferenceDestroy>;

// A matrix, or a batch of them: element (r, c) of batch b sits at
// b * batch_stride + r * row_stride + c * col_stride elements past data.
struct Matrix {
  void* data;
  int64_t rows;
  int64_t cols;
  int64_t row_stride;
  int64_t col_stride;
  int64_t batch_stride;
};

Matrix matrix_of(Tensor* t) {
  const int64_t d = static_cast<int64_t>(t->dim());
  return Matrix{
      t->data_ptr(),
      t->size(d - 2),
      t->size(d - 1),
      t->stride(d - 2),
      t->stride(d - 1),
      d == 3 ? t->stride(0) : 0};
}

Matrix transposed(Matrix m) {
  std::swap(m.rows, m.cols);
  std::swap(m.row_stride, m.col_stride);
  return m;
}

// Capped at cuBLASLt's default of 256 bytes, as PyTorch does.
uint32_t alignment_of(const void* pointer) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  uint32_t alignment = 1;
  while (alignment < 256 && address % (alignment * 2) == 0) {
    alignment *= 2;
  }
  return alignment;
}

// cuBLAS reads column-major matrices: consecutive elements of a column are
// adjacent, and columns start a leading dimension apart.
bool is_column_major(const Matrix& m, int64_t* ld) {
  if (m.row_stride != 1 && m.rows != 1) {
    return false;
  }
  *ld = m.cols == 1 ? std::max<int64_t>(m.rows, 1) : m.col_stride;
  return *ld >= std::max<int64_t>(m.rows, 1);
}

bool is_row_major(const Matrix& m, int64_t* ld) {
  return is_column_major(transposed(m), ld);
}

// self as addmm broadcasts it to rows x cols: a missing or size-1 dimension
// repeats with a zero stride.
Matrix broadcast(Tensor* self, int64_t rows, int64_t cols) {
  const int64_t d = static_cast<int64_t>(self->dim());
  const auto stride = [&](int64_t dim) {
    return dim >= 0 && self->size(dim) != 1 ? self->stride(dim) : 0;
  };
  return Matrix{self->data_ptr(), rows, cols, stride(d - 2), stride(d - 1), 0};
}

// Elements move as raw bits, so one kernel per element size serves every dtype.
template <typename T>
__global__ void copy_matrix_kernel(Matrix dst, Matrix src) {
  const int64_t numel = dst.rows * dst.cols;
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; i < numel;
       i += int64_t{gridDim.x} * blockDim.x) {
    const int64_t r = i / dst.cols;
    const int64_t c = i % dst.cols;
    static_cast<T*>(dst.data)[r * dst.row_stride + c * dst.col_stride] =
        static_cast<const T*>(
            src.data)[r * src.row_stride + c * src.col_stride];
  }
}

Error copy_matrix(
    const char* name,
    const Matrix& dst,
    const Matrix& src,
    size_t element_size,
    c10::DeviceIndex device) {
  auto stream = getCurrentCUDAStream(device);
  ET_CHECK_OK_OR_RETURN_ERROR(stream.error(), "%s: no CUDA stream", name);
  constexpr int64_t kThreads = 256;
  const int blocks = static_cast<int>(
      std::min<int64_t>((dst.rows * dst.cols + kThreads - 1) / kThreads, 4096));
  switch (element_size) {
    case sizeof(uint32_t):
      copy_matrix_kernel<uint32_t>
          <<<blocks, kThreads, 0, stream.get()>>>(dst, src);
      break;
    case sizeof(uint16_t):
      copy_matrix_kernel<uint16_t>
          <<<blocks, kThreads, 0, stream.get()>>>(dst, src);
      break;
    default:
      ET_CHECK_OR_RETURN_ERROR(
          false,
          NotSupported,
          "%s: unsupported element size %zu",
          name,
          element_size);
  }
  ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
  return Error::Ok;
}

// How cuBLAS reads an operand: as stored when it is column-major, or as the
// transpose of the column-major matrix it is when it is row-major.
struct Operand {
  cublasOperation_t op;
  uint64_t stored_rows;
  uint64_t stored_cols;
  int64_t ld;
};

bool as_operand(const Matrix& m, Operand* operand) {
  int64_t ld = 0;
  if (is_column_major(m, &ld)) {
    *operand = Operand{CUBLAS_OP_N, uint64_t(m.rows), uint64_t(m.cols), ld};
    return true;
  }
  if (is_row_major(m, &ld)) {
    *operand = Operand{CUBLAS_OP_T, uint64_t(m.cols), uint64_t(m.rows), ld};
    return true;
  }
  return false;
}

Error make_layout(
    LayoutDesc& layout,
    cudaDataType_t type,
    uint64_t rows,
    uint64_t cols,
    int64_t ld,
    int32_t batch,
    int64_t batch_stride) {
  ET_CUBLAS_CHECK_OR_RETURN_ERROR(
      lt()->MatrixLayoutCreate(&layout.value, type, rows, cols, ld));
  if (batch > 1) {
    ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatrixLayoutSetAttribute(
        layout.value,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch,
        sizeof(batch)));
    ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatrixLayoutSetAttribute(
        layout.value,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &batch_stride,
        sizeof(batch_stride)));
  }
  return Error::Ok;
}

// d = alpha * a @ b + beta * d, plus bias added to every column of d when bias
// is not null. d must be column-major.
Error gemm(
    const char* name,
    const Matrix& d,
    const Matrix& a,
    const Matrix& b,
    int32_t batch,
    cudaDataType_t type,
    c10::DeviceIndex device,
    const void* bias,
    float alpha,
    float beta) {
  int64_t ldd = 0;
  ET_CHECK_OR_RETURN_ERROR(
      is_column_major(d, &ldd),
      Internal,
      "%s: result is not column-major",
      name);
  Operand op_a{};
  Operand op_b{};
  ET_CHECK_OR_RETURN_ERROR(
      as_operand(a, &op_a) && as_operand(b, &op_b),
      NotSupported,
      "%s: an operand is neither row-major nor column-major",
      name);

  ET_CHECK_OR_RETURN_ERROR(
      lt() != nullptr, NotFound, "%s: cannot load cuBLASLt", name);

  auto stream = getCurrentCUDAStream(device);
  ET_CHECK_OK_OR_RETURN_ERROR(stream.error(), "%s: no CUDA stream", name);

  thread_local LtHandles handles;
  cublasLtHandle_t handle = nullptr;
  ET_CUBLAS_CHECK_OR_RETURN_ERROR(handles.get(device, &handle));

  MatmulDesc desc;
  ET_CUBLAS_CHECK_OR_RETURN_ERROR(
      lt()->MatmulDescCreate(&desc.value, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatmulDescSetAttribute(
      desc.value, CUBLASLT_MATMUL_DESC_TRANSA, &op_a.op, sizeof(op_a.op)));
  ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatmulDescSetAttribute(
      desc.value, CUBLASLT_MATMUL_DESC_TRANSB, &op_b.op, sizeof(op_b.op)));
  if (bias != nullptr) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatmulDescSetAttribute(
        desc.value,
        CUBLASLT_MATMUL_DESC_EPILOGUE,
        &epilogue,
        sizeof(epilogue)));
    ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatmulDescSetAttribute(
        desc.value, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  }

  LayoutDesc layout_a;
  LayoutDesc layout_b;
  LayoutDesc layout_d;
  ET_CHECK_OK_OR_RETURN_ERROR(make_layout(
      layout_a,
      type,
      op_a.stored_rows,
      op_a.stored_cols,
      op_a.ld,
      batch,
      a.batch_stride));
  ET_CHECK_OK_OR_RETURN_ERROR(make_layout(
      layout_b,
      type,
      op_b.stored_rows,
      op_b.stored_cols,
      op_b.ld,
      batch,
      b.batch_stride));
  ET_CHECK_OK_OR_RETURN_ERROR(make_layout(
      layout_d,
      type,
      uint64_t(d.rows),
      uint64_t(d.cols),
      ldd,
      batch,
      d.batch_stride));

  PreferenceDesc preference;
  ET_CUBLAS_CHECK_OR_RETURN_ERROR(
      lt()->MatmulPreferenceCreate(&preference.value));
  ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatmulPreferenceSetAttribute(
      preference.value,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &kMaxWorkspaceBytes,
      sizeof(kMaxWorkspaceBytes)));
  // The heuristic otherwise assumes 256-byte aligned pointers, and can pick an
  // algorithm that then fails on a view that starts partway into its storage.
  const std::pair<cublasLtMatmulPreferenceAttributes_t, const void*>
      alignments[] = {
          {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a.data},
          {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b.data},
          {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, d.data},
          {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, d.data}};
  for (const auto& [attribute, pointer] : alignments) {
    const uint32_t alignment = alignment_of(pointer);
    ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatmulPreferenceSetAttribute(
        preference.value, attribute, &alignment, sizeof(alignment)));
  }

  cublasLtMatmulHeuristicResult_t heuristic{};
  int found = 0;
  ET_CUBLAS_CHECK_OR_RETURN_ERROR(lt()->MatmulAlgoGetHeuristic(
      handle,
      desc.value,
      layout_a.value,
      layout_b.value,
      layout_d.value,
      layout_d.value,
      preference.value,
      1,
      &heuristic,
      &found));
  ET_CHECK_OR_RETURN_ERROR(
      found > 0, NotSupported, "%s: cuBLASLt has no algorithm", name);

  void* workspace = nullptr;
  if (heuristic.workspaceSize > 0) {
    auto allocation = CudaAllocator::allocate_async(
        heuristic.workspaceSize, device, stream.get());
    ET_CHECK_OK_OR_RETURN_ERROR(
        allocation.error(), "%s: workspace allocation failed", name);
    workspace = allocation.get();
  }
  const cublasStatus_t status = lt()->Matmul(
      handle,
      desc.value,
      &alpha,
      a.data,
      layout_a.value,
      b.data,
      layout_b.value,
      &beta,
      d.data,
      layout_d.value,
      d.data,
      layout_d.value,
      &heuristic.algo,
      workspace,
      heuristic.workspaceSize,
      stream.get());
  if (workspace != nullptr) {
    CudaAllocator::deallocate_async(workspace, device, stream.get());
  }
  ET_CHECK_OR_RETURN_ERROR(
      status == CUBLAS_STATUS_SUCCESS,
      Internal,
      "%s: cublasLtMatmul failed: %s",
      name,
      lt()->GetStatusString(status));
  return Error::Ok;
}

Error cuda_data_type(
    const char* name,
    c10::ScalarType dtype,
    cudaDataType_t* type) {
  switch (dtype) {
    case c10::ScalarType::Float:
      *type = CUDA_R_32F;
      return Error::Ok;
    case c10::ScalarType::Half:
      *type = CUDA_R_16F;
      return Error::Ok;
    case c10::ScalarType::BFloat16:
      *type = CUDA_R_16BF;
      return Error::Ok;
    default:
      ET_CHECK_OR_RETURN_ERROR(
          false,
          NotSupported,
          "%s: unsupported dtype %d",
          name,
          static_cast<int>(dtype));
  }
}

// Checks out = self @ mat2 for 2-D (ndim 2) or batched (ndim 3) operands.
Error check_operands(
    const char* name,
    Tensor* out,
    Tensor* self,
    Tensor* mat2,
    size_t ndim) {
  ET_CHECK_OR_RETURN_ERROR(
      out != nullptr && self != nullptr && mat2 != nullptr,
      InvalidArgument,
      "%s: a tensor is null",
      name);
  ET_CHECK_OR_RETURN_ERROR(
      out->dim() == ndim && self->dim() == ndim && mat2->dim() == ndim,
      InvalidArgument,
      "%s: expected %zu-D tensors",
      name,
      ndim);
  ET_CHECK_OR_RETURN_ERROR(
      out->device().is_cuda() && out->device_index() >= 0 &&
          self->device() == out->device() && mat2->device() == out->device(),
      InvalidArgument,
      "%s: expected tensors on one CUDA device",
      name);
  ET_CHECK_OR_RETURN_ERROR(
      self->dtype() == out->dtype() && mat2->dtype() == out->dtype(),
      InvalidArgument,
      "%s: tensors differ in dtype",
      name);
  const size_t m = ndim - 2;
  ET_CHECK_OR_RETURN_ERROR(
      self->size(m + 1) == mat2->size(m) && out->size(m) == self->size(m) &&
          out->size(m + 1) == mat2->size(m + 1),
      InvalidArgument,
      "%s: shapes do not multiply",
      name);
  ET_CHECK_OR_RETURN_ERROR(
      ndim == 2 ||
          (self->size(0) == out->size(0) && mat2->size(0) == out->size(0)),
      InvalidArgument,
      "%s: batch sizes differ",
      name);
  ET_CHECK_OR_RETURN_ERROR(
      self->size(m + 1) > 0 || out->numel() == 0,
      NotSupported,
      "%s: an empty inner dimension is not supported",
      name);
  return Error::Ok;
}

// out = alpha * self @ mat2 + beta * out, plus a bias row added to every row of
// out when bias is not null.
Error multiply(
    const char* name,
    Tensor* out,
    Tensor* self,
    Tensor* mat2,
    const void* bias,
    float alpha,
    float beta) {
  if (out->numel() == 0) {
    return Error::Ok;
  }
  cudaDataType_t type{};
  ET_CHECK_OK_OR_RETURN_ERROR(cuda_data_type(name, out->dtype(), &type));
  const int64_t batch = out->dim() == 3 ? out->size(0) : 1;
  ET_CHECK_OR_RETURN_ERROR(
      batch <= INT32_MAX, NotSupported, "%s: batch too large", name);

  Matrix d = matrix_of(out);
  Matrix a = matrix_of(self);
  Matrix b = matrix_of(mat2);
  ET_CHECK_OR_RETURN_ERROR(
      batch == 1 || d.batch_stride != 0,
      InvalidArgument,
      "%s: the result repeats one matrix over the batch",
      name);
  int64_t ld = 0;
  // cuBLAS writes column-major results, so a row-major out is computed as
  // out^T = mat2^T @ self^T. That orientation also lines a bias row up with
  // the rows of the column-major result, which is what the epilogue adds.
  if (is_row_major(d, &ld)) {
    d = transposed(d);
    std::swap(a, b);
    a = transposed(a);
    b = transposed(b);
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        is_column_major(d, &ld),
        NotSupported,
        "%s: the result is neither row-major nor column-major",
        name);
    ET_CHECK_OR_RETURN_ERROR(
        bias == nullptr,
        NotSupported,
        "%s: a bias needs a row-major result",
        name);
  }
  return gemm(
      name,
      d,
      a,
      b,
      static_cast<int32_t>(batch),
      type,
      out->device_index(),
      bias,
      alpha,
      beta);
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda_mm_out(Tensor* out, Tensor* self, Tensor* mat2) {
  constexpr const char* name = "aoti_torch_cuda_mm_out";
  ET_CHECK_OK_OR_RETURN_ERROR(check_operands(name, out, self, mat2, 2));
  return multiply(name, out, self, mat2, nullptr, 1.0f, 0.0f);
}

AOTITorchError aoti_torch_cuda_addmm_out(
    Tensor* out,
    Tensor* self,
    Tensor* mat1,
    Tensor* mat2,
    double beta,
    double alpha) {
  constexpr const char* name = "aoti_torch_cuda_addmm_out";
  ET_CHECK_OK_OR_RETURN_ERROR(check_operands(name, out, mat1, mat2, 2));
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr && self->device() == out->device() &&
          self->dtype() == out->dtype(),
      InvalidArgument,
      "%s: self must be on the result's device and of its dtype",
      name);
  if (out->numel() == 0) {
    return Error::Ok;
  }
  const int64_t rows = out->size(0);
  const int64_t cols = out->size(1);
  const int64_t self_dim = static_cast<int64_t>(self->dim());
  ET_CHECK_OR_RETURN_ERROR(
      self_dim <= 2 &&
          (self_dim < 1 || self->size(self_dim - 1) == cols ||
           self->size(self_dim - 1) == 1) &&
          (self_dim < 2 || self->size(0) == rows || self->size(0) == 1),
      InvalidArgument,
      "%s: self does not broadcast to the result",
      name);

  const Matrix src = broadcast(self, rows, cols);
  const Matrix dst = matrix_of(out);
  int64_t ld = 0;
  // A row added to every row of a row-major result runs as the cuBLASLt bias
  // epilogue, as PyTorch does for a linear layer.
  if (beta == 1.0 && src.row_stride == 0 &&
      (src.col_stride == 1 || cols == 1) && is_row_major(dst, &ld)) {
    return multiply(
        name, out, mat1, mat2, self->data_ptr(), static_cast<float>(alpha), 0);
  }
  // Any other self is written into out and accumulated there with beta. With
  // beta == 0, cuBLAS does not read out, so self is ignored, as in PyTorch.
  const bool self_is_out = src.data == dst.data &&
      src.row_stride == dst.row_stride && src.col_stride == dst.col_stride;
  if (beta != 0.0 && !self_is_out) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        copy_matrix(name, dst, src, out->itemsize(), out->device_index()));
  }
  return multiply(
      name,
      out,
      mat1,
      mat2,
      nullptr,
      static_cast<float>(alpha),
      static_cast<float>(beta));
}

AOTITorchError
aoti_torch_cuda_bmm_out(Tensor* out, Tensor* self, Tensor* mat2) {
  constexpr const char* name = "aoti_torch_cuda_bmm_out";
  ET_CHECK_OK_OR_RETURN_ERROR(check_operands(name, out, self, mat2, 3));
  return multiply(name, out, self, mat2, nullptr, 1.0f, 0.0f);
}

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
