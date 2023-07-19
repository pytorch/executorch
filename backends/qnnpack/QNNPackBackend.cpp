#include <executorch/backends/qnnpack/executor/QNNExecutor.h>
#include <executorch/backends/qnnpack/qnnpack_schema_generated.h>
#include <executorch/backends/qnnpack/utils/utils.h>
#include <executorch/extension/fb/threadpool/threadpool.h>
#include <executorch/runtime/backend/backend_registry.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/util/memory_utils.h>
#include <pytorch_qnnpack.h>
#include <qnnpack_func.h>
#include <string>

namespace torch {
namespace executor {

// On x86, the bias tensor data is loaded using the 128-bit MOVAPS instruction
// ("Move Aligned Packed Single-Precision Floating-Point Values"), which will
// generate an exception if it does not receive 16-byte-aligned data.
static constexpr size_t kTensorDataAlignment = 16;

namespace {
Error allocate_tensor(
    const size_t ndim,
    const exec_aten::SizesType* sizes,
    const ScalarType type,
    MemoryAllocator* runtime_allocator,
    const size_t pad_bytes,
    TensorImpl** tensor_impl_ptr) {
  exec_aten::SizesType* tensor_sizes = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      runtime_allocator, exec_aten::SizesType, ndim);
  std::memcpy(tensor_sizes, sizes, ndim * sizeof(exec_aten::SizesType));
  // We don't really need to allocate strides, but resizes modify strides.
  // TensorImpl constructor however is ok taking nullptr as strides so resize
  // impl needs to account for this difference.
  exec_aten::DimOrderType* tensor_dim_order = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      runtime_allocator, exec_aten::DimOrderType, ndim);
  exec_aten::StridesType* tensor_strides = ET_ALLOCATE_LIST_OR_RETURN_ERROR(
      runtime_allocator, exec_aten::StridesType, ndim);
  for (size_t i = 0; i < ndim; ++i) {
    tensor_dim_order[i] = static_cast<exec_aten::DimOrderType>(i);
  }
  tensor_strides[ndim - 1] = 1;
  for (size_t i = ndim - 1; i > 0; --i) {
    // For sizes[i] == 0, treat it as 1 to be consistent with core Pytorch
    if (sizes[i] == 0) {
      tensor_strides[i - 1] = tensor_strides[i];
    } else {
      tensor_strides[i - 1] = tensor_strides[i] * sizes[i];
    }
  }
  auto tensor_impl =
      ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, TensorImpl);
  new (tensor_impl) TensorImpl(
      type,
      ndim,
      tensor_sizes,
      nullptr,
      tensor_dim_order,
      tensor_strides,
      TensorShapeDynamism::DYNAMIC_BOUND);
  size_t nbytes = tensor_impl->nbytes();
  void* tensor_storage =
      runtime_allocator->allocate(nbytes + pad_bytes, kTensorDataAlignment);
  if (tensor_storage == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  tensor_impl->set_data(tensor_storage);
  *tensor_impl_ptr = tensor_impl;
  return Error::Ok;
}

Error allocate_and_copy_tensor(
    const size_t ndim,
    const exec_aten::SizesType* sizes,
    const void* data,
    const ScalarType type,
    MemoryAllocator* runtime_allocator,
    const size_t pad_bytes,
    TensorImpl** tensor_impl_ptr) {
  ET_CHECK_MSG(
      *tensor_impl_ptr == nullptr,
      "Tensor impl pointer must be null initialized");
  ET_CHECK_MSG(
      allocate_tensor(
          ndim, sizes, type, runtime_allocator, pad_bytes, tensor_impl_ptr) ==
          Error::Ok,
      "Could not allocate tensor in QNNPACK backend.");
  TensorImpl* tensor_impl = *tensor_impl_ptr;
  std::memcpy(tensor_impl->mutable_data(), data, tensor_impl->nbytes());
  return Error::Ok;
}
} // namespace

using namespace qnnpack_utils;

class QnnpackBackend final : public PyTorchBackendInterface {
 public:
  ~QnnpackBackend() = default;

  bool is_available() const override {
    return pytorch_qnnp_status_success == pytorch_qnnp_initialize();
  }

  Result<DelegateHandle*> init(
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs,
      MemoryAllocator* runtime_allocator) const override {
    auto dynamic_linear = fb_qnnpack::GetQNNDynamicLinear(processed->data());
    auto bias = dynamic_linear->bias();

    constexpr size_t pre_pad_bytes = 16;
    // Create + copy Bias Tensor
    TensorImpl* bias_buf = nullptr;
    allocate_and_copy_tensor(
        bias->shape()->size(),
        bias->shape()->data(),
        bias->buffer()->data(),
        ScalarType::Float,
        runtime_allocator,
        pre_pad_bytes,
        &bias_buf);

    // Create + copy Weight Zero-Points Tensor
    auto weights_zp = dynamic_linear->weights_zero_point();
    TensorImpl* zp_buf = nullptr;
    allocate_and_copy_tensor(
        weights_zp->shape()->size(),
        weights_zp->shape()->data(),
        weights_zp->buffer()->data(),
        ScalarType::QUInt8,
        runtime_allocator,
        0,
        &zp_buf);

    // Create + copy Weight Scales Tensor
    auto weights_scale = dynamic_linear->weights_scale();
    TensorImpl* scale_buf = nullptr;
    allocate_and_copy_tensor(
        weights_scale->shape()->size(),
        weights_scale->shape()->data(),
        weights_scale->buffer()->data(),
        ScalarType::Float,
        runtime_allocator,
        0,
        &scale_buf);

    // Create Quantized Input Tensor
    auto input_shape = dynamic_linear->input_shape();
    TensorImpl* qinput_buf = nullptr;
    allocate_tensor(
        input_shape->size(),
        input_shape->data(),
        ScalarType::QUInt8,
        runtime_allocator,
        // Add prepadding to make qnnpack happy
        pre_pad_bytes,
        &qinput_buf);
    qinput_buf->set_data(
        static_cast<uint8_t*>(qinput_buf->mutable_data()) + pre_pad_bytes);

    // Pack Weights
    auto weights = dynamic_linear->weights();
    auto packed_weights = std::make_unique<qnnpack::PackBMatrix>(
        weights->shape()->Get(0), /* input_channels */
        weights->shape()->Get(1), /* output_channels */
        weights_zp->buffer()->data(), /* kernel_zero_points */
        reinterpret_cast<const float*>(
            weights_scale->buffer()->data()), /* requantization_scales */
        weights->buffer()->data(), /* kernel */
        nullptr /* bias */
    );

    // Create QNNExecutor
    QNNExecutor* executor =
        ET_ALLOCATE_INSTANCE_OR_RETURN_ERROR(runtime_allocator, QNNExecutor);

    // NOTE: Since we use placement new and since this type is not trivially
    // destructible, we must call the destructor manually in destroy().
    new (executor) QNNExecutor(
        std::move(packed_weights), bias_buf, qinput_buf, scale_buf, zp_buf);

    // TODO(T144120904): Remove this MMAP block once all users switch to
    // MmapDataLoader.
#if defined(ET_MMAP_SUPPORTED)
    torch::executor::util::mark_memory_as_unused(
        const_cast<void*>(processed->data()), processed->size());
#endif
    processed->Free();

    return executor;
  }

  Error execute(DelegateHandle* handle, EValue** args) const override {
    static constexpr size_t kMaxDims = 16;

    QNNExecutor* etor = static_cast<QNNExecutor*>(handle);

    const Tensor rinput = args[0]->toTensor();
    ET_CHECK_OR_RETURN_ERROR(
        rinput.dim() <= kMaxDims,
        Internal,
        "rinput.dim() %u > kMaxDims %zu",
        (unsigned int)rinput.dim(),
        kMaxDims);
    Tensor::SizesType expected_output_size[kMaxDims];
    for (int32_t i = 0; i < rinput.dim() - 1; ++i) {
      expected_output_size[i] = rinput.size(i);
    }
    expected_output_size[rinput.dim() - 1] = etor->bias_.size(0);

    Tensor output = args[1]->toTensor();
    auto error = resize_tensor(
        output, {expected_output_size, static_cast<size_t>(rinput.dim())});
    if (error != Error::Ok) {
      std::string message("Failed to resize output tensor for size:{");
      for (int32_t i = 0; i < rinput.dim(); i++) {
        message += std::to_string(expected_output_size[i]) + ", ";
      }
      message += "}";
      ET_CHECK_MSG(false, "%s", message.c_str());
    }

    float rinput_min, rinput_max;
    std::tie(rinput_min, rinput_max) = GetMinMax(rinput);
    QuantizationParams input_qparam;

    uint8_t qmin = std::numeric_limits<uint8_t>::min();
    uint8_t qmax = std::numeric_limits<uint8_t>::max();
    Error e = ChooseQuantizationParams(
        rinput_min,
        rinput_max,
        qmin,
        qmax,
        input_qparam,
        false, /* preserve_sparsity */
        false, /* force_scale_power_of_two */
        false /* reduce_range */
    );
    ET_CHECK_OR_RETURN_ERROR(
        e == Error::Ok, Internal, "ChooseQuantizationParams() failed");

    ET_CHECK_OR_RETURN_ERROR(
        input_qparam.zero_point <= qmax && input_qparam.zero_point >= qmin,
        Internal,
        "ChooseQuantizationParams() selected invalid input_zero_point: %d",
        input_qparam.zero_point);

    std::vector<float> dequantization_scales;
    e = GenerateRequantizationScale(
        etor->weight_scales_,
        input_qparam.scale,
        1.0f /* output_scale */,
        dequantization_scales);

    ET_CHECK_OR_RETURN_ERROR(
        e == Error::Ok, Internal, "GenerateRequantizationScale() failed");

    // padding to handle out of bounds access
    dequantization_scales.resize(dequantization_scales.size() + 4);

    // Need to resize quantized tensor to match fp32 tensor sizes
    // Have to do this conditionally since only joiner of asr model has
    if (etor->qinput_.dim() == rinput.dim()) {
      resize(etor->qinput_, rinput.sizes());
    }
    e = QuantizePerTensor(
        rinput, etor->qinput_, input_qparam.scale, input_qparam.zero_point);

    ET_CHECK_OR_RETURN_ERROR(
        e == Error::Ok, Internal, "QuantizePerTensor() failed");

    size_t rows_weight = etor->bias_.size(0);
    size_t rows_input = 1;
    size_t cols_input = rinput.size(rinput.dim() - 1);
    size_t cols_weight = etor->packed_weight_.get()->getInputChannels();
    for (int i = 0; i < rinput.dim() - 1; ++i) {
      rows_input *= rinput.size(i);
    }

    ET_CHECK_OR_RETURN_ERROR(
        cols_input == cols_weight,
        Internal,
        "Can not multiple matrices, size mismatch input[-1]: %zd, weight[1]: %zd",
        cols_input,
        cols_weight);

    auto pthreadpool = torch::executorch::threadpool::get_pthreadpool();
    enum pytorch_qnnp_status status = qnnpack::qnnpackLinearDynamic(
        rows_input, /* const size_t batch_size */
        cols_input, /* const size_t input_channels */
        rows_weight, /* const size_t output_channels */
        input_qparam.zero_point, /* const uint8_t input_zero_point */
        etor->weight_zero_points_
            .const_data_ptr<uint8_t>(), /* const uint8_t* kernel_zero_points */
        dequantization_scales.data(), /* const float* dequantization_scales */
        etor->qinput_.const_data_ptr<uint8_t>(), /* const uint8_t* input */
        cols_input, /* const size_t input_stride */
        etor->packed_weight_.get()
            ->getPackedWeights(), /* void* packed_weights */
        etor->bias_.const_data_ptr<float>(), /* const float* bias */
        output.mutable_data_ptr<float>(), /* float* output */
        rows_weight, /* const size_t output_stride */
        pthreadpool /* pthreadpool_t threadpool */
    );

    ET_CHECK_OR_RETURN_ERROR(
        status == pytorch_qnnp_status_success,
        Internal,
        "qnnpackLinearDynamic failed");

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (handle != nullptr) {
      auto executor = static_cast<QNNExecutor*>(handle);
      // QNNExecutor is not trivially destructible. Since this was constructed
      // manually in init(), we must destroy it manually here.
      executor->~QNNExecutor();
    }
  }
};

namespace {
auto cls = QnnpackBackend();
Backend backend{"QnnpackBackend", &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace

} // namespace executor
} // namespace torch
