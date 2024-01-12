#include <executorch/backends/xnnpack/threadpool/threadpool.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>
#include <xnnpack.h>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

struct ModelArgs {
  int32_t dim = 4096;
  int32_t n_layers = 32;
  int32_t n_heads = 32;
  int32_t n_kv_heads = 32;
  int32_t vocab_size = 32000;
  int32_t multiple_of = 256;
  float ffn_dim_multiplier;
  float norm_eps = 1e-5;
  int32_t max_batch_size = 32;
  int32_t max_seq_len = 256;
  int32_t kv_cache_size = 256;
};

enum class FullyConnectedOpType : uint8_t {
  QD8_F16_QC4W = 0,
  QD8_F16_QC8W,
  QD8_F32_QC4W,
  QD8_F32_QC8W,
};

enum class ComputeType : uint8_t {
  F16 = 0,
  F32,
  QuantizedInt8,
};

using torch::executor::ScalarType;
using torch::executor::testing::TensorFactory;

namespace {
static void* aligned_allocate(size_t alignment, size_t size) {
#if defined(__ANDROID__)
  return memalign(alignment, size);
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void* memory_ptr = NULL;
  if (posix_memalign(&memory_ptr, alignment, size) != 0) {
    return NULL;
  }
  return memory_ptr;
#endif
}

// Due to lack of support for short types in ExecuTorch just using Short for now
ScalarType MyHalf = ScalarType::Short;
class TensorFactoryWrapper {
 public:
  static torch::executor::Tensor make_tensor(
      ScalarType dtype,
      const std::vector<int32_t>& sizes) {
    sizes_vector_.push_back(sizes);
    auto& size = sizes_vector_.back();
    size[0] = size[0] * 4;
    return make_tensor_impl(dtype, sizes);
  }

  static torch::executor::Tensor make_tensor_impl(
      ScalarType dtype,
      const std::vector<int32_t>& sizes);

  static std::vector<std::vector<int32_t>> sizes_vector_;
  static TensorFactory<ScalarType::Char> tf_int8_;
  static TensorFactory<ScalarType::Byte> tf_uint8_;
  static TensorFactory<ScalarType::Int> tf_int_;
  static TensorFactory<ScalarType::Float> tf_float_;
  static TensorFactory<ScalarType::Short> tf_short_;
};
std::vector<std::vector<int32_t>> TensorFactoryWrapper::sizes_vector_;
TensorFactory<ScalarType::Char> TensorFactoryWrapper::tf_int8_;
TensorFactory<ScalarType::Byte> TensorFactoryWrapper::tf_uint8_;
TensorFactory<ScalarType::Int> TensorFactoryWrapper::tf_int_;
TensorFactory<ScalarType::Float> TensorFactoryWrapper::tf_float_;
TensorFactory<ScalarType::Short> TensorFactoryWrapper::tf_short_;

torch::executor::Tensor TensorFactoryWrapper::make_tensor_impl(
    ScalarType dtype,
    const std::vector<int32_t>& sizes) {
  if (dtype == torch::executor::ScalarType::Char) {
    return tf_int8_.ones(sizes);
  } else if (dtype == torch::executor::ScalarType::Byte) {
    return tf_uint8_.ones(sizes);
  } else if (dtype == torch::executor::ScalarType::Int) {
    return tf_int_.ones(sizes);
  } else if (dtype == torch::executor::ScalarType::Float) {
    return tf_float_.ones(sizes);
  } else if (dtype == ::MyHalf) {
    return tf_short_.ones(sizes);
  }
  ET_CHECK_MSG(
      false, "Cannot make tensor of type %s", torch::executor::toString(dtype));
}

xnn_operator_t create_fully_connected(
    const FullyConnectedOpType fc_type,
    const int32_t input_channels,
    const int32_t output_channels) {
  xnn_operator_t op = nullptr;
  if (fc_type == FullyConnectedOpType::QD8_F16_QC4W ||
      fc_type == FullyConnectedOpType::QD8_F32_QC4W) {
    std::vector<uint8_t> kernel(output_channels * input_channels / 2, 1);
    std::vector<float> kernel_scales(output_channels, 1.0);
    const float output_min = -std::numeric_limits<float>::infinity();
    const float output_max = +std::numeric_limits<float>::infinity();

    if (fc_type == FullyConnectedOpType::QD8_F16_QC4W) {
      const xnn_status status = xnn_create_fully_connected_nc_qd8_f16_qc4w(
          input_channels,
          output_channels,
          input_channels,
          output_channels,
          8,
          kernel_scales.data(),
          kernel.data(),
          nullptr,
          output_min,
          output_max,
          0,
          nullptr,
          nullptr,
          &op);
      ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    } else {
      const xnn_status status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
          input_channels,
          output_channels,
          input_channels,
          output_channels,
          8,
          kernel_scales.data(),
          kernel.data(),
          nullptr,
          output_min,
          output_max,
          0,
          nullptr,
          nullptr,
          &op);
      ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    }
    return op;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", fc_type);
}

xnn_operator_t create_sdpa(const ComputeType compute_type) {
  xnn_operator_t op = nullptr;
  if (compute_type == ComputeType::F16) {
    const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f16(
        xnn_attention_logits_cap_type_none, nullptr, 0, &op);
    ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    return op;
  } else if (compute_type == ComputeType::F32) {
    const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f32(
        xnn_attention_logits_cap_type_none, nullptr, 0, &op);
    ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    return op;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", compute_type);
}

xnn_operator_t create_convert_dq(const FullyConnectedOpType fc_type) {
  xnn_operator_t op = nullptr;
  if (fc_type == FullyConnectedOpType::QD8_F16_QC4W) {
    const xnn_status status = xnn_create_convert_nc_f16_qd8(0, &op);
    ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    return op;
  } else if (fc_type == FullyConnectedOpType::QD8_F32_QC4W) {
    const xnn_status status = xnn_create_convert_nc_f32_qd8(0, &op);
    ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    return op;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", fc_type);
}

xnn_operator_t create_transpose_nd(const ComputeType type) {
  xnn_operator_t op = nullptr;
  if (type == ComputeType::F16) {
    const xnn_status status = xnn_create_transpose_nd_x16(0, &op);
    ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    return op;
  } else if (type == ComputeType::F32) {
    const xnn_status status = xnn_create_transpose_nd_x32(0, &op);
    ET_CHECK_MSG(op != nullptr, "Failed to create xnnpack operator");
    return op;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", type);
}

void run_fully_connected(
    const FullyConnectedOpType fc_type,
    xnn_operator_t op,
    const xnn_dynamic_quantization_params* qparams,
    const torch::executor::Tensor& input,
    torch::executor::Tensor& output) {
  auto threadpool = torch::executorch::threadpool::get_pthreadpool();
  int32_t batch_size = input.size(0);
  ET_CHECK_MSG(input.scalar_type() == ScalarType::Char, "Input must be int8");
  if (fc_type == FullyConnectedOpType::QD8_F16_QC4W) {
    ET_CHECK_MSG(output.scalar_type() == ::MyHalf, "Output must be Half");
    xnn_status status =
        xnn_reshape_fully_connected_nc_qd8_f16_qc4w(op, 1, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to reshape xnnpack operator");

    status = xnn_setup_fully_connected_nc_qd8_f16_qc4w(
        op,
        input.const_data_ptr<int8_t>(),
        output.mutable_data_ptr(),
        reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
            qparams));
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  } else if (fc_type == FullyConnectedOpType::QD8_F32_QC4W) {
    ET_CHECK_MSG(
        output.scalar_type() == ScalarType::Float, "Output must be float");
    xnn_status status =
        xnn_reshape_fully_connected_nc_qd8_f32_qc4w(op, 1, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to reshape xnnpack operator");

    status = xnn_setup_fully_connected_nc_qd8_f32_qc4w(
        op,
        input.const_data_ptr<int8_t>(),
        output.mutable_data_ptr<float>(),
        reinterpret_cast<const struct xnn_dynamic_quantization_params*>(
            qparams));
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", fc_type);
}

void run_sdpa(
    const ComputeType compute_type,
    xnn_operator_t op,
    const int32_t batch_size,
    const int32_t query_heads,
    const int32_t query_tokens,
    const int32_t kv_heads,
    const int32_t kv_tokens,
    const int32_t qk_channels,
    const int32_t v_channels,
    torch::executor::Tensor& query,
    torch::executor::Tensor& key,
    torch::executor::Tensor& value,
    torch::executor::Tensor& scale,
    torch::executor::Tensor& mask,
    torch::executor::Tensor& output) {
  auto threadpool = torch::executorch::threadpool::get_pthreadpool();
  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  if (compute_type == ComputeType::F16) {
    ET_CHECK_MSG(query.scalar_type() == ::MyHalf, "Query must be half");
    ET_CHECK_MSG(key.scalar_type() == ::MyHalf, "Key must be half");
    ET_CHECK_MSG(value.scalar_type() == ::MyHalf, "Value must be half");
    ET_CHECK_MSG(scale.scalar_type() == ::MyHalf, "Scalemust be half");
    ET_CHECK_MSG(mask.scalar_type() == ::MyHalf, "Mask must be half");
    ET_CHECK_MSG(output.scalar_type() == ::MyHalf, "Output must be half");
    xnn_status status = xnn_reshape_scaled_dot_product_attention_nhtc_f16(
        op,
        batch_size,
        query_heads,
        query_tokens,
        kv_heads,
        kv_tokens,
        qk_channels,
        v_channels,
        &workspace_size,
        &workspace_alignment,
        threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success,
        "Failed to reshape sdpa xnnpack operator");

    constexpr std::ptrdiff_t alignment{64};
    void* workspace_ptr = aligned_allocate(alignment, workspace_size);
    ET_CHECK_MSG(
        workspace_ptr != nullptr, "Failed to allocate workspace memory");
    status = xnn_setup_scaled_dot_product_attention_nhtc_f16(
        op,
        workspace_ptr,
        query.const_data_ptr<uint16_t>(),
        key.const_data_ptr<uint16_t>(),
        value.const_data_ptr<uint16_t>(),
        scale.const_data_ptr<uint16_t>(),
        mask.const_data_ptr<uint16_t>(),
        output.mutable_data_ptr<void>());
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup sdpa xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    free(workspace_ptr);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  } else if (compute_type == ComputeType::F32) {
    ET_CHECK_MSG(
        query.scalar_type() == ScalarType::Float, "Query must be float");
    ET_CHECK_MSG(key.scalar_type() == ScalarType::Float, "Key must be float");
    ET_CHECK_MSG(
        value.scalar_type() == ScalarType::Float, "Value must be float");
    ET_CHECK_MSG(
        scale.scalar_type() == ScalarType::Float, "Scalemust be float");
    ET_CHECK_MSG(mask.scalar_type() == ScalarType::Float, "Mask must be float");
    ET_CHECK_MSG(
        output.scalar_type() == ScalarType::Float, "Output must be float");
    xnn_status status = xnn_reshape_scaled_dot_product_attention_nhtc_f32(
        op,
        batch_size,
        query_heads,
        query_tokens,
        kv_heads,
        kv_tokens,
        qk_channels,
        v_channels,
        &workspace_size,
        &workspace_alignment,
        threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success,
        "Failed to reshape sdpa xnnpack operator");

    constexpr std::ptrdiff_t alignment{64};
    void* workspace_ptr = aligned_allocate(alignment, workspace_size);
    ET_CHECK_MSG(
        workspace_ptr != nullptr, "Failed to allocate workspace memory");
    status = xnn_setup_scaled_dot_product_attention_nhtc_f32(
        op,
        workspace_ptr,
        query.const_data_ptr<float>(),
        key.const_data_ptr<float>(),
        value.const_data_ptr<float>(),
        scale.const_data_ptr<float>(),
        mask.const_data_ptr<float>(),
        output.mutable_data_ptr<float>());
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup sdpa xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    free(workspace_ptr);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  }
  ET_CHECK_MSG(false, "Unsupported operator type:%hhu", compute_type);
}

void run_convert(
    const FullyConnectedOpType fc_type,
    xnn_operator_t op,
    xnn_dynamic_quantization_params* qparams,
    const torch::executor::Tensor& input,
    torch::executor::Tensor& output) {
  auto threadpool = torch::executorch::threadpool::get_pthreadpool();
  ET_CHECK_MSG(input.dim() == 2, "Expected input tensor dim of 2");
  const int32_t batch_size = input.size(0);
  const int32_t input_channels = input.size(1);
  if (fc_type == FullyConnectedOpType::QD8_F16_QC4W) {
    ET_CHECK_MSG(input.scalar_type() == ::MyHalf, "Input must be half");
    ET_CHECK_MSG(
        output.scalar_type() == ScalarType::Char, "Output must be int8");
    xnn_status status = xnn_reshape_convert_nc_f16_qd8(
        op,
        batch_size,
        input_channels,
        input_channels,
        input_channels,
        threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to reshape xnnpack operator");

    status = xnn_setup_convert_nc_f16_qd8(
        op,
        input.const_data_ptr<uint16_t>(),
        output.mutable_data_ptr<int8_t>(),
        reinterpret_cast<struct xnn_dynamic_quantization_params*>(qparams));
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  } else if (fc_type == FullyConnectedOpType::QD8_F32_QC4W) {
    ET_CHECK_MSG(
        input.scalar_type() == ScalarType::Float, "Input must be float");
    ET_CHECK_MSG(
        output.scalar_type() == ScalarType::Char, "Output must be int8");
    xnn_status status = xnn_reshape_convert_nc_f32_qd8(
        op,
        batch_size,
        input_channels,
        input_channels,
        input_channels,
        threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to reshape xnnpack operator");

    status = xnn_setup_convert_nc_f32_qd8(
        op,
        input.const_data_ptr<float>(),
        output.mutable_data_ptr<int8_t>(),
        reinterpret_cast<struct xnn_dynamic_quantization_params*>(qparams));
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  }
}

void run_transpose_nd(
    const ComputeType type,
    xnn_operator_t op,
    const int32_t num_dims,
    const std::vector<size_t>& sizes,
    const std::vector<size_t>& perm,
    const torch::executor::Tensor& input,
    torch::executor::Tensor& output) {
  auto threadpool = torch::executorch::threadpool::get_pthreadpool();
  if (type == ComputeType::F16) {
    ET_CHECK_MSG(input.scalar_type() == ::MyHalf, "Input must be half");
    ET_CHECK_MSG(output.scalar_type() == ::MyHalf, "Input must be half");
    xnn_status status = xnn_reshape_transpose_nd_x16(
        op, num_dims, sizes.data(), perm.data(), threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to reshape xnnpack operator");

    status = xnn_setup_transpose_nd_x16(
        op,
        input.const_data_ptr<uint16_t>(),
        output.mutable_data_ptr<uint16_t>());
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  } else if (type == ComputeType::F32) {
    ET_CHECK_MSG(
        input.scalar_type() == ScalarType::Float, "Input must be float");
    ET_CHECK_MSG(
        output.scalar_type() == ScalarType::Float, "Input must be float");
    xnn_status status = xnn_reshape_transpose_nd_x32(
        op, num_dims, sizes.data(), perm.data(), threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to reshape xnnpack operator");

    status = xnn_setup_transpose_nd_x32(
        op, input.const_data_ptr<float>(), output.mutable_data_ptr<float>());
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to setup xnnpack operator");

    status = xnn_run_operator(op, threadpool);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to run xnnpack operator");
    return;
  }
}
} // namespace

class MultiHeadedAttention {
 public:
  /*
  What is not accounted for:
  1. RoPE
  */
  explicit MultiHeadedAttention(
      const ModelArgs& args,
      const FullyConnectedOpType linear_type,
      const ComputeType sdpa_type)
      : linear_type_(linear_type), sdpa_type_(sdpa_type) {
    ET_CHECK_MSG(
        args.n_heads == args.n_kv_heads,
        "MultiHeadedAttention only supports n_heads == n_kv_heads");
    int32_t input_channels = args.dim;
    int32_t output_channels = args.n_heads * (args.dim / args.n_heads);
    int32_t head_dim = args.dim / args.n_heads;

    convert_op_ = create_convert_dq(linear_type);

    q_proj_ =
        create_fully_connected(linear_type, input_channels, output_channels);
    k_proj_ =
        create_fully_connected(linear_type, input_channels, output_channels);
    v_proj_ =
        create_fully_connected(linear_type, input_channels, output_channels);

    // Tranpose q, k, v
    // Use single op to do the tranpose
    // Since all q, k and v projected output are of the same size
    qkv_sdpa_out_transpose_ = create_transpose_nd(sdpa_type_);
    qkv_dims_[0] = 1;
    qkv_dims_[1] = 1;
    qkv_dims_[2] = args.n_heads;
    qkv_dims_[3] = head_dim;

    // Not doing RoPE
    sdpa_op_ = create_sdpa(sdpa_type);
    convert_sdpa_output_op_ = create_convert_dq(linear_type);
    sdpa_dims_[0] = 1;
    sdpa_dims_[1] = args.n_heads;
    sdpa_dims_[2] = 1; // Seq len = 1 for 1 token at a time
    sdpa_dims_[3] = head_dim;
    o_proj_ =
        create_fully_connected(linear_type, output_channels, input_channels);

    torch::executor::ScalarType qkv_output_dtype;
    torch::executor::ScalarType q_input_dtype;

    torch::executor::ScalarType qkv_linear_dtype =
        torch::executor::ScalarType::Char;
    if (linear_type_ == FullyConnectedOpType::QD8_F16_QC4W) {
      qkv_output_dtype = ::MyHalf;
      q_input_dtype = ::MyHalf;
    } else if (linear_type_ == FullyConnectedOpType::QD8_F32_QC4W) {
      qkv_output_dtype = torch::executor::ScalarType::Float;
      q_input_dtype = torch::executor::ScalarType::Float;
    } else {
      ET_CHECK_MSG(false, "Unsupported operator type:%hhu", linear_type);
    }

    query_input_float_ = TensorFactoryWrapper::make_tensor(
        q_input_dtype, {benchmarking_batch_size_, input_channels});
    query_input_ = TensorFactoryWrapper::make_tensor(
        qkv_linear_dtype, {benchmarking_batch_size_, input_channels});

    query_output_ = TensorFactoryWrapper::make_tensor(
        qkv_output_dtype, {benchmarking_batch_size_, output_channels});
    key_output_ = TensorFactoryWrapper::make_tensor(
        qkv_output_dtype, {benchmarking_batch_size_, output_channels});
    value_output_ = TensorFactoryWrapper::make_tensor(
        qkv_output_dtype, {benchmarking_batch_size_, output_channels});

    sdpa_output_ = TensorFactoryWrapper::make_tensor(
        qkv_output_dtype, {benchmarking_batch_size_, output_channels});
    sdpa_output_dq_ = TensorFactoryWrapper::make_tensor(
        qkv_linear_dtype, {benchmarking_batch_size_, output_channels});

    qparams_.resize(benchmarking_batch_size_ + XNN_EXTRA_QUANTIZATION_PARAMS);

    k_cache_ = TensorFactoryWrapper::make_tensor(
        qkv_output_dtype,
        {benchmarking_batch_size_, args.n_heads, args.max_seq_len, head_dim});
    v_cache_ = TensorFactoryWrapper::make_tensor(
        qkv_output_dtype,
        {benchmarking_batch_size_, args.n_heads, args.max_seq_len, head_dim});
    scales_ = TensorFactoryWrapper::make_tensor(qkv_output_dtype, {head_dim});
    mask_ = TensorFactoryWrapper::make_tensor(
        qkv_output_dtype, {1, args.max_seq_len});

    query_heads_ = args.n_heads;
    kv_heads_ = args.n_heads;
    query_tokens_ = 1;
    kv_tokens_ = args.max_seq_len;
    qk_channels_ = head_dim;
    v_channels_ = head_dim;
  }

  MultiHeadedAttention(const MultiHeadedAttention& other) =
      delete; // Copy constructor
  MultiHeadedAttention& operator=(const MultiHeadedAttention& other) =
      delete; // Assignment operator
  MultiHeadedAttention(MultiHeadedAttention&& other) noexcept =
      delete; // Move constructor
  MultiHeadedAttention& operator=(MultiHeadedAttention&& other) noexcept =
      delete; // Move assignment operator

  ~MultiHeadedAttention() {
    xnn_delete_operator(convert_op_);
    xnn_delete_operator(convert_sdpa_output_op_);
    xnn_delete_operator(q_proj_);
    xnn_delete_operator(k_proj_);
    xnn_delete_operator(v_proj_);
    xnn_delete_operator(qkv_sdpa_out_transpose_);
    xnn_delete_operator(o_proj_);
    xnn_delete_operator(sdpa_op_);
  }

  void run(
      const torch::executor::Tensor& q,
      const torch::executor::Tensor& k_cache,
      const torch::executor::Tensor& v_cache) {
    // Fake it
  }

  void run_bench() {
    run_convert(
        linear_type_,
        convert_op_,
        qparams_.data(),
        query_input_float_.value(),
        query_input_.value());
    run_fully_connected(
        linear_type_,
        q_proj_,
        qparams_.data(),
        query_input_.value(),
        query_output_.value());
    run_fully_connected(
        linear_type_,
        k_proj_,
        qparams_.data(),
        query_input_.value(),
        key_output_.value());
    run_fully_connected(
        linear_type_,
        v_proj_,
        qparams_.data(),
        query_input_.value(),
        value_output_.value());

    // Missing
    // - apply rotary embedding on q and k
    run_transpose_nd(
        sdpa_type_,
        qkv_sdpa_out_transpose_,
        4,
        qkv_dims_,
        qkv_sdpa_out_perm_,
        query_output_.value(),
        query_output_.value());
    run_transpose_nd(
        sdpa_type_,
        qkv_sdpa_out_transpose_,
        4,
        qkv_dims_,
        qkv_sdpa_out_perm_,
        key_output_.value(),
        key_output_.value());
    run_transpose_nd(
        sdpa_type_,
        qkv_sdpa_out_transpose_,
        4,
        qkv_dims_,
        qkv_sdpa_out_perm_,
        value_output_.value(),
        value_output_.value());

    run_sdpa(
        sdpa_type_,
        sdpa_op_,
        benchmarking_batch_size_,
        query_heads_,
        query_tokens_,
        kv_heads_,
        kv_tokens_,
        qk_channels_,
        v_channels_,
        query_output_.value(),
        k_cache_.value(),
        v_cache_.value(),
        scales_.value(),
        mask_.value(),
        sdpa_output_.value());
    run_transpose_nd(
        sdpa_type_,
        qkv_sdpa_out_transpose_,
        4,
        sdpa_dims_,
        qkv_sdpa_out_perm_,
        sdpa_output_.value(),
        sdpa_output_.value());
    run_convert(
        linear_type_,
        convert_sdpa_output_op_,
        qparams_.data(),
        sdpa_output_.value(),
        sdpa_output_dq_.value());
    run_fully_connected(
        linear_type_,
        o_proj_,
        qparams_.data(),
        sdpa_output_dq_.value(),
        value_output_.value());
  }

 private:
  int32_t benchmarking_batch_size_{1};
  FullyConnectedOpType linear_type_;
  ComputeType sdpa_type_;
  std::vector<xnn_dynamic_quantization_params> qparams_;
  exec_aten::optional<torch::executor::Tensor> query_input_float_;
  exec_aten::optional<torch::executor::Tensor> query_input_, k_cache_, v_cache_;
  exec_aten::optional<torch::executor::Tensor> scales_, mask_;
  exec_aten::optional<torch::executor::Tensor> sdpa_output_, sdpa_output_dq_;
  exec_aten::optional<torch::executor::Tensor> query_output_, key_output_,
      value_output_, output_;
  xnn_operator_t convert_op_, convert_sdpa_output_op_;
  xnn_operator_t qkv_sdpa_out_transpose_;
  xnn_operator_t q_proj_;
  xnn_operator_t k_proj_;
  xnn_operator_t v_proj_;
  xnn_operator_t o_proj_;
  xnn_operator_t sdpa_op_;
  std::vector<size_t> qkv_sdpa_out_perm_{0, 2, 1, 3};
  std::vector<size_t> qkv_dims_ = std::vector<size_t>(4, 1);
  std::vector<size_t> sdpa_dims_ = std::vector<size_t>(4, 1);
  int32_t query_heads_, kv_heads_, query_tokens_, kv_tokens_, qk_channels_,
      v_channels_;
};

class FeedForward {
 public:
  /*
  What is not accounted for:
  1. Silu
  */
  explicit FeedForward(
      const ModelArgs& args,
      const FullyConnectedOpType linear_type)
      : linear_type_(linear_type) {
    int32_t hidden_dim = 4 * args.dim;
    int32_t n_hidden = 2 * hidden_dim / 3;
    int32_t mask = ~(args.multiple_of - 1);
    int32_t intermediate_size = (n_hidden + args.multiple_of - 1) & mask;

    convert_op_ffn_input_ = create_convert_dq(linear_type);
    w1_ = create_fully_connected(linear_type, args.dim, intermediate_size);
    w3_ = create_fully_connected(linear_type, args.dim, intermediate_size);
    convert_op_w2_input_ = create_convert_dq(linear_type);
    w2_ = create_fully_connected(linear_type, intermediate_size, args.dim);

    torch::executor::ScalarType linear_out_dtype;

    torch::executor::ScalarType ffn_linear_dtype =
        torch::executor::ScalarType::Char;
    if (linear_type_ == FullyConnectedOpType::QD8_F16_QC4W) {
      linear_out_dtype = ::MyHalf;
    } else if (linear_type_ == FullyConnectedOpType::QD8_F32_QC4W) {
      linear_out_dtype = torch::executor::ScalarType::Float;
    } else {
      ET_CHECK_MSG(false, "Unsupported operator type:%hhu", linear_type);
    }

    ffw_input_float_ = TensorFactoryWrapper::make_tensor(
        linear_out_dtype, {benchmarking_batch_size_, args.dim});
    ffw_input_ = TensorFactoryWrapper::make_tensor(
        ffn_linear_dtype, {benchmarking_batch_size_, args.dim});
    w1_output_ = TensorFactoryWrapper::make_tensor(
        linear_out_dtype, {benchmarking_batch_size_, intermediate_size});
    w3_output_ = TensorFactoryWrapper::make_tensor(
        linear_out_dtype, {benchmarking_batch_size_, intermediate_size});
    w2_input_ = TensorFactoryWrapper::make_tensor(
        ffn_linear_dtype, {benchmarking_batch_size_, intermediate_size});
    w2_output_ = TensorFactoryWrapper::make_tensor(
        linear_out_dtype, {benchmarking_batch_size_, args.dim});

    qparams_.resize(benchmarking_batch_size_ + XNN_EXTRA_QUANTIZATION_PARAMS);
  }

  FeedForward(const FeedForward& other) = delete;
  FeedForward& operator=(const FeedForward& other) = delete;
  FeedForward(FeedForward&& other) = delete; // Move constructor
  FeedForward& operator=(FeedForward&& other) noexcept =
      delete; // Move assignment operator

  ~FeedForward() {
    xnn_delete_operator(convert_op_ffn_input_);
    xnn_delete_operator(convert_op_w2_input_);
    xnn_delete_operator(w1_);
    xnn_delete_operator(w2_);
    xnn_delete_operator(w3_);
    // xnn_delete_operator(silu_);
  }

  void run(const torch::executor::Tensor& mha_output) {
    // Fake it
  }

  void run_bench() {
    run_convert(
        linear_type_,
        convert_op_ffn_input_,
        qparams_.data(),
        ffw_input_float_.value(),
        ffw_input_.value());
    run_fully_connected(
        linear_type_,
        w1_,
        qparams_.data(),
        ffw_input_.value(),
        w1_output_.value());
    run_fully_connected(
        linear_type_,
        w3_,
        qparams_.data(),
        ffw_input_.value(),
        w3_output_.value());
    run_convert(
        linear_type_,
        convert_op_w2_input_,
        qparams_.data(),
        w3_output_.value(),
        w2_input_.value());
    run_fully_connected(
        linear_type_,
        w2_,
        qparams_.data(),
        w2_input_.value(),
        w2_output_.value());
  }

 private:
  int32_t benchmarking_batch_size_{1};
  FullyConnectedOpType linear_type_;
  std::vector<xnn_dynamic_quantization_params> qparams_;
  exec_aten::optional<torch::executor::Tensor> ffw_input_, w1_output_,
      w3_output_, w2_output_;
  exec_aten::optional<torch::executor::Tensor> ffw_input_float_, w2_input_;
  xnn_operator_t convert_op_ffn_input_, convert_op_w2_input_;
  xnn_operator_t w1_;
  xnn_operator_t w2_;
  xnn_operator_t w3_;
  xnn_operator_t silu_;
};

class RMSNorm {
 public:
  explicit RMSNorm(const ModelArgs& args, const ComputeType compute_type)
      : compute_type_(compute_type), input_dims_1_(8, 1), input_dims_2_(8, 1) {
    xnn_status status = xnn_create_square_nc_f16(0, &square_op_);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to create square operator");

    status = xnn_create_mean_nd_f16(0, &mean_op_);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to create mean operator");

    status = xnn_create_square_root_nc_f16(0, &sqrt_op_);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to create sqrt operator");

    const float output_min = -std::numeric_limits<float>::infinity();
    const float output_max = +std::numeric_limits<float>::infinity();
    status = xnn_create_divide_nd_f16(output_min, output_max, 0, &divide_op_);
    ET_CHECK_MSG(
        status == xnn_status_success, "Failed to create sqrt operator");

    status = xnn_create_multiply_nd_f16(output_min, output_max, 0, &mul_op_);
    ET_CHECK_MSG(status == xnn_status_success, "Failed to create mul operator");

    torch::executor::ScalarType tensor_dtype;
    if (compute_type_ == ComputeType::F16) {
      tensor_dtype = ::MyHalf;
    } else if (compute_type_ == ComputeType::F32) {
      tensor_dtype = torch::executor::ScalarType::Float;
    } else {
      ET_CHECK_MSG(false, "Unsupported operator type:%hhu", compute_type_);
    }

    weight_ = TensorFactoryWrapper::make_tensor(
        tensor_dtype, {benchmarking_batch_size_, args.dim});
    rmsnorm_input_ = TensorFactoryWrapper::make_tensor(
        tensor_dtype, {benchmarking_batch_size_, args.dim});
    square_output_ = TensorFactoryWrapper::make_tensor(
        tensor_dtype, {benchmarking_batch_size_, args.dim});
    // second dim is one because mean is doing reduction across that dim
    mean_sqrt_output_ = TensorFactoryWrapper::make_tensor(
        tensor_dtype, {benchmarking_batch_size_, 1});
    // second dim is back to args.dim because output of sqrt
    // is broadcast divided by the rmsnorm_input_
    divide_mul_output_ = TensorFactoryWrapper::make_tensor(
        tensor_dtype, {benchmarking_batch_size_, args.dim});
  }

  void run() {
    // Fake it
  }

  void run_bench() {
    run_bench_square_op(rmsnorm_input_.value(), square_output_.value());
    run_bench_mean_op(square_output_.value(), mean_sqrt_output_.value());
    run_bench_square_root_op(
        mean_sqrt_output_.value(), mean_sqrt_output_.value());
    run_bench_divide_op(
        rmsnorm_input_.value(),
        mean_sqrt_output_.value(),
        divide_mul_output_.value());
    run_bench_mul_op(
        weight_.value(),
        divide_mul_output_.value(),
        divide_mul_output_.value());
  }

 private:
  void run_bench_square_op(
      const torch::executor::Tensor& input,
      torch::executor::Tensor& output) {
    auto threadpool = torch::executorch::threadpool::get_pthreadpool();
    size_t input_channels = input.size(1);
    if (compute_type_ == ComputeType::F16) {
      xnn_status status = xnn_reshape_square_nc_f16(
          square_op_,
          benchmarking_batch_size_,
          input_channels,
          input_channels,
          input_channels,
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      status = xnn_setup_square_nc_f16(
          square_op_,
          input.const_data_ptr<uint16_t>(),
          output.mutable_data_ptr<uint16_t>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(square_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    } else if (compute_type_ == ComputeType::F32) {
      xnn_status status = xnn_reshape_square_nc_f32(
          square_op_,
          benchmarking_batch_size_,
          input_channels,
          input_channels,
          input_channels,
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      status = xnn_setup_square_nc_f32(
          square_op_,
          input.const_data_ptr<float>(),
          output.mutable_data_ptr<float>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(square_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    }
  }

  void run_bench_mean_op(
      const torch::executor::Tensor& input,
      torch::executor::Tensor& output) {
    auto threadpool = torch::executorch::threadpool::get_pthreadpool();
    const size_t num_input_dims = input.dim();
    for (size_t i = 0; i < num_input_dims; ++i) {
      input_dims_1_[i] = input.size(i);
    }
    if (compute_type_ == ComputeType::F16) {
      size_t workspace_size;
      size_t workspace_alignment;
      xnn_status status = xnn_reshape_mean_nd_f16(
          mean_op_,
          1,
          mean_axis_.data(),
          num_input_dims,
          input_dims_1_.data(),
          &workspace_size,
          &workspace_alignment,
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      constexpr std::ptrdiff_t alignment{64};
      void* workspace_ptr = aligned_allocate(alignment, workspace_size);
      ET_CHECK_MSG(
          workspace_ptr != nullptr, "Failed to allocate workspace memory");

      status = xnn_setup_mean_nd_f16(
          mean_op_,
          workspace_ptr,
          input.const_data_ptr<uint16_t>(),
          output.mutable_data_ptr<uint16_t>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(mean_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    } else if (compute_type_ == ComputeType::F32) {
      size_t workspace_size;
      size_t workspace_alignment;
      xnn_status status = xnn_reshape_mean_nd_f32(
          mean_op_,
          1,
          mean_axis_.data(),
          num_input_dims,
          input_dims_1_.data(),
          &workspace_size,
          &workspace_alignment,
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      constexpr std::ptrdiff_t alignment{64};
      void* workspace_ptr = aligned_allocate(alignment, workspace_size);
      ET_CHECK_MSG(
          workspace_ptr != nullptr, "Failed to allocate workspace memory");

      status = xnn_setup_mean_nd_f32(
          mean_op_,
          workspace_ptr,
          input.const_data_ptr<float>(),
          output.mutable_data_ptr<float>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(mean_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    }
  }

  void run_bench_square_root_op(
      const torch::executor::Tensor& input,
      torch::executor::Tensor& output) {
    auto threadpool = torch::executorch::threadpool::get_pthreadpool();
    size_t input_channels = input.size(1);
    if (compute_type_ == ComputeType::F16) {
      xnn_status status = xnn_reshape_square_root_nc_f16(
          sqrt_op_,
          benchmarking_batch_size_,
          input_channels,
          input_channels,
          input_channels,
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      status = xnn_setup_square_root_nc_f16(
          sqrt_op_,
          input.const_data_ptr<uint16_t>(),
          output.mutable_data_ptr<uint16_t>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(sqrt_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    } else if (compute_type_ == ComputeType::F32) {
      xnn_status status = xnn_reshape_square_root_nc_f32(
          sqrt_op_,
          benchmarking_batch_size_,
          input_channels,
          input_channels,
          input_channels,
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      status = xnn_setup_square_root_nc_f32(
          sqrt_op_,
          input.const_data_ptr<float>(),
          output.mutable_data_ptr<float>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(sqrt_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    }
  }

  void run_bench_divide_op(
      const torch::executor::Tensor& input1,
      const torch::executor::Tensor& input2,
      torch::executor::Tensor& output) {
    auto threadpool = torch::executorch::threadpool::get_pthreadpool();
    const size_t num_input1_dims = input1.dim();
    const size_t num_input2_dims = input2.dim();
    for (size_t i = 0; i < num_input1_dims; ++i) {
      input_dims_1_[i] = input1.size(i);
    }
    for (size_t i = 0; i < num_input2_dims; ++i) {
      input_dims_2_[i] = input2.size(i);
    }
    if (compute_type_ == ComputeType::F16) {
      xnn_status status = xnn_reshape_divide_nd_f16(
          divide_op_,
          num_input1_dims,
          input_dims_1_.data(),
          num_input2_dims,
          input_dims_2_.data(),
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      status = xnn_setup_divide_nd_f16(
          divide_op_,
          input1.const_data_ptr<uint16_t>(),
          input2.const_data_ptr<uint16_t>(),
          output.mutable_data_ptr<uint16_t>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(mean_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    } else if (compute_type_ == ComputeType::F32) {
      xnn_status status = xnn_reshape_divide_nd_f32(
          divide_op_,
          num_input1_dims,
          input_dims_1_.data(),
          num_input2_dims,
          input_dims_2_.data(),
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      status = xnn_setup_divide_nd_f32(
          divide_op_,
          input1.const_data_ptr<float>(),
          input2.const_data_ptr<float>(),
          output.mutable_data_ptr<float>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(mean_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    }
  }

  void run_bench_mul_op(
      const torch::executor::Tensor& input1,
      const torch::executor::Tensor& input2,
      torch::executor::Tensor& output) {
    auto threadpool = torch::executorch::threadpool::get_pthreadpool();
    const size_t num_input1_dims = input1.dim();
    const size_t num_input2_dims = input2.dim();
    for (size_t i = 0; i < num_input1_dims; ++i) {
      input_dims_1_[i] = input1.size(i);
    }
    for (size_t i = 0; i < num_input2_dims; ++i) {
      input_dims_2_[i] = input2.size(i);
    }
    if (compute_type_ == ComputeType::F16) {
      xnn_status status = xnn_reshape_multiply_nd_f16(
          mul_op_,
          num_input1_dims,
          input_dims_1_.data(),
          num_input2_dims,
          input_dims_2_.data(),
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      status = xnn_setup_multiply_nd_f16(
          mul_op_,
          input1.const_data_ptr<uint16_t>(),
          input2.const_data_ptr<uint16_t>(),
          output.mutable_data_ptr<uint16_t>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(mean_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    } else if (compute_type_ == ComputeType::F32) {
      xnn_status status = xnn_reshape_multiply_nd_f32(
          mul_op_,
          num_input1_dims,
          input_dims_1_.data(),
          num_input2_dims,
          input_dims_2_.data(),
          threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to reshape xnnpack operator");

      status = xnn_setup_multiply_nd_f32(
          mul_op_,
          input1.const_data_ptr<float>(),
          input2.const_data_ptr<float>(),
          output.mutable_data_ptr<float>());
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to setup xnnpack operator");

      status = xnn_run_operator(mean_op_, threadpool);
      ET_CHECK_MSG(
          status == xnn_status_success, "Failed to run xnnpack operator");
      return;
    }
  }

  int32_t benchmarking_batch_size_{1};
  ComputeType compute_type_;
  exec_aten::optional<torch::executor::Tensor> weight_;
  exec_aten::optional<torch::executor::Tensor> rmsnorm_input_;
  exec_aten::optional<torch::executor::Tensor> square_output_;
  exec_aten::optional<torch::executor::Tensor> mean_sqrt_output_,
      divide_mul_output_;
  xnn_operator_t square_op_;
  xnn_operator_t mean_op_;
  xnn_operator_t sqrt_op_;
  xnn_operator_t divide_op_;
  xnn_operator_t mul_op_;
  std::vector<size_t> mean_axis_{1};
  std::vector<size_t> input_dims_1_;
  std::vector<size_t> input_dims_2_;
};

class TransformerBlock {
 public:
  explicit TransformerBlock(
      const ModelArgs& args,
      const FullyConnectedOpType linear_type,
      const ComputeType sdpa_type)
      : multi_headed_attention_(args, linear_type, sdpa_type),
        feedforward_(args, linear_type),
        attention_norm_(args, sdpa_type),
        ffn_norm_(args, sdpa_type) {}

  void run(const torch::executor::Tensor& mha_output) {
    // Fake it
  }

  void run_bench() {
    attention_norm_.run_bench();
    multi_headed_attention_.run_bench();
    ffn_norm_.run_bench();
    feedforward_.run_bench();
  }

 private:
  MultiHeadedAttention multi_headed_attention_;
  FeedForward feedforward_;
  RMSNorm attention_norm_;
  RMSNorm ffn_norm_;
};

class Transformer {
 public:
  /*
  What is not accounted for:
  1. Generating and selecting input pos dependent causal mask
  2. frequ_cis populating for RoPE
  In run_bench
  1. Token embeddings
  */
  explicit Transformer(
      const ModelArgs& args,
      const FullyConnectedOpType linear_type,
      const ComputeType sdpa_type)
      : linear_type_(linear_type) {
    transformer_blocks_.reserve(args.n_layers);
    for (int i = 0; i < args.n_layers; ++i) {
      transformer_blocks_.emplace_back(
          std::make_unique<TransformerBlock>(args, linear_type, sdpa_type));
    }

    // Not sure if we should quantize the last linear layer or not. For now
    // assuming we do.
    out_logits_ =
        create_fully_connected(linear_type, args.dim, args.vocab_size);

    torch::executor::ScalarType out_logits_out_type;

    torch::executor::ScalarType out_logits_linear_type =
        torch::executor::ScalarType::Char;
    if (linear_type_ == FullyConnectedOpType::QD8_F16_QC4W) {
      out_logits_out_type = ::MyHalf;
    } else if (linear_type_ == FullyConnectedOpType::QD8_F32_QC4W) {
      out_logits_out_type = torch::executor::ScalarType::Float;
    } else {
      ET_CHECK_MSG(false, "Unsupported operator type:%hhu", linear_type);
    }
    out_logits_input_ = TensorFactoryWrapper::make_tensor(
        out_logits_linear_type, {benchmarking_batch_size_, args.dim});
    out_logits_output_ = TensorFactoryWrapper::make_tensor(
        out_logits_out_type, {benchmarking_batch_size_, args.vocab_size});

    qparams_.resize(1 + XNN_EXTRA_QUANTIZATION_PARAMS);
    std::generate(qparams_.begin(), qparams_.end(), [&]() {
      return xnn_dynamic_quantization_params{0, 1.f};
    });
  }

  Transformer(const Transformer& other) = delete;
  Transformer& operator=(const Transformer& other) = delete;
  Transformer(Transformer&& other) = delete; // Move constructor
  Transformer& operator=(Transformer&& other) noexcept =
      delete; // Move assignment operator

  void run(const torch::executor::Tensor& mha_output) {
    // Fake it
  }

  void run_bench() {
    for (auto& transformer_block : transformer_blocks_) {
      transformer_block->run_bench();
    }
    run_fully_connected(
        linear_type_,
        out_logits_,
        qparams_.data(),
        out_logits_input_.value(),
        out_logits_output_.value());
  }

 private:
  std::vector<std::unique_ptr<TransformerBlock>> transformer_blocks_;
  xnn_operator_t out_logits_;
  int32_t benchmarking_batch_size_{1};
  FullyConnectedOpType linear_type_;
  std::vector<xnn_dynamic_quantization_params> qparams_;
  exec_aten::optional<torch::executor::Tensor> out_logits_input_,
      out_logits_output_;
};

// #define BENCHMARK_FP32
static void benchmark_llama2_7b() {
  ModelArgs args;
  const int32_t kWarmupIterations = 10;
  const int32_t kIterations = 50;
// Need to benchmark pre-fill separately.
#if defined(BENCHMARK_FP32)
  Transformer transformer(
      args, FullyConnectedOpType::QD8_F32_QC4W, ComputeType::F32);
#else
  Transformer transformer(
      args, FullyConnectedOpType::QD8_F16_QC4W, ComputeType::F16);
#endif
  for (int i = 0; i < kWarmupIterations; ++i) {
    transformer.run_bench();
  }
  auto start_time = std::chrono::steady_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    transformer.run_bench();
  }
  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time - start_time)
                        .count();
  std::cout << "Elapsed time: " << elapsed_us << " microseconds" << std::endl;
  std::cout << "Elapsed time per iter(" << kIterations
            << "): " << elapsed_us / kIterations << " microseconds"
            << std::endl;
}

// BENCHMARK(benchmark_llama2_7b)->Unit(benchmark::kMicrosecond)->UseRealTime();
int main(int argc, char** argv) {
  torch::executor::runtime_init();
  xnn_status status = xnn_initialize(/*allocator =*/nullptr);
  ET_CHECK_MSG(status == xnn_status_success, "failed to initialize xnnpack");
  benchmark_llama2_7b();
}
