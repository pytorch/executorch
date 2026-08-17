#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/core/dtype.h>
#include <executorch/backends/xnnpack/runtime/core/quant_params.h>
#include <executorch/backends/xnnpack/runtime/core/tensor.h>

using namespace executorch::backends::xnnpack::core;

// --- is_quantized ---

TEST(TestQuantParams, is_quantized_float) {
  EXPECT_FALSE(is_quantized(DType::Float32));
}

TEST(TestQuantParams, is_quantized_qint8sym) {
  EXPECT_TRUE(is_quantized(DType::QInt8));
}

TEST(TestQuantParams, is_quantized_qint4sym) {
  EXPECT_TRUE(is_quantized(DType::QInt4));
}

TEST(TestQuantParams, is_quantized_quint8asym) {
  EXPECT_TRUE(is_quantized(DType::QUInt8));
}

TEST(TestQuantParams, is_quantized_nonquantized_types) {
  EXPECT_FALSE(is_quantized(DType::Float16));
  EXPECT_FALSE(is_quantized(DType::BFloat16));
  EXPECT_FALSE(is_quantized(DType::Int64));
  EXPECT_FALSE(is_quantized(DType::UInt64));
}

// --- is_asymmetric (now derived from QuantParams, not DType) ---

TEST(TestQuantParams, is_asymmetric_sym) {
  EXPECT_FALSE(is_asymmetric(qint8_per_channel_sym(0)));
  EXPECT_FALSE(is_asymmetric(qint8_per_tensor_sym(0.5f)));
  EXPECT_FALSE(is_asymmetric(qint4_blockwise_sym(1, 32)));
}

TEST(TestQuantParams, is_asymmetric_asym) {
  EXPECT_TRUE(is_asymmetric(quint8_per_tensor_asym(0.25f, 128)));
  EXPECT_TRUE(is_asymmetric(quint8_per_row_asym(-1)));
  EXPECT_TRUE(is_asymmetric(quint8_per_token_asym()));
}

// --- is_subbyte / byte_stride ---

TEST(TestQuantParams, is_subbyte) {
  EXPECT_TRUE(is_subbyte(DType::QInt4));
  EXPECT_FALSE(is_subbyte(DType::Float32));
  EXPECT_FALSE(is_subbyte(DType::Float16));
  EXPECT_FALSE(is_subbyte(DType::BFloat16));
  EXPECT_FALSE(is_subbyte(DType::Int64));
  EXPECT_FALSE(is_subbyte(DType::UInt64));
  EXPECT_FALSE(is_subbyte(DType::QInt8));
  EXPECT_FALSE(is_subbyte(DType::QUInt8));
  EXPECT_FALSE(is_subbyte(DType::QInt32));
}

TEST(TestQuantParams, byte_stride) {
  EXPECT_EQ(byte_stride(DType::QInt8), 1);
  EXPECT_EQ(byte_stride(DType::QUInt8), 1);
  EXPECT_EQ(byte_stride(DType::Float16), 2);
  EXPECT_EQ(byte_stride(DType::BFloat16), 2);
  EXPECT_EQ(byte_stride(DType::Float32), 4);
  EXPECT_EQ(byte_stride(DType::QInt32), 4);
  EXPECT_EQ(byte_stride(DType::Int64), 8);
  EXPECT_EQ(byte_stride(DType::UInt64), 8);
}

// --- compute_storage_size ---

TEST(TestQuantParams, storage_size_qint8sym) {
  const uint64_t sizes[] = {4, 8};
  auto r = compute_storage_size(sizes, DType::QInt8);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.get(), 32);
}

TEST(TestQuantParams, storage_size_quint8asym) {
  const uint64_t sizes[] = {2, 5};
  auto r = compute_storage_size(sizes, DType::QUInt8);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.get(), 10);
}

TEST(TestQuantParams, storage_size_float16) {
  const uint64_t sizes[] = {4, 8};
  auto r = compute_storage_size(sizes, DType::Float16);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.get(), 64); // 32 elements * 2 bytes
}

TEST(TestQuantParams, storage_size_int64) {
  const uint64_t sizes[] = {4, 8};
  auto r = compute_storage_size(sizes, DType::Int64);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.get(), 256); // 32 elements * 8 bytes
}

TEST(TestQuantParams, storage_size_qint4sym_even) {
  const uint64_t sizes[] = {2, 4};
  auto r = compute_storage_size(sizes, DType::QInt4);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.get(), 4);
}

TEST(TestQuantParams, storage_size_qint4sym_odd) {
  const uint64_t sizes[] = {7};
  auto r = compute_storage_size(sizes, DType::QInt4);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.get(), 4);
}

TEST(TestQuantParams, storage_size_qint4sym_one) {
  const uint64_t sizes[] = {1};
  auto r = compute_storage_size(sizes, DType::QInt4);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.get(), 1);
}

TEST(TestQuantParams, storage_size_overflow_returns_error) {
  const uint64_t sizes[] = {SIZE_MAX, 2};
  auto r = compute_storage_size(sizes, DType::QInt8);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.error(), executorch::runtime::Error::InvalidArgument);
}

TEST(TestQuantParams, storage_size_byte_overflow_returns_error) {
  // num_elements fits in size_t but num_elements * 4 overflows.
  const uint64_t sizes[] = {SIZE_MAX / 2};
  auto r = compute_storage_size(sizes, DType::Float32);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.error(), executorch::runtime::Error::InvalidArgument);
}

// --- Preset factories ---

TEST(TestQuantParams, preset_qint8_per_channel_sym) {
  auto p = qint8_per_channel_sym(0);
  auto* pa = std::get_if<PerAxisQuantParams>(&p);
  ASSERT_NE(pa, nullptr);
  EXPECT_EQ(pa->axis, 0);
  EXPECT_EQ(pa->scale_dtype, DType::Float32);
  EXPECT_FALSE(pa->has_zero_point);
}

TEST(TestQuantParams, preset_qint8_per_tensor_sym) {
  auto p = qint8_per_tensor_sym(0.5f);
  auto* pt = std::get_if<PerTensorQuantParams>(&p);
  ASSERT_NE(pt, nullptr);
  EXPECT_FLOAT_EQ(pt->scale, 0.5f);
  EXPECT_EQ(pt->zero_point, 0);
  EXPECT_FALSE(pt->has_zero_point);
}

TEST(TestQuantParams, preset_quint8_per_tensor_asym) {
  auto p = quint8_per_tensor_asym(0.25f, 128);
  auto* pt = std::get_if<PerTensorQuantParams>(&p);
  ASSERT_NE(pt, nullptr);
  EXPECT_FLOAT_EQ(pt->scale, 0.25f);
  EXPECT_EQ(pt->zero_point, 128);
  EXPECT_TRUE(pt->has_zero_point);
}

TEST(TestQuantParams, preset_quint8_per_row_asym) {
  auto p = quint8_per_row_asym(1);
  auto* pr = std::get_if<PerRowQuantParams>(&p);
  ASSERT_NE(pr, nullptr);
  EXPECT_EQ(pr->axis, 1);
  EXPECT_EQ(pr->scale_dtype, DType::Float32);
  EXPECT_TRUE(pr->has_zero_point);
}

TEST(TestQuantParams, preset_quint8_per_token_asym) {
  auto p = quint8_per_token_asym();
  auto* pr = std::get_if<PerRowQuantParams>(&p);
  ASSERT_NE(pr, nullptr);
  EXPECT_EQ(pr->axis, -1);
  EXPECT_TRUE(pr->has_zero_point);
}

TEST(TestQuantParams, preset_qint4_blockwise_sym) {
  auto p = qint4_blockwise_sym(1, 32);
  auto* pb = std::get_if<PerBlockQuantParams>(&p);
  ASSERT_NE(pb, nullptr);
  EXPECT_EQ(pb->axis, 1);
  EXPECT_EQ(pb->block_size, 32);
  EXPECT_EQ(pb->scale_dtype, DType::Float32);
}

// --- aux_buffer_count ---

TEST(TestQuantParams, aux_buffer_count_float) {
  QuantParams dummy = PerTensorQuantParams{};
  EXPECT_EQ(aux_buffer_count(DType::Float32, dummy), 0);
}

TEST(TestQuantParams, aux_buffer_count_sym) {
  auto p = qint8_per_channel_sym(0);
  EXPECT_EQ(aux_buffer_count(DType::QInt8, p), 1);
}

TEST(TestQuantParams, aux_buffer_count_asym) {
  auto p = quint8_per_tensor_asym(1.0f, 0);
  EXPECT_EQ(aux_buffer_count(DType::QUInt8, p), 2);
}

// --- compute_aux_storage_sizes ---

TEST(TestQuantParams, aux_sizes_per_tensor_sym) {
  auto p = qint8_per_tensor_sym(1.0f);
  const uint64_t shape[] = {4, 8};
  auto sizes = compute_aux_storage_sizes(shape, DType::QInt8, p).get();
  ASSERT_EQ(sizes.size(), 1);
  EXPECT_EQ(sizes[0], sizeof(float)); // 1 scale, float32
}

TEST(TestQuantParams, aux_sizes_per_axis_keep_axis0) {
  // [4, 8], keep axis=0 -> one scale per index along axis 0 -> 4 scales.
  auto p = qint8_per_channel_sym(0);
  const uint64_t shape[] = {4, 8};
  auto sizes = compute_aux_storage_sizes(shape, DType::QInt8, p).get();
  ASSERT_EQ(sizes.size(), 1);
  EXPECT_EQ(sizes[0], 4 * sizeof(float));
}

TEST(TestQuantParams, aux_sizes_per_axis_keep_axis1) {
  // [4, 8], keep axis=1 -> 8 scales.
  auto p = qint8_per_channel_sym(1);
  const uint64_t shape[] = {4, 8};
  auto sizes = compute_aux_storage_sizes(shape, DType::QInt8, p).get();
  ASSERT_EQ(sizes.size(), 1);
  EXPECT_EQ(sizes[0], 8 * sizeof(float));
}

TEST(TestQuantParams, aux_sizes_per_channel_conv3d) {
  // conv3d weight [out=4, in=8, kT=3, kH=3, kW=3], per-output-channel keeps
  // axis 0 and reduces the rest -> one scale per output channel -> 4.
  auto p = qint8_per_channel_sym(0);
  const uint64_t shape[] = {4, 8, 3, 3, 3};
  auto sizes = compute_aux_storage_sizes(shape, DType::QInt8, p).get();
  ASSERT_EQ(sizes.size(), 1);
  EXPECT_EQ(sizes[0], 4 * sizeof(float));
}

TEST(TestQuantParams, aux_sizes_per_row_asym_2d) {
  // [4, 8], per-token reduces the last dim -> one scale per row -> 4 scales
  // + 4 zero_points.
  auto p = quint8_per_token_asym();
  const uint64_t shape[] = {4, 8};
  auto sizes = compute_aux_storage_sizes(shape, DType::QUInt8, p).get();
  ASSERT_EQ(sizes.size(), 2);
  EXPECT_EQ(sizes[0], 4 * sizeof(float)); // scales
  EXPECT_EQ(sizes[1], 4 * sizeof(int32_t)); // zero_points
}

TEST(TestQuantParams, aux_sizes_per_token_3d) {
  // [batch=2, seqlen=3, features=8], per-token reduces the last dim ->
  // one scale per [batch, seqlen] combo -> 2*3 = 6 scales.
  auto p = quint8_per_token_asym();
  const uint64_t shape[] = {2, 3, 8};
  auto sizes = compute_aux_storage_sizes(shape, DType::QUInt8, p).get();
  ASSERT_EQ(sizes.size(), 2);
  EXPECT_EQ(sizes[0], 6 * sizeof(float)); // scales
  EXPECT_EQ(sizes[1], 6 * sizeof(int32_t)); // zero_points
}

TEST(TestQuantParams, aux_sizes_per_row_explicit_dim) {
  // [2, 3, 8], reduce dim 1 -> keep dims 0 and 2 -> 2*8 = 16 scales.
  auto p = quint8_per_row_asym(1);
  const uint64_t shape[] = {2, 3, 8};
  auto sizes = compute_aux_storage_sizes(shape, DType::QUInt8, p).get();
  ASSERT_EQ(sizes.size(), 2);
  EXPECT_EQ(sizes[0], 16 * sizeof(float));
  EXPECT_EQ(sizes[1], 16 * sizeof(int32_t));
}

TEST(TestQuantParams, aux_sizes_per_row_negative_dim) {
  // dim=-1 on [2, 3, 8] reduces the last dim -> 6 scales (same as per-token).
  auto p = quint8_per_row_asym(-1);
  const uint64_t shape[] = {2, 3, 8};
  auto sizes = compute_aux_storage_sizes(shape, DType::QUInt8, p).get();
  ASSERT_EQ(sizes.size(), 2);
  EXPECT_EQ(sizes[0], 6 * sizeof(float));
}

TEST(TestQuantParams, aux_sizes_per_row_dim_out_of_range_errors) {
  // dim=3 is invalid for a 3-dim tensor.
  auto p = quint8_per_row_asym(3);
  const uint64_t shape[] = {2, 3, 8};
  auto result = compute_aux_storage_sizes(shape, DType::QUInt8, p);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), executorch::runtime::Error::InvalidArgument);
}

TEST(TestQuantParams, aux_sizes_per_row_negative_dim_out_of_range_errors) {
  // dim=-4 resolves to -1 for a 3-dim tensor, which is invalid.
  auto p = quint8_per_row_asym(-4);
  const uint64_t shape[] = {2, 3, 8};
  auto result = compute_aux_storage_sizes(shape, DType::QUInt8, p);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), executorch::runtime::Error::InvalidArgument);
}

TEST(TestQuantParams, aux_sizes_blockwise_sym) {
  // [4, 128], blockwise along axis=1, block_size=32
  // num_blocks = 128/32 = 4
  // other_dims = 4 (axis=1, so dim 0 contributes)
  // total scales = 4 * 4 = 16
  auto p = qint4_blockwise_sym(1, 32);
  const uint64_t shape[] = {4, 128};
  auto sizes = compute_aux_storage_sizes(shape, DType::QInt4, p).get();
  ASSERT_EQ(sizes.size(), 1);
  EXPECT_EQ(sizes[0], 16 * sizeof(float));
}

TEST(TestQuantParams, aux_sizes_per_block_not_divisible_errors) {
  // [4, 100], block_size=32 along axis=1 does not evenly divide 100.
  auto p = qint4_blockwise_sym(1, 32);
  const uint64_t shape[] = {4, 100};
  auto result = compute_aux_storage_sizes(shape, DType::QInt4, p);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), executorch::runtime::Error::InvalidArgument);
}

// --- compute_aux_storage_sizes validation ---

TEST(TestQuantParams, aux_sizes_axis_out_of_range_errors) {
  // axis=2 is invalid for a 2-dim tensor.
  auto p = qint8_per_channel_sym(2);
  const uint64_t shape[] = {4, 8};
  auto result = compute_aux_storage_sizes(shape, DType::QInt8, p);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), executorch::runtime::Error::InvalidArgument);
}

TEST(TestQuantParams, aux_sizes_negative_axis_errors) {
  auto p = qint8_per_channel_sym(-1);
  const uint64_t shape[] = {4, 8};
  auto result = compute_aux_storage_sizes(shape, DType::QInt8, p);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), executorch::runtime::Error::InvalidArgument);
}

TEST(TestQuantParams, aux_sizes_zero_block_size_errors) {
  auto p = qint4_blockwise_sym(1, 0);
  const uint64_t shape[] = {4, 128};
  auto result = compute_aux_storage_sizes(shape, DType::QInt4, p);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), executorch::runtime::Error::InvalidArgument);
}
