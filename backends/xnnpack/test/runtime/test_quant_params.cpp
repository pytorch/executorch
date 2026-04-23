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
    EXPECT_TRUE(is_quantized(DType::QInt8Sym));
}

TEST(TestQuantParams, is_quantized_qint4sym) {
    EXPECT_TRUE(is_quantized(DType::QInt4Sym));
}

TEST(TestQuantParams, is_quantized_quint8asym) {
    EXPECT_TRUE(is_quantized(DType::QUInt8Asym));
}

// --- is_asymmetric ---

TEST(TestQuantParams, is_asymmetric_sym) {
    EXPECT_FALSE(is_asymmetric(DType::QInt8Sym));
    EXPECT_FALSE(is_asymmetric(DType::QInt4Sym));
}

TEST(TestQuantParams, is_asymmetric_asym) {
    EXPECT_TRUE(is_asymmetric(DType::QUInt8Asym));
}

// --- element_size ---

TEST(TestQuantParams, element_size_float32) {
    EXPECT_EQ(element_size(DType::Float32), 4);
}

TEST(TestQuantParams, element_size_qint8) {
    EXPECT_EQ(element_size(DType::QInt8Sym), 1);
    EXPECT_EQ(element_size(DType::QUInt8Asym), 1);
}

// --- compute_storage_size ---

TEST(TestQuantParams, storage_size_qint8sym) {
    EXPECT_EQ(compute_storage_size({4, 8}, DType::QInt8Sym), 32);
}

TEST(TestQuantParams, storage_size_quint8asym) {
    EXPECT_EQ(compute_storage_size({2, 5}, DType::QUInt8Asym), 10);
}

TEST(TestQuantParams, storage_size_qint4sym_even) {
    EXPECT_EQ(compute_storage_size({2, 4}, DType::QInt4Sym), 4);
}

TEST(TestQuantParams, storage_size_qint4sym_odd) {
    EXPECT_EQ(compute_storage_size({7}, DType::QInt4Sym), 4);
}

TEST(TestQuantParams, storage_size_qint4sym_one) {
    EXPECT_EQ(compute_storage_size({1}, DType::QInt4Sym), 1);
}

// --- Preset factories ---

TEST(TestQuantParams, preset_qint8_per_channel_sym) {
    auto p = qint8_per_channel_sym(0);
    auto* pa = std::get_if<PerAxisQuant>(&p);
    ASSERT_NE(pa, nullptr);
    EXPECT_EQ(pa->axis, 0);
    EXPECT_EQ(pa->scale_dtype, DType::Float32);
}

TEST(TestQuantParams, preset_qint8_per_tensor_sym) {
    auto p = qint8_per_tensor_sym(0.5f);
    auto* pt = std::get_if<PerTensorQuant>(&p);
    ASSERT_NE(pt, nullptr);
    EXPECT_FLOAT_EQ(pt->scale, 0.5f);
    EXPECT_EQ(pt->zero_point, 0);
}

TEST(TestQuantParams, preset_quint8_per_tensor_asym) {
    auto p = quint8_per_tensor_asym(0.25f, 128);
    auto* pt = std::get_if<PerTensorQuant>(&p);
    ASSERT_NE(pt, nullptr);
    EXPECT_FLOAT_EQ(pt->scale, 0.25f);
    EXPECT_EQ(pt->zero_point, 128);
}

TEST(TestQuantParams, preset_quint8_per_token_asym) {
    auto p = quint8_per_token_asym(0);
    auto* pa = std::get_if<PerAxisQuant>(&p);
    ASSERT_NE(pa, nullptr);
    EXPECT_EQ(pa->axis, 0);
}

TEST(TestQuantParams, preset_qint4_blockwise_sym) {
    auto p = qint4_blockwise_sym(1, 32);
    auto* pb = std::get_if<BlockwiseQuant>(&p);
    ASSERT_NE(pb, nullptr);
    EXPECT_EQ(pb->axis, 1);
    EXPECT_EQ(pb->block_size, 32);
    EXPECT_EQ(pb->scale_dtype, DType::Float32);
}

// --- aux_buffer_count ---

TEST(TestQuantParams, aux_buffer_count_float) {
    QuantParams dummy = PerTensorQuant{};
    EXPECT_EQ(aux_buffer_count(DType::Float32, dummy), 0);
}

TEST(TestQuantParams, aux_buffer_count_sym) {
    auto p = qint8_per_channel_sym(0);
    EXPECT_EQ(aux_buffer_count(DType::QInt8Sym, p), 1);
}

TEST(TestQuantParams, aux_buffer_count_asym) {
    auto p = quint8_per_tensor_asym(1.0f, 0);
    EXPECT_EQ(aux_buffer_count(DType::QUInt8Asym, p), 2);
}

// --- compute_aux_storage_sizes ---

TEST(TestQuantParams, aux_sizes_per_tensor_sym) {
    auto p = qint8_per_tensor_sym(1.0f);
    auto sizes = compute_aux_storage_sizes({4, 8}, DType::QInt8Sym, p);
    ASSERT_EQ(sizes.size(), 1);
    EXPECT_EQ(sizes[0], sizeof(float)); // 1 scale, float32
}

TEST(TestQuantParams, aux_sizes_per_axis_sym) {
    // [4, 8], per-channel along axis=0 (quant over dim 0).
    // Scales indexed by all other dims -> 8 scales.
    auto p = qint8_per_channel_sym(0);
    auto sizes = compute_aux_storage_sizes({4, 8}, DType::QInt8Sym, p);
    ASSERT_EQ(sizes.size(), 1);
    EXPECT_EQ(sizes[0], 8 * sizeof(float));
}

TEST(TestQuantParams, aux_sizes_per_axis_sym_weights) {
    // [out_ch=4, in_ch=8], per-channel along axis=1 (quant over in_ch).
    // One scale per output channel -> 4 scales.
    auto p = qint8_per_channel_sym(1);
    auto sizes = compute_aux_storage_sizes({4, 8}, DType::QInt8Sym, p);
    ASSERT_EQ(sizes.size(), 1);
    EXPECT_EQ(sizes[0], 4 * sizeof(float));
}

TEST(TestQuantParams, aux_sizes_per_axis_asym) {
    // [4, 8], per-token along axis=1 (quant over feature dim).
    // One scale per row -> 4 scales + 4 zero_points.
    auto p = quint8_per_token_asym(1);
    auto sizes = compute_aux_storage_sizes({4, 8}, DType::QUInt8Asym, p);
    ASSERT_EQ(sizes.size(), 2);
    EXPECT_EQ(sizes[0], 4 * sizeof(float));   // scales
    EXPECT_EQ(sizes[1], 4 * sizeof(int32_t)); // zero_points
}

TEST(TestQuantParams, aux_sizes_per_token_3d) {
    // [batch=2, seqlen=3, features=8], per-token along axis=2.
    // One scale per [batch, seqlen] combo -> 2*3 = 6 scales.
    auto p = quint8_per_token_asym(2);
    auto sizes = compute_aux_storage_sizes({2, 3, 8}, DType::QUInt8Asym, p);
    ASSERT_EQ(sizes.size(), 2);
    EXPECT_EQ(sizes[0], 6 * sizeof(float));   // scales
    EXPECT_EQ(sizes[1], 6 * sizeof(int32_t)); // zero_points
}

TEST(TestQuantParams, aux_sizes_blockwise_sym) {
    // [4, 128], blockwise along axis=1, block_size=32
    // num_blocks = 128/32 = 4
    // other_dims = 4 (axis=1, so dim 0 contributes)
    // total scales = 4 * 4 = 16
    auto p = qint4_blockwise_sym(1, 32);
    auto sizes = compute_aux_storage_sizes({4, 128}, DType::QInt4Sym, p);
    ASSERT_EQ(sizes.size(), 1);
    EXPECT_EQ(sizes[0], 16 * sizeof(float));
}

TEST(TestQuantParams, aux_sizes_blockwise_sym_not_divisible) {
    // [4, 100], blockwise along axis=1, block_size=32
    // num_blocks = ceil(100/32) = 4
    // total scales = 4 * 4 = 16
    auto p = qint4_blockwise_sym(1, 32);
    auto sizes = compute_aux_storage_sizes({4, 100}, DType::QInt4Sym, p);
    ASSERT_EQ(sizes.size(), 1);
    EXPECT_EQ(sizes[0], 16 * sizeof(float));
}
